from django.shortcuts import render
from django.conf import settings
import googleapiclient.discovery
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import isodate
from sklearn.preprocessing import MinMaxScaler
import numpy as np

api_service_name = "youtube"
api_version = "v3"
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=settings.YOUTUBE_API_KEY
)

def fetch_videos(query, max_videos):
    video_details = []
    next_page_token = None

    while len(video_details) < max_videos:
        request = youtube.search().list(
            part="snippet",
            maxResults=max_videos,
            order="date",
            type="video",
            q=query,
            pageToken=next_page_token
        )
        response = request.execute()
        video_details.extend(response['items'])
        next_page_token = response.get('nextPageToken')
        if next_page_token is None:
            break

    return video_details

def get_video_durations(video_ids, videoQty):
    durations = []
    for i in range(0, len(video_ids), videoQty):
        request = youtube.videos().list(
            part="contentDetails",
            id=','.join(video_ids[i:i+videoQty])
        )
        response = request.execute()
        for item in response['items']:
            duration = item['contentDetails']['duration']
            durations.append(duration)
    return durations

def parse_duration(duration):
    duration = isodate.parse_duration(duration)
    return duration.total_seconds()

def load_and_prepare_data(query, videoQty):
    videos = fetch_videos(query, videoQty)
    df = pd.DataFrame(videos)

    file_path = 'video_details.csv'
    df.to_csv(file_path, index=False)

    video_data = pd.read_csv(file_path)
    video_data['videoId'] = video_data['id'].apply(lambda x: eval(x)['videoId'])
    video_data['title'] = video_data['snippet'].apply(lambda x: eval(x)['title'])
    video_data['description'] = video_data['snippet'].apply(lambda x: eval(x)['description'])
    video_data['thumbnail'] = video_data['snippet'].apply(lambda x: eval(x)['thumbnails']['high']['url'])

    video_ids = video_data['videoId'].tolist()
    durations = get_video_durations(video_ids,videoQty)
    video_data['duration'] = durations
    video_data['duration'] = video_data['duration'].apply(parse_duration)
    video_data_cleaned = video_data.drop(columns=['kind', 'id', 'snippet'])
    video_data_cleaned['combined'] = video_data_cleaned['title'] + ' ' + video_data_cleaned['description']

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(video_data_cleaned['combined'])
    scaler = StandardScaler()
    video_data_cleaned['duration_scaled'] = scaler.fit_transform(video_data_cleaned[['duration']])
    duration_sparse = csr_matrix(video_data_cleaned[['duration_scaled']].values)
    combined_features = hstack([tfidf_matrix, duration_sparse])

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(combined_features)
    knn2 = NearestNeighbors(metric='euclidean', algorithm='brute')
    knn2.fit(combined_features)
    
    return video_data_cleaned, combined_features, knn, knn2

def get_recommendations(video_id, video_data_cleaned, combined_features, knn, knn2, k):
    if video_id not in video_data_cleaned['videoId'].values:
        return None

    idx = video_data_cleaned.index[video_data_cleaned['videoId'] == video_id][0]
    query_features = combined_features[idx]
    distances, indices = knn.kneighbors(query_features, n_neighbors=k + 1)
    distances2, indices2 = knn2.kneighbors(query_features, n_neighbors=k + 1)
    
    indices = indices.flatten()[1:]
    distances = distances.flatten()[1:]
    indices2 = indices2.flatten()[1:]
    distances2 = distances2.flatten()[1:]
    recommended_videos = video_data_cleaned.iloc[indices]
    recommended_videos2 = video_data_cleaned.iloc[indices2]

    normalized_distances_cosine = normalize_distances(distances)
    normalized_distances_euclidean = normalize_distances(distances2)

    weighted_scores_cosine = calculate_weighted_scores(normalized_distances_cosine)
    weighted_scores_euclidean = calculate_weighted_scores(normalized_distances_euclidean)

    recommendations = []
    for _, row in recommended_videos.iterrows():
        recommendations.append({
            'title': row['title'],
            'description': row['description'],
            'duration': row['duration'],
            'thumbnail': row['thumbnail'],
            'url': 'https://www.youtube.com/watch?v='+row['videoId'],
            'score': weighted_scores_cosine
        })

    recommendations2 = []
    for _, row in recommended_videos2.iterrows():
        recommendations2.append({
            'title': row['title'],
            'description': row['description'],
            'duration': row['duration'],
            'thumbnail': row['thumbnail'],
            'url': 'https://www.youtube.com/watch?v='+row['videoId'],
            'score': weighted_scores_euclidean
        })

    return normalized_distances_cosine, recommendations, normalized_distances_euclidean, recommendations2

def home(request):
    return render(request, 'appta/index.html')

def preferences_form(request):
    if request.method == 'POST':
        topic = request.POST.get('topic')
        difficulty = request.POST.get('difficulty')
        query = f"{topic} {difficulty}"

        # Store the query in the session
        request.session['query'] = query

        video_data_cleaned, combined_features, knn, knn2 = load_and_prepare_data(query, 50)
        initial_videos = video_data_cleaned.head(5)
        return render(request, 'appta/initial_videos.html', {'initial_videos': initial_videos})

    return render(request, 'appta/preferences.html')

# Function to normalize distances using Min-Max Normalization
def normalize_distances(distances):
    scaler = MinMaxScaler()
    distances = np.array(distances).reshape(-1, 1)
    normalized_distances = scaler.fit_transform(distances)
    return normalized_distances.flatten()

def calculate_weighted_scores(distances):
    # lower distances imply higher relevance
    max_distance = np.max(distances)
    weighted_scores = max_distance - distances
    return weighted_scores

def select_video(request):
    if request.method == 'POST':
        video_id = request.POST.get('selected_video_id')
        video_data_cleaned, combined_features, knn, knn2 = load_and_prepare_data(request.session['query'], 50)
        normalized_distances_cosine, recommendations, normalized_distances_euclidean, recommendations2 = get_recommendations(video_id, video_data_cleaned, combined_features, knn, knn2, k=5)

        recommendations_with_distances = zip(recommendations, normalized_distances_cosine)
        recommendations_with_distances2 = zip(recommendations2, normalized_distances_euclidean)

        # Calculate average weighted scores
        avg_score_cosine = np.mean([rec['score'] for rec in recommendations])
        avg_score_euclidean = np.mean([rec['score'] for rec in recommendations2])

        return render(request, 'appta/recommendations.html', {'recommendations_with_distances': recommendations_with_distances,'recommendations_with_distances2': recommendations_with_distances2,'average_normalized_distance_cosine': avg_score_cosine,'average_normalized_distance_euclidean': avg_score_euclidean})
