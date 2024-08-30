from django.shortcuts import render
from django.conf import settings
import googleapiclient.discovery
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import isodate
import ast

api_service_name = "youtube"
api_version = "v3"
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=settings.YOUTUBE_API_KEY
)

def fetch_videos(query, max_videos=50):
    video_details = []
    next_page_token = None

    while len(video_details) < max_videos:
        request = youtube.search().list(
            part="snippet",
            maxResults=50,
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

    video_ids = [video['id']['videoId'] for video in video_details]

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute()
        for item in response['items']:
            video_id = item['id']
            like_count = int(item['statistics']['likeCount']) if 'likeCount' in item['statistics'] else None
            view_count = int(item['statistics']['viewCount']) if 'viewCount' in item['statistics'] else None
            for video in video_details:
                if video['id']['videoId'] == video_id:
                    video['statistics'] = {'likeCount': like_count, 'viewCount': view_count}
                    break

    for video in video_details:
        if 'statistics' not in video or not isinstance(video['statistics'], dict):
            video['statistics'] = {'likeCount': None, 'viewCount': None}

    return video_details

def get_video_durations(video_ids):
    durations = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="contentDetails",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute()
        for item in response['items']:
            duration = item['contentDetails']['duration']
            durations.append(duration)
    return durations

def parse_duration(duration):
    duration = isodate.parse_duration(duration)
    return duration.total_seconds()

def load_and_prepare_data(query):
    videos = fetch_videos(query, 50)
    df = pd.DataFrame(videos)

    file_path = 'video_details.csv'
    df.to_csv(file_path, index=False)

    video_data = pd.read_csv(file_path)
    video_data['statistics'] = video_data['statistics'].apply(ast.literal_eval)
    video_data['videoId'] = video_data['id'].apply(lambda x: eval(x)['videoId'])
    video_data['publishedAt'] = video_data['snippet'].apply(lambda x: eval(x)['publishedAt'])
    video_data['channelId'] = video_data['snippet'].apply(lambda x: eval(x)['channelId'])
    video_data['title'] = video_data['snippet'].apply(lambda x: eval(x)['title'])
    video_data['description'] = video_data['snippet'].apply(lambda x: eval(x)['description'])
    video_data['channelTitle'] = video_data['snippet'].apply(lambda x: eval(x)['channelTitle'])
    video_data['liveBroadcastContent'] = video_data['snippet'].apply(lambda x: eval(x)['liveBroadcastContent'])
    video_data['likeCount'] = video_data['statistics'].apply(lambda x: x['likeCount'])
    video_data['viewCount'] = video_data['statistics'].apply(lambda x: x['viewCount'])

    video_ids = video_data['videoId'].tolist()
    durations = get_video_durations(video_ids)
    video_data['duration'] = durations
    video_data['duration'] = video_data['duration'].apply(parse_duration)
    video_data_cleaned = video_data.drop(columns=['kind', 'id', 'snippet', 'statistics'])
    video_data_cleaned['publishedAt'] = pd.to_datetime(video_data_cleaned['publishedAt'])
    video_data_cleaned['combined'] = video_data_cleaned['title'] + ' ' + video_data_cleaned['description']

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(video_data_cleaned['combined'])
    scaler = StandardScaler()
    video_data_cleaned['duration_scaled'] = scaler.fit_transform(video_data_cleaned[['duration']])
    duration_sparse = csr_matrix(video_data_cleaned[['duration_scaled']].values)
    combined_features = hstack([tfidf_matrix, duration_sparse])

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(combined_features)
    
    return video_data_cleaned, combined_features, knn

def get_recommendations(video_id, video_data_cleaned, combined_features, knn, k=5):
    if video_id not in video_data_cleaned['videoId'].values:
        return None

    idx = video_data_cleaned.index[video_data_cleaned['videoId'] == video_id][0]
    query_features = combined_features[idx]
    distances, indices = knn.kneighbors(query_features, n_neighbors=k + 1)
    
    indices = indices.flatten()[1:]
    recommended_videos = video_data_cleaned.iloc[indices]

    recommendations = []
    for _, row in recommended_videos.iterrows():
        recommendations.append({
            'title': row['title'],
            'description': row['description'],
            'duration': row['duration']
        })

    return recommendations

def home(request):
    return render(request, 'appta/index.html')

def preferences_form(request):
    if request.method == 'POST':
        topic = request.POST.get('topic')
        difficulty = request.POST.get('difficulty')
        query = f"{topic} {difficulty}"

        video_data_cleaned, combined_features, knn = load_and_prepare_data(query)
        example_video_id = video_data_cleaned['videoId'].iloc[0]
        recommendations = get_recommendations(example_video_id, video_data_cleaned, combined_features, knn, k=5)

        return render(request, 'appta/recommendations.html', {'recommendations': recommendations})

    return render(request, 'appta/preferences.html')
