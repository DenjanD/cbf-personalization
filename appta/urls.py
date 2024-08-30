from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('select_video/', views.select_video, name='select_video'),
    path('preferences/', views.preferences_form, name='preferences_form'),
    path('', views.home, name='home'),  # Root URL points to preferences_form
]
