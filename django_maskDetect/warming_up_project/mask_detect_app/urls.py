from django.urls import path
from . import views

urlpatterns = [
    path('', views.maskDetect, name='maskDetect'),
    path('start/', views.camera, name='camera'),
]
