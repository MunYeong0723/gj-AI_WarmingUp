from django.urls import path
from . import views

urlpatterns = [
    path('', views.qrDecode, name='qrDecode'),
    path('viewCamera', views.viewCamera, name='viewCamera'),
]
