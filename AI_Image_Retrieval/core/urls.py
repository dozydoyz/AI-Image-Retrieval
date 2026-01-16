from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('history/', views.history, name='history'),
    path('register/', views.register, name='register'),
    path('search/<int:pk>/', views.search_again, name='search_again'),
]