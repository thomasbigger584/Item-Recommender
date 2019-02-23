from django.urls import path
from app import views

urlpatterns = [
    path("", views.home, name="home"),
    path("app/<name>", views.hello_there, name="hello_there"),
]
