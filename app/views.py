import re
from datetime import datetime
from django.http import HttpResponse
from django.shortcuts import render, redirect
from app.forms import LogMessageForm
from app.models import LogMessage, ItemRecommender
from app.transform import DataTransform
from django.views.generic import ListView
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView

class ItemRecommenderView(APIView):
    def post(self, request, format=None):
        ItemRecommender().trainModels()
        return Response(status=status.HTTP_200_OK)

    def get(self, request):
        ItemRecommender().query()
        return Response(status=status.HTTP_200_OK)

class DataTransformView(APIView):
    def post(self, request, format=None):
        DataTransform().transform()
        return Response(status=status.HTTP_200_OK)

class HomeListView(ListView):
    """Renders the home page, with a list of all messages."""
    model = LogMessage

    def get_context_data(self, **kwargs):
        context = super(HomeListView, self).get_context_data(**kwargs)
        return context


def about(request):
    return render(request, "app/about.html")


def contact(request):
    return render(request, "app/contact.html")


def hello_there(request, name):
    return render(
        request,
        'app/hello_there.html',
        {
            'name': name,
            'date': datetime.now()
        }
    )


def log_message(request):
    form = LogMessageForm(request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            message = form.save(commit=False)
            message.log_date = datetime.now()
            message.save()
            return redirect("home")
    else:
        return render(request, "app/log_message.html", {"form": form})
