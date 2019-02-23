import re
from datetime import datetime
from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, "app/home.html")

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
