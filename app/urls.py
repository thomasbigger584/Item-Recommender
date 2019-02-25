from django.urls import path
from app import views
from app.models import LogMessage

home_list_view = views.HomeListView.as_view(
    # :5 limits the results to the five most recent
    queryset=LogMessage.objects.order_by("-log_date")[:5],
    context_object_name="message_list",
    template_name="app/home.html",
)

item_recommender_view = views.ItemRecommenderView.as_view()

urlpatterns = [
    path("", home_list_view, name="home"),
    path("app/<name>", views.hello_there, name="hello_there"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("log/", views.log_message, name="log"),
    path('item-recommender/', item_recommender_view)
]
