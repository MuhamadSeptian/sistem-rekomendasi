"""URL configuration for the movies app."""

from django.urls import path

from . import views

urlpatterns = [
    # Catalog & detail
    path("", views.catalog, name="catalog"),
    path("movie/<int:tmdb_id>/", views.movie_detail, name="movie_detail"),
    path("movie/<int:tmdb_id>/rate/", views.rate_movie, name="rate_movie"),
    path("movie/<int:tmdb_id>/delete-rating/", views.delete_rating, name="delete_rating"),

    # Dashboard
    path("dashboard/", views.dashboard, name="dashboard"),

    # Auth
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
]
