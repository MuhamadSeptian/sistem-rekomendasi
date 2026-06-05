"""Admin registration for the movies app."""

from django.contrib import admin

from .models import Genre, Movie, UserRating


@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    list_display = ("id", "name")
    search_fields = ("name",)


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ("tmdb_id", "title", "vote_average", "release_date")
    search_fields = ("title",)
    list_filter = ("genres",)


@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ("user", "movie", "rating", "timestamp")
    list_filter = ("rating",)
    raw_id_fields = ("user", "movie")
