"""
Movies app models.

Defines Genre, Movie, and UserRating for the recommendation system.
"""

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Genre(models.Model):
    """A movie genre sourced from TMDB (uses TMDB's own ID as the PK)."""

    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class Movie(models.Model):
    """A movie record fetched from the TMDB API."""

    tmdb_id = models.IntegerField(unique=True, db_index=True)
    title = models.CharField(max_length=255)
    overview = models.TextField(blank=True, default="")
    release_date = models.DateField(null=True, blank=True)
    poster_path = models.CharField(max_length=255, blank=True, default="")
    vote_average = models.FloatField(default=0.0)
    genres = models.ManyToManyField(Genre, related_name="movies", blank=True)

    class Meta:
        ordering = ["-vote_average"]

    def __str__(self) -> str:
        return f"{self.title} ({self.tmdb_id})"


class UserRating(models.Model):
    """A user's rating (1-5) for a specific movie."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ratings",
    )
    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        related_name="ratings",
    )
    rating = models.FloatField(
        validators=[MinValueValidator(0.5), MaxValueValidator(5.0)],
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "movie")
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        return f"User {self.user_id} → {self.movie.title}: {self.rating}"
