"""Views for the movies app — catalog, detail, dashboard, auth, AJAX rating."""

import json
import logging

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import LoginForm, RegisterForm
from .models import Movie, UserRating
from .recommender import get_recommender

logger = logging.getLogger(__name__)

# Cache recommender instance at module level for performance
_recommender = None


def _get_cached_recommender():
    global _recommender
    if _recommender is None:
        _recommender = get_recommender()
        try:
            _recommender.train()
        except ValueError:
            logger.warning("Not enough data to train recommender.")
            _recommender = None
    return _recommender


# ------------------------------------------------------------------
# Catalog
# ------------------------------------------------------------------

def catalog(request):
    query = request.GET.get("q", "").strip()
    movies_qs = Movie.objects.prefetch_related("genres").all()

    if query:
        movies_qs = movies_qs.filter(
            Q(title__icontains=query) | Q(overview__icontains=query)
        )

    paginator = Paginator(movies_qs, 20)
    page = request.GET.get("page", 1)
    movies = paginator.get_page(page)

    return render(request, "movies/catalog.html", {
        "movies": movies,
        "query": query,
    })


# ------------------------------------------------------------------
# Movie Detail
# ------------------------------------------------------------------

def movie_detail(request, tmdb_id):
    movie = get_object_or_404(
        Movie.objects.prefetch_related("genres"), tmdb_id=tmdb_id
    )
    user_rating = None
    if request.user.is_authenticated:
        user_rating = UserRating.objects.filter(
            user=request.user, movie=movie
        ).first()

    # Build checked attributes for star rating (avoids custom templatetag)
    checked = {}
    if user_rating:
        for i in range(1, 6):
            checked[f"checked_{i}"] = "checked" if user_rating.rating == i else ""
    else:
        for i in range(1, 6):
            checked[f"checked_{i}"] = ""

    return render(request, "movies/detail.html", {
        "movie": movie,
        "user_rating": user_rating,
        **checked,
    })


# ------------------------------------------------------------------
# AJAX: Rate Movie
# ------------------------------------------------------------------

@login_required
@require_POST
def rate_movie(request, tmdb_id):
    movie = get_object_or_404(Movie, tmdb_id=tmdb_id)

    try:
        data = json.loads(request.body)
        rating_val = int(data.get("rating", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return JsonResponse({"success": False, "error": "Invalid data"}, status=400)

    if not 1 <= rating_val <= 5:
        return JsonResponse({"success": False, "error": "Rating must be 1-5"}, status=400)

    UserRating.objects.update_or_create(
        user=request.user,
        movie=movie,
        defaults={"rating": rating_val},
    )

    # Invalidate recommender cache so next dashboard load re-trains
    global _recommender
    _recommender = None

    return JsonResponse({"success": True, "rating": rating_val})


# ------------------------------------------------------------------
# AJAX: Delete Rating
# ------------------------------------------------------------------

@login_required
@require_POST
def delete_rating(request, tmdb_id):
    movie = get_object_or_404(Movie, tmdb_id=tmdb_id)
    deleted, _ = UserRating.objects.filter(
        user=request.user, movie=movie
    ).delete()

    if deleted:
        # Invalidate recommender cache
        global _recommender
        _recommender = None
        return JsonResponse({"success": True, "message": "Rating deleted"})

    return JsonResponse({"success": False, "error": "No rating found"}, status=404)


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

@login_required
def dashboard(request):
    user_ratings = (
        UserRating.objects
        .filter(user=request.user)
        .select_related("movie")
        .prefetch_related("movie__genres")
        .order_by("-timestamp")
    )

    recommendations = []
    rec_details = {}  # tmdb_id → {cbf_score, cf_score, hybrid_score}
    rec_method_info = {}  # metadata about the recommender

    rec = _get_cached_recommender()
    if rec:
        try:
            # For 0 ratings, detailed predict falls back to CBF natively (pseudo-profile)
            # For >= 1 ratings, it will use Hybrid automatically.
            detailed = rec.predict_detailed(user_id=request.user.id, k=10)
            top_ids = [d["tmdb_id"] for d in detailed]
            rec_details = {d["tmdb_id"]: d for d in detailed}

            recommendations = list(
                Movie.objects.filter(tmdb_id__in=top_ids)
                .prefetch_related("genres")
            )
            # Preserve the recommender's ordering
            id_order = {tid: i for i, tid in enumerate(top_ids)}
            recommendations.sort(key=lambda m: id_order.get(m.tmdb_id, 999))

            if not user_ratings.exists():
                rec_method_info = {
                    "cbf_weight": 1.0,
                    "cf_weight": 0.0,
                    "cbf_algo": "TF-IDF + Cosine Similarity",
                    "cf_algo": "SVD (Butuh minimal 1 rating)",
                    "hybrid_algo": "Content-Based Only (Sistem User Baru)",
                }
            else:
                rec_method_info = {
                    "cbf_weight": rec.cbf_weight,
                    "cf_weight": rec.cf_weight,
                    "cbf_algo": "TF-IDF + Cosine Similarity",
                    "cf_algo": "SVD (Singular Value Decomposition)",
                    "hybrid_algo": "Weighted Hybrid Filtering",
                }
        except Exception as exc:
                logger.error("Recommendation error: %s", exc)

    return render(request, "movies/dashboard.html", {
        "recommendations": recommendations,
        "rec_details": rec_details,
        "rec_method_info": rec_method_info,
        "user_ratings": user_ratings,
    })


# ------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------

def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(
                request,
                f"Akun '{user.username}' berhasil dibuat! 🎉 Silakan login untuk melanjutkan."
            )
            return redirect("login")
    else:
        form = RegisterForm()

    return render(request, "auth/register.html", {"form": form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            messages.success(request, "Welcome back! 👋")
            next_url = request.GET.get("next", "catalog")
            return redirect(next_url)
    else:
        form = LoginForm()

    return render(request, "auth/login.html", {"form": form})


def logout_view(request):
    logout(request)
    messages.info(request, "You've been logged out.")
    return redirect("catalog")
