"""
Management command to ingest movies from the TMDB API.

Usage:
    python manage.py ingest_tmdb                        # default: semua strategi
    python manage.py ingest_tmdb --pages 100            # 100 halaman per endpoint
    python manage.py ingest_tmdb --year-start 2000      # discover mulai tahun 2000
    python manage.py ingest_tmdb --year-end 2026        # discover sampai tahun 2026
    python manage.py ingest_tmdb --skip-discover        # hanya popular + top_rated

Strategi pengambilan data:
  1. Popular  (maks 500 halaman = 10 000 film)
  2. Top Rated (maks 500 halaman = 10 000 film)
  3. Discover per tahun (maks 500 halaman/tahun) → ribuan film tambahan

The TMDB API key is read from settings.TMDB_API_KEY.
"""

import time
from datetime import datetime

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from tqdm import tqdm

from movies.models import Genre, Movie

TMDB_API_BASE = "https://api.themoviedb.org/3"
MAX_PAGES_PER_ENDPOINT = 500  # TMDB hard limit


class Command(BaseCommand):
    help = "Fetch movies from TMDB (popular, top_rated, discover) and store them."

    def add_arguments(self, parser):
        parser.add_argument(
            "--pages", type=int, default=50,
            help="Max pages to fetch per endpoint/year (20 movies/page). Default: 50",
        )
        parser.add_argument(
            "--year-start", type=int, default=2000,
            help="Start year for the discover endpoint. Default: 2000",
        )
        parser.add_argument(
            "--year-end", type=int, default=2026,
            help="End year for the discover endpoint. Default: 2026",
        )
        parser.add_argument(
            "--skip-discover", action="store_true", default=False,
            help="Skip the discover-by-year step (only fetch popular + top_rated).",
        )

    def handle(self, *args, **options):
        api_key = getattr(settings, "TMDB_API_KEY", None)
        if not api_key:
            raise CommandError(
                "TMDB_API_KEY is not set in settings.py. "
                "Please add it before running this command."
            )

        max_pages = min(options["pages"], MAX_PAGES_PER_ENDPOINT)
        year_start = options["year_start"]
        year_end = options["year_end"]
        skip_discover = options["skip_discover"]

        self.api_key = api_key
        self.headers = {"accept": "application/json"}
        self.movies_created = 0
        self.movies_updated = 0

        # ---- 1. Popular ----
        self.stdout.write(self.style.NOTICE("\n📥 Fetching POPULAR movies..."))
        self._fetch_endpoint(f"{TMDB_API_BASE}/movie/popular", max_pages)

        # ---- 2. Top Rated ----
        self.stdout.write(self.style.NOTICE("\n📥 Fetching TOP RATED movies..."))
        self._fetch_endpoint(f"{TMDB_API_BASE}/movie/top_rated", max_pages)

        # ---- 3. Discover by year ----
        if not skip_discover:
            self.stdout.write(self.style.NOTICE(
                f"\n📥 Fetching DISCOVER movies ({year_start}–{year_end})..."
            ))
            for year in range(year_start, year_end + 1):
                self.stdout.write(f"  Year {year}...")
                extra_params = {
                    "sort_by": "popularity.desc",
                    "primary_release_year": year,
                    "vote_count.gte": 10,
                }
                self._fetch_endpoint(
                    f"{TMDB_API_BASE}/discover/movie",
                    max_pages,
                    extra_params=extra_params,
                    label=f"Discover {year}",
                )

        # ---- Sync genre names ----
        self._sync_genre_names()

        total = self.movies_created + self.movies_updated
        self.stdout.write(self.style.SUCCESS(
            f"\n✅ Done!  Total movies in DB now: ~{Movie.objects.count()}\n"
            f"   Created {self.movies_created}, updated {self.movies_updated} "
            f"(processed {total})."
        ))

    # ------------------------------------------------------------------
    # Core: fetch one paginated endpoint
    # ------------------------------------------------------------------
    def _fetch_endpoint(self, url: str, max_pages: int,
                        extra_params: dict | None = None,
                        label: str = "") -> None:
        params_base = {"api_key": self.api_key, "language": "en-US"}
        if extra_params:
            params_base.update(extra_params)

        desc = label or url.split("/")[-1]

        for page in tqdm(range(1, max_pages + 1), desc=desc, leave=False):
            params = {**params_base, "page": page}

            try:
                response = requests.get(
                    url, params=params, headers=self.headers, timeout=15
                )
                # Rate-limit: TMDB allows ~40 req/10s on free tier
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2))
                    time.sleep(retry_after)
                    response = requests.get(
                        url, params=params, headers=self.headers, timeout=15
                    )
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                self.stderr.write(
                    self.style.ERROR(f"  Failed page {page}: {exc}")
                )
                continue

            data = response.json()
            results = data.get("results", [])
            total_pages_available = data.get("total_pages", 1)

            if not results:
                break

            for item in results:
                self._upsert_movie(item)

            # Don't request pages beyond what TMDB actually has
            if page >= total_pages_available:
                break

    # ------------------------------------------------------------------
    # Upsert a single movie dict from any TMDB endpoint
    # ------------------------------------------------------------------
    def _upsert_movie(self, item: dict) -> None:
        genre_ids = item.get("genre_ids", [])
        for gid in genre_ids:
            Genre.objects.get_or_create(
                id=gid, defaults={"name": f"Genre-{gid}"}
            )

        release_date = None
        raw_date = item.get("release_date")
        if raw_date:
            try:
                release_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError:
                pass

        movie, created = Movie.objects.update_or_create(
            tmdb_id=item["id"],
            defaults={
                "title": item.get("title") or "",
                "overview": item.get("overview") or "",
                "release_date": release_date,
                "poster_path": item.get("poster_path") or "",
                "vote_average": item.get("vote_average") or 0.0,
            },
        )
        movie.genres.set(genre_ids)

        if created:
            self.movies_created += 1
        else:
            self.movies_updated += 1

    # ------------------------------------------------------------------
    # Sync genre names from TMDB genre-list endpoint
    # ------------------------------------------------------------------
    def _sync_genre_names(self) -> None:
        url = "https://api.themoviedb.org/3/genre/movie/list"
        try:
            resp = requests.get(
                url,
                params={"api_key": self.api_key, "language": "en"},
                timeout=15,
            )
            resp.raise_for_status()
            for g in resp.json().get("genres", []):
                Genre.objects.filter(id=g["id"]).update(name=g["name"])
        except requests.exceptions.RequestException:
            pass
