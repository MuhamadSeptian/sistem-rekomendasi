"""
Management command to load MovieLens ml-latest-small dataset into the database.

Reads movies.csv, links.csv, and ratings.csv from the MovieLens dataset.
Fetches movie metadata (title, overview, release_date, poster, vote_average,
genres) from the TMDB API using the tmdbId mapping in links.csv.

Usage:
    python manage.py load_movielens
    python manage.py load_movielens --skip-tmdb
    python manage.py load_movielens --movies-csv path/to/movies.csv --ratings-csv path/to/ratings.csv --links-csv path/to/links.csv
"""

import csv
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from tqdm import tqdm

from movies.models import Genre, Movie

TMDB_API_BASE = "https://api.themoviedb.org/3"


class Command(BaseCommand):
    help = "Load MovieLens ml-latest-small dataset with TMDB metadata into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--movies-csv",
            type=str,
            default="ml-latest-small/movies.csv",
            help="Path to MovieLens movies.csv",
        )
        parser.add_argument(
            "--ratings-csv",
            type=str,
            default="ml-latest-small/ratings.csv",
            help="Path to MovieLens ratings.csv",
        )
        parser.add_argument(
            "--links-csv",
            type=str,
            default="ml-latest-small/links.csv",
            help="Path to MovieLens links.csv",
        )
        parser.add_argument(
            "--skip-tmdb",
            action="store_true",
            default=False,
            help="Skip fetching TMDB metadata (use only CSV data).",
        )

    def handle(self, *args, **options):
        movies_path = Path(options["movies_csv"])
        ratings_path = Path(options["ratings_csv"])
        links_path = Path(options["links_csv"])
        self.skip_tmdb = options["skip_tmdb"]

        for p, label in [(movies_path, "movies.csv"), (ratings_path, "ratings.csv"), (links_path, "links.csv")]:
            if not p.exists():
                raise CommandError(f"File not found: {p} ({label})")

        self.api_key = getattr(settings, "TMDB_API_KEY", None)
        if not self.skip_tmdb and not self.api_key:
            raise CommandError(
                "TMDB_API_KEY not set in settings.py. "
                "Set it or use --skip-tmdb to skip TMDB metadata."
            )

        # ── Step 1: Read CSV data ──
        self.stdout.write(self.style.NOTICE("\n[1/5] Reading CSV files..."))
        ml_movies = self._read_movies_csv(movies_path)
        ml_to_tmdb = self._read_links_csv(links_path)
        self.stdout.write(f"  Movies in CSV: {len(ml_movies):,}")
        self.stdout.write(f"  Links (ML->TMDB): {len(ml_to_tmdb):,}")

        # ── Step 2: Clear old data ──
        self.stdout.write(self.style.WARNING("\n[2/5] Clearing old data..."))
        self._clear_database()

        # ── Step 3: Sync genre names from TMDB ──
        if not self.skip_tmdb:
            self.stdout.write(self.style.NOTICE("\n[3/5] Syncing genre list from TMDB..."))
            self._sync_genre_names()

        # ── Step 4: Load movies with TMDB metadata ──
        self.stdout.write(self.style.NOTICE("\n[4/5] Loading movies..."))
        movies_loaded = self._load_movies(ml_movies, ml_to_tmdb)
        self.stdout.write(self.style.SUCCESS(f"  Done: {movies_loaded:,} movies loaded."))

        # ── Step 5: Import ratings ──
        self.stdout.write(self.style.NOTICE("\n[5/5] Importing ratings..."))
        self._import_ratings(ratings_path, ml_to_tmdb)

        self.stdout.write(self.style.SUCCESS(
            f"\nAll done!\n"
            f"   Movies: {Movie.objects.count():,}\n"
            f"   Genres: {Genre.objects.count():,}\n"
        ))

    # ==================================================================
    #  CSV Readers
    # ==================================================================

    @staticmethod
    def _read_movies_csv(path: Path) -> dict:
        """Return {movieId: {'title': ..., 'genres': [...]}}."""
        movies = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    movie_id = int(row["movieId"])
                    title = row["title"].strip()
                    genres_raw = row["genres"].strip()
                    genres = genres_raw.split("|") if genres_raw and genres_raw != "(no genres listed)" else []
                    movies[movie_id] = {"title": title, "genres": genres}
                except (KeyError, ValueError):
                    continue
        return movies

    @staticmethod
    def _read_links_csv(path: Path) -> dict:
        """Return {movieId: tmdbId}."""
        mapping = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    movie_id = int(row["movieId"])
                    tmdb_raw = row["tmdbId"].strip()
                    if not tmdb_raw:
                        continue
                    tmdb_id = int(float(tmdb_raw))
                    if tmdb_id > 0:
                        mapping[movie_id] = tmdb_id
                except (KeyError, ValueError, TypeError):
                    continue
        return mapping

    # ==================================================================
    #  Clear Database
    # ==================================================================

    def _clear_database(self):
        """Remove all ratings, movies, genres, and auto-created users."""
        from django.contrib.auth.models import User
        from movies.models import UserRating

        deleted_ratings = UserRating.objects.all().delete()[0]
        self.stdout.write(f"  Deleted {deleted_ratings:,} ratings")

        # Clear movie-genre M2M then movies
        Movie.genres.through.objects.all().delete()
        deleted_movies = Movie.objects.all().delete()[0]
        self.stdout.write(f"  Deleted {deleted_movies:,} movies")

        deleted_genres = Genre.objects.all().delete()[0]
        self.stdout.write(f"  Deleted {deleted_genres:,} genres")

        deleted_users = User.objects.filter(username__startswith="user_").delete()[0]
        self.stdout.write(f"  Deleted {deleted_users:,} auto-created users")

    # ==================================================================
    #  Sync Genre Names from TMDB
    # ==================================================================

    def _sync_genre_names(self):
        """Fetch genre list from TMDB and create Genre objects."""
        url = f"{TMDB_API_BASE}/genre/movie/list"
        try:
            resp = requests.get(
                url,
                params={"api_key": self.api_key, "language": "en"},
                timeout=15,
            )
            resp.raise_for_status()
            for g in resp.json().get("genres", []):
                Genre.objects.update_or_create(
                    id=g["id"],
                    defaults={"name": g["name"]},
                )
            self.stdout.write(f"  {Genre.objects.count()} genres synced.")
        except requests.exceptions.RequestException as e:
            self.stderr.write(self.style.ERROR(f"  Failed to sync genres: {e}"))

    # ==================================================================
    #  Load Movies (with TMDB metadata)
    # ==================================================================

    def _load_movies(self, ml_movies: dict, ml_to_tmdb: dict) -> int:
        """Load movies from CSV, enriching with TMDB metadata."""
        loaded = 0
        failed_tmdb = 0
        no_tmdb_id = 0

        # Build list of (movieId, tmdbId) pairs
        movie_list = []
        for ml_id, info in ml_movies.items():
            tmdb_id = ml_to_tmdb.get(ml_id)
            if tmdb_id:
                movie_list.append((ml_id, tmdb_id, info))
            else:
                no_tmdb_id += 1

        if no_tmdb_id > 0:
            self.stdout.write(f"  WARNING: {no_tmdb_id} movies have no TMDB mapping, skipped.")

        for ml_id, tmdb_id, info in tqdm(movie_list, desc="Loading movies", unit="movies"):
            if not self.skip_tmdb:
                success = self._fetch_and_save_tmdb(tmdb_id, info)
                if success:
                    loaded += 1
                else:
                    failed_tmdb += 1
                    # Fallback: save with CSV data only
                    self._save_movie_from_csv(tmdb_id, info)
                    loaded += 1
            else:
                self._save_movie_from_csv(tmdb_id, info)
                loaded += 1

        if failed_tmdb > 0:
            self.stdout.write(self.style.WARNING(
                f"  WARNING: {failed_tmdb} movies failed TMDB fetch (used CSV fallback)."
            ))

        return loaded

    def _fetch_and_save_tmdb(self, tmdb_id: int, csv_info: dict) -> bool:
        """Fetch a single movie from TMDB API and save to database."""
        url = f"{TMDB_API_BASE}/movie/{tmdb_id}"
        try:
            resp = requests.get(
                url,
                params={"api_key": self.api_key, "language": "en-US"},
                headers={"accept": "application/json"},
                timeout=15,
            )

            # Handle rate limiting
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                time.sleep(retry_after)
                resp = requests.get(
                    url,
                    params={"api_key": self.api_key, "language": "en-US"},
                    headers={"accept": "application/json"},
                    timeout=15,
                )

            resp.raise_for_status()
            data = resp.json()

            # Parse release date
            release_date = None
            raw_date = data.get("release_date")
            if raw_date:
                try:
                    release_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
                except ValueError:
                    pass

            # Upsert genres
            genre_ids = []
            for g in data.get("genres", []):
                Genre.objects.get_or_create(
                    id=g["id"],
                    defaults={"name": g["name"]},
                )
                genre_ids.append(g["id"])

            # Create/update movie
            movie, _ = Movie.objects.update_or_create(
                tmdb_id=tmdb_id,
                defaults={
                    "title": data.get("title") or csv_info["title"],
                    "overview": data.get("overview") or "",
                    "release_date": release_date,
                    "poster_path": data.get("poster_path") or "",
                    "vote_average": data.get("vote_average") or 0.0,
                },
            )
            movie.genres.set(genre_ids)
            return True

        except requests.exceptions.RequestException:
            return False

    def _save_movie_from_csv(self, tmdb_id: int, info: dict):
        """Save a movie using only CSV data (no TMDB metadata)."""
        # Map CSV genre names to existing Genre objects, or create simple ones
        genre_ids = []
        for genre_name in info.get("genres", []):
            genre, _ = Genre.objects.get_or_create(
                name=genre_name,
                defaults={"id": abs(hash(genre_name)) % 100000},
            )
            genre_ids.append(genre.id)

        movie, _ = Movie.objects.update_or_create(
            tmdb_id=tmdb_id,
            defaults={
                "title": info["title"],
                "overview": "",
                "release_date": None,
                "poster_path": "",
                "vote_average": 0.0,
            },
        )
        movie.genres.set(genre_ids)

    # ==================================================================
    #  Import Ratings (high-performance raw SQL)
    # ==================================================================

    def _import_ratings(self, ratings_path: Path, ml_to_tmdb: dict):
        """Import ratings using raw SQL for performance."""
        db_name = connection.settings_dict["NAME"]
        db_path = Path(db_name)
        if not db_path.is_absolute():
            db_path = Path(settings.BASE_DIR) / db_path

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA cache_size = -200000")
            cur = conn.cursor()

            # Build tmdb_id → movie PK map
            cur.execute("SELECT tmdb_id, id FROM movies_movie")
            movie_map = dict(cur.fetchall())
            self.stdout.write(f"  {len(movie_map):,} movies in DB for mapping.")

            user_cache = {}
            users_created = 0
            ratings_upserted = 0
            skipped_no_mapping = 0
            skipped_no_movie = 0
            batch = []
            batch_size = 50000

            insert_user_sql = (
                "INSERT OR IGNORE INTO auth_user "
                "(password, last_login, is_superuser, username, last_name, email, "
                "is_staff, is_active, date_joined, first_name) "
                "VALUES (?, NULL, 0, ?, '', '', 0, 1, CURRENT_TIMESTAMP, '')"
            )
            upsert_rating_sql = (
                "INSERT INTO movies_userrating (user_id, movie_id, rating, timestamp) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id, movie_id) DO UPDATE SET "
                "rating = excluded.rating, "
                "timestamp = CURRENT_TIMESTAMP"
            )

            conn.execute("BEGIN")
            with ratings_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header

                for row in tqdm(reader, desc="Importing ratings", unit="rows"):
                    try:
                        csv_user_id = row[0].strip()
                        ml_movie_id = int(row[1].strip())
                        rating_val = float(row[2].strip())
                        # Clamp to valid range
                        rating_val = max(0.5, min(5.0, rating_val))
                    except (IndexError, ValueError, AttributeError):
                        continue

                    tmdb_id = ml_to_tmdb.get(ml_movie_id)
                    if tmdb_id is None:
                        skipped_no_mapping += 1
                        continue

                    movie_pk = movie_map.get(tmdb_id)
                    if movie_pk is None:
                        skipped_no_movie += 1
                        continue

                    user_pk = user_cache.get(csv_user_id)
                    if user_pk is None:
                        username = f"user_{csv_user_id}"
                        cur.execute(insert_user_sql, ("!", username))
                        if cur.rowcount == 1:
                            users_created += 1
                            user_pk = cur.lastrowid
                        else:
                            cur.execute(
                                "SELECT id FROM auth_user WHERE username = ?",
                                (username,),
                            )
                            user_pk = cur.fetchone()[0]
                        user_cache[csv_user_id] = user_pk

                    batch.append((user_pk, movie_pk, rating_val))
                    if len(batch) >= batch_size:
                        cur.executemany(upsert_rating_sql, batch)
                        ratings_upserted += len(batch)
                        batch.clear()
                        conn.commit()
                        conn.execute("BEGIN")

            if batch:
                cur.executemany(upsert_rating_sql, batch)
                ratings_upserted += len(batch)
                batch.clear()
            conn.commit()

        finally:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.close()

        self.stdout.write(self.style.SUCCESS(
            f"\n  Ratings imported!\n"
            f"     Users created: {users_created:,}\n"
            f"     Ratings written: {ratings_upserted:,}\n"
            f"     Skipped (no ML->TMDB mapping): {skipped_no_mapping:,}\n"
            f"     Skipped (movie not in DB): {skipped_no_movie:,}"
        ))
