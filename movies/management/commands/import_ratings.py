"""
Management command to import user ratings from MovieLens ratings.csv.

Expected CSV format:
    userId,movieId,rating,timestamp

This command uses MovieLens links.csv to map movieId values to TMDB ids.
The movie catalog itself comes from TMDB, so only user ratings are imported here.

Usage:
    python manage.py import_ratings "path/to/ratings.csv" --clear
    python manage.py import_ratings "path/to/ratings.csv" --links "path/to/links.csv" --clear
"""

import csv
import sqlite3
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from tqdm import tqdm


class Command(BaseCommand):
    help = "Import MovieLens user ratings using TMDB mapping from links.csv."

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_file",
            type=str,
            help="Path to the MovieLens ratings.csv file.",
        )
        parser.add_argument(
            "--links",
            type=str,
            default="ml-32m/links.csv",
            help="Path to MovieLens links.csv used to map MovieLens ids to TMDB ids.",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            default=False,
            help="Clear imported ratings and auto-created user_* accounts before importing.",
        )

    def handle(self, *args, **options):
        ratings_path = Path(options["csv_file"])
        links_path = Path(options["links"])
        clear = options["clear"]

        if not ratings_path.exists():
            raise CommandError(f"File not found: {ratings_path}")
        if not links_path.exists():
            raise CommandError(f"Links file not found: {links_path}")

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

            self.stdout.write("  Loading movie database mapping...")
            movie_map = self._load_movie_map(cur)
            if not movie_map:
                raise CommandError("No movies found in the database. Import TMDB movie data first.")
            self.stdout.write(f"  {len(movie_map):,} movies loaded.")

            self.stdout.write("  Loading MovieLens -> TMDB mapping from links.csv...")
            ml_to_tmdb = self._build_mapping_from_links(links_path)
            self.stdout.write(f"  {len(ml_to_tmdb):,} mappings loaded.")

            if clear:
                deleted_ratings = cur.execute("DELETE FROM movies_userrating").rowcount
                deleted_users = cur.execute(
                    "DELETE FROM auth_user WHERE username LIKE 'user_%'"
                ).rowcount
                conn.commit()
                self.stdout.write(self.style.WARNING(
                    f"Cleared {deleted_ratings} ratings and {deleted_users} auto-created users."
                ))

            user_cache = {}
            users_created = 0
            ratings_upserted = 0
            skipped_no_mapping = 0
            skipped_no_movie = 0
            batch = []
            batch_size = 100000

            insert_user_sql = (
                "INSERT OR IGNORE INTO auth_user "
                "(password, last_login, is_superuser, username, last_name, email, is_staff, is_active, date_joined, first_name) "
                "VALUES (?, NULL, 0, ?, '', '', 0, 1, CURRENT_TIMESTAMP, '')"
            )
            upsert_rating_sql = (
                "INSERT INTO movies_userrating (user_id, movie_id, rating, timestamp) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(user_id, movie_id) DO UPDATE SET "
                "rating = excluded.rating, "
                "timestamp = CURRENT_TIMESTAMP"
            )

            self.stdout.write(self.style.NOTICE(f"\nImporting {ratings_path}..."))
            conn.execute("BEGIN")
            with ratings_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)

                for row in tqdm(reader, desc="Importing", unit="rows"):
                    try:
                        csv_user_id = row[0].strip()
                        ml_movie_id = int(row[1].strip())
                        rating_int = max(1, min(5, round(float(row[2].strip()))))
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
                            cur.execute("SELECT id FROM auth_user WHERE username = ?", (username,))
                            user_pk = cur.fetchone()[0]
                        user_cache[csv_user_id] = user_pk

                    batch.append((user_pk, movie_pk, rating_int))
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
            f"\nDone.\n"
            f"  Users loaded: {len(user_cache):,} (created: {users_created:,})\n"
            f"  Ratings written: {ratings_upserted:,}\n"
            f"  Skipped (no ML->TMDB mapping): {skipped_no_mapping:,}\n"
            f"  Skipped (movie not in DB): {skipped_no_movie:,}"
        ))

    @staticmethod
    def _load_movie_map(cursor) -> dict:
        cursor.execute("SELECT tmdb_id, id FROM movies_movie")
        return dict(cursor.fetchall())

    @staticmethod
    def _build_mapping_from_links(links_path: Path) -> dict:
        mapping = {}
        with links_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                try:
                    movie_id = int(row[0])
                    tmdb_raw = row[2].strip()
                    if not tmdb_raw:
                        continue
                    tmdb_id = int(float(tmdb_raw))
                    if tmdb_id > 0:
                        mapping[movie_id] = tmdb_id
                except (IndexError, ValueError, TypeError):
                    pass
        return mapping
