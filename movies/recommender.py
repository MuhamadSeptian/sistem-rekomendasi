"""
Hybrid Recommender Engine.

Provides three recommendation strategies:
1. **ContentBasedRecommender (CBF)** — TF-IDF + Cosine Similarity
   on movie genre & overview text.
2. **CollaborativeRecommender (CF)** — Truncated SVD (Singular Value
   Decomposition) via ``scipy.sparse.linalg.svds``.
3. **HybridRecommender** — Weighted Hybrid that combines CBF and CF
   scores with configurable weights  (default: α=0.4 CBF, β=0.6 CF).

Use ``get_recommender()`` to get a ready-to-use ``HybridRecommender``.
If the user has no ratings yet, the system gracefully falls back to
pure content-based recommendations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from movies.models import Movie, UserRating

logger = logging.getLogger(__name__)


# ======================================================================
# 1.  Content-Based Filtering  —  TF-IDF + Cosine Similarity
# ======================================================================

class ContentBasedRecommender:
    """
    Content-based recommender that builds a TF-IDF matrix from each
    movie's combined genre names and overview text, then uses cosine
    similarity to find movies most similar to a user's highly-rated ones.
    """

    def __init__(self):
        self._movie_ids: List[int] = []
        self._tfidf_matrix = None          # sparse TF-IDF matrix
        self._tmdb_to_idx: Dict[int, int] = {}

    # ---- build TF-IDF matrix ----

    def train(self) -> None:
        movies = list(Movie.objects.all())
        if not movies:
            raise ValueError("No movies in the database. Run ingest_tmdb first.")

        self._movie_ids = [m.tmdb_id for m in movies]
        self._tmdb_to_idx = {tid: i for i, tid in enumerate(self._movie_ids)}

        # Pre-load all movie→genre mappings via the through table
        from django.db import connection
        genre_map: Dict[int, List[str]] = {m.pk: [] for m in movies}
        movie_pk_to_idx = {m.pk: i for i, m in enumerate(movies)}

        for mg in Movie.genres.through.objects.select_related("genre").all():
            pk = mg.movie_id
            if pk in genre_map:
                genre_map[pk].append(mg.genre.name)

        # Combine title, genre names, and overview into a single text corpus
        corpus: List[str] = []
        for m in movies:
            genres_str = " ".join(genre_map.get(m.pk, []))
            # Use equal weights (1x) for title, genres, and overview
            text = f"{m.title} {genres_str} {m.overview}"
            corpus.append(text)

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = vectorizer.fit_transform(corpus)

        logger.info(
            "CBF (TF-IDF) ready — %d movies, %d features.",
            len(movies),
            self._tfidf_matrix.shape[1],
        )

    # ---- build user profile (cached) ----

    def _build_profile(self, user_id: int, user_ratings=None):
        """Return (profile_vector, rated_tmdb_ids) for a user."""
        if user_ratings is not None:
            ratings = user_ratings
        else:
            ratings = list(
                UserRating.objects.filter(user_id=user_id)
                .values_list("movie__tmdb_id", "rating")
            )

        # Untuk profil "Minat", kita hanya peduli pada rating yang bersifat positif/suka (>= 3.0)
        # Rating buruk (1 atau 2) tidak boleh dijadikan acuan untuk mencari kemiripan konten.
        positive_ratings = [(tid, r) for tid, r in ratings if r >= 3.0]

        if not positive_ratings:
            # Fallback for cold-start (new user with 0 ratings OR 0 positive ratings):
            # Create a pseudo-profile from globally popular & highly rated movies
            from django.db.models import Count, Avg
            top_movies = (
                UserRating.objects
                .values("movie__tmdb_id")
                .annotate(count=Count("id"), avg=Avg("rating"))
                .filter(count__gte=5, avg__gte=4.0)
                .order_by("-count")[:10]
            )
            positive_ratings = [(m["movie__tmdb_id"], m["avg"]) for m in top_movies]
            if not positive_ratings:
                raise ValueError(f"User {user_id} has no positive ratings and no fallback data available.")

        profile = np.zeros(self._tfidf_matrix.shape[1])
        total_weight = 0.0
        rated_ids = {tid for tid, _ in ratings} # Keep track of ALL rated movies to exclude them from predictions

        for tmdb_id, rating in positive_ratings:
            idx = self._tmdb_to_idx.get(tmdb_id)
            if idx is not None:
                # Modifikasi tambahan: Berikan bobot lebih murni dengan mengurangi threshold dasar (misal -3.0)
                # Secara operasional jika kita sudah filter r>=3.0, maka kita bisa pakai rating tsb langsung.
                profile += self._tfidf_matrix[idx].toarray().flatten() * rating
                total_weight += rating

        if total_weight > 0:
            profile /= total_weight

        return profile, rated_ids

    # ---- predict ----

    def predict(self, user_id: int, k: int = 10, user_ratings=None) -> List[Tuple[int, float]]:
        """
        Return a list of ``(tmdb_id, score)`` pairs for the top-*k* movies
        recommended for *user_id* based on content similarity.
        """
        if self._tfidf_matrix is None:
            raise RuntimeError("Model not trained. Call train() first.")

        profile, rated_ids = self._build_profile(user_id, user_ratings=user_ratings)

        # Cosine similarity between the user profile and every movie
        scores = cosine_similarity(
            profile.reshape(1, -1), self._tfidf_matrix
        )[0]

        # Zero-out already-rated movies
        for tmdb_id in rated_ids:
            idx = self._tmdb_to_idx.get(tmdb_id)
            if idx is not None:
                scores[idx] = -1.0

        top_indices = np.argsort(-scores)[:k]
        return [(self._movie_ids[i], float(scores[i])) for i in top_indices]

    # ---- score a single movie for a user ----

    def score_one(self, user_id: int, tmdb_id: int, _profile=None, user_ratings=None) -> float:
        """
        Return the cosine-similarity score for a single movie.
        Accepts an optional pre-built profile to avoid recomputation.
        """
        if self._tfidf_matrix is None:
            return 0.0
        idx = self._tmdb_to_idx.get(tmdb_id)
        if idx is None:
            return 0.0
        if _profile is None:
            try:
                _profile, _ = self._build_profile(user_id, user_ratings=user_ratings)
            except ValueError:
                return 0.0
        sim = cosine_similarity(
            _profile.reshape(1, -1), self._tfidf_matrix[idx]
        )[0, 0]
        return float(sim)

    # convenience: return just IDs
    def predict_ids(self, user_id: int, k: int = 10) -> List[int]:
        return [tid for tid, _ in self.predict(user_id, k)]


# ======================================================================
# 2.  Collaborative Filtering  —  SVD  (scipy truncated SVD)
# ======================================================================

class CollaborativeRecommender:
    """
    Collaborative-filtering recommender using **Truncated SVD** (Singular
    Value Decomposition) implemented with ``scipy.sparse.linalg.svds``.

    The algorithm decomposes the user-item rating matrix  R  into three
    low-rank matrices:

        R ≈ U · Σ · Vᵀ

    where:
      - U  (n_users × k)   — user latent-factor matrix
      - Σ  (k × k)         — diagonal matrix of singular values
      - Vᵀ (k × n_items)   — item latent-factor matrix

    The predicted rating for user *u* and item *i* is:

        r̂(u, i) = μ + bᵤ + bᵢ + Uᵤ · Σ · Vᵢᵀ

    where μ is the global mean rating, bᵤ the user bias, and bᵢ the
    item bias.
    """

    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors
        # Learned parameters (set in train())
        self._U = None               # user latent factors
        self._sigma = None           # singular values
        self._Vt = None              # item latent factors
        self._global_mean: float = 0.0
        self._user_bias = None       # per-user bias
        self._item_bias = None       # per-item bias
        self._user_to_idx: Dict[int, int] = {}
        self._item_to_idx: Dict[int, int] = {}
        self._idx_to_item: Dict[int, int] = {}
        self._n_users: int = 0
        self._n_items: int = 0
        self._n_ratings: int = 0
        self._trained = False

    # ---- train ----

    def train(self, ratings_data=None) -> None:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds

        if ratings_data is not None:
            ratings = ratings_data
        else:
            ratings_qs = UserRating.objects.values_list(
                "user_id", "movie__tmdb_id", "rating"
            )
            ratings = list(ratings_qs)
        if not ratings:
            raise ValueError("No ratings found. Cannot train SVD.")

        # Build user / item index mappings
        user_ids_set = sorted({r[0] for r in ratings})
        item_ids_set = sorted({r[1] for r in ratings})

        self._user_to_idx = {uid: i for i, uid in enumerate(user_ids_set)}
        self._item_to_idx = {iid: i for i, iid in enumerate(item_ids_set)}
        self._idx_to_item = {i: iid for iid, i in self._item_to_idx.items()}

        self._n_users = len(user_ids_set)
        self._n_items = len(item_ids_set)
        self._n_ratings = len(ratings)

        # Build sparse rating matrix  (users × items)
        rows, cols, vals = [], [], []
        for uid, iid, rating in ratings:
            rows.append(self._user_to_idx[uid])
            cols.append(self._item_to_idx[iid])
            vals.append(float(rating))

        R = csr_matrix(
            (vals, (rows, cols)),
            shape=(self._n_users, self._n_items),
            dtype=np.float64,
        )

        # Global mean (over observed entries only)
        self._global_mean = float(np.mean(vals))

        # User and item biases
        R_dense = R.toarray()
        R_mask = (R_dense != 0).astype(np.float64)

        user_sums = R_dense.sum(axis=1)
        user_counts = R_mask.sum(axis=1)
        self._user_bias = np.where(
            user_counts > 0,
            user_sums / user_counts - self._global_mean,
            0.0,
        )

        item_sums = R_dense.sum(axis=0)
        item_counts = R_mask.sum(axis=0)
        self._item_bias = np.where(
            item_counts > 0,
            item_sums / item_counts - self._global_mean,
            0.0,
        )

        # Centre the matrix: subtract global mean + biases for SVD
        bias_matrix = self._global_mean + self._user_bias[:, np.newaxis] + self._item_bias[np.newaxis, :]
        R_centred = R_dense - (R_mask * bias_matrix)
                # leave unobserved as 0

        # Truncated SVD
        k = min(self.n_factors, min(self._n_users, self._n_items) - 1)
        U, sigma, Vt = svds(csr_matrix(R_centred), k=k)

        # svds returns singular values in ascending order — reverse
        idx = np.argsort(-sigma)
        self._U = U[:, idx]
        self._sigma = sigma[idx]
        self._Vt = Vt[idx, :]
        self._trained = True

        logger.info(
            "CF (SVD) trained — %d users, %d items, %d ratings, %d factors.",
            self._n_users, self._n_items, self._n_ratings, k,
        )

    # ---- predict a single rating ----

    def _predict_rating(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for internal indices."""
        dot = self._U[user_idx] @ np.diag(self._sigma) @ self._Vt[:, item_idx]
        est = self._global_mean + self._user_bias[user_idx] + self._item_bias[item_idx] + dot
        # Clamp to valid range
        return float(np.clip(est, 1.0, 5.0))

    # ---- top-k predict ----

    def predict(self, user_id: int, k: int = 10, rated_tmdb_ids=None) -> List[Tuple[int, float]]:
        """
        Return ``(tmdb_id, estimated_rating)`` for the top-*k* unseen movies.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        user_idx = self._user_to_idx.get(user_id)
        if user_idx is None:
            # Unknown user — return empty (cold start)
            return []

        if rated_tmdb_ids is not None:
            rated_ids = set(rated_tmdb_ids)
        else:
            rated_ids = set(
                UserRating.objects.filter(user_id=user_id)
                .values_list("movie__tmdb_id", flat=True)
            )

        predictions = []
        for item_idx in range(self._n_items):
            tmdb_id = self._idx_to_item[item_idx]
            if tmdb_id in rated_ids:
                continue
            est = self._predict_rating(user_idx, item_idx)
            predictions.append((tmdb_id, est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:k]

    def score_one(self, user_id: int, tmdb_id: int) -> float:
        """Return the predicted rating for a single user-item pair."""
        if not self._trained:
            return 0.0
        user_idx = self._user_to_idx.get(user_id)
        item_idx = self._item_to_idx.get(tmdb_id)
        if user_idx is None or item_idx is None:
            return 0.0
        return self._predict_rating(user_idx, item_idx)

    def predict_ids(self, user_id: int, k: int = 10) -> List[int]:
        return [tid for tid, _ in self.predict(user_id, k)]


# ======================================================================
# 3.  Weighted Hybrid Filtering
# ======================================================================

class HybridRecommender:
    """
    Weighted Hybrid recommender that linearly combines scores from
    Content-Based Filtering (CBF) and Collaborative Filtering (CF).

        hybrid_score = α × cbf_score_norm + β × cf_score_norm

    where  α + β = 1  (default α = 0.4, β = 0.6).

    If the user has no ratings (cold-start), the system falls back to
    pure CBF using popular movies as a pseudo-profile.
    """

    def __init__(self, cbf_weight: float = 0.4, cf_weight: float = 0.6):
        if abs(cbf_weight + cf_weight - 1.0) >= 1e-6:
            raise ValueError("Weights must sum to 1.")
        self.cbf_weight = cbf_weight
        self.cf_weight = cf_weight
        self.cbf = ContentBasedRecommender()
        self.cf = CollaborativeRecommender()
        self._trained = False

    # ---- train both sub-models ----

    def train(self, ratings_data=None) -> None:
        self.cbf.train()
        try:
            self.cf.train(ratings_data=ratings_data)
        except Exception as exc:
            logger.warning("CF training skipped: %s — will use CBF only.", exc)
        self._trained = True

    # ---- predict (weighted merge) ----

    def predict(self, user_id: int, k: int = 10, user_ratings=None) -> List[int]:
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        candidates = self._build_candidates(user_id, k, user_ratings=user_ratings)
        if not candidates:
            return []
        return [c["tmdb_id"] for c in candidates]

    # ---- detailed predict (with per-method scores for dashboard) ----

    def predict_detailed(self, user_id: int, k: int = 10, user_ratings=None) -> List[dict]:
        """
        Return a list of dicts with tmdb_id, cbf_score, cf_score,
        hybrid_score for the top-k recommendations.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self._build_candidates(user_id, k, user_ratings=user_ratings)

    # ---- core: build candidate list with BOTH scores filled in ----

    def _build_candidates(self, user_id: int, k: int = 10, user_ratings=None) -> List[dict]:
        """
        1. Collect top candidates from CBF and CF independently.
        2. For EVERY candidate, compute the score from BOTH methods
           so no movie ends up with 0.0 on one side.
        3. Normalise and merge with weighted hybrid formula.
        """
        cbf_results: List[Tuple[int, float]] = []
        cf_results: List[Tuple[int, float]] = []

        # Build CBF user profile once (reused for score_one calls)
        cbf_profile = None
        pool_size = max(k, 10)

        try:
            cbf_profile, _ = self.cbf._build_profile(user_id, user_ratings=user_ratings)
            cbf_results = self.cbf.predict(user_id, k=pool_size, user_ratings=user_ratings)
        except (ValueError, RuntimeError):
            pass

        if self.cf._trained:
            try:
                rated_ids = [tid for tid, _ in user_ratings] if user_ratings is not None else None
                cf_results = self.cf.predict(user_id, k=pool_size, rated_tmdb_ids=rated_ids)
            except Exception:
                pass

        if not cbf_results and not cf_results:
            return []

        # If only one model has results, use that alone
        if not cf_results:
            cbf_norm = self._normalise(cbf_results)
            return [
                {"tmdb_id": tid, "cbf_score": round(s, 4),
                 "cf_score": 0.0, "hybrid_score": round(s, 4)}
                for tid, s in sorted(cbf_norm.items(), key=lambda x: -x[1])[:k]
            ]
        if not cbf_results:
            cf_norm = self._normalise(cf_results)
            return [
                {"tmdb_id": tid, "cbf_score": 0.0,
                 "cf_score": round(s, 4), "hybrid_score": round(s, 4)}
                for tid, s in sorted(cf_norm.items(), key=lambda x: -x[1])[:k]
            ]

        # ---- Collect all candidate IDs from both sides ----
        cbf_dict = dict(cbf_results)   # raw scores (not yet normalised)
        cf_dict = dict(cf_results)
        all_ids = set(cbf_dict.keys()) | set(cf_dict.keys())

        # ---- Fill missing scores by querying the other model ----
        raw_cbf: Dict[int, float] = {}
        raw_cf: Dict[int, float] = {}

        for tid in all_ids:
            # CBF score
            if tid in cbf_dict:
                raw_cbf[tid] = cbf_dict[tid]
            else:
                raw_cbf[tid] = self.cbf.score_one(user_id, tid, _profile=cbf_profile, user_ratings=user_ratings)

            # CF score
            if tid in cf_dict:
                raw_cf[tid] = cf_dict[tid]
            else:
                raw_cf[tid] = self.cf.score_one(user_id, tid)

        # ---- Min-Max Normalization (0-1) ----
        
        cbf_min = min(raw_cbf.values()) if raw_cbf else 0
        cbf_max = max(raw_cbf.values()) if raw_cbf else 1
        if cbf_max == cbf_min: cbf_max += 1e-9
        
        cf_min = min(raw_cf.values()) if raw_cf else 0
        cf_max = max(raw_cf.values()) if raw_cf else 1
        if cf_max == cf_min: cf_max += 1e-9

        cbf_norm = {}
        cf_norm = {}
        for tid in all_ids:
            cbf_norm[tid] = (raw_cbf[tid] - cbf_min) / (cbf_max - cbf_min)
            cf_norm[tid] = (raw_cf[tid] - cf_min) / (cf_max - cf_min)

        results = []
        for tid in all_ids:
            hybrid_s = self.cbf_weight * cbf_norm[tid] + self.cf_weight * cf_norm[tid]

            results.append({
                "tmdb_id": tid,
                "cbf_score": round(cbf_norm[tid], 4),
                "cf_score": round(cf_norm[tid], 4),
                "hybrid_score": round(hybrid_s, 6),
            })

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]


    # ---- helpers ----

    @staticmethod
    def _normalise(item_scores: List[Tuple[int, float]], scale_max=1.0) -> Dict[int, float]:
        if not item_scores:
            return {}
        scores = [s for _, s in item_scores]
        s_min, s_max = min(scores), max(scores)
        
        # Avoid division by zero if all scores are identical
        if s_max - s_min < 1e-6:
            return {tid: scale_max for tid, _ in item_scores}
            
        return {tid: ((s - s_min) / (s_max - s_min)) * scale_max for tid, s in item_scores}


# ======================================================================
# 4.  Factory helper
# ======================================================================

def get_recommender(cbf_weight: float = 0.4, cf_weight: float = 0.6) -> HybridRecommender:
    """Return a HybridRecommender with the given weights."""
    return HybridRecommender(cbf_weight=cbf_weight, cf_weight=cf_weight)
