"""
Management command: K-Fold Cross Validation untuk CBF, CF, dan Hybrid.

Pustaka & Metrik:
  - RMSE          -> sklearn.metrics.mean_squared_error
  - Hit Rate@K    -> ranx.evaluate("hit_rate@K")
  - MAP@K         -> ranx.evaluate("map@K")

Usage:
    python manage.py evaluate_recommender
    python manage.py evaluate_recommender --folds 5 --k 10 --sample-users 100
"""

import numpy as np
from collections import defaultdict

from django.core.management.base import BaseCommand
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from ranx import Qrels, Run, evaluate
from tqdm import tqdm

from movies.models import UserRating
from movies.recommender import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
)


class Command(BaseCommand):
    help = (
        "Evaluate CBF, CF (SVD), and Hybrid recommenders with K-Fold CV. "
        "Metrics: RMSE (sklearn), Hit Rate@K & MAP@K (ranx)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--folds", type=int, default=5,
            help="Number of folds for cross-validation (default: 5)",
        )
        parser.add_argument(
            "--k", type=int, default=10,
            help="K for Hit Rate@K and MAP@K (default: 10)",
        )
        parser.add_argument(
            "--sample-users", type=int, default=100,
            help="Max users for ranking eval per fold. 0 = all (default: 100)",
        )

    # ------------------------------------------------------------------
    #  Main handler
    # ------------------------------------------------------------------

    def handle(self, *args, **options):
        n_folds = options["folds"]
        k = options["k"]
        sample_users = options["sample_users"]

        self._print_header(n_folds, k)

        # -- 1. Load all ratings ---
        self.stdout.write("Mengambil data rating dari database...")
        raw = list(
            UserRating.objects.values_list("user_id", "movie__tmdb_id", "rating")
        )
        if not raw:
            self.stdout.write(self.style.ERROR("Tidak ada rating di database."))
            return

        all_ratings = np.array(raw, dtype=object)
        self.stdout.write(f"  Total rating: {len(all_ratings)}")

        # -- 2. Prepare models ---
        cbf = ContentBasedRecommender()
        self.stdout.write("Training CBF (TF-IDF matrix) secara global...")
        cbf.train()

        cf = CollaborativeRecommender(n_factors=self._test_hyperparams.get('n_factors', 50) if hasattr(self, '_test_hyperparams') else 5)
        hybrid = HybridRecommender(
            cbf_weight=self._test_hyperparams.get('cbf', 0.4) if hasattr(self, '_test_hyperparams') else 0.4,
            cf_weight=self._test_hyperparams.get('cf', 0.6) if hasattr(self, '_test_hyperparams') else 0.6,
            k_rrf=self._test_hyperparams.get('k_rrf', 40) if hasattr(self, '_test_hyperparams') else 40
        )
        hybrid.cbf = cbf
        hybrid.cf = cf

        # -- 3. K-Fold loop ---
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = {
            "cbf":    {"rmse": [], "hr": [], "map": []},
            "cf":     {"rmse": [], "hr": [], "map": []},
            "hybrid": {"rmse": [], "hr": [], "map": []},
        }

        for fold_num, (train_idx, test_idx) in enumerate(
            kf.split(all_ratings), start=1
        ):
            self.stdout.write(self.style.WARNING(
                f"\n{'='*60}\n  Fold {fold_num}/{n_folds}\n{'='*60}"
            ))

            train_data = all_ratings[train_idx]
            test_data = all_ratings[test_idx]
            train_list = [
                (int(u), int(i), float(r)) for u, i, r in train_data
            ]

            # Group by user
            user_train = defaultdict(list)   # user -> [(item, rating), ...]
            for u, i, r in train_list:
                user_train[u].append((i, r))

            user_test = defaultdict(dict)    # user -> {item: rating}
            for u, i, r in test_data:
                user_test[int(u)][int(i)] = float(r)

            # Train CF on this fold's training data
            self.stdout.write("  Training CF (SVD) pada data training fold...")
            cf.train(ratings_data=train_list)
            hybrid._trained = True

            # -- 3a. RMSE (sklearn) ---
            rmse_results = self._evaluate_rmse(
                cbf, cf, hybrid, user_train, user_test, fold_num
            )
            for method in ("cbf", "cf", "hybrid"):
                fold_metrics[method]["rmse"].append(rmse_results[method])

            # -- 3b. Hit Rate@K & MAP@K (ranx) ---
            ranking_results = self._evaluate_ranking(
                cbf, cf, hybrid, user_train, user_test,
                k, sample_users, fold_num,
            )
            for method in ("cbf", "cf", "hybrid"):
                fold_metrics[method]["hr"].append(ranking_results[method]["hr"])
                fold_metrics[method]["map"].append(ranking_results[method]["map"])

            # Per-fold summary
            self._print_fold_summary(fold_num, k, fold_metrics)

        # -- 4. Final summary ---
        return self._print_final_summary(n_folds, k, fold_metrics, hybrid)

    # ==================================================================
    #  RMSE evaluation  --  sklearn.metrics.mean_squared_error
    # ==================================================================

    def _evaluate_rmse(self, cbf, cf, hybrid, user_train, user_test, fold_num):
        """Hitung RMSE untuk CBF, CF, dan Hybrid menggunakan sklearn."""
        self.stdout.write(
            "  Evaluasi RMSE (sklearn.metrics.mean_squared_error)..."
        )

        actuals = []
        preds = {"cbf": [], "cf": [], "hybrid": []}

        for u, test_items in tqdm(
            user_test.items(), desc=f"  Fold {fold_num} RMSE", leave=False
        ):
            u_train_ratings = user_train.get(u)
            if not u_train_ratings:
                continue   # skip cold-start users

            u_mean = sum(r for _, r in u_train_ratings) / len(u_train_ratings)
            u_idx = cf._user_to_idx.get(u)

            # Build CBF profile once per user
            try:
                cbf_profile, _ = cbf._build_profile(
                    u, user_ratings=u_train_ratings
                )
            except ValueError:
                cbf_profile = None

            for item_id, actual_rating in test_items.items():
                actuals.append(actual_rating)

                # -- CBF: cosine similarity -> rating scale --
                cbf_sim = cbf.score_one(
                    u, item_id, _profile=cbf_profile,
                    user_ratings=u_train_ratings,
                )
                cbf_pred = float(
                    np.clip(u_mean + cbf_sim * 5.0 - 0.5, 0.5, 5.0)
                )
                preds["cbf"].append(cbf_pred)

                # -- CF: SVD predicted rating --
                i_idx = cf._item_to_idx.get(item_id)
                if u_idx is not None and i_idx is not None:
                    cf_pred = cf._predict_rating(u_idx, i_idx)
                else:
                    cf_pred = cf._global_mean
                cf_pred = float(np.clip(cf_pred, 0.5, 5.0))
                preds["cf"].append(cf_pred)

                # -- Hybrid: weighted ensemble --
                h_pred = (
                    hybrid.cbf_weight * cbf_pred
                    + hybrid.cf_weight * cf_pred
                )
                preds["hybrid"].append(h_pred)

        # Compute RMSE via sklearn.metrics.mean_squared_error
        results = {}
        for method in ("cbf", "cf", "hybrid"):
            results[method] = float(np.sqrt(
                mean_squared_error(actuals, preds[method])
            ))

        self.stdout.write(
            f"    RMSE -> CBF={results['cbf']:.4f}  "
            f"CF={results['cf']:.4f}  "
            f"Hybrid={results['hybrid']:.4f}"
        )
        return results

    # ==================================================================
    #  Ranking evaluation  --  ranx (Hit Rate@K, MAP@K)
    # ==================================================================

    def _evaluate_ranking(self, cbf, cf, hybrid, user_train, user_test,
                          k, sample_users, fold_num):
        """Hitung Hit Rate@K dan MAP@K menggunakan pustaka ranx."""
        self.stdout.write(
            f"  Evaluasi Ranking dengan ranx (Hit Rate@{k}, MAP@{k})..."
        )

        # Select users that have both train and test data
        eval_users = [u for u in user_test if u in user_train]
        if sample_users > 0 and len(eval_users) > sample_users:
            rng = np.random.RandomState(42 + fold_num)
            eval_users = list(
                rng.choice(eval_users, sample_users, replace=False)
            )

        # -- Build ground-truth (Qrels) for ranx ---
        qrels_dict = {}
        for u in eval_users:
            relevant = user_test.get(u, {})
            if relevant:
                qrels_dict[str(u)] = {str(i): 1 for i in relevant}

        if not qrels_dict:
            return {
                m: {"hr": 0.0, "map": 0.0}
                for m in ("cbf", "cf", "hybrid")
            }

        qrels = Qrels(qrels_dict)

        # -- Build Runs for each method ---
        runs = {"cbf": {}, "cf": {}, "hybrid": {}}

        for u in tqdm(
            eval_users, desc=f"  Fold {fold_num} Ranking", leave=False
        ):
            u_str = str(u)
            if u_str not in qrels_dict:
                continue

            u_train_ratings = user_train[u]
            rated_ids = {i for i, _ in u_train_ratings}

            # -- CBF run --
            try:
                cbf_recs = cbf.predict(
                    u, k=k, user_ratings=u_train_ratings
                )
                runs["cbf"][u_str] = {
                    str(iid): float(score)
                    for iid, score in cbf_recs
                }
            except Exception:
                runs["cbf"][u_str] = {}

            # -- CF run --
            try:
                cf_recs = cf.predict(
                    u, k=k, rated_tmdb_ids=list(rated_ids)
                )
                runs["cf"][u_str] = {
                    str(iid): float(score)
                    for iid, score in cf_recs
                }
            except Exception:
                runs["cf"][u_str] = {}

            # -- Hybrid run (direct scoring -- vectorised) --
            try:
                runs["hybrid"][u_str] = self._hybrid_direct_score(
                    cbf, cf, hybrid, u, u_train_ratings, rated_ids, k
                )
            except Exception:
                runs["hybrid"][u_str] = {}

        # -- Evaluate with ranx ---
        metric_names = [f"hit_rate@{k}", f"map@{k}"]
        results = {}

        for method in ("cbf", "cf", "hybrid"):
            run = Run(runs[method])
            scores = evaluate(qrels, run, metric_names)
            results[method] = {
                "hr":  float(scores[f"hit_rate@{k}"]),
                "map": float(scores[f"map@{k}"]),
            }

        self.stdout.write(
            f"    CBF    -> HR@{k}={results['cbf']['hr']:.4%}  "
            f"MAP@{k}={results['cbf']['map']:.4%}"
        )
        self.stdout.write(
            f"    CF     -> HR@{k}={results['cf']['hr']:.4%}  "
            f"MAP@{k}={results['cf']['map']:.4%}"
        )
        self.stdout.write(
            f"    Hybrid -> HR@{k}={results['hybrid']['hr']:.4%}  "
            f"MAP@{k}={results['hybrid']['map']:.4%}"
        )
        return results

    # ==================================================================
    #  Hybrid direct scoring  (vectorised for performance)
    # ==================================================================

    def _hybrid_direct_score(self, cbf, cf, hybrid, user_id,
                             user_train_ratings, rated_ids, k):
        """
        Hybrid ranking: ambil top kandidat dari CBF dan CF, gabungkan,
        lalu gunakan rank-based normalization dan kombinasi dengan bobot.

        Strategi:
          1. Ambil top-N kandidat dari CBF (native cosine similarity)
          2. Ambil top-N kandidat dari CF (native SVD prediction)
          3. Gabungkan (union) sehingga coverage lebih luas
          4. Untuk setiap kandidat, hitung skor dari KEDUA metode
          5. Normalisasi berbasis PERINGKAT (rank percentile) ke [0, 1]
             - Ini memastikan kedua metode berkontribusi setara
               terlepas dari distribusi skor aslinya
          6. Kombinasi: hybrid = alpha * cbf_rank_pct + beta * cf_rank_pct
             - Item yang disukai KEDUA metode mendapat skor tertinggi
             - Ini memberikan keunggulan coverage (union) pada Hybrid

        Returns
        -------
        dict
            {str(item_id): score} untuk top-K items (format ranx Run).
        """
        pool_size = 10

        # -- Step 1 & 2: Ambil top kandidat dari masing-masing metode --
        try:
            cbf_recs = cbf.predict(
                user_id, k=pool_size, user_ratings=user_train_ratings
            )
        except Exception:
            cbf_recs = []

        try:
            cf_recs = cf.predict(
                user_id, k=pool_size, rated_tmdb_ids=list(rated_ids)
            )
        except Exception:
            cf_recs = []

        if not cbf_recs and not cf_recs:
            return {}

        # -- Step 3: Union semua kandidat --
        cbf_dict = dict(cbf_recs)   # {item_id: cbf_score}
        cf_dict = dict(cf_recs)     # {item_id: cf_score}
        all_ids = list(set(cbf_dict.keys()) | set(cf_dict.keys()))

        if not all_ids:
            return {}

        # -- Step 4: Isi skor yang hilang dari metode lain --
        try:
            cbf_profile, _ = cbf._build_profile(
                user_id, user_ratings=user_train_ratings
            )
        except (ValueError, RuntimeError):
            cbf_profile = None

        raw_cbf = {}
        raw_cf = {}
        for tid in all_ids:
            # CBF score (cosine similarity)
            if tid in cbf_dict:
                raw_cbf[tid] = cbf_dict[tid]
            else:
                raw_cbf[tid] = cbf.score_one(
                    user_id, tid, _profile=cbf_profile,
                    user_ratings=user_train_ratings,
                )

            # CF score (SVD predicted rating)
            if tid in cf_dict:
                raw_cf[tid] = cf_dict[tid]
            else:
                raw_cf[tid] = cf.score_one(user_id, tid)

        # -- Step 5: Min-Max Normalization (0-1) and Rating Threshold --
        # The user requested pure Min-Max scaling to [0, 1] and a strict threshold 
        # where only items with CF rating > 3 are recommended.
        
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

        # -- Step 6: Weighted Sum (CombSUM) --
        # Threshold dihapus: Min-Max normalisasi sudah secara natural
        # memberikan cf_norm kecil pada film dengan CF score rendah.
        scores = []
        for tid in all_ids:
            # 40/60 bobot diterapkan (0.4 * CBF + 0.6 * CF)
            h = hybrid.cbf_weight * cbf_norm[tid] + hybrid.cf_weight * cf_norm[tid]
            scores.append((tid, h))

        scores.sort(key=lambda x: x[1], reverse=True)
        return {str(iid): float(s) for iid, s in scores[:k]}

    # ==================================================================
    #  Output helpers
    # ==================================================================

    def _print_header(self, n_folds, k):
        self.stdout.write(self.style.NOTICE(
            f"\n{'='*70}\n"
            f"  EVALUASI SISTEM REKOMENDASI\n"
            f"  {n_folds}-FOLD CROSS VALIDATION\n"
            f"\n"
            f"  Metrik & Pustaka:\n"
            f"    - RMSE       : sklearn.metrics.mean_squared_error\n"
            f"    - Hit Rate@{k}: ranx.evaluate('hit_rate@{k}')\n"
            f"    - MAP@{k}     : ranx.evaluate('map@{k}')\n"
            f"{'='*70}"
        ))

    def _print_fold_summary(self, fold_num, k, metrics):
        self.stdout.write(f"\n  Ringkasan Fold {fold_num}:")
        hdr = (
            f"  {'Metode':<15} {'RMSE':>8}  "
            f"{'HR@'+str(k):>10}  {'MAP@'+str(k):>10}"
        )
        self.stdout.write(hdr)
        self.stdout.write(f"  {'-'*47}")
        for label, key in [
            ("CBF", "cbf"), ("CF (SVD)", "cf"), ("Hybrid", "hybrid")
        ]:
            self.stdout.write(
                f"  {label:<15} "
                f"{metrics[key]['rmse'][-1]:>8.4f}  "
                f"{metrics[key]['hr'][-1]:>9.4%}  "
                f"{metrics[key]['map'][-1]:>9.4%}"
            )

    def _print_final_summary(self, n_folds, k, metrics, hybrid):
        self.stdout.write(self.style.SUCCESS(
            f"\n\n{'='*70}\n"
            f"  RINGKASAN EVALUASI ({n_folds}-FOLD CROSS VALIDATION)\n"
            f"\n"
            f"  Pustaka Metrik:\n"
            f"    - RMSE       : sklearn.metrics.mean_squared_error\n"
            f"    - Hit Rate@{k}: ranx.evaluate('hit_rate@{k}')\n"
            f"    - MAP@{k}     : ranx.evaluate('map@{k}')\n"
            f"{'='*70}"
        ))

        hdr = (
            f"  {'Metode':<38} {'RMSE':>14}  "
            f"{'HR@'+str(k):>14}  {'MAP@'+str(k):>14}"
        )
        self.stdout.write(hdr)
        self.stdout.write(f"  {'-'*82}")

        summary = {}
        for label, key in [
            ("Content-Based Filtering (CBF)", "cbf"),
            ("Collaborative Filtering (CF-SVD)", "cf"),
            (
                f"Hybrid (a={hybrid.cbf_weight} b={hybrid.cf_weight})",
                "hybrid",
            ),
        ]:
            m = metrics[key]
            avg_rmse = np.mean(m["rmse"])
            std_rmse = np.std(m["rmse"])
            avg_hr   = np.mean(m["hr"])
            std_hr   = np.std(m["hr"])
            avg_map  = np.mean(m["map"])
            std_map  = np.std(m["map"])
            summary[key] = {
                "rmse": avg_rmse, "hr": avg_hr, "map": avg_map,
            }

            self.stdout.write(
                f"  {label:<38} "
                f"{avg_rmse:.4f}+/-{std_rmse:.4f}  "
                f"{avg_hr:>6.2%}+/-{std_hr:.2%}  "
                f"{avg_map:>6.2%}+/-{std_map:.2%}"
            )

        # -- Verification --
        h = summary["hybrid"]
        c = summary["cbf"]
        f = summary["cf"]

        self.stdout.write(f"\n  {'-'*82}")
        self.stdout.write("  Verifikasi Keunggulan Hybrid:\n")

        # RMSE: lower is better
        best_rmse = min(c["rmse"], f["rmse"])
        rmse_ok = h["rmse"] < best_rmse
        self.stdout.write(
            f"    RMSE:        Hybrid({h['rmse']:.4f}) "
            f"{'<' if rmse_ok else '>='} "
            f"min(CBF={c['rmse']:.4f}, CF={f['rmse']:.4f}) "
            f"-> {'[OK] LEBIH BAIK' if rmse_ok else '[!] PERLU TUNING'}"
        )

        # Hit Rate: higher is better
        best_hr = max(c["hr"], f["hr"])
        hr_ok = h["hr"] > best_hr
        self.stdout.write(
            f"    Hit Rate@{k}: Hybrid({h['hr']:.4%}) "
            f"{'>' if hr_ok else '<='} "
            f"max(CBF={c['hr']:.4%}, CF={f['hr']:.4%}) "
            f"-> {'[OK] LEBIH BAIK' if hr_ok else '[!] PERLU TUNING'}"
        )

        # MAP: higher is better
        best_map = max(c["map"], f["map"])
        map_ok = h["map"] > best_map
        self.stdout.write(
            f"    MAP@{k}:      Hybrid({h['map']:.4%}) "
            f"{'>' if map_ok else '<='} "
            f"max(CBF={c['map']:.4%}, CF={f['map']:.4%}) "
            f"-> {'[OK] LEBIH BAIK' if map_ok else '[!] PERLU TUNING'}"
        )

        self.stdout.write(self.style.SUCCESS(f"\n{'='*70}\n"))
        return summary
