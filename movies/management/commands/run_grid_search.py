import logging
import random
import time
from django.core.management.base import BaseCommand
from django.db.models import Prefetch

import ranx
from ranx import Qrels, Run

from movies.models import UserRating
from movies.recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from movies.management.commands.evaluate_recommender import Command as EvaluateCommand

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Run grid search over CF n_factors and CBF params to find optimal Hybrid performance (using k-fold)."

    def add_arguments(self, parser):
        parser.add_argument('--folds', type=int, default=5, help="Number of folds (default 5)")
        parser.add_argument('--sample-users', type=int, default=50, help="Number of users to evaluate (0 for all)")

    def handle(self, *args, **options):
        n_folds = options['folds']
        sample_users = options['sample_users']

        self.stdout.write(self.style.NOTICE(f"=== Starting Grid Search (Folds: {n_folds}, Users: {sample_users}) ==="))

        eval_cmd = EvaluateCommand()
        eval_cmd.stdout = self.stdout
        eval_cmd.style = self.style

        # -- 1. Load all ratings ---
        self.stdout.write("Mengambil data rating dari database...")
        raw = list(
            UserRating.objects.values_list("user_id", "movie__tmdb_id", "rating")
        )
        if not raw:
            self.stdout.write(self.style.ERROR("Tidak ada rating di database."))
            return

        import numpy as np
        from sklearn.model_selection import KFold
        from collections import defaultdict
        all_ratings = np.array(raw, dtype=object)

        # Get unique users for sampling if needed
        all_user_ids = list(set([r[0] for r in raw]))
        if sample_users > 0 and len(all_user_ids) > sample_users:
            random.seed(42)
            sampled = set(random.sample(all_user_ids, sample_users))
            # Filter ratings for only sampled users
            filtered_raw = [r for r in raw if r[0] in sampled]
            all_ratings = np.array(filtered_raw, dtype=object)
            
        self.stdout.write(f"  Total rating for evaluation: {len(all_ratings)}")

        # We will grid search n_factors for CF and pool_size for Hybrid
        cf_factors = [5, 10, 15, 20, 30]
        pool_sizes = [10, 20, 50, 100]
        
        best_hybrid_hr = -1
        best_config = None
        
        # Train CBF once since it doesn't depend on CF parameters (and doesn't use ratings directly)
        cbf = ContentBasedRecommender()
        self.stdout.write("Training CBF (TF-IDF matrix) secara global...")
        cbf.train()

        for n_factors in cf_factors:
            for pool_size in pool_sizes:
                self.stdout.write(self.style.WARNING(f"\nEvaluating n_factors={n_factors}, pool_size={pool_size}"))
                
                cbf_metrics = {'hr': [], 'map': []}
                cf_metrics = {'hr': [], 'map': []}
                hybrid_metrics = {'hr': [], 'map': []}

                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                for fold_num, (train_idx, test_idx) in enumerate(kf.split(all_ratings), start=1):
                    train_data = all_ratings[train_idx]
                    test_data = all_ratings[test_idx]
                    train_list = [(int(u), int(i), float(r)) for u, i, r in train_data]

                    # Group by user
                    user_train = defaultdict(list)
                    for u, i, r in train_list:
                        user_train[u].append((i, r))

                    user_test = defaultdict(dict)
                    for u, i, r in test_data:
                        user_test[int(u)][int(i)] = float(r)
                    
                    # Train CF on this fold's training data
                    cf = CollaborativeRecommender(n_factors=n_factors)
                    cf.train(ratings_data=train_list)
                    
                    hybrid = HybridRecommender(cbf_weight=0.4, cf_weight=0.6)
                    hybrid.cbf = cbf
                    hybrid.cf = cf
                    hybrid._trained = True

                    qrels_dict = {}
                    run_cbf_dict = {}
                    run_cf_dict = {}
                    run_hybrid_dict = {}
                    
                    # For sample progress tracking
                    users_in_test = list(user_test.keys())

                    for u in users_in_test:
                        # Ground truth (rating >= 3.0)
                        u_test_relevant = {str(iid): r for iid, r in user_test[u].items() if r >= 3.0}
                        if not u_test_relevant:
                            continue
                        
                        qrels_dict[str(u)] = u_test_relevant
                        
                        # List of tuples [(item, rating), ...]
                        u_train_ratings = user_train[u]
                        rated_ids = set([i for i, r in u_train_ratings])
                        
                        k = 10
                        
                        # Single models
                        try:
                            cbf_recs = cbf.predict(u, k=k, user_ratings=u_train_ratings)
                        except Exception: cbf_recs = []
                            
                        try:
                            cf_recs = cf.predict(u, k=k, rated_tmdb_ids=list(rated_ids))
                        except Exception: cf_recs = []
                            
                        # Hybrid direct scoring BUT WITH CUSTOM POOL SIZE
                        try:
                            cbf_pool = cbf.predict(u, k=pool_size, user_ratings=u_train_ratings)
                        except: cbf_pool = []
                        try:
                            cf_pool = cf.predict(u, k=pool_size, rated_tmdb_ids=list(rated_ids))
                        except: cf_pool = []
                        
                        cbf_dict = dict(cbf_pool)
                        cf_dict = dict(cf_pool)
                        all_ids = list(set(cbf_dict.keys()) | set(cf_dict.keys()))
                        
                        raw_cbf = {}
                        raw_cf = {}
                        cbf_profile, _ = cbf._build_profile(u, user_ratings=u_train_ratings)
                        for tid in all_ids:
                            raw_cbf[tid] = cbf_dict[tid] if tid in cbf_dict else cbf.score_one(u, tid, _profile=cbf_profile, user_ratings=u_train_ratings)
                            raw_cf[tid] = cf_dict[tid] if tid in cf_dict else cf.score_one(u, tid)
                            
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

                        scores = []
                        for tid in all_ids:
                            h = 0.4 * cbf_norm[tid] + 0.6 * cf_norm[tid]
                            scores.append((tid, h))
                        
                        scores.sort(key=lambda x: x[1], reverse=True)
                        hybrid_recs = {str(iid): float(s) for iid, s in scores[:k]}

                        if cbf_recs:
                            run_cbf_dict[str(u)] = {str(iid): float(s) for iid, s in cbf_recs}
                        if cf_recs:
                            run_cf_dict[str(u)] = {str(iid): float(s) for iid, s in cf_recs}
                        if hybrid_recs:
                            run_hybrid_dict[str(u)] = hybrid_recs
                            
                    if not qrels_dict:
                        continue

                    qrels = Qrels(qrels_dict)
                    if run_cbf_dict:
                        res_cbf = ranx.evaluate(qrels, Run(run_cbf_dict), ["hit_rate@10", "map@10"])
                        cbf_metrics['hr'].append(res_cbf["hit_rate@10"])
                        cbf_metrics['map'].append(res_cbf["map@10"])
                    else:
                        cbf_metrics['hr'].append(0)
                        cbf_metrics['map'].append(0)
                    
                    if run_cf_dict:
                        res_cf = ranx.evaluate(qrels, Run(run_cf_dict), ["hit_rate@10", "map@10"])
                        cf_metrics['hr'].append(res_cf["hit_rate@10"])
                        cf_metrics['map'].append(res_cf["map@10"])
                    else:
                        cf_metrics['hr'].append(0)
                        cf_metrics['map'].append(0)

                    if run_hybrid_dict:
                        res_hybrid = ranx.evaluate(qrels, Run(run_hybrid_dict), ["hit_rate@10", "map@10"])
                        hybrid_metrics['hr'].append(res_hybrid["hit_rate@10"])
                        hybrid_metrics['map'].append(res_hybrid["map@10"])
                    else:
                        hybrid_metrics['hr'].append(0)
                        hybrid_metrics['map'].append(0)
                        
                # Compute averages
                def avg(lst): return sum(lst)/len(lst) if lst else 0.0
                
                cbf_hr_avg = avg(cbf_metrics['hr'])
                cf_hr_avg = avg(cf_metrics['hr'])
                hybrid_hr_avg = avg(hybrid_metrics['hr'])
                
                cbf_map_avg = avg(cbf_metrics['map'])
                cf_map_avg = avg(cf_metrics['map'])
                hybrid_map_avg = avg(hybrid_metrics['map'])
                
                self.stdout.write(f"  CBF    HR@10: {cbf_hr_avg:.4f}  MAP@10: {cbf_map_avg:.4f}")
                self.stdout.write(f"  CF     HR@10: {cf_hr_avg:.4f}  MAP@10: {cf_map_avg:.4f}")
                self.stdout.write(f"  Hybrid HR@10: {hybrid_hr_avg:.4f}  MAP@10: {hybrid_map_avg:.4f}")
                
                hr_beats = hybrid_hr_avg > max(cbf_hr_avg, cf_hr_avg)
                map_beats = hybrid_map_avg >= max(cbf_map_avg, cf_map_avg)
                
                if hr_beats and map_beats:
                    self.stdout.write(self.style.SUCCESS("  => Hybrid beats single models in BOTH HR and MAP!"))
                elif hr_beats:
                    self.stdout.write(self.style.WARNING("  => Hybrid beats single models in HR but not MAP."))
                elif map_beats:
                    self.stdout.write(self.style.WARNING("  => Hybrid beats single models in MAP but not HR."))
                else:
                    self.stdout.write(self.style.ERROR("  => Hybrid failed to beat single models in both."))
                    
                if hr_beats and map_beats and hybrid_hr_avg > best_hybrid_hr:
                    best_hybrid_hr = hybrid_hr_avg
                    best_config = {'n_factors': n_factors, 'pool_size': pool_size}
                    
        self.stdout.write(self.style.SUCCESS(f"\nBest Config (Beats both): {best_config} (Hybrid HR@10: {best_hybrid_hr:.4f})"))


