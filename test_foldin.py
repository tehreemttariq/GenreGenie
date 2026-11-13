"""Test fold-in: compare folded vectors to trained vectors"""
import numpy as np
from model_loader import load_model
from foldin import fold_in_user, recommend_new_user
from pathlib import Path
import csv

def get_user_ratings_from_csv(user_idx, mappings):
    """Load ratings for a user from ratings.csv"""
    ratings_path = Path("data/raw") / "ratings.csv"
    if not ratings_path.exists():
        return {}
    
    ratings_dict = {}
    user_to_idx_inv = {v: k for k, v in mappings['user_to_idx'].items()}
    user_id = user_to_idx_inv.get(user_idx)
    
    if user_id is None:
        return {}
    
    with open(ratings_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['userId']) == int(user_id):
                movie_id = int(row['movieId'])
                rating = float(row['rating'])
                ratings_dict[movie_id] = rating
    
    return ratings_dict

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_top_n_overlap(recs_foldin, recs_trained, n=10):
    """
    Compare two recommendation lists.
    EXCLUDE already-rated movies from both.
    """
    recs_foldin_ids = set([mid for mid, _ in recs_foldin[:n]])
    recs_trained_ids = set([mid for mid, _ in recs_trained[:n]])
    
    overlap = recs_foldin_ids & recs_trained_ids
    union = recs_foldin_ids | recs_trained_ids
    
    jaccard = len(overlap) / len(union) if union else 0
    
    return {
        "overlap_count": len(overlap),
        "jaccard_similarity": jaccard,
        "overlap_movies": sorted(list(overlap))
    }

def test_fold_in_with_partial_ratings():
    """
    Test fold-in: for 50 users, keep only 10 ratings, fold in,
    and compare to trained vector.
    Report RMSE on held-out ratings.
    """
    print("\n" + "=" * 80)
    print("FOLD-IN TEST: Partial Ratings (10-rating fold-in)")
    print("=" * 80)
    
    model_data, mappings = load_model()
    if model_data is None:
        print("ERROR: Could not load model")
        return
    
    print(f"\n✓ Model loaded: {len(mappings['user_to_idx'])} users, {len(mappings['movie_to_idx'])} movies")
    
    # Test 50 random users
    user_indices = list(mappings['user_to_idx'].values())[:50]
    
    print(f"\nTesting {len(user_indices)} users with 10-rating fold-in:\n")
    
    results = []
    rmse_values = []
    
    for i, user_idx in enumerate(user_indices):
        trained_vector = model_data['user_factors'][user_idx]
        all_ratings = get_user_ratings_from_csv(user_idx, mappings)
        
        if len(all_ratings) < 10:
            continue
        
        # Randomly select 10 ratings for fold-in
        movie_ids = list(all_ratings.keys())
        np.random.seed(i)
        fold_in_movie_ids = np.random.choice(movie_ids, 10, replace=False)
        holdout_movie_ids = [m for m in movie_ids if m not in fold_in_movie_ids]
        
        fold_in_ratings = {m: all_ratings[m] for m in fold_in_movie_ids}
        
        # Fold-in with 10 ratings
        folded_vector = fold_in_user(fold_in_ratings, model_data, mappings, lambda_reg=1.0)
        
        if folded_vector is None:
            continue
        
        # Compute cosine similarity
        cos_sim = cosine_similarity(trained_vector, folded_vector)
        
        # Get recommendations
        rated_ids = set(fold_in_movie_ids)
        recs_trained = recommend_new_user(trained_vector, rated_ids, model_data, mappings, n_recs=20)
        recs_folded = recommend_new_user(folded_vector, rated_ids, model_data, mappings, n_recs=20)
        
        overlap = get_top_n_overlap(recs_folded, recs_trained, n=10)
        
        # Compute RMSE on holdout ratings
        if holdout_movie_ids:
            holdout_preds = []
            for movie_id in holdout_movie_ids[:5]:  # Check first 5 holdout
                try:
                    movie_idx = mappings['movie_to_idx'].get(str(movie_id))
                    if movie_idx is not None:
                        score = model_data['global_mean'] + folded_vector @ model_data['item_factors'][movie_idx]
                        holdout_preds.append((all_ratings[movie_id], score))
                except:
                    pass
            
            if holdout_preds:
                true_vals = np.array([t[0] for t in holdout_preds])
                pred_vals = np.array([t[1] for t in holdout_preds])
                rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
                rmse_values.append(rmse)
            else:
                rmse = None
        else:
            rmse = None
        
        result = {
            "user_idx": user_idx,
            "num_fold_in": len(fold_in_ratings),
            "num_holdout": len(holdout_movie_ids),
            "cosine_sim": cos_sim,
            "overlap_top10": overlap["overlap_count"],
            "jaccard": overlap["jaccard_similarity"],
            "rmse": rmse
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1} users...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY (50 Users, 10-Rating Fold-In)")
    print("=" * 80)
    
    if results:
        avg_cos_sim = np.mean([r["cosine_sim"] for r in results])
        avg_overlap = np.mean([r["overlap_top10"] for r in results])
        avg_jaccard = np.mean([r["jaccard"] for r in results])
        avg_rmse = np.mean([r["rmse"] for r in results if r["rmse"] is not None])
        
        print(f"\nAveraged over {len(results)} users (trained: all ratings, tested: 10-rating fold-in):")
        print(f"  Cosine similarity: {avg_cos_sim:.4f}")
        print(f"  Top-10 overlap: {avg_overlap:.1f}/10 movies")
        print(f"  Jaccard index: {avg_jaccard:.4f}")
        if not np.isnan(avg_rmse):
            print(f"  RMSE (holdout): {avg_rmse:.3f}")
        
        print(f"\nLambda regularization: 1.0 (default)")
        print(f"  • Balances personalization vs smoothness")
        print(f"  • Standard in collaborative filtering")
        print(f"  • Chosen via lambda tuning test")
        
        print(f"\n✓ FOLD-IN TEST COMPLETE")
    else:
        print("ERROR: No valid test results")

if __name__ == "__main__":
    test_fold_in_with_partial_ratings()

