"""Tune lambda parameter for fold-in"""
import numpy as np
from model_loader import load_model
from foldin import fold_in_user, recommend_new_user
from pathlib import Path
import csv

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_user_ratings_from_csv(user_idx, mappings):
    ratings_path = Path("data/raw") / "ratings.csv"
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

print("TUNING LAMBDA FOR FOLD-IN\n")

model_data, mappings = load_model()
user_indices = list(mappings['user_to_idx'].values())[:5]

lambdas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

print(f"{'Lambda':<8} {'Avg Cosine Sim':<18} {'Description':<40}")
print("-" * 70)

results_per_lambda = {}

for lam in lambdas:
    cos_sims = []
    
    for user_idx in user_indices:
        trained_vector = model_data['user_factors'][user_idx]
        ratings_dict = get_user_ratings_from_csv(user_idx, mappings)
        
        if len(ratings_dict) < 3:
            continue
        
        folded_vector = fold_in_user(ratings_dict, model_data, mappings, lambda_reg=lam)
        if folded_vector is None:
            continue
        
        cos_sim = cosine_similarity(trained_vector, folded_vector)
        cos_sims.append(cos_sim)
    
    if cos_sims:
        avg_sim = np.mean(cos_sims)
        results_per_lambda[lam] = avg_sim
        
        if lam < 1.0:
            desc = "Strong personalization (may overfit)"
        elif lam == 1.0:
            desc = "Balanced (DEFAULT)"
        else:
            desc = "Smooth solutions (conservative)"
        
        print(f"{lam:<8.2f} {avg_sim:<18.4f} {desc:<40}")

print("\n" + "=" * 70)
best_lambda = max(results_per_lambda, key=results_per_lambda.get)
best_score = results_per_lambda[best_lambda]

print(f"\nBEST LAMBDA: {best_lambda}")
print(f"Cosine similarity: {best_score:.4f}")
print(f"\nRECOMMENDATION:")
if best_lambda < 1.0:
    print(f"  λ = {best_lambda} provides stronger personalization")
    print(f"  Use this if you want recommendations closely tailored to ratings")
elif best_lambda == 1.0:
    print(f"  λ = 1.0 is already optimal! Balanced fit.")
else:
    print(f"  λ = {best_lambda} provides smoother recommendations")
    print(f"  Use this if you want more diverse, less narrow recommendations")

print(f"\nFINAL CHOICE: λ = 1.0 (default ridge regression)")
print(f"  - Balances personalization vs smoothness")
print(f"  - Standard in collaborative filtering")
print(f"  - Works well across user types")
