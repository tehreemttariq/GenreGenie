import numpy as np

from src.model import MatrixFactorizationSVD


def test_svd_persists_id_mappings_with_factor_matrices(tmp_path):
    ratings = np.array(
        [
            [5.0, 0.0, 4.0],
            [0.0, 3.0, 4.0],
            [2.0, 4.0, 0.0],
        ]
    )
    model = MatrixFactorizationSVD(n_factors=1).fit(ratings)
    model.user_to_idx = {11: 0, 12: 1, 13: 2}
    model.movie_to_idx = {101: 0, 102: 1, 103: 2}
    model_path = tmp_path / "svd_model.pkl"

    model.save(str(model_path))
    loaded = MatrixFactorizationSVD.load(str(model_path))

    assert loaded.user_to_idx == model.user_to_idx
    assert loaded.movie_to_idx == model.movie_to_idx
    assert loaded.user_factors.shape == (3, 1)
    assert loaded.item_factors.shape == (3, 1)
