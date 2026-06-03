import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, movie_feature_dim: int,
                 embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_tower = nn.Sequential(
            nn.Linear(movie_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(user_ids)

    def encode_movie(self, movie_features: torch.Tensor) -> torch.Tensor:
        return self.movie_tower(movie_features)
