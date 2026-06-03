from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models import TwoTowerModel
from tmdb_service import letterboxd_url_from_tmdb


class Recommender:
    def __init__(self,
                 checkpoint_path: str,
                 movies_csv_path: str,
                 positive_threshold: float = 0.6,
                 negative_threshold: float = 0.3):
        self.checkpoint_path = checkpoint_path
        self.movies_csv_path = movies_csv_path
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):

        ckpt = torch.load(self.checkpoint_path, map_location=self.device,
                          weights_only=False)

        self.model = TwoTowerModel(
            n_users=ckpt["n_users"],
            movie_feature_dim=ckpt["movie_feature_dim"],
            embedding_dim=ckpt["embedding_dim"],
            hidden_dim=ckpt["hidden_dim"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()


        self.user2id = ckpt["user2id"]
        self.user2id_lc = {u.lower(): i for u, i in self.user2id.items()}
        self.id2user = ckpt["id2user"]
        self.movie2id = ckpt["movie2id"]
        self.id2movie = ckpt["id2movie"]


        feats = torch.tensor(ckpt["movie_features"], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            self.all_movie_embeddings = self.model.encode_movie(feats)
        self.n_movies = self.all_movie_embeddings.shape[0]

       
        self.metrics = ckpt.get("metrics", {})
        self.version = ckpt.get("version", "unknown")
        self.n_users = ckpt["n_users"]

        
        movies = pd.read_csv(self.movies_csv_path)
        for col in ["tmdb_genres", "nanogenres", "themes", "director", "title_original"]:
            if col in movies.columns:
                movies[col] = movies[col].fillna("")
        self.movies_by_title = movies.set_index("title_normalized")

 
    def is_known_user(self, username: Optional[str]) -> bool:
        if not username:
            return False
        return username.lower() in self.user2id_lc

    def get_user_id(self, username: str) -> int:
        return self.user2id_lc[username.lower()]

   
    def get_user_embedding_known(self, username: str) -> torch.Tensor:
        
        user_id = self.get_user_id(username)
        with torch.no_grad():
            emb = self.model.encode_user(
                torch.tensor([user_id], device=self.device)
            ).squeeze(0)
        return emb

    def build_coldstart_embedding(
        self,
        ratings_df: pd.DataFrame,
        use_quadratic: bool = True,
        use_anti: bool = True,
        anti_weight: float = 0.5,
        use_finetuning: bool = True,
        finetune_steps: int = 100,
        finetune_lr: float = 0.05,
        finetune_l2: float = 0.01,
    ) -> dict:
        
        in_catalog = ratings_df[ratings_df["title_normalized"].isin(self.movie2id)].copy()
        n_in_catalog = len(in_catalog)
        n_out_catalog = len(ratings_df) - n_in_catalog

        
        pos_df = in_catalog[in_catalog["rating_norm"] > self.positive_threshold]
        if len(pos_df) == 0:
            raise ValueError(
                f"Not enough movies > {self.positive_threshold} "       
            )

        pos_ids = [self.movie2id[t] for t in pos_df["title_normalized"]]
        pos_ratings = pos_df["rating_norm"].values

        if use_quadratic:
            pos_w_np = np.maximum(0.0, pos_ratings - 0.5) ** 2
        else:
            pos_w_np = pos_ratings.copy()

        nz = pos_w_np > 1e-6
        pos_ids = [m for m, k in zip(pos_ids, nz) if k]
        pos_w_np = pos_w_np[nz]
        if len(pos_ids) == 0:
            
            pos_ids = [self.movie2id[t] for t in pos_df["title_normalized"]]
            pos_w_np = pos_df["rating_norm"].values

        pos_w = torch.tensor(pos_w_np, dtype=torch.float32, device=self.device)
        pos_embs = self.all_movie_embeddings[pos_ids]
        pos_emb = (pos_embs * pos_w.unsqueeze(1)).sum(dim=0) / pos_w.sum()

       
        n_neg_used = 0
        neg_df = in_catalog[in_catalog["rating_norm"] < self.negative_threshold]
        if use_anti and len(neg_df) > 0:
            neg_ids = [self.movie2id[t] for t in neg_df["title_normalized"]]
            neg_ratings = neg_df["rating_norm"].values
            if use_quadratic:
                neg_w_np = np.maximum(0.0, 0.5 - neg_ratings) ** 2
            else:
                neg_w_np = (1.0 - neg_ratings)
            nz = neg_w_np > 1e-6
            neg_ids = [m for m, k in zip(neg_ids, nz) if k]
            neg_w_np = neg_w_np[nz]
            if len(neg_ids) > 0:
                neg_w = torch.tensor(neg_w_np, dtype=torch.float32, device=self.device)
                neg_embs = self.all_movie_embeddings[neg_ids]
                neg_emb = (neg_embs * neg_w.unsqueeze(1)).sum(dim=0) / neg_w.sum()
                user_emb = pos_emb - anti_weight * neg_emb
                n_neg_used = len(neg_ids)
            else:
                user_emb = pos_emb
        else:
            user_emb = pos_emb

        info = {
            "n_in_catalog": n_in_catalog,
            "n_out_catalog": n_out_catalog,
            "n_positives_used": len(pos_ids),
            "n_negatives_used": n_neg_used,
            "norm_initial": float(user_emb.norm().item()),
        }
        finetune_losses = []

        
        if use_finetuning and len(pos_ids) > 0:
            seen_ids = set(self.movie2id[t] for t in in_catalog["title_normalized"])
            user_emb, finetune_losses = self._finetune_user_embedding(
                user_emb,
                pos_ids,
                seen_ids,
                steps=finetune_steps,
                lr=finetune_lr,
                l2=finetune_l2,
            )
            info["norm_final"] = float(user_emb.norm().item())

        return {
            "embedding": user_emb,
            "info": info,
            "finetune_losses": finetune_losses,
        }

    def _finetune_user_embedding(self,
                                 initial_embedding: torch.Tensor,
                                 pos_ids: List[int],
                                 seen_ids: set,
                                 steps: int,
                                 lr: float,
                                 l2: float):
        
        u = initial_embedding.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=lr)
        pos_embs_cached = self.all_movie_embeddings[pos_ids].detach()

        losses = []
        rng = np.random.default_rng(42)
        for _ in range(steps):
            neg_ids = []
            for _ in range(len(pos_ids)):
                while True:
                    cand = int(rng.integers(0, self.n_movies))
                    if cand not in seen_ids:
                        neg_ids.append(cand)
                        break
            neg_embs = self.all_movie_embeddings[neg_ids].detach()

            score_pos = (u.unsqueeze(0) * pos_embs_cached).sum(dim=1)
            score_neg = (u.unsqueeze(0) * neg_embs).sum(dim=1)
            loss = -F.logsigmoid(score_pos - score_neg).mean() + l2 * (u * u).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        return u.detach(), losses


    def recommend(self,
                  user_embedding: torch.Tensor,
                  seen_titles_normalized: set,
                  top_k: int = 10) -> List[dict]:
        
        seen_ids = [self.movie2id[t] for t in seen_titles_normalized
                    if t in self.movie2id]

        with torch.no_grad():
            scores = (user_embedding * self.all_movie_embeddings).sum(dim=1)
        if seen_ids:
            scores[seen_ids] = -1e9

        
        n_available = self.n_movies - len(seen_ids)
        effective_k = max(1, min(top_k, n_available))

        topk = torch.topk(scores, k=effective_k)
        top_ids = topk.indices.cpu().tolist()
        top_scores = topk.values.cpu().tolist()

        return [self._build_movie_dict(rank + 1, mid, score)
                for rank, (mid, score) in enumerate(zip(top_ids, top_scores))]

    def _build_movie_dict(self, rank: int, movie_id: int, score: float) -> dict:
        title_norm = self.id2movie[movie_id]
        out = {
            "rank": rank,
            "title_normalized": title_norm,
            "score": round(float(score), 4),
        }
        if title_norm in self.movies_by_title.index:
            row = self.movies_by_title.loc[title_norm]
            tmdb_id = row.get("tmdb_id", "")
            tmdb_id = "" if tmdb_id is None else str(tmdb_id)
            
            if tmdb_id.endswith(".0"):
                tmdb_id = tmdb_id[:-2]

            
            lb_url = row.get("lb_url", "")
            lb_url = "" if lb_url is None else str(lb_url).strip()
            if lb_url in ("", "nan"):
                lb_url = letterboxd_url_from_tmdb(tmdb_id) if tmdb_id else None

            out.update({
                "title":       str(row.get("title_original", title_norm) or title_norm),
                "year":        str(row.get("year", "") or ""),
                "director":    str(row.get("director", "") or ""),
                "tmdb_genres": str(row.get("tmdb_genres", "") or ""),
                "nanogenres":  str(row.get("nanogenres", "") or ""),
                "themes":      str(row.get("themes", "") or ""),
                "tmdb_id":     tmdb_id,
                "letterboxd_url": lb_url or None,
            })
        else:
            out["title"] = title_norm
        return out

    def get_user_favorites(self, ratings_df: pd.DataFrame, k: int = 5) -> List[dict]:
        in_catalog = ratings_df[ratings_df["title_normalized"].isin(self.movie2id)].copy()
        top = in_catalog.sort_values("rating_norm", ascending=False).head(k)

        out = []
        for _, r in top.iterrows():
            title_norm = r["title_normalized"]
            d = {"title_normalized": title_norm,
                 "rating_norm": round(float(r["rating_norm"]), 3)}
            if title_norm in self.movies_by_title.index:
                row = self.movies_by_title.loc[title_norm]
                tmdb_id = row.get("tmdb_id", "")
                tmdb_id = "" if tmdb_id is None else str(tmdb_id)
                if tmdb_id.endswith(".0"):
                    tmdb_id = tmdb_id[:-2]
                lb_url = row.get("lb_url", "")
                lb_url = "" if lb_url is None else str(lb_url).strip()
                if lb_url in ("", "nan"):
                    lb_url = None
                d.update({
                    "title":       str(row.get("title_original", title_norm) or title_norm),
                    "year":        str(row.get("year", "") or ""),
                    "tmdb_genres": str(row.get("tmdb_genres", "") or ""),
                    "tmdb_id":     tmdb_id,
                    "letterboxd_url": lb_url,
                })
            else:
                d["title"] = title_norm
            out.append(d)
        return out
