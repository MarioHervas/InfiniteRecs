
from typing import List, Optional
from pydantic import BaseModel


class MovieRecommendation(BaseModel):
    rank: int
    title: str
    title_normalized: str
    year: Optional[str] = ""
    director: Optional[str] = ""
    tmdb_genres: Optional[str] = ""
    nanogenres: Optional[str] = ""
    themes: Optional[str] = ""
    score: float
    tmdb_id: Optional[str] = ""
    poster_url: Optional[str] = None
    letterboxd_url: Optional[str] = None


class UserFavorite(BaseModel):
    title: str
    title_normalized: str
    year: Optional[str] = ""
    tmdb_genres: Optional[str] = ""
    rating_norm: float
    tmdb_id: Optional[str] = ""
    poster_url: Optional[str] = None
    letterboxd_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    username: Optional[str]
    scenario: str 
    n_total_ratings: int
    n_in_catalog: int
    n_out_catalog: int
    n_positives_used: int
    n_negatives_used: int
    fine_tuning_used: bool
    finetune_losses: List[float] = []
    favorites: List[UserFavorite]
    recommendations: List[MovieRecommendation]


class HealthResponse(BaseModel):
    status: str
    version: str
    n_users: int
    n_movies: int
    metrics: dict
