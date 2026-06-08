import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from recommender import Recommender
from tmdb_service import poster_service
from schemas import (
    HealthResponse, MovieRecommendation,
    RecommendationResponse, UserFavorite,
)
from zip_processor import (
    extract_username_from_filename,
    load_ratings_from_zip_bytes,
    load_seen_titles_from_zip_bytes,
)

BASE_DIR = Path(__file__).parent
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(BASE_DIR / "checkpoint.pt"),
)
MOVIES_CSV_PATH = os.environ.get(
    "MOVIES_CSV_PATH",
    str(BASE_DIR / "movies.csv"),
)
FRONTEND_DIR = (BASE_DIR.parent / "frontend").resolve()


app = FastAPI(
    title="InfiniteRecs",
    description="Personalized Letterboxd recommender system",
    version="1.0",
)



_recommender: Optional[Recommender] = None


def get_recommender() -> Recommender:
    global _recommender
    if _recommender is None:
        if not Path(CHECKPOINT_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No checkpoint found!"
                ),
            )
        if not Path(MOVIES_CSV_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "movies.csv not found!"
                ),
            )
        _recommender = Recommender(
            checkpoint_path=CHECKPOINT_PATH,
            movies_csv_path=MOVIES_CSV_PATH,
        )
    return _recommender



@app.get("/api/health", response_model=HealthResponse)
def health():
    rec = get_recommender()
    return HealthResponse(
        status="ok",
        version=rec.version,
        n_users=rec.n_users,
        n_movies=rec.n_movies,
        metrics={k: v for k, v in rec.metrics.items() if k != "per_user"},
    )
@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    use_finetuning: bool = Form(True),
    use_quadratic: bool = Form(True),
    use_anti: bool = Form(True),
):
    rec = get_recommender()
    username = extract_username_from_filename(file.filename or "")
    try:
        zip_bytes = await file.read()
        ratings_df, stats = load_ratings_from_zip_bytes(zip_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Not valid zip file",
        )

    fine_tuning_used = False
    finetune_losses = []
    info = {}

    if username and rec.is_known_user(username):
        scenario = "A"
        user_embedding = rec.get_user_embedding_known(username)

        in_catalog = ratings_df[ratings_df["title_normalized"].isin(rec.movie2id)]
        info = {
            "n_in_catalog": int(len(in_catalog)),
            "n_out_catalog": int(len(ratings_df) - len(in_catalog)),
            "n_positives_used": int((in_catalog["rating_norm"] > rec.positive_threshold).sum()),
            "n_negatives_used": int((in_catalog["rating_norm"] < rec.negative_threshold).sum()),
        }
    else:
        scenario = "B"
        try:
            result = rec.build_coldstart_embedding(
                ratings_df,
                use_quadratic=use_quadratic,
                use_anti=use_anti,
                anti_weight=0.5,
                use_finetuning=use_finetuning,
                finetune_steps=100,
                finetune_lr=0.05,
                finetune_l2=0.01,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        user_embedding = result["embedding"]
        info = result["info"]
        finetune_losses = result["finetune_losses"]
        fine_tuning_used = use_finetuning and len(finetune_losses) > 0
    seen_titles = set(ratings_df["title_normalized"])
    seen_titles |= load_seen_titles_from_zip_bytes(zip_bytes)
    recommendations_raw = rec.recommend(user_embedding, seen_titles, top_k=top_k)
    for r in recommendations_raw:
        r["poster_url"] = poster_service.get_poster_url(r.get("tmdb_id"))


    favorites = rec.get_user_favorites(ratings_df, k=5)
    for f in favorites:
        f["poster_url"] = poster_service.get_poster_url(f.get("tmdb_id"))

    
    return RecommendationResponse(
        username=username,
        scenario=scenario,
        n_total_ratings=stats["n_total"],
        n_in_catalog=info.get("n_in_catalog", 0),
        n_out_catalog=info.get("n_out_catalog", 0),
        n_positives_used=info.get("n_positives_used", 0),
        n_negatives_used=info.get("n_negatives_used", 0),
        fine_tuning_used=fine_tuning_used,
        finetune_losses=finetune_losses,
        favorites=[UserFavorite(**f) for f in favorites],
        recommendations=[MovieRecommendation(**r) for r in recommendations_raw],
    )


@app.get("/")
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail=f"index.html no encontrado en {FRONTEND_DIR}")
    return FileResponse(index)



if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
