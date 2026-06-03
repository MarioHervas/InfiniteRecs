import json
from pathlib import Path
import pandas as pd

class MovieStore:

    def __init__(self, store_path: str = "movie_store.json"):
        self.store_path = Path(store_path)
        self.store = self._load()

    def is_processed(self, title_normalized: str) -> bool:
        return title_normalized in self.store

    def add_movie(self, movie: dict) -> None:
        title_normalized = movie["title_normalized"]
        self.store[title_normalized] = movie
        self._save()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.store, orient="index").reset_index(drop=True)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.store, orient="index").reset_index(drop=True)

    def _save(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2, ensure_ascii=False)

    def _load(self) -> dict:
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def update_themes(self, title_normalized: str, themes: str) -> None:
        if title_normalized in self.store:
            self.store[title_normalized]["themes"] = themes