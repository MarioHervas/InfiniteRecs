
import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import requests

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_API_TOKEN = os.environ.get("TMDB_API_TOKEN", "").strip()


IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
CACHE_PATH = Path(__file__).parent / "poster_cache.json"

REQUEST_TIMEOUT = 4.0


class PosterService:
    def __init__(self):
        self._cache: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()
        self._load_cache()
        self.enabled = bool(TMDB_API_KEY or TMDB_API_TOKEN)
    def _load_cache(self):
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self):
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except OSError:
            pass

    def _fetch_poster_path(self, tmdb_id: str) -> Optional[str]:
        if not self.enabled:
            return None

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        headers = {}
        params = {}
        if TMDB_API_TOKEN:
            headers["Authorization"] = f"Bearer {TMDB_API_TOKEN}"
        else:
            params["api_key"] = TMDB_API_KEY

        try:
            resp = requests.get(url, headers=headers, params=params,
                                timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("poster_path")
        except (requests.RequestException, ValueError):
            return None

    def get_poster_url(self, tmdb_id) -> Optional[str]:
        if tmdb_id is None or str(tmdb_id).strip() in ("", "nan"):
            return None
        key = str(int(float(tmdb_id))) if _is_floatish(tmdb_id) else str(tmdb_id).strip()
        if key in self._cache:
            path = self._cache[key]
            return f"{IMAGE_BASE}{path}" if path else None
        path = self._fetch_poster_path(key)
        with self._lock:
            self._cache[key] = path
            self._save_cache()
        return f"{IMAGE_BASE}{path}" if path else None


def _is_floatish(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False



poster_service = PosterService()


def letterboxd_url_from_tmdb(tmdb_id) -> Optional[str]:

    if tmdb_id is None or str(tmdb_id).strip() in ("", "nan"):
        return None
    key = str(int(float(tmdb_id))) if _is_floatish(tmdb_id) else str(tmdb_id).strip()
    return f"https://letterboxd.com/tmdb/{key}"
