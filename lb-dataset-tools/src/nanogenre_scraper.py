# src/nanogenre_scraper.py
import time
import random
import requests
import json
from pathlib import Path
from bs4 import BeautifulSoup
from curl_cffi import requests as cf_requests

class NanogenreScraper:

    def __init__(self, cookies_path: str = "cookies.json", cache_path: str = "nanogenre_cache.json"):
        self.session = cf_requests.Session(impersonate="chrome120")
        self._load_cookies(cookies_path)
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()

    def _load_cookies(self, cookies_path: str) -> None:
        with open(cookies_path, "r") as f:
            cookies_list = json.load(f)
        for cookie in cookies_list:
            self.session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie["domain"],
                path=cookie.get("path", "/")
            )

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def scrape(self, letterboxd_uri: str) -> list[str]:
        if letterboxd_uri in self.cache:
            return self.cache[letterboxd_uri]

        url = self._get_url(letterboxd_uri)
        nanogenres = self._get_nanogenres(url)


        self.cache[letterboxd_uri] = nanogenres
        self._save_cache()

        return nanogenres

    def _get_url(self, letterboxd_uri: str) -> str:
        response = self.session.get(letterboxd_uri)
        return f"{response.url}nanogenres"

    def _get_nanogenres(self, url: str) -> list[str]:
        response = self.session.get(url, timeout=15)
        print(f"Status: {response.status_code} for {url}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        nanogenres = []
        for a in soup.select('a[href*="/films/nanogenre/"]'):
            label = a.select_one("span.label")
            if label:
                text = label.get_text(strip=True)
                if text and text not in nanogenres:
                    nanogenres.append(text)
        time.sleep(random.uniform(1.5, 3.5))
        return nanogenres[:10]

    def close(self) -> None:
        self.session.close()