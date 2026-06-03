from src.zip_extractor import LetterboxdData
from src.tmdb_client import TMDBClient
from src.nanogenre_scraper import NanogenreScraper
from src.movie_store import MovieStore
import pandas as pd
from pathlib import Path


class DatasetConstructor:

    def __init__(self, user_data: list[LetterboxdData], tmdb_client: TMDBClient, nanogenre_scraper: NanogenreScraper,
                 movie_store: MovieStore):
        self.user_data = user_data
        self.tmdb_client = tmdb_client
        self.nanogenre_scraper = nanogenre_scraper
        self.movie_store = movie_store

    def build_catalog(self) -> pd.DataFrame:
        all_movies = pd.concat(
            [user.ratings[["Name", "Year", "Letterboxd URI"]] for user in self.user_data],
            ignore_index=True
        ).drop_duplicates(subset=["Name", "Year"])

        total = len(all_movies)

        for i, (_, movie) in enumerate(all_movies.iterrows(), 1):
            title_normalized = movie["Name"].lower()
            print(f"[{i}/{total}] {movie['Name']}")

            if self.movie_store.is_processed(title_normalized):
                print("Found")
                continue

            # TMDB
            tmdb_id = self.tmdb_client.get_tmdb_id(movie["Name"], movie["Year"])
            if tmdb_id is None:
                print(" ✗ Not found on tmdb")
                continue

            details = self.tmdb_client.get_movie_request(tmdb_id)

            nanogenres = self.nanogenre_scraper.scrape(movie["Letterboxd URI"])
            if nanogenres:
                print(f"{len(nanogenres)} nanogenres")
            else:
                print("No Nano Found")

            entry = {
                "title_normalized": title_normalized,
                "title_original": movie["Name"],
                "tmdb_id": tmdb_id,
                "year": details["year"],
                "director": "|".join(details["director"]),
                "tmdb_genres": "|".join(details["genres"]),
                "overview": details["overview"],
                "country": details["country"],
                "nanogenres": "|".join(nanogenres),
                "themes": "",  # se rellena en la fase BERTopic (BertopicEnricher)
                "lb_url": movie["Letterboxd URI"],
            }

            self.movie_store.add_movie(entry)

        return self.movie_store.to_dataframe()

    def build_interactions(self) -> pd.DataFrame:
        frames = []
        for user in self.user_data:
            df = user.ratings[["Name", "Year", "Rating", "Date"]].copy()
            df["username"] = user.username
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def export(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.movie_store.to_dataframe().to_csv(output_path / "movies.csv", index=False)
        self.build_interactions().to_csv(output_path / "interactions.csv", index=False)