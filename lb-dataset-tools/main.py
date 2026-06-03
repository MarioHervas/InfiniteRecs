from pathlib import Path
from dotenv import load_dotenv
import os

from src.zip_extractor import ZipExtractor
from src.preprocessor import preprocess
from src.tmdb_client import TMDBClient
from src.nanogenre_scraper import NanogenreScraper
from src.movie_store import MovieStore
from src.dataset_builder import DatasetConstructor
from src.bertopic_enricher import BertopicEnricher

# 1. Configuration
load_dotenv()
ZIPS_DIR = Path("zips")
OUTPUT_DIR = Path("output_dataset")

# 2. Extraction
extractor = ZipExtractor()
user_data = []

zip_paths = sorted(ZIPS_DIR.glob("*.zip"))

if not zip_paths:
    print(f"No zips found!")
    exit(1)

for zip_path in zip_paths:
    try:
        data = extractor.extract(zip_path)
        user_data.append(data)
        print(f"loaded {data.username:20s}  ratings: {len(data.ratings):4d}  diary: {len(data.diary):4d}")
    except Exception as e:
        print(f"Error")

print(f"\nTotal of {len(user_data)} users loaded")
print("Loaded users:")
for i, data in enumerate(user_data, 1):
    print(f"  {i:2d}. {data.username}")

# 3. Preprocessing
for data in user_data:
    preprocess(data)

# 4. Catalogue construction
tmdb_client = TMDBClient(api_key=os.getenv("TMDB_API_KEY"))
scraper = NanogenreScraper()
movie_store = MovieStore("movie_store.json")

dataset = DatasetConstructor(user_data, tmdb_client, scraper, movie_store)
dataset.build_catalog()
print(f"Catalogue size: {len(movie_store.store)}")

scraper.close()

# 5. BERTopci Enrichment
enricher = BertopicEnricher(movie_store)
enricher.enrich()

# 6. Exporting the CSVs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

movies_df = movie_store.to_dataframe()
interactions_df = dataset.build_interactions()

movies_df.to_csv(OUTPUT_DIR / "movies.csv", index=False)
interactions_df.to_csv(OUTPUT_DIR / "interactions.csv", index=False)

print(f"movies.csv: {len(movies_df)} rows")
print(f"interactions.csv: {len(interactions_df)} rows")