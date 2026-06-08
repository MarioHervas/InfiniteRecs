# InfiniteRecs

InfiniteRecs is a personalized movie recommender system based on Letterboxd user profiles.

The system processes Letterboxd export files, enriches the movie catalogue with TMDB metadata, Letterboxd nanogenres and BERTopic themes, trains a PyTorch Two-Tower recommender model with Bayesian Personalized Ranking loss, and serves the trained model through a FastAPI web application.

This project was developed as part of the final degree project:

**Development of a Personalised Recommender System Based on Letterboxd User Profiles**  
Mario Hervás Núñez  
Grado en Ingeniería del Software  
ETSISI - Universidad Politécnica de Madrid

## Repository structure

```text
.
├── lb-dataset-tools/
│   ├── src/
│   ├── zips/
│   ├── output_dataset/
│   ├── main.py
│   ├── movie_store.json
│   ├── nanogenre_cache.json
│   ├── cookies.json
│   ├── .env.example
│   └── requirements.txt
│
├── pytorch-implementation/
│   ├── lb-two-tower.ipynb
│   ├── movies.csv
│   ├── interactions.csv
│   └── checkpoint.pt
│
└── fastapi-implementation/
    ├── backend/
    │   ├── main.py
    │   ├── recommender.py
    │   ├── models.py
    │   ├── schemas.py
    │   ├── zip_processor.py
    │   ├── tmdb_service.py
    │   ├── checkpoint.pt
    │   ├── movies.csv
    │   ├── interactions.csv
    │   ├── .env.example
    │   └── requirements.txt
    │
    ├── frontend/
    │   ├── index.html
    │   ├── app.js
    │   ├── styles.css
    │   └── favicon.ico
    │
    └── evaluation/
        ├── evaluate.ipynb
        ├── figures/
        └── zips/
```

## Main components

## `lb-dataset-tools`

Contains the data pipeline used to create the training dataset.

The pipeline reads Letterboxd export ZIP files, extracts user ratings, normalizes ratings, enriches movies with TMDB metadata, retrieves Letterboxd nanogenres, applies BERTopic to movie overviews, and exports the final CSV files used by the model.

Main output files:

```text
lb-dataset-tools/output_dataset/movies.csv
lb-dataset-tools/output_dataset/interactions.csv
```

## `pytorch-implementation`

Contains the notebook used to train the recommendation model.

The model is implemented in PyTorch using a Two-Tower architecture:

- the user tower learns a user embedding;
- the item tower encodes movie features;
- the affinity score is computed with a dot product;
- training is performed with Bayesian Personalized Ranking loss.

The notebook uses:

```text
pytorch-implementation/movies.csv
pytorch-implementation/interactions.csv
```

and generates:

```text
pytorch-implementation/checkpoint.pt
```

## `fastapi-implementation`

Contains the deployed recommendation system.

The backend loads the trained model checkpoint and exposes an API with FastAPI. The frontend is a static web interface where users can upload their Letterboxd export ZIP and receive movie recommendations.

The frontend is served directly by the FastAPI backend, so no separate JavaScript build step is required.

## Requirements

The project requires Python and pip. A virtual environment is recommended.

Recommended version:

```text
Python 3.10+
```

Main external requirements:

- TMDB API key for metadata and poster retrieval;
- Letterboxd export ZIP files for dataset creation and recommendation testing;
- Jupyter Notebook or Google Colab for model training;
- PyTorch for training and inference;
- FastAPI and Uvicorn for serving the web application.

## Environment variables

Both the dataset pipeline and the backend include a `.env.example` file.

Create a `.env` file from the example:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then add your TMDB API key:

```env
TMDB_API_KEY=your_api_key_here
```

The backend also supports the following optional variables:

```env
TMDB_API_TOKEN=your_tmdb_bearer_token_here
CHECKPOINT_PATH=path/to/checkpoint.pt
MOVIES_CSV_PATH=path/to/movies.csv
```

If `CHECKPOINT_PATH` and `MOVIES_CSV_PATH` are not provided, the backend uses the files located in `fastapi-implementation/backend/`.

## Quick start: run the web application

A quick and simple start is included after creating the .env file. The scritpt run.bat works on Windows. Use this option if the trained model checkpoint and the dataset CSV files are already available inside `fastapi-implementation/backend/`.

From the repository root:

```bash
cd fastapi-implementation/backend
python -m venv .venv
```

Activate the environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and add your TMDB API key:

```env
TMDB_API_KEY=your_api_key_here
```

Start the application:

```bash
python main.py
```

Alternatively:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

## Using the web application

1. Export your data from Letterboxd.
2. Upload the exported `.zip` file in the InfiniteRecs web interface.
3. Choose the number of recommendations.
4. Optionally enable or disable the advanced cold-start options.
5. Submit the file and wait for the recommendations.

The uploaded ZIP must contain at least:

```text
ratings.csv
```

If available, the backend also uses:

```text
watched.csv
```

to filter out movies the user has already watched.

The default Letterboxd filename format is recommended:

```text
letterboxd-<username>-<date>.zip
```

If the username matches one of the users known by the trained model, the backend uses the learned user embedding. Otherwise, it builds a cold-start embedding from the ratings contained in the uploaded ZIP.

## API endpoints

## Health check

```http
GET /api/health
```

Returns model status, version, number of users, number of movies, and stored evaluation metrics.

## Recommendation endpoint

```http
POST /api/recommend
```

Expected form fields:

| Field | Type | Description |
|---|---|---|
| `file` | ZIP file | Letterboxd export file |
| `top_k` | integer | Number of recommendations to return |
| `use_finetuning` | boolean | Whether to refine the cold-start embedding with gradient descent |
| `use_quadratic` | boolean | Whether to apply quadratic rating weighting |
| `use_anti` | boolean | Whether to subtract an anti-embedding built from disliked movies |

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/api/recommend" \
  -F "file=@/path/to/letterboxd-user-date.zip" \
  -F "top_k=10" \
  -F "use_finetuning=true" \
  -F "use_quadratic=true" \
  -F "use_anti=true"
```

## Rebuilding the dataset

Use this option if you want to regenerate `movies.csv` and `interactions.csv` from Letterboxd export ZIP files.

From the repository root:

```bash
cd lb-dataset-tools
python -m venv .venv
```

Activate the environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
TMDB_API_KEY=your_api_key_here
```

Add the Letterboxd export ZIP files to:

```text
lb-dataset-tools/zips/
```

Then run the pipeline:

```bash
python main.py
```

The generated dataset will be written to:

```text
lb-dataset-tools/output_dataset/movies.csv
lb-dataset-tools/output_dataset/interactions.csv
```

## Dataset pipeline details

The dataset pipeline performs the following steps:

1. Reads Letterboxd ZIP exports from `lb-dataset-tools/zips/`.
2. Extracts `ratings.csv`, `diary.csv`, `watched.csv`, and optionally `watchlist.csv` and `reviews.csv`.
3. Normalizes ratings from the Letterboxd 0.5-5 scale into a 0-1 range.
4. Queries TMDB to retrieve metadata such as genres, overview, director, country, release year and TMDB ID.
5. Scrapes Letterboxd nanogenres for each movie.
6. Stores intermediate movie metadata in `movie_store.json`.
7. Stores nanogenre results in `nanogenre_cache.json`.
8. Applies BERTopic over movie overviews to generate thematic features.
9. Exports `movies.csv` and `interactions.csv`.

## Nanogenre scraping note

The dataset pipeline can retrieve Letterboxd nanogenres by scraping public movie pages.  
However, depending on Letterboxd access restrictions, the scraper may require valid browser cookies from a Letterboxd session.

To generate the required `cookies.json` file:

1. Log in to Letterboxd from your browser.
2. Install a browser extension such as **Cookie-Editor**.
3. Open Letterboxd in the browser.
4. Use Cookie-Editor to export the cookies in JSON format.
5. Save the exported content as:

```text
lb-dataset-tools/cookies.json
```

## Training the model

The model training is performed in:

```text
pytorch-implementation/lb-two-tower.ipynb
```

The notebook was designed to run in Google Colab. By default, it expects:

```python
MOVIES_CSV = "/content/movies.csv"
INTERACTIONS_CSV = "/content/interactions.csv"
CHECKPOINT_PATH = "/content/checkpoint.pt"
```

To train the model:

1. Open the notebook in Google Colab or Jupyter.
2. Upload or provide `movies.csv` and `interactions.csv`.
3. Update the paths at the top of the notebook if necessary.
4. Run all cells.
5. Download or save the generated `checkpoint.pt`.
6. Copy the checkpoint to:

```text
fastapi-implementation/backend/checkpoint.pt
```

The notebook uses a temporal validation split and stores the evaluation metrics inside the checkpoint.

## Running the evaluation notebook

The evaluation notebook is located at:

```text
fastapi-implementation/evaluation/evaluate.ipynb
```

It reuses the same recommender logic used by the web application and generates evaluation figures.

To run it:

1. Place the required Letterboxd ZIP files in:

```text
fastapi-implementation/evaluation/zips/
```

2. Open the notebook.
3. Run all cells.

Generated figures are saved in:

```text
fastapi-implementation/evaluation/figures/
```

## Data files

## `movies.csv`

Contains enriched movie metadata. Main columns include:

| Column | Description |
|---|---|
| `title_normalized` | Lowercase movie title used as internal key |
| `title_original` | Original movie title |
| `tmdb_id` | TMDB movie identifier |
| `year` | Release year |
| `director` | Movie director or directors |
| `tmdb_genres` | TMDB genres separated by `|` |
| `overview` | TMDB overview |
| `country` | Production country |
| `nanogenres` | Letterboxd nanogenres separated by `|` |
| `themes` | BERTopic-derived theme words |
| `lb_url` | Letterboxd URL |

## `interactions.csv`

Contains user-rating interactions. Main columns include:

| Column | Description |
|---|---|
| `Name` | Movie title |
| `Year` | Movie year |
| `Rating` | Normalized rating |
| `Date` | Rating date |
| `username` | Letterboxd username |

## Privacy notes

Raw Letterboxd ZIP exports are not included in this repository for privacy reasons. Usernames and interactions are offered with permission of the participating users.

Folders that may contain private user exports are empty:

```text
lb-dataset-tools/zips/
fastapi-implementation/evaluation/zips/
```

## Troubleshooting

## `No zips found!`

The dataset pipeline did not find any Letterboxd export ZIPs.

Place ZIP files in:

```text
lb-dataset-tools/zips/
```

and run:

```bash
python main.py
```

## `No checkpoint found!`

The backend could not find the trained model.

Check that this file exists:

```text
fastapi-implementation/backend/checkpoint.pt
```

or define a custom path:

```env
CHECKPOINT_PATH=path/to/checkpoint.pt
```

## `movies.csv not found!`

The backend could not find the movie catalogue.

Check that this file exists:

```text
fastapi-implementation/backend/movies.csv
```

or define a custom path:

```env
MOVIES_CSV_PATH=path/to/movies.csv
```

## `No ratings.csv found!`

The uploaded file is not a valid Letterboxd export ZIP, or the ZIP does not contain `ratings.csv`.

Export your data directly from Letterboxd and upload the generated ZIP file without manually modifying its internal structure.

## Posters are missing

Poster retrieval requires a TMDB API key or token.

Add one of the following to `.env`:

```env
TMDB_API_KEY=your_api_key_here
```

or:

```env
TMDB_API_TOKEN=your_tmdb_bearer_token_here
```

The recommender can still work without posters, but the visual interface will be less complete.

## Notes for development

- The frontend is static and is served from `fastapi-implementation/frontend/`.
- The backend API is implemented in `fastapi-implementation/backend/main.py`.
- The deployed recommender logic is implemented in `fastapi-implementation/backend/recommender.py`.
- The model architecture used at inference time is defined in `fastapi-implementation/backend/models.py`.
- The dataset creation pipeline starts from `lb-dataset-tools/main.py`.
- The training notebook contains hardcoded Colab paths by default; update them if running locally.

