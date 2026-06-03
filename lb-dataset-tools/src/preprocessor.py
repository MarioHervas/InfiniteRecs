import pandas as pd
from src.zip_extractor import ZipExtractor, LetterboxdData
def preprocess(data: LetterboxdData):
    convert_to_date(data)
    normalize_ratings(data)
def convert_to_date(data: LetterboxdData):
    data.ratings["Date"] = pd.to_datetime(data.ratings["Date"])

#TODO - Add different kind of normalization based on date and time elapsed
def normalize_ratings(data: LetterboxdData):
    data.ratings["Rating"] = (data.ratings["Rating"]-0.5)/(5.0-0.5)
