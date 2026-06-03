import zipfile
from typing import Optional

import pandas as pd
from pathlib import Path
from dataclasses import dataclass

@dataclass
class LetterboxdData:
    ratings: pd.DataFrame
    diary: pd.DataFrame
    watched: pd.DataFrame
    watchlist: Optional[pd.DataFrame] = None
    reviews: Optional[pd.DataFrame] = None
    username: str = ""

class ZipExtractor:
    def extract(self,zip_path: Path) -> LetterboxdData:
        with self._open_zip(zip_path) as zf:
            username = self._extract_username(zip_path)
            ratings = self._load_csv(zf,"ratings.csv")
            diary = self._load_csv(zf, "diary.csv")
            watched = self._load_csv(zf, "watched.csv")
            watchlist = None
            reviews = None
            if "watchlist.csv" in zf.namelist():
                watchlist = self._load_csv(zf, "watchlist.csv")
            if "reviews.csv" in zf.namelist():
                reviews = self._load_csv(zf, "reviews.csv")
            return LetterboxdData(ratings=ratings,diary=diary,watched=watched,watchlist=watchlist,reviews=reviews,username=username)

    def _open_zip(self, zip_path: Path) -> zipfile.ZipFile:
        zf = zipfile.ZipFile(zip_path,"r")
        return zf
    def _extract_username(self,zip_path: Path) -> str:
        return zip_path.name.split('-')[1]
    def _load_csv(self,zf : zipfile.ZipFile,filename: str) -> pd.DataFrame:
        csv = zf.open(filename)
        return pd.read_csv(csv)




