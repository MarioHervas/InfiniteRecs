import requests

class TMDBClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_info(self,title: str, year : int) -> dict:
        return(self.get_movie_request(self.get_tmdb_id(title,year)))


    def get_tmdb_id(self, title: str, year: int) -> int:
        response = requests.get(f"https://api.themoviedb.org/3/search/movie?api_key={self.api_key}&query={title}&year={year}")
        results = response.json()["results"]
        if results:
            return results[0]["id"]
        return None

    def get_movie_request(self,tmdb_id: int) -> dict:
        response = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={self.api_key}&append_to_response=credits")
        data = response.json()
        director = []
        for member in data["credits"]["crew"]:
            if member["job"]=="Director":
                director.append(member["name"])

        return {
            "genres": [g["name"] for g in data ["genres"]],
            "overview": data["overview"],
            "director": director,
            "country": data["production_countries"][0]["iso_3166_1"] if data["production_countries"] else None,
            "year": data["release_date"][:4] if data["release_date"] else None,
        }

