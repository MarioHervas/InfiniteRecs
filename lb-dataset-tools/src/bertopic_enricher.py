from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from src.movie_store import MovieStore


class BertopicEnricher:

    def __init__(self, movie_store: MovieStore):
        self.movie_store = movie_store

    def enrich(self) -> None:
        titles, docs = self._extract_overviews()
        print(f"[BERTopic] Training over {len(docs)} synopses")
        topic_model = self._build_model()
        topics, _= topic_model.fit_transform(docs)

        self._update_store(titles, topics, topic_model)
        self.movie_store._save()

        n_topics = len(set(t for t in topics if t != -1))
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"[BERTopic] done. {n_topics} found topics, {n_outliers} outliers.")

    def _extract_overviews(self) -> tuple[list[str], list[str]]:
        titles, docs = [], []
        for title, data in self.movie_store.store.items():
            overview = data.get("overview", "").strip()
            if overview:
                titles.append(title)
                docs.append(overview)
        return titles, docs

    def _build_model(self) -> BERTopic:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model = UMAP(n_neighbors=15, n_components=5, random_state=42)
        cluster_model = KMeans(n_clusters=20, random_state=42)
        vectorizer = CountVectorizer(stop_words="english")

        return BERTopic(
            embedding_model=sentence_model,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer,
        )

    def _update_store(self, titles: list[str], topics: list[int], topic_model: BERTopic) -> None:
        for i, title in enumerate(titles):
            topic_id = topics[i]
            if topic_id == -1:
                themes = ""
            else:
                words  = [word for word, _ in topic_model.get_topic(topic_id)]
                themes = "|".join(words[:5])
            self.movie_store.update_themes(title, themes)