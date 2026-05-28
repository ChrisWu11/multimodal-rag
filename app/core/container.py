from functools import lru_cache

from app.core.config import Settings, get_settings
from app.rag.embeddings import get_embedding_provider
from app.rag.generator import RagAnswerGenerator
from app.rag.ingestion import DocumentIngestor
from app.rag.model_providers import active_embedding_model
from app.rag.pipeline import RagPipeline
from app.rag.retriever import HybridRetriever
from app.rag.storage import RagStore
from app.services.image_analysis import ImageAnalyzer


class AppContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = RagStore(settings.sqlite_path)
        self.store.init_db()
        self.embeddings = get_embedding_provider(settings)
        self.image_analyzer = ImageAnalyzer(settings)
        self.ingestor = DocumentIngestor(
            settings=settings,
            store=self.store,
            embeddings=self.embeddings,
            image_analyzer=self.image_analyzer,
        )
        self.retriever = HybridRetriever(
            store=self.store,
            embeddings=self.embeddings,
            embedding_provider=settings.embedding_provider,
            embedding_model=active_embedding_model(settings),
        )
        self.generator = RagAnswerGenerator(settings=settings)
        self.pipeline = RagPipeline(
            settings=settings,
            retriever=self.retriever,
            generator=self.generator,
            image_analyzer=self.image_analyzer,
        )

    def create_pipeline(self, settings: Settings) -> RagPipeline:
        embeddings = get_embedding_provider(settings)
        image_analyzer = ImageAnalyzer(settings)
        return RagPipeline(
            settings=settings,
            retriever=HybridRetriever(
                store=self.store,
                embeddings=embeddings,
                embedding_provider=settings.embedding_provider,
                embedding_model=active_embedding_model(settings),
            ),
            generator=RagAnswerGenerator(settings=settings),
            image_analyzer=image_analyzer,
        )

    def create_ingestor(self, settings: Settings) -> DocumentIngestor:
        return DocumentIngestor(
            settings=settings,
            store=self.store,
            embeddings=get_embedding_provider(settings),
            image_analyzer=ImageAnalyzer(settings),
        )


@lru_cache
def get_container() -> AppContainer:
    return AppContainer(get_settings())
