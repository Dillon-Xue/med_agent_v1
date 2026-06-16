import os
import dashscope
from langchain_core.embeddings import Embeddings

class DashscopeEmbeddings(Embeddings):
    def __init__(self):
        self.model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")

    def embed_documents(self, texts):
        return [
            dashscope.TextEmbedding.call(
                model=self.model,
                input=t
            ).output["embeddings"][0]["embedding"]
            for t in texts
        ]

    def embed_query(self, text):
        return dashscope.TextEmbedding.call(
            model=self.model,
            input=text
        ).output["embeddings"][0]["embedding"]
