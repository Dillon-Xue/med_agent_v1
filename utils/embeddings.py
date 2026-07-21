import dashscope
from langchain_core.embeddings import Embeddings
from utils.config import get_embedding_model

class DashscopeEmbeddings(Embeddings):
    def __init__(self):
        self.model = get_embedding_model()

    def embed_documents(self, texts):
        results = []
        for t in texts:
            try:
                resp = dashscope.TextEmbedding.call(model=self.model, input=t)
                emb = resp.output["embeddings"][0]["embedding"]
                results.append(emb)
            except Exception as e:
                print(f"[EmbedError] embed_documents failed: {e}, returning zero vector")
                results.append([0.0] * 1536)
        return results

    def embed_query(self, text):
        try:
            resp = dashscope.TextEmbedding.call(model=self.model, input=text)
            return resp.output["embeddings"][0]["embedding"]
        except Exception as e:
            print(f"[EmbedError] embed_query failed: {e}, returning zero vector")
            return [0.0] * 1536
