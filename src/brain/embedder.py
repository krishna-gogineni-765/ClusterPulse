from langchain.embeddings import OpenAIEmbeddings


class Embedder:
    def __init__(self, openai_api_key):
        self.embedder = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

    def embed(self, text):
        return self.embedder.embed_query(text=text)

    async def async_embed(self, text):
        return await self.embedder.aembed_query(text=text)
