import uuid
from typing import List

from pydantic import BaseModel


class PulseDocument:
    def __init__(self, document_text, original_document_obj):
        self.text: str = document_text
        self.original_document_obj = original_document_obj
        self.document_id = uuid.uuid4()
        self.processed_text: str = None
        self.embedding: List[float] = None
        self.cluster_id = None
        self.cluster_probability: float = 0.0

class PulseTopic(BaseModel):
    topic: str
    description: str
    original_sentences: List[str]


class PulseDocumentFactory:
    @staticmethod
    def create_document_from_dataset_item(dataset_item):
        document_text = (dataset_item["system"] + "\n" + dataset_item["instruction"]).strip()
        return PulseDocument(document_text=document_text, original_document_obj=dataset_item)

    @staticmethod
    def create_document_from_topic(topic):
        return PulseDocument(document_text=f"{topic.topic} : {topic.description}", original_document_obj=topic)

    @staticmethod
    def create_documents_from_topics(topics):
        return [PulseDocumentFactory.create_document_from_topic(topic) for topic in topics]
