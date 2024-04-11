import asyncio
import os
import uuid

import numpy as np
from langchain import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans

from src.entities.pulse_cluster import PulseCluster
from src.brain.prompt_templates import *
from src.utils.async_utils import gather_with_concurrency
from src.utils.serialization import Serializer


class ClusterPulse:
    def __init__(self, documents, llm_model, embedding_model,
                 document_processing_prompt=DEFAULT_DOCUMENT_PROCESSING_PROMPT_TEMPLATE,
                 cluster_labelling_prompt=DEFAULT_CLUSTER_LABELING_PROMPT_TEMPLATE,
                 cluster_description_prompt=DEFAULT_CLUSTER_DESCRIPTION_PROMPT_TEMPLATE):
        self.documents = documents
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.document_processing_prompt = document_processing_prompt
        self.cluster_labelling_prompt = cluster_labelling_prompt
        self.cluster_description_prompt = cluster_description_prompt

    def train(self, min_cluster_size=5):
        if len(self.documents) <= min_cluster_size:
            print("Not enough documents to cluster, skipping clustering")
            return
        for document in self.documents:
            self.process_document_with_llm(document)
            self.assign_embedding_to_document(document)
        self.clusters = self.cluster_documents_kmeans(cluster_size=min_cluster_size)
        self.assign_cluster_description()
        self.assign_differentiated_cluster_labels_v2()

    async def async_train(self, min_cluster_size=5, cache_directory=None):
        async def async_process_document_and_get_embedding(document, cache_directory):
            try:
                await self.async_process_document_with_llm(document)
                await self.async_assign_embedding_to_document(document)
                if cache_directory is not None:
                    cache_directory = os.path.join(cache_directory, "pulse_documents")
                    await Serializer.serialize_pulse_document_to_disk(document, cache_directory)
                return document.embedding
            except Exception as e:
                print("Error processing document: {}".format(e))
                return None

        if len(self.documents) <= min_cluster_size:
            print("Not enough documents to cluster, skipping clustering")
            return
        tasks = []
        for document in self.documents:
            tasks.append(async_process_document_and_get_embedding(document, cache_directory))
        await gather_with_concurrency(15, *tasks)
        self.documents = [document for document in self.documents if document.embedding is not None]
        print("Finished processing documents, starting clustering")
        self.clusters = self.cluster_documents_kmeans(cluster_size=min_cluster_size)
        await self.async_assign_cluster_description()
        await self.async_assign_differentiated_cluster_labels_v2()

    def process_document_with_llm(self, document):
        if document.processed_text is not None:
            return
        if self.document_processing_prompt is None:
            document.processed_text = document.text
        else:
            document.processed_text = self.llm_model.simple_generate(
                self.get_document_processing_prompt(document.text))
        print("Processed text: {}".format(document.processed_text))

    async def async_process_document_with_llm(self, document):
        if document.processed_text is not None:
            return
        if self.document_processing_prompt is None:
            document.processed_text = document.text
        else:
            document.processed_text = await self.llm_model.async_simple_generate(
                self.get_document_processing_prompt(document.text))
        # print("Processed text: {}".format(document.processed_text))

    def assign_embedding_to_document(self, document):
        if document.embedding is not None:
            return
        document.embedding = self.embedding_model.embed(document.processed_text)

    async def async_assign_embedding_to_document(self, document):
        if document.embedding is not None:
            return
        document.embedding = await self.embedding_model.async_embed(document.processed_text)

    def cluster_documents_kmeans(self, cluster_size=10):
        kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(
            np.array([document.embedding for document in self.documents]))
        pulse_clusters = {}
        self.sklearn_cluster_id_to_pulse_cluster_id = {}
        for document, sklearn_cluster_id in zip(self.documents, kmeans.labels_):
            if sklearn_cluster_id < 0:
                cluster_id = int(sklearn_cluster_id)
                self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id] = cluster_id
            elif sklearn_cluster_id in self.sklearn_cluster_id_to_pulse_cluster_id:
                cluster_id = self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id]
            else:
                cluster_id = uuid.uuid4()
                self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id] = cluster_id
            document.cluster_id = cluster_id
            document.cluster_probability = 1
            if str(cluster_id) not in pulse_clusters:
                pulse_clusters[str(cluster_id)] = PulseCluster(cluster_id=cluster_id)
            pulse_clusters[str(cluster_id)].add_document(document)
        for sklearn_cluster_id, cluster_id in self.sklearn_cluster_id_to_pulse_cluster_id.items():
            if str(cluster_id) in pulse_clusters and sklearn_cluster_id >= 0:
                pulse_clusters[str(cluster_id)].cluster_centroid = kmeans.cluster_centers_[sklearn_cluster_id]

        self._clustering_processor = kmeans
        self._cluster_centroids = kmeans.cluster_centers_
        return pulse_clusters

    def cluster_documents_hdb(self, min_cluster_size=5):
        hdb = HDBSCAN(min_cluster_size=min(min_cluster_size, len(self.documents)), store_centers="centroid")
        hdb.fit(np.array([document.embedding for document in self.documents]))

        pulse_clusters = {}
        self.sklearn_cluster_id_to_pulse_cluster_id = {}
        for document, sklearn_cluster_id, prob in zip(self.documents, hdb.labels_, hdb.probabilities_):
            if sklearn_cluster_id < 0:
                cluster_id = int(sklearn_cluster_id)
                self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id] = cluster_id
            elif sklearn_cluster_id in self.sklearn_cluster_id_to_pulse_cluster_id:
                cluster_id = self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id]
            else:
                cluster_id = uuid.uuid4()
                self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id] = cluster_id
            document.cluster_id = cluster_id
            document.cluster_probability = prob
            if cluster_id not in pulse_clusters:
                pulse_clusters[cluster_id] = PulseCluster(cluster_id=cluster_id)
            pulse_clusters[cluster_id].add_document(document)

        for sklearn_cluster_id, cluster_id in self.sklearn_cluster_id_to_pulse_cluster_id.items():
            if cluster_id in pulse_clusters and sklearn_cluster_id >= 0:
                pulse_clusters[cluster_id].cluster_centroid = hdb.centroids_[sklearn_cluster_id]
        self._clustering_processor = hdb
        self._cluster_centroids = hdb.centroids_
        return pulse_clusters

    def assign_cluster_description(self):
        for cluster in self.clusters.values():
            if type(cluster.cluster_id) == int and cluster.cluster_id < 0:
                cluster.cluster_label = "Noise"
                cluster.cluster_description = "Noise"
                continue
            representative_points = cluster.get_representative_points()
            representative_points_lst = [point.processed_text for point in representative_points]
            # cluster_label = self.llm_model.simple_generate(
            #     self.get_cluster_labelling_prompt(representative_points_lst))
            # if len(cluster_label.split(" ")) > 7:
            #     raise Exception("Cluster label too long")
            # cluster.cluster_label = cluster_label
            cluster_description = self.llm_model.simple_generate(
                self.get_cluster_description_prompt(representative_points_lst))
            cluster.cluster_description = cluster_description

    async def async_assign_cluster_description(self):
        async def async_get_cluster_description_helper(cluster):
            if type(cluster.cluster_id) == int and cluster.cluster_id < 0:
                cluster.cluster_label = "Noise"
                cluster.cluster_description = "Noise"
                return
            representative_points = cluster.get_representative_points()
            representative_points_lst = [point.processed_text for point in representative_points]
            # cluster_label = await self.llm_model.async_simple_generate(
            #     self.get_cluster_labelling_prompt(representative_points_lst))
            # if len(cluster_label.split(" ")) > 12:
            #     raise Exception("Cluster label too long")
            # cluster.cluster_label = cluster_label
            cluster_description = await self.llm_model.async_simple_generate(
                self.get_cluster_description_prompt(representative_points_lst))
            cluster.cluster_description = cluster_description

        tasks = []
        for cluster in self.clusters.values():
            tasks.append(async_get_cluster_description_helper(cluster))
        await asyncio.gather(*tasks)

    def assign_differentiated_cluster_labels(self):
        for cluster in self.clusters.values():
            if type(cluster.cluster_id) == int and cluster.cluster_id < 0:
                cluster.cluster_label = "Noise"
                cluster.cluster_description = "Noise"
                continue
            other_cluster_descriptions = [other_cluster.cluster_description for other_cluster in self.clusters.values()
                                          if other_cluster.cluster_id != cluster.cluster_id]
            cluster_label = self.llm_model.simple_generate(
                self.get_cluster_labelling_prompt(cluster.cluster_description, other_cluster_descriptions))
            cluster.cluster_label = cluster_label

    async def async_assign_differentiated_cluster_labels(self):
        async def async_get_cluster_label_helper(cluster):
            if type(cluster.cluster_id) == int and cluster.cluster_id < 0:
                cluster.cluster_label = "Noise"
                cluster.cluster_description = "Noise"
                return
            other_cluster_descriptions = [other_cluster.cluster_description for other_cluster in self.clusters.values()
                                          if other_cluster.cluster_id != cluster.cluster_id]
            cluster_label = await self.llm_model.async_simple_generate(
                self.get_cluster_labelling_prompt(cluster.cluster_description, other_cluster_descriptions))
            cluster.cluster_label = cluster_label

        tasks = []
        for cluster in self.clusters.values():
            tasks.append(async_get_cluster_label_helper(cluster))
        await asyncio.gather(*tasks)

    async def async_add_keyphrases_for_cluster(self, cluster, cache_directory=None, custom_llm=None):
        if custom_llm is None:
            custom_llm = self.llm_model
        if hasattr(cluster, "representative_keyphrases") and cluster.representative_keyphrases:
            return
        keyphrase_parser = PydanticOutputParser(pydantic_object=RepresentativeKeyphrasesListLLMOutput)
        keyphrase_extraction_prompt_template = PromptTemplate(template=KEYPHRASE_EXTRACTION_PROMPT_TEMPLATE,
                                                              input_variables=['representative_points_text', 'intent'],
                                                              partial_variables={
                                                                  'format_instructions':
                                                                      keyphrase_parser.get_format_instructions()})
        representative_points = cluster.get_representative_points()
        representative_points_lst = [point.processed_text for point in representative_points]
        try:
            model_output = await (custom_llm.async_simple_generate(
                keyphrase_extraction_prompt_template.format(
                    representative_points_text="\n".join(representative_points_lst),
                    intent=(cluster.cluster_label + " : " + cluster.cluster_description))))
            parsed_keyphrases_output = keyphrase_parser.parse(model_output)
            print("For cluster {} : {}".format(cluster.cluster_label, parsed_keyphrases_output))
            cluster.representative_keyphrases = [str(i.keyphrase) for i in parsed_keyphrases_output.keyphrases]
            if cache_directory is not None:
                await Serializer.async_serialize_to_disk(self, os.path.join(cache_directory, "call_driver_cluster_pulse.pkl"))
        except Exception as e:
            print(e)
            cluster.representative_keyphrases = []

    def assign_differentiated_cluster_labels_v2(self):
        all_cluster_descriptions_lst = [cluster.cluster_description for cluster in self.clusters.values()]
        cluster_label_parser = PydanticOutputParser(pydantic_object=ClusterLabelsListLLMOutput)
        cluster_label_extraction_prompt_template = PromptTemplate(
            template=CALL_DRIVER_CLUSTER_LABELING_PROMPT_TEMPLATE_V2,
            input_variables=['cluster_descriptions'],
            partial_variables={
                'format_instructions':
                    cluster_label_parser.get_format_instructions()})
        try:
            model_output = (self.llm_model.simple_generate(
                cluster_label_extraction_prompt_template.format(
                    cluster_descriptions="\n".join(all_cluster_descriptions_lst))))
            parsed_cluster_labels_output = cluster_label_parser.parse(model_output)
            for cluster, cluster_label in zip(self.clusters.values(), parsed_cluster_labels_output.cluster_labels):
                cluster.cluster_label = cluster_label.cluster_label
        except Exception as e:
            print("Error in cluster label extraction: {}".format(e))

    async def async_assign_differentiated_cluster_labels_v2(self):
        all_cluster_descriptions_lst = [cluster.cluster_description for cluster in self.clusters.values()]
        cluster_label_parser = PydanticOutputParser(pydantic_object=ClusterLabelsListLLMOutput)
        cluster_label_extraction_prompt_template = PromptTemplate(
            template=CALL_DRIVER_CLUSTER_LABELING_PROMPT_TEMPLATE_V2,
            input_variables=['cluster_descriptions'],
            partial_variables={
                'format_instructions':
                    cluster_label_parser.get_format_instructions()})
        try:
            model_output = await (self.llm_model.async_simple_generate(
                cluster_label_extraction_prompt_template.format(
                    cluster_descriptions="\n".join(all_cluster_descriptions_lst))))
            parsed_cluster_labels_output = cluster_label_parser.parse(model_output)
            for cluster, cluster_label in zip(self.clusters.values(), parsed_cluster_labels_output.cluster_labels):
                cluster.cluster_label = cluster_label.cluster_label
        except Exception as e:
            print("Error in cluster label extraction: {}".format(e))

    def classify_new_embedding(self, new_embedding):
        distances = np.linalg.norm(self._cluster_centroids - new_embedding, axis=1)
        sklearn_cluster_id = np.argmin(distances)
        return self.sklearn_cluster_id_to_pulse_cluster_id[sklearn_cluster_id]

    def predict(self, document):
        self.process_document_with_llm(document)
        self.assign_embedding_to_document(document)
        cluster_id = self.classify_new_embedding(document.embedding)
        document.cluster_id = cluster_id
        relevant_cluster = self.clusters[cluster_id]
        return relevant_cluster

    async def async_predict(self, document):
        await self.async_process_document_with_llm(document)
        await self.async_assign_embedding_to_document(document)
        cluster_id = self.classify_new_embedding(document.embedding)
        document.cluster_id = cluster_id
        relevant_cluster = self.clusters[cluster_id]
        return relevant_cluster

    def get_document_processing_prompt(self, document_text):
        if self.document_processing_prompt is None:
            return None
        return self.document_processing_prompt.format(document_text=document_text)

    def get_cluster_description_prompt(self, representative_points_lst):
        if self.cluster_description_prompt is None:
            return None
        return self.cluster_description_prompt.format(representative_points_text="\n".join(representative_points_lst))

    def get_cluster_labelling_prompt(self, focus_cluster_description, other_cluster_descriptions):
        if self.cluster_labelling_prompt is None:
            return None
        return self.cluster_labelling_prompt.format(focus_cluster_description=focus_cluster_description,
                                                    other_cluster_descriptions="\n".join(other_cluster_descriptions))
