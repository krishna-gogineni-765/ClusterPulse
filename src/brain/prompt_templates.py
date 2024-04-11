from typing import List

from pydantic import BaseModel, Field

DEFAULT_DOCUMENT_PROCESSING_PROMPT_TEMPLATE = "I want you to act as a concise document summarizer. \
I will input text from a certain document and you are expected to summarize it. \
The following is the text from the document: \n \
{document_text}\n \
Summarize the above text into 2-3 sentences while trying to preserve the key events/points in the text"

DEFAULT_CLUSTER_LABELING_PROMPT_TEMPLATE = "I have clustered some data and have descriptions that represent what each \
cluster is about. We will focus on one such cluster, I will give you it's description and you are expected to give \
a concise name (less than 5 words) that captures the essence of the focus cluster. Here is the description: \n\
{focus_cluster_description}\n \
I am also sharing the descriptions of other clusters so that you can compare and differentiate the above cluster using \
it's specific nuances if need be. Here are the descriptions of other clusters: \n\
{other_cluster_descriptions}\n \
Based on these descriptions, what would be an appropriate name for the focus cluster?"

DEFAULT_CLUSTER_DESCRIPTION_PROMPT_TEMPLATE = "I have clustered data and picked few representative points \
from a particular cluster. Given these points, provide a brief description, no longer than two sentences, \
that captures the main theme or essence of this cluster. Here is the text from the points: \n\
{representative_points_text}\n \
What concise description can best capture the essence of this cluster?"

CALL_DRIVER_CLUSTER_LABELING_PROMPT_TEMPLATE_V2 = "I have clustered some data and have \
descriptions for each cluster. Each description represents the description behind the cluster. \
I will give you the descriptions of all the clusters and you are expected to give each of the clusters a concise \
name (less than 5 words) that captures the essence of the cluster. Here are the descriptions of all the clusters: \n\
{cluster_descriptions}\n \
Based on these descriptions, what would be an appropriate name for each of the clusters? Name the clusters \
such that they are distinct from each other using the nuances in the descriptions if need be. \n{format_instructions}"


class ClusterLabelLLMOutput(BaseModel):
    cluster_label: str = Field(description="Name of the cluster")


class ClusterLabelsListLLMOutput(BaseModel):
    cluster_labels: List[ClusterLabelLLMOutput]


class RepresentativeKeyphraseLLMOutput(BaseModel):
    keyphrase: str = Field(description="Keyphrase as is")


class RepresentativeKeyphrasesListLLMOutput(BaseModel):
    keyphrases: List[RepresentativeKeyphraseLLMOutput]


KEYPHRASE_EXTRACTION_PROMPT_TEMPLATE = "I am going to give you a list of call center call ASR transcriptions \
all of which are for the same caller intent. I want you to come up with keyphrases that can be used to accurately \
identify more such calls for this intent from the transcriptions. \n\
Call Transcriptions : \n\
{representative_points_text}\n\
Caller Intent : {intent}\n\
What are some 8-10 keyphrases that can accurately identify more such calls for this intent.\n " \
                                       "The keyphrases must ideally occur as is in the calls? \n{format_instructions}"
