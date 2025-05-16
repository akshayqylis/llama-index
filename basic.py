from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import SummaryIndex
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine

from qazure import get_llm, get_embedder

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = get_llm()
embed_model = get_embedder()

Settings.llm = llm
Settings.embed_model = embed_model

# Simple query by indexing
# TODO: Try multimedia inputs like mp3 and mp4
documents = SimpleDirectoryReader(
    input_files=["feedback.docx"]
).load_data()
feedback_index = VectorStoreIndex.from_documents(documents)

query = "What is the feedback about? Give me names."
query_engine = feedback_index.as_query_engine()
answer = query_engine.query(query)

print(answer.get_formatted_sources())
print("query was:", query)
print("answer was:", answer)

# Summarization
summary_index = SummaryIndex.from_documents(documents)

query_engine = summary_index.as_query_engine(response_mode="tree_summarize", streaming=True)
response = query_engine.query("<summarization_query>")
response.print_response_stream()
print('')

# Text input index for an English essay
documents = SimpleDirectoryReader(
    input_files=["paul_graham_essay.txt"]
).load_data()
essay_index = VectorStoreIndex.from_documents(documents)

# Define query engines and tools
tool1 = QueryEngineTool.from_defaults(
    query_engine=feedback_index.as_query_engine(),
    description="Use this query engine for feedback form.",
    # description="Use this query engine for an English essay.", Switch
)
tool2 = QueryEngineTool.from_defaults(
    query_engine=essay_index.as_query_engine(),
    description="Use this query engine for an English essay.",
    # description="Use this query engine for feedback form.", Switch
)

# Router engine
# [IMP] The description needs to be accurate for the router to function correctly
router_engine = RouterQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2]
)

response = router_engine.query(
    "What is the feedback form about? Give me names."
)
print(response.get_formatted_sources())
print('Response:', response)

# SubQuestion Query Engine
# [IMP] Doesn't really work for interview feedback. Doesn't work on heterogenous sources?
sub_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2]
)

response = sub_query_engine.query(
    'What is the feedback of Sohel\'s interview? Give gist.'
)
print(response.get_formatted_sources())
print('Response:', response)
