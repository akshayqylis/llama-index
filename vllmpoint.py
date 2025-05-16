from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

def get_vllm_llm():
	llm = OpenAILike(
		model="meta-llama/Llama-3.1-8B-Instruct",
		api_base="http://g3.ai.qylis.com:8000/v1",
		api_key="fake",
		context_window=128000,
		is_chat_model=True,
		is_function_calling_model=True,
	)
	return llm

def get_vllm_embedder():
	embedding = OpenAILikeEmbedding(
		model_name="BAAI/bge-m3",
		api_base="http://g3.ai.qylis.com:7000/v1",
		api_key="fake",
		embed_batch_size=10,
	)
	return embedding
