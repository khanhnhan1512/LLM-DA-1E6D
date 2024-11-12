
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAIEmbeddings(api_key=API_KEY, model="text-embedding-3-large")
l1 = ["I am a teacher", "I am a student"]
l2 = ["I am a pupil", "I am a student", "I am a doctor"]
l1_emb = llm.embed_documents(l1)
l2_emb = llm.embed_documents(l2)
similarity_matrix = cosine_similarity(l1_emb, l2_emb)
print(similarity_matrix)

