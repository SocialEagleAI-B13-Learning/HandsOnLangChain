from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

loader = TextLoader("sample.txt")
documents = loader.load()

print("Number of documents:", len(documents))
print("Content:")
print(documents[0].page_content)
print("Total Characters:", len(documents[0].page_content))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,      # each chunk is about 80 characters
    chunk_overlap=10    # 10 characters repeat between chunks
)

chunks = splitter.split_text(documents[0].page_content)

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)


documents = TextLoader("sample.txt").load()
chunks1 = splitter.split_documents(documents)
print("Number of chunks1:", len(chunks1))



# A small, free, popular embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Embedding model is ready!")

vector = embeddings.embed_query(documents[0].page_content)
print("How many numbers in this vector?", len(vector))
print("First 5 numbers:", vector[:5])

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# A small, cheap, high-quality OpenAI embedding model
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key = api_key)

print("OpenAI embedding model is ready!")


vector = openai_embeddings.embed_query(documents[0].page_content)
print("How many numbers in this vector-OpeAPI?", len(vector))
print("First 5 numbers-OpenAPI:", vector[:5])