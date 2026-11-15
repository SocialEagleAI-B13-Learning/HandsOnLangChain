from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Our small knowledge base (a list of sentences)
texts = [
    # Similar sentences - Batting
    "Mithhran scored 100 runs in the match.",
    "The batter made a century during the game.",
    "The player scored a hundred runs in cricket.",

    # Similar sentences - Bowling
    "Senthil took 5 wickets in the match.",
    "The fast bowler dismissed five batsmen.",
    "The bowler claimed five wickets during the game.",

    # Similar sentences - Team victory
    "Our team won the cricket tournament.",
    "The team secured victory in the final match.",

    # Related cricket sentences
    "The cricket match starts at 10 AM.",
    "The captain selected eleven players for the match.",

    # Different topics
    "Python is a programming language.",
    "Dogs are friendly animals.",
    "Cats like to sleep a lot.",
    "The weather is sunny today."
]

# Free, local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Ready!")

# Convert text into vectors
vectors = embeddings.embed_documents(texts)

print("Total sentences:", len(vectors))
print("Vector dimensions:", len(vectors[0]))
print("First vector:", vectors[0][:5])



vector_store = FAISS.from_texts(texts, embeddings)

print("All texts are stored in the vector store!")

query = "Tell me about cricket"

results = vector_store.similarity_search(query, k=2)   # k=2 means top 2 matches

print("Top matches for:", query)
for r in results:
    print("-", r.page_content)

query = "How Many Runs Scored"

results = vector_store.similarity_search(query, k=2)   # k=2 means top 2 matches

print("Runs Scored:", query)
for r in results:
    print("-", r.page_content)

## Step 5: Save and load the store

##You can save the store to disk and load it later, so you don't rebuild it every time.
##```python
vector_store.save_local("my_store")

# load it back later
vector_store = FAISS.load_local("my_store", embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # return top 2 results

print("Retriever is ready!")

question = "Who Scored century"

results = retriever.invoke(question)

print("Question:", question)
print("\nRelevant information found:")
for r in results:
    print("-", r.page_content)

question = "Who has taken Wicket?"

results = retriever.invoke(question)

print("Question:", question)
print("\nRelevant information found:")
for r in results:
    print("-", r.page_content)

# 1. Prompt: a template with a blank {topic} to fill in
prompt = ChatPromptTemplate.from_template("Explain {topic} in one simple sentence.")

# 2. Model: the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")

# 3. Output parser: turn the model's reply into plain text
output_parser = StrOutputParser()

print("All three pieces are ready!")

chain = prompt | model | output_parser

print("Chain is ready!")

answer = chain.invoke({"topic": "Python"})

print(answer)

tip_prompt = ChatPromptTemplate.from_template(
    "Give one short, friendly tip for a beginner learning {skill}."
)

tip_chain = tip_prompt | model | output_parser

print(tip_chain.invoke({"skill": "Python"}))



# Turn LangSmith tracking ON
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"   # <-- paste your key
os.environ["LANGSMITH_PROJECT"] = "Tracing-app"          # runs are grouped under this name
os.environ["OPENAI_API_KEY"] = "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"         # <-- paste your key

print("LangSmith is connected!")