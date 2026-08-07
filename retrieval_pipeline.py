import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv() 

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embedding_model = HuggingFaceEmbeddings(model= "BAAI/bge-small-en-v1.5")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)


query = "What is 1+1?"

retriever = db.as_retriever(search_kwargs={"k":3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

print("-----RESULTS-----")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


#----------------------------------------------------------------

combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using the information from these documents. If you can't find answer in the documents,
say "I don't have enough information provided to answer your query.". Don't say anything after that."""


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]

result = llm.invoke(messages)

print("\n---GENERATED RESPONSE---")

print(result.content)
