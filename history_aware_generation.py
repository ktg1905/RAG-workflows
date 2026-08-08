import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

chat_history=[]

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be searchable and standalone. Just return the rewritten question")
        ] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = llm.invoke(messages)
        search_question = str(result.content).strip()
        print(f"Searching for: {search_question}")

    else:
        search_question = user_question


    retriever = db.as_retriever(search_kwargs={"k":3})
    relevant_docs = retriever.invoke(search_question)

    print(f"Found {len(relevant_docs)} relevant documents: ")

    for i, doc in enumerate(relevant_docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"\nDoc {i}: {preview}...")

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in relevant_docs])}
    Please provide a clear, helpful answer using the information from these documents. If you can't find answer in the documents,
    say "I don't have enough information provided to answer your query.". Don't say anything after that."""

    messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on provided documnets and conversation history"),
    HumanMessage(content=combined_input)
    ]

    result = llm.invoke(messages)
    answer = str(result.content).strip()

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer



def start_chat():
    print("Ask me questions! or type 'quit' to exit")

    while True:
        question = input("\nYour question: ")

        if question.lower()=="quit":
            print("Goodbyee!")
            break

        ask_question(question)


def main():
    start_chat()

if __name__=="__main__":
    main()