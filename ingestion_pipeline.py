import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



def load_documents(docs_path = "docs"):
    print(f"Loading all documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",  #just reads text files
        loader_cls = TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents)==0:
        raise FileNotFoundError(f"No txt files found in {docs_path}. Please add your company files.")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    print("Creating embeddings and storing in chroma DB")

    embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )

    print("--creating vector store--")
    vectorstore = Chroma.from_documents(documents = chunks, embedding = embedding_model, persist_directory=persist_directory,
                                        collection_metadata={"hnsw:space":"cosine"})
    print("finished creating vectorstore")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    documents = load_documents(docs_path = "docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)



if __name__=="__main__":
    main()



