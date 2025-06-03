# chatbot.py

import os
import warnings
import sys
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore", category=FutureWarning, module='langchain_community.vectorstores.faiss')
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*HuggingFaceEmbeddings.*")

DATA_PATH = "Mahabharata_Gita_Light_Edition.txt"  
VECTORSTORE_PATH = "faiss_index_gemma_local" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
OLLAMA_MODEL_NAME = "mistral:7b-instruct-v0.2-q4_K_M"
INDEX_BATCH_SIZE = 500 

def load_or_create_vectorstore(data_path, vectorstore_path, embedding_model, batch_size):
    """Loads an existing FAISS index or creates a new one from the text file using batch processing."""
    if os.path.exists(vectorstore_path):
        print(f"Loading existing vector index from '{vectorstore_path}'...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            print("Vector index loaded successfully!")
            return vectorstore
        except Exception as e:
            print(f"ERROR loading index: {e}. Attempting to recreate it.")

    print(f"Creating new vector index from '{data_path}'...")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file '{data_path}' not found!")
        return None

    try:
        loader = TextLoader(data_path, encoding="utf-8")
        documents = loader.load()
        print(f"{len(documents)} document(s) loaded from the file.")
        if not documents:
            print("ERROR: No documents loaded. Is the file empty?")
            return None

        # Chunk-Größe reduziert von 1000 auf 800, Overlap von 150 auf 100
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        total_docs = len(docs)
        print(f"{total_docs} text chunks created.")
        if not docs:
            print("ERROR: No chunks created after splitting.")
            return None

        print(f"Creating embeddings using '{embedding_model}' (this will take a while)...")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        print(f"Processing {total_docs} chunks in batches of {batch_size}...")
        print("Initializing FAISS index with the first batch...")
        first_batch_docs = docs[:batch_size]
        if not first_batch_docs:
            print("ERROR: No documents available for the first batch!")
            return None

        vectorstore = FAISS.from_documents(first_batch_docs, embeddings)
        total_batches = (total_docs + batch_size - 1) // batch_size
        print(f"Processed Batch 1/{total_batches}.")

        for i in range(batch_size, total_docs, batch_size):
            current_batch_num = (i // batch_size) + 1
            print(f"Processing Batch {current_batch_num}/{total_batches} (Chunks {i} to {min(i + batch_size, total_docs)})...")

            current_batch_docs = docs[i:min(i + batch_size, total_docs)]
            if not current_batch_docs:
                continue

            vectorstore.add_documents(current_batch_docs)

        print("All batches processed. Embeddings created and FAISS index built.")

        vectorstore.save_local(vectorstore_path)
        print(f"Vector index created successfully and saved in '{vectorstore_path}'!")
        return vectorstore

    except Exception as e:
        print(f"An unexpected ERROR occurred during index creation: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_rag_chain(vectorstore, ollama_model_name):
    """Initializes Ollama and creates the RetrievalQA Chain."""
    print(f"Initializing Ollama with model: {ollama_model_name}...")
    print("Make sure the Ollama service is running!")
    try:
        llm = Ollama(model=ollama_model_name)
        print(f"Ollama LLM '{ollama_model_name}' initialized successfully.")
    except Exception as e:
        print(f"\nERROR: Could not initialize Ollama LLM '{ollama_model_name}'.")
        print("Possible reasons:")
        print("- Is the Ollama service running? (Start it or use 'ollama serve')")
        print(f"- Is the model '{ollama_model_name}' downloaded correctly? ('ollama list')")
        print(f"Error message: {e}")
        return None

    print("Creating the Prompt Template...")
    prompt_template = """<|start_of_turn|>user
You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say "I cannot answer this question based on the provided context.". Do not make things up.
Always respond in English.

Context:
{context}

Question: {question}<|end_of_turn|>
<|start_of_turn|>model
Answer in English:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    print("Creating the Retriever...")
    # k-Wert reduziert von 5 auf 3
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Creating the RetrievalQA Chain...")
    # Wieder zurück zu "stuff", aber k schon reduziert → schneller als vorher
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("RAG Chain created successfully!")
    return qa_chain

if __name__ == "__main__":
    print("Starting the RAG Chatbot...")

    vector_store = load_or_create_vectorstore(DATA_PATH, VECTORSTORE_PATH, EMBEDDING_MODEL_NAME, INDEX_BATCH_SIZE)

    if vector_store:
        try:
            print(f"Number of vectors in the store: {vector_store.index.ntotal}")
        except AttributeError:
            print("Could not retrieve exact vector count from loaded index.")

        rag_chain = create_rag_chain(vector_store, OLLAMA_MODEL_NAME)

        if rag_chain:
            print("\nChatbot is ready! Ask your questions.")
            print("Type 'quit' or 'exit' to stop the chatbot.")

            while True:
                user_question = input("\nYour question: ")

                if user_question.lower() in ["quit", "exit"]:
                    print("Exiting chatbot. Goodbye!")
                    break

                if not user_question.strip():
                    continue

                print("Thinking...")
                try:
                    result = rag_chain.invoke({"query": user_question})

                    print("\n--- Answer ---")
                    print(result["result"])
                    print("-" * 15)

                    show_sources = True
                    if show_sources and result.get("source_documents"):
                        print("\n--- Sources Used (Excerpts) ---")
                        for i, doc in enumerate(result["source_documents"]):
                            page_content_oneline = " ".join(doc.page_content.splitlines())
                            print(f"Source {i+1}: '{page_content_oneline[:300]}...'")
                        print("-" * 15)

                except Exception as e:
                    print(f"\nERROR processing question: {e}")
                    print("Please try again, or restart the chatbot if the issue persists.")
        else:
            print("Chatbot could not be initialized (RAG Chain creation failed).")
    else:
        print("Chatbot could not be initialized (Vector Store creation/loading failed).")
