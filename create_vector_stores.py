
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from services import get_pdf_text_from_directory, get_text_chunks
from config import GOOGLE_API_KEY

def main():
    print("Starting vector store creation process...")
    

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found. Please check your .env file.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # insurcnace doc processing
    insurance_docs_path = "documents/"
    insurance_index_path = "faiss_index_insurance" 
    
    print(f"\nProcessing insurance documents from: {insurance_docs_path}")
    insurance_raw_text = get_pdf_text_from_directory(insurance_docs_path)
    if insurance_raw_text.strip():
        insurance_chunks = get_text_chunks(insurance_raw_text)
        insurance_vector_store = FAISS.from_texts(insurance_chunks, embedding=embeddings)
        insurance_vector_store.save_local(insurance_index_path)
        print(f"Successfully created and saved insurance vector store to: {insurance_index_path}")
    else:
        print("No insurance documents found to process.")

    # law doc processing
    law_docs_path = "documents_law/"
    law_index_path = "faiss_index_law" 

    print(f"\nProcessing law documents from: {law_docs_path}")
    law_raw_text = get_pdf_text_from_directory(law_docs_path)
    if law_raw_text.strip():
        law_chunks = get_text_chunks(law_raw_text)
        law_vector_store = FAISS.from_texts(law_chunks, embedding=embeddings)
        law_vector_store.save_local(law_index_path)
        print(f"Successfully created and saved law vector store to: {law_index_path}")
    else:
        print("No law documents found to process.")

    print("\nVector store creation complete.")


if __name__ == "__main__":
    main()
