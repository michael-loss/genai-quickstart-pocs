import os
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from tqdm import tqdm
from dotenv import load_dotenv
import time
from typing import List
from langchain.docstore.document import Document

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

class PDFVectorizer:
    def __init__(self):
        load_dotenv()
        
        self.opensearch_host = os.getenv('opensearch_host')
        self.vector_index_name = os.getenv('vector_index_name')
        self.vector_field_name = os.getenv('vector_field_name')
        self.profile_name = os.getenv('profile_name')
        
        self.embeddings = BedrockEmbeddings(
            credentials_profile_name=self.profile_name,
            model_id="amazon.titan-embed-text-v1"
        )
        
        
        self.vector_store = None

    def process_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            splits = self.text_splitter.split_documents(documents)
            return splits
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def initialize_vector_store(self):
        self.vector_store = OpenSearchVectorSearch(
            index_name=self.vector_index_name,
            embedding_function=self.embeddings,
            opensearch_url=self.opensearch_host,
            vector_field=self.vector_field_name,
            is_aoss=True,
        )

    def batch_process_documents(self, documents: List[Document], batch_size: int = 100):
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
            batch = documents[i:i + batch_size]
            try:
                self.vector_store.add_documents(batch)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")

    def process_folder(self, folder_path: str):
        all_splits = []
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        
        # Process PDFs
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            file_path = os.path.join(folder_path, pdf_file)
            splits = self.process_pdf(file_path)
            all_splits.extend(splits)
        
        # Initialize vector store if not already done
        if not self.vector_store:
            self.initialize_vector_store()
        
        # Process documents in batches
        self.batch_process_documents(all_splits)
        
        return len(all_splits)

def main():
    # Initialize vectorizer
    vectorizer = PDFVectorizer()
    
    # Process PDFs
    folder_path = "/Users/TheDr/Desktop/position-statements"  # Replace with your folder path
    total_chunks = vectorizer.process_folder(folder_path)
    
    print(f"\nProcessing complete. Total chunks processed: {total_chunks}")
    
    # Optional: Test search
    if vectorizer.vector_store:
        results = vectorizer.vector_store.similarity_search(
            "Your test query here",
            k=3
        )
        print("\nTest Search Results:")
        for doc in results:
            print(f"\nContent: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata}")

if __name__ == "__main__":
    main()
