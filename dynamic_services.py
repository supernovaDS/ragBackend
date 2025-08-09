# dynamic_services.py

import asyncio
from typing import List
from io import BytesIO
import requests

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

# Import configurations and schemas
from config import GOOGLE_API_KEY
from schemas import BatchAnswers

def _get_text_from_url(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts its text."""
    print(f"Downloading document from: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        print("Successfully extracted text from downloaded PDF.")
        return text
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download document: {e}")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}")

def get_batch_qa_chain() -> Runnable:
    """Creates a QA chain that extracts exact answers from a document."""
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.0, 
        google_api_key=GOOGLE_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    parser = JsonOutputParser(pydantic_object=BatchAnswers)

    # ** UPDATED PROMPT: This now instructs the model to extract exact text. **
    prompt_template = """
    You are an AI assistant that functions as a precise information extractor. 
    Your task is to answer each question from the list by finding and quoting the **exact sentence or phrase** from the provided text context that contains the answer. 
    Do not paraphrase, summarize, or generate new sentences.
    If a direct quote answering the question cannot be found in the context, you must state: "A direct answer was not found in the provided document."
    
    Return your response as a single JSON object with a key "answers" that contains a list of strings. Each string in the list should be the extracted answer to the corresponding question.

    Context:
    {context}

    Questions:
    {questions}

    JSON Response:
    """
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "questions"],
        partial_variables={"json_format": parser.get_format_instructions()}
    ) | model | parser

async def process_dynamic_document(document_url: str, questions: List[str]) -> List[str]:
    """
    Orchestrates the optimized processing of a document from a URL.
    """
    try:
        raw_text = _get_text_from_url(document_url)
        if not raw_text.strip():
            return ["The document is empty or no text could be extracted."] * len(questions)
    except (ConnectionError, ValueError) as e:
        return [str(e)] * len(questions)

    print("Creating in-memory vector store...")
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print("In-memory vector store created.")

    print("Gathering context for all questions...")
    all_docs = []
    for question in questions:
        all_docs.extend(vector_store.similarity_search(question, k=2))
    
    unique_docs = {doc.page_content for doc in all_docs}
    context_text = "\n\n".join(unique_docs)

    print("Answering all questions in a single batch...")
    qa_chain = get_batch_qa_chain()
    
    formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    
    response_data = await qa_chain.ainvoke({
        "context": context_text, 
        "questions": formatted_questions
    })
    
    print("All questions processed.")
    return response_data.get("answers", ["Failed to generate an answer."] * len(questions))

# Helper function needed by this module
def get_text_chunks(text: str) -> List[str]:
    """Splits the input text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)
