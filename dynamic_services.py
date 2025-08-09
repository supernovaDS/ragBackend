import asyncio
from typing import List
from io import BytesIO
import requests
import numpy as np
import faiss

from pypdfium2 import PdfDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

from config import GOOGLE_API_KEY
from schemas import BatchAnswers


def _get_text_from_url(pdf_url: str) -> str:
    print(f"Downloading document from: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        pdf_reader = PdfDocument(pdf_file)
        text = ""
        for page in pdf_reader:
            text += page.get_textpage().get_text_range() or ""
        print("Successfully extracted text from downloaded PDF.")
        return text
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download document: {e}")
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}")


def get_batch_qa_chain() -> Runnable:
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    parser = JsonOutputParser(pydantic_object=BatchAnswers)

    prompt_template = """
    You are an AI assistant that functions as a precise information extractor.
    Your task is to answer each question from the list by finding and quoting the **exact sentence or phrase** from the provided text context that contains the answer.
    Do not paraphrase, summarize, or generate new sentences.
    If a direct quote answering the question cannot be found in the context, you must state: "A direct answer was not found in the provided document."

    Return your response as a single JSON object with a key "answers" that contains a list of strings.

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


def get_text_chunks(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

async def process_dynamic_document(document_url: str, questions: List[str]) -> List[str]:
    try:
        raw_text = _get_text_from_url(document_url)
        if not raw_text.strip():
            return ["The document is empty or no text could be extracted."] * len(questions)
    except (ConnectionError, ValueError) as e:
        return [str(e)] * len(questions)

    print("Creating in-memory vector store...")
    text_chunks = get_text_chunks(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=GOOGLE_API_KEY
    )

    #Embed chunks
    chunk_embeddings = embeddings.embed_documents(text_chunks)
    faiss_index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
    faiss_index.add(np.array(chunk_embeddings, dtype="float32"))
    print("In-memory vector store created.")

    #Embed all questions at once
    question_embeddings = embeddings.embed_documents(questions)

    print("Performing global FAISS search...")
    k = 5
    #Search for each question
    all_indices = []
    for q_emb in question_embeddings:
        _, idx = faiss_index.search(np.array([q_emb], dtype="float32"), k)
        all_indices.extend(idx[0])

    #Remove duplicates, keep order
    unique_indices = list(dict.fromkeys(all_indices))
    combined_context = "\n\n".join(text_chunks[i] for i in unique_indices if i < len(text_chunks))

    print("Answering all questions in one batch...")
    qa_chain = get_batch_qa_chain()
    resp = await qa_chain.ainvoke({
        "context": combined_context,
        "questions": questions
    })

    #Ensure output matches question order
    return resp.get("answers", ["Failed to generate an answer."] * len(questions))
