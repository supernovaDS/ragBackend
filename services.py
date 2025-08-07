

import os
import asyncio
import re
from typing import List
from io import BytesIO
import requests

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from config import GOOGLE_API_KEY, PDF_DOCUMENTS_PATH
from schemas import (
    PolicyDecision, LawDecision, SimplifiedPolicyResponse, 
    SimplifiedLawResponse, BatchAnswers
)


insurance_vector_store = None
law_vector_store = None


def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

def get_pdf_text_from_directory(directory_path: str) -> str:
    text = ""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}. Please add your document files there.")
        return ""
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Warning: No PDF files found in the '{directory_path}' directory.")
        return ""
    for pdf_file in pdf_files:
        path = os.path.join(directory_path, pdf_file)
        print(f"Processing PDF: {pdf_file}")
        with open(path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    return text

def get_text_chunks(text: str) -> List[str]:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def initialize_vector_stores():
 
    global insurance_vector_store, law_vector_store
    
    print("Loading vector stores from disk...")
    ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    insurance_index_path = "faiss_index_insurance"
    if os.path.exists(insurance_index_path):
        insurance_vector_store = FAISS.load_local(insurance_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Insurance Vector Store loaded successfully.")
    else:
        print(f"Warning: Insurance index not found at '{insurance_index_path}'. Please run create_vector_stores.py.")

    law_index_path = "faiss_index_law"
    if os.path.exists(law_index_path):
        law_vector_store = FAISS.load_local(law_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Law Vector Store loaded successfully.")
    else:
        print(f"Warning: Law index not found at '{law_index_path}'. Please run create_vector_stores.py.")


async def _get_simplified_explanation(decision_object: BaseModel, context: str) -> str:
   
    #print(f"Generating simplified explanation for {context} context...")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, google_api_key=GOOGLE_API_KEY)
    
    prompt_text = ""
    if context == "insurance":
        prompt_text = """
        You are an AI assistant that explains complex insurance decisions in simple terms.
        Based on the following structured decision, write a clear, concise, and easy-to-understand explanation for the end-user.
        Do not just repeat the information. Explain *why* the decision was made by referencing the justification, and clarify what it means for the user.

        Structured Decision:
        {decision_json}

        Simplified Explanation for the User:
        """
    elif context == "law":
        prompt_text = """
        You are an AI assistant that explains complex legal topics in simple terms for a non-lawyer.
        Based on the following structured legal analysis, write a clear and easy-to-understand explanation.
        Explain the key takeaways from the relevant clauses and what the suggested next steps mean in practical terms. **Do not give legal advice.**

        Structured Analysis:
        {decision_json}

        Simplified Explanation for the User:
        """
    else:
        return "No explanation available."

    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | model
    
    response = await chain.ainvoke({"decision_json": decision_object.model_dump_json(indent=2)})
    explanation = response.content.strip()
    #print("Simplified explanation generated.")
    return explanation



async def _expand_insurance_query(user_question: str) -> str:
  
    #print(f"Original insurance query: '{user_question}'")
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
    
    prompt_template = """
    You are an expert insurance claims assistant. Your task is to take a user's simple, plain-English query or statement and expand it into a more detailed and precise search query suitable for an insurance policy database. 
    Focus on identifying the key subjects (e.g., age, gender, health history, desired coverage) and rephrase the query to be unambiguous for semantic search.
    Only return the expanded query text, with no preamble.

    Original Query: '{user_question}'

    Expanded Insurance Search Query:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    chain = prompt | model
    
    response = await chain.ainvoke({"user_question": user_question})
    expanded_query = response.content.strip()
    
    #print(f"Expanded insurance query: '{expanded_query}'")
    return expanded_query

def get_insurance_qa_chain() -> Runnable:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY, model_kwargs={"response_format": {"type": "json_object"}})
    parser = JsonOutputParser(pydantic_object=PolicyDecision)
    

    prompt_template = """
    You are an expert insurance policy analysis agent. Your task is to analyze the user's query 
    against the provided insurance policy clauses from the context.
    Based *only* on these clauses, make a final decision. Do not use any external knowledge.
    Return a structured JSON response with the following format:
    {json_format}

    Context (Policy Clauses):
    {context}
    User's Question:
    {question}
    JSON Response:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={"json_format": parser.get_format_instructions()}
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

async def get_policy_decision(user_question: str) -> SimplifiedPolicyResponse:
    if insurance_vector_store is None:
        raise ConnectionError("Insurance vector store is not available.")
        
    expanded_query = await _expand_insurance_query(user_question)
    docs = insurance_vector_store.similarity_search(expanded_query, k=5)
    
    if not docs:
        structured_decision = PolicyDecision(
            decision="Not Found",
            amount=None,
            justification=["No relevant clauses found in the provided documents for your query."]
        )
    else:
        chain = get_insurance_qa_chain()
        response = await chain.ainvoke({"input_documents": docs, "question": user_question})
        output_text = response.get('output_text', '{}')
        json_match = re.search(r'```(json)?(.*)```', output_text, re.DOTALL)
        clean_json_str = json_match.group(2).strip() if json_match else output_text.strip()
        structured_decision = PolicyDecision.model_validate_json(clean_json_str)

    simplified_explanation = await _get_simplified_explanation(structured_decision, "insurance")
    
    return SimplifiedPolicyResponse(
        structured_decision=structured_decision,
        simplified_explanation=simplified_explanation
    )


async def _expand_legal_query(user_question: str) -> str:
  
    print(f"Original legal query: '{user_question}'")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)
    
    prompt_template = """
    You are an expert legal assistant. Your task is to take a user's simple, plain-English query and expand it into a more detailed and precise search query suitable for a legal database. 
    Focus on identifying the key subjects (e.g., persons, animals, property) and actions (e.g., theft, injury, defamation) and rephrase the query to be unambiguous for semantic search.
    Only return the expanded query text, with no preamble.

    Original Query: '{user_question}'

    Expanded Legal Search Query:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    chain = prompt | model
    
    response = await chain.ainvoke({"user_question": user_question})
    expanded_query = response.content.strip()
    
    print(f"Expanded legal query: '{expanded_query}'")
    return expanded_query

def get_law_qa_chain() -> Runnable:
 
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY, model_kwargs={"response_format": {"type": "json_object"}})
    parser = JsonOutputParser(pydantic_object=LawDecision)
    

    prompt_template = """
    You are an AI assistant specialized in Indian Law. Your task is to analyze a user's query 
    against the provided legal documents (like the Indian Penal Code, Constitution of India, etc.).
    
    1.  Identify the most relevant legal code or act from the context.
    2.  Extract the key sections, articles, or clauses that apply to the query.
    3.  Provide a clear, neutral summary of what the law states regarding the query.
    4.  Suggest general next steps. **You are not a lawyer. Do not provide legal advice.** Your primary next step should always be to recommend consulting a qualified legal professional.

    Return a structured JSON response in the following format:
    {json_format}

    Context (Legal Clauses):
    {context}
    User's Question:
    {question}
    JSON Response:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={"json_format": parser.get_format_instructions()}
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

async def get_law_decision(user_question: str) -> SimplifiedLawResponse:

    if law_vector_store is None:
        raise ConnectionError("Legal vector store is not available. Check server logs and ensure documents are in 'documents_law/'.")

    expanded_query = await _expand_legal_query(user_question)
    docs = law_vector_store.similarity_search(expanded_query, k=7)
    
    if not docs:
        structured_decision = LawDecision(
            relevant_law="Not Found",
            key_clauses=[],
            summary="No relevant clauses or articles could be found in the provided legal documents for your query.",
            next_steps=["Please rephrase your query or consult a legal professional."]
        )
    else:
        chain = get_law_qa_chain()
        response = await chain.ainvoke({"input_documents": docs, "question": user_question})
        output_text = response.get('output_text', '{}')
        json_match = re.search(r'```(json)?(.*)```', output_text, re.DOTALL)
        clean_json_str = json_match.group(2).strip() if json_match else output_text.strip()
        structured_decision = LawDecision.model_validate_json(clean_json_str)

    simplified_explanation = await _get_simplified_explanation(structured_decision, "law")

    return SimplifiedLawResponse(
        structured_decision=structured_decision,
        simplified_explanation=simplified_explanation
    )

#general

def _get_text_from_url(pdf_url: str) -> str:
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
 
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0, 
        google_api_key=GOOGLE_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    parser = JsonOutputParser(pydantic_object=BatchAnswers)

    prompt_template = """
    You are an AI assistant. Your task is to answer all of the questions in the provided list based *only* on the given text context.
    Provide a direct answer for each question. If the answer is not in the context, state that the information is not available in the document.
    Return your response as a single JSON object with a key "answers" that contains a list of strings. Each string in the list should be the answer to the corresponding question in the input list.

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
