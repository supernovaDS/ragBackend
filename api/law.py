

from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, SimplifiedLawResponse 
from services import get_law_decision


router = APIRouter()

@router.post("/process_law_query", response_model=SimplifiedLawResponse) 
async def process_law_query(request: QueryRequest):
    try:
        decision = await get_law_decision(request.question)
        return decision
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Error processing the law query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request: {e}")
