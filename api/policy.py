

from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, SimplifiedPolicyResponse 
from services import get_policy_decision


router = APIRouter()

@router.post("/process_query", response_model=SimplifiedPolicyResponse) 
async def process_query(request: QueryRequest):
    try:
        decision = await get_policy_decision(request.question)
        return decision
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"Error processing the query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request: {e}")
