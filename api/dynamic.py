# api/dynamic.py

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from schemas import DynamicRequest, DynamicResponse
from dynamic_services import process_dynamic_document # <-- Updated import

router = APIRouter()
security = HTTPBearer()

def check_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    valid_api_key = "b6576e2fa82c4dd0cb99fcfb82c75dcc2597d031c433c743ae42c7a5f47fb03b"
    
    if credentials.scheme != "Bearer" or credentials.credentials != valid_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )
    return credentials.credentials


@router.post("/run", response_model=DynamicResponse)
async def run_dynamic_processing(
    request: DynamicRequest,
    api_key: str = Security(check_api_key)
):
  
    try:
        answers = await process_dynamic_document(
            document_url=str(request.documents),
            questions=request.questions
        )
        return DynamicResponse(answers=answers)
    except Exception as e:
        print(f"An unexpected error occurred in the dynamic endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during document processing.")
