from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

#base schema
class PolicyDecision(BaseModel):
    decision: str = Field()
    amount: Optional[float] = Field()
    justification: List[str] = Field()

class LawDecision(BaseModel):
    relevant_law: str = Field()
    key_clauses: List[str] = Field()
    summary: str = Field()
    next_steps: List[str] = Field()

class QueryRequest(BaseModel):
    question: str


#simplified api res
class SimplifiedPolicyResponse(BaseModel):
  
    structured_decision: PolicyDecision
    simplified_explanation: str = Field()

class SimplifiedLawResponse(BaseModel):
    
    structured_decision: LawDecision
    simplified_explanation: str = Field()


class DynamicRequest(BaseModel):
    """Defines the structure for a dynamic document processing request."""
    documents: HttpUrl # Use HttpUrl for automatic URL validation
    questions: List[str]

class DynamicResponse(BaseModel):
    """Defines the structure for a dynamic document processing response."""
    answers: List[str]
