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
  
    documents: HttpUrl 
    questions: List[str]

class DynamicResponse(BaseModel):

    answers: List[str]

class BatchAnswers(BaseModel):
 
    answers: List[str]