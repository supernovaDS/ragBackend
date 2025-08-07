

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.policy import router as policy_router
from api.law import router as law_router  
from api.dynamic import router as dynamic_router
from services import initialize_vector_stores 


app = FastAPI(
    title="Multi-Domain Analyzer API",
    description="An API to query Insurance and Indian Law documents.",
    version="0.0.1" 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://meow.com"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    initialize_vector_stores()

app.include_router(policy_router, prefix="/insurance", tags=["Insurance Policy"])
app.include_router(law_router, prefix="/law", tags=["Indian Law"])
app.include_router(dynamic_router, prefix="/hackrx", tags=["Dynamic Document Processing"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Domain Analyzer API."}
