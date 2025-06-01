from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# Dummy classifier & masker for demonstration
def dummy_classify(email_body: str) -> str:
    if "refund" in email_body.lower():
        return "billing"
    elif "password" in email_body.lower():
        return "technical"
    else:
        return "general"

def dummy_masker(email_body: str) -> (str, List[Dict[str, Any]]):
    # Dummy PII masking: Replace "test" with "***"
    masked_email = email_body.replace("test", "***")
    entities = []
    index = email_body.lower().find("test")
    if index != -1:
        entities.append({
            "position": [index, index + len("test")],
            "classification": "name",
            "entity": "test"
        })
    return masked_email, entities

app = FastAPI()

class EmailRequest(BaseModel):
    input_email_body: str

class Entity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[Entity]
    masked_email: str
    category_of_the_email: str

@app.post("/classify", response_model=EmailResponse)
def classify_email(request: EmailRequest):
    email_body = request.input_email_body
    category = dummy_classify(email_body)
    masked_email, entities = dummy_masker(email_body)

    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }
