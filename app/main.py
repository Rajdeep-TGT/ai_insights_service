from fastapi import FastAPI
from pydantic import BaseModel
from app.ml_logic import analyze_receipt
from app.summary import get_summary

app = FastAPI()

class Receipt(BaseModel):
    tenantId: str
    receiptId: str
    amount: float
    date: str
    vendor: str
    category: str
    items: list[str]

@app.post("/analyze")
def analyze(receipt: Receipt):
    try:
        result = analyze_receipt(receipt)
        return result
    except Exception as e:
        print("🔥 ERROR in /analyze:", e)
        return {"error": str(e)}

@app.get("/summary/{tenant_id}")
def summary(tenant_id: str):
    return get_summary(tenant_id)
