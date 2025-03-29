from pydantic import BaseModel

class StockAnalysisRequest(BaseModel):
    prompt: str

class StockAnalysisResponse(BaseModel):
    ticker: str
    period: str
    interval: str
    summary: dict
    graph_data: dict