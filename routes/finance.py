from fastapi import APIRouter, HTTPException
from schemas.stock_schemas import StockAnalysisRequest
from services.flow_service import FinancialAnalysisState, app

router = APIRouter()


@router.post("/analyze")
async def analyze_prompt(request: StockAnalysisRequest):
    try:
        # Initialize the state with default values along with the received prompt.
        initial_state: FinancialAnalysisState = {
            "ticker": "",
            "period": "1y",
            "interval": "1d",
            "prompt": request.prompt,
            "stock_data": {},
            "predictions": [],
            "report": ""
        }

        final_state = app.invoke(initial_state)
        return {
            "stockName": final_state['ticker'],
            "predictions": final_state["predictions"],
            "report": final_state['report'],
            "period": final_state['period']
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))