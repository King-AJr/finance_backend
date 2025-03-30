from fastapi import FastAPI
from routes import finance
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(finance.router, prefix="/api", tags=["api"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Youtube Assistant API"} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
