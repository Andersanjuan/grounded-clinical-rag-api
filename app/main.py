from fastapi import FastAPI

app = FastAPI(title="MedRAG Clinical Assistant API")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MedRAG API is running"}
