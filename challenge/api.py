import fastapi
import joblib
import pandas as pd
from fastapi import Request
from fastapi.responses import JSONResponse
from sklearn.linear_model import LogisticRegression


app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(req: Request) -> JSONResponse:
    features = pd.DataFrame((await req.json())["flights"])

    model: LogisticRegression = joblib.load("data/model.joblib")
    model.predict(features)

    return JSONResponse(content={"predictions": features.to_dict()})
