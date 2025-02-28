import fastapi
import joblib
import pandas as pd
from fastapi import Request
from fastapi.responses import JSONResponse
from google.cloud.error_reporting import Client as ErrClient
from sklearn.linear_model import LogisticRegression

from .model import top_10_features


err_client = ErrClient()


class MonthError(Exception):
    message = "Month must be between 1 and 12"

    def __init__(self) -> None:
        super().__init__(self.message)


class TypeFlightError(Exception):
    message = "Type of flight must be 'I' or 'N'"

    def __init__(self) -> None:
        super().__init__(self.message)


class AirlineError(Exception):
    message = "Airline must be 'Latin American Wings', 'Grupo LATAM', 'Sky Airline' or 'Copa Air'"

    def __init__(self) -> None:
        super().__init__(self.message)


def transform_opera(opera: str, compare: str) -> int:
    if opera not in ["Latin American Wings", "Grupo LATAM", "Sky Airline", "Copa Air"]:
        raise AirlineError
    return 1 if opera == compare else 0


def transform_mes(mes: int, compare: int) -> int:
    if mes > 12:
        raise MonthError
    return 1 if mes == compare else 0


def transform_tipovuelo(tipovuelo: str, compare: str) -> int:
    if tipovuelo not in ["I", "N"]:
        raise TypeFlightError
    return 1 if tipovuelo == compare else 0


app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(req: Request) -> JSONResponse:
    flights = await req.json()

    if not flights or "flights" not in flights or not flights["flights"]:
        err_client.report_exception()
        return JSONResponse(
            content={"error": "Payload is empty or 'flights' key is missing"}, status_code=400
        )

    try:
        features_list = []
        for flight in flights["flights"]:
            features = {
                "OPERA_Latin American Wings": transform_opera(
                    flight["OPERA"], "Latin American Wings"
                ),
                "MES_7": transform_mes(flight["MES"], 7),
                "MES_10": transform_mes(flight["MES"], 10),
                "OPERA_Grupo LATAM": transform_opera(flight["OPERA"], "Grupo LATAM"),
                "MES_12": transform_mes(flight["MES"], 12),
                "TIPOVUELO_I": transform_tipovuelo(flight["TIPOVUELO"], "I"),
                "MES_4": transform_mes(flight["MES"], 4),
                "MES_11": transform_mes(flight["MES"], 11),
                "OPERA_Sky Airline": transform_opera(flight["OPERA"], "Sky Airline"),
                "OPERA_Copa Air": transform_opera(flight["OPERA"], "Copa Air"),
            }
            features_list.append(features)

        features_df = pd.DataFrame(features_list, columns=top_10_features)

    except MonthError:
        err_client.report_exception()
        return JSONResponse(content={"error": MonthError.message}, status_code=400)
    except TypeFlightError:
        err_client.report_exception()
        return JSONResponse(content={"error": TypeFlightError.message}, status_code=400)
    except AirlineError:
        err_client.report_exception()
        return JSONResponse(content={"error": AirlineError.message}, status_code=400)

    model: LogisticRegression = joblib.load("data/reg_model.pkl")
    return JSONResponse(
        content={"predict": model.predict(features_df.values).tolist()},
        status_code=200,
    )
