from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap

with open('modelo_mora.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('explainer_shap.pkl', 'rb') as f:
    explainer = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.options("/predecir")
async def options_predecir():
    return {"message": "OK"}

@app.get("/")
def inicio():
    return {"estado": "MoraScope API funcionando"}

class ClienteData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime3059DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime6089DaysPastDueNotWorse: int
    NumberOfDependents: int

@app.post("/predecir")
def predecir(cliente: ClienteData):
    datos_dict = {
        'RevolvingUtilizationOfUnsecuredLines': cliente.RevolvingUtilizationOfUnsecuredLines,
        'age': cliente.age,
        'NumberOfTime30-59DaysPastDueNotWorse': cliente.NumberOfTime3059DaysPastDueNotWorse,
        'DebtRatio': cliente.DebtRatio,
        'MonthlyIncome': cliente.MonthlyIncome,
        'NumberOfOpenCreditLinesAndLoans': cliente.NumberOfOpenCreditLinesAndLoans,
        'NumberOfTimes90DaysLate': cliente.NumberOfTimes90DaysLate,
        'NumberRealEstateLoansOrLines': cliente.NumberRealEstateLoansOrLines,
        'NumberOfTime60-89DaysPastDueNotWorse': cliente.NumberOfTime6089DaysPastDueNotWorse,
        'NumberOfDependents': cliente.NumberOfDependents
    }
    datos = pd.DataFrame([datos_dict])[feature_names]

    prob_mora = modelo.predict_proba(datos)[0][1]
    score = round(prob_mora * 100, 1)

    if score < 30:
        nivel = "bajo"
        color = "verde"
    elif score < 60:
        nivel = "medio"
        color = "amarillo"
    else:
        nivel = "alto"
        color = "rojo"

    shap_vals = explainer.shap_values(datos)[0]
    factores = pd.DataFrame({
        'variable': feature_names,
        'impacto': shap_vals
    }).sort_values('impacto', key=abs, ascending=False)

    top3 = factores.head(3).to_dict('records')

    factor_principal = top3[0]['variable']
    diagnostico = generar_diagnostico(score, factor_principal, cliente)

    return {
        "score_mora": score,
        "nivel": nivel,
        "color": color,
        "probabilidad": round(prob_mora, 4),
        "top_factores": top3,
        "diagnostico": diagnostico
    }

def generar_diagnostico(score, factor_principal, cliente):
    nombres_legibles = {
        "RevolvingUtilizationOfUnsecuredLines": "alta utilización de líneas de crédito",
        "NumberOfTimes90DaysLate": "múltiples atrasos graves",
        "NumberOfTime30-59DaysPastDueNotWorse": "atrasos leves frecuentes",
        "DebtRatio": "alto ratio de endeudamiento",
        "MonthlyIncome": "nivel de ingresos bajo",
        "age": "perfil de edad del cliente",
        "NumberOfOpenCreditLinesAndLoans": "exceso de créditos abiertos",
        "NumberOfDependents": "número de dependientes",
        "NumberRealEstateLoansOrLines": "créditos hipotecarios",
        "NumberOfTime60-89DaysPastDueNotWorse": "atrasos moderados recurrentes"
    }

    factor_texto = nombres_legibles.get(factor_principal, factor_principal)

    if score >= 60:
        return (
            f"Este cliente presenta señales de alto riesgo de mora. "
            f"El principal factor de alerta es {factor_texto}. "
            f"Con un score de {score}/100, se recomienda activar protocolo "
            f"de seguimiento preventivo de forma inmediata."
        )
    elif score >= 30:
        return (
            f"Este cliente muestra señales moderadas de riesgo financiero. "
            f"El factor que más contribuye al riesgo es {factor_texto}. "
            f"Con un score de {score}/100, se sugiere monitoreo periódico."
        )
    else:
        return (
            f"Este cliente presenta un perfil financiero saludable. "
            f"El score de {score}/100 indica baja probabilidad de mora. "
            f"Se recomienda mantener las condiciones actuales del crédito."
        )
