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

def generar_diagnostico(score, factor_principal):
    nombres_legibles = {
        "RevolvingUtilizationOfUnsecuredLines": "alta utilizacion de lineas de credito",
        "NumberOfTimes90DaysLate": "multiples atrasos graves",
        "NumberOfTime30-59DaysPastDueNotWorse": "atrasos leves frecuentes",
        "DebtRatio": "alto ratio de endeudamiento",
        "MonthlyIncome": "nivel de ingresos bajo",
        "age": "perfil de edad del cliente",
        "NumberOfOpenCreditLinesAndLoans": "exceso de creditos abiertos",
        "NumberOfDependents": "numero de dependientes",
        "NumberRealEstateLoansOrLines": "creditos hipotecarios",
        "NumberOfTime60-89DaysPastDueNotWorse": "atrasos moderados recurrentes"
    }

    factor_texto = nombres_legibles.get(factor_principal, factor_principal)

    if score >= 60:
        return (
            "Este cliente presenta senales de alto riesgo de mora. "
            "El principal factor de alerta es " + factor_texto + ". "
            "Con un score de " + str(score) + "/100, se recomienda activar protocolo "
            "de seguimiento preventivo de forma inmediata."
        )
    elif score >= 30:
        return (
            "Este cliente muestra senales moderadas de riesgo financiero. "
            "El factor que mas contribuye al riesgo es " + factor_texto + ". "
            "Con un score de " + str(score) + "/100, se sugiere monitoreo periodico."
        )
    else:
        return (
            "Este cliente presenta un perfil financiero saludable. "
            "El score de " + str(score) + "/100 indica baja probabilidad de mora. "
            "Se recomienda mantener las condiciones actuales del credito."
        )

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
    score = round(float(prob_mora) * 100, 1)

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

    factores_lista = []
    for i in range(len(feature_names)):
        factores_lista.append({
            'variable': str(feature_names[i]),
            'impacto': float(shap_vals[i])
        })

    factores_lista.sort(key=lambda x: abs(x['impacto']), reverse=True)
    top3 = factores_lista[:3]

    factor_principal = top3[0]['variable']
    diagnostico = generar_diagnostico(score, factor_principal)

    return {
        "score_mora": score,
        "nivel": nivel,
        "color": color,
        "probabilidad": round(float(prob_mora), 4),
        "top_factores": top3,
        "diagnostico": diagnostico
    }
