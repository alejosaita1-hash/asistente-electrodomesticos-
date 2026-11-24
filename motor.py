"""Motor de cálculo determinístico: kWh -> pesos. Sin IA: mismo dato, mismo resultado.
E = P x h ; C = E x tarifa ; Ahorro = C_actual - C_recomendado (mostrado como rango)."""
import csv, os
from dataclasses import dataclass, field

DIR = os.path.join(os.path.dirname(__file__), "data")
SEMANAS_MES = 4.33
INCERTIDUMBRE = 0.15  # +-15% sobre el ahorro: supuesto declarado

# Supuestos de uso (editables en la app). Cada uno se muestra al usuario.
FACTOR_COMPRESOR_AA = 0.6   # fracción del tiempo que el compresor trabaja a potencia nominal
HORAS_CICLO_LAVA_FALLBACK = 1.5
FACTOR_LAVA_FALLBACK = 0.5


def cargar_datos():
    d = {}
    with open(os.path.join(DIR, "datos_calculo.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["id"]] = {k: (float(v) if v not in ("", None) and k not in ("id", "fuente") else v)
                          for k, v in r.items()}
    return d


def cargar_fichas():
    with open(os.path.join(DIR, "fichas_productos.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@dataclass
class Resultado:
    kwh_mes: float
    pesos_mes: float
    metodo: str
    supuestos: list = field(default_factory=list)
    confiable: bool = True


def consumo_mensual(pid, uso, tarifa, datos):
    """uso (dict) segun categoria:
    AA: horas_dia, dias_mes, modo ('frio'|'calor')
    HE: (sin parametros)
    LA: ciclos_semana
    MI: minutos_dia, dias_mes
    """
    d = datos[pid]
    cat = pid[:2]
    sup = []
    if cat == "AA":
        p = d["potencia_calor_w"] if uso.get("modo") == "calor" and d.get("potencia_calor_w") else d["potencia_w"]
        h = uso["horas_dia"] * uso["dias_mes"]
        kwh = p / 1000 * h * FACTOR_COMPRESOR_AA
        sup += [f"Potencia nominal {p:.0f} W ({uso.get('modo','frio')})",
                f"El compresor trabaja a potencia nominal el {FACTOR_COMPRESOR_AA:.0%} del tiempo (supuesto)"]
        metodo = "E = P × h × factor de uso"
    elif cat == "HE":
        kwh = d["kwh_anio"] / 12
        sup += [f"Consumo anual de etiqueta: {d['kwh_anio']:.0f} kWh/año ÷ 12 (la heladera funciona todo el año)"]
        metodo = "E = consumo anual ÷ 12"
    elif cat == "LA":
        ciclos = uso["ciclos_semana"] * SEMANAS_MES
        if d.get("kwh_ciclo"):
            kwh = d["kwh_ciclo"] * ciclos
            sup += [f"{d['kwh_ciclo']} kWh por ciclo (programa algodón, según manual)"]
            metodo = "E = kWh por ciclo × ciclos por mes"
        else:
            kwh = d["potencia_w"] / 1000 * HORAS_CICLO_LAVA_FALLBACK * FACTOR_LAVA_FALLBACK * ciclos
            sup += ["El manual no informa kWh por ciclo: estimación con potencia, 1,5 h por ciclo y factor 0,5 (poco confiable)"]
            metodo = "E = P × h × factor (estimación)"
            return Resultado(kwh, kwh * tarifa, metodo, sup, False)
    elif cat == "MI":
        kwh = d["potencia_w"] / 1000 * uso["minutos_dia"] / 60 * uso["dias_mes"]
        sup += [f"Potencia de entrada {d['potencia_w']:.0f} W"]
        metodo = "E = P × h"
    else:
        raise ValueError(pid)
    return Resultado(kwh, kwh * tarifa, metodo, sup, True)


def ahorro(actual: Resultado, recomendado: Resultado):
    """Rango de ahorro mensual en pesos y kWh."""
    base = actual.pesos_mes - recomendado.pesos_mes
    kwh = actual.kwh_mes - recomendado.kwh_mes
    lo, hi = sorted([base * (1 - INCERTIDUMBRE), base * (1 + INCERTIDUMBRE)])
    return {"pesos": base, "kwh": kwh, "pesos_min": lo, "pesos_max": hi,
            "pesos_anio_min": lo * 12, "pesos_anio_max": hi * 12}


def fmt(n):
    return f"${n:,.0f}".replace(",", ".")
