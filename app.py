import os
import streamlit as st
import pandas as pd
from pinecone import Pinecone
from google import genai
import motor

st.set_page_config(page_title="Asistente de electrodomésticos", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1100px}
.kpi{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:16px 18px}
.kpi small{opacity:.65;text-transform:uppercase;letter-spacing:.04em;font-size:.72rem}
.kpi h2{margin:.2rem 0 0 0;font-size:1.9rem}
.cita{font-size:.82rem;opacity:.7;border-left:3px solid #f5a623;padding-left:10px;margin:.3rem 0}
</style>""", unsafe_allow_html=True)


def secreto(k, default=None):
    try:
        return st.secrets[k]
    except Exception:
        return os.environ.get(k, default)


@st.cache_resource
def clientes():
    pc = Pinecone(api_key=secreto("PINECONE_API_KEY"))
    return pc.Index("manuales-electro"), genai.Client(api_key=secreto("GEMINI_API_KEY"))


@st.cache_data
def datos():
    return motor.cargar_fichas(), motor.cargar_datos()


fichas, calc = datos()
por_id = {f["id"]: f for f in fichas}
CATS = {"Aire acondicionado": "AA", "Heladera": "HE", "Lavarropas": "LA", "Microondas": "MI"}
MODELO = secreto("GEMINI_MODEL", "gemini-2.5-flash")

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚡ Asistente")
    st.caption("Recomendaciones de uso eficiente basadas en el manual real de cada producto.")
    cat = st.radio("Categoría", list(CATS), index=0)
    opciones = [f for f in fichas if f["id"].startswith(CATS[cat])]
    pid = st.selectbox("Producto", [f["id"] for f in opciones],
                       format_func=lambda i: f"{por_id[i]['marca']} {por_id[i]['modelo']}")
    ficha = por_id[pid]
    st.divider()
    tarifa = st.number_input("Tarifa ($ por kWh)", min_value=1.0, value=100.0, step=10.0,
                             help="Cargá el valor de tu factura. 100 es solo un ejemplo.")
    with st.expander("Datos de la ficha"):
        st.write(f"**Tipo:** {ficha['tipo']}")
        st.write(f"**Potencia:** {ficha['potencia_nominal_w']}")
        st.write(f"**Consumo:** {ficha['consumo_etiqueta']}")
        st.write(f"**Eficiencia:** {ficha['clase_eficiencia']}")

st.title(f"{ficha['marca']} {ficha['modelo']}")
st.caption(f"{cat} · {ficha['tipo']}")
tab1, tab2 = st.tabs(["💬 Preguntarle al manual", "💰 Calcular consumo y ahorro"])

# ---------- Tab 1: RAG ----------
SISTEMA = """Sos un asistente de una tienda de electrodomésticos. Respondés SOLO con la información de los
fragmentos del manual que te paso. Reglas:
- Respondé en español rioplatense, claro y breve.
- Después de cada dato indicá la página entre paréntesis, por ejemplo (p. 12).
- Si el manual no trae el dato, decí exactamente: "Eso no figura en el manual." y no inventes.
- No hagas cuentas de dinero: para eso existe la pestaña de cálculo."""


def preguntar(q):
    idx, ia = clientes()
    r = idx.search(namespace="manuales", query={
        "top_k": 6, "inputs": {"text": q}, "filter": {"producto_id": {"$eq": pid}}})
    hits = r["result"]["hits"]
    ctx = "\n\n".join(f"[Página {h['fields']['pagina']:.0f}]\n{h['fields']['texto']}" for h in hits)
    prompt = f"{SISTEMA}\n\nProducto: {ficha['marca']} {ficha['modelo']}\n\nFRAGMENTOS:\n{ctx}\n\nPREGUNTA: {q}"
    resp = ia.models.generate_content(model=MODELO, contents=prompt)
    pags = sorted({int(h["fields"]["pagina"]) for h in hits})
    return resp.text, pags


with tab1:
    ejemplos = {"AA": "¿Qué temperatura recomienda para ahorrar energía?",
                "HE": "¿Cómo debo ubicar la heladera para que consuma menos?",
                "LA": "¿Cómo cargo el lavarropas para gastar menos energía?",
                "MI": "¿Qué cuidados tiene el uso del microondas?"}[CATS[cat]]
    if "chat" not in st.session_state or st.session_state.get("chat_pid") != pid:
        st.session_state.chat, st.session_state.chat_pid = [], pid
    for m in st.session_state.chat:
        with st.chat_message(m["rol"]):
            st.markdown(m["txt"])
            if m.get("pags"):
                st.markdown(f"<div class='cita'>Páginas consultadas: {', '.join(map(str, m['pags']))}</div>",
                            unsafe_allow_html=True)
    q = st.chat_input(ejemplos)
    if q:
        st.session_state.chat.append({"rol": "user", "txt": q})
        try:
            txt, pags = preguntar(q)
        except Exception as e:
            txt, pags = f"No pude consultar el manual ahora ({type(e).__name__}). Probá de nuevo.", []
        st.session_state.chat.append({"rol": "assistant", "txt": txt, "pags": pags})
        st.rerun()

# ---------- Tab 2: cálculo ----------
with tab2:
    st.markdown("Cálculo determinístico: **E = P × h**, **C = E × tarifa**, **Ahorro = C actual − C recomendado**.")
    c = CATS[cat]
    if c == "AA":
        a, b = st.columns(2)
        with a:
            st.subheader("Uso actual")
            h1 = st.slider("Horas por día", 0.0, 24.0, 10.0, 0.5, key="h1")
        with b:
            st.subheader("Uso recomendado")
            h2 = st.slider("Horas por día", 0.0, 24.0, 7.0, 0.5, key="h2")
        dias = st.slider("Días de uso por mes", 1, 31, 30)
        modo = st.radio("Modo", ["frio", "calor"], horizontal=True, format_func=str.capitalize)
        u1 = {"horas_dia": h1, "dias_mes": dias, "modo": modo}
        u2 = {"horas_dia": h2, "dias_mes": dias, "modo": modo}
    elif c == "LA":
        a, b = st.columns(2)
        with a:
            st.subheader("Uso actual")
            c1 = st.slider("Lavados por semana", 0, 14, 6, key="c1")
        with b:
            st.subheader("Uso recomendado")
            c2 = st.slider("Lavados por semana (cargas completas)", 0, 14, 4, key="c2")
        u1, u2 = {"ciclos_semana": c1}, {"ciclos_semana": c2}
    elif c == "MI":
        a, b = st.columns(2)
        with a:
            st.subheader("Uso actual")
            m1 = st.slider("Minutos por día", 0, 120, 30, key="m1")
        with b:
            st.subheader("Uso recomendado")
            m2 = st.slider("Minutos por día", 0, 120, 20, key="m2")
        dias = st.slider("Días de uso por mes", 1, 31, 30)
        u1, u2 = {"minutos_dia": m1, "dias_mes": dias}, {"minutos_dia": m2, "dias_mes": dias}
    else:
        u1 = u2 = {}
        st.info("La heladera funciona todo el año: su consumo mensual es el consumo anual de etiqueta dividido 12. "
                "Los consejos para gastar menos (ubicación, apertura de puerta, temperatura) están en la pestaña del manual.")

    r1 = motor.consumo_mensual(pid, u1, tarifa, calc)
    r2 = motor.consumo_mensual(pid, u2, tarifa, calc)
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'><small>Consumo mensual actual</small><h2>{r1.kwh_mes:.1f} kWh</h2>"
                f"<span>{motor.fmt(r1.pesos_mes)} por mes</span></div>", unsafe_allow_html=True)
    if c != "HE":
        s = motor.ahorro(r1, r2)
        k2.markdown(f"<div class='kpi'><small>Consumo recomendado</small><h2>{r2.kwh_mes:.1f} kWh</h2>"
                    f"<span>{motor.fmt(r2.pesos_mes)} por mes</span></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><small>Ahorro estimado por mes</small>"
                    f"<h2>{motor.fmt(s['pesos_min'])} – {motor.fmt(s['pesos_max'])}</h2>"
                    f"<span>{motor.fmt(s['pesos_anio_min'])} – {motor.fmt(s['pesos_anio_max'])} por año</span></div>",
                    unsafe_allow_html=True)
        st.bar_chart(pd.DataFrame({"$ por mes": [r1.pesos_mes, r2.pesos_mes]}, index=["Actual", "Recomendado"]))
    if not r1.confiable:
        st.warning("Este manual no informa el consumo por ciclo: el resultado es una estimación poco confiable.")
    with st.expander("Cómo se calculó y qué supuestos usamos", expanded=True):
        st.write(f"**Método:** {r1.metodo}")
        for x in r1.supuestos:
            st.write(f"- {x}")
        st.write(f"- Tarifa: {motor.fmt(tarifa)} por kWh (la cargás vos)")
        if c != "HE":
            st.write(f"- El ahorro se muestra como rango (±{motor.INCERTIDUMBRE:.0%}) porque el uso real varía.")
        st.caption("Fuente de los datos: " + calc[pid]["fuente"])
