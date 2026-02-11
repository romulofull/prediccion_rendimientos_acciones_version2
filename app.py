import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf


st.set_page_config(
    page_title="Predicción de Rendimiento de Acciones",
    layout="wide"
)
st.markdown("""
<style>
/* Number inputs más compactos */
div[data-testid="stNumberInput"] {
    max-width: 370px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Tipografía general */
html, body, [class*="css"]  {
    font-size: 17px;
}

h1 {
    font-size: 60px !important;
}
h2 {
    font-size: 50px !important;
}
h3 {
    font-size: 32px !important;
}

section[data-testid="stSidebar"] {
    width: 360px !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 70px;
    display: flex;
    flex-direction: column;
    align-items: center;
}
section[data-testid="stSidebar"] input {
    max-width: 260px;
    border-radius: 8px;
    padding: 6px;
}


table {
    font-size: 55px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align:center;'>Modelo de Retornos de Precios de Acciones</h1>",
    unsafe_allow_html=True
)


st.markdown("""
<div style="margin-top:30px;">
  <h2>📌 Descripción del proyecto</h2>
  <ul>
    <li><b>Objetivo</b>: Proyectar el rendimiento diario del precio de cierre de una acción.</li>
    <li><b>Metodología</b>: Enfoque multifactorial que integra información de precios, volumen, commodities y variables macroeconómicas.</li>
  </ul>
</div>
<div style="margin-top:30px;">
<h2>🧠 Variables utilizadas por el modelo</h2>
<ul>
  <li>📈 <b>Ret_precio_apertura</b>: Rendimiento porcentual diario del precio de apertura.</li>
  <li>📊 <b>Ret_precio_maximo</b>: Rendimiento porcentual del precio máximo del día anterior.</li>
  <li>📉 <b>Ret_precio_minimo</b>: Rendimiento porcentual del precio mínimo del día anterior.</li>
  <li>🔄 <b>Ret_volumen</b>: Variación logarítmica diaria del volumen transado (rezago de un día).</li>
  <li>🌍 <b>Sp500</b>: Rendimiento porcentual diario del índice S&P 500 (rezago de un día).</li>
  <li>🛢️ <b>Ret_petroleo_usd</b>: Rendimiento porcentual diario del precio del petróleo (rezago de un día).</li>
  <li>🏦 <b>D_tasa_tesoro_10y</b>: Variación diaria de la tasa del Tesoro de EE. UU. a 10 años (rezago de un día).</li>
  <li>🔩 <b>Ret_cobre_usd</b>: Rendimiento porcentual diario del precio del cobre (rezago de un día).</li>
  <li>🏦 <b>D_tasa_tesoro_3m</b>: Variación diaria de la tasa del Tesoro de EE. UU. a 3 meses (rezago de un día).</li>
  <li>🌏 <b>Ret_usd_yuan</b>: Rendimiento diario del tipo de cambio USD/CHINA. expectativas sobre comercio global y cadenas de suministro (rezago de un día).</li>
 </div>
</ul>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SELECCIÓN DE EMPRESA
# --------------------------------------------------
st.markdown("### 🏢 Seleccione la empresa")

empresas = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOGL)": "GOOGL",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "Netflix (NFLX)": "NFLX"
}

empresa_seleccionada = st.selectbox(
    "Empresa",
    options=list(empresas.keys())
)

ticker = empresas[empresa_seleccionada]
# --------------------------------------------------
# DESCARGA DATOS PRINCIPALES
# --------------------------------------------------
@st.cache_data(ttl=3600)
def cargar_precio(ticker):
    df = yf.download(ticker, period="10d", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()

df = cargar_precio(ticker)

if df is None or len(df) < 3:
    st.error("❌ No hay suficientes datos de precios")
    st.stop()

today, yesterday, day_before = df.iloc[-1], df.iloc[-2], df.iloc[-3]

# --------------------------------------------------
# FEATURES DE LA ACCIÓN
# --------------------------------------------------
ret_precio_apertura = (today["Open"] / yesterday["Open"]) - 1
ret_precio_maximo = (yesterday["High"] / day_before["High"]) - 1
ret_precio_minimo = (yesterday["Low"] / day_before["Low"]) - 1

vol_raw = (yesterday["Volume"] / day_before["Volume"]) - 1
ret_volumen = np.log1p(vol_raw) if vol_raw > -1 else 0.0

# --------------------------------------------------
# FUNCIÓN SEGURA PARA MERCADO
# --------------------------------------------------
def retorno_seguro(ticker_ref):
    df_ref = yf.download(ticker_ref, period="10d", interval="1d", progress=False)

    if isinstance(df_ref.columns, pd.MultiIndex):
        df_ref.columns = df_ref.columns.get_level_values(0)

    ret = df_ref["Close"].pct_change().dropna()

    return ret.iloc[-1] if len(ret) > 0 else 0.0

# --------------------------------------------------
# VARIABLES MACRO
# --------------------------------------------------
sp500 = retorno_seguro("^GSPC")
ret_petroleo_usd = retorno_seguro("CL=F")
ret_cobre_usd = retorno_seguro("HG=F")
ret_usd_yuan = retorno_seguro("CNY=X")

# Tasas → diferencias, NO retornos
def diff_tasa(ticker_ref):
    df_ref = yf.download(ticker_ref, period="10d", interval="1d", progress=False)
    if isinstance(df_ref.columns, pd.MultiIndex):
        df_ref.columns = df_ref.columns.get_level_values(0)

    diff = df_ref["Close"].diff().dropna()
    return diff.iloc[-1] if len(diff) > 0 else 0.0

d_tasa_tesoro_10y = diff_tasa("^TNX")
d_tasa_tesoro_3m = diff_tasa("^IRX")

# --------------------------------------------------
# DATAFRAME FINAL
# --------------------------------------------------
input_data = pd.DataFrame({
    "ret_precio_apertura": [ret_precio_apertura],
    "ret_precio_maximo": [ret_precio_maximo],
    "ret_precio_minimo": [ret_precio_minimo],
    "ret_volumen": [ret_volumen],
    "sp500": [sp500],
    "ret_petroleo_usd": [ret_petroleo_usd],
    "d_tasa_tesoro_10y": [d_tasa_tesoro_10y],
    "ret_cobre_usd": [ret_cobre_usd],
    "d_tasa_tesoro_3m": [d_tasa_tesoro_3m],
    "ret_usd_yuan": [ret_usd_yuan]
}).round(6)

st.subheader("📋 Escenario de mercado")
st.dataframe(input_data, use_container_width=True, hide_index=True)

# --------------------------------------------------
# MODELO
# --------------------------------------------------
try:
    data = joblib.load("modelo_rendimientos_catboost.pkl")
    model = data["model"]
    st.success("✅ Modelo cargado")
except Exception as e:
    st.error(e)
    st.stop()

# --------------------------------------------------
# PREDICCIÓN
# --------------------------------------------------
if st.button("📊 Ejecutar predicción"):
    if input_data.isna().any().any():
        st.error("❌ Existen valores NaN. No se puede predecir.")
        st.stop()

    X = input_data[model.feature_names_]
    pred = model.predict(X)

    st.metric(
        "📌 Rendimiento esperado del cierre",
        f"{pred[0]*100:.2f} %"
    )