from __future__ import annotations

from pathlib import Path
import base64
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Markov Regime-Switching | Fundos de Ações Portugueses",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

DATA_FINAL_FILE = DATA_DIR / "tabela_final_com_carhart.xlsx"
MARKOV_FILE = DATA_DIR / "comparacao_modelos_markov_variaveis_reduzidas.xlsx"
STATIONARITY_FILE = DATA_DIR / "output_python_testes_estacionaridade.txt"
SMOOTHED_PROBS_FILE = DATA_DIR / "probabilidades_suavizadas_markov_v2.xlsx"
LOGO_FILE = ASSETS_DIR / "ipca_esg_logo.png"

MODEL_CODE = "M1_STDMKT_FLOW"
MODEL_LABEL = "Especificação parcimoniosa"
MODEL_DESC = "Modelo Markov Regime-Switching com fund_return explicado por STDMKT e FLOW"
DEPENDENT_VARIABLE = "fund_return"
EXCLUDED_FUNDS = {"IMGA Global Equities Selection"}

VARIABLE_LABELS = {
    "fund_return": "Rendibilidade do fundo",
    "alpha_carhart": "Alpha de Carhart",
    "FLOW": "Fluxos líquidos",
    "STDMKT": "Risco de mercado",
    "STDFND": "Risco do fundo",
    "LNTNA": "Log(TNA)",
    "DELTA_LNTNA": "Δ Log(TNA)",
    "LNAGE": "Log idade",
    "ECOSIZE": "Dimensão económica",
    "LOG_ECOSIZE": "Log dimensão económica",
    "DELTA_LOG_ECOSIZE": "Δ Log dimensão económica",
    "UP": "Valor da UP",
    "TNA": "Valor líquido global",
}

EVENTS = [
    {"date": "2011-06-01", "label": "Crise da dívida soberana", "detail": "Período de tensão na dívida soberana europeia."},
    {"date": "2020-03-01", "label": "COVID-19", "detail": "Choque de mercado associado ao início da pandemia."},
    {"date": "2022-02-01", "label": "Guerra na Ucrânia / inflação", "detail": "Choque geopolítico e aceleração das pressões inflacionistas."},
    {"date": "2023-07-01", "label": "Taxas de juro elevadas", "detail": "Contexto de política monetária restritiva e níveis de preços elevados."},
]

BENCHMARK_BY_CATEGORY = {
    "Portugal": "PSI 20",
    "Europa": "STOXX Europe 600",
    "América": "S&P 500",
    "Global": "FTSE All-World",
    "Emergentes": "MSCI Emerging Markets",
    "Setoriais": "STOXX Europe 600 sectorial",
}


VARIABLES_TABLE = [
    {
        "Sigla": "fund_return",
        "Nome": "Rendibilidade do fundo",
        "Significado": "Rendibilidade mensal calculada a partir do valor da unidade de participação. É a variável dependente usada na especificação final.",
        "Fonte": "APFIPP — Valor da UP",
        "Modelo final": "Usada",
    },
    {
        "Sigla": "alpha_carhart",
        "Nome": "Alpha de Carhart",
        "Significado": "Medida alternativa de performance ajustada ao risco, estimada com fatores de mercado, dimensão, valor e momentum.",
        "Fonte": "Cálculo próprio com fatores de Carhart/Fama-French",
        "Modelo final": "Testada / não usada na especificação final",
    },
    {
        "Sigla": "FLOW",
        "Nome": "Fluxos líquidos do fundo",
        "Significado": "Entrada ou saída líquida estimada de capitais do fundo, ajustada pela rendibilidade do período.",
        "Fonte": "Cálculo próprio com TNA/VLG e rendibilidade",
        "Modelo final": "Usada",
    },
    {
        "Sigla": "STDMKT",
        "Nome": "Risco de mercado",
        "Significado": "Volatilidade do índice de mercado relevante para a categoria do fundo. Ajuda a distinguir períodos de menor e maior instabilidade de mercado.",
        "Fonte": "Índices de mercado por categoria APFIPP",
        "Modelo final": "Usada",
    },
    {
        "Sigla": "LNTNA",
        "Nome": "Logaritmo do total líquido global",
        "Significado": "Proxy da dimensão do fundo, calculada como logaritmo do valor líquido global do fundo.",
        "Fonte": "CMVM — Valor líquido global do fundo",
        "Modelo final": "Testada / removida por estacionaridade",
    },
    {
        "Sigla": "DELTA_LNTNA",
        "Nome": "Variação do logaritmo do TNA",
        "Significado": "Variação mensal da dimensão do fundo após transformação logarítmica.",
        "Fonte": "Cálculo próprio com VLG/CMVM",
        "Modelo final": "Testada / robustez",
    },
    {
        "Sigla": "AGE / LNAGE",
        "Nome": "Idade do fundo / log idade",
        "Significado": "Número de anos desde o início do fundo, podendo ser transformado em logaritmo.",
        "Fonte": "Informação institucional do fundo e cálculo próprio",
        "Modelo final": "Testada / não usada na especificação final",
    },
    {
        "Sigla": "STDFND",
        "Nome": "Risco específico do fundo",
        "Significado": "Volatilidade histórica da rendibilidade do próprio fundo.",
        "Fonte": "Cálculo próprio a partir da rendibilidade do fundo",
        "Modelo final": "Testada / robustez",
    },
    {
        "Sigla": "ECOSIZE / LOG_ECOSIZE",
        "Nome": "Dimensão económica",
        "Significado": "Proxy macroeconómica baseada no PIB relevante para o mercado/categoria do fundo.",
        "Fonte": "Séries macroeconómicas externas e cálculo próprio",
        "Modelo final": "Testada / removida por estacionaridade",
    },
    {
        "Sigla": "DELTA_LOG_ECOSIZE",
        "Nome": "Variação do logaritmo da dimensão económica",
        "Significado": "Transformação estacionária da dimensão económica, usada para testes de robustez.",
        "Fonte": "Cálculo próprio a partir de séries de PIB",
        "Modelo final": "Testada / robustez",
    },
]

def benchmark_from_category(category: str, current: str = "—") -> str:
    cat = str(category or "").strip()
    # Corrige casos em que o índice ficou mal preenchido ou ausente.
    for key, bench in BENCHMARK_BY_CATEGORY.items():
        if key.lower() in cat.lower():
            return bench
    return current if current and current != "nan" else "—"

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    :root{
        --primary:#34596A; --soft:#EDF5F8; --line:#D7E4EA; --text:#203743; --muted:#60717C;
    }
    .block-container{padding-top:2.8rem; padding-bottom:2.2rem; max-width:1500px;}
    [data-testid="stSidebar"]{background:#F7FAFB;border-right:1px solid #E0E8EC;}
    .sidebar-logo-wrap{padding:0.25rem 0.1rem 1.05rem 0.1rem;margin-bottom:.7rem;border-bottom:1px solid #E1E8EC;text-align:center;}
    .sidebar-logo{width:200px;max-width:94%;height:auto;object-fit:contain;}
    .main-header{
        background:linear-gradient(135deg,#F8FCFD 0%,#EAF3F7 100%);
        border:1px solid #D2E2E9;border-radius:18px;padding:1.25rem 1.45rem;margin:0.20rem 0 1.10rem 0;
        box-shadow:0 1px 5px rgba(20,45,60,.055);overflow:visible;box-sizing:border-box;
    }
    .main-title{font-size:1.28rem;font-weight:760;color:#213946;line-height:1.22;margin:0 0 .28rem 0;letter-spacing:-.018em;}
    .main-subtitle{font-size:.82rem;color:#526A78;margin:0;}
    .metric-card{background:white;border:1px solid #DCE6EB;border-radius:15px;padding:.76rem .9rem;min-height:74px;box-shadow:0 1px 2px rgba(20,45,60,.04);}
    .metric-label{color:#617582;font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;font-weight:760;margin-bottom:.22rem;}
    .metric-value{color:#1F3642;font-size:1.08rem;font-weight:760;line-height:1.12;}
    .metric-note{color:#758792;font-size:.66rem;margin-top:.12rem;}
    .section-title{font-size:1.02rem;font-weight:760;color:#243B4A;margin-top:.75rem;margin-bottom:.35rem;}
    .info-box{background:#F8FBFC;border:1px solid #DCE7EC;border-radius:14px;padding:.88rem 1rem;color:#334E5C;font-size:.90rem;}
    .formula-box{background:#FBFCFD;border-left:4px solid #6C93A6;border-top:1px solid #E0E8EC;border-right:1px solid #E0E8EC;border-bottom:1px solid #E0E8EC;border-radius:12px;padding:.85rem 1rem;margin:.45rem 0 .65rem 0;font-size:.93rem;}
    .matrix-box{background:white;border:0;padding:.35rem .2rem;margin-top:.45rem;text-align:center;font-size:1.05rem;color:#243B4A;}
    .matrix-table{margin:0 auto;border-collapse:collapse;font-size:1rem;}
    .matrix-table td{padding:.18rem .55rem;text-align:center;}
    .matrix-bracket{font-size:2.2rem;font-weight:300;color:#243B4A;line-height:1;}
    .small-muted{color:#60717C;font-size:.80rem;}
    .pill{display:inline-block;border:1px solid #DCE6EB;border-radius:999px;padding:.18rem .52rem;background:#F8FBFC;color:#34596A;font-size:.72rem;font-weight:650;margin-right:.25rem;}
    .var-note{font-size:.78rem;color:#60717C;margin-top:.25rem;margin-bottom:.5rem;}
    h1,h2,h3{color:#243B4A;}
    .stTabs [data-baseweb="tab-list"]{gap:6px;}
    .stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0;padding:.45rem .75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def file_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def stop_if_missing_files() -> None:
    missing = [p for p in [DATA_FINAL_FILE, MARKOV_FILE, STATIONARITY_FILE] if not p.exists()]
    if missing:
        st.error("Faltam ficheiros na pasta data/.")
        st.code(
            "dashboard/\n"
            "├── app_dashboard_markov_v14_final.py\n"
            "├── assets/ipca_esg_logo.png\n"
            "└── data/\n"
            "    ├── tabela_final_com_carhart.xlsx\n"
            "    ├── comparacao_modelos_markov_variaveis_reduzidas.xlsx\n"
            "    ├── output_python_testes_estacionaridade.txt\n"
            "    └── probabilidades_suavizadas_markov_v2.xlsx  # opcional"
        )
        st.write("Em falta:", ", ".join(p.name for p in missing))
        st.stop()


def fmt_num(x, decimals=3):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):,.{decimals}f}".replace(",", " ")
    except Exception:
        return str(x)


def fmt_pct(x, decimals=1):
    if pd.isna(x):
        return "—"
    return f"{float(x)*100:.{decimals}f}%"


def signif_bool(s) -> bool:
    if pd.isna(s):
        return False
    return str(s).strip() in {"*", "**", "***"}





def ensure_datetime_series(s: pd.Series) -> pd.Series:
    """Converte datas de forma robusta.

    O Excel usado no projeto tem datas no formato 01-2010, 02-2010, etc.
    Esta função evita que essas datas sejam interpretadas como números/timestamps
    e garante que o eixo temporal aparece como mês-ano.
    """
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    # Datas Excel em número ou timestamps acidentais
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        med = vals.dropna().abs().median() if vals.notna().any() else np.nan
        if pd.notna(med) and med > 1e12:
            return pd.to_datetime(vals, unit="ns", errors="coerce")
        if pd.notna(med) and med > 20000:
            return pd.to_datetime(vals, unit="D", origin="1899-12-30", errors="coerce")
        return pd.to_datetime(vals, errors="coerce")

    txt = s.astype(str).str.strip()
    txt = txt.replace({"nan": None, "NaT": None, "None": None})

    # Formato principal do projeto: MM-YYYY
    parsed = pd.to_datetime(txt, format="%m-%Y", errors="coerce")
    # Outros formatos possíveis
    if parsed.notna().sum() == 0 or parsed.isna().mean() > 0.5:
        parsed2 = pd.to_datetime(txt, format="%d-%m-%Y", errors="coerce")
        if parsed2.notna().sum() > parsed.notna().sum():
            parsed = parsed2
    if parsed.notna().sum() == 0 or parsed.isna().mean() > 0.5:
        parsed2 = pd.to_datetime(txt, errors="coerce", dayfirst=True)
        if parsed2.notna().sum() > parsed.notna().sum():
            parsed = parsed2

    return parsed

def clean_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = ensure_datetime_series(df["date"])
        df = df[df["date"].notna()].copy()
        df = df[(df["date"].dt.year >= 1990) & (df["date"].dt.year <= 2035)].copy()
    return df


def normalize_name(value: str) -> str:
    import unicodedata
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def build_fund_sheet_map(model_funds: list[str], fund_sheets: list[str]) -> dict[str, str]:
    """Mapeia o nome oficial do modelo para a folha do Excel, tolerando nomes cortados."""
    import difflib
    mapping = {}
    norm_sheets = {normalize_name(s): s for s in fund_sheets}
    for fund in model_funds:
        nf = normalize_name(fund)
        if nf in norm_sheets:
            mapping[fund] = norm_sheets[nf]
            continue
        # trata casos em que o nome da folha ficou truncado, por exemplo Améric / Servic
        candidates = []
        for ns, sheet in norm_sheets.items():
            if nf.startswith(ns) or ns.startswith(nf):
                candidates.append((1.0, sheet))
            else:
                candidates.append((difflib.SequenceMatcher(None, nf, ns).ratio(), sheet))
        best_score, best_sheet = max(candidates, key=lambda x: x[0])
        if best_score >= 0.86:
            mapping[fund] = best_sheet
    return mapping

@st.cache_data(show_spinner=False)
def load_fund_data() -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(DATA_FINAL_FILE)
    out = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(DATA_FINAL_FILE, sheet_name=sheet)
        if sheet in EXCLUDED_FUNDS:
            continue
        if "date" in df.columns:
            raw = df["date"].copy()
            # Conversão robusta de datas: evita interpretar datas como números enormes no eixo X
            if pd.api.types.is_datetime64_any_dtype(raw):
                df["date"] = pd.to_datetime(raw, errors="coerce")
            elif pd.api.types.is_numeric_dtype(raw):
                vals = pd.to_numeric(raw, errors="coerce")
                med = vals.dropna().abs().median() if vals.notna().any() else np.nan
                if pd.notna(med) and med > 1e12:
                    df["date"] = pd.to_datetime(vals, unit="ns", errors="coerce")
                elif pd.notna(med) and med > 20000:
                    df["date"] = pd.to_datetime(vals, unit="D", origin="1899-12-30", errors="coerce")
                else:
                    df["date"] = pd.to_datetime(raw, errors="coerce", dayfirst=True)
            else:
                txt = raw.astype(str).str.strip()
                parsed = pd.to_datetime(txt, format="%m-%Y", errors="coerce")
                if parsed.isna().mean() > 0.5:
                    parsed = pd.to_datetime(txt, format="%d-%m-%Y", errors="coerce")
                if parsed.isna().mean() > 0.5:
                    parsed = pd.to_datetime(txt, errors="coerce", dayfirst=True)
                df["date"] = parsed
        for c in df.columns:
            if c not in ["date", "fund_name", "categoria_apfipp", "market_index", "inicio_fundo"]:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() > 0:
                    df[c] = converted
        out[sheet] = df
    return out


@st.cache_data(show_spinner=False)
def load_markov_data() -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(MARKOV_FILE, sheet_name=None)
    for k, df in list(sheets.items()):
        if "Fundo" in df.columns:
            sheets[k] = df[~df["Fundo"].isin(EXCLUDED_FUNDS)].copy()
    return sheets


@st.cache_data(show_spinner=False)
def load_stationarity_text() -> str:
    return STATIONARITY_FILE.read_text(encoding="utf-8", errors="ignore")


@st.cache_data(show_spinner=False)
def load_smoothed_probabilities() -> pd.DataFrame:
    """Carrega probabilidades suavizadas de forma robusta.

    Evita o erro `arg must be a list, tuple, 1-d array, or Series`, que pode ocorrer
    quando há colunas duplicadas depois da normalização dos nomes.
    """
    if not SMOOTHED_PROBS_FILE.exists():
        return pd.DataFrame()
    try:
        xls = pd.ExcelFile(SMOOTHED_PROBS_FILE)
        sheet = "Probabilidades_Suavizadas" if "Probabilidades_Suavizadas" in xls.sheet_names else xls.sheet_names[0]
        raw = pd.read_excel(SMOOTHED_PROBS_FILE, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    norm_map = {c: normalize_name(c) for c in raw.columns}

    def find_col(candidates):
        candidates = {normalize_name(x) for x in candidates}
        for c, nc in norm_map.items():
            if nc in candidates:
                return c
        return None

    fund_col = find_col(["Fundo", "Fund", "Nome do fundo"])
    date_col = find_col(["Date", "Data"])
    low_col = find_col(["Prob_Low_Volatility", "Prob Low Volatility", "Prob Low", "Prob_Regime_0", "Prob Regime 0"])
    high_col = find_col(["Prob_High_Volatility", "Prob High Volatility", "Prob High", "Prob_Regime_1", "Prob Regime 1"])
    regime_col = find_col(["Regime_mais_provavel", "Regime mais provavel", "Regime classificado", "Regime"])

    if fund_col is None or date_col is None or low_col is None or high_col is None:
        return pd.DataFrame()

    df = pd.DataFrame({
        "Fundo": raw[fund_col].astype(str),
        "date": ensure_datetime_series(raw[date_col]),
        "Prob_Low_Volatility": pd.to_numeric(raw[low_col], errors="coerce"),
        "Prob_High_Volatility": pd.to_numeric(raw[high_col], errors="coerce"),
    })

    if regime_col is not None:
        df["Regime_mais_provavel"] = raw[regime_col].astype(str)
    else:
        df["Regime_mais_provavel"] = np.where(
            df["Prob_Low_Volatility"] >= df["Prob_High_Volatility"],
            "Low-Volatility",
            "High-Volatility",
        )

    df = df.dropna(subset=["date", "Prob_Low_Volatility", "Prob_High_Volatility"]).copy()
    df = df[(df["date"].dt.year >= 1990) & (df["date"].dt.year <= 2035)].copy()
    df = df[~df["Fundo"].isin(EXCLUDED_FUNDS)].copy()
    return df.sort_values(["Fundo", "date"])

def probabilities_for_fund(probs: pd.DataFrame, fund: str) -> pd.DataFrame:
    if probs.empty or "Fundo" not in probs.columns:
        return pd.DataFrame()
    m = probs[probs["Fundo"].astype(str).eq(fund)].copy()
    if not m.empty:
        return m.sort_values("date") if "date" in m.columns else m
    nf = normalize_name(fund)
    tmp = probs.copy()
    tmp["_norm_fundo"] = tmp["Fundo"].astype(str).map(normalize_name)
    # tolera nomes truncados
    m = tmp[(tmp["_norm_fundo"].apply(lambda x: nf.startswith(x) or x.startswith(nf)))].copy()
    if m.empty:
        import difflib
        tmp["_score"] = tmp["_norm_fundo"].apply(lambda x: difflib.SequenceMatcher(None, nf, x).ratio())
        best = tmp["_score"].max()
        if pd.notna(best) and best >= 0.86:
            m = tmp[tmp["_score"].eq(best)].copy()
    if m.empty:
        return pd.DataFrame()
    m = m.drop(columns=[c for c in ["_norm_fundo", "_score"] if c in m.columns], errors="ignore")
    return m.sort_values("date") if "date" in m.columns else m


def add_probability_event_markers(fig: go.Figure) -> go.Figure:
    for ev in EVENTS:
        dt = pd.to_datetime(ev["date"])
        fig.add_vline(x=dt, line_width=1, line_dash="dot", opacity=.25)
    return fig


def render_smoothed_probabilities(prob_df: pd.DataFrame) -> None:
    if prob_df.empty or "date" not in prob_df.columns:
        st.info("Não foram encontradas probabilidades suavizadas para este fundo.")
        return
    low_col = "Prob_Low_Volatility" if "Prob_Low_Volatility" in prob_df.columns else None
    high_col = "Prob_High_Volatility" if "Prob_High_Volatility" in prob_df.columns else None
    if low_col is None and "Prob_Regime_0" in prob_df.columns:
        low_col = "Prob_Regime_0"
    if high_col is None and "Prob_Regime_1" in prob_df.columns:
        high_col = "Prob_Regime_1"
    if low_col is None or high_col is None:
        st.info("O ficheiro de probabilidades existe, mas não tem as colunas de probabilidade esperadas.")
        return
    df = prob_df[["date", low_col, high_col] + (["Regime_mais_provavel"] if "Regime_mais_provavel" in prob_df.columns else [])].dropna(subset=["date"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[low_col], mode="lines", name="Low-Volatility", line=dict(width=2.2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df[high_col], mode="lines", name="High-Volatility", line=dict(width=2.2)))
    add_probability_event_markers(fig)
    fig.update_yaxes(range=[0,1], tickformat=".0%", title="Probabilidade suavizada")
    fig.update_xaxes(type="date", tickformat="%Y", hoverformat="%m-%Y")
    fig.update_layout(height=430, margin=dict(l=20,r=20,t=25,b=20), hovermode="x unified", legend=dict(orientation="h", y=-.18))
    st.plotly_chart(fig, use_container_width=True)


def render_regime_timeline(prob_df: pd.DataFrame) -> None:
    if prob_df.empty or "date" not in prob_df.columns:
        st.info("Não foram encontradas probabilidades suavizadas para desenhar a timeline dos regimes.")
        return
    df = prob_df.copy().dropna(subset=["date"])
    low_col = "Prob_Low_Volatility" if "Prob_Low_Volatility" in df.columns else "Prob_Regime_0" if "Prob_Regime_0" in df.columns else None
    high_col = "Prob_High_Volatility" if "Prob_High_Volatility" in df.columns else "Prob_Regime_1" if "Prob_Regime_1" in df.columns else None
    if "Regime_mais_provavel" not in df.columns:
        if low_col and high_col:
            df["Regime_mais_provavel"] = np.where(df[low_col] >= df[high_col], "Low-Volatility", "High-Volatility")
        else:
            st.info("Não há informação suficiente para classificar o regime mais provável.")
            return
    # blocos por mês: usa scatter com quadrados para aspeto de timeline compacto
    y = [1] * len(df)
    color_map = {"Low-Volatility": "#A9D4F4", "High-Volatility": "#5A94D6"}
    fig = go.Figure()
    for regime, g in df.groupby("Regime_mais_provavel"):
        fig.add_trace(go.Scatter(
            x=g["date"], y=[1]*len(g), mode="markers", name=str(regime),
            marker=dict(symbol="square", size=12, color=color_map.get(str(regime), "#9BB")),
            hovertemplate="%{x|%m-%Y}<br>Regime: " + str(regime) + "<extra></extra>"
        ))
    add_probability_event_markers(fig)
    fig.update_yaxes(visible=False, range=[0.8,1.2])
    fig.update_xaxes(type="date", tickformat="%Y", hoverformat="%m-%Y")
    fig.update_layout(height=170, margin=dict(l=20,r=20,t=10,b=25), legend=dict(orientation="h", y=-.35))
    st.plotly_chart(fig, use_container_width=True)


def final_model_rows(markov: dict[str, pd.DataFrame], sheet: str) -> pd.DataFrame:
    df = markov.get(sheet, pd.DataFrame()).copy()
    if df.empty or "Modelo" not in df.columns:
        return df
    return df[df["Modelo"].astype(str).eq(MODEL_CODE)].copy()


def get_category_from_fund_df(df: pd.DataFrame) -> str:
    if "categoria_apfipp" not in df.columns:
        return "Sem categoria"
    vals = df["categoria_apfipp"].dropna().astype(str).unique()
    return vals[0] if len(vals) else "Sem categoria"


def row_for_fund(compar: pd.DataFrame, fund: str) -> pd.Series | None:
    m = compar[compar["Fundo"].astype(str).eq(fund)]
    if m.empty:
        return None
    return m.iloc[0]


def coefs_for_fund(coefs: pd.DataFrame, fund: str) -> pd.DataFrame:
    if coefs.empty:
        return coefs
    return coefs[(coefs["Fundo"].astype(str).eq(fund)) & (coefs["Modelo"].astype(str).eq(MODEL_CODE))].copy()


def coef_value(cdf: pd.DataFrame, regime: str, var: str):
    m = cdf[(cdf["Regime"].astype(str).eq(regime)) & (cdf["Variável"].astype(str).eq(var))]
    if m.empty:
        return np.nan
    return float(m.iloc[0]["Coeficiente"])


def coef_star(cdf: pd.DataFrame, regime: str, var: str):
    m = cdf[(cdf["Regime"].astype(str).eq(regime)) & (cdf["Variável"].astype(str).eq(var))]
    if m.empty or pd.isna(m.iloc[0].get("Significancia")):
        return ""
    return str(m.iloc[0]["Significancia"])


def build_formula(cdf: pd.DataFrame, regime: str) -> str:
    b0 = coef_value(cdf, regime, "Intercept")
    b1 = coef_value(cdf, regime, "STDMKT")
    b2 = coef_value(cdf, regime, "FLOW")
    s0 = coef_star(cdf, regime, "Intercept")
    s1 = coef_star(cdf, regime, "STDMKT")
    s2 = coef_star(cdf, regime, "FLOW")
    if pd.isna(b0) or pd.isna(b1) or pd.isna(b2):
        return "fund_return_t = β₀ + β₁·STDMKT_t + β₂·FLOW_t + ε_t"
    sign1 = "+" if b1 >= 0 else "−"
    sign2 = "+" if b2 >= 0 else "−"
    return (
        f"fund_return_t = {b0:.4f}{s0} {sign1} {abs(b1):.4f}{s1}·STDMKT_t "
        f"{sign2} {abs(b2):.4f}{s2}·FLOW_t + ε_t"
    )


def parse_stationarity_summary(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    rows = []
    pattern = re.compile(r"^([A-Z_]+|fund_return|FLOW|LNTNA|LNAGE|STDFND|STDMKT|ECOSIZE)\s+\|\s+ADF:\s+(.*?)\s+\|\s+KPSS:\s+(.*?)\s+\|\s+(.*)$")
    for line in lines:
        m = pattern.match(line.strip())
        if m:
            rows.append({"Variável": m.group(1), "ADF": m.group(2), "KPSS": m.group(3), "Recomendação": m.group(4)})
    return pd.DataFrame(rows)


def transition_matrix(row: pd.Series | None) -> np.ndarray:
    if row is None:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])
    return np.array([
        [row.get("P(Low→Low)", np.nan), row.get("P(Low→High)", np.nan)],
        [row.get("P(High→Low)", np.nan), row.get("P(High→High)", np.nan)],
    ], dtype=float)


def add_event_markers(fig: go.Figure, dfp: pd.DataFrame, y_column: str | None = None) -> go.Figure:
    """Adiciona eventos como linhas discretas sem substituir a evolução das variáveis."""
    if dfp.empty:
        return fig
    for ev in EVENTS:
        dt = pd.to_datetime(ev["date"])
        fig.add_vline(x=dt, line_width=1, line_dash="dot", opacity=.30)
        fig.add_annotation(
            x=dt, y=1.02, xref="x", yref="paper",
            text="●", showarrow=False,
            hovertext=f"<b>{ev['label']}</b><br>Data: {dt.strftime('%m-%Y')}<br>{ev['detail']}",
            hoverlabel=dict(bgcolor="white"),
            font=dict(size=14),
        )
    return fig


def render_event_legend() -> None:
    labels = " &nbsp; ".join([
        f"<span class='pill'>{pd.to_datetime(ev['date']).strftime('%m-%Y')}: {ev['label']}</span>" for ev in EVENTS
    ])
    st.markdown(f"<div style='margin-top:-.25rem;margin-bottom:.45rem'>{labels}</div>", unsafe_allow_html=True)


def simple_metric(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fund_interpretation(fund: str, row: pd.Series | None, cdf: pd.DataFrame) -> str:
    if row is None:
        return "Não foi encontrada a linha do modelo final para este fundo."
    p_ll = row.get("P(Low→Low)", np.nan)
    p_hh = row.get("P(High→High)", np.nan)
    d_l = row.get("Duracao_Low_meses", np.nan)
    d_h = row.get("Duracao_High_meses", np.nan)
    bic = row.get("BIC", np.nan)

    dominant = "baixa volatilidade" if p_ll >= p_hh else "alta volatilidade"
    risk_msg = ""
    for reg in ["Low-Volatility", "High-Volatility"]:
        m = cdf[(cdf["Regime"].eq(reg)) & (cdf["Variável"].eq("STDMKT"))]
        if not m.empty and signif_bool(m.iloc[0].get("Significancia")):
            coef = float(m.iloc[0]["Coeficiente"])
            direction = "positivo" if coef > 0 else "negativo"
            risk_msg += f" No regime {reg.replace('-', ' ').lower()}, STDMKT apresenta efeito {direction} e estatisticamente significativo."
    if not risk_msg:
        risk_msg = " Nesta especificação, o efeito de STDMKT deve ser interpretado com cautela quando não apresenta significância estatística."

    return (
        f"Para **{fund}**, o modelo final apresenta maior persistência no regime de **{dominant}**. "
        f"A duração esperada é de {fmt_num(d_l,1)} meses no regime Low-Volatility e {fmt_num(d_h,1)} meses no regime High-Volatility. "
        f"O BIC do modelo é {fmt_num(bic,2)}, devendo ser usado para comparação relativa entre modelos: valores menores indicam melhor equilíbrio entre ajustamento e parcimónia."
        f"{risk_msg}"
    )



def build_fund_metadata_table(available_funds: list[str], fund_sheet_map: dict[str, str], fund_data: dict[str, pd.DataFrame], category_map: dict[str, str]) -> pd.DataFrame:
    rows = []
    for fund in available_funds:
        df = fund_data[fund_sheet_map[fund]].copy()
        category = category_map.get(fund, "—")

        # Benchmark final por categoria, conforme definido para o trabalho
        idx = "—"
        if "market_index" in df.columns:
            vals = df["market_index"].dropna().astype(str).unique()
            if len(vals):
                idx = vals[0]
        idx = benchmark_from_category(category, idx)

        periodo = "—"
        if "date" in df.columns:
            datas = ensure_datetime_series(df["date"])
            valid = datas.notna()
            # O período da amostra é calculado entre a primeira e a última observação útil do fundo.
            if DEPENDENT_VARIABLE in df.columns:
                valid = valid & df[DEPENDENT_VARIABLE].notna()
            datas_validas = datas[valid]
            datas_validas = datas_validas[(datas_validas.dt.year >= 1990) & (datas_validas.dt.year <= 2035)]
            if datas_validas.notna().any():
                periodo = f"{datas_validas.min().strftime('%m-%Y')} a {datas_validas.max().strftime('%m-%Y')}"

        obs = int(df[DEPENDENT_VARIABLE].notna().sum()) if DEPENDENT_VARIABLE in df.columns else int(len(df))
        rows.append({
            "Fundo": fund,
            "Categoria APFIPP": category,
            "Benchmark usado para STDMKT": idx,
            "Período da amostra": periodo,
            "Observações": obs,
        })
    return pd.DataFrame(rows)


def build_variables_overview_table() -> pd.DataFrame:
    return pd.DataFrame(VARIABLES_TABLE)


def render_variables_table() -> None:
    st.markdown("<div class='section-title'>Tabela geral das variáveis consideradas</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='var-note'>A tabela distingue as variáveis usadas na especificação parcimoniosa das variáveis testadas ou usadas apenas em análises de robustez.</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(build_variables_overview_table(), hide_index=True, use_container_width=True)


def render_regime_methodology() -> None:
    st.markdown("<div class='section-title'>Explicação metodológica dos regimes</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='info-box'>
        O modelo Markov Regime-Switching assume que a performance observada do fundo pode alternar entre dois estados latentes não observados diretamente. Estes estados são estimados endogenamente pelo modelo a partir dos dados, sem impor previamente as datas de mudança de regime. Nesta dashboard, os regimes são interpretados como <b>Low-Volatility</b> e <b>High-Volatility</b>. O primeiro representa períodos de menor instabilidade relativa; o segundo representa períodos em que a volatilidade e o risco de mercado tendem a ser mais elevados. A variável <b>STDMKT</b> é central nesta leitura, porque representa a volatilidade do mercado de referência do fundo e ajuda a interpretar a sensibilidade da performance ao contexto de mercado.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_transition_diagram_matlab(P: np.ndarray) -> None:
    """Diagrama compacto de transição com loops de permanência corretamente orientados."""
    p_ll, p_lh, p_hl, p_hh = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
    fig = go.Figure()

    low_x, high_x, y0 = 0.28, 0.72, 0.42

    # Nós
    fig.add_trace(go.Scatter(
        x=[low_x, high_x], y=[y0, y0],
        mode="markers+text",
        marker=dict(
            size=50,
            color=["#A9D4F4", "#5A94D6"],
            line=dict(width=2, color="#2E5874"),
        ),
        text=["Low<br>Volatility", "High<br>Volatility"],
        textfont=dict(size=9, color="#173040"),
        textposition="middle center",
        hoverinfo="skip",
        showlegend=False,
    ))

    annotations = [
        # Low -> High
        dict(ax=0.35, ay=0.51, x=0.65, y=0.51, xref="x", yref="y", axref="x", ayref="y",
             text=f"p₁₂ = {fmt_pct(p_lh)}", showarrow=True, arrowhead=3, arrowsize=1,
             arrowwidth=1.9, arrowcolor="#425D70", font=dict(size=10, color="#425D70"),
             bgcolor="rgba(255,255,255,.85)"),
        # High -> Low
        dict(ax=0.65, ay=0.33, x=0.35, y=0.33, xref="x", yref="y", axref="x", ayref="y",
             text=f"p₂₁ = {fmt_pct(p_hl)}", showarrow=True, arrowhead=3, arrowsize=1,
             arrowwidth=1.9, arrowcolor="#425D70", font=dict(size=10, color="#425D70"),
             bgcolor="rgba(255,255,255,.85)"),
        # Labels dos loops
        dict(x=low_x, y=0.66, text=f"p₁₁ = {fmt_pct(p_ll)}", showarrow=False,
             font=dict(size=10, color="#2F5368"), bgcolor="rgba(255,255,255,.88)"),
        dict(x=high_x, y=0.66, text=f"p₂₂ = {fmt_pct(p_hh)}", showarrow=False,
             font=dict(size=10, color="#2F5368"), bgcolor="rgba(255,255,255,.88)"),
        # Cabeças dos loops: terminam no próprio círculo, por baixo/lateral, não para fora
        dict(ax=0.20, ay=0.60, x=0.245, y=0.47, xref="x", yref="y", axref="x", ayref="y",
             text="", showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2,
             arrowcolor="#2F5368"),
        dict(ax=0.80, ay=0.60, x=0.755, y=0.47, xref="x", yref="y", axref="x", ayref="y",
             text="", showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=2,
             arrowcolor="#2F5368"),
    ]

    shapes = [
        # Loop Low — compacto, junto ao círculo, com abertura lateral
        dict(type="path", path="M 0.245 0.49 C 0.13 0.58, 0.15 0.74, 0.28 0.74 C 0.40 0.74, 0.42 0.58, 0.315 0.49",
             line=dict(color="#2F5368", width=2)),
        # Loop High
        dict(type="path", path="M 0.755 0.49 C 0.87 0.58, 0.85 0.74, 0.72 0.74 C 0.60 0.74, 0.58 0.58, 0.685 0.49",
             line=dict(color="#2F5368", width=2)),
    ]

    fig.update_layout(
        annotations=annotations,
        shapes=shapes,
        height=230,
        width=600,
        xaxis=dict(visible=False, range=[0.08, 0.92]),
        yaxis=dict(visible=False, range=[0.22, 0.79]),
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    diag_l, diag_c, diag_r = st.columns([0.9, 2.2, 0.9])
    with diag_c:
        st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False})

def render_method_intro() -> None:
    st.markdown("<div class='section-title'>Objetivo da análise</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='info-box'>
        A análise pretende avaliar se a performance dos fundos de ações portugueses se comporta de forma diferente em regimes de mercado distintos. O modelo Markov Regime-Switching permite identificar dois estados latentes, associados a períodos de menor e maior volatilidade, e estimar como o risco de mercado e os fluxos dos fundos se relacionam com a rendibilidade em cada regime.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='info-box'><b>1. Performance</b><br>A variável dependente é a rendibilidade mensal do fundo (<i>fund_return</i>).</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='info-box'><b>2. Regimes</b><br>O modelo distingue regimes de baixa e alta volatilidade, estimando probabilidades de transição entre estados.</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='info-box'><b>3. Determinantes</b><br>A especificação parcimoniosa considera <i>STDMKT</i> e <i>FLOW</i> como variáveis explicativas.</div>""", unsafe_allow_html=True)


def render_math_matrix(P: np.ndarray) -> None:
    p_ll, p_lh, p_hl, p_hh = P[0,0], P[0,1], P[1,0], P[1,1]
    def f(x):
        return "—" if pd.isna(x) else f"{float(x):.3f}"
    html = f"""
    <div class='matrix-box'>
      <table class='matrix-table'>
        <tr>
          <td style='font-weight:700;padding-right:.25rem;'>P =</td>
          <td class='matrix-bracket'>[</td>
          <td>
            <table class='matrix-table'>
              <tr><td>{f(p_ll)}</td><td>{f(p_lh)}</td></tr>
              <tr><td>{f(p_hl)}</td><td>{f(p_hh)}</td></tr>
            </table>
          </td>
          <td class='matrix-bracket'>]</td>
        </tr>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# CARREGAMENTO
# ============================================================
stop_if_missing_files()
fund_data = load_fund_data()
markov = load_markov_data()
stationarity_text = load_stationarity_text()
smoothed_probs = load_smoothed_probabilities()

compar = final_model_rows(markov, "Comparacao_modelos")
coefs_all = final_model_rows(markov, "Coeficientes")
matrices = final_model_rows(markov, "Matrizes_transicao")
durations = final_model_rows(markov, "Duracoes")
stationarity_summary = parse_stationarity_summary(stationarity_text)

# Fundos disponíveis: usa os nomes oficiais dos resultados do modelo e mapeia para as folhas do Excel.
# Isto evita perder fundos quando uma folha está truncada, por exemplo "Améric" vs "América".
model_funds = sorted([f for f in compar.get("Fundo", pd.Series(dtype=str)).dropna().astype(str).unique() if f not in EXCLUDED_FUNDS])
fund_sheet_map = build_fund_sheet_map(model_funds, list(fund_data.keys()))
available_funds = [f for f in model_funds if f in fund_sheet_map]

category_map = {}
for f in available_funds:
    r = row_for_fund(compar, f)
    if r is not None and pd.notna(r.get("Categoria_APFIPP")):
        category_map[f] = str(r.get("Categoria_APFIPP"))
    else:
        category_map[f] = get_category_from_fund_df(fund_data[fund_sheet_map[f]])
categories = sorted(set(category_map.values()))
fund_metadata = build_fund_metadata_table(available_funds, fund_sheet_map, fund_data, category_map)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    logo_b64 = file_to_base64(LOGO_FILE)
    if logo_b64:
        st.markdown(f'<div class="sidebar-logo-wrap"><img class="sidebar-logo" src="data:image/png;base64,{logo_b64}" /></div>', unsafe_allow_html=True)
    else:
        st.markdown("### ESG · IPCA")

    st.markdown("### Navegação")
    selected_category = st.selectbox("Categoria APFIPP", ["— Visão geral —"] + categories, index=0)
    if selected_category != "— Visão geral —":
        funds_in_cat = sorted([f for f, c in category_map.items() if c == selected_category])
        selected_fund = st.selectbox("Fundo", ["— Selecionar fundo —"] + funds_in_cat, index=0, key="selected_fund")
    else:
        selected_fund = "— Selecionar fundo —"

    st.markdown("---")
    st.markdown("### Visualização")
    if selected_fund != "— Selecionar fundo —":
        sheet_name = fund_sheet_map[selected_fund]
        var_options = [v for v in ["fund_return", "FLOW", "STDMKT", "STDFND", "TNA", "UP", "DELTA_LNTNA", "DELTA_LOG_ECOSIZE"] if v in fund_data[sheet_name].columns]
        selected_vars = st.multiselect("Variáveis no gráfico", var_options, default=[v for v in ["fund_return", "STDMKT", "FLOW"] if v in var_options])
    else:
        selected_vars = []
    dual_axis = st.checkbox("Usar dois eixos", value=True)
    show_events = st.checkbox("Mostrar eventos históricos", value=False)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">Modelo Markov Regime-Switching para análise da performance de fundos de ações portugueses</div>
        <div class="main-subtitle">André Fangueiro | Projeto Profissional · ESG-IPCA · Análise econométrica da performance por regimes de mercado</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Página inicial neutra: só mostra o detalhe quando uma categoria e um fundo forem escolhidos.
if selected_fund == "— Selecionar fundo —":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        simple_metric("Fundos analisados", str(len(available_funds)))
    with c2:
        simple_metric("Categorias APFIPP", str(len(categories)))
    with c3:
        simple_metric("Modelo", MODEL_LABEL)
    with c4:
        bic_med = compar["BIC"].median() if "BIC" in compar.columns and not compar.empty else np.nan
        simple_metric("BIC mediano", fmt_num(bic_med, 2))

    render_method_intro()

    st.markdown("<div class='section-title'>Fundos, categorias e benchmarks utilizados</div>", unsafe_allow_html=True)
    st.dataframe(fund_metadata, hide_index=True, use_container_width=True)

    render_variables_table()

    st.markdown("<div class='section-title'>Como utilizar a dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='info-box'>
        Para consultar os resultados econométricos, seleciona primeiro a categoria APFIPP no menu lateral e depois o fundo. A página do fundo apresenta a fórmula estimada, matriz de transição, persistência mensal, coeficientes por regime, evolução das variáveis e testes de estacionaridade.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

sheet_name = fund_sheet_map[selected_fund]
selected_df = clean_date_frame(fund_data[sheet_name])
st.markdown(f"<div class='section-title' style='font-size:1.18rem;margin-top:.2rem;'>Fundo selecionado: {selected_fund}</div>", unsafe_allow_html=True)
selected_row = row_for_fund(compar, selected_fund)
selected_coefs = coefs_for_fund(coefs_all, selected_fund)
selected_probs = probabilities_for_fund(smoothed_probs, selected_fund)

# ============================================================
# CARDS RESUMO
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    simple_metric("Fundos analisados", str(len(available_funds)))
with c2:
    simple_metric("Categorias APFIPP", str(len(categories)))
with c3:
    simple_metric("Modelo", MODEL_LABEL)
with c4:
    bic_med = compar["BIC"].median() if "BIC" in compar.columns and not compar.empty else np.nan
    simple_metric("BIC mediano", fmt_num(bic_med, 2))
with c5:
    obs = int(selected_row.get("N_obs", selected_df[DEPENDENT_VARIABLE].notna().sum())) if selected_row is not None else selected_df[DEPENDENT_VARIABLE].notna().sum()
    simple_metric("Observações", str(obs))


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resumo do fundo",
    "Regimes e transições",
    "Coeficientes",
    "Variáveis e timeline",
    "Análise agregada",
    "Metodologia e testes",
])

with tab1:
    st.markdown("<div class='section-title'>Fórmula estimada do modelo final</div>", unsafe_allow_html=True)
    col_l, col_h = st.columns(2)
    with col_l:
        st.markdown("<b>Regime Low-Volatility</b>", unsafe_allow_html=True)
        st.markdown(f"<div class='formula-box'>{build_formula(selected_coefs, 'Low-Volatility')}</div>", unsafe_allow_html=True)
    with col_h:
        st.markdown("<b>Regime High-Volatility</b>", unsafe_allow_html=True)
        st.markdown(f"<div class='formula-box'>{build_formula(selected_coefs, 'High-Volatility')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Interpretação</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-box'>{fund_interpretation(selected_fund, selected_row, selected_coefs)}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Resumo estatístico da rendibilidade</div>", unsafe_allow_html=True)
    if DEPENDENT_VARIABLE in selected_df.columns:
        s = selected_df[DEPENDENT_VARIABLE].dropna()
        stats = pd.DataFrame({
            "Métrica": ["Média", "Mediana", "Mínimo", "Máximo", "Desvio-padrão", "Assimetria", "Curtose"],
            "Valor": [s.mean(), s.median(), s.min(), s.max(), s.std(), s.skew(), s.kurtosis()],
        })
        stats["Valor"] = stats["Valor"].map(lambda x: fmt_num(x, 4))
        st.dataframe(stats, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("<div class='section-title'>Matriz de transição</div>", unsafe_allow_html=True)
    P = transition_matrix(selected_row)
    col_a, col_b = st.columns([1.05, .95])
    with col_a:
        fig = px.imshow(
            P,
            x=["Low", "High"],
            y=["Low", "High"],
            text_auto=".2%",
            aspect="auto",
            labels=dict(x="Regime seguinte", y="Regime atual", color="Probabilidade"),
            color_continuous_scale="Blues",
            zmin=0,
            zmax=1,
        )
        fig.update_layout(height=390, margin=dict(l=20, r=20, t=30, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        render_math_matrix(P)
    with col_b:
        p_ll, p_lh, p_hl, p_hh = P[0,0], P[0,1], P[1,0], P[1,1]
        d_low = selected_row.get("Duracao_Low_meses", np.nan) if selected_row is not None else np.nan
        d_high = selected_row.get("Duracao_High_meses", np.nan) if selected_row is not None else np.nan
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown(f"**Persistência mensal Low:** {fmt_pct(p_ll)}")
        st.markdown(f"**Persistência mensal High:** {fmt_pct(p_hh)}")
        st.markdown(f"**Duração esperada Low:** {fmt_num(d_low,1)} meses")
        st.markdown(f"**Duração esperada High:** {fmt_num(d_high,1)} meses")
        st.markdown("<span class='small-muted'>A persistência mensal corresponde à probabilidade estimada de permanecer no mesmo regime de um mês para o mês seguinte.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Diagrama de transição</div>", unsafe_allow_html=True)
    render_transition_diagram_matlab(P)

    st.markdown("<div class='section-title'>Probabilidades suavizadas dos regimes</div>", unsafe_allow_html=True)
    render_smoothed_probabilities(selected_probs)

with tab3:
    st.markdown("<div class='section-title'>Coeficientes estimados por regime</div>", unsafe_allow_html=True)
    if not selected_coefs.empty:
        display = selected_coefs[["Regime", "Variável", "Coeficiente", "p_value", "Significancia"]].copy()
        display["Coeficiente"] = display["Coeficiente"].map(lambda x: fmt_num(x, 4))
        display["p_value"] = display["p_value"].map(lambda x: fmt_num(x, 4))
        st.dataframe(display, hide_index=True, use_container_width=True)

        plot_df = selected_coefs[selected_coefs["Variável"].isin(["STDMKT", "FLOW", "Intercept"])].copy()
        fig = px.bar(plot_df, x="Variável", y="Coeficiente", color="Regime", barmode="group", text="Significancia", labels={"Coeficiente":"Coeficiente estimado"})
        fig.update_layout(height=410, margin=dict(l=20,r=20,t=25,b=20), legend_title_text="Regime")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Significância por variável no conjunto dos fundos</div>", unsafe_allow_html=True)
    sig = coefs_all[coefs_all["Variável"].isin(["STDMKT", "FLOW", "Intercept"])].copy()
    sig["Significativo"] = sig["Significancia"].apply(signif_bool)
    agg = sig.groupby(["Regime", "Variável"], as_index=False)["Significativo"].sum()
    fig = px.bar(agg, x="Variável", y="Significativo", color="Regime", barmode="group", labels={"Significativo":"N.º de fundos com significância"})
    fig.update_layout(height=360, margin=dict(l=20,r=20,t=25,b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("<div class='section-title'>Evolução das variáveis do fundo</div>", unsafe_allow_html=True)
    if selected_vars:
        dfp = clean_date_frame(selected_df)
        if "date" in dfp.columns:
            dfp["date"] = ensure_datetime_series(dfp["date"])
            dfp = dfp[dfp["date"].notna()].copy()
            dfp = dfp[(dfp["date"].dt.year >= 1990) & (dfp["date"].dt.year <= 2035)].copy()
        if dual_axis and len(selected_vars) >= 2:
            left = selected_vars[0]
            right_vars = selected_vars[1:]
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=dfp["date"], y=dfp[left], name=VARIABLE_LABELS.get(left, left), mode="lines"), secondary_y=False)
            for v in right_vars:
                fig.add_trace(go.Scatter(x=dfp["date"], y=dfp[v], name=VARIABLE_LABELS.get(v, v), mode="lines"), secondary_y=True)
            fig.update_yaxes(title_text=VARIABLE_LABELS.get(left, left), secondary_y=False)
            fig.update_yaxes(title_text=" / ".join([VARIABLE_LABELS.get(v, v) for v in right_vars]), secondary_y=True)
        else:
            fig = go.Figure()
            for v in selected_vars:
                fig.add_trace(go.Scatter(x=dfp["date"], y=dfp[v], name=VARIABLE_LABELS.get(v, v), mode="lines"))
        if show_events:
            event_ref_var = selected_vars[0] if selected_vars else None
            fig = add_event_markers(fig, dfp, event_ref_var)
        fig.update_xaxes(type="date", tickformat="%Y", hoverformat="%m-%Y")
        fig.update_layout(height=520, margin=dict(l=20,r=20,t=25,b=20), legend=dict(orientation="h", y=-.18), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        if show_events:
            render_event_legend()
    else:
        st.info("Seleciona pelo menos uma variável no menu lateral para visualizar a série temporal.")

    st.markdown("<div class='section-title'>Regime timeline</div>", unsafe_allow_html=True)
    render_regime_timeline(selected_probs)

with tab5:
    st.markdown("<div class='section-title'>Análise cross-sectional por categoria APFIPP</div>", unsafe_allow_html=True)
    if not compar.empty:
        agg_cat = compar.groupby("Categoria_APFIPP", as_index=False).agg(
            Fundos=("Fundo", "nunique"),
            BIC_mediano=("BIC", "median"),
            Persistencia_Low=("P(Low→Low)", "mean"),
            Persistencia_High=("P(High→High)", "mean"),
            Duracao_Low_meses=("Duracao_Low_meses", "mean"),
            Duracao_High_meses=("Duracao_High_meses", "mean"),
        )
        st.dataframe(agg_cat, hide_index=True, use_container_width=True)
        fig = px.bar(agg_cat, x="Categoria_APFIPP", y=["Persistencia_Low", "Persistencia_High"], barmode="group", labels={"value":"Persistência média", "variable":"Regime"})
        fig.update_layout(height=420, margin=dict(l=20,r=20,t=25,b=90), xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Variáveis mais relevantes por regime</div>", unsafe_allow_html=True)
    sig2 = coefs_all[coefs_all["Variável"].isin(["STDMKT", "FLOW"])].copy()
    sig2["Significativo"] = sig2["Significancia"].apply(signif_bool)
    rel = sig2.groupby(["Regime", "Variável"], as_index=False).agg(
        Coeficiente_medio=("Coeficiente", "mean"),
        Fundos_significativos=("Significativo", "sum"),
    )
    st.dataframe(rel, hide_index=True, use_container_width=True)

with tab6:
    render_regime_methodology()
    render_variables_table()

    st.markdown("<div class='section-title'>Enquadramento metodológico</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='info-box'>
        A análise estima a performance mensal dos fundos através de um modelo Markov Regime-Switching com dois regimes latentes. O modelo permite que os coeficientes associados às variáveis explicativas sejam diferentes entre regimes, captando alterações na relação entre performance, risco de mercado e fluxos dos fundos ao longo do tempo. A especificação apresentada utiliza <b>fund_return</b> como variável dependente e <b>STDMKT</b> e <b>FLOW</b> como variáveis explicativas.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Testes de estacionaridade</div>", unsafe_allow_html=True)
    if not stationarity_summary.empty:
        st.dataframe(stationarity_summary, hide_index=True, use_container_width=True)
    else:
        st.text(stationarity_text[:5000])

    st.markdown("<div class='section-title'>Notas para a próxima estimação</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='info-box'>
        As probabilidades suavizadas permitem acompanhar, mês a mês, a probabilidade estimada de cada fundo estar em regime Low-Volatility ou High-Volatility. Esta componente torna visível a dinâmica temporal dos regimes e aproxima a apresentação dos resultados ao formato usado em aplicações econométricas de mudança de regime.
        </div>
        """,
        unsafe_allow_html=True,
    )

