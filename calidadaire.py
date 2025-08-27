# app.py
# Visor mensual 2x2 con bandas ICCA (PM1/PM2.5/PM10, NO2, SO2, O3)
# Fuente: carpeta de GitHub (Contents API -> fallback HTML -> raw URLs)
# Requisitos: streamlit, pandas, plotly, requests, python-dateutil (implícito)

import io
import os
import re
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =================== CONFIG POR DEFECTO ===================
DEFAULT_GITHUB_OWNER = "danipoleo"
DEFAULT_GITHUB_REPO  = "calidadaire"
DEFAULT_PATH_IN_REPO = "datos"
DEFAULT_BRANCH       = "main"
# ==========================================================

# ------------------ Mapeo de estaciones -------------------
STATION_MAP: Dict[str, str] = {
    "1637": "San José",
    "1653": "Cigefi",
    "1735": "Cartago TEC",
    "1762": "Santa Lucía",
    "1775": "Belén CTP",
    "1776": "Fabio Baudrit",
    "Z01777": "Municipalidad de Santa Ana",
}

# Columnas posibles (nombres varían por exporte)
COL_CANDIDATES: Dict[str, List[str]] = {
    "PM1":   ["PM1 Concentration", "PM1"],
    "PM25":  ["PM2.5 Concentration", "PM2.5", "PM2_5", "PM25"],
    "PM10":  ["PM10 Concentration", "PM10"],
    "NO2":   ["NO2 Concentration", "NO2", "NO₂"],
    "SO2":   ["SO2 Concentration", "SO2", "SO₂"],
    "O3":    ["O3 Concentration",  "O3", "Ozone", "O₃"],
    "CO":    ["CO Concentration",  "CO"],
    "DATE":  ["Date (Local)", "Date", "Fecha"],
    "TIME":  ["Time (Local)", "Time", "Hora"],
    "UTC":   ["UTC Time Stamp", "UTC", "Timestamp", "Datetime", "DateTime"],
    # Columnas de unidades (si existen)
    "NO2_UNIT": ["NO2 Unit", "NO₂ Unit", "NO2_Unit", "NO2 units"],
    "SO2_UNIT": ["SO2 Unit", "SO₂ Unit", "SO2_Unit", "SO2 units"],
    "O3_UNIT":  ["O3 Unit",  "O₃ Unit",  "O3_Unit",  "O3 units"],
    "CO_UNIT":  ["CO Unit", "CO_Unit", "CO units"],
    "PM1_UNIT": ["PM1 Unit", "PM1 units"],
    "PM25_UNIT":["PM2.5 Unit", "PM2_5 Unit", "PM25 Unit", "PM2.5 units"],
    "PM10_UNIT":["PM10 Unit", "PM10 units"],
}

# ============ Regex y parser del nombre de archivo ============
FILENAME_REGEX = re.compile(
    r"^device_(?P<sid>[A-Za-z0-9]+)_(?P<start>\d{12})_(?P<end>\d{12})_1hr\.csv$"
)

def parse_from_filename(name: str):
    """
    Extrae sid, dt_start, dt_end, year, month desde:
    'device_<sid>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>_1hr.csv'.
    Si no coincide, devuelve claves con None.
    """
    m = FILENAME_REGEX.match(name or "")
    if not m:
        return {"sid": None, "dt_start": None, "dt_end": None, "year": None, "month": None}
    sid = m.group("sid")
    start = m.group("start")
    end   = m.group("end")
    try:
        dt_start = datetime.strptime(start, "%Y%m%d%H%M")
        dt_end   = datetime.strptime(end,   "%Y%m%d%H%M")
        return {
            "sid": sid,
            "dt_start": dt_start,
            "dt_end": dt_end,
            "year": dt_start.year,
            "month": dt_start.month
        }
    except Exception:
        return {"sid": sid, "dt_start": None, "dt_end": None, "year": None, "month": None}
# =============================================================

# ---------- ICCA (rangos base) ----------
ICCA: Dict[str, List[Tuple[str, str, float, float]]] = {
    "PM10": [
        ("Verde",      "#00A65A", 0,    60),
        ("Amarillo",   "#FFC107", 61,   100),
        ("Anaranjado", "#FF9800", 101,  200),
        ("Rojo",       "#E53935", 201,  250),
        ("Púrpura",    "#8E24AA", 250,  math.inf),
    ],
    "PM2.5": [
        ("Verde",      "#00A65A", 0,    15),
        ("Amarillo",   "#FFC107", 15.1, 40),
        ("Anaranjado", "#FF9800", 40.1, 65),
        ("Rojo",       "#E53935", 66,   100),
        ("Púrpura",    "#8E24AA", 100,  math.inf),
    ],
    "NO2_ppm": [
        ("Verde",      "#00A65A", 0,     0.105),
        ("Amarillo",   "#FFC107", 0.106, 0.210),
        ("Anaranjado", "#FF9800", 0.211, 0.315),
        ("Rojo",       "#E53935", 0.316, 0.420),
        ("Púrpura",    "#8E24AA", 0.420, math.inf),
    ],
    "O3_ppm": [
        ("Verde",      "#00A65A", 0,     0.055),
        ("Amarillo",   "#FFC107", 0.056, 0.110),
        ("Anaranjado", "#FF9800", 0.111, 0.165),
        ("Rojo",       "#E53935", 0.166, 0.220),
        ("Púrpura",    "#8E24AA", 0.220, math.inf),
    ],
    "SO2_ppm": [
        ("Verde",      "#00A65A", 0,     0.065),
        ("Amarillo",   "#FFC107", 0.066, 0.130),
        ("Anaranjado", "#FF9800", 0.131, 0.195),
        ("Rojo",       "#E53935", 0.196, 0.260),
        ("Púrpura",    "#8E24AA", 0.260,  math.inf),
    ],
    "CO_ppm": [
        ("Verde",      "#00A65A", 0,     5.50),
        ("Amarillo",   "#FFC107", 5.51,  11.0),
        ("Anaranjado", "#FF9800", 11.01, 16.5),
        ("Rojo",       "#E53935", 16.51, 22.0),
        ("Púrpura",    "#8E24AA", 22.0,  math.inf),
    ],
}
MW = {"NO2": 46.01, "O3": 48.00, "SO2": 64.07, "CO": 28.01}

# ========================== Utilidades ==========================

def _gh_headers() -> Dict[str, str]:
    """Headers opcionales con token para evitar rate-limits en GitHub (Streamlit Secrets o env)."""
    token = st.secrets.get("GITHUB_TOKEN", None) or os.environ.get("GITHUB_TOKEN", None)
    return {"Authorization": f"Bearer {token}"} if token else {}

def find_column(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    # Coincidencia exacta
    for candidate in options:
        for col in df.columns:
            if col.strip().lower() == candidate.strip().lower():
                return col
    # Coincidencia parcial
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in options:
        c = candidate.strip().lower()
        for lc, real in lower_cols.items():
            if c in lc:
                return real
    return None

def _clean_earthsense_text(txt: str) -> str:
    """
    Devuelve SOLO la parte útil del CSV: desde la fila de encabezados reales
    ('Date (Local)', 'UTC Time Stamp', ...) hacia abajo.
    También descarta líneas sueltas/rotas que no tengan suficientes comas.
    """
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    if txt.startswith("\ufeff"):
        txt = txt.lstrip("\ufeff")

    lines = txt.split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        l = line.strip('"').lower()
        if "date (local)" in l and "utc time stamp" in l:
            header_idx = i
            break

    if header_idx is None:
        return txt  # no encontré cabecera — leer tal cual

    useful = lines[header_idx:]
    filtered = [ln for ln in useful if ln.count(",") >= 5 or ln.strip().lower().startswith("date (local)")]
    return "\n".join(filtered)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    """
    Descarga y lee un CSV tipo EarthSense con metadatos arriba.
    Usa engine='python' y on_bad_lines='skip' para tolerar líneas raras.
    """
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    # decodificar como texto
    try:
        txt = r.content.decode("utf-8")
    except UnicodeDecodeError:
        txt = r.content.decode("latin-1", errors="replace")

    clean_txt = _clean_earthsense_text(txt)

    # Intento principal: separador ','
    try:
        return pd.read_csv(
            io.StringIO(clean_txt),
            engine="python",
            sep=",",
            quotechar='"',
            on_bad_lines="skip",
            na_values=["N/A", "NA", "", "null", "None"],
        )
    except Exception:
        # Alterno: separador ';'
        return pd.read_csv(
            io.StringIO(clean_txt),
            engine="python",
            sep=";",
            quotechar='"',
            on_bad_lines="skip",
            na_values=["N/A", "NA", "", "null", "None"],
        )

def _list_with_api(owner: str, repo: str, path: str, branch: str):
    """Devuelve lista con la Contents API o [] si no se puede (no levanta excepción)."""
    url_api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = _gh_headers()
    try:
        r = requests.get(url_api, headers=headers, timeout=30)
        if r.status_code == 403:
            return []  # rate limit -> sin excepciones
        r.raise_for_status()
        items = r.json()
        if isinstance(items, dict) and items.get("type") == "file":
            items = [items]
        return [
            {
                "name": it["name"],
                "download_url": it.get("download_url"),
                "path": it.get("path"),
            }
            for it in items
            if it.get("type") == "file" and it.get("name", "").lower().endswith(".csv")
        ]
    except Exception:
        return []  # cualquier error -> [] y que el caller pruebe el HTML

def _list_with_html(owner: str, repo: str, path: str, branch: str):
    """Parsea la página HTML pública de GitHub para listar CSV (sin API)."""
    url_html = f"https://github.com/{owner}/{repo}/tree/{branch}/{path}".rstrip("/")
    try:
        rh = requests.get(url_html, timeout=30)
        if rh.status_code >= 400:
            return []
        html = rh.text
        path_prefix = (path.strip("/") + "/") if path.strip("/") else ""
        pattern = re.compile(
            rf'href="/{re.escape(owner)}/{re.escape(repo)}/blob/{re.escape(branch)}/({re.escape(path_prefix)}[^"<>]+?\.csv)"'
        )
        matches = pattern.findall(html)
        files, seen = [], set()
        for rel in matches:
            if rel in seen:
                continue
            seen.add(rel)
            name = rel.split("/")[-1]
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}"
            files.append({"name": name, "download_url": raw, "path": rel})
        return files
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def github_list_files(owner: str, repo: str, path: str, branch: str, force_html: bool = False):
    """
    Lista CSV en la carpeta usando:
    - API (si no hay rate limit) salvo que force_html=True
    - Fallback HTML si la API no devuelve nada o se pide forzar HTML
    Nunca levanta excepción; devuelve [] si no encontró nada.
    """
    files = []
    if not force_html:
        files = _list_with_api(owner, repo, path, branch)
    if not files:
        files = _list_with_html(owner, repo, path, branch)
    return files

def raw_url(owner: str, repo: str, path: str, branch: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

def unit_from_column(df: pd.DataFrame, unit_col: Optional[str], default: str) -> str:
    if unit_col and unit_col in df.columns:
        val = df[unit_col].dropna().astype(str)
        if len(val):
            return val.iloc[0].strip()
    return default

def convert_icca_ranges_ppm(ranges_ppm, target_unit, mw):
    tu = (target_unit or "").lower().replace("ug/m3","µg/m³").replace("ug/m³","µg/m³")
    out = []
    for label, color, lo, hi in ranges_ppm:
        if tu in ["ppm"]:
            lo2, hi2 = lo, hi
        elif tu in ["µg/m³", "μg/m³"]:
            factor = mw * 1000.0 / 24.45
            lo2 = 0 if (lo == 0) else (None if lo is None else lo*factor)
            hi2 = math.inf if hi is None or math.isinf(hi) else hi*factor
        elif tu in ["mg/m³", "mg/m3"]:
            factor = mw / 24.45
            lo2 = 0 if (lo == 0) else (None if lo is None else lo*factor)
            hi2 = math.inf if hi is None or math.isinf(hi) else hi*factor
        else:
            lo2, hi2 = lo, hi
        out.append((label, color, lo2, hi2))
    return out

def make_bands(x0, x1, ranges, opacity=0.12):
    shapes, lines = [], []
    for label, color, lo, hi in ranges:
        y0 = -math.inf if lo is None else lo
        y1 =  math.inf if (hi is None or math.isinf(hi)) else hi
        shapes.append(dict(type="rect", xref="x", yref="y",
                           x0=x0, x1=x1, y0=y0, y1=y1,
                           fillcolor=color, opacity=opacity, layer="below", line=dict(width=0)))
        if hi is not None and not math.isinf(hi):
            lines.append(dict(type="line", xref="x", yref="y",
                              x0=x0, x1=x1, y0=hi, y1=hi,
                              line=dict(color=color, width=1, dash="dot"), layer="below"))
    return shapes, lines

def make_line_figure(x, y, title, color, y_title, icca_ranges=None, show_icca=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color)))
    fig.update_layout(
        title=title, template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
        xaxis_title="Fecha/Hora", yaxis_title=y_title,
        height=350, legend_title_text=None
    )
    if show_icca and icca_ranges:
        x0, x1 = (pd.Series(x).min(), pd.Series(x).max())
        shapes, lines = make_bands(x0, x1, icca_ranges, opacity=0.12)
        fig.update_layout(shapes=shapes + lines)
    return fig

def make_pm_figure(df, pm_map, pm_band_choice, show_icca=True):
    fig = go.Figure()
    for _, label, color in pm_map:
        lw = 2.5 if label == "PM2.5" else 2
        fig.add_trace(go.Scatter(x=df["dt"], y=df[label], name=label, mode="lines",
                                 line=dict(color=color, width=lw)))
    if show_icca:
        x0, x1 = df["dt"].min(), df["dt"].max()
        icca_ranges = ICCA["PM2.5"] if pm_band_choice == "PM2.5" else ICCA["PM10"]
        shapes, lines = make_bands(x0, x1, icca_ranges, opacity=0.12)
        fig.update_layout(shapes=shapes + lines)

    fig.update_layout(
        title=f"Particulados — PM1 / PM2.5 / PM10 (Bandas ICCA: {pm_band_choice})",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
        xaxis_title="Fecha/Hora",
        yaxis_title="Concentración (µg/m³)",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig

# =============================== UI / APP ===============================
st.set_page_config(page_title="Visor mensual de calidad de aire (GitHub)", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; margin: 0 auto;}
h1, h2, h3 {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("Visor mensual de calidad de aire — Fuente: GitHub")
st.caption("Carga automática desde la carpeta del repositorio y visualización 2×2 con bandas ICCA.")

with st.sidebar:
    st.header("Fuente de datos (GitHub)")
    owner  = st.text_input("Owner", value=DEFAULT_GITHUB_OWNER)
    repo   = st.text_input("Repo",  value=DEFAULT_GITHUB_REPO)
    path   = st.text_input("Ruta en el repo", value=DEFAULT_PATH_IN_REPO)
    branch = st.text_input("Branch", value=DEFAULT_BRANCH)

    st.divider()
    show_icca = st.checkbox("Mostrar bandas ICCA", value=True)
    pm_band_choice = st.radio("Umbral para fondo de particulados", ["PM2.5","PM10"], index=0)

    st.caption("Opciones avanzadas:")
    force_html = st.checkbox("Forzar fallback (HTML)", value=False)
    if st.button("Recargar / limpiar caché"):
        st.cache_data.clear()
        st.success("Caché limpiada. Vuelve a seleccionar opciones si es necesario.")

    st.caption("También puedes pegar la URL de carpeta (opcional):")
    gh_url = st.text_input("URL de carpeta (opcional)", value="", placeholder="https://github.com/owner/repo/tree/branch/carpeta")
    if gh_url.strip():
        try:
            parts = gh_url.strip().split("github.com/")[-1].split("/")
            owner = parts[0]; repo = parts[1]
            idx_tree = parts.index("tree") if "tree" in parts else -1
            if idx_tree != -1 and len(parts) > idx_tree+1:
                branch = parts[idx_tree+1]
                path = "/".join(parts[idx_tree+2:]) if len(parts) > idx_tree+2 else ""
        except Exception:
            st.warning("No pude interpretar la URL; usando los campos owner/repo/ruta/branch.")

# --- Listar archivos CSV (API -> fallback HTML, sin excepciones) ---
files = github_list_files(owner, repo, path, branch, force_html=force_html)

if not files:
    st.error(
        "No encontré CSV en esa carpeta del repositorio.\n"
        "Si ves límites de GitHub, prueba 'Forzar fallback (HTML)' o agrega un GITHUB_TOKEN en Secrets."
    )
    st.stop()

# Construir índice e intentar parsear año/mes del nombre
rows = []
for f in files:
    fname = f.get("name") or ""
    if not fname:
        continue
    meta = parse_from_filename(fname)
    sid = meta["sid"]
    station = STATION_MAP.get(sid, sid if sid else "Desconocida")
    rows.append({
        "name": fname,
        "url": f.get("download_url") or f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{f.get('path','')}",
        "sid": sid,
        "station": station,
        "dt_start": meta["dt_start"],
        "dt_end": meta["dt_end"],
        "year": meta["year"],
        "month": meta["month"],
    })
files_df = pd.DataFrame(rows)

# Si no pudimos extraer año/mes desde el nombre, inferir desde el CSV
if files_df["year"].isna().any() or files_df["month"].isna().any():
    need_rows = files_df[files_df["year"].isna() | files_df["month"].isna()]
    for i, r in need_rows.iterrows():
        try:
            tmp = fetch_csv(r["url"])
            col_utc  = find_column(tmp, COL_CANDIDATES["UTC"])
            col_date = find_column(tmp, COL_CANDIDATES["DATE"])
            col_time = find_column(tmp, COL_CANDIDATES["TIME"])
            if col_utc is not None:
                dt = pd.to_datetime(tmp[col_utc], errors="coerce", utc=True).dt.tz_convert(None)
            elif (col_date is not None) and (col_time is not None):
                dt = pd.to_datetime(tmp[col_date].astype(str)+" "+tmp[col_time].astype(str),
                                    errors="coerce", dayfirst=True)
            else:
                dt = pd.to_datetime(pd.NaT)
            dt = pd.Series(dt).dropna()
            if not dt.empty:
                files_df.at[i, "dt_start"] = dt.min()
                files_df.at[i, "dt_end"]   = dt.max()
                files_df.at[i, "year"]     = int(dt.min().year)
                files_df.at[i, "month"]    = int(dt.min().month)
        except Exception:
            pass

# Controles (Año/Mes/Estación)
valid_years = sorted([int(y) for y in files_df["year"].dropna().unique()], reverse=True)
if not valid_years:
    st.error("No pude resolver año/mes de los archivos. Revisa el patrón del nombre o las columnas de tiempo.")
    st.stop()

year_sel = st.selectbox("Año", valid_years)
month_names = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
               7:"Julio",8:"Agosto",9:"Setiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
months = sorted(files_df.loc[files_df["year"]==year_sel, "month"].dropna().unique())
month_sel = st.selectbox("Mes", months, format_func=lambda m: f"{month_names.get(int(m), m)} ({int(m):02d})")

subset = files_df[(files_df["year"]==year_sel) & (files_df["month"]==month_sel)]
if subset.empty:
    st.warning("No hay archivos para ese año/mes.")
    st.stop()

# Selector de estación por etiqueta amigable
subset = subset.copy()
subset["label"] = subset.apply(
    lambda r: f"{(r['station'] or 'Desconocida')} ({r['sid']})" if pd.notna(r["sid"]) else r["name"], axis=1
)
station_label = st.selectbox("Estación", list(subset["label"]))
row = subset[subset["label"] == station_label].iloc[0]

st.caption(f"Archivo: `{row['name']}`  ·  Estación: **{row['station']}**  ·  Periodo: {row['dt_start']} → {row['dt_end']}")

# Cargar CSV elegido
try:
    df_raw = fetch_csv(row["url"])
except Exception as e:
    st.error(f"No pude descargar/leer el CSV: {row['url']}\nDetalle: {e}")
    st.stop()

# Resolver fecha/hora
col_utc  = find_column(df_raw, COL_CANDIDATES["UTC"])
col_date = find_column(df_raw, COL_CANDIDATES["DATE"])
col_time = find_column(df_raw, COL_CANDIDATES["TIME"])
if col_utc is not None:
    dt = pd.to_datetime(df_raw[col_utc], errors="coerce", utc=True).dt.tz_convert(None)
elif (col_date is not None) and (col_time is not None):
    dt = pd.to_datetime(df_raw[col_date].astype(str)+" "+df_raw[col_time].astype(str),
                        errors="coerce", dayfirst=True)
else:
    st.error("No encontré columnas de tiempo (UTC o Date+Time).")
    st.stop()

# Mapear columnas de interés
mapped = {k: find_column(df_raw, COL_CANDIDATES[k]) for k in ["PM1","PM25","PM10","NO2","SO2","O3","CO"]}
unit_cols = {
    "NO2": find_column(df_raw, COL_CANDIDATES["NO2_UNIT"]),
    "SO2": find_column(df_raw, COL_CANDIDATES["SO2_UNIT"]),
    "O3":  find_column(df_raw, COL_CANDIDATES["O3_UNIT"]),
    "CO":  find_column(df_raw, COL_CANDIDATES["CO_UNIT"]),
    "PM1": find_column(df_raw, COL_CANDIDATES["PM1_UNIT"]),
    "PM25":find_column(df_raw, COL_CANDIDATES["PM25_UNIT"]),
    "PM10":find_column(df_raw, COL_CANDIDATES["PM10_UNIT"]),
}

no2_col = mapped.get("NO2")
so2_col = mapped.get("SO2")
o3_col  = mapped.get("O3")

gases_missing = [name for name, col in {"NO₂": no2_col, "SO₂": so2_col, "O₃": o3_col}.items() if col is None]
if gases_missing:
    st.error("El CSV no contiene: " + ", ".join(gases_missing))
    st.stop()

pm_map  = []
if mapped.get("PM1"):  pm_map.append((mapped["PM1"],  "PM1",   "#8ECAE6"))
if mapped.get("PM25"): pm_map.append((mapped["PM25"], "PM2.5", "#FB8500"))
if mapped.get("PM10"): pm_map.append((mapped["PM10"], "PM10",  "#219EBC"))
if len(pm_map) == 0:
    st.error("No se encontraron columnas de particulado (PM1/PM2.5/PM10) en el CSV.")
    st.stop()

# Detectar unidades (defaults razonables si no hay columnas de unidad)
no2_unit = unit_from_column(df_raw, unit_cols.get("NO2"), "µg/m³")
o3_unit  = unit_from_column(df_raw, unit_cols.get("O3"),  "µg/m³")
so2_unit = unit_from_column(df_raw, unit_cols.get("SO2"), "µg/m³")
co_unit  = unit_from_column(df_raw, unit_cols.get("CO"),  "mg/m³")

# DataFrame final
data = {
    "dt": dt,
    "NO₂": pd.to_numeric(df_raw[no2_col], errors="coerce"),
    "SO₂": pd.to_numeric(df_raw[so2_col], errors="coerce"),
    "O₃":  pd.to_numeric(df_raw[o3_col],  errors="coerce")
}
for col, label, _ in pm_map:
    data[label] = pd.to_numeric(df_raw[col], errors="coerce")
df = pd.DataFrame(data).sort_values("dt").reset_index(drop=True)

# ---- Layout centrado: grilla 2x2 ----
sp_left, center_col, sp_right = st.columns([0.1, 0.8, 0.1])
with center_col:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(make_pm_figure(df, pm_map, pm_band_choice, show_icca), use_container_width=True)

    with c2:
        no2_icca = convert_icca_ranges_ppm(ICCA["NO2_ppm"], no2_unit, MW["NO2"])
        st.plotly_chart(
            make_line_figure(df["dt"], df["NO₂"], f"Dióxido de Nitrógeno — NO₂ ({no2_unit})",
                             "#2A9D8F", f"Concentración ({no2_unit})",
                             icca_ranges=no2_icca, show_icca=show_icca),
            use_container_width=True
        )

    c3, c4 = st.columns(2, gap="large")
    with c3:
        so2_icca = convert_icca_ranges_ppm(ICCA["SO2_ppm"], so2_unit, MW["SO2"])
        st.plotly_chart(
            make_line_figure(df["dt"], df["SO₂"], f"Dióxido de Azufre — SO₂ ({so2_unit})",
                             "#E76F51", f"Concentración ({so2_unit})",
                             icca_ranges=so2_icca, show_icca=show_icca),
            use_container_width=True
        )
    with c4:
        o3_icca = convert_icca_ranges_ppm(ICCA["O3_ppm"], o3_unit, MW["O3"])
        st.plotly_chart(
            make_line_figure(df["dt"], df["O₃"], f"Ozono — O₃ ({o3_unit})",
                             "#264653", f"Concentración ({o3_unit})",
                             icca_ranges=o3_icca, show_icca=show_icca),
            use_container_width=True
        )

# Estadísticas rápidas
with st.expander("Estadísticas del mes (valores originales)"):
    cols_for_stats = [lab for _, lab, _ in pm_map] + ["NO₂","SO₂","O₃"]
    st.dataframe(
        df[cols_for_stats].describe().T.rename(columns={
            "mean":"media", "std":"desv.std", "min":"mín", "max":"máx"
        })[["count","media","desv.std","mín","25%","50%","75%","máx"]]
    )

# Descarga del CSV limpio
safe_year = int(subset.iloc[0]['year']) if pd.notna(subset.iloc[0]['year']) else None
safe_month = int(subset.iloc[0]['month']) if pd.notna(subset.iloc[0]['month']) else None
file_stub = f"clean_{row.get('sid') or 'NA'}_{safe_year or 'YYYY'}{(safe_month or 0):02d if safe_month else 'MM'}"

st.download_button(
    "Descargar CSV limpio (este archivo)",
    data=df.to_csv(index=False),
    file_name=f"{file_stub}.csv",
    mime="text/csv"
)

with st.expander("Ayuda / Supuestos"):
    st.markdown(f"""
- Para evitar límites, agrega **GITHUB_TOKEN** en *Settings → Secrets* (PAT de solo lectura).  
- Si ves rate limit, usa **“Forzar fallback (HTML)”** o pulsa **“Recargar / limpiar caché”**.
- Limpieza EarthSense: se descartan metadatos superiores hasta la fila con **Date (Local)** y **UTC Time Stamp**.
- Detección de tiempo: **UTC Time Stamp** si existe; si no, **Date (Local) + Time (Local)** (`dayfirst=True`).
- **Bandas ICCA**: particulados (PM10/PM2.5) en µg/m³; gases (NO₂/O₃/SO₂) a ppm convertidos a la unidad del CSV.
- Conversión (25°C, 1 atm): µg/m³ = ppm × MW × 1000 / 24.45; mg/m³ = ppm × MW / 24.45.
""")
