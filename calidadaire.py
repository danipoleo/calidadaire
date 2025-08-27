# app.py
# Visor mensual 2x2 con bandas ICCA (PM1/PM2.5/PM10, NO2, SO2, O3)
# Robusto contra rate limits: PRIORIDAD ZIP (codeload.github.com) -> HTML -> API
# Lee TODOS los device_*_1hr.csv desde la carpeta indicada; si está vacía, auto-descubre en todo el repo

import io
import os
import re
import math
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =================== CONFIG POR DEFECTO ===================
DEFAULT_GITHUB_OWNER = "danipoleo"
DEFAULT_GITHUB_REPO  = "calidadaire"
DEFAULT_PATH_IN_REPO = "datos"       # ⇦ puedes poner "datosCA" si quieres por defecto
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
    "1777": "Municipalidad de Santa Ana",
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
    m = FILENAME_REGEX.match(name or "")
    if not m:
        return {"sid": None, "dt_start": None, "dt_end": None, "year": None, "month": None}
    sid = m.group("sid"); start = m.group("start"); end = m.group("end")
    try:
        dt_start = datetime.strptime(start, "%Y%m%d%H%M")
        dt_end   = datetime.strptime(end,   "%Y%m%d%H%M")
        return {"sid": sid, "dt_start": dt_start, "dt_end": dt_end, "year": dt_start.year, "month": dt_start.month}
    except Exception:
        return {"sid": sid, "dt_start": None, "dt_end": None, "year": None, "month": None}

# ---------- ICCA (rangos base) ----------
ICCA: Dict[str, List[Tuple[str, str, float, float]]] = {
    "PM10": [("Verde","#00A65A",0,60),("Amarillo","#FFC107",61,100),("Anaranjado","#FF9800",101,200),("Rojo","#E53935",201,250),("Púrpura","#8E24AA",250,math.inf)],
    "PM2.5":[("Verde","#00A65A",0,15),("Amarillo","#FFC107",15.1,40),("Anaranjado","#FF9800",40.1,65),("Rojo","#E53935",66,100),("Púrpura","#8E24AA",100,math.inf)],
    "NO2_ppm":[("Verde","#00A65A",0,0.105),("Amarillo","#FFC107",0.106,0.210),("Anaranjado","#FF9800",0.211,0.315),("Rojo","#E53935",0.316,0.420),("Púrpura","#8E24AA",0.420,math.inf)],
    "O3_ppm":[("Verde","#00A65A",0,0.055),("Amarillo","#FFC107",0.056,0.110),("Anaranjado","#FF9800",0.111,0.165),("Rojo","#E53935",0.166,0.220),("Púrpura","#8E24AA",0.220,math.inf)],
    "SO2_ppm":[("Verde","#00A65A",0,0.065),("Amarillo","#FFC107",0.066,0.130),("Anaranjado","#FF9800",0.131,0.195),("Rojo","#E53935",0.196,0.260),("Púrpura","#8E24AA",0.260,math.inf)],
    "CO_ppm":[("Verde","#00A65A",0,5.50),("Amarillo","#FFC107",5.51,11.0),("Anaranjado","#FF9800",11.01,16.5),("Rojo","#E53935",16.51,22.0),("Púrpura","#8E24AA",22.0,math.inf)],
}
MW = {"NO2": 46.01, "O3": 48.00, "SO2": 64.07, "CO": 28.01}

# ========================== Utilidades ==========================
def _gh_headers() -> Dict[str, str]:
    token = st.secrets.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}

def find_column(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for candidate in options:
        for col in df.columns:
            if col.strip().lower() == candidate.strip().lower():
                return col
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in options:
        c = candidate.strip().lower()
        for lc, real in lower_cols.items():
            if c in lc:
                return real
    return None

def _clean_earthsense_text(txt: str) -> str:
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    if txt.startswith("\ufeff"):  # BOM
        txt = txt.lstrip("\ufeff")
    lines = txt.split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        l = line.strip('"').lower()
        if "date (local)" in l and "utc time stamp" in l:
            header_idx = i; break
    if header_idx is None:
        return txt
    useful = lines[header_idx:]
    filtered = [ln for ln in useful if ln.count(",") >= 5 or ln.strip().lower().startswith("date (local)")]
    return "\n".join(filtered)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_csv_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
    # Lee CSV desde bytes (evita nuevas llamadas a Internet)
    try:
        txt = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw_bytes.decode("latin-1", errors="replace")
    clean_txt = _clean_earthsense_text(txt)
    try:
        return pd.read_csv(io.StringIO(clean_txt), engine="python", sep=",", quotechar='"',
                           on_bad_lines="skip", na_values=["N/A","NA","","null","None"])
    except Exception:
        return pd.read_csv(io.StringIO(clean_txt), engine="python", sep=";", quotechar='"',
                           on_bad_lines="skip", na_values=["N/A","NA","","null","None"])

@st.cache_data(ttl=300, show_spinner=False)
def fetch_csv_http(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    return fetch_csv_from_bytes(r.content)

# -------- ZIP (PRIORIDAD) --------
@st.cache_data(ttl=1800, show_spinner=False)
def download_repo_zip(owner: str, repo: str, branch: str) -> bytes:
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    r = requests.get(zip_url, timeout=60)
    r.raise_for_status()
    return r.content

def list_csvs_from_zip(zip_bytes: bytes, path: str, branch: str) -> List[dict]:
    # Devuelve archivos .csv bajo "path" (si path vacío, busca en todo el repo)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    root_prefix = None
    # El zip expande como <repo>-<branch>/
    for name in zf.namelist():
        if name.endswith("/") and name.count("/") == 1:
            root_prefix = name  # ej: calidadaire-main/
            break
    if not root_prefix:
        root_prefix = f"{DEFAULT_GITHUB_REPO}-{branch}/"  # fallback

    rel_folder = path.strip("/")
    # Si path vacío, escanear todo el repo
    wanted_prefix = (root_prefix + rel_folder + "/") if rel_folder else root_prefix

    rows = []
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        if rel_folder:
            if not name.startswith(wanted_prefix):
                continue
        # Solo nombres device_*_1hr.csv (pero si quieres todos los .csv, comenta este if)
        base = name.split("/")[-1]
        if not FILENAME_REGEX.match(base):
            continue
        rel = name[len(root_prefix):]  # ruta relativa dentro del repo
        rows.append({
            "name": base,
            "path": rel,  # p.ej: datos/device_...csv
            "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}",
            "source": "zip"
        })

    return rows

def autodiscover_device_csvs_zip(zip_bytes: bytes, branch: str) -> List[dict]:
    # Busca device_* en TODO el repo (por si pusiste path mal o vacío)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    root_prefix = None
    for name in zf.namelist():
        if name.endswith("/") and name.count("/") == 1:
            root_prefix = name
            break
    if not root_prefix:
        return []
    rows = []
    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue
        base = name.split("/")[-1]
        if not FILENAME_REGEX.match(base):
            continue
        rel = name[len(root_prefix):]
        rows.append({
            "name": base,
            "path": rel,
            "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}",
            "source": "zip_auto"
        })
    return rows

# -------- HTML (fallback 1) --------
def _normalize_rel_for_raw(rel: str, branch: str) -> str:
    if not rel: return rel
    rel = rel.lstrip("/")
    if rel.startswith("refs/heads/"):
        rel = rel[len("refs/heads/"):]
    prefix = f"{branch}/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return rel

def _extract_csv_links(html: str) -> List[str]:
    return re.findall(r'href="([^"]+?/blob/[^"]+?\.csv[^"]*)"', html, flags=re.IGNORECASE)

def list_with_html(owner: str, repo: str, path: str, branch: str, plain: bool):
    url = f"https://github.com/{owner}/{repo}/tree/{branch}/{path}".rstrip("/")
    if plain:
        url += "?plain=1"
    r = requests.get(url, timeout=30)
    if r.status_code >= 400:
        return [], {"status": r.status_code, "where": "html_plain" if plain else "html"}
    hrefs = _extract_csv_links(r.text)
    sub = path.strip("/")
    hrefs = [h for h in hrefs if (not sub) or (f"/{sub}/" in h) or h.endswith("/"+sub)]
    files, seen = [], set()
    for h in hrefs:
        rel = h.split("/blob/", 1)[-1].split("?", 1)[0]
        rel = _normalize_rel_for_raw(rel, branch)
        name = rel.split("/")[-1]
        if not FILENAME_REGEX.match(name or ""):
            continue
        raw  = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}"
        if rel not in seen:
            seen.add(rel)
            files.append({"name": name, "download_url": raw, "path": rel, "source": "html"})
    return files, {"status": r.status_code, "where": "html_plain" if plain else "html", "count": len(files)}

# -------- API (fallback 2) --------
def list_with_git_tree(owner: str, repo: str, branch: str, subpath: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    try:
        r = requests.get(url, headers=_gh_headers(), timeout=30)
        if r.status_code == 403:
            return [], {"status": 403, "where": "git_tree"}
        r.raise_for_status()
        data = r.json()
        tree = data.get("tree", [])
        sub  = subpath.strip("/")
        out  = []
        for node in tree:
            p = node.get("path", "")
            name = p.split("/")[-1]
            if node.get("type") == "blob" and name.lower().endswith(".csv"):
                if not sub or p.startswith(sub + "/") or p == sub:
                    if not FILENAME_REGEX.match(name or ""):
                        continue
                    out.append({"name": name,
                                "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{p}",
                                "path": p, "source": "git_tree"})
        return out, {"status": r.status_code, "where": "git_tree", "count": len(out)}
    except Exception as e:
        return [], {"status": "ERR", "where": "git_tree", "error": str(e)}

def list_with_contents_api(owner: str, repo: str, path: str, branch: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    try:
        r = requests.get(url, headers=_gh_headers(), timeout=30)
        if r.status_code == 403:
            return [], {"status": 403, "where": "contents_api"}
        r.raise_for_status()
        items = r.json()
        if isinstance(items, dict) and items.get("type") == "file":
            items = [items]
        files = []
        for it in items:
            if it.get("type") != "file":
                continue
            name = str(it.get("name",""))
            if name.lower().endswith(".csv") and FILENAME_REGEX.match(name):
                files.append({"name": name, "download_url": it.get("download_url"), "path": it.get("path"), "source": "contents"})
        return files, {"status": r.status_code, "where": "contents_api", "count": len(files)}
    except Exception as e:
        return [], {"status": "ERR", "where": "contents_api", "error": str(e)}

# ================ Conversión y figuras ================
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
    for _, color, lo, hi in ranges:
        y0 = -math.inf if lo is None else lo
        y1 =  math.inf if (hi is None or math.isinf(hi)) else hi
        shapes.append(dict(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1,
                           fillcolor=color, opacity=opacity, layer="below", line=dict(width=0)))
        if hi is not None and not math.isinf(hi):
            lines.append(dict(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=hi, y1=hi,
                              line=dict(color=color, width=1, dash="dot"), layer="below"))
    return shapes, lines

def make_line_figure(x, y, title, color, y_title, icca_ranges=None, show_icca=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color)))
    fig.update_layout(title=title, template="plotly_white",
                      margin=dict(l=20, r=20, t=60, b=30),
                      xaxis_title="Fecha/Hora", yaxis_title=y_title, height=350)
    if show_icca and icca_ranges:
        x0, x1 = (pd.Series(x).min(), pd.Series(x).max())
        shapes, lines = make_bands(x0, x1, icca_ranges, opacity=0.12)
        fig.update_layout(shapes=shapes + lines)
    return fig

def make_pm_figure(df, pm_labels, pm_band_choice, show_icca=True):
    fig = go.Figure()
    color_map = {"PM1":"#8ECAE6","PM2.5":"#FB8500","PM10":"#219EBC"}
    for label in pm_labels:
        if label in df.columns:
            lw = 2.5 if label == "PM2.5" else 2
            fig.add_trace(go.Scatter(x=df["dt"], y=df[label], name=label, mode="lines",
                                     line=dict(color=color_map[label], width=lw)))
    if show_icca and not df.empty:
        x0, x1 = df["dt"].min(), df["dt"].max()
        icca_ranges = ICCA["PM2.5"] if pm_band_choice == "PM2.5" else ICCA["PM10"]
        shapes, lines = make_bands(x0, x1, icca_ranges, opacity=0.12)
        fig.update_layout(shapes=shapes + lines)
    fig.update_layout(title=f"Particulados — PM1 / PM2.5 / PM10 (Bandas ICCA: {pm_band_choice})",
                      template="plotly_white", margin=dict(l=20, r=20, t=60, b=30),
                      xaxis_title="Fecha/Hora", yaxis_title="Concentración (µg/m³)",
                      height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig

# =============================== UI / APP ===============================
st.set_page_config(page_title="Visor mensual de calidad de aire (GitHub ZIP-first)", layout="wide")
st.markdown("""<style>.block-container{max-width:1200px;margin:0 auto;}h1,h2,h3{text-align:center;}</style>""",
            unsafe_allow_html=True)

st.title("Visor mensual — ZIP primero (sin rate limits)")
st.caption("Carga automática desde carpeta del repo o auto-descubre; 2×2 con bandas ICCA.")

with st.sidebar:
    st.header("Fuente de datos (GitHub)")
    owner  = st.text_input("Owner", value=DEFAULT_GITHUB_OWNER)
    repo   = st.text_input("Repo",  value=DEFAULT_GITHUB_REPO)
    path   = st.text_input("Ruta en el repo", value=DEFAULT_PATH_IN_REPO)   # p.ej. datos o datosCA
    branch = st.text_input("Branch", value=DEFAULT_BRANCH)

    prefer_zip = st.checkbox("Preferir ZIP (recomendado)", value=True,
                             help="Usa codeload.github.com (no depende de API ni HTML).")
    allow_html = st.checkbox("Permitir HTML como fallback", value=True)
    allow_api  = st.checkbox("Permitir API como último recurso", value=False,
                             help="Sin GITHUB_TOKEN te puedes topar rate limit.")

    st.divider()
    show_icca = st.checkbox("Mostrar bandas ICCA", value=True)
    pm_band_choice = st.radio("Umbral fondo particulados", ["PM2.5","PM10"], index=0)
    if st.button("Recargar / limpiar caché"):
        st.cache_data.clear()
        st.success("Caché limpiada.")

# =============== LISTADO ROBUSTO ===============
debug = []
files: List[dict] = []

zip_bytes = None
if prefer_zip:
    try:
        zip_bytes = download_repo_zip(owner, repo, branch)
        files = list_csvs_from_zip(zip_bytes, path, branch)
        debug.append({"where":"zip_list", "count":len(files)})
        if not files:
            # auto-descubrir en TODO el repo
            files = autodiscover_device_csvs_zip(zip_bytes, branch)
            debug.append({"where":"zip_autodiscover", "count":len(files)})
    except Exception as e:
        debug.append({"where":"zip_error", "error":str(e)})

if not files and allow_html:
    # HTML plain
    f1, m1 = list_with_html(owner, repo, path, branch, plain=True);  debug.append(m1)
    if not f1:
        f2, m2 = list_with_html(owner, repo, path, branch, plain=False); debug.append(m2)
        files = f2
    else:
        files = f1

if not files and allow_api:
    f3, m3 = list_with_git_tree(owner, repo, branch, path);          debug.append(m3)
    if not f3:
        f4, m4 = list_with_contents_api(owner, repo, path, branch);  debug.append(m4)
        files = f4
    else:
        files = f3

with st.expander("Diagnóstico de listado"):
    st.write({"owner": owner, "repo": repo, "path": path, "branch": branch,
              "prefer_zip": prefer_zip, "allow_html": allow_html, "allow_api": allow_api})
    st.write(debug)

if not files:
    st.error("No encontré CSV con el patrón device_*_1hr.csv en la ruta ni en el repo.")
    st.stop()

# Índice por archivo
rows = []
for f in files:
    fname = f.get("name") or ""
    if not fname or not FILENAME_REGEX.match(fname):
        continue
    meta = parse_from_filename(fname)
    sid = meta["sid"]; station = STATION_MAP.get(sid, sid if sid else "Desconocida")
    rows.append({
        "name": fname,
        "url": f.get("download_url"),
        "path": f.get("path"),
        "sid": sid,
        "station": station,
        "dt_start": meta["dt_start"],
        "dt_end": meta["dt_end"],
        "year": meta["year"],
        "month": meta["month"],
        "source": f.get("source")
    })
files_df = pd.DataFrame(rows)
if files_df.empty:
    st.error("No se pudo construir índice; revisa nombres de archivo.")
    st.stop()

# Controles (Año/Mes/Estación)
valid_years = sorted([int(y) for y in files_df["year"].dropna().unique()], reverse=True)
if not valid_years:
    st.error("No pude resolver año/mes desde los nombres."); st.stop()

year_sel = st.selectbox("Año", valid_years)
month_names = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
               7:"Julio",8:"Agosto",9:"Setiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
months = sorted(files_df.loc[files_df["year"]==year_sel, "month"].dropna().unique())
month_sel = st.selectbox("Mes", months, format_func=lambda m: f"{month_names.get(int(m), m)} ({int(m):02d})")

subset = files_df[(files_df["year"]==year_sel) & (files_df["month"]==month_sel)]
if subset.empty:
    st.warning("No hay archivos para ese año/mes."); st.stop()

subset = subset.copy()
subset["label"] = subset.apply(lambda r: f"{(r['station'] or 'Desconocida')} ({r['sid']})", axis=1)
station_label = st.selectbox("Estación", list(subset["label"]))
row = subset[subset["label"] == station_label].iloc[0]

st.caption(f"Archivo: `{row['name']}` · Estación: **{row['station']}** · Periodo: {row['dt_start']} → {row['dt_end']} · Fuente: {row['source']}")

# =============== CARGA DEL CSV (desde ZIP si lo tenemos; si no, HTTP raw) ===============
def read_selected_csv(row, zip_bytes: Optional[bytes]) -> pd.DataFrame:
    if zip_bytes and row.get("path"):
        # leer directo del zip (más robusto y sin llamadas web)
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        # Buscar entrada exacta dentro del zip: <repo>-<branch>/<row['path']>
        # Averiguar prefijo root del zip:
        root_prefix = None
        for name in zf.namelist():
            if name.endswith("/") and name.count("/") == 1:
                root_prefix = name
                break
        if root_prefix:
            inzip = root_prefix + row["path"]
            if inzip in zf.namelist():
                with zf.open(inzip) as fh:
                    content = fh.read()
                return fetch_csv_from_bytes(content)
    # fallback: HTTP
    return fetch_csv_http(row["url"])

try:
    df_raw = read_selected_csv(row, zip_bytes)
except Exception as e:
    st.error(f"No pude leer el CSV seleccionado.\nDetalle: {e}")
    st.stop()

# =============== PARSEO DE TIEMPO Y MAPEO DE COLUMNAS ===============
col_utc  = find_column(df_raw, COL_CANDIDATES["UTC"])
col_date = find_column(df_raw, COL_CANDIDATES["DATE"])
col_time = find_column(df_raw, COL_CANDIDATES["TIME"])
if col_utc is not None:
    dt = pd.to_datetime(df_raw[col_utc], errors="coerce", utc=True).dt.tz_convert(None)
elif (col_date is not None) and (col_time is not None):
    dt = pd.to_datetime(df_raw[col_date].astype(str)+" "+df_raw[col_time].astype(str),
                        errors="coerce", dayfirst=True)
else:
    st.error("No encontré columnas de tiempo (UTC o Date+Time)."); st.stop()

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
no2_col, so2_col, o3_col = mapped.get("NO2"), mapped.get("SO2"), mapped.get("O3")

def unit_from_column(df, unit_col, default):
    if unit_col and unit_col in df.columns:
        v = df[unit_col].dropna().astype(str)
        if len(v): return v.iloc[0].strip()
    return default

no2_unit = unit_from_column(df_raw, unit_cols.get("NO2"), "µg/m³")
o3_unit  = unit_from_column(df_raw, unit_cols.get("O3"),  "µg/m³")
so2_unit = unit_from_column(df_raw, unit_cols.get("SO2"), "µg/m³")

def to_numeric(series): return pd.to_numeric(series, errors="coerce")

data = {"dt": dt}
pm_labels = []
if mapped.get("PM1"):  data["PM1"]  = to_numeric(df_raw[mapped["PM1"]]);  pm_labels.append("PM1")
if mapped.get("PM25"): data["PM2.5"]= to_numeric(df_raw[mapped["PM25"]]); pm_labels.append("PM2.5")
if mapped.get("PM10"): data["PM10"] = to_numeric(df_raw[mapped["PM10"]]); pm_labels.append("PM10")

# Gases: si faltan en el archivo, se muestra solo PM (no crashea)
if no2_col is not None: data["NO₂"] = to_numeric(df_raw[no2_col])
if so2_col is not None: data["SO₂"] = to_numeric(df_raw[so2_col])
if o3_col  is not None: data["O₃"]  = to_numeric(df_raw[o3_col])

df = pd.DataFrame(data).sort_values("dt").reset_index(drop=True)

# =============== GRÁFICAS ===============
sp_left, center_col, sp_right = st.columns([0.1,0.8,0.1])
with center_col:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(make_pm_figure(df, pm_labels, pm_band_choice, show_icca), use_container_width=True)
    with c2:
        if "NO₂" in df.columns:
            no2_icca = convert_icca_ranges_ppm(ICCA["NO2_ppm"], no2_unit, MW["NO2"])
            st.plotly_chart(
                make_line_figure(df["dt"], df["NO₂"], f"NO₂ ({no2_unit})",
                                 "#2A9D8F", f"Concentración ({no2_unit})",
                                 icca_ranges=no2_icca, show_icca=show_icca),
                use_container_width=True
            )
        else:
            st.info("El CSV no tiene NO₂; se omite el gráfico.")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        if "SO₂" in df.columns:
            so2_icca = convert_icca_ranges_ppm(ICCA["SO2_ppm"], so2_unit, MW["SO2"])
            st.plotly_chart(
                make_line_figure(df["dt"], df["SO₂"], f"SO₂ ({so2_unit})",
                                 "#E76F51", f"Concentración ({so2_unit})",
                                 icca_ranges=so2_icca, show_icca=show_icca),
                use_container_width=True
            )
        else:
            st.info("El CSV no tiene SO₂; se omite el gráfico.")
    with c4:
        if "O₃" in df.columns:
            o3_icca = convert_icca_ranges_ppm(ICCA["O3_ppm"], o3_unit, MW["O3"])
            st.plotly_chart(
                make_line_figure(df["dt"], df["O₃"], f"O₃ ({o3_unit})",
                                 "#264653", f"Concentración ({o3_unit})",
                                 icca_ranges=o3_icca, show_icca=show_icca),
                use_container_width=True
            )
        else:
            st.info("El CSV no tiene O₃; se omite el gráfico.")

# =============== Estadísticas y descarga ===============
with st.expander("Estadísticas del mes (valores originales)"):
    cols_for_stats = [c for c in ["PM1","PM2.5","PM10","NO₂","SO₂","O₃"] if c in df.columns]
    st.dataframe(
        df[cols_for_stats].describe().T.rename(
            columns={"mean":"media","std":"desv.std","min":"mín","max":"máx"}
        )[["count","media","desv.std","mín","25%","50%","75%","máx"]]
    )

year_str  = f"{int(row['year'])}"  if pd.notna(row['year'])  else "YYYY"
month_str = f"{int(row['month']):02d}" if pd.notna(row['month']) else "MM"
sid_str   = row.get('sid') or 'NA'
file_stub = f"clean_{sid_str}_{year_str}{month_str}"
st.download_button("Descargar CSV limpio (este archivo)", data=df.to_csv(index=False),
                   file_name=f"{file_stub}.csv", mime="text/csv")
