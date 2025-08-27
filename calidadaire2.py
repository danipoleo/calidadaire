# app.py
# Visor mensual 2x2 con bandas ICCA (usa TODOS los device_*_1hr.csv de la carpeta)
# Fuente: GitHub (HTML y/o API) + Plan B URL directa
# Promedia múltiples estaciones seleccionadas (gases convertidos a µg/m³)

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
DEFAULT_PATH_IN_REPO = "datosCA"   # usa "datos" si es tu carpeta actual
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
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=_gh_headers(), timeout=60)
    r.raise_for_status()
    try:
        txt = r.content.decode("utf-8")
    except UnicodeDecodeError:
        txt = r.content.decode("latin-1", errors="replace")
    clean_txt = _clean_earthsense_text(txt)
    try:
        return pd.read_csv(io.StringIO(clean_txt), engine="python", sep=",", quotechar='"',
                           on_bad_lines="skip", na_values=["N/A","NA","","null","None"])
    except Exception:
        return pd.read_csv(io.StringIO(clean_txt), engine="python", sep=";", quotechar='"',
                           on_bad_lines="skip", na_values=["N/A","NA","","null","None"])

def _normalize_rel_for_raw(rel: str, branch: str) -> str:
    if not rel: return rel
    rel = rel.lstrip("/")
    if rel.startswith("refs/heads/"):
        rel = rel[len("refs/heads/"):]
    prefix = f"{branch}/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return rel

def _to_tree_url_if_blob(url: str) -> str:
    if not url: return url
    u = url.split("?", 1)[0]
    return u.replace("/blob/", "/tree/")

def _raw_from_blob_or_raw(owner: str, repo: str, branch: str, url: str) -> str:
    if "raw.githubusercontent.com" in url:
        return url.split("?", 1)[0]
    if "/blob/" in url:
        rel = url.split("/blob/", 1)[-1].split("?", 1)[0]
        rel = _normalize_rel_for_raw(rel, branch)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel}"
    return url.split("?", 1)[0]

# --------- LISTADO DE ARCHIVOS (HTML y/o API) ----------
def _extract_csv_links(html: str) -> List[str]:
    return re.findall(r'href="([^"]+?/blob/[^"]+?\.csv[^"]*)"', html, flags=re.IGNORECASE)

def _list_with_html_plain(owner: str, repo: str, path: str, branch: str):
    url = f"https://github.com/{owner}/{repo}/tree/{branch}/{path}".rstrip("/") + "?plain=1"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code >= 400:
            return [], {"status": r.status_code, "where": "html_plain"}
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
                files.append({"name": name, "download_url": raw, "path": rel})
        return files, {"status": r.status_code, "where": "html_plain", "count": len(files)}
    except Exception as e:
        return [], {"status": "ERR", "where": "html_plain", "error": str(e)}

def _list_with_html(owner: str, repo: str, path: str, branch: str):
    url = f"https://github.com/{owner}/{repo}/tree/{branch}/{path}".rstrip("/")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code >= 400:
            return [], {"status": r.status_code, "where": "html"}
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
                files.append({"name": name, "download_url": raw, "path": rel})
        return files, {"status": r.status_code, "where": "html", "count": len(files)}
    except Exception as e:
        return [], {"status": "ERR", "where": "html", "error": str(e)}

def _list_with_git_tree(owner: str, repo: str, branch: str, subpath: str):
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
                                "path": p})
        return out, {"status": r.status_code, "where": "git_tree", "count": len(out)}
    except Exception as e:
        return [], {"status": "ERR", "where": "git_tree", "error": str(e)}

def _list_with_contents_api(owner: str, repo: str, path: str, branch: str):
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
                files.append({"name": name, "download_url": it.get("download_url"), "path": it.get("path")})
        return files, {"status": r.status_code, "where": "contents_api", "count": len(files)}
    except Exception as e:
        return [], {"status": "ERR", "where": "contents_api", "error": str(e)}

# ---------- Función cacheada de listado (30 minutos) ----------
@st.cache_data(ttl=1800, show_spinner=False)
def github_list_files(owner: str, repo: str, path: str, branch: str,
                      prefer_html: bool, token_present: bool):
    debug = []
    files = []

    def try_and_log(fn, *args):
        f, meta = fn(*args)
        debug.append(meta)
        return f

    if prefer_html or not token_present:
        files = try_and_log(_list_with_html_plain, owner, repo, path, branch)
        if not files:
            files = try_and_log(_list_with_html, owner, repo, path, branch)
        if not files and token_present:
            files = try_and_log(_list_with_contents_api, owner, repo, path, branch)
            if not files:
                files = try_and_log(_list_with_git_tree, owner, repo, branch, path)
    else:
        files = try_and_log(_list_with_git_tree, owner, repo, branch, path)
        if not files:
            files = try_and_log(_list_with_contents_api, owner, repo, path, branch)
        if not files:
            files = try_and_log(_list_with_html_plain, owner, repo, path, branch)
            if not files:
                files = try_and_log(_list_with_html, owner, repo, path, branch)

    return files, debug

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
    if show_icca:
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
st.set_page_config(page_title="Visor mensual de calidad de aire (GitHub)", layout="wide")
st.markdown("""<style>.block-container{max-width:1200px;margin:0 auto;}h1,h2,h3{text-align:center;}</style>""",
            unsafe_allow_html=True)

st.title("Visor mensual de calidad de aire — Fuente: GitHub")
st.caption("Usa TODOS los 'device_*_1hr.csv' de la carpeta; permite promediar varias estaciones.")

with st.sidebar:
    st.header("Fuente de datos (GitHub)")
    owner  = st.text_input("Owner", value=DEFAULT_GITHUB_OWNER)
    repo   = st.text_input("Repo",  value=DEFAULT_GITHUB_REPO)
    path   = st.text_input("Ruta en el repo", value=DEFAULT_PATH_IN_REPO)
    branch = st.text_input("Branch", value=DEFAULT_BRANCH)

    st.caption("Pega URL de carpeta (acepta /tree/ o /blob/):")
    gh_url = st.text_input("URL de carpeta", value="", placeholder="https://github.com/owner/repo/tree/branch/carpeta")
    if gh_url.strip():
        try:
            url_fixed = _to_tree_url_if_blob(gh_url.strip())
            parts = url_fixed.split("github.com/")[-1].split("/")
            owner = parts[0]; repo = parts[1]
            idx_tree = parts.index("tree") if "tree" in parts else -1
            if idx_tree != -1 and len(parts) > idx_tree+1:
                branch = parts[idx_tree+1]
                path = "/".join(parts[idx_tree+2:]) if len(parts) > idx_tree+2 else ""
        except Exception:
            st.warning("No pude interpretar la URL; usando owner/repo/ruta/branch.")

    st.divider()
    token_present = bool(_gh_headers())
    prefer_html = st.checkbox("Forzar solo HTML (sin API)", value=(not token_present),
                              help="Útil si no tienes GITHUB_TOKEN o ves rate limits.")
    direct_csv_url = st.text_input("Plan B: URL directa a un CSV (raw o blob)", value="",
                                   placeholder="https://github.com/.../blob/main/datosCA/device_...csv")
    st.divider()
    show_icca = st.checkbox("Mostrar bandas ICCA", value=True)
    pm_band_choice = st.radio("Umbral fondo particulados", ["PM2.5","PM10"], index=0)
    if st.button("Recargar / limpiar caché"):
        st.cache_data.clear()
        st.success("Caché limpiada.")

# --- Listar CSV (solo los que cumplen el patrón device_*_1hr.csv) ---
files, debug = github_list_files(owner, repo, path, branch, prefer_html, token_present)

with st.expander("Diagnóstico de listado"):
    st.write({"owner": owner, "repo": repo, "path": path, "branch": branch,
              "prefer_html": prefer_html, "token_present": token_present})
    st.write(debug)

if not files and direct_csv_url.strip():
    raw = _raw_from_blob_or_raw(owner, repo, branch, direct_csv_url.strip())
    name = raw.split("/")[-1].split("?")[0]
    if FILENAME_REGEX.match(name):
        files = [{"name": name, "download_url": raw, "path": raw}]

if not files:
    st.error("No encontré CSV con el patrón device_*_1hr.csv. Verifica branch/carpeta o usa 'Plan B'.")
    st.stop()

# Construir índice e intentar parsear año/mes del nombre
rows = []
for f in files:
    fname = f.get("name") or ""
    if not fname or not FILENAME_REGEX.match(fname):
        continue
    meta = parse_from_filename(fname)
    sid = meta["sid"]; station = STATION_MAP.get(sid, sid if sid else "Desconocida")
    url = f.get("download_url") or f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{f.get('path','')}"
    rows.append({"name": fname, "url": url, "sid": sid, "station": station,
                 "dt_start": meta["dt_start"], "dt_end": meta["dt_end"],
                 "year": meta["year"], "month": meta["month"]})
files_df = pd.DataFrame(rows)
if files_df.empty:
    st.error("No se pudo construir índice; revisa nombres de archivo.")
    st.stop()

# Controles (Año/Mes/Estación MÚLTIPLE)
valid_years = sorted([int(y) for y in files_df["year"].dropna().unique()], reverse=True)
if not valid_years:
    st.error("No pude resolver año/mes desde los nombres.")
    st.stop()

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
all_labels = list(subset["label"])

sel_labels = st.multiselect("Estaciones (puedes elegir varias; promedio en gráficos)", 
                            options=all_labels, default=all_labels)

if not sel_labels:
    st.warning("Selecciona al menos una estación."); st.stop()

chosen = subset[subset["label"].isin(sel_labels)].reset_index(drop=True)

# ---- Carga y normalización de CADA archivo; gases -> µg/m³ ----
def _unit_from_column(df, unit_col, default):
    if unit_col and unit_col in df.columns:
        v = df[unit_col].dropna().astype(str)
        if len(v): return v.iloc[0].strip()
    return default

def _to_ugm3(series: pd.Series, from_unit: str, mw: float) -> pd.Series:
    fu = (from_unit or "").lower().replace("ug/m3","µg/m³").replace("ug/m³","µg/m³")
    if fu in ["µg/m³", "μg/m³", "ug/m3", "ug/m³", "μg/m3"]:
        return pd.to_numeric(series, errors="coerce")
    if fu == "ppm":
        factor = mw * 1000.0 / 24.45
        return pd.to_numeric(series, errors="coerce") * factor
    if fu in ["mg/m³", "mg/m3"]:
        factor = 1000.0  # mg/m³ -> µg/m³
        return pd.to_numeric(series, errors="coerce") * factor
    # si desconocida, no convierto
    return pd.to_numeric(series, errors="coerce")

def load_one(url: str):
    df_raw = fetch_csv(url)

    col_utc  = find_column(df_raw, COL_CANDIDATES["UTC"])
    col_date = find_column(df_raw, COL_CANDIDATES["DATE"])
    col_time = find_column(df_raw, COL_CANDIDATES["TIME"])
    if col_utc is not None:
        dt = pd.to_datetime(df_raw[col_utc], errors="coerce", utc=True).dt.tz_convert(None)
    elif (col_date is not None) and (col_time is not None):
        dt = pd.to_datetime(df_raw[col_date].astype(str)+" "+df_raw[col_time].astype(str),
                            errors="coerce", dayfirst=True)
    else:
        raise ValueError("No encontré columnas de tiempo (UTC o Date+Time).")

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
    # si faltan gases, lo omito (pero no reviento)
    gases_ok = all([no2_col, so2_col, o3_col])

    # PM: asumo µg/m³ ya
    data = {"dt": dt}
    if mapped.get("PM1"):  data["PM1"]  = pd.to_numeric(df_raw[mapped["PM1"]], errors="coerce")
    if mapped.get("PM25"): data["PM2.5"]= pd.to_numeric(df_raw[mapped["PM25"]], errors="coerce")
    if mapped.get("PM10"): data["PM10"] = pd.to_numeric(df_raw[mapped["PM10"]], errors="coerce")

    if gases_ok:
        no2_unit = _unit_from_column(df_raw, unit_cols.get("NO2"), "µg/m³")
        so2_unit = _unit_from_column(df_raw, unit_cols.get("SO2"), "µg/m³")
        o3_unit  = _unit_from_column(df_raw, unit_cols.get("O3"),  "µg/m³")
        data["NO2_µg/m³"] = _to_ugm3(df_raw[no2_col], no2_unit, MW["NO2"])
        data["SO2_µg/m³"] = _to_ugm3(df_raw[so2_col], so2_unit, MW["SO2"])
        data["O3_µg/m³"]  = _to_ugm3(df_raw[o3_col],  o3_unit,  MW["O3"])

    df = pd.DataFrame(data).dropna(subset=["dt"]).sort_values("dt")
    return df, gases_ok

loaded, used, skipped = [], 0, 0
errors = []
for _, r in chosen.iterrows():
    try:
        dfi, gases_ok = load_one(r["url"])
        if not gases_ok:
            skipped += 1
            errors.append(f"Omitido (sin NO2/SO2/O3): {r['name']}")
            continue
        loaded.append(dfi.set_index("dt"))
        used += 1
    except Exception as e:
        skipped += 1
        errors.append(f"Error con {r['name']}: {e}")

if not loaded:
    st.error("No se pudo cargar ninguna estación válida para ese mes (faltan gases o errores de lectura).")
    if errors: st.write(errors)
    st.stop()

# Unir por índice de tiempo y promediar columnas homónimas
wide = pd.concat(loaded, axis=1)
# Promedios ignorando NaN
def _avg(cols):
    return wide[cols].mean(axis=1, skipna=True) if any(c in wide.columns for c in cols) else pd.Series(dtype="float64")

avg = pd.DataFrame(index=wide.index)
pm_labels = []
for lab in ["PM1","PM2.5","PM10"]:
    cols = [c for c in wide.columns if c.endswith(lab)]
    # si el CSV de cada estación no repite sufijos, seleccione por nombre exacto
    if not cols:
        cols = [c for c in wide.columns if c.split("_")[-1] == lab or c == lab]
    if any(lab == c or c.endswith(lab) for c in wide.columns):
        # normalizar nombres columna: exactos lab
        # intentamos columnas exactas:
        cand = [c for c in wide.columns if c == lab]
        if cand:
            avg[lab] = _avg(cand)
        else:
            cand = [c for c in wide.columns if c.endswith(lab)]
            avg[lab] = _avg(cand)
        pm_labels.append(lab)

for gas in ["NO2","SO2","O3"]:
    cols = [c for c in wide.columns if c.startswith(f"{gas}_")]
    if cols:
        avg[gas] = _avg(cols)

avg = avg.reset_index().rename(columns={"index":"dt"})
avg = avg.sort_values("dt")

# ---- Render UI / Info de selección ----
st.caption(f"Archivos seleccionados: {used} usados · {skipped} omitidos")
if errors:
    with st.expander("Detalles de archivos omitidos / errores"):
        for e in errors:
            st.write("- " + e)

# ---- Figuras (ICCA en µg/m³ para gases) ----
sp_left, center_col, sp_right = st.columns([0.1,0.8,0.1])
with center_col:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(make_pm_figure(avg, pm_labels, pm_band_choice, show_icca), use_container_width=True)
    with c2:
        no2_icca = convert_icca_ranges_ppm(ICCA["NO2_ppm"], "µg/m³", MW["NO2"])
        st.plotly_chart(
            make_line_figure(avg["dt"], avg["NO2"], f"NO₂ (µg/m³)",
                             "#2A9D8F", "Concentración (µg/m³)",
                             icca_ranges=no2_icca, show_icca=show_icca),
            use_container_width=True
        )
    c3, c4 = st.columns(2, gap="large")
    with c3:
        so2_icca = convert_icca_ranges_ppm(ICCA["SO2_ppm"], "µg/m³", MW["SO2"])
        st.plotly_chart(
            make_line_figure(avg["dt"], avg["SO2"], f"SO₂ (µg/m³)",
                             "#E76F51", "Concentración (µg/m³)",
                             icca_ranges=so2_icca, show_icca=show_icca),
            use_container_width=True
        )
    with c4:
        o3_icca = convert_icca_ranges_ppm(ICCA["O3_ppm"], "µg/m³", MW["O3"])
        st.plotly_chart(
            make_line_figure(avg["dt"], avg["O3"], f"O₃ (µg/m³)",
                             "#264653", "Concentración (µg/m³)",
                             icca_ranges=o3_icca, show_icca=show_icca),
            use_container_width=True
        )

# Estadísticas
with st.expander("Estadísticas del mes (promedio de estaciones seleccionadas)"):
    cols_for_stats = [c for c in ["PM1","PM2.5","PM10","NO2","SO2","O3"] if c in avg.columns]
    st.dataframe(
        avg[cols_for_stats].describe().T.rename(
            columns={"mean":"media","std":"desv.std","min":"mín","max":"máx"}
        )[["count","media","desv.std","mín","25%","50%","75%","máx"]]
    )

# Descarga del CSV promedio
year_str  = f"{int(year_sel)}"
month_str = f"{int(month_sel):02d}"
file_stub = f"clean_PROM_{year_str}{month_str}"
st.download_button("Descargar CSV limpio (promedio seleccionado)", data=avg.to_csv(index=False),
                   file_name=f"{file_stub}.csv", mime="text/csv")
