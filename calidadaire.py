# app.py
# Visor mensual 2x2 centrado con bandas ICCA: Particulados (PM1/PM2.5/PM10 juntos), NO2, SO2, O3 y conversión de unidades.
# Requisitos: streamlit, pandas, plotly, python-dateutil

import re
import math
import pathlib
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# ===============  EDITAR AQUÍ (RUTA DE DATOS)  ===============
DATA_DIR = r"C:\Users\dpoleo\Downloads"   # <-- CAMBIA ESTA RUTA CUANDO LO NECESITES
# ============================================================

# Mapeo de estaciones
STATION_MAP = {
    "1637": "San José",
    "1653": "Cigefi",
    "1735": "Cartago TEC",
    "1762": "Santa Lucía",
    "1775": "Belén CTP",
    "1776": "Fabio Baudrit",
    "Z01777": "Municipalidad de Santa Ana",
}

# Columnas posibles (nombres varían por exporte)
COL_CANDIDATES = {
    "PM1":   ["PM1 Concentration", "PM1"],
    "PM25":  ["PM2.5 Concentration", "PM2.5", "PM2_5", "PM25"],
    "PM10":  ["PM10 Concentration", "PM10"],
    "NO2":   ["NO2 Concentration", "NO2", "NO₂"],
    "SO2":   ["SO2 Concentration", "SO2", "SO₂"],
    "O3":    ["O3 Concentration",  "O3", "Ozone", "O₃"],
    "CO":    ["CO Concentration",  "CO"],
    "DATE":  ["Date (Local)", "Date"],
    "TIME":  ["Time (Local)", "Time"],
    "UTC":   ["UTC Time Stamp", "UTC", "Timestamp"],
    # Columnas de unidades (si existen)
    "NO2_UNIT": ["NO2 Unit", "NO₂ Unit", "NO2_Unit"],
    "SO2_UNIT": ["SO2 Unit", "SO₂ Unit", "SO2_Unit"],
    "O3_UNIT":  ["O3 Unit",  "O₃ Unit",  "O3_Unit"],
    "CO_UNIT":  ["CO Unit", "CO_Unit"],
    "PM1_UNIT": ["PM1 Unit"],
    "PM25_UNIT":["PM2.5 Unit", "PM2_5 Unit", "PM25 Unit"],
    "PM10_UNIT":["PM10 Unit"],
}

# Regex del nombre de archivo: device_<sid>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>_1hr.csv
FILENAME_REGEX = re.compile(
    r"^device_(?P<sid>[A-Za-z0-9]+)_(?P<start>\d{12})_(?P<end>\d{12})_1hr\.csv$"
)

# ---------- ICCA (rangos base) ----------
# PMs en µg/m³, gases en ppm (se convertirán a la unidad del CSV)
ICCA = {
    "PM10": [
        ("Verde",      "#00A65A", 0,    60),
        ("Amarillo",   "#FFC107", 61,   100),
        ("Anaranjado", "#FF9800", 101,  200),
        ("Rojo",       "#E53935", 201,  250),
        ("Púrpura",    "#8E24AA", 250,  math.inf),  # >250
    ],
    "PM2.5": [
        ("Verde",      "#00A65A", 0,    15),
        ("Amarillo",   "#FFC107", 15.1, 40),
        ("Anaranjado", "#FF9800", 40.1, 65),
        ("Rojo",       "#E53935", 66,   100),
        ("Púrpura",    "#8E24AA", 100,  math.inf),  # >100
    ],
    # Gases base en ppm (luego se convierten a la unidad del CSV)
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
        ("Púrpura",    "#8E24AA", 0.260, math.inf),
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

def find_column(df, options):
    # Exact match
    for candidate in options:
        for col in df.columns:
            if col.strip().lower() == candidate.strip().lower():
                return col
    # Fallback: contiene
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in options:
        c = candidate.strip().lower()
        for lc, real in lower_cols.items():
            if c in lc:
                return real
    return None

def scan_files(data_dir):
    rows = []
    p = pathlib.Path(data_dir)
    for f in p.glob("device_*_1hr.csv"):
        m = FILENAME_REGEX.match(f.name)
        if not m:
            continue
        sid = m.group("sid")
        start = m.group("start")  # YYYYMMDDHHMM
        end   = m.group("end")
        try:
            dt_start = datetime.strptime(start, "%Y%m%d%H%M")
            dt_end   = datetime.strptime(end,   "%Y%m%d%H%M")
        except Exception:
            dt_start = dt_end = None
        rows.append({
            "file": str(f),
            "sid": sid,
            "station": STATION_MAP.get(sid, sid),
            "dt_start": dt_start,
            "dt_end": dt_end,
            "year": dt_start.year if dt_start else None,
            "month": dt_start.month if dt_start else None,
            "fname": f.name
        })
    cols = ["file","sid","station","dt_start","dt_end","year","month","fname"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

def load_month_file(path):
    # N/As a NaN
    df = pd.read_csv(path, na_values=["N/A", "NA", "", "null", "None"])
    # Fecha/hora
    col_utc  = find_column(df, COL_CANDIDATES["UTC"])
    col_date = find_column(df, COL_CANDIDATES["DATE"])
    col_time = find_column(df, COL_CANDIDATES["TIME"])

    if col_utc is not None:
        dt = pd.to_datetime(df[col_utc], errors="coerce", utc=True).dt.tz_convert(None)
    elif (col_date is not None) and (col_time is not None):
        dt = pd.to_datetime(
            df[col_date].astype(str) + " " + df[col_time].astype(str),
            errors="coerce",
            dayfirst=True
        )
    else:
        dt = pd.NaT

    # Mapear columnas de interés
    mapped = {k: find_column(df, COL_CANDIDATES[k]) for k in ["PM1","PM25","PM10","NO2","SO2","O3","CO"]}
    # Mapear columnas de unidades (si existen)
    unit_cols = {
        "NO2": find_column(df, COL_CANDIDATES["NO2_UNIT"]),
        "SO2": find_column(df, COL_CANDIDATES["SO2_UNIT"]),
        "O3":  find_column(df, COL_CANDIDATES["O3_UNIT"]),
        "CO":  find_column(df, COL_CANDIDATES["CO_UNIT"]),
        "PM1": find_column(df, COL_CANDIDATES["PM1_UNIT"]),
        "PM25":find_column(df, COL_CANDIDATES["PM25_UNIT"]),
        "PM10":find_column(df, COL_CANDIDATES["PM10_UNIT"]),
    }

    df["dt"] = dt
    return df, mapped, unit_cols

def unit_from_column(df, unit_col, default):
    """Lee la unidad desde la columna de unidades; si no hay, devuelve default."""
    if unit_col and unit_col in df.columns:
        val = df[unit_col].dropna().astype(str)
        if len(val):
            return val.iloc[0].strip()
    return default

# Conversión de umbrales ICCA (gases) a la unidad del CSV
def convert_icca_ranges_ppm(ranges_ppm, target_unit, mw):
    """
    ranges_ppm: lista de tuplas (label, color, low_ppm, high_ppm)
    target_unit: 'ppm', 'µg/m³' (o 'ug/m3'), 'mg/m³'
    """
    tu = (target_unit or "").lower().replace("ug/m3","µg/m³").replace("ug/m³","µg/m³")
    out = []
    for label, color, lo, hi in ranges_ppm:
        if tu in ["ppm"]:
            lo2, hi2 = lo, hi
        elif tu in ["µg/m³", "μg/m³"]:
            # µg/m³ = ppm * MW * 1000 / 24.45
            factor = mw * 1000.0 / 24.45
            lo2 = None if (lo is None or lo==0 and math.isinf(lo)) else (0 if lo==0 else lo*factor)
            hi2 = math.inf if hi is None or math.isinf(hi) else hi*factor
        elif tu in ["mg/m³", "mg/m3"]:
            # mg/m³ = ppm * MW / 24.45
            factor = mw / 24.45
            lo2 = None if (lo is None or lo==0 and math.isinf(lo)) else (0 if lo==0 else lo*factor)
            hi2 = math.inf if hi is None or math.isinf(hi) else hi*factor
        else:
            # Desconocida: devolvemos ppm
            lo2, hi2 = lo, hi
        out.append((label, color, lo2, hi2))
    return out

# Crear shapes (bandas) para un eje 'y' dado
def make_bands(x0, x1, ranges, opacity=0.12, y_padding=0.0):
    shapes = []
    lines  = []
    for label, color, lo, hi in ranges:
        y0 = -math.inf if lo is None else lo
        y1 =  math.inf if (hi is None or math.isinf(hi)) else hi
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, opacity=opacity, layer="below", line=dict(width=0)
        ))
        # Línea límite superior (si no es infinito) para marcar el corte
        if hi is not None and not math.isinf(hi):
            lines.append(dict(
                type="line", xref="x", yref="y",
                x0=x0, x1=x1, y0=hi, y1=hi,
                line=dict(color=color, width=1, dash="dot"), layer="below"
            ))
    return shapes, lines

# -------------------- UI --------------------
st.set_page_config(page_title="Visor mensual de calidad de aire", layout="wide")

# CSS para centrar y ancho máximo
st.markdown("""
<style>
.block-container {max-width: 1200px; margin: 0 auto;}
h1, h2, h3 {text-align: center;}
</style>
""", unsafe_allow_html=True)

st.title("Visor mensual de calidad de aire")
st.caption("Grilla 2×2 centrada: Particulados (PM1/PM2.5/PM10 juntos), NO₂, SO₂ y O₃. Bandas ICCA incluidas.")

with st.sidebar:
    st.header("Configuración")
    DATA_DIR = st.text_input("Carpeta con CSV (device_*_1hr.csv)", DATA_DIR)

    files_df = scan_files(DATA_DIR)
    if files_df.empty:
        st.error("No se encontraron archivos con patrón device_<id>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>_1hr.csv en la carpeta.")
        st.stop()

    years = sorted([y for y in files_df["year"].dropna().unique()], reverse=True)
    year_sel = st.selectbox("Año", years)

    month_names = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                   7:"Julio",8:"Agosto",9:"Setiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    months = sorted(files_df.loc[files_df["year"]==year_sel,"month"].dropna().unique())
    month_sel = st.selectbox("Mes", months, format_func=lambda m: f"{month_names.get(m, m)} ({m:02d})")

    stations = files_df.loc[(files_df["year"]==year_sel)&(files_df["month"]==month_sel), ["sid","station"]].drop_duplicates()
    label_map = {f"{r.station} ({r.sid})": r.sid for r in stations.itertuples()}
    station_label = st.selectbox("Estación", list(label_map.keys()))
    sid_sel = label_map[station_label]

    show_icca = st.checkbox("Mostrar bandas ICCA", value=True)
    pm_band_choice = st.radio("Umbral para el fondo de particulados", ["PM2.5","PM10"], index=0)

# Localiza el archivo del mes/estación
row = files_df[(files_df["year"]==year_sel) & (files_df["month"]==month_sel) & (files_df["sid"]==sid_sel)]
if row.empty:
    st.warning("No se encontró el archivo para esa combinación.")
    st.stop()

file_path = row.iloc[0]["file"]
st.caption(f"Archivo: `{row.iloc[0]['fname']}`  ·  Estación: **{row.iloc[0]['station']}**  ·  Periodo: {row.iloc[0]['dt_start']} → {row.iloc[0]['dt_end']}")

# Carga
df_raw, cols, unit_cols = load_month_file(file_path)

# Columnas requeridas (gases) + PMs disponibles
no2_col = cols.get("NO2")
so2_col = cols.get("SO2")
o3_col  = cols.get("O3")

gases_missing = [name for name, col in {"NO₂": no2_col, "SO₂": so2_col, "O₃": o3_col}.items() if col is None]
if gases_missing:
    st.error("El CSV no contiene: " + ", ".join(gases_missing))
    st.stop()

# Particulados disponibles
pm_map  = []  # (colname_in_df, label_to_show, color)
if cols.get("PM1"):  pm_map.append((cols["PM1"],  "PM1",   "#8ECAE6"))
if cols.get("PM25"): pm_map.append((cols["PM25"], "PM2.5", "#FB8500"))  # destacado
if cols.get("PM10"): pm_map.append((cols["PM10"], "PM10",  "#219EBC"))
if len(pm_map) == 0:
    st.error("No se encontraron columnas de particulado (PM1/PM2.5/PM10) en el CSV.")
    st.stop()

# Detectar unidades (si existen). Defaults comunes si no hay columnas de unidades
no2_unit = unit_from_column(df_raw, unit_cols.get("NO2"), "µg/m³")  # típicamente µg/m³ en tus CSV
o3_unit  = unit_from_column(df_raw, unit_cols.get("O3"),  "µg/m³")
so2_unit = unit_from_column(df_raw, unit_cols.get("SO2"), "µg/m³")
co_unit  = unit_from_column(df_raw, unit_cols.get("CO"),  "mg/m³")  # CO suele venir en mg/m³

# DataFrame base (valores originales), ordenado por tiempo
data = {
    "dt": df_raw["dt"],
    "NO₂": pd.to_numeric(df_raw[no2_col], errors="coerce"),
    "SO₂": pd.to_numeric(df_raw[so2_col], errors="coerce"),
    "O₃":  pd.to_numeric(df_raw[o3_col],  errors="coerce")
}
for col, label, _ in pm_map:
    data[label] = pd.to_numeric(df_raw[col], errors="coerce")

df = pd.DataFrame(data).sort_values("dt").reset_index(drop=True)

def make_line_figure(x, y, title, color, y_title, icca_ranges=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color)))
    fig.update_layout(
        title=title, template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
        xaxis_title="Fecha/Hora", yaxis_title=y_title,
        height=350, legend_title_text=None
    )
    if show_icca and icca_ranges:
        x0, x1 = (x.min(), x.max()) if hasattr(x, "min") else (df["dt"].min(), df["dt"].max())
        shapes, lines = make_bands(x0, x1, icca_ranges, opacity=0.12)
        fig.update_layout(shapes=shapes + lines)
    return fig

def make_pm_figure(df, pm_map, pm_band_choice):
    fig = go.Figure()
    for _, label, color in pm_map:
        lw = 2.5 if label == "PM2.5" else 2
        fig.add_trace(go.Scatter(
            x=df["dt"], y=df[label], name=label, mode="lines",
            line=dict(color=color, width=lw)
        ))

    # Bandas ICCA para particulados
    if show_icca:
        x0, x1 = df["dt"].min(), df["dt"].max()
        if pm_band_choice == "PM2.5":
            icca_ranges = ICCA["PM2.5"]
        else:
            icca_ranges = ICCA["PM10"]
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

# ---- Layout centrado: grilla 2x2 ----
sp_left, center_col, sp_right = st.columns([0.1, 0.8, 0.1])
with center_col:
    # --------- Particulados (arriba-izquierda) ---------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(make_pm_figure(df, pm_map, pm_band_choice), use_container_width=True)

    # --------- NO2 (arriba-derecha) ---------
    with c2:
        # Convertir ICCA (ppm) a unidad del CSV para NO2
        no2_icca = convert_icca_ranges_ppm(ICCA["NO2_ppm"], no2_unit, MW["NO2"])
        st.plotly_chart(
            make_line_figure(df["dt"], df["NO₂"], f"Dióxido de Nitrógeno — NO₂  ({no2_unit})",
                             "#2A9D8F", f"Concentración ({no2_unit})", icca_ranges=no2_icca),
            use_container_width=True
        )
    # --------- SO2 (abajo-izquierda) ---------
    c3, c4 = st.columns(2, gap="large")
    with c3:
        so2_icca = convert_icca_ranges_ppm(ICCA["SO2_ppm"], so2_unit, MW["SO2"])
        st.plotly_chart(
            make_line_figure(df["dt"], df["SO₂"], f"Dióxido de Azufre — SO₂  ({so2_unit})",
                             "#E76F51", f"Concentración ({so2_unit})", icca_ranges=so2_icca),
            use_container_width=True
        )
    # --------- O3 (abajo-derecha) ---------
    with c4:
        o3_icca = convert_icca_ranges_ppm(ICCA["O3_ppm"], o3_unit, MW["O3"])
        st.plotly_chart(
            make_line_figure(df["dt"], df["O₃"], f"Ozono — O₃  ({o3_unit})",
                             "#264653", f"Concentración ({o3_unit})", icca_ranges=o3_icca),
            use_container_width=True
        )

# Estadísticas del mes (solo columnas presentes)
with st.expander("Estadísticas del mes (valores originales)"):
    cols_for_stats = [lab for _, lab, _ in pm_map] + ["NO₂","SO₂","O₃"]
    st.dataframe(
        df[cols_for_stats].describe().T.rename(columns={
            "mean":"media", "std":"desv.std", "min":"mín", "max":"máx"
        })[["count","media","desv.std","mín","25%","50%","75%","máx"]]
    )

# Descarga del CSV limpio
st.download_button(
    "Descargar CSV limpio (este mes/estación)",
    data=df.to_csv(index=False),
    file_name=f"clean_{sid_sel}_{year_sel}{month_sel:02d}.csv",
    mime="text/csv"
)

with st.expander("Ayuda / Supuestos"):
    st.markdown(f"""
- Detección de archivos: **`device_<id>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>_1hr.csv`**.
- Eje temporal: **UTC Time Stamp** si existe; si no, **Date (Local) + Time (Local)**.
- **Bandas ICCA**:
  - **Particulados** (PM10 y PM2.5) en **µg/m³** (según rangos proporcionados).
  - **Gases (NO₂, O₃, SO₂, CO)**: ICCA en **ppm** convertido automáticamente a la **unidad del CSV**:
    - NO₂ ({no2_unit}), O₃ ({o3_unit}), SO₂ ({so2_unit}), CO ({co_unit}).
- Conversión (25°C, 1 atm): µg/m³ = ppm × MW × 1000 / 24.45; mg/m³ = ppm × MW / 24.45.
- Si las columnas de unidad no existen, se asume: NO₂/O₃/SO₂ en **µg/m³** y CO en **mg/m³**.
- En el gráfico de **Particulados**, el fondo muestra el umbral seleccionado (**{pm_band_choice}**); los otros PM se grafican como curvas para comparación.
""")
