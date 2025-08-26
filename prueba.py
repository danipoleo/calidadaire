# app.py (smoke test)
import streamlit as st
import pandas as pd
import plotly
import plotly.graph_objects as go

st.write("✅ Streamlit ok")
st.write("✅ pandas", pd.__version__)
st.write("✅ plotly", plotly.__version__)

fig = go.Figure(data=go.Scatter(y=[1,3,2,4]))
fig.update_layout(title="Prueba Plotly")
st.plotly_chart(fig, use_container_width=True)
