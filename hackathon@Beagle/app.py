import streamlit as st
import folium
from streamlit_folium import st_folium

from geo_utils import make_bbox, fetch_osm, grid_bbox, features
from models import apply_scenario, uhi_delta, traffic_delay_pct, pm25_delta
from viz import add_heat_layer

st.set_page_config(page_title="UrbanTwin AI", layout="wide")
st.title("UrbanTwin AI — Generative City Simulation")

with st.sidebar:
    st.header("Area")
    with st.form("controls"):
        lat = st.number_input("Latitude", value=43.000000, format="%.6f")
        lon = st.number_input("Longitude", value=-78.790000, format="%.6f")
        km = st.slider("Tile size (km)", 0.5, 2.0, 1.0, 0.5)

        st.header("Scenario")
        add_b = st.slider("% buildings change", -50, 100, 30, 5)
        add_g = st.slider("% green change", -50, 100, 0, 5)

        submit = st.form_submit_button("Run Simulation")

if submit:
    bbox = make_bbox(lat, lon, km)
    bld, roads, green, poly_m = fetch_osm(bbox)

    if len(bld) == 0 and len(green) == 0 and len(roads) == 0:
        st.warning("No OSM features in this tile. Try a different location/size.")
        st.stop()

    grid = grid_bbox(poly_m, cell=50)
    base = features(grid, bld, roads, green)

    # NEW: persist results instead of rendering immediately
    st.session_state["sim"] = {
        "lat": lat,
        "lon": lon,
        "base": base,
    }

# NEW: render from session (stable across reruns)
if "sim" in st.session_state:
    sim = st.session_state["sim"]
    base = sim["base"]
    lat  = sim["lat"]
    lon  = sim["lon"]

    scenario = apply_scenario(base, add_b, add_g)
    uhi = uhi_delta(scenario)
    delay = traffic_delay_pct(scenario)
    pm = pm25_delta(scenario)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Δ UHI (°C, mean)", f"{uhi.mean():+.2f}")
    c2.metric("Δ Traffic delay (%)", f"{delay.mean():+.1f}%")
    c3.metric("Δ PM2.5 (µg/m³)", f"{pm.mean():+.2f}")

    # Map
    m = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        tiles="cartodbpositron",
        width="100%",  # make it full-width
        height="100%"  # optional, but helps scaling
    )

    add_heat_layer(m, scenario[["geometry"]], uhi, "UHI Δ (°C)")

    # wider and taller map in Streamlit
    st_folium(
        m,
        width=1200,     # increase width for more horizontal space
        height=600,     # increase height for easier visualization
        key="map"
    )

    st.caption("Heuristic surrogate models; values shown as Δ vs baseline.")