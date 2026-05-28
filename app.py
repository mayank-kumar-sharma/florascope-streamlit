import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import json
import tempfile
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import html
from shapely.geometry import mapping

from modules.eligibility_engine import run_eligibility_assessment
from modules.report_generator import generate_report
from modules.geometry_tools import (
    calculate_area, export_to_kml, export_to_geojson,
    buffer_geometry, simplify_geometry, _extract_geometry
)
from modules.audit_engine import run_full_audit
from modules.canopy_tree_analysis import demo_canopy_analysis


st.set_page_config(layout="wide", page_title="FloraScope", page_icon="🌿")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    background-color: #ffffff;
}
[data-testid="stSidebar"] {
    background-color: #f0f7f0;
}
[data-testid="stSidebar"] .stRadio label {
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] p {
    color: #1a1a1a !important;
}
.block-container h1, .block-container h2, .block-container h3 {
    color: #1a9850;
}
.block-container p {
    color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)


# COPY THESE EXACT FUNCTIONS into app.py as-is:

def validate_geojson(data: dict) -> dict:
    try:
        geom = _extract_geometry(data)
        if geom is None:
            raise ValueError("Could not extract geometry from input")
        if not geom.is_valid:
            geom = geom.buffer(0)
        area_info = calculate_area(data)
        centroid_lat = area_info.get("centroid_lat", 0)
        centroid_lon = area_info.get("centroid_lon", 0)
        if data.get("type") == "FeatureCollection":
            normalized = data
        elif data.get("type") == "Feature":
            normalized = {"type": "FeatureCollection", "features": [data]}
        else:
            normalized = {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": mapping(geom), "properties": {"name": "AOI"}}]
            }
        return {
            "valid": True,
            "geojson": normalized,
            "centroid": {"lat": centroid_lat, "lon": centroid_lon},
            "area_ha_approx": area_info.get("area_hectares", 0),
            "area": area_info,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def generate_demo_analysis(geojson_data: dict) -> dict:
    import random
    from datetime import datetime
    validated = validate_geojson(geojson_data)
    centroid = validated.get("centroid", {"lat": 0, "lon": 0})
    area = validated.get("area_ha_approx", 100)
    current_year = datetime.now().year
    start_year = current_year - 10
    random.seed(42)
    base_trees = random.uniform(5, 25)
    base_shrub = random.uniform(20, 50)
    base_crop = random.uniform(10, 30)
    base_bare = random.uniform(5, 15)
    lulc_ts = {}
    for year in range(start_year, current_year + 1):
        yr_offset = (year - start_year) / 10
        lulc_ts[str(year)] = {
            "Trees": round(max(0, area * (base_trees + random.uniform(-2, 3) * yr_offset) / 100), 2),
            "Rangeland": round(max(0, area * (base_shrub + random.uniform(-3, 3)) / 100), 2),
            "Crops": round(max(0, area * (base_crop + random.uniform(-2, 2)) / 100), 2),
            "Bare Ground": round(max(0, area * (base_bare + random.uniform(-1, 1)) / 100), 2),
            "Water": round(area * random.uniform(0, 3) / 100, 2),
            "Built Area": round(area * random.uniform(0, 2) / 100, 2),
            "Flooded Vegetation": round(area * random.uniform(0, 2) / 100, 2),
        }
    hansen_loss = {yr: round(random.uniform(0, area * 0.005), 2) for yr in range(2001, 2024)}
    ndvi_ts = {str(y): round(random.uniform(0.2, 0.5), 3) for y in range(start_year, current_year + 1)}
    is_coastal = abs(centroid["lat"]) < 30 and random.random() < 0.3
    return {
        "metadata": {
            "analysis_date": datetime.now().isoformat(),
            "analysis_period": f"{start_year}-{current_year}",
            "centroid_lon": round(centroid["lon"], 6),
            "centroid_lat": round(centroid["lat"], 6),
            "data_source": "DEMO MODE",
        },
        "lulc_timeseries": lulc_ts,
        "hansen_forest": {
            "total_area_ha": round(area, 2),
            "forest_area_2000_ha": round(area * base_trees / 100, 2),
            "forest_gain_ha": round(area * random.uniform(0, 0.03), 2),
            "forest_loss_total_ha": round(sum(hansen_loss.values()), 2),
            "loss_by_year": hansen_loss,
            "mean_tree_cover_2000_pct": round(base_trees, 1),
            "forest_cover_pct_2000": round(base_trees, 1),
        },
        "mangrove": {
            "mangrove_present": is_coastal,
            "mangrove_area_ha": round(area * 0.05, 2) if is_coastal else 0,
            "historical_mangrove": is_coastal,
            "tidal_wetland": is_coastal,
        },
        "ndvi_timeseries": ndvi_ts,
        "terrain": {
            "mean_elevation_m": round(random.uniform(50, 800), 1),
            "min_elevation_m": round(random.uniform(10, 100), 1),
            "max_elevation_m": round(random.uniform(200, 1200), 1),
            "mean_slope_deg": round(random.uniform(2, 15), 1),
            "max_slope_deg": round(random.uniform(10, 35), 1),
        },
        "climate": {
            "annual_rainfall_mm": round(random.uniform(400, 2500), 0),
            "mean_temperature_c": round(random.uniform(18, 30), 1),
            "climate_zone": random.choice(["Tropical Wet", "Tropical Moist", "Tropical Dry", "Subtropical Moist"]),
        },
        "protected_areas": {"overlaps_protected_area": random.random() < 0.15, "protected_area_count": 0},
        "soil_carbon": {
            "mean_soc_g_per_kg": round(random.uniform(8, 40), 1),
            "min_soc_g_per_kg": round(random.uniform(3, 15), 1),
            "max_soc_g_per_kg": round(random.uniform(20, 60), 1),
        },
    }


NAV_OPTIONS = ["🏠 Home", "🔍 Pre-Feasibility", "📋 Project Audit", "🌳 Canopy & Biomass", "🗺️ GIS Tools"]

LULC_COLORS = {
    "Trees": "#1a9850",
    "Crops": "#e8d354",
    "Rangeland": "#a0c93d",
    "Bare Ground": "#a39171",
    "Water": "#0096ff",
    "Built Area": "#d13c1b",
    "Flooded Vegetation": "#7a87c6",
}

ELIGIBILITY_CARD_STYLES = {
    "Highly Suitable": {"bg": "#d4edda", "border": "#1a9850"},
    "Suitable": {"bg": "#c8e6c9", "border": "#66bd63"},
    "Possibly Suitable": {"bg": "#fff3cd", "border": "#fee08b"},
    "Not Suitable": {"bg": "#f8d7da", "border": "#d73027"},
    "Needs Further Investigation": {"bg": "#fff3cd", "border": "#fdae61"},
}


def _safe_json_loads(raw_bytes: bytes) -> dict:
    return json.loads(raw_bytes.decode("utf-8"))


def _load_geojson_from_uploader(uploaded) -> dict | None:
    if uploaded is None:
        return None
    try:
        return json.loads(uploaded.read())
    except Exception:
        try:
            uploaded.seek(0)
        except Exception:
            pass
        return _safe_json_loads(uploaded.getvalue())


def _ensure_session_defaults():
    st.session_state.setdefault("project_name", "My Forest Project")
    st.session_state.setdefault("geojson", None)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("eligibility", None)
    st.session_state.setdefault("canopy_geojson", None)
    st.session_state.setdefault("canopy_result", None)


def _render_feature_card(title: str, body: str):
    safe_title = html.escape(str(title))
    safe_body = html.escape(str(body))
    st.markdown(
        f"""
<div class="feature-card">
  <div style="font-size: 18px; font-weight: 700; color: #1a9850; margin-bottom: 6px;">{safe_title}</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='color:#262730; font-size:14px;'>{safe_body}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_eligibility_card(r: dict):
    status = str(r.get("status", "Needs Further Investigation"))
    style = ELIGIBILITY_CARD_STYLES.get(status, ELIGIBILITY_CARD_STYLES["Needs Further Investigation"])
    reasons = r.get("reasons") or []
    reasons_html = "".join(f"<li>{html.escape(str(x))}</li>" for x in reasons)
    methodology = html.escape(str(r.get("methodology", "")))
    project_type = html.escape(str(r.get("project_type", "")))
    confidence = float(r.get("confidence", 0) or 0)

    st.markdown(
        f"""
<div class="fs-elig-card" style="background: {style['bg']}; border: 1px solid {style['border']};">
  <div style="display:flex; justify-content:space-between; gap: 12px; flex-wrap: wrap;">
    <div style="font-size:16px; font-weight:700;">{project_type}</div>
    <div style="font-size:14px;"><b>{status}</b> · <b>{confidence:.0f}%</b></div>
  </div>
  <div style="margin-top:8px; font-size:13px;">
    <div style="margin-bottom:6px;"><b>Methodology:</b> {methodology}</div>
    <div><b>Reasons:</b></div>
    <ul style="margin: 6px 0 0 18px;">{reasons_html}</ul>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_home():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        col1, col2 = st.columns([1, 7])
        with col1:
            st.image(logo_path, width=90)
        with col2:
            st.markdown("<h1 style='color:#1a9850; margin-top:15px;'>FloraScope — Forestry Carbon GIS Toolkit</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='color:#1a9850'>🌿 FloraScope — Forestry Carbon GIS Toolkit</h1>", unsafe_allow_html=True)

    st.markdown(
        "<div class='fs-subtle' style='font-size: 18px; margin-bottom: 14px;'>Pre-Feasibility Screening · Audit · Canopy Analysis · GIS Tools</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_feature_card(
            "🔍 Pre-Feasibility",
            "Upload an AOI GeoJSON and run a demo analysis (LULC, NDVI, forest change) plus Gold Standard eligibility screening.",
        )
    with c2:
        _render_feature_card(
            "📋 Project Audit",
            "Audit boundary + farm plot datasets for containment, overlaps, duplicates, geometry validity and area consistency.",
        )
    with c3:
        _render_feature_card(
            "🌳 Canopy & Biomass",
            "Run demo canopy cover trend, tree count proxy, canopy height summary, and carbon stock estimates.",
        )
    with c4:
        _render_feature_card(
            "🗺️ GIS Tools",
            "Compute area/perimeter, buffer geometries, simplify vertices, and export as GeoJSON or KML.",
        )

    st.divider()
    st.success("Ready. Choose a module from the sidebar to begin.")


def page_pre_feasibility():
    st.session_state["project_name"] = st.text_input("Project Name", value=st.session_state["project_name"])

    uploaded = st.file_uploader("Upload GeoJSON", type=["geojson", "json"])
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("📂 Use Sample GeoJSON"):
            try:
                with open("sample_aoi.geojson", "r", encoding="utf-8") as f:
                    st.session_state["geojson"] = json.load(f)
                st.success("Loaded sample GeoJSON.")
            except Exception as e:
                st.error(f"Could not load `sample_aoi.geojson`: {e}")

    if uploaded is not None:
        try:
            st.session_state["geojson"] = _load_geojson_from_uploader(uploaded)
            st.success("GeoJSON uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to read uploaded GeoJSON: {e}")

    geojson_data = st.session_state.get("geojson")
    if not geojson_data:
        st.info("Upload a GeoJSON or use the sample to see demo results.")
        return

    try:
        validated = validate_geojson(geojson_data)
        if not validated.get("valid"):
            st.error(validated.get("error", "Invalid GeoJSON"))
            return

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Approx. Area (ha)", f"{validated.get('area_ha_approx', 0):,.2f}")
        with m2:
            st.metric("Centroid Latitude", f"{validated.get('centroid', {}).get('lat', 0):.6f}")
        with m3:
            st.metric("Centroid Longitude", f"{validated.get('centroid', {}).get('lon', 0):.6f}")

        st.session_state["geojson"] = validated.get("geojson")
    except Exception as e:
        st.error(f"Validation failed: {e}")
        return

    if st.button("🚀 Run Demo Analysis"):
        try:
            with st.spinner("Analysing..."):
                analysis = generate_demo_analysis(st.session_state["geojson"])
                eligibility = run_eligibility_assessment(analysis)
                st.session_state["analysis"] = analysis
                st.session_state["eligibility"] = eligibility
            st.success("Demo analysis completed.")
            st.balloons()
        except Exception as e:
            st.error(f"Analysis failed: {e}")

    analysis = st.session_state.get("analysis")
    eligibility = st.session_state.get("eligibility")
    if not analysis or not eligibility:
        return

    tabs = st.tabs(["🗺️ Land Use", "📈 NDVI Trend", "🌲 Forest Loss", "✅ Eligibility", "🌍 Site Info"])

    with tabs[0]:
        try:
            lulc_ts = analysis.get("lulc_timeseries", {}) or {}
            years = sorted(lulc_ts.keys(), key=lambda x: int(x))
            if not years:
                st.info("No LULC time series available.")
            else:
                classes = list(LULC_COLORS.keys())
                fig = go.Figure()
                for cls in classes:
                    fig.add_trace(
                        go.Bar(
                            name=cls,
                            x=years,
                            y=[(lulc_ts.get(y, {}) or {}).get(cls, 0) for y in years],
                            marker_color=LULC_COLORS.get(cls, "#cccccc"),
                        )
                    )
                fig.update_layout(barmode="stack", template="plotly_white", height=450, legend_title_text="Class")
                st.plotly_chart(fig, use_container_width=True)
                st.success("Land use chart generated.")
        except Exception as e:
            st.error(f"Could not render Land Use tab: {e}")

    with tabs[1]:
        try:
            ndvi_ts = analysis.get("ndvi_timeseries", {}) or {}
            years = sorted(ndvi_ts.keys(), key=lambda x: int(x))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=[ndvi_ts.get(y) for y in years],
                    mode="lines+markers",
                    line=dict(color="#1a9850", width=3),
                )
            )
            fig.update_layout(template="plotly_white", height=420, yaxis_title="NDVI", xaxis_title="Year")
            st.plotly_chart(fig, use_container_width=True)
            st.success("NDVI chart generated.")
        except Exception as e:
            st.error(f"Could not render NDVI tab: {e}")

    with tabs[2]:
        try:
            loss = (analysis.get("hansen_forest", {}) or {}).get("loss_by_year", {}) or {}
            years = sorted(loss.keys())
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[str(y) for y in years],
                    y=[loss.get(y, 0) for y in years],
                    marker_color="#d73027",
                )
            )
            fig.update_layout(template="plotly_white", height=420, yaxis_title="Forest loss (ha)", xaxis_title="Year")
            st.plotly_chart(fig, use_container_width=True)
            st.success("Forest loss chart generated.")
        except Exception as e:
            st.error(f"Could not render Forest Loss tab: {e}")

    with tabs[3]:
        try:
            results = eligibility.get("results", []) or []
            if not results:
                st.info("No eligibility results found.")
            else:
                for r in results:
                    _render_eligibility_card(r)
                st.success("Eligibility cards rendered.")
        except Exception as e:
            st.error(f"Could not render Eligibility tab: {e}")

    with tabs[4]:
        try:
            t = analysis.get("terrain", {}) or {}
            c = analysis.get("climate", {}) or {}
            s = analysis.get("soil_carbon", {}) or {}
            p = analysis.get("protected_areas", {}) or {}

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Mean Elevation (m)", t.get("mean_elevation_m", 0))
                st.metric("Mean Slope (deg)", t.get("mean_slope_deg", 0))
            with r2:
                st.metric("Annual Rainfall (mm)", c.get("annual_rainfall_mm", 0))
                st.metric("Mean Temp (°C)", c.get("mean_temperature_c", 0))
            with r3:
                st.metric("Mean SOC (g/kg)", s.get("mean_soc_g_per_kg", 0))
                st.metric("SOC Max (g/kg)", s.get("max_soc_g_per_kg", 0))
            with r4:
                st.metric("Protected Overlap", "Yes" if p.get("overlaps_protected_area") else "No")
                st.metric("Protected Area Count", p.get("protected_area_count", 0))

            st.divider()
            st.markdown(eligibility.get("narrative", ""))
            st.success("Site info rendered.")
        except Exception as e:
            st.error(f"Could not render Site Info tab: {e}")

    st.divider()
    if st.button("📥 Download PDF Report"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                generate_report(analysis, eligibility, f.name, st.session_state["project_name"])
                with open(f.name, "rb") as rf:
                    pdf_bytes = rf.read()
            st.success("PDF generated.")
            st.download_button("Download PDF", pdf_bytes, "report.pdf", "application/pdf")
        except Exception as e:
            st.error(f"Failed to generate PDF report: {e}")


def page_project_audit():
    st.session_state["project_name"] = st.text_input("Project Name", value=st.session_state["project_name"])
    col1, col2 = st.columns(2)

    with col1:
        boundary_up = st.file_uploader("Project Boundary GeoJSON", type=["geojson", "json"], key="audit_boundary")
    with col2:
        farms_up = st.file_uploader("Farm Plots GeoJSON", type=["geojson", "json"], key="audit_farms")

    buffer_distance_m = st.number_input("Buffer Distance (m)", value=15.0, min_value=0.0)

    boundary_geojson = None
    farms_geojson = None
    try:
        boundary_geojson = _load_geojson_from_uploader(boundary_up) if boundary_up else None
        farms_geojson = _load_geojson_from_uploader(farms_up) if farms_up else None
    except Exception as e:
        st.error(f"Could not read uploaded files: {e}")

    if st.button("🔍 Run Audit"):
        if not boundary_geojson or not farms_geojson:
            st.error("Please upload both the project boundary and farm plots GeoJSON files.")
            return
        try:
            with st.spinner("Running audit..."):
                result = run_full_audit(boundary_geojson, farms_geojson, st.session_state["project_name"], buffer_distance_m)
            if result.get("error"):
                st.error(result["error"])
                return
            st.success("Audit completed.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Issues", result.get("total_issues", 0))
            m2.metric("Critical", result.get("critical_count", 0))
            m3.metric("Warnings", result.get("warning_count", 0))
            m4.metric("Pass", result.get("pass_count", 0))

            st.divider()
            for issue in result.get("issues", []) or []:
                sev = str(issue.get("severity", "")).lower()
                if sev == "critical":
                    label = "🔴 CRITICAL"
                elif sev == "warning":
                    label = "🟡 WARNING"
                elif sev == "pass":
                    label = "🟢 PASS"
                else:
                    label = "🔵 INFO"
                with st.expander(f"{label} — {issue.get('check_name', 'Check')}"):
                    st.write(issue.get("message", ""))
                    st.write(f"Affected features: {issue.get('affected_features', [])}")
                    st.write(f"Area: {issue.get('area_ha', 0)} ha")
                    details = issue.get("details")
                    if details:
                        st.json(details)
        except Exception as e:
            st.error(f"Audit failed: {e}")


def page_canopy_biomass():
    uploaded = st.file_uploader("Upload GeoJSON", type=["geojson", "json"], key="canopy_geojson_up")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if st.button("📂 Use Sample GeoJSON", key="canopy_sample"):
            try:
                with open("sample_aoi.geojson", "r", encoding="utf-8") as f:
                    st.session_state["canopy_geojson"] = json.load(f)
                st.success("Loaded sample GeoJSON.")
            except Exception as e:
                st.error(f"Could not load `sample_aoi.geojson`: {e}")

    if uploaded is not None:
        try:
            st.session_state["canopy_geojson"] = _load_geojson_from_uploader(uploaded)
            st.success("GeoJSON uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to read uploaded GeoJSON: {e}")

    geojson_data = st.session_state.get("canopy_geojson")
    if not geojson_data:
        st.info("Upload a GeoJSON or use the sample to run the demo canopy analysis.")
        return

    area_ha = 0.0
    try:
        validated = validate_geojson(geojson_data)
        if not validated.get("valid"):
            st.error(validated.get("error", "Invalid GeoJSON"))
            return
        area_ha = float(validated.get("area_ha_approx", 0) or 0)
        st.metric("Approx. Area (ha)", f"{area_ha:,.2f}")
        st.session_state["canopy_geojson"] = validated.get("geojson")
    except Exception as e:
        st.error(f"Could not compute area: {e}")
        return

    if st.button("🌿 Run Canopy Analysis"):
        try:
            with st.spinner("Analysing canopy..."):
                result = demo_canopy_analysis(area_ha)
                st.session_state["canopy_result"] = result
            st.success("Canopy analysis completed.")
            st.balloons()
        except Exception as e:
            st.error(f"Canopy analysis failed: {e}")

    result = st.session_state.get("canopy_result")
    if not result:
        return

    tabs = st.tabs(["🌿 Canopy Cover", "🌳 Tree Count", "📏 Canopy Height", "💨 Carbon Stock"])

    with tabs[0]:
        try:
            data = (((result.get("canopy_cover") or {}).get("hansen_adjusted_timeseries")) or {})
            years = sorted(data.keys())
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[str(y) for y in years],
                    y=[(data.get(y) or {}).get("mean_canopy_pct", 0) for y in years],
                    mode="lines+markers",
                    line=dict(color="#1a9850", width=3),
                )
            )
            fig.update_layout(template="plotly_white", height=420, yaxis_title="Mean canopy (%)", xaxis_title="Year")
            st.plotly_chart(fig, use_container_width=True)
            st.success("Canopy cover chart generated.")
        except Exception as e:
            st.error(f"Could not render canopy cover: {e}")

    with tabs[1]:
        try:
            data = (((result.get("tree_count") or {}).get("combined_estimate")) or {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Estimated Tree Count", data.get("estimated_tree_count", 0))
            c2.metric("Trees / Hectare", data.get("trees_per_hectare", 0))
            c3.metric("Canopy Cover (%)", data.get("canopy_cover_pct", 0))
            c4.metric("Total Area (ha)", data.get("total_area_ha", 0))
            st.success("Tree count metrics rendered.")
        except Exception as e:
            st.error(f"Could not render tree count: {e}")

    with tabs[2]:
        try:
            data = (((result.get("canopy_height") or {}).get("eth_canopy_height")) or {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Height (m)", data.get("mean_height_m", 0))
            c2.metric("Max Height (m)", data.get("max_height_m", 0))
            c3.metric("Median Height (m)", data.get("median_height_m", 0))
            st.success("Canopy height metrics rendered.")
        except Exception as e:
            st.error(f"Could not render canopy height: {e}")

    with tabs[3]:
        try:
            data = (((result.get("biomass") or {}).get("carbon_stock")) or {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Carbon / ha (tC)", data.get("mean_carbon_per_ha_tC", 0))
            c2.metric("Total Carbon (tC)", data.get("total_carbon_tC", 0))
            c3.metric("Mean CO2e / ha", data.get("mean_co2e_per_ha", 0))
            c4.metric("Total CO2e", data.get("total_co2e", 0))
            st.success("Carbon stock metrics rendered.")
        except Exception as e:
            st.error(f"Could not render carbon stock: {e}")


def page_gis_tools():
    uploaded = st.file_uploader("Upload GeoJSON", type=["geojson", "json"], key="gis_upload")
    if not uploaded:
        st.info("Upload a GeoJSON file to use the GIS tools.")
        return

    geojson_data = None
    try:
        geojson_data = _load_geojson_from_uploader(uploaded)
        st.success("GeoJSON uploaded successfully.")
    except Exception as e:
        st.error(f"Failed to read uploaded GeoJSON: {e}")
        return

    try:
        area_info = calculate_area(geojson_data) or {}
        if area_info.get("error"):
            st.error(area_info["error"])
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Area (ha)", area_info.get("area_hectares", 0))
        c2.metric("Area (acres)", area_info.get("area_acres", 0))
        c3.metric("Area (sq km)", area_info.get("area_sq_km", 0))

        r1, r2 = st.columns(2)
        r1.metric("Area (sq m)", area_info.get("area_sq_m", 0))
        r2.metric("Perimeter (km)", area_info.get("perimeter_km", 0))

        st.success("Area calculated successfully.")
    except Exception as e:
        st.error(f"Area calculation failed: {e}")
        return

    st.divider()

    st.subheader("Buffer Tool")
    distance_m = st.slider("Buffer Distance (m)", 10, 5000, 100)
    if st.button("Apply Buffer"):
        try:
            buffered = buffer_geometry(geojson_data, float(distance_m))
            area_info2 = calculate_area(buffered) or {}
            st.success(f"Buffered area: {area_info2.get('area_hectares', 0)} ha")
            st.download_button(
                "⬇ Download Buffered GeoJSON",
                json.dumps(buffered),
                "buffered.geojson",
                "application/json",
            )
        except Exception as e:
            st.error(f"Buffer failed: {e}")

    st.divider()

    st.subheader("Simplify Tool")
    tolerance = st.slider("Tolerance", 0.00001, 0.001, 0.0001, step=0.00001, format="%.5f")
    if st.button("Simplify"):
        try:
            simplified = simplify_geometry(geojson_data, float(tolerance))
            a, b = st.columns(2)
            a.metric("Original Vertices", (simplified.get("properties") or {}).get("original_vertices", 0))
            b.metric("Simplified Vertices", (simplified.get("properties") or {}).get("simplified_vertices", 0))
            st.success("Geometry simplified successfully.")
            st.download_button(
                "⬇ Download Simplified GeoJSON",
                json.dumps(simplified),
                "simplified.geojson",
                "application/json",
            )
        except Exception as e:
            st.error(f"Simplify failed: {e}")

    st.divider()

    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Export as KML"):
            try:
                kml_str = export_to_kml(geojson_data, "FloraScope Export")
                st.success("KML generated successfully.")
                st.download_button(
                    "⬇ Download KML",
                    kml_str,
                    "export.kml",
                    "application/vnd.google-earth.kml+xml",
                )
            except Exception as e:
                st.error(f"KML export failed: {e}")
    with col2:
        if st.button("⬇ Export as GeoJSON"):
            try:
                geojson_str = export_to_geojson(geojson_data)
                st.success("GeoJSON export generated successfully.")
                st.download_button(
                    "⬇ Download GeoJSON",
                    geojson_str,
                    "export.geojson",
                    "application/json",
                )
            except Exception as e:
                st.error(f"GeoJSON export failed: {e}")


def main():
    _ensure_session_defaults()

    st.sidebar.markdown("### FloraScope")
    page = st.sidebar.radio("Navigation", NAV_OPTIONS)

    if page == "🏠 Home":
        page_home()
    elif page == "🔍 Pre-Feasibility":
        page_pre_feasibility()
    elif page == "📋 Project Audit":
        page_project_audit()
    elif page == "🌳 Canopy & Biomass":
        page_canopy_biomass()
    elif page == "🗺️ GIS Tools":
        page_gis_tools()
    else:
        page_home()


if __name__ == "__main__":
    main()

