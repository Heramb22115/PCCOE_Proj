import streamlit as st
import requests
import json 
from streamlit_folium import st_folium
import folium
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO
from streamlit_js_eval import get_geolocation 

# --- Page Configuration ---
st.set_page_config(page_title="Reforestation Dashboard", page_icon="🌳", layout="wide")

# --- API Endpoints ---
# Use localhost for local dev, or the service name in Docker
API_BASE_URL = "http://localhost:8000" 
API_HEALTH_URL = f"{API_BASE_URL}/api/health"
API_REPORT_URL = f"{API_BASE_URL}/api/get-full-report"
API_ZONE_URL = f"{API_BASE_URL}/api/register-zone"
API_CARBON_URL = f"{API_BASE_URL}/api/estimate-carbon-credits"
API_PDF_URL = f"{API_BASE_URL}/api/get-pdf-report"
API_FIRE_URL = f"{API_BASE_URL}/api/trigger-fire-check"

# --- Mock Data Catalog (for map centering) ---
MOCK_SITE_COORDS = {
    "Sanjay Park, India (Degraded)": [19.2296, 72.8711],
    "Tadoba National Park (Protected)": [20.2505, 79.3377],
    "Solapur Barren Land (Wasteland)": [17.6599, 75.9064]
}
DEFAULT_SITE = "Sanjay Park, India (Degraded)"
LIVE_DATA_MODE = "🛰️ Live Data (Manual/Clicked)"

# --- Session State Initialization ---
if 'recommended_crop' not in st.session_state:
    st.session_state.recommended_crop = "N/A"
if 'suitability' not in st.session_state:
    st.session_state.suitability = "N/A"
if 'last_clicked_lat' not in st.session_state:
    st.session_state.last_clicked_lat = MOCK_SITE_COORDS[DEFAULT_SITE][0]
if 'last_clicked_lon' not in st.session_state:
    st.session_state.last_clicked_lon = MOCK_SITE_COORDS[DEFAULT_SITE][1]
if 'map_center' not in st.session_state:
    st.session_state.map_center = MOCK_SITE_COORDS[DEFAULT_SITE]
if 'last_report_data' not in st.session_state:
    st.session_state.last_report_data = None
if 'last_pdf' not in st.session_state:
    st.session_state.last_pdf = None
if 'data_mode_select' not in st.session_state:
    st.session_state.data_mode_select = DEFAULT_SITE


# --- Main Dashboard ---
st.title("🌳 AI-Driven Reforestation Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Smart Site Report", 
    "✍️ Register Zone & Alerts", 
    "C Carbon Credit Estimation", 
    "🛠️ Developer Tools"
])

# --- Tab 1: Smart Site Report ---
with tab1:
    st.header("Smart Site Suitability Report")
    
    # --- Data Mode Selection ---
    st.write("Click a location on the map, select a mock site, or get your current location.")
    data_mode_options = [LIVE_DATA_MODE] + list(MOCK_SITE_COORDS.keys())
    data_mode = st.selectbox(
        "Select Data Mode",
        options=data_mode_options,
        key="data_mode_select",
    )

    # --- MODIFIED: Location Input Logic ---
    if data_mode == LIVE_DATA_MODE:
        st.info("Click the button to try and get your location, click the map, or type coordinates manually.")
        
        if st.button("📍 Get My Current Location (May be blocked by browser)"):
            with st.spinner("Requesting browser location... Please check for a popup!"):
                try:
                    location = get_geolocation()
                    if location:
                        st.session_state.last_clicked_lat = location['coords']['latitude']
                        st.session_state.last_clicked_lon = location['coords']['longitude']
                        st.session_state.map_center = [location['coords']['latitude'], location['coords']['longitude']]
                        st.success("Location found!")
                        st.rerun() # Refresh the app to update inputs
                    else:
                        st.error("Could not get location. Please allow browser permissions.")
                except Exception as e:
                    st.error(f"Error getting location. Browser security on 'http://' is likely blocking this.")
        
        # --- NEW: Manual coordinate input ---
        st.write("Or enter coordinates manually:")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.last_clicked_lat = st.number_input(
                "Latitude", 
                value=st.session_state.last_clicked_lat, 
                format="%.6f"
            )
        with col2:
            st.session_state.last_clicked_lon = st.number_input(
                "Longitude", 
                value=st.session_state.last_clicked_lon, 
                format="%.6f"
            )
        # Update map center if manual input changes
        st.session_state.map_center = [st.session_state.last_clicked_lat, st.session_state.last_clicked_lon]


    # --- Interactive Map ---
    if data_mode in MOCK_SITE_COORDS:
        # If user picks a mock site, update map center
        st.session_state.map_center = MOCK_SITE_COORDS[data_mode]
        st.session_state.last_clicked_lat = MOCK_SITE_COORDS[data_mode][0]
        st.session_state.last_clicked_lon = MOCK_SITE_COORDS[data_mode][1]

    
    m = folium.Map(location=st.session_state.map_center, zoom_start=10)
    folium.Marker(
        [st.session_state.last_clicked_lat, st.session_state.last_clicked_lon],
        popup="Selected Location", tooltip="Selected Location"
    ).add_to(m)
    map_data = st_folium(m, width=700, height=400)

    # --- Map Click Logic ---
    if map_data and map_data["last_clicked"]:
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        
        # Check if the click is different from the current state to avoid loops
        if (clicked_lat != st.session_state.last_clicked_lat or 
            clicked_lon != st.session_state.last_clicked_lon):
            
            st.session_state.last_clicked_lat = clicked_lat
            st.session_state.last_clicked_lon = clicked_lon
            st.session_state.map_center = [clicked_lat, clicked_lon]
            
            # When user clicks, always switch to Live Data mode
            if st.session_state.data_mode_select != LIVE_DATA_MODE:
                st.session_state.data_mode_select = LIVE_DATA_MODE
            
            st.rerun() 

    st.write(f"**Selected Coordinates:** `{st.session_state.last_clicked_lat:.4f}, {st.session_state.last_clicked_lon:.4f}`")

    # --- Report Generation Form ---
    with st.form("report_form"):
        report_button = st.form_submit_button("🌱 Generate Full Report")

    if report_button:
        # Clear old report data
        st.session_state.last_report_data = None
        st.session_state.last_pdf = None

        dev_mode = data_mode in MOCK_SITE_COORDS
        mock_site = data_mode if dev_mode else None
        
        spinner_text = (f"Fetching MOCK data for {mock_site}..." if dev_mode 
                        else "Fetching LIVE data from all APIs... (This may take a moment)")

        with st.spinner(spinner_text):
            payload = {
                "latitude": st.session_state.last_clicked_lat,
                "longitude": st.session_state.last_clicked_lon,
                "dev_mode": dev_mode,
                "mock_site": mock_site # FIXED: Key name was wrong
            }
            try:
                response = requests.post(API_REPORT_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    st.session_state.last_report_data = response.json()
                    # FIXED: Parse the correct flat keys from the response
                    st.session_state.recommended_crop = st.session_state.last_report_data.get('recommended_crop', 'N/A')
                    st.session_state.suitability = st.session_state.last_report_data.get('suitability_assessment', 'N/A')
                else:
                    st.error(f"❌ Error from backend (Status {response.status_code}):")
                    try:
                        st.json(response.json())
                    except:
                        st.text(response.text)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to connect to backend: {e}")

    # --- Display Report and Download Button (if report exists) ---
    if st.session_state.last_report_data:
        data = st.session_state.last_report_data
        st.subheader(f"✅ Report Generated for {data.get('location_name', 'Selected Area')}")
        
        st.markdown(f"**Suitability Assessment: `{data.get('suitability_assessment', 'N/A')}`**")
        st.markdown(f"**Recommended Crop:** `{data.get('recommended_crop', 'N/A')}`")
        
        # FIXED: This key now comes from the backend
        if data.get('is_already_registered'):
            st.warning("⚠️ **This location is already registered (within 500m of an existing zone).**")

        # PDF Download Button
        # The ReportData model in main.py must match this structure
        pdf_payload = st.session_state.last_report_data
        
        try:
            pdf_response = requests.post(API_PDF_URL, json=pdf_payload, timeout=20)
            if pdf_response.status_code == 200:
                st.session_state.last_pdf = pdf_response.content
            else:
                st.session_state.last_pdf = None
                st.error(f"PDF Generation Failed (Status {pdf_response.status_code}): {pdf_response.text}")
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")

        if st.session_state.last_pdf:
            st.download_button(
                label="Download Report as PDF",
                data=st.session_state.last_pdf,
                file_name=f"Reforestation_Report_{data.get('location_name', 'site').replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

        with st.expander("Show Full JSON Response"):
            st.json(data)

# --- Tab 2: Register Zone & Alerts ---
with tab2:
    st.header("Register Your Reforestation Zone")
    st.write("Register your selected coordinates to monitor them and receive alerts.")
    
    with st.form("zone_form"):
        # --- MODIFIED: Automatically loads coordinates from session state ---
        st.write(f"**Coordinates to Register:** `{st.session_state.last_clicked_lat:.4f}, {st.session_state.last_clicked_lon:.4f}`")
        zone_name = st.text_input("Zone Name", value="My Reforestation Plot")
        phone_number = st.text_input("Your WhatsApp Number (e.g., whatsapp:+919876543210)", 
                                     help="Must be the number you linked to the Twilio Sandbox.")
        
        register_button = st.form_submit_button("Register Zone")

    if register_button:
        with st.spinner("Registering zone and sending confirmation..."):
            payload = {
                "zone_name": zone_name,
                "phone_number": phone_number,
                # --- MODIFIED: Sends the coordinates from session state ---
                "latitude": st.session_state.last_clicked_lat,
                "longitude": st.session_state.last_clicked_lon
            }
            try:
                response = requests.post(API_ZONE_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    st.success(f"✅ Zone '{zone_name}' registered successfully!")
                    st.success("A confirmation message has been sent to your WhatsApp.")
                else:
                    st.error(f"❌ Error from backend (Status {response.status_code}):")
                    st.json(response.json())
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to connect to backend: {e}")


# --- Tab 3: Carbon Credit Estimation ---
with tab3:
    st.header("Carbon Credit Estimation")
    st.write("Estimate the potential carbon credits for your registered plot.")
    
    with st.form("carbon_form"):
        st.write(f"**Selected Crop:** `{st.session_state.recommended_crop}`")
        st.write(f"**Suitability:** `{st.session_state.suitability}`")
        
        area_hectares = st.number_input("Area to Plant (in Hectares)", min_value=0.1, value=1.0, step=0.5)
        
        # FIXED: Added Age slider
        age_years = st.slider(
            "Project Duration (Years)", 
            min_value=1, 
            max_value=50, 
            value=10, 
            help="Carbon credits are calculated based on the project's lifetime."
        )

        carbon_button = st.form_submit_button("Estimate Carbon Credits")

    if carbon_button:
        if st.session_state.suitability != "Suitable for Reforestation":
            st.error("Cannot calculate carbon credits. The selected area is not suitable for reforestation crops.")
        else:
            with st.spinner("Calculating..."):
                # FIXED: Payload matches the backend API Contract
                payload = {
                    "crop_type": st.session_state.recommended_crop,
                    "area_hectares": area_hectares,
                    "age_years": age_years
                }
                try:
                    response = requests.post(API_CARBON_URL, json=payload, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # FIXED: Parse the correct response keys
                        total_tons = data.get('carbon_credits', 0)
                        details = data.get('calculation_details', {})
                        age = details.get('age_years', age_years if age_years > 0 else 1)
                        per_year = total_tons / age
                        
                        st.subheader(f"Total Sequestration ({age} yrs): {total_tons:.2f} tonnes CO₂")
                        st.markdown(f"**Equivalent to {per_year:.2f} tonnes CO₂ / year**")
                        st.text(f"Total Credits: {data.get('carbon_credits', 0)}")
                        
                        with st.expander("Show Calculation Details"):
                            st.json(data)
                    else:
                        st.error(f"❌ Error from backend (Status {response.status_code}):")
                        st.json(response.json())
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to connect to backend: {e}")

# --- Tab 4: Developer Tools ---
with tab4:
    st.header("Developer & Admin Tools")
    
    st.subheader("Fire Alert System (Manual Trigger)")
    st.write("This tool will check ALL registered zones for active fires (using MOCK data) and send alerts.")
    
    if st.button("🔥 Test Fire Alert System"):
        with st.spinner("Checking all zones with mock fire data..."):
            try:
                # Use a long timeout, as this worker job could take time
                response = requests.post(API_FIRE_URL, timeout=300) 
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Fire check complete! {data.get('alerts_sent', 0)} test alerts sent.")
                    st.json(data)
                else:
                    st.error(f"❌ Error from backend (Status {response.status_code}):")
                    st.json(response.json())
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to connect to backend: {e}")

    st.divider()
    
    st.subheader("System Status")
    if st.button("Check System Health"):
        with st.spinner("Checking system health..."):
            try:
                response = requests.get(API_HEALTH_URL, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("database_status") == "connected":
                        st.success("✅ All systems operational.")
                    else:
                        st.error("❌ API is online, but Database is disconnected.")
                    st.json(data)
                else:
                    st.error(f"❌ Backend is reachable, but reported an error.")
                    st.json(response.json())
            except requests.exceptions.ConnectionError:
                st.error("❌ **CRITICAL: FastAPI Backend is unreachable.**")
                st.error("Is the backend service running? `docker-compose up --build`")