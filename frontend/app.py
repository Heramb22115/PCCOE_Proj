import streamlit as st
import requests
import json 
from streamlit_folium import st_folium
import folium
import pandas as pd
import altair as alt
from io import BytesIO
from streamlit_js_eval import get_geolocation 

# --- Page Configuration ---
st.set_page_config(page_title="Reforestation Dashboard", page_icon="🌳", layout="wide")

# --- API Endpoints ---
API_HEALTH_URL = "http://localhost:8000/api/health"
API_REPORT_URL = "http://localhost:8000/api/get-full-report"
API_ZONE_URL = "http://localhost:8000/api/register-zone"
API_CARBON_URL = "http://localhost:8000/api/estimate-carbon-credits"
API_PDF_URL = "http://localhost:8000/api/get-pdf-report"
API_FIRE_URL = "http://localhost:8000/api/trigger-fire-check"
API_GET_ZONES_URL = "http://localhost:8000/api/get-all-zones"
API_NGO_URL = "http://localhost:8000/api/find-local-ngos"
API_CHAT_URL = "http://localhost:8000/api/ask-basic-bot"
API_VERIFY_URL = "http://localhost:8000/api/verify-plot"
API_LEADERBOARD_URL = "http://localhost:8000/api/get-leaderboard"

# --- Mock Data Catalog (for map centering) ---
MOCK_SITE_COORDS = {
    "Sanjay Park, India (Degraded)": [19.2296, 72.8711],
    "Tadoba National Park (Protected)": [20.2505, 79.3377],
    "Solapur Barren Land (Wasteland)": [17.6599, 75.9064]
}
DEFAULT_SITE = "Sanjay Park, India (Degraded)"
LIVE_DATA_MODE = "🛰️ Live Data (Manual/Clicked)"

# --- Session State Initialization ---
if 'last_report_data' not in st.session_state:
    st.session_state.last_report_data = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = MOCK_SITE_COORDS[DEFAULT_SITE]
if 'lat' not in st.session_state:
    st.session_state.lat = MOCK_SITE_COORDS[DEFAULT_SITE][0]
if 'lon' not in st.session_state:
    st.session_state.lon = MOCK_SITE_COORDS[DEFAULT_SITE][1]
if 'data_mode_select' not in st.session_state:
    st.session_state.data_mode_select = DEFAULT_SITE
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Main Dashboard ---
st.title("🌳 AI-Driven Reforestation Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Smart Site Report", 
    "📈 My Carbon Projects",
    "🤝 Find Local NGOs",
    "🏆 Community Leaderboard",
    "🤖 Basic Helpbot",
    "🛠️ Developer Tools"
])

# --- Helper Function for Map ---
def create_map(center_lat, center_lon, zoom=10):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
    folium.Marker(
        [center_lat, center_lon],
        popup="Selected Location", tooltip="Selected Location"
    ).add_to(m)
    return st_folium(m, width=700, height=400, returned_objects=[])

# --- Tab 1: Smart Site Report ---
with tab1:
    st.header("Smart Site Suitability Report")
    
    col1, col2 = st.columns([0.6, 0.4]) # Give map more space
    
    with col1:
        st.subheader("1. Select a Location")
        
        data_mode_options = [LIVE_DATA_MODE] + list(MOCK_SITE_COORDS.keys())
        data_mode = st.selectbox(
            "Select Data Mode",
            options=data_mode_options,
            key="data_mode_select",
            index=data_mode_options.index(st.session_state.data_mode_select) # Maintain state
        )

        if data_mode == LIVE_DATA_MODE:
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.session_state.lat = st.number_input("Latitude", value=st.session_state.lat, format="%.6f")
            with c2:
                st.session_state.lon = st.number_input("Longitude", value=st.session_state.lon, format="%.6f")
            with c3:
                st.markdown("    ") # Vertical alignment
                if st.button("📍 Get My Current Location"):
                    with st.spinner("Requesting browser location..."):
                        try:
                            location = get_geolocation()
                            if location:
                                st.session_state.lat = location['coords']['latitude']
                                st.session_state.lon = location['coords']['longitude']
                                st.session_state.map_center = [st.session_state.lat, st.session_state.lon]
                                st.success("Location found!")
                                st.rerun()
                            else:
                                st.error("Could not get location. Please allow permissions.")
                        except:
                            st.error("Error: Location requests may be blocked on non-HTTPS sites.")
            
            st.session_state.map_center = [st.session_state.lat, st.session_state.lon]

        else: # Mock Site Selected
            st.session_state.map_center = MOCK_SITE_COORDS[data_mode]
            st.session_state.lat = MOCK_SITE_COORDS[data_mode][0]
            st.session_state.lon = MOCK_SITE_COORDS[data_mode][1]

        # --- Interactive Map ---
        st.write(f"**Selected Coordinates:** `{st.session_state.lat:.4f}, {st.session_state.lon:.4f}`")
        map_data = create_map(st.session_state.map_center[0], st.session_state.map_center[1])
        
        if map_data and map_data["last_clicked"]:
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            if (clicked_lat != st.session_state.lat or clicked_lon != st.session_state.lon):
                st.session_state.lat = clicked_lat
                st.session_state.lon = clicked_lon
                st.session_state.map_center = [clicked_lat, clicked_lon]
                st.session_state.data_mode_select = LIVE_DATA_MODE
                st.rerun() 

        # --- Report Generation Form ---
        if st.button("🌱 Generate Full Report", type="primary", use_container_width=True):
            st.session_state.last_report_data = None
            dev_mode = data_mode != LIVE_DATA_MODE
            mock_site = data_mode if dev_mode else None
            
            spinner_text = (f"Fetching MOCK data for {mock_site}..." if dev_mode 
                            else "Fetching LIVE data from all APIs...")

            with st.spinner(spinner_text):
                payload = {
                    "latitude": st.session_state.lat,
                    "longitude": st.session_state.lon,
                    "dev_mode": dev_mode,
                    "mock_site": mock_site
                }
                try:
                    response = requests.post(API_REPORT_URL, json=payload, timeout=60)
                    if response.status_code == 200:
                        st.session_state.last_report_data = response.json()
                    else:
                        st.error(f"❌ Error (Status {response.status_code}):")
                        try:
                            st.json(response.json())
                        except:
                            st.text(response.text)
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to connect to backend: {e}")

    with col2:
        st.subheader("2. Review & Register")
        
        if not st.session_state.last_report_data:
            st.info("Please generate a report to see the results and register your plot.")
        
        if st.session_state.last_report_data:
            data = st.session_state.last_report_data
            
            # --- Display Report ---
            st.success("**Report Generated!**")
            
            suitability = data.get('suitability_analysis', {})
            st.metric(
                label="Suitability Assessment",
                value=suitability.get('recommendation', 'N/A')
            )
            st.caption(f"Reason: {suitability.get('reason', 'N/A')}")
            
            # --- NEW: AI Crop Suitability Ranking ---
            st.markdown("---")
            st.markdown("**AI Crop Suitability Ranking**")
            ranking = data.get('crop_recommendation', {}).get('top_5_ranking', [])
            
            if ranking:
                for item in ranking:
                    crop = item.get('crop', 'N/A').title()
                    score = item.get('score', 0)
                    st.text(f"**{crop}**")
                    st.progress(score, text=f"{score*100:.1f}% Suitability")
            else:
                st.text("No crop recommendation available.")
            
            # --- 7-Day Forecast Charts ---
            st.markdown("---")
            st.markdown("**7-Day Weather Forecast**")
            forecast = data.get('forecast_data', {})
            if forecast and forecast.get('dates'):
                forecast_df = pd.DataFrame({
                    'Date': forecast['dates'],
                    'Temperature (°C)': forecast['temp'],
                    'Rainfall (mm)': forecast['rain']
                })
                
                temp_chart = alt.Chart(forecast_df).mark_line(point=True).encode(
                    x=alt.X('Date', axis=alt.Axis(title='Date', format="%m-%d")),
                    y=alt.Y('Temperature (°C)', axis=alt.Axis(title='Temp (°C)')),
                    tooltip=['Date', 'Temperature (°C)']
                ).interactive()
                st.altair_chart(temp_chart, use_container_width=True)
                
                rain_chart = alt.Chart(forecast_df).mark_bar().encode(
                    x=alt.X('Date', axis=alt.Axis(title='Date', format="%m-%d")),
                    y=alt.Y('Rainfall (mm)', axis=alt.Axis(title='Rain (mm)')),
                    tooltip=['Date', 'Rainfall (mm)']
                ).interactive()
                st.altair_chart(rain_chart, use_container_width=True)

            # --- PDF Download Button ---
            try:
                pdf_response = requests.post(API_PDF_URL, json=data, timeout=20)
                if pdf_response.status_code == 200:
                    st.download_button(
                        label="Download Full Report as PDF",
                        data=pdf_response.content,
                        file_name="Reforestation_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("PDF generation failed.")
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")

            # --- Register Plot Form ---
            st.markdown("---")
            st.subheader("Register This Plot")
            with st.form("zone_form"):
                zone_name = st.text_input("Zone Name", value="My New Reforestation Plot")
                area_hectares = st.number_input("Area (in Hectares)", min_value=0.1, value=1.0, step=0.5)
                phone_number = st.text_input("WhatsApp Number (e.g., whatsapp:+919876543210)", 
                                             help="Must be the number you linked to the Twilio Sandbox.")
                register_button = st.form_submit_button("Register Zone & Get Alerts", type="primary", use_container_width=True)

                if register_button:
                    top_crop = ranking[0]['crop'] if ranking else 'N/A'
                    with st.spinner("Registering zone and sending confirmation..."):
                        payload = {
                            "zone_name": zone_name,
                            "phone_number": phone_number,
                            "latitude": st.session_state.lat,
                            "longitude": st.session_state.lon,
                            "recommended_crop": top_crop,
                            "area_hectares": area_hectares
                        }
                        try:
                            response = requests.post(API_ZONE_URL, json=payload, timeout=30)
                            if response.status_code == 200:
                                st.success(f"✅ Zone '{zone_name}' registered successfully!")
                                st.success("A confirmation message has been sent to your WhatsApp.")
                                st.session_state.last_report_data = None # Clear report
                            else:
                                st.error(f"❌ Error (Status {response.status_code}):")
                                st.json(response.json())
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Failed to connect to backend: {e}")

# --- Tab 2: My Carbon Projects ---
with tab2:
    st.header("📈 My Carbon Projects")
    st.write("Track the status and carbon sequestration of all your registered plots.")
    
    if st.button("Refresh My Plots"):
        st.rerun()
    
    try:
        response = requests.get(API_GET_ZONES_URL, timeout=10)
        if response.status_code == 200:
            zones = response.json()
            if not zones:
                st.info("You haven't registered any plots yet. Go to the 'Smart Site Report' tab to get started.")
            
            for zone in zones:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.subheader(zone.get('zone_name', 'Unnamed Zone'))
                    st.caption(f"ID: {zone.get('id')} | Crop: {zone.get('recommended_crop', 'N/A').title()} | Area: {zone.get('area_hectares', 0)} ha")
                
                with col2:
                    status = zone.get('status', 'N/A')
                    st.metric("Status", status)

                with col3:
                    # Calculate carbon credits
                    age_in_years = zone.get('age_in_days', 0) / 365.25
                    payload = {
                        "crop_type": zone.get('recommended_crop', 'default'),
                        "area_hectares": zone.get('area_hectares', 0),
                        "age_years": age_in_years
                    }
                    carbon_response = requests.post(API_CARBON_URL, json=payload, timeout=10)
                    if carbon_response.status_code == 200:
                        carbon_data = carbon_response.json()
                        st.metric("Carbon (Tons CO₂)", f"{carbon_data.get('carbon_credits', 0):.2f}")
                    else:
                        st.metric("Carbon (Tons CO₂)", "Error")
                
                # Verification Button
                if status == 'Pending':
                    if st.button(f"Set Status to 'Verified'", key=f"verify_{zone.get('id')}"):
                        verify_payload = {"zone_id": zone.get('id'), "new_status": "Verified"}
                        try:
                            requests.post(API_VERIFY_URL, json=verify_payload, timeout=10)
                            st.success(f"Plot {zone.get('id')} verified! It will now appear on the leaderboard.")
                            st.rerun()
                        except:
                            st.error("Failed to verify plot.")
                
                st.divider()

        else:
            st.error("Failed to fetch registered plots from the server.")
            st.json(response.json())
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")

# --- Tab 3: Find Local NGOs ---
with tab3:
    st.header("🤝 Find Local NGOs")
    st.write("Find environmental organizations near your registered plots to get help with funding and planting.")
    
    # Fetch user's registered plots to populate dropdown
    try:
        zones_response = requests.get(API_GET_ZONES_URL, timeout=10)
        if zones_response.status_code == 200:
            zones = zones_response.json()
            if not zones:
                st.warning("You must register a plot first (in the 'Smart Site Report' tab) to find nearby NGOs.")
            else:
                zone_names = {z['zone_name']: z for z in zones}
                selected_zone_name = st.selectbox("Select one of your registered plots:", options=zone_names.keys())
                
                selected_zone = zone_names[selected_zone_name]
                
                ngo_dev_mode = st.checkbox("🚀 Developer Mode (Use Mock NGO Data)", value=True)
                
                if st.button("Find Local NGOs", type="primary", use_container_width=True):
                    with st.spinner(f"Searching for NGOs within 50km of {selected_zone_name}..."):
                        payload = {
                            "latitude": selected_zone['lat'],
                            "longitude": selected_zone['lon'],
                            "dev_mode": ngo_dev_mode
                        }
                        try:
                            ngo_response = requests.post(API_NGO_URL, json=payload, timeout=20)
                            if ngo_response.status_code == 200:
                                st.success("Found local organizations!")
                                results = ngo_response.json().get('results', [])
                                if not results:
                                    st.info("No organizations found in a 50km radius.")
                                
                                for org in results:
                                    st.subheader(org.get('name', 'N/A'))
                                    st.caption(f"Rating: {org.get('rating', 'N/A')} ⭐")
                                    st.write(f"Address: {org.get('vicinity', 'N/A')}")
                                    st.divider()
                            else:
                                st.error(f"Error finding NGOs (Status {ngo_response.status_code}):")
                                st.json(ngo_response.json())
                        except Exception as e:
                            st.error(f"Error connecting to backend: {e}")
        else:
            st.error("Failed to fetch your registered plots.")

    except Exception as e:
        st.error(f"Error connecting to backend to fetch plots: {e}")

# --- Tab 4: Community Leaderboard ---
with tab4:
    st.header("🏆 Community Leaderboard")
    st.write("Top 'Verified' reforestation projects, ranked by total carbon sequestered.")
    
    if st.button("Refresh Leaderboard"):
        st.rerun()

    try:
        response = requests.get(API_LEADERBOARD_URL, timeout=10)
        if response.status_code == 200:
            leaderboard_data = response.json()
            if not leaderboard_data:
                st.info("No 'Verified' projects found. Verify a plot in 'My Carbon Projects' to see it here.")
            else:
                st.dataframe(
                    leaderboard_data,
                    column_config={
                        "zone_name": "Project Name",
                        "crop": "Primary Crop",
                        "total_carbon_tons": st.column_config.NumberColumn(
                            "Total Carbon (Tons CO₂)",
                            format="%.2f",
                            help="Total estimated carbon sequestered to date."
                        )
                    },
                    use_container_width=True
                )
        else:
            st.error("Failed to fetch leaderboard data.")
            st.json(response.json())
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")

# --- Tab 5: Basic Helpbot ---
with tab5:
    st.header("🤖 Basic Helpbot")
    st.write("Get instant answers to common questions. 100% local, no external AI.")
    
    lang = st.selectbox("Select Language", options=["en", "hi", "mr"], format_func=lambda x: {"en": "English", "hi": "हिंदी (Hindi)", "mr": "मराठी (Marathi)"}[x])
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Get user input
    prompt = st.chat_input("Ask a question (e.g., 'how to start')")
    
    if prompt:
        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_CHAT_URL, json={"message": prompt, "language": lang}, timeout=5)
                if response.status_code == 200:
                    bot_message = response.json().get("response", "Sorry, something went wrong.")
                else:
                    bot_message = "Error: Could not connect to the bot service."
            except Exception:
                bot_message = "Error: Backend is not reachable."
        
        # Add bot message to history and display
        st.session_state.chat_history.append({"role": "assistant", "content": bot_message})
        with st.chat_message("assistant"):
            st.markdown(bot_message)

# --- Tab 6: Developer Tools ---
with tab6:
    st.header("🛠️ Developer & Admin Tools")
    
    st.subheader("Fire Alert System (Manual Trigger)")
    st.write("This tool will check ALL registered zones for active fires and send alerts.")
    
    if st.button("🔥 Check All Zones for Fires"):
        with st.spinner("Triggering manual fire check... Check logs and phone for test alerts."):
            try:
                response = requests.post(API_FIRE_URL, timeout=30) 
                if response.status_code == 200:
                    st.success("✅ Fire check triggered successfully!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Error from backend (Status {response.status_code}):")
                    st.json(response.json())
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Failed to connect to backend: {e}")

    st.divider()
    
    st.subheader("Phase 1: System Status")
    with st.spinner("Checking system health..."):
        try:
            response = requests.get(API_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                st.success("✅ All systems operational.")
                st.json(response.json())
            else:
                st.error(f"❌ Backend is reachable, but database check failed.")
                st.json(response.json())
        except requests.exceptions.ConnectionError:
            st.error("❌ **CRITICAL: FastAPI Backend is unreachable.**")