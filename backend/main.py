import asyncio
import io
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv # Import dotenv

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from geoalchemy2 import Geography, WKTElement
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from sqlalchemy import (Column, Float, Integer, MetaData, String, Table,
                        create_engine, func, inspect, insert, select, text)
from sqlalchemy.exc import OperationalError
from twilio.rest import Client

<<<<<<< HEAD
# --- Load Environment Variables ---
load_dotenv()

# --- API Keys & Secrets (FIXED: Loaded from .env) ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") # FIXED
=======

try:
    from googlemaps import Client as GoogleClient
except ImportError:
    GoogleClient = None

try:
    from chatbot_logic import get_bot_response
except ImportError:
    def get_bot_response(message: str, language: str):
        return "Chatbot logic file not found."

OPENWEATHER_API_KEY = "fc15e1b874046adab12c981b6f0dab30"
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password")
DB_NAME = os.getenv("DB_NAME", "reforestation")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

_MOCK_DATA_CATALOG = {
    "Tadoba National Park (Protected)": {
        "coords": {"lat": 20.2505, "lon": 79.3377},
<<<<<<< HEAD
        "land_type": "Protected Forest", # NEW: Land type flag
        "soil": {"ph": {"value": 6.8}, "N": {"value": 150.0}, "P": {"value": 22.0}, "K": {"value": 140.0}, "soc": {"value": 18.0}},
        "weather": {"temperature": 27.0, "humidity": 70.0, "rainfall": 0.2},
        "crop": "coffee" # (Would be blocked, but good for data)
=======
        "land_type": "Protected Forest",
        "soil": {"ph": {"value": 6.8}, "N": {"value": 150.0}, "P": {"value": 22.0}, "K": {"value": 140.0}, "soc": {"value": 18.2}},
        "weather": {"current": {"temperature": 27.0, "humidity": 70.0, "rainfall": 0.2},
                    "forecast": {"dates": ["2025-11-10", "2025-11-11", "2025-11-12"], "temp": [27, 28, 26], "rain": [0.2, 1.1, 0.0]}},
        "crop_ranking": [("coffee", 0.95), ("blackgram", 0.72), ("pigeonpeas", 0.65), ("jute", 0.51), ("rice", 0.40)]
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    },
    "Solapur Barren Land (Wasteland)": {
        "coords": {"lat": 17.6599, "lon": 75.9064},
        "land_type": "Wasteland",
<<<<<<< HEAD
        "soil": {"ph": {"value": 7.9}, "N": {"value": 25.0}, "P": {"value": 30.0}, "K": {"value": 35.0}, "soc": {"value": 5.0}},
        "weather": {"temperature": 32.0, "humidity": 40.0, "rainfall": 0.0},
        "crop": "chickpea"
=======
        "soil": {"ph": {"value": 7.9}, "N": {"value": 25.0}, "P": {"value": 30.0}, "K": {"value": 35.0}, "soc": {"value": 3.1}},
        "weather": {"current": {"temperature": 32.0, "humidity": 40.0, "rainfall": 0.0},
                    "forecast": {"dates": ["2025-11-10", "2025-11-11", "2025-11-12"], "temp": [32, 33, 31], "rain": [0.0, 0.0, 0.0]}},
        "crop_ranking": [("chickpea", 0.99), ("mothbeans", 0.92), ("pigeonpeas", 0.85), ("maize", 0.70), ("lentil", 0.60)]
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    },
    "Sanjay Park, India (Degraded)": {
        "coords": {"lat": 19.2296, "lon": 72.8711},
        "land_type": "Reforestation Candidate",
<<<<<<< HEAD
        "soil": {"ph": {"value": 6.8}, "N": {"value": 90.0}, "P": {"value": 42.0}, "K": {"value": 43.0}, "soc": {"value": 11.0}},
        "weather": {"temperature": 29.0, "humidity": 80.0, "rainfall": 0.5},
        "crop": "rice"
=======
        "soil": {"ph": {"value": 6.8}, "N": {"value": 90.0}, "P": {"value": 42.0}, "K": {"value": 43.0}, "soc": {"value": 8.5}},
        "weather": {"current": {"temperature": 29.0, "humidity": 80.0, "rainfall": 0.5},
                    "forecast": {"dates": ["2025-11-10", "2025-11-11", "2025-11-12"], "temp": [29, 29, 30], "rain": [0.5, 2.5, 0.1]}},
        "crop_ranking": [("rice", 0.98), ("maize", 0.88), ("jute", 0.75), ("banana", 0.68), ("blackgram", 0.52)]
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    }
}

_MOCK_FIRE_DATA = {
    "Sanjay Park, India (Degraded)": {
        "events": [{"id": "EONET_MOCK_FIRE_1", "title": "Mock Wildfire at Sanjay Park", "geometry": [{"type": "Point", "coordinates": [72.87, 19.22]}]}]
    },
    "Tadoba National Park (Protected)": {"events": []},
    "Solapur Barren Land (Wasteland)": {"events": []}
}

_MOCK_CARBON_RATES = {
    "rice": 1.5, "maize": 2.2, "jute": 2.5, "cotton": 1.8, "coconut": 3.0,
    "papaya": 1.2, "orange": 2.8, "apple": 3.5, "muskmelon": 1.1, "watermelon": 1.3,
    "grapes": 2.7, "mango": 3.2, "banana": 2.0, "pomegranate": 2.9, "lentil": 1.4,
    "blackgram": 1.3, "mungbean": 1.2, "mothbeans": 1.1, "pigeonpeas": 1.6,
    "kidneybeans": 1.7, "chickpea": 1.8, "coffee": 4.0, "default": 2.0
}
REFORESTATION_CROPS = [
    'coffee', 'coconut', 'papaya', 'orange', 'apple',
    'grapes', 'mango', 'banana', 'pomegranate'
]
<<<<<<< HEAD
# Soil Organic Carbon threshold (in g/kg) to be considered an existing forest
EXISTING_FOREST_SOC_THRESHOLD = 15.0
=======
EXISTING_FOREST_SOC_THRESHOLD = 15.0 
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4

# --- Pydantic Models ---
class PlotCoordinates(BaseModel):
    latitude: float
    longitude: float
    dev_mode: bool = False
    mock_site: Optional[str] = "Sanjay Park, India (Degraded)"

class BoundingBox(BaseModel):
    min_lon: float; min_lat: float; max_lon: float; max_lat: float
    dev_mode: bool = False
    mock_site: Optional[str] = "Sanjay Park, India (Degraded)"

class CropPredictionInputs(BaseModel):
    N: float; P: float; K: float;
    temperature: float; humidity: float; ph: float; rainfall: float

# FIXED: Matches UI Screenshot #7 (Age is optional, defaults to 10)
class CarbonCreditInputs(BaseModel):
<<<<<<< HEAD
    crop_type: str
    area_hectares: float
    age_years: int = 10 # Default to 10 years
=======
    crop_type: str; area_hectares: float; age_years: int
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4

class ZoneRegistration(BaseModel):
    zone_name: str; latitude: float; longitude: float; phone_number: str
    recommended_crop: str; area_hectares: float

class ReportData(BaseModel):
    # This model is for the PDF endpoint, needs to match the new flat structure
    report_status: str
    coordinates: Dict[str, Any]
<<<<<<< HEAD
    suitability_assessment: str
    recommended_crop: str
    fetched_soil_data: Dict[str, Any]
    fetched_weather_data: Dict[str, Any]
    # Keep the detailed objects optional for PDF generation
    crop_recommendation_details: Optional[Dict[str, Any]] = None
    suitability_analysis_details: Optional[Dict[str, Any]] = None
    location_name: Optional[str] = "N/A"
    is_already_registered: Optional[bool] = False

=======
    suitability_analysis: Dict[str, Any]
    crop_recommendation: Dict[str, Any]
    fetched_soil_data: Dict[str, Any]
    fetched_weather_data: Dict[str, Any]
    forecast_data: Optional[Dict[str, List]] = None

class NGOQuery(BaseModel):
    latitude: float
    longitude: float
    dev_mode: bool = False

class ChatbotQuery(BaseModel):
    message: str
    language: str = 'en'
    
class PlotVerification(BaseModel):
    zone_id: int
    new_status: str
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4

# --- FastAPI App ---
app = FastAPI(title="Reforestation Project API")
engine = None
crop_model = None
twilio_client = None
google_maps_client = None
db_metadata = MetaData()

# Define the table
registered_zones_table = Table(
    'registered_zones', db_metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('zone_name', String(255)),
    Column('phone_number', String(50)),
<<<<<<< HEAD
    Column('location', Geography(geometry_type='POINT', srid=4326))
=======
    Column('location', Geography(geometry_type='POINT', srid=4326)),
    Column('recommended_crop', String(100)),
    Column('area_hectares', Float),
    Column('registration_date', func.now()),
    Column('status', String(50), default='Pending') 
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
)

def create_tables():
    """ Creates the registered_zones table if it doesn't exist. """
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.commit()

            inspector = inspect(conn)
            if not inspector.has_table('registered_zones'):
                registered_zones_table.create(conn)
                conn.commit()
                print("Table 'registered_zones' created.")
            else:
                print("Table 'registered_zones' already exists.")
    except Exception as e:
        print(f"Error creating tables: {e}")

@app.on_event("startup")
def startup_event():
<<<<<<< HEAD
    global engine, crop_model, twilio_client

=======
    global engine, crop_model, twilio_client, google_maps_client
    
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    # Connect to DB
    retries = 5; delay = 5
    for i in range(retries):
        try:
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            with engine.connect() as connection:
                print("Database connection established successfully!")
                create_tables()
                break
        except OperationalError:
            print(f"Database connection failed. Retrying... ({i+1}/{retries})")
            time.sleep(delay)

    # Load ML Model
    try:
        crop_model = joblib.load("rf_crop_recommendation_model.pkl")
        print("Crop recommendation model loaded successfully!")
    except Exception as e: print(f"Error loading model: {e}"); crop_model = None

    # Init Twilio
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER:
        try:
            twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            print("Twilio client initialized successfully.")
        except Exception as e: print(f"Error initializing Twilio client: {e}"); twilio_client = None
    else: print("Twilio credentials not found. WhatsApp alerts will be disabled.")
<<<<<<< HEAD

    # --- NEW: Start the background worker ---
=======
    
    # Init Google Maps
    if GOOGLE_MAPS_API_KEY and GoogleClient:
        try:
            google_maps_client = GoogleClient(key=GOOGLE_MAPS_API_KEY)
            print("Google Maps client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Maps client: {e}")
            google_maps_client = None
    else:
        print("Google Maps API Key not found or library not installed. NGO finder will be disabled.")
    
    # Start the background worker
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    print("Starting background fire alert worker (30 min loop)...")
    asyncio.create_task(fire_alert_worker())

def send_whatsapp_message(to_number: str, body: str):
    if not twilio_client:
        print(f"Twilio not configured. SKIPPING message to {to_number}")
        return {"status": "skipped", "reason": "Twilio not configured"}
    try:
        message = twilio_client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=to_number)
        print(f"WhatsApp alert SENT to {to_number}")
        return {"status": "sent", "sid": message.sid}
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/", tags=["Root"])
def read_root(): return {"message": "Welcome to the AI-Driven Reforestation API!"}

@app.get("/api/health", tags=["Monitoring"])
def get_health_check():
    # Simple health check, startup handles the complex logic
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        print(f"Health check DB error: {e}")
    
    return {
        "api_status": "ok", 
        "database_status": "connected" if db_ok else "disconnected"
    }


# --- Data Acquisition Endpoints ---
@app.post("/api/get-soil-data", tags=["Data Acquisition"])
def get_soil_data(coordinates: PlotCoordinates):
    if coordinates.dev_mode:
        if coordinates.mock_site in _MOCK_DATA_CATALOG:
            return _MOCK_DATA_CATALOG[coordinates.mock_site]["soil"]
        else: raise HTTPException(status_code=404, detail=f"Mock site '{coordinates.mock_site}' not found.")

    LANDGIS_URL = "https://landgisapi.opengeohub.org/query/point"
    layers_to_query = [
        "ph.h2o_usda.4c1a2a_m_250m_b0cm_2018", "n_tot.ncs_m_250m_b0cm_2018",
        "p.ext_usda.4g1a1_m_250m_b0cm_2018", "k.ext_usda.4g1a1_m_250m_b0cm_2018",
        "soc.usda.6a1c_m_250m_b0cm_2018"
    ]
    params = {'lon': coordinates.longitude, 'lat': coordinates.latitude, 'layers': ",".join(layers_to_query)}

    try:
        r = requests.get(LANDGIS_URL, params=params, timeout=20)
        r.raise_for_status()
        return _parse_landmap_response(r.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=504, detail=f"Failed to connect to OpenLandMap API: {str(e)}")

@app.post("/api/get-weather-data", tags=["Data Acquisition"])
def get_weather_data(coordinates: PlotCoordinates):
    if coordinates.dev_mode:
        if coordinates.mock_site in _MOCK_DATA_CATALOG:
            return _MOCK_DATA_CATALOG[coordinates.mock_site]["weather"]
        else: raise HTTPException(status_code=404, detail=f"Mock site '{coordinates.mock_site}' not found.")

    if not OPENWEATHER_API_KEY: raise HTTPException(status_code=400, detail="OpenWeatherMap API key is not configured")

    WEATHER_URL = "https://api.openweathermap.org/data/3.0/onecall"
<<<<<<< HEAD
    params = {'lat': coordinates.latitude, 'lon': coordinates.longitude, 'appid': OPENWEATHER_API_KEY, 'units': 'metric', 'exclude': 'minutely,hourly,daily,alerts'}

=======
    params = {'lat': coordinates.latitude, 'lon': coordinates.longitude, 'appid': OPENWEATHER_API_KEY, 'units': 'metric', 'exclude': 'minutely,hourly,alerts'}
    
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    try:
        response = requests.get(WEATHER_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current_data = data['current']
        rainfall = current_data.get('rain', {}).get('1h', 0.0)
<<<<<<< HEAD
        return {"temperature": current_data.get('temp'), "humidity": current_data.get('humidity'), "rainfall": rainfall}
    except Exception as e:
=======
        current_weather = {"temperature": current_data.get('temp'), "humidity": current_data.get('humidity'), "rainfall": rainfall}

        forecast_temp = []
        forecast_rain = []
        forecast_dates = []
        if 'daily' in data:
            for day in data['daily'][:7]: # Get next 7 days
                forecast_dates.append(pd.to_datetime(day['dt'], unit='s').strftime('%Y-%m-%d'))
                forecast_temp.append(day['temp']['day'])
                forecast_rain.append(day.get('rain', 0.0))
        
        forecast_data = {"temp": forecast_temp, "rain": forecast_rain, "dates": forecast_dates}
        
        return {"current": current_weather, "forecast": forecast_data}
    
    except Exception as e: 
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
        raise HTTPException(status_code=504, detail=f"Failed to connect to OpenWeatherMap API: {str(e)}")

@app.post("/api/get-fire-events", tags=["Data Acquisition"])
def get_fire_events(bbox: BoundingBox):
    if bbox.dev_mode:
        return _MOCK_FIRE_DATA.get(bbox.mock_site, {"events": []})

    EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
    bbox_str = f"[{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}]"
    params = {'category': 'wildfires', 'status': 'open', 'bbox': bbox_str}

    try:
        r = requests.get(EONET_URL, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=504, detail=f"Failed to connect to NASA EONET API: {e}")

# --- AI Model Endpoint (UPGRADED for RANKING) ---
@app.post("/api/get-crop-recommendation", tags=["AI Model"])
def get_crop_recommendation(inputs: CropPredictionInputs):
    """
    Predicts a ranked list of suitable crops using predict_proba().
    """
    if crop_model is None: raise HTTPException(status_code=503, detail="Crop model is not loaded.")
    try:
<<<<<<< HEAD
        # Ensure order matches training: N, P, K, temp, humidity, ph, rainfall
        input_data = [
            inputs.N, inputs.P, inputs.K, 
            inputs.temperature, inputs.humidity, inputs.ph, inputs.rainfall
        ]
        # Create DataFrame with explicit column order to be safe
        input_df = pd.DataFrame([input_data], columns=[
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
        ])
        
        prediction = crop_model.predict(input_df)
        return {"recommended_crop": prediction[0], "input_features": inputs.dict()}
=======
        input_df = pd.DataFrame([inputs.dict()], columns=[
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
        ])
        
        probabilities = crop_model.predict_proba(input_df)[0]
        
        ranked_crops = sorted(
            zip(crop_model.classes_, probabilities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_5_crops = [
            {"crop": crop, "score": round(float(score), 4)} 
            for crop, score in ranked_crops[:5]
        ]
        
        return {
            "top_recommendation": top_5_crops[0],
            "top_5_ranking": top_5_crops,
            "input_features": inputs.dict()
        }
        
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    except Exception as e:
        print(f"Prediction error: {e}")
        if hasattr(crop_model, 'feature_names_in_'):
            print(f"Model expects: {crop_model.feature_names_in_}")
        print(f"We sent: {input_df.columns.tolist()}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Master "Smart Report" Endpoint (UPGRADED for RANKING) ---
@app.post("/api/get-full-report", tags=["Master Report"])
def get_full_report(coordinates: PlotCoordinates):
    print(f"Generating full report for {coordinates.dict()}")
    crop_recommendation = {}
    recommended_crop = None
<<<<<<< HEAD
    is_registered = False # FIXED: Flag to match UI
    suitability_analysis = {}

=======
    forecast_data = {}
    
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    if coordinates.dev_mode:
        if coordinates.mock_site not in _MOCK_DATA_CATALOG:
            raise HTTPException(status_code=404, detail="Mock site not found.")

        print(f"DEV MODE: Using mock data for {coordinates.mock_site}")
        mock_data = _MOCK_DATA_CATALOG[coordinates.mock_site]
<<<<<<< HEAD

        # Check for protected land in mock data
        if mock_data.get("land_type") == "Protected Forest":
             suitability_analysis["recommendation"] = "Not Suitable"
             suitability_analysis["reason"] = f"This location ({coordinates.mock_site}) is a protected forest and cannot be used for new planting."
             recommended_crop = "N/A"
             soil_data = mock_data["soil"]
             weather_data = mock_data["weather"]
             crop_recommendation = {"recommended_crop": "N/A", "input_features": "N/A"}
        else:
            soil_data = mock_data["soil"]
            weather_data = mock_data["weather"]
            recommended_crop = mock_data["crop"]
            model_inputs = CropPredictionInputs(
                N=soil_data.get('N').get('value'), P=soil_data.get('P').get('value'), K=soil_data.get('K').get('value'),
                temperature=weather_data.get('temperature'), humidity=weather_data.get('humidity'),
                ph=soil_data.get('ph').get('value'), rainfall=weather_data.get('rainfall')
            )
            crop_recommendation = {"recommended_crop": recommended_crop, "input_features": model_inputs.dict()}
=======
        
        if mock_data.get("land_type") == "Protected Forest":
            raise HTTPException(
                status_code=409, 
                detail=f"Site Not Suitable: This location ({coordinates.mock_site}) is a protected forest and cannot be used for new planting."
            )
            
        soil_data = mock_data["soil"]
        weather_data = mock_data["weather"]["current"]
        forecast_data = mock_data["weather"].get("forecast", {})
        
        top_5_ranking_tuples = mock_data["crop_ranking"]
        crop_recommendation = {
            "top_recommendation": {"crop": top_5_ranking_tuples[0][0], "score": top_5_ranking_tuples[0][1]},
            "top_5_ranking": [{"crop": crop, "score": score} for crop, score in top_5_ranking_tuples]
        }
        recommended_crop = top_5_ranking_tuples[0][0]
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4

    else:
        print("LIVE MODE: Checking database and fetching live data...")
        with engine.connect() as conn:
            point_wkt = f'POINT({coordinates.longitude} {coordinates.latitude})'
            stmt = select(registered_zones_table).where(
                func.ST_DWithin(
                    registered_zones_table.c.location,
                    text(f"ST_SetSRID(ST_GeomFromText('{point_wkt}'), 4326)::geography"),
                    100 
                )
            )
            existing_zone = conn.execute(stmt).first()
            # FIXED: Set flag instead of raising error, to match UI
            if existing_zone:
<<<<<<< HEAD
                is_registered = True
                print(f"Note: This location is already registered as part of the '{existing_zone.zone_name}' zone.")

        try:
            soil_data = get_soil_data(coordinates)

            # Check for existing forest based on live Soil Organic Carbon (SOC)
=======
                raise HTTPException(
                    status_code=409, 
                    detail=f"This location is already registered (within 100m) as part of the '{existing_zone.zone_name}' zone. Cannot register duplicate plot."
                )
        
        try:
            soil_data = get_soil_data(coordinates)
            weather_data_full = get_weather_data(coordinates)
            weather_data = weather_data_full["current"]
            forecast_data = weather_data_full["forecast"]
            
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
            soc_value = soil_data.get('soc', {}).get('value', 0.0)
            if soc_value is not None and soc_value > EXISTING_FOREST_SOC_THRESHOLD:
                 suitability_analysis["recommendation"] = "Not Suitable"
                 suitability_analysis["reason"] = f"Site appears to be an existing forest (Soil Organic Carbon is {soc_value} g/kg, above threshold)."
                 recommended_crop = "N/A"
                 weather_data = get_weather_data(coordinates) # Still get weather
                 crop_recommendation = {"recommended_crop": "N/A", "input_features": "N/A"}
            
            else:
                # Only run model if not a forest
                weather_data = get_weather_data(coordinates)
                model_inputs = CropPredictionInputs(
                    N=soil_data.get('N', {}).get('value') or 90.0,
                    P=soil_data.get('P', {}).get('value') or 42.0,
                    K=soil_data.get('K', {}).get('value') or 43.0,
                    temperature=weather_data.get('temperature') or 20.0,
                    humidity=weather_data.get('humidity') or 80.0,
                    ph=soil_data.get('ph', {}).get('value') or 6.5,
                    rainfall=weather_data.get('rainfall') or 200.0
                )
<<<<<<< HEAD
                crop_recommendation = get_crop_recommendation(model_inputs)
                recommended_crop = crop_recommendation.get("recommended_crop")

        except Exception as e:
            # Catch exceptions from API calls or model
            print(f"Error during live data processing: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred during live data fetching: {str(e)}")

    # Suitability Analysis (only if not already decided)
    if not suitability_analysis:
        if recommended_crop in REFORESTATION_CROPS:
            suitability_analysis["recommendation"] = "Suitable for Reforestation"
            suitability_analysis["reason"] = f"The model recommended '{recommended_crop}', which is a high-value, long-term reforestation crop."
        else:
            suitability_analysis["recommendation"] = "Suitable for Agriculture"
            suitability_analysis["reason"] = f"The model recommended '{recommended_crop}', which is a short-term agricultural crop. This area is better for farming."

    # FIXED: Return structure matches what app.py expects
    return {
        "report_status": "Success",
        "coordinates": coordinates.dict(),
        "fetched_soil_data": soil_data,
        "fetched_weather_data": weather_data,
        
        # Detailed objects for PDF
        "crop_recommendation_details": crop_recommendation,
        "suitability_analysis_details": suitability_analysis,
        
        # Flat keys for UI (Screenshot #5)
        "recommended_crop": recommended_crop,
        "suitability_assessment": suitability_analysis.get("recommendation", "N/A"),
        "location_name": coordinates.mock_site if (coordinates.dev_mode and coordinates.mock_site) else f"Site at {coordinates.latitude:.3f}, {coordinates.longitude:.3f}",
        "is_already_registered": is_registered
    }


# --- Carbon Credits Endpoint ---
@app.post("/api/estimate-carbon-credits", tags=["Carbon Credits"])
def estimate_carbon_credits(inputs: CarbonCreditInputs):
    crop_key = inputs.crop_type.lower()
    rate = _MOCK_CARBON_RATES.get(crop_key, _MOCK_CARBON_RATES["default"])
    
    # Total CO2 sequestered over the *entire* period
    total_co2_sequestered = rate * inputs.area_hectares * inputs.age_years
    
    return {
        "carbon_credits": round(total_co2_sequestered, 2), # This is the total
        "calculation_details": {
            "crop_type": inputs.crop_type,
            "sequestration_rate_per_ha_yr": rate,
            "area_hectares": inputs.area_hectares,
            "age_years": inputs.age_years,
            "total_co2_sequestered_tons": round(total_co2_sequestered, 2)
        }
    }

# --- Register Zone Endpoint (FIXED) ---
@app.post("/api/register-zone", tags=["Alerts & Registration"])
def register_zone(registration: ZoneRegistration):
    print(f"Registering new zone: {registration.zone_name}")
    try:
        with engine.connect() as conn:
            point_wkt = f'POINT({registration.longitude} {registration.latitude})'

            # NEW: Check for location proximity (100m) instead of unique name
            stmt_check = select(registered_zones_table).where(
                func.ST_DWithin(
                    registered_zones_table.c.location,
                    text(f"ST_SetSRID(ST_GeomFromText('{point_wkt}'), 4326)::geography"),
                    100 # 100 meters
                )
            )
            existing_zone = conn.execute(stmt_check).first()
            if existing_zone:
                raise HTTPException(
                    status_code=409, # 409 Conflict
                    detail=f"This location is already registered (within 100m) as part of the '{existing_zone.zone_name}' zone. Cannot register duplicate plot."
                )

            # If no conflict, insert the new zone
            stmt = insert(registered_zones_table).values(
                zone_name=registration.zone_name,
                phone_number=registration.phone_number,
                location=text(f"ST_SetSRID(ST_GeomFromText('{point_wkt}'), 4326)")
            )
            conn.execute(stmt)
            conn.commit()
            print("Zone successfully saved to database.")

    except HTTPException as e: raise e # Re-raise our own 409 error
    except Exception as e:
        print(f"Error saving to database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save zone to database: {str(e)}")

    body = (
        f"🎉 Welcome to the Reforestation Monitoring System! 🎉\n\n"
        f"You have successfully registered the zone: *{registration.zone_name}*\n\n"
        f"📍 Location: ({registration.latitude:.4f}, {registration.longitude:.4f})\n\n"
        f"You will now receive critical alerts (like fire warnings) for this zone at this number. 🔥🌲"
    )

    message_status = send_whatsapp_message(
        to_number=registration.phone_number,
        body=body
    )
    return {
        "registration_status": "success", "database_status": "saved",
        "zone_name": registration.zone_name, "message_status": message_status
    }

# --- PDF Report Endpoint ---
=======
            
            model_inputs = CropPredictionInputs(
                N=soil_data.get('N', {}).get('value') or 90.0,
                P=soil_data.get('P', {}).get('value') or 42.0,
                K=soil_data.get('K', {}).get('value') or 43.0,
                temperature=weather_data.get('temperature') or 20.0,
                humidity=weather_data.get('humidity') or 80.0,
                ph=soil_data.get('ph', {}).get('value') or 6.5,
                rainfall=weather_data.get('rainfall') or 200.0
            )
            crop_recommendation = get_crop_recommendation(model_inputs)
            recommended_crop = crop_recommendation.get("top_recommendation", {}).get("crop")
        
        except Exception as e:
            if isinstance(e, HTTPException): raise e
            raise HTTPException(status_code=500, detail=f"An error occurred during live data fetching: {str(e)}")

    suitability_analysis = {}
    if recommended_crop in REFORESTATION_CROPS:
        suitability_analysis["recommendation"] = "Suitable for Reforestation"
        suitability_analysis["reason"] = f"The model's top recommendation ('{recommended_crop}') is a high-value, long-term reforestation crop."
    else:
        suitability_analysis["recommendation"] = "Suitable for Agriculture"
        suitability_analysis["reason"] = f"The model's top recommendation ('{recommended_crop}') is a short-term agricultural crop. This area is better for farming."
    
    return {
        "report_status": "Success", "coordinates": coordinates.dict(),
        "crop_recommendation": crop_recommendation, "fetched_soil_data": soil_data,
        "fetched_weather_data": weather_data, "suitability_analysis": suitability_analysis,
        "forecast_data": forecast_data
    }

# --- PDF Report Endpoint (UPGRADED for RANKING) ---
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
@app.post("/api/get-pdf-report", tags=["Master Report"])
def get_pdf_report(report_data: ReportData):
    try:
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        y = height - inch
        p.setFont("Helvetica-Bold", 16)
        p.drawCentredString(width / 2.0, y, "Smart Site Suitability Report")
        y -= 0.5*inch

        # 1. Suitability Assessment
        p.setFont("Helvetica-Bold", 12)
        p.drawString(inch, y, "1. Suitability Assessment")
        y -= 0.25*inch

        suitability = report_data.suitability_assessment
        p.setFont("Helvetica-Bold", 11)
        p.drawString(inch * 1.2, y, f"Assessment: {suitability}")
        y -= 0.25*inch

        p.setFont("Helvetica", 10)
        # Get reason from the detailed object
        reason = report_data.suitability_analysis_details.get('reason', 'N/A')
        p.drawString(inch * 1.2, y, f"Reason: {reason}")
        y -= 0.5*inch

        # 2. AI Recommendation (Now shows ranked list)
        p.setFont("Helvetica-Bold", 12)
        p.drawString(inch, y, "2. AI Crop Suitability Ranking")
        y -= 0.25*inch

        p.setFont("Helvetica", 10)
        lat = report_data.coordinates.get('latitude', 0.0)
        lon = report_data.coordinates.get('longitude', 0.0)
        p.drawString(inch * 1.2, y, f"For Coordinates: ({lat:.4f}, {lon:.4f})")
        y -= 0.25*inch

        p.setFont("Helvetica-Bold", 11)
<<<<<<< HEAD
        crop = report_data.recommended_crop.title()
        p.drawString(inch * 1.2, y, f"Recommended Crop: {crop}")
        y -= 0.5*inch
=======
        ranking = report_data.crop_recommendation.get('top_5_ranking', [])
        if ranking:
            for i, item in enumerate(ranking):
                crop = item.get('crop', 'N/A').title()
                score = item.get('score', 0) * 100
                p.drawString(inch * 1.2, y, f"{i+1}. {crop} ({score:.1f}% suitability)")
                y -= 0.25*inch
        else:
            p.drawString(inch * 1.2, y, "No recommendation available.")
            y -= 0.25*inch
        
        y -= 0.25*inch # Extra padding
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4

        # 3. Environmental Data
        p.setFont("Helvetica-Bold", 12)
        p.drawString(inch, y, "3. Environmental Data Used")
        y -= 0.25*inch

        p.setFont("Helvetica-Bold", 11)
        p.drawString(inch * 1.2, y, "Weather Data (Current):")
        y -= 0.2*inch
        p.setFont("Helvetica", 10)
        weather = report_data.fetched_weather_data
        p.drawString(inch * 1.4, y, f"Temperature: {weather.get('temperature')} °C")
        y -= 0.2*inch
        p.drawString(inch * 1.4, y, f"Humidity: {weather.get('humidity')} %")
        y -= 0.2*inch
        p.drawString(inch * 1.4, y, f"Rainfall (1h): {weather.get('rainfall')} mm")
        y -= 0.3*inch

        p.setFont("Helvetica-Bold", 11)
        p.drawString(inch * 1.2, y, "Soil Data (Topsoil, 0-5cm):")
        y -= 0.2*inch
        p.setFont("Helvetica", 10)
        soil = report_data.fetched_soil_data
        p.drawString(inch * 1.4, y, f"Nitrogen (N): {soil.get('N', {}).get('value')} {soil.get('N', {}).get('unit', '')}")
        y -= 0.2*inch
        p.drawString(inch * 1.4, y, f"Phosphorus (P): {soil.get('P', {}).get('value')} {soil.get('P', {}).get('unit', '')}")
        y -= 0.2*inch
        p.drawString(inch * 1.4, y, f"Potassium (K): {soil.get('K', {}).get('value')} {soil.get('K', {}).get('unit', '')}")
        y -= 0.2*inch
        p.drawString(inch * 1.4, y, f"Soil pH: {soil.get('ph', {}).get('value')} {soil.get('ph', {}).get('unit', '')}")
        y -= 0.2*inch
        p.drawString(inch * 1.4, y, f"Soil Organic Carbon: {soil.get('soc', {}).get('value')} {soil.get('soc', {}).get('unit', '')}")

        p.setFont("Helvetica-Oblique", 9)
        p.drawCentredString(width / 2.0, inch * 0.75, "Report generated by AI-Driven Reforestation Planning System")

        p.showPage()
        p.save()
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="application/pdf", headers={
            "Content-Disposition": "attachment; filename=Reforestation_Report.pdf"
        })
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


# --- Carbon Credits Endpoint ---
@app.post("/api/estimate-carbon-credits", tags=["Carbon Credits"])
def estimate_carbon_credits(inputs: CarbonCreditInputs):
    crop_key = inputs.crop_type.lower()
    rate = _MOCK_CARBON_RATES.get(crop_key, _MOCK_CARBON_RATES["default"])
    total_co2_sequestered = rate * inputs.area_hectares * inputs.age_years
    return {
        "carbon_credits": round(total_co2_sequestered, 2),
        "calculation_details": {
            "crop_type": inputs.crop_type, "sequestration_rate_per_ha_yr": rate,
            "area_hectares": inputs.area_hectares, "age_years": inputs.age_years,
            "total_co2_sequestered_tons": round(total_co2_sequestered, 2)
        }
    }

# --- Register Zone Endpoint (UPGRADED) ---
@app.post("/api/register-zone", tags=["Alerts & Registration"])
def register_zone(registration: ZoneRegistration):
    print(f"Registering new zone: {registration.zone_name}")
    try:
        with engine.connect() as conn:
            point_wkt = f'POINT({registration.longitude} {registration.latitude})'
            
            stmt_check = select(registered_zones_table).where(
                func.ST_DWithin(
                    registered_zones_table.c.location,
                    text(f"ST_SetSRID(ST_GeomFromText('{point_wkt}'), 4326)::geography"),
                    100 
                )
            )
            existing_zone = conn.execute(stmt_check).first()
            if existing_zone:
                raise HTTPException(
                    status_code=409, 
                    detail=f"This location is already registered (within 100m) as part of the '{existing_zone.zone_name}' zone. Cannot register duplicate plot."
                )
            
            stmt = insert(registered_zones_table).values(
                zone_name=registration.zone_name,
                phone_number=registration.phone_number,
                location=text(f"ST_SetSRID(ST_GeomFromText('{point_wkt}'), 4326)"),
                recommended_crop=registration.recommended_crop,
                area_hectares=registration.area_hectares,
                status="Pending" 
            )
            conn.execute(stmt)
            conn.commit()
            print("Zone successfully saved to database.")
            
    except HTTPException as e: raise e 
    except Exception as e:
        print(f"Error saving to database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save zone to database: {str(e)}")

    body = (
        f"🎉 Welcome to the Reforestation Monitoring System! 🎉\n\n"
        f"You have successfully registered the zone: *{registration.zone_name}*\n"
        f"Recommended Crop: *{registration.recommended_crop}*\n"
        f"Area: *{registration.area_hectares} hectares*\n\n"
        f"You will now receive critical alerts (like fire warnings) for this zone."
    )
        
    message_status = send_whatsapp_message(to_number=registration.phone_number, body=body)
    return {
        "registration_status": "success", "database_status": "saved",
        "zone_name": registration.zone_name, "message_status": message_status
    }

# --- Get All Zones Endpoint ---
@app.get("/api/get-all-zones", tags=["Alerts & Registration"])
def get_all_zones():
    """
    Fetches all registered zones from the database for the dashboard.
    """
    try:
        with engine.connect() as conn:
            age_in_days = func.extract('epoch', func.now() - registered_zones_table.c.registration_date) / (60*60*24)
            
            stmt = select(
                registered_zones_table.c.id,
                registered_zones_table.c.zone_name,
                registered_zones_table.c.recommended_crop,
                registered_zones_table.c.area_hectares,
                registered_zones_table.c.status,
                registered_zones_table.c.registration_date,
                age_in_days.label('age_in_days'),
                func.ST_X(registered_zones_table.c.location).label('lon'),
                func.ST_Y(registered_zones_table.c.location).label('lat')
            )
            zones = conn.execute(stmt).fetchall()
            
            return [dict(zone._mapping) for zone in zones]
            
    except Exception as e:
        print(f"Error fetching all zones: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch zones: {str(e)}")

# --- NGO Finder Endpoint ---
@app.post("/api/find-local-ngos", tags=["NGOs"])
def find_local_ngos(query: NGOQuery):
    if query.dev_mode:
        print("DEV MODE: Returning mock NGO data.")
        return {
            "results": [
                {"name": "Mock Green Foundation (Pune)", "vicinity": "123 Green St, Pune", "rating": 4.5},
                {"name": "Mock Srushti Sevabhavi Sanstha", "vicinity": "456 Main Rd, Solapur", "rating": 4.8}
            ], "status": "OK_MOCK"
        }
    
    if not google_maps_client:
        raise HTTPException(status_code=503, detail="Google Maps client is not configured or library not installed.")
    
    print(f"LIVE MODE: Searching Google Maps for NGOs near ({query.latitude}, {query.longitude})")
    try:
        places_result = google_maps_client.places_nearby(
            location=(query.latitude, query.longitude),
            radius=50000, 
            keyword="environmental organization" 
        )
        return places_result
    except Exception as e:
        print(f"Google Maps API error: {e}")
        raise HTTPException(status_code=500, detail=f"Google Maps API error: {str(e)}")

# --- Basic Chatbot Endpoint ---
@app.post("/api/ask-basic-bot", tags=["Chatbot"])
def ask_basic_bot(query: ChatbotQuery):
    response = get_bot_response(query.message, query.language)
    return {"response": response}

# --- Plot Verification Endpoint ---
@app.post("/api/verify-plot", tags=["Carbon Credits"])
def verify_plot(verification: PlotVerification):
    """
    Updates the status of a plot (e.g., from 'Pending' to 'Verified').
    """
    try:
        with engine.connect() as conn:
            stmt = (
                registered_zones_table.update()
                .where(registered_zones_table.c.id == verification.zone_id)
                .values(status=verification.new_status)
            )
            result = conn.execute(stmt)
            conn.commit()
            
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail="Plot not found.")
                
            return {"status": "success", "zone_id": verification.zone_id, "new_status": verification.new_status}
    except Exception as e:
        print(f"Error updating plot status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")

# --- Leaderboard Endpoint ---
@app.get("/api/get-leaderboard", tags=["Carbon Credits"])
def get_leaderboard():
    """
    Fetches all 'Verified' plots and ranks them by their carbon value.
    """
    try:
        zones = get_all_zones()
        
        leaderboard = []
        for zone in zones:
            if zone['status'] == 'Verified':
                inputs = CarbonCreditInputs(
                    crop_type=zone['recommended_crop'],
                    area_hectares=zone['area_hectares'],
                    age_years=zone['age_in_days'] / 365.25
                )
                carbon_data = estimate_carbon_credits(inputs)
                
                leaderboard.append({
                    "zone_name": zone['zone_name'],
                    "crop": zone['recommended_crop'],
                    "total_carbon_tons": carbon_data['carbon_credits']
                })
        
        leaderboard.sort(key=lambda x: x['total_carbon_tons'], reverse=True)
        
        return leaderboard
        
    except Exception as e:
        print(f"Error generating leaderboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate leaderboard: {str(e)}")

# --- Background Fire Alert Worker ---
async def fire_alert_worker():
<<<<<<< HEAD
    """
    Runs in the background, checking for fires every 30 minutes.
    """
    await asyncio.sleep(10) # Initial delay to let server start

=======
    await asyncio.sleep(10) 
    
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    while True:
        print("WORKER: Running scheduled fire check for all registered zones...")
        alerts_sent_count = 0
        try:
            with engine.connect() as conn:
                stmt = select(
                    registered_zones_table.c.zone_name,
                    registered_zones_table.c.phone_number,
                    func.ST_X(registered_zones_table.c.location).label('lon'),
                    func.ST_Y(registered_zones_table.c.location).label('lat')
                )
                all_zones = conn.execute(stmt).fetchall()
                print(f"WORKER: Found {len(all_zones)} zone(s) to check.")

                for zone in all_zones:
                    zone_name, phone_number, lon, lat = zone
<<<<<<< HEAD

                    # Create a 50km bounding box around the zone's center point
=======
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
                    bbox = BoundingBox(
                        min_lon=lon - 0.25, min_lat=lat - 0.25,
                        max_lon=lon + 0.25, max_lat=lat + 0.25,
                        dev_mode=False 
                    )

                    print(f"WORKER: Checking for fires near '{zone_name}' ({lat:.2f}, {lon:.2f})...")
<<<<<<< HEAD

                    # Call the live EONET API
=======
                    
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
                    fire_data = get_fire_events(bbox)

                    if fire_data.get("events"):
                        num_events = len(fire_data["events"])
                        fire_title = fire_data["events"][0].get("title", "Unknown Fire")
                        print(f"WORKER: 🔥 FIRE DETECTED for '{zone_name}'! Sending alert...")

                        body = (
                            f"🔥🔥🔥 FIRE ALERT 🔥🔥🔥\n\n"
                            f"A new fire has been detected near your registered zone: *{zone_name}*.\n\n"
                            f"Event: *{fire_title}*\n"
                            f"Number of fire points detected: {num_events}\n"
                            f"Approx. Location: ({lat:.4f}, {lon:.4f})\n\n"
                            f"Please check the area and take necessary precautions."
                        )
                        send_whatsapp_message(to_number=phone_number, body=body)
                        alerts_sent_count += 1
                    else:
<<<<<<< HEAD
                        print(f"WORKER: No fires found for '{zone_name}'.")

                    # Small delay to avoid hitting API rate limits
                    await asyncio.sleep(5)

        except Exception as e:
            print(f"WORKER: Error during fire check: {e}")

        # Wait for 30 minutes before running again
        print(f"WORKER: Fire check complete. {alerts_sent_count} alerts sent. Sleeping for 30 minutes...")
        await asyncio.sleep(1800) # 1800 seconds = 30 minutes

# --- Manual Trigger for Fire Check (for testing) ---
@app.post("/api/trigger-fire-check", tags=["Alerts & Registration"])
async def trigger_fire_check():
    """
    Manually triggers a *single run* of the fire check worker for testing.
    This will run in the background and return a status.
    """
=======
                        print(f"WORKD_IN_PROG: No fires found for '{zone_name}'.")
                    await asyncio.sleep(5) 
            
        except Exception as e:
            print(f"WORKER: Error during fire check: {e}")

        print("WORKER: Fire check complete. Sleeping for 30 minutes...")
        await asyncio.sleep(1800) # 30 minutes

# --- Manual Trigger for Fire Check (for testing) ---
@app.post("/api/trigger-fire-check", tags=["Alerts & Registration"])
def trigger_fire_check():
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    print("ADMIN: Manual fire check triggered.")
    # Run the check one time, not in a loop
    alerts_sent = await fire_alert_worker_manual(use_mock=True) 
    return {
        "status": "success", 
        "message": "Manual fire check complete. Check logs and phone for test alerts.",
        "alerts_sent": alerts_sent
    }

<<<<<<< HEAD
async def fire_alert_worker_manual(use_mock: bool = False):
    """
    A single run of the fire alert worker for manual testing.
    """
=======
async def fire_alert_worker_manual():
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    print("MANUAL WORKER: Running manual fire check...")
    alerts_sent_count = 0
    try:
        with engine.connect() as conn:
            stmt = select(
                registered_zones_table.c.zone_name,
                registered_zones_table.c.phone_number,
                func.ST_X(registered_zones_table.c.location).label('lon'),
                func.ST_Y(registered_zones_table.c.location).label('lat')
            )
            all_zones = conn.execute(stmt).fetchall()
            print(f"MANUAL WORKER: Found {len(all_zones)} zone(s).")
            if not all_zones: return

<<<<<<< HEAD
            if not all_zones:
                print("MANUAL WORKER: No zones registered. Aborting test.")
                return 0

            # Test on all registered zones
            for zone in all_zones:
                zone_name, phone_number, lon, lat = zone
                
                bbox = BoundingBox(
                    min_lon=lon - 0.25, min_lat=lat - 0.25,
                    max_lon=lon + 0.25, max_lat=lat + 0.25,
                    dev_mode=use_mock,
                    mock_site="Sanjay Park, India (Degraded)" # Default mock
=======
            zone_name, phone_number, lon, lat = all_zones[0]
            
            mock_site_to_use = "Sanjay Park, India (Degraded)" 
            for site in _MOCK_DATA_CATALOG:
                mock_coords = _MOCK_DATA_CATALOG[site]["coords"]
                if abs(mock_coords["lat"] - lat) < 0.1 and abs(mock_coords["lon"] - lon) < 0.1:
                    mock_site_to_use = site
                    break
            
            print(f"MANUAL WORKER: Testing alert for zone '{zone_name}' using mock data for '{mock_site_to_use}'...")
            
            bbox = BoundingBox(
                min_lon=lon - 0.25, min_lat=lat - 0.25,
                max_lon=lon + 0.25, max_lat=lat + 0.25,
                dev_mode=True,
                mock_site=mock_site_to_use
            )
            fire_data = get_fire_events(bbox)
            
            if fire_data.get("events"):
                fire_title = fire_data["events"][0].get("title", "Unknown Fire")
                print(f"MANUAL WORKER: 🔥 MOCK FIRE DETECTED for '{zone_name}'! Sending test alert...")
                body = (
                    f"🔥🔥🔥 *TEST* FIRE ALERT 🔥🔥🔥\n\n"
                    f"This is a test of the alert system for your zone: *{zone_name}*.\n\n"
                    f"Event: *{fire_title}*\n"
                    f"Location: ({lat:.4f}, {lon:.4f})"
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
                )
                
                print(f"MANUAL WORKER: Checking zone '{zone_name}' (Mock={use_mock})...")
                fire_data = get_fire_events(bbox)

                if fire_data.get("events"):
                    fire_title = fire_data["events"][0].get("title", "Unknown Fire")
                    print(f"MANUAL WORKER: 🔥 MOCK FIRE DETECTED for '{zone_name}'! Sending test alert...")
                    body = (
                        f"🔥🔥🔥 *TEST* FIRE ALERT 🔥🔥🔥\n\n"
                        f"This is a test of the alert system for your zone: *{zone_name}*.\n\n"
                        f"Event: *{fire_title}*\n"
                        f"Location: ({lat:.4f}, {lon:.4f})"
                    )
                    send_whatsapp_message(to_number=phone_number, body=body)
                    alerts_sent_count += 1
                else:
                    print(f"MANUAL WORKER: No mock fires found for '{zone_name}'. No alert sent.")
                
                await asyncio.sleep(2) # Avoid rate limits

    except Exception as e:
        print(f"MANUAL WORKER: Error during manual fire check: {e}")
    
    return alerts_sent_count

# Helper: Parse OpenLandMap Response
def _parse_landmap_response(response_json):
    parsed_data = {}
    layers = {
        "ph": "ph.h2o_usda.4c1a2a_m_250m_b0cm_2018",
        "N": "n_tot.ncs_m_250m_b0cm_2018",
        "P": "p.ext_usda.4g1a1_m_250m_b0cm_2018",
        "K": "k.ext_usda.4g1a1_m_250m_b0cm_2018",
        "soc": "soc.usda.6a1c_m_250m_b0cm_2018"
    }
    units = {
        "ph": {"unit": "pH", "scale": 0.1},
        "N": {"unit": "g/kg", "scale": 0.01}, 
        "P": {"unit": "mg/kg", "scale": 0.1}, 
        "K": {"unit": "mg/kg", "scale": 0.1}, 
        "soc": {"unit": "g/kg", "scale": 0.1} 
    }

    if "layers" not in response_json:
        return {"error": "Invalid response from OpenLandMap", "data": response_json}

    for key, layer_name in layers.items():
        if layer_name in response_json["layers"]:
            raw_value = response_json["layers"][layer_name]["value"]
            if raw_value is not None:
                scaled_value = round(raw_value * units[key]["scale"], 2)
                parsed_data[key] = {"value": scaled_value, "unit": units[key]["unit"]}
            else:
                parsed_data[key] = {"value": None, "unit": units[key].get("unit")}
        else:
<<<<<<< HEAD
            # Layer itself wasn't found in the response
            parsed_data[key] = {"value": None, "unit": "N/A", "error": f"Layer '{layer_name}' not found in response"}

=======
            parsed_data[key] = {"value": None, "unit": "N/A", "error": f"Layer '{layer_name}' not found"}
            
>>>>>>> 6aa45908e9d9b7968563867b3d15a41259813bf4
    return parsed_data