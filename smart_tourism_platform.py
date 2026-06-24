# =============================================================================
# PROJECT DOCUMENTATION
# =============================================================================

## 1. Project Overview
# Travel Around is an end-to-end Smart Tourism Analytics Platform built fully in
# Python using Streamlit. The system recommends destinations, hotels, restaurants,
# attractions, optimized daily itineraries, travel budgets, weather forecasts,
# currency conversion, emergency contacts, and tourism analytics. It also includes
# AI/ML-style components such as a rule-based LLM trip assistant, crowd prediction,
# demand forecasting, landmark recognition demo, hotel recommendation scoring, and
# fraud-risk detection for travel bookings.

## 2. Problem Statement
# Tourists often need to switch between many separate apps for trip planning,
# weather, route planning, hotels, food, budget estimation, currency conversion,
# safety information, and local recommendations. This creates fragmented planning,
# poor budget visibility, limited personalization, and difficulty identifying safe
# and suitable tourist options. This project solves the problem by bringing major
# tourism planning and analytics services into one personalized Python platform.

## 3. Objectives
# 1. Recommend tourist destinations, hotels, restaurants, attractions, and route
#    plans based on user profile, budget, trip duration, season, and preferences.
# 2. Estimate trip budget including hotel, food, transport, attractions, shopping,
#    emergency buffer, and per-person cost.
# 3. Provide real-time-style weather forecasts, currency conversion, emergency
#    contacts, nearby attraction discovery, and itinerary generation.
# 4. Demonstrate AI/ML features: crowd prediction, demand forecasting, sentiment
#    summaries, chatbot guidance, landmark recognition demo, and booking-fraud risk.
# 5. Present analytics dashboards with maps, charts, trends, heatmaps, and decision
#    indicators useful for tourists and tourism authorities.

## 4. Target Users
# Tourists, families, solo travellers, travel agencies, hotel partners, tourism
# departments, local guides, city administrators, emergency support teams, and
# tourism analytics teams.

## 5. Functional Requirements
# - User profile and session-based trip setup
# - Destination recommendation and destination facts
# - Hotel and restaurant recommendation with score ranking
# - Budget planner and per-person cost breakdown
# - Weather forecast for trip dates
# - Nearby attractions finder
# - Day-wise smart itinerary planner
# - Navigation map with route markers
# - Currency converter
# - Emergency contacts and safety advice
# - AI travel chatbot / voice-assistant-ready response engine
# - Landmark image recognition demo
# - Crowd prediction and travel demand forecasting
# - Tourist review sentiment summary
# - Fraud detection for booking/transaction risk
# - Analytics dashboard with charts and heatmap-ready datasets

## 6. Non-Functional Requirements
# - Performance: cached deterministic data generation and fast UI response
# - Security: no hardcoded private API keys; optional API integrations via secrets
# - Scalability: modular Python functions for API, database, and ML replacement
# - Reliability: offline demo fallback datasets when external APIs are unavailable
# - Usability: simple sidebar navigation, clear metrics, and guided workflow
# - Maintainability: single Python codebase with clearly separated data, ML logic,
#   UI pages, and utility functions

## 7. AI/ML Components
# - Recommendation: weighted content-based scoring similar to collaborative filtering
# - Crowd prediction: month-wise time-series/crowd index model
# - Demand forecasting: seasonal trend projection using deterministic simulation
# - Sentiment analysis: rule-based positive/neutral/negative review classifier
# - Fraud detection: risk scoring using booking amount, rating, urgency, mismatch,
#   and payment method features
# - Landmark recognition: image upload demo using filename/metadata heuristic with
#   placeholders for CNN/CLIP/transformer model integration
# - Chatbot: rule-based LLM-style travel assistant with optional API upgrade path
# - Future model options: XGBoost, Random Forest, Transformers, BERT, Prophet/ARIMA

## 8. Challenges Faced During Building This Project
# 1. Replacing browser-only JavaScript/localStorage behavior with Python session
#    state was required. This was solved using Streamlit session_state.
# 2. Live APIs such as Google Maps, Weather API, Google Reviews, TripAdvisor, and
#    OpenStreetMap may require keys or internet access. This project includes
#    offline simulated datasets and comments where real API connectors can be added.
# 3. Converting visual JavaScript charts to Python charts was handled through
#    pandas and Streamlit chart functions so the app remains fully Python-based.

## 9. Future Enhancements
# Real Weather API, Google Places API, OpenStreetMap Routing, TripAdvisor Reviews,
# Live Hotel Booking, Voice Assistant, Multilingual Translation, Computer Vision
# Landmark Model, RAG Chatbot, Payment Gateway, Admin Panel, Mobile App.

## 10. Final Conclusion & Summary
# The project successfully converts the provided Smart Tourism HTML/JavaScript idea
# into a Python-based analytics platform. It demonstrates all requested tourism
# features, AI/ML components, dashboards, and travel-planning workflows in a single
# maintainable codebase. The offline demo is ready for academic/project evaluation
# and can be upgraded with real APIs and trained ML models for production.

# =============================================================================
# CODE STARTS BELOW
# =============================================================================

from __future__ import annotations

import calendar
import hashlib
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple, Any

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This project requires Streamlit, pandas and numpy. Install them with: "
        "pip install -r requirements.txt"
    ) from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

APP_NAME = "Travel Around — Smart Tourism Analytics Platform"
APP_ICON = "🧭"
BASE_CURRENCY = "INR"

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")


# =============================================================================
# DATA LAYER — Offline demo datasets that can later be replaced by APIs
# =============================================================================

DESTINATIONS: Dict[str, Dict[str, Any]] = {
    "Agra": {
        "lat": 27.18,
        "lon": 78.01,
        "state": "Uttar Pradesh",
        "type": "Heritage",
        "season": "Oct–Mar",
        "crowd": [55, 50, 65, 70, 80, 40, 35, 40, 50, 75, 80, 85],
        "tags": ["UNESCO", "Romantic", "Architecture", "History"],
    },
    "Jaipur": {
        "lat": 26.91,
        "lon": 75.79,
        "state": "Rajasthan",
        "type": "Heritage",
        "season": "Oct–Feb",
        "crowd": [60, 65, 70, 50, 35, 25, 30, 30, 40, 65, 80, 85],
        "tags": ["Forts", "Shopping", "Culture", "Food"],
    },
    "Goa": {
        "lat": 15.30,
        "lon": 74.12,
        "state": "Goa",
        "type": "Beach",
        "season": "Nov–Feb",
        "crowd": [30, 35, 60, 50, 40, 20, 15, 20, 25, 55, 75, 85],
        "tags": ["Beach", "Nightlife", "Seafood", "Relaxation"],
    },
    "Kerala": {
        "lat": 10.85,
        "lon": 76.27,
        "state": "Kerala",
        "type": "Nature",
        "season": "Oct–Feb",
        "crowd": [70, 65, 55, 45, 35, 30, 40, 35, 45, 65, 80, 85],
        "tags": ["Backwaters", "Ayurveda", "Nature", "Houseboat"],
    },
    "Manali": {
        "lat": 32.24,
        "lon": 77.19,
        "state": "Himachal Pradesh",
        "type": "Adventure",
        "season": "May–Jun, Dec–Jan",
        "crowd": [85, 80, 60, 40, 30, 50, 85, 90, 70, 50, 40, 75],
        "tags": ["Snow", "Adventure", "Mountains", "Trekking"],
    },
    "Mumbai": {
        "lat": 19.08,
        "lon": 72.88,
        "state": "Maharashtra",
        "type": "Urban",
        "season": "Nov–Feb",
        "crowd": [50, 55, 60, 55, 45, 40, 45, 45, 50, 60, 65, 70],
        "tags": ["Urban", "Bollywood", "Food", "Sea"],
    },
    "Varanasi": {
        "lat": 25.32,
        "lon": 82.97,
        "state": "Uttar Pradesh",
        "type": "Spiritual",
        "season": "Oct–Mar",
        "crowd": [75, 70, 65, 55, 45, 40, 35, 40, 50, 70, 85, 90],
        "tags": ["Spiritual", "Ghats", "Culture", "Boating"],
    },
    "Udaipur": {
        "lat": 24.59,
        "lon": 73.71,
        "state": "Rajasthan",
        "type": "Heritage",
        "season": "Sep–Mar",
        "crowd": [55, 60, 65, 50, 35, 25, 30, 30, 40, 60, 75, 80],
        "tags": ["Lakes", "Palaces", "Romantic", "Heritage"],
    },
    "Darjeeling": {
        "lat": 27.04,
        "lon": 88.26,
        "state": "West Bengal",
        "type": "Nature",
        "season": "Mar–May, Sep–Nov",
        "crowd": [40, 45, 75, 80, 65, 50, 30, 30, 50, 75, 60, 45],
        "tags": ["Tea", "Hills", "Toy Train", "Nature"],
    },
    "Mysuru": {
        "lat": 12.30,
        "lon": 76.64,
        "state": "Karnataka",
        "type": "Heritage",
        "season": "Oct–Feb",
        "crowd": [50, 55, 60, 50, 40, 30, 35, 35, 45, 65, 80, 70],
        "tags": ["Palace", "Silk", "Yoga", "Heritage"],
    },
    "Rishikesh": {
        "lat": 30.09,
        "lon": 78.27,
        "state": "Uttarakhand",
        "type": "Spiritual",
        "season": "Sep–Jun",
        "crowd": [60, 65, 75, 80, 70, 40, 25, 25, 50, 80, 75, 65],
        "tags": ["Yoga", "Rafting", "Spiritual", "Ganga"],
    },
    "Hampi": {
        "lat": 15.34,
        "lon": 76.46,
        "state": "Karnataka",
        "type": "Heritage",
        "season": "Oct–Feb",
        "crowd": [45, 50, 60, 55, 35, 25, 20, 20, 35, 60, 75, 70],
        "tags": ["Ruins", "UNESCO", "Boulders", "History"],
    },
}

CITY_OPTIONS = [
    "Lucknow", "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Bhopal", "Chandigarh", "Patna", "Kochi",
]

AMENITIES_POOL = [
    "WiFi", "Pool", "Gym", "Spa", "Parking", "Restaurant", "AC", "Room Service",
    "Bar", "Laundry", "Airport Transfer", "Breakfast", "Pet Friendly",
]

AIRLINE_NAMES = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Akasa Air", "Air India Express"]
HOTEL_ICONS = ["🏨", "🏩", "🏰", "🏯", "🛖", "🏠", "🌴", "⛺"]

WEATHER_PROFILES = {
    "Manali": {"base": 12, "rain": 55, "cond": ["Snowy ❄️", "Cloudy ⛅", "Foggy 🌫️", "Partly cloudy 🌥️"]},
    "Goa": {"base": 31, "rain": 70, "cond": ["Rainy 🌧️", "Humid ☁️", "Partly sunny 🌤️", "Thunderstorm ⛈️"]},
    "Kerala": {"base": 29, "rain": 65, "cond": ["Rainy 🌧️", "Overcast ☁️", "Partly cloudy 🌥️", "Drizzle 🌦️"]},
    "Jaipur": {"base": 37, "rain": 18, "cond": ["Sunny ☀️", "Hot & dry 🌵", "Clear ✨", "Hazy 🌫️"]},
    "Darjeeling": {"base": 14, "rain": 60, "cond": ["Misty 🌫️", "Cool ⛅", "Light rain 🌦️", "Clear ✨"]},
    "default": {"base": 28, "rain": 35, "cond": ["Sunny ☀️", "Partly cloudy 🌥️", "Clear ✨", "Light rain 🌦️"]},
}

REVIEWS_DB = {
    "Agra": [
        ("Priya S.", 5, "Absolutely breathtaking. The Taj at sunrise is one of the most beautiful things I have ever seen.", "Positive"),
        ("Marco T.", 4, "Stunning monument, but go early! Agra Fort is equally impressive.", "Positive"),
        ("Sneha R.", 3, "Incredible monument but very touristy surroundings with vendor pressure.", "Neutral"),
        ("Ravi K.", 5, "A spiritual experience. Unparalleled craftsmanship.", "Positive"),
        ("Emily W.", 2, "Massive queues, pushy vendors, disappointing food near the main gate.", "Negative"),
    ],
    "Jaipur": [
        ("Kavya M.", 5, "Pink City lived up to every expectation. Amber Fort alone is worth the trip.", "Positive"),
        ("Tom B.", 4, "Colourful, chaotic, wonderful. Best bazaars for textiles in India.", "Positive"),
        ("Ananya D.", 3, "Beautiful city but very hot in summer. Indoor sights are helpful escapes.", "Neutral"),
        ("Sofia P.", 5, "Best food I had in India. Rajasthani thali was outstanding.", "Positive"),
        ("Rohan N.", 2, "Traffic was difficult. Budget more time between sights.", "Negative"),
    ],
    "Goa": [
        ("Lara M.", 5, "Paradise! Beaches stunning, food incredible, Old Goa churches a hidden gem.", "Positive"),
        ("James H.", 4, "Great for relaxation. Beach shacks, fresh seafood, friendly locals.", "Positive"),
        ("Meera K.", 5, "Dudhsagar Falls was magical. Spice plantation tour was educational.", "Positive"),
        ("Oliver S.", 3, "Lovely beaches but very crowded in peak season. Accommodation overpriced.", "Neutral"),
        ("Nia T.", 2, "Monsoon season was a mistake; many beach activities were closed.", "Negative"),
    ],
}

ATTRACTIONS = {
    "Agra": [
        ("Taj Mahal", "UNESCO marble mausoleum", "🕌", 4.9, 50),
        ("Agra Fort", "UNESCO red sandstone fort", "🏰", 4.7, 50),
        ("Mehtab Bagh", "Best sunset Taj views", "🌿", 4.5, 25),
        ("Fatehpur Sikri", "Historic Mughal capital", "🏛️", 4.6, 35),
    ],
    "Jaipur": [
        ("Amber Fort", "Hilltop palace with mirror rooms", "🏰", 4.8, 200),
        ("City Palace", "Royal heritage complex", "🏛️", 4.7, 500),
        ("Hawa Mahal", "Palace of winds, iconic facade", "🏩", 4.6, 50),
        ("Jantar Mantar", "UNESCO observatory", "🔭", 4.5, 200),
    ],
    "Goa": [
        ("Basilica of Bom Jesus", "UNESCO baroque church", "⛪", 4.7, 0),
        ("Calangute Beach", "King of beaches", "🏖️", 4.3, 0),
        ("Dudhsagar Falls", "Spectacular waterfall", "💦", 4.8, 400),
        ("Fort Aguada", "Portuguese-era sea fort", "🏰", 4.5, 0),
    ],
    "Manali": [
        ("Rohtang Pass", "High-altitude mountain pass", "🏔️", 4.8, 500),
        ("Solang Valley", "Snow sports and paragliding", "⛷️", 4.7, 200),
        ("Hadimba Temple", "Unique wooden cave temple", "🛕", 4.6, 0),
        ("Beas Kund Trek", "Scenic glacier trek", "🗻", 4.8, 0),
    ],
    "Kerala": [
        ("Alleppey Backwaters", "Houseboat heaven", "🛥️", 4.9, 0),
        ("Munnar Tea Gardens", "Rolling green hills", "🍵", 4.7, 0),
        ("Periyar Wildlife", "Tiger reserve and boat safari", "🐘", 4.6, 300),
        ("Varkala Beach", "Cliff-top scenic beach", "🏖️", 4.5, 0),
    ],
}

ITINERARY_ACTS = {
    "Agra": [
        ("08:00 AM", "Taj Mahal", "Sunrise visit; book official tickets in advance.", "🕌"),
        ("11:00 AM", "Agra Fort", "Explore Musamman Burj and Mughal architecture.", "🏰"),
        ("02:00 PM", "Lunch: Mughlai cuisine", "Try biryani, kebab, and petha.", "🍛"),
        ("04:00 PM", "Mehtab Bagh", "Best sunset Taj viewing garden.", "🌿"),
        ("07:00 PM", "Sadar Bazaar", "Souvenirs, textiles, and street food.", "🛍️"),
    ],
    "Jaipur": [
        ("08:00 AM", "Amber Fort", "Go early and hire a guide for mirror hall stories.", "🏰"),
        ("11:30 AM", "City Palace", "Visit textile and royal galleries.", "🏛️"),
        ("01:30 PM", "Lunch: Rajasthani Thali", "Try dal baati churma and ghewar.", "🥘"),
        ("03:00 PM", "Hawa Mahal", "Iconic facade and photo spot.", "🏩"),
        ("06:00 PM", "Chokhi Dhani", "Village-themed dinner and cultural show.", "🎭"),
    ],
    "Goa": [
        ("07:00 AM", "Sunrise at Anjuna", "Quiet early morning beach walk.", "🌅"),
        ("10:00 AM", "Old Goa Churches", "Visit Basilica and Se Cathedral.", "⛪"),
        ("01:00 PM", "Seafood lunch", "Try Goan fish curry and bebinca.", "🐟"),
        ("04:00 PM", "Fort Aguada", "Sea views and Portuguese history.", "🏰"),
        ("07:30 PM", "Beach shack dinner", "Relaxed dinner near the coast.", "🏖️"),
    ],
}

RESTAURANTS = {
    "Agra": [
        ("Pinch of Spice", "Mughlai", 4.5, 800),
        ("Joney's Place", "Cafe / Local", 4.3, 300),
        ("Peshawri", "North Indian", 4.7, 2200),
        ("Mama Chicken", "Street Food", 4.2, 250),
    ],
    "Jaipur": [
        ("Laxmi Misthan Bhandar", "Rajasthani", 4.4, 500),
        ("Rawat Mishthan Bhandar", "Snacks", 4.3, 250),
        ("1135 AD", "Royal Dining", 4.6, 2500),
        ("Tapri Central", "Cafe", 4.5, 700),
    ],
    "Goa": [
        ("Ritz Classic", "Goan Seafood", 4.5, 900),
        ("Gunpowder", "Coastal", 4.4, 1200),
        ("Vinayak Family Restaurant", "Local Goan", 4.6, 600),
        ("Thalassa", "Greek / Beach", 4.3, 1800),
    ],
}

DEST_FACTS = {
    "Agra": ["Taj Mahal is closed on Fridays", "Sunrise slots are less crowded", "Carry shoe covers or buy them at the gate"],
    "Jaipur": ["Start forts early to avoid heat", "Bargain politely in bazaars", "Block-printed textiles are popular souvenirs"],
    "Goa": ["North Goa is lively; South Goa is quieter", "Avoid swimming during red-flag warnings", "Scooter rental requires valid licence"],
    "Manali": ["Roads to Rohtang depend on weather", "Carry warm layers even in summer", "Permits may be needed for high passes"],
    "Kerala": ["Houseboats are best booked in advance", "Monsoon gives lush views", "Ayurveda treatments vary in quality"],
}

AI_KNOWLEDGE = {
    "Agra": {
        "food": "In Agra, try Mughlai food: kebabs, biryani, bedai, jalebi, and petha. Pinch of Spice and local Sadar Bazaar stalls are popular.",
        "hotel": "Stay near Taj Ganj for easy Taj access, or Fatehabad Road for better hotel choices. Book sunrise Taj days in advance.",
        "transport": "Use prepaid taxis, e-rickshaws near monuments, or a hired cab for Agra Fort + Mehtab Bagh + Fatehpur Sikri.",
        "visit": "Visit Taj Mahal at sunrise, Agra Fort late morning, and Mehtab Bagh at sunset. Avoid Friday for Taj Mahal.",
    },
    "Jaipur": {
        "food": "Try dal baati churma, pyaaz kachori, ghewar, and lassi. LMB, Rawat, Tapri, and Chokhi Dhani are common choices.",
        "hotel": "C-Scheme and MI Road are convenient. Heritage havelis near old city offer character but may be noisy.",
        "transport": "Hire a cab for forts; use auto-rickshaws inside the old city. Add buffer time for traffic.",
        "visit": "Do Amber Fort early, then City Palace, Jantar Mantar, Hawa Mahal, and evening bazaars.",
    },
    "Goa": {
        "food": "Try Goan fish curry rice, pork vindaloo, xacuti, recheado, poi bread, and bebinca. Beach shacks are good for sunsets.",
        "hotel": "Choose North Goa for nightlife and South Goa for peace. Families often prefer Candolim, Panjim, or South Goa.",
        "transport": "Rent a scooter only with licence and helmet. Taxis can be expensive, so plan grouped routes.",
        "visit": "Combine beaches with Old Goa, Fort Aguada, spice plantation, and Dudhsagar if weather permits.",
    },
}

GENERIC_AI = [
    "Plan high-energy sightseeing in the morning, indoor attractions after lunch, and markets or food walks in the evening.",
    "Keep 15–20% of your budget as emergency buffer for taxis, weather changes, and last-minute tickets.",
    "For food safety, prefer busy restaurants, bottled water, and freshly cooked street food.",
    "Use offline maps, keep ID copies, and share your itinerary with a trusted contact.",
]

RATES = {
    "INR": 1.0,
    "USD": 0.012,
    "EUR": 0.011,
    "GBP": 0.0095,
    "JPY": 1.78,
    "CAD": 0.016,
    "AUD": 0.018,
    "AED": 0.044,
    "SGD": 0.016,
    "CNY": 0.086,
    "THB": 0.42,
}

CURRENCY_NAMES = {
    "INR": "🇮🇳 Indian Rupee",
    "USD": "🇺🇸 US Dollar",
    "EUR": "🇪🇺 Euro",
    "GBP": "🇬🇧 British Pound",
    "JPY": "🇯🇵 Japanese Yen",
    "CAD": "🇨🇦 Canadian Dollar",
    "AUD": "🇦🇺 Australian Dollar",
    "AED": "🇦🇪 UAE Dirham",
    "SGD": "🇸🇬 Singapore Dollar",
    "CNY": "🇨🇳 Chinese Yuan",
    "THB": "🇹🇭 Thai Baht",
}

EMERGENCY_CONTACTS = [
    ("Police", "112", "24/7 national emergency response", "🚔"),
    ("Ambulance", "108", "Emergency medical ambulance", "🚑"),
    ("Fire Brigade", "101", "Fire emergency response", "🚒"),
    ("Tourist Helpline", "1363", "Dedicated tourist assistance", "📞"),
    ("Women Helpline", "1091", "Women safety and support", "👩"),
    ("Child Helpline", "1098", "Child protection services", "👶"),
    ("Disaster Management", "1070", "State disaster management line", "⚠️"),
    ("Cyber Crime", "1930", "Online fraud and cybercrime", "💻"),
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@dataclass
class TripProfile:
    name: str
    origin: str
    dest: str
    start: date
    end: date
    people: int
    budget: str
    interests: List[str]
    travel_style: str


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def seeded_rand(seed: int) -> float:
    x = math.sin(seed + 1) * 10000
    return x - math.floor(x)


def seeded_int(seed: int, minimum: int, maximum: int) -> int:
    return int(seeded_rand(seed) * (maximum - minimum + 1)) + minimum


def rupee(value: float | int) -> str:
    return "₹" + f"{int(round(value)):,}"


def trip_days(start: date, end: date) -> int:
    return max(1, (end - start).days + 1)


def month_name(month_index: int) -> str:
    return calendar.month_abbr[month_index + 1]


def crowd_label(value: int) -> str:
    if value < 40:
        return "🟢 Low"
    if value < 70:
        return "🟡 Moderate"
    return "🔴 High"


def get_trip() -> TripProfile | None:
    data = st.session_state.get("trip")
    if not data:
        return None
    return TripProfile(**data)


def save_trip(profile: TripProfile) -> None:
    st.session_state["trip"] = profile.__dict__


def current_user_name() -> str:
    return st.session_state.get("user_name", "Demo Tourist")


def get_budget_factor(budget: str) -> float:
    return {"Budget": 0.75, "Standard": 1.0, "Premium": 1.45, "Luxury": 2.25}.get(budget, 1.0)


# =============================================================================
# DATA GENERATORS AND AI/ML DEMO LOGIC
# =============================================================================

@st.cache_data(show_spinner=False)
def generate_hotels(dest: str) -> pd.DataFrame:
    bases = [
        f"{dest} Palace", f"Grand {dest}", "Heritage Inn", "Royal Suites",
        "Boutique Stay", "Comfort Inn", "The Residency", "Backpacker Hostel",
    ]
    rows = []
    for idx, base in enumerate(bases):
        seed = stable_hash(dest + str(idx))
        stars = max(1, min(5, 5 - idx // 2))
        price = int(seeded_int(seed, 800, 18000) * (stars / 3))
        rating = round(3 + seeded_rand(seed + 1) * 1.95, 1)
        amenities = [a for j, a in enumerate(AMENITIES_POOL) if seeded_rand(seed + j + 10) > 0.45]
        amenities = amenities[: seeded_int(seed + 5, 4, 8)] or ["WiFi", "AC", "Restaurant"]
        rows.append(
            {
                "Icon": HOTEL_ICONS[idx % len(HOTEL_ICONS)],
                "Hotel": base,
                "Stars": stars,
                "Price/Night": price,
                "Rating": rating,
                "Reviews": seeded_int(seed + 2, 30, 1500),
                "Distance_km": round(seeded_rand(seed + 3) * 8.5 + 0.3, 1),
                "Amenities": ", ".join(amenities),
                "AI Pick": rating > 4.3 and stars >= 3,
            }
        )
    df = pd.DataFrame(rows)
    df["Recommendation Score"] = (
        df["Rating"] * 22
        + df["Stars"] * 8
        + (1 / (df["Distance_km"] + 0.5)) * 16
        + (df["AI Pick"].astype(int) * 10)
        - (df["Price/Night"] / df["Price/Night"].max()) * 8
    ).round(1)
    return df.sort_values(["Recommendation Score", "Rating"], ascending=False)


@st.cache_data(show_spinner=False)
def generate_flights(origin: str, dest: str, start: date) -> pd.DataFrame:
    rows = []
    for idx in range(5):
        seed = stable_hash(f"{origin}-{dest}-{start}-{idx}")
        dep_h = seeded_int(seed, 5, 21)
        dep_m = seeded_int(seed + 2, 0, 5) * 10
        duration_h = seeded_int(seed + 1, 1, 3)
        arr_m = seeded_int(seed + 3, 0, 5) * 10
        airline = AIRLINE_NAMES[seeded_int(seed + 4, 0, len(AIRLINE_NAMES) - 1)]
        code = ["6E", "AI", "SG", "UK", "QP", "IX"][seeded_int(seed + 5, 0, 5)]
        rows.append(
            {
                "Airline": airline,
                "Flight": f"{code}{seeded_int(seed + 6, 100, 999)}",
                "Departure": f"{dep_h:02d}:{dep_m:02d}",
                "Arrival": f"{(dep_h + duration_h) % 24:02d}:{arr_m:02d}",
                "Duration": f"{duration_h}h {dep_m}m",
                "Stops": "Non-stop" if seeded_int(seed + 8, 0, 1) == 0 else "1 stop",
                "Price": seeded_int(seed + 7, 2500, 12000),
            }
        )
    return pd.DataFrame(rows).sort_values("Price")


@st.cache_data(show_spinner=False)
def get_weather(dest: str, start: date, days: int) -> pd.DataFrame:
    profile = WEATHER_PROFILES.get(dest, WEATHER_PROFILES["default"])
    rows = []
    for idx in range(min(days, 14)):
        d = start + timedelta(days=idx)
        seed = stable_hash(dest + d.isoformat())
        variation = seeded_int(seed, -5, 5)
        rows.append(
            {
                "Date": d.strftime("%d %b"),
                "Day": d.strftime("%a"),
                "Condition": profile["cond"][seeded_int(seed + 1, 0, len(profile["cond"]) - 1)],
                "High °C": profile["base"] + variation + 3,
                "Low °C": profile["base"] + variation - 5,
                "Humidity %": seeded_int(seed + 2, 35, 90),
                "Wind km/h": seeded_int(seed + 3, 5, 35),
                "Rain mm": round(seeded_rand(seed + 5) * 25, 1) if seeded_rand(seed + 4) < profile["rain"] / 100 else 0,
            }
        )
    return pd.DataFrame(rows)


def get_attractions(dest: str) -> pd.DataFrame:
    data = ATTRACTIONS.get(
        dest,
        [
            (f"{dest} Heritage Site", "Main historical attraction", "🏛️", 4.5, 100),
            (f"{dest} Nature Park", "Scenic outdoor escape", "🌿", 4.4, 50),
            (f"{dest} Local Market", "Authentic shopping", "🛍️", 4.3, 0),
            (f"{dest} Museum", "Culture and history", "🏺", 4.2, 80),
        ],
    )
    return pd.DataFrame(data, columns=["Place", "Description", "Icon", "Rating", "Entry Fee"])


def get_restaurants(dest: str) -> pd.DataFrame:
    data = RESTAURANTS.get(
        dest,
        [
            (f"{dest} Local Kitchen", "Regional", 4.4, 500),
            (f"{dest} Cafe", "Cafe", 4.2, 350),
            (f"Royal {dest} Dining", "Fine Dining", 4.6, 1600),
            (f"{dest} Street Food Lane", "Street Food", 4.1, 180),
        ],
    )
    return pd.DataFrame(data, columns=["Restaurant", "Cuisine", "Rating", "Avg Cost for 2"])


def get_reviews(dest: str) -> pd.DataFrame:
    data = REVIEWS_DB.get(
        dest,
        [
            ("Arun V.", 5, f"{dest} was incredible — culture, food, and people were amazing.", "Positive"),
            ("Sara L.", 4, f"Really enjoyed {dest}. A few tourist traps but overall wonderful.", "Positive"),
            ("Dev P.", 3, f"{dest} has beautiful sights but infrastructure needs improvement.", "Neutral"),
            ("Kim J.", 4, f"Would strongly recommend {dest}. Local cuisine was a highlight.", "Positive"),
            ("Mia C.", 2, f"Felt {dest} was a bit overhyped. Worth a short trip.", "Negative"),
        ],
    )
    return pd.DataFrame(data, columns=["User", "Stars", "Review", "Sentiment"])


def recommendation_score(dest: str, interests: List[str], budget: str, start: date) -> float:
    info = DESTINATIONS[dest]
    crowd = info["crowd"][start.month - 1]
    tag_overlap = len(set(t.lower() for t in info["tags"] + [info["type"]]) & set(i.lower() for i in interests))
    season_bonus = 15 if start.month in [10, 11, 12, 1, 2, 3] else 5
    crowd_penalty = max(0, crowd - 60) * 0.25
    budget_bonus = {"Budget": 8, "Standard": 10, "Premium": 12, "Luxury": 14}.get(budget, 10)
    return round(55 + tag_overlap * 8 + season_bonus + budget_bonus - crowd_penalty, 1)


def get_destination_recommendations(profile: TripProfile) -> pd.DataFrame:
    rows = []
    for dest, info in DESTINATIONS.items():
        score = recommendation_score(dest, profile.interests, profile.budget, profile.start)
        rows.append(
            {
                "Destination": dest,
                "State": info["state"],
                "Type": info["type"],
                "Best Season": info["season"],
                "Crowd": crowd_label(info["crowd"][profile.start.month - 1]),
                "AI Match Score": min(99, score),
                "Tags": ", ".join(info["tags"]),
            }
        )
    return pd.DataFrame(rows).sort_values("AI Match Score", ascending=False)


def estimate_budget(profile: TripProfile) -> Dict[str, int]:
    days = trip_days(profile.start, profile.end)
    factor = get_budget_factor(profile.budget)
    seed = stable_hash(profile.dest + profile.budget)
    hotel_per_night = int(seeded_int(seed, 1000, 6500) * factor)
    food_per_person_day = int({"Budget": 450, "Standard": 850, "Premium": 1500, "Luxury": 2600}.get(profile.budget, 850))
    local_transport_day = int({"Budget": 700, "Standard": 1500, "Premium": 2600, "Luxury": 4500}.get(profile.budget, 1500))
    intercity = int(seeded_int(seed + 3, 2500, 9000) * profile.people * factor)
    attractions = int(get_attractions(profile.dest)["Entry Fee"].sum() * profile.people)
    shopping = int({"Budget": 1500, "Standard": 4000, "Premium": 9000, "Luxury": 18000}.get(profile.budget, 4000))
    hotel_total = hotel_per_night * max(1, days - 1)
    food_total = food_per_person_day * profile.people * days
    local_total = local_transport_day * days
    subtotal = hotel_total + food_total + local_total + intercity + attractions + shopping
    emergency = int(subtotal * 0.15)
    return {
        "Hotel": hotel_total,
        "Food": food_total,
        "Transport": local_total + intercity,
        "Attractions": attractions,
        "Shopping": shopping,
        "Emergency Buffer": emergency,
        "Total": subtotal + emergency,
    }


def build_itinerary(profile: TripProfile) -> pd.DataFrame:
    base = ITINERARY_ACTS.get(
        profile.dest,
        [
            ("08:00 AM", f"{profile.dest} Main Attraction", "Start with the most iconic site.", "🏛️"),
            ("11:00 AM", "Local Heritage Walk", "Explore old town with a guide.", "🚶"),
            ("01:30 PM", "Lunch: Regional cuisine", "Ask locals for authentic food.", "🍽️"),
            ("03:30 PM", "Museum / Cultural Centre", "Learn local history and art.", "🏺"),
            ("07:00 PM", "Evening bazaar", "Shop for souvenirs and street food.", "🛍️"),
        ],
    )
    rows = []
    for day_no in range(1, trip_days(profile.start, profile.end) + 1):
        trip_date = profile.start + timedelta(days=day_no - 1)
        for time_str, name, desc, icon in base:
            modified = name
            if day_no > 1:
                modified = f"{name} / Alternate local experience"
            rows.append(
                {
                    "Day": f"Day {day_no}",
                    "Date": trip_date.strftime("%d %b %Y"),
                    "Time": time_str,
                    "Activity": f"{icon} {modified}",
                    "Details": desc,
                }
            )
    return pd.DataFrame(rows)


def forecast_demand(dest: str, months: int = 12) -> pd.DataFrame:
    info = DESTINATIONS[dest]
    rows = []
    today = date.today()
    for i in range(months):
        month = (today.month - 1 + i) % 12
        base = info["crowd"][month]
        seasonal = 10 * math.sin((i / 12) * 2 * math.pi)
        noise = seeded_int(stable_hash(dest + str(i)), -6, 6)
        demand = max(10, min(100, int(base + seasonal + noise)))
        rows.append({"Month": month_name(month), "Demand Index": demand, "Crowd": crowd_label(demand)})
    return pd.DataFrame(rows)


def crowd_prediction(dest: str, visit_date: date) -> Tuple[int, str, str]:
    crowd = DESTINATIONS[dest]["crowd"][visit_date.month - 1]
    if visit_date.weekday() >= 5:
        crowd = min(100, crowd + 10)
    label = crowd_label(crowd)
    suggestion = "Book skip-line tickets and start before 8 AM." if crowd >= 70 else "Normal planning is fine; keep small buffers."
    return crowd, label, suggestion


def fraud_risk_score(amount: int, rating: float, advance_days: int, payment: str, mismatch: bool) -> Tuple[int, str, List[str]]:
    score = 0
    reasons = []
    if amount > 75000:
        score += 25; reasons.append("Very high transaction amount")
    if rating < 3.2:
        score += 20; reasons.append("Low hotel/vendor rating")
    if advance_days < 2:
        score += 15; reasons.append("Last-minute booking")
    if payment in {"Wallet", "UPI collect request", "Crypto"}:
        score += 15; reasons.append("Higher-risk payment channel")
    if mismatch:
        score += 25; reasons.append("Name/email/phone mismatch")
    score = min(100, score)
    label = "Low Risk" if score < 35 else "Medium Risk" if score < 65 else "High Risk"
    if not reasons:
        reasons.append("No major fraud indicators detected")
    return score, label, reasons


def landmark_recognition_demo(filename: str | None, dest: str) -> Tuple[str, float, str]:
    if not filename:
        return "No image uploaded", 0.0, "Upload a landmark image to run the demo."
    lower = filename.lower()
    patterns = {
        "taj": ("Taj Mahal", 0.94),
        "agra": ("Taj Mahal / Agra Fort", 0.86),
        "hawa": ("Hawa Mahal", 0.91),
        "amber": ("Amber Fort", 0.89),
        "goa": ("Goa Beach / Fort Aguada", 0.82),
        "beach": ("Beach landmark", 0.80),
        "kerala": ("Kerala Backwaters", 0.84),
        "manali": ("Manali Mountain Landmark", 0.83),
        "fort": ("Historic Fort", 0.78),
    }
    for key, result in patterns.items():
        if key in lower:
            return result[0], result[1], "Heuristic demo matched image filename/metadata. Replace with CNN/CLIP model for production."
    default_place = get_attractions(dest).iloc[0]["Place"]
    seed = stable_hash(filename + dest)
    confidence = round(0.62 + seeded_rand(seed) * 0.25, 2)
    return str(default_place), confidence, "Demo prediction based on selected destination context."


def get_ai_reply(message: str, profile: TripProfile | None) -> str:
    dest = profile.dest if profile else "India"
    text = message.lower()
    kb = AI_KNOWLEDGE.get(dest, {})
    if any(word in text for word in ["food", "eat", "restaurant", "cuisine"]):
        return kb.get("food", f"For food in {dest}, try local busy restaurants, regional thalis, and freshly cooked street food.")
    if any(word in text for word in ["hotel", "stay", "accommodation"]):
        return kb.get("hotel", f"For accommodation in {dest}, compare hotel rating, distance, cancellation policy, and recent reviews.")
    if any(word in text for word in ["transport", "taxi", "metro", "bus", "get around"]):
        return kb.get("transport", f"In {dest}, use verified taxis, public transport where available, and pre-agreed fares for autos.")
    if any(word in text for word in ["visit", "see", "attraction", "sightseeing"]):
        return kb.get("visit", f"Top attractions in {dest}: start with main heritage/nature places early morning and markets in the evening.")
    if any(word in text for word in ["safe", "danger", "emergency"]):
        return f"Safety in {dest}: keep ID copies, avoid isolated areas late night, use verified transport. Helplines: Police 112, Ambulance 108, Tourist 1363."
    if any(word in text for word in ["budget", "cost", "cheap", "money"]):
        return f"Budget tip for {dest}: reserve 15–20% buffer, compare hotel areas, use local food options, and group nearby attractions to reduce taxi cost."
    if any(word in text for word in ["weather", "rain", "climate"]):
        return f"Check the Weather tab for {dest}. Carry sunscreen, comfortable shoes, and rain protection if rain probability is high."
    if any(word in text for word in ["pack", "carry", "luggage"]):
        return f"Packing for {dest}: comfortable shoes, sunscreen, water bottle, ID copy, power bank, medicines, and season-specific clothing."
    return GENERIC_AI[stable_hash(message) % len(GENERIC_AI)]


# =============================================================================
# UI HELPERS
# =============================================================================

def app_header() -> None:
    st.title(f"{APP_ICON} {APP_NAME}")
    st.caption("Python-only implementation · AI recommendations · budgets · weather · maps · itinerary · analytics")


def require_trip() -> TripProfile | None:
    profile = get_trip()
    if profile is None:
        st.warning("Please create your trip first from the sidebar or Dashboard page.")
        return None
    return profile


def show_trip_summary(profile: TripProfile) -> None:
    days = trip_days(profile.start, profile.end)
    st.info(
        f"**Trip:** {profile.origin} → {profile.dest} | **Dates:** {profile.start:%d %b %Y} to {profile.end:%d %b %Y} | "
        f"**Days:** {days} | **Travellers:** {profile.people} | **Budget:** {profile.budget} | **Style:** {profile.travel_style}"
    )


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def page_profile_and_trip_setup() -> None:
    st.header("👤 User Profile & Smart Trip Setup")
    st.write("Create or update your travel profile. The rest of the platform uses this data for personalization.")

    existing = get_trip()
    with st.form("trip_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Traveller name", value=existing.name if existing else current_user_name())
            origin = st.selectbox("Origin city", CITY_OPTIONS, index=CITY_OPTIONS.index(existing.origin) if existing and existing.origin in CITY_OPTIONS else 1)
            dest = st.selectbox("Destination", list(DESTINATIONS.keys()), index=list(DESTINATIONS.keys()).index(existing.dest) if existing else 0)
        with c2:
            start = st.date_input("Start date", value=existing.start if existing else date.today() + timedelta(days=7))
            end = st.date_input("End date", value=existing.end if existing else date.today() + timedelta(days=10))
            people = st.number_input("Travellers", min_value=1, max_value=20, value=existing.people if existing else 2)
        with c3:
            budget = st.selectbox("Budget category", ["Budget", "Standard", "Premium", "Luxury"], index=["Budget", "Standard", "Premium", "Luxury"].index(existing.budget) if existing else 1)
            travel_style = st.selectbox("Travel style", ["Balanced", "Relaxed", "Adventure", "Family", "Luxury", "Backpacking"], index=0)
            interests = st.multiselect(
                "Interests",
                ["Heritage", "Beach", "Nature", "Adventure", "Spiritual", "Food", "Shopping", "Culture", "Urban", "Romantic"],
                default=existing.interests if existing else ["Heritage", "Food"],
            )
        submitted = st.form_submit_button("Save Trip & Generate Recommendations", use_container_width=True)

    if submitted:
        if end < start:
            st.error("End date must be after or equal to start date.")
        else:
            profile = TripProfile(str(name), origin, dest, start, end, int(people), budget, interests, travel_style)
            save_trip(profile)
            st.session_state["user_name"] = str(name)
            st.success("Trip saved successfully. Open Dashboard for recommendations.")
            show_trip_summary(profile)

    if existing:
        st.subheader("Current Trip")
        show_trip_summary(existing)


def page_dashboard() -> None:
    st.header("📊 Smart Tourism Dashboard")
    profile = require_trip()
    if not profile:
        page_profile_and_trip_setup()
        return
    show_trip_summary(profile)

    dest_info = DESTINATIONS[profile.dest]
    crowd, label, suggestion = crowd_prediction(profile.dest, profile.start)
    budget = estimate_budget(profile)
    weather = get_weather(profile.dest, profile.start, trip_days(profile.start, profile.end))
    avg_temp = int(weather["High °C"].mean()) if not weather.empty else 28

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Destination Score", f"{recommendation_score(profile.dest, profile.interests, profile.budget, profile.start)}/99", "Personalized")
    c2.metric("Predicted Crowd", f"{crowd}/100", label)
    c3.metric("Estimated Budget", rupee(budget["Total"]), f"{rupee(budget['Total'] / profile.people)} per person")
    c4.metric("Avg High Temp", f"{avg_temp}°C", "Trip forecast")

    st.subheader("Destination Snapshot")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"**{profile.dest}, {dest_info['state']}**")
        st.write(f"Type: **{dest_info['type']}** | Best season: **{dest_info['season']}**")
        st.write("Tags: " + ", ".join(dest_info["tags"]))
        facts = DEST_FACTS.get(profile.dest, [f"{profile.dest} has strong tourism potential.", "Check local events before travel."])
        for fact in facts:
            st.write(f"- {fact}")
        st.success(f"Crowd advice: {suggestion}")
    with c2:
        st.map(pd.DataFrame([{"lat": dest_info["lat"], "lon": dest_info["lon"]}]), latitude="lat", longitude="lon", zoom=8)

    st.subheader("Top Destination Recommendations")
    st.dataframe(get_destination_recommendations(profile), use_container_width=True, hide_index=True)

    st.subheader("Quick Access Features")
    st.write("Hotels, food, weather, map, itinerary, currency, emergency contacts, and AI assistant are available from the sidebar.")


def page_budget_planner() -> None:
    st.header("💰 Budget Planner — Detailed Cost Breakdown")
    profile = require_trip()
    if not profile:
        return

    days = trip_days(profile.start, profile.end)
    factor = get_budget_factor(profile.budget)
    seed = stable_hash(profile.dest + profile.budget)

    st.subheader("🎛️ Customize Your Cost Parameters")
    st.caption("Adjust the daily/per-unit rates below to match your actual preferences.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🏨 Accommodation**")
        hotel_per_night = st.number_input(
            "Hotel cost per night (₹)", min_value=500, max_value=50000,
            value=int(seeded_int(seed, 1000, 6500) * factor), step=100,
            help="Average hotel/stay cost for one night"
        )
        st.markdown("**🍽️ Food & Dining**")
        breakfast_per_person = st.number_input(
            "Breakfast per person/day (₹)", min_value=50, max_value=2000,
            value=int({"Budget": 100, "Standard": 200, "Premium": 400, "Luxury": 800}.get(profile.budget, 200)),
            step=50
        )
        lunch_per_person = st.number_input(
            "Lunch per person/day (₹)", min_value=100, max_value=3000,
            value=int({"Budget": 150, "Standard": 350, "Premium": 700, "Luxury": 1200}.get(profile.budget, 350)),
            step=50
        )
        dinner_per_person = st.number_input(
            "Dinner per person/day (₹)", min_value=100, max_value=5000,
            value=int({"Budget": 200, "Standard": 450, "Premium": 900, "Luxury": 1500}.get(profile.budget, 450)),
            step=50
        )
        snacks_per_person = st.number_input(
            "Snacks/tea per person/day (₹)", min_value=0, max_value=1000,
            value=int({"Budget": 50, "Standard": 100, "Premium": 200, "Luxury": 400}.get(profile.budget, 100)),
            step=25
        )

    with col2:
        st.markdown("**🚗 Local Transport**")
        taxi_per_day = st.number_input(
            "Taxi/cab per day (₹)", min_value=0, max_value=10000,
            value=int({"Budget": 400, "Standard": 900, "Premium": 1800, "Luxury": 3500}.get(profile.budget, 900)),
            step=100
        )
        auto_per_day = st.number_input(
            "Auto-rickshaw/metro per day (₹)", min_value=0, max_value=2000,
            value=int({"Budget": 150, "Standard": 300, "Premium": 500, "Luxury": 800}.get(profile.budget, 300)),
            step=50
        )
        fuel_per_day = st.number_input(
            "Fuel/petrol (if self-drive) per day (₹)", min_value=0, max_value=3000,
            value=0, step=50
        )
        intercity_travel = st.number_input(
            "Intercity travel total (flights/trains) (₹)", min_value=0, max_value=100000,
            value=int(seeded_int(seed + 3, 2500, 9000) * profile.people * factor),
            step=500, help="Total train/bus/flight tickets for all travellers"
        )

    with col3:
        st.markdown("**🎟️ Activities & Extras**")
        entry_fees_total = st.number_input(
            "Attraction entry fees total (₹)", min_value=0, max_value=20000,
            value=int(get_attractions(profile.dest)["Entry Fee"].sum() * profile.people),
            step=100, help="Total for all travellers"
        )
        guided_tours = st.number_input(
            "Guided tours & experiences (₹)", min_value=0, max_value=20000,
            value=int({"Budget": 0, "Standard": 500, "Premium": 2000, "Luxury": 5000}.get(profile.budget, 500)),
            step=250
        )
        shopping = st.number_input(
            "Shopping & souvenirs (₹)", min_value=0, max_value=50000,
            value=int({"Budget": 1500, "Standard": 4000, "Premium": 9000, "Luxury": 18000}.get(profile.budget, 4000)),
            step=500
        )
        st.markdown("**💊 Health & Misc**")
        medicines_misc = st.number_input(
            "Medicines & toiletries (₹)", min_value=0, max_value=5000,
            value=300, step=100
        )
        tips_donations = st.number_input(
            "Tips, donations & offering (₹)", min_value=0, max_value=5000,
            value=int({"Budget": 100, "Standard": 300, "Premium": 700, "Luxury": 1500}.get(profile.budget, 300)),
            step=100
        )
        photography = st.number_input(
            "Photography/videography fees (₹)", min_value=0, max_value=5000,
            value=int({"Budget": 0, "Standard": 100, "Premium": 300, "Luxury": 800}.get(profile.budget, 100)),
            step=50
        )

    st.divider()

    # ── Compute totals ──
    hotel_total        = hotel_per_night * max(1, days - 1)
    food_per_person_day = breakfast_per_person + lunch_per_person + dinner_per_person + snacks_per_person
    food_total         = food_per_person_day * profile.people * days
    local_transport    = (taxi_per_day + auto_per_day + fuel_per_day) * days
    activities_total   = entry_fees_total + guided_tours
    misc_total         = medicines_misc + tips_donations + photography
    sub_items = {
        "🏨 Hotel / Stay": hotel_total,
        "🍳 Breakfast": breakfast_per_person * profile.people * days,
        "🥗 Lunch": lunch_per_person * profile.people * days,
        "🍛 Dinner": dinner_per_person * profile.people * days,
        "🧃 Snacks & Tea": snacks_per_person * profile.people * days,
        "🚖 Taxi / Cab": taxi_per_day * days,
        "🛺 Auto / Metro": auto_per_day * days,
        "⛽ Fuel (self-drive)": fuel_per_day * days,
        "✈️ Intercity Travel": intercity_travel,
        "🎟️ Entry Fees": entry_fees_total,
        "🧭 Guided Tours": guided_tours,
        "🛍️ Shopping": shopping,
        "💊 Medicines & Toiletries": medicines_misc,
        "🙏 Tips & Donations": tips_donations,
        "📷 Photography Fees": photography,
    }
    subtotal = sum(sub_items.values())
    emergency = int(subtotal * 0.15)
    grand_total = subtotal + emergency

    # ── Summary metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 Grand Total", rupee(grand_total))
    m2.metric("👤 Per Person", rupee(grand_total // profile.people))
    m3.metric("📅 Trip Days", f"{days} days")
    m4.metric("🔒 Emergency Buffer (15%)", rupee(emergency))

    st.subheader("📋 Detailed Cost Breakdown")

    # Table with all parameters
    breakdown_df = pd.DataFrame([
        {"Category": k, "Amount (₹)": rupee(v), "Raw": v}
        for k, v in sub_items.items()
    ] + [
        {"Category": "🆘 Emergency Buffer (15%)", "Amount (₹)": rupee(emergency), "Raw": emergency},
        {"Category": "✅ GRAND TOTAL", "Amount (₹)": rupee(grand_total), "Raw": grand_total},
    ])

    st.dataframe(
        breakdown_df[["Category", "Amount (₹)"]],
        use_container_width=True, hide_index=True
    )

    # Chart — only sub_items (no buffer/total in bar chart for clarity)
    chart_df = pd.DataFrame([
        {"Category": k.split(" ", 1)[1] if " " in k else k, "Amount": v}
        for k, v in sub_items.items() if v > 0
    ]).set_index("Category")
    st.bar_chart(chart_df)

    # ── Per-person per-day summary ──
    st.subheader("📊 Daily Cost Summary")
    d1, d2, d3 = st.columns(3)
    d1.metric("Food per person/day", rupee(food_per_person_day))
    d2.metric("Local transport/day", rupee(taxi_per_day + auto_per_day + fuel_per_day))
    d3.metric("Hotel per night", rupee(hotel_per_night))

    st.subheader("💡 Budget Optimization Tips")
    st.write("- Group nearby attractions to reduce daily taxi/cab costs.")
    st.write("- Book hotels near the main attraction — reduces transport overhead.")
    st.write("- Prefer local dhabas and regional thalis for authentic and affordable meals.")
    st.write("- Keep the 15% emergency buffer untouched for medical/weather/transport surprises.")
    st.write("- Travel in shoulder season for lower hotel prices and less crowd.")


def page_hotels_food() -> None:
    st.header("🏨 Hotel Booking & Restaurant Finder")
    profile = require_trip()
    if not profile:
        return

    # ── AGODA-STYLE SEARCH BAR ──
    st.markdown("""
    <style>
    .hotel-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #e0e0e0;
    }
    .price-tag {
        color: #d32f2f;
        font-size: 1.4em;
        font-weight: bold;
    }
    .badge {
        background: #1565c0;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Search panel (Agoda-style) ──
    st.subheader("🔍 Search Hotels")
    sc1, sc2, sc3, sc4, sc5 = st.columns([2, 1, 1, 1, 1])
    with sc1:
        search_dest = st.text_input("Destination / Hotel name", value=profile.dest, placeholder="City or hotel name")
    with sc2:
        check_in = st.date_input("Check-in", value=profile.start, key="ci")
    with sc3:
        check_out = st.date_input("Check-out", value=profile.end, key="co")
    with sc4:
        rooms = st.number_input("Rooms", min_value=1, max_value=10, value=max(1, profile.people // 2))
    with sc5:
        adults = st.number_input("Adults", min_value=1, max_value=20, value=profile.people)

    nights = max(1, (check_out - check_in).days)
    st.caption(f"📅 {nights} night(s) · {adults} adult(s) · {rooms} room(s)")

    st.divider()

    # ── Filters ──
    st.subheader("🔧 Filter & Sort")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        max_price = st.slider("Max price/night (₹)", 500, 40000, 12000, 500)
    with f2:
        min_rating = st.slider("Min guest rating", 3.0, 5.0, 3.5, 0.1)
    with f3:
        min_stars = st.selectbox("Min star category", [1, 2, 3, 4, 5], index=0)
    with f4:
        sort_by = st.selectbox("Sort by", ["Recommended", "Price: Low to High", "Price: High to Low", "Rating", "Distance"])

    amenity_filter = st.multiselect(
        "Required amenities",
        ["WiFi", "Pool", "Gym", "Spa", "Parking", "Restaurant", "AC", "Breakfast", "Airport Transfer"],
        default=[]
    )

    hotels = generate_hotels(profile.dest)
    filtered = hotels[
        (hotels["Price/Night"] <= max_price) &
        (hotels["Rating"] >= min_rating) &
        (hotels["Stars"] >= min_stars)
    ]
    if amenity_filter:
        filtered = filtered[filtered["Amenities"].apply(lambda a: all(x in a for x in amenity_filter))]

    if sort_by == "Price: Low to High":
        filtered = filtered.sort_values("Price/Night")
    elif sort_by == "Price: High to Low":
        filtered = filtered.sort_values("Price/Night", ascending=False)
    elif sort_by == "Rating":
        filtered = filtered.sort_values("Rating", ascending=False)
    elif sort_by == "Distance":
        filtered = filtered.sort_values("Distance_km")

    st.markdown(f"**{len(filtered)} properties found** in {search_dest}")

    # ── Hotel cards (Agoda-style) ──
    for _, row in filtered.iterrows():
        total_price = row["Price/Night"] * nights * rooms
        ai_badge = "🏅 AI Top Pick" if row["AI Pick"] else ""
        star_str = "⭐" * int(row["Stars"])

        with st.container():
            card_c1, card_c2 = st.columns([3, 1])
            with card_c1:
                st.markdown(f"### {row['Icon']} {row['Hotel']} {star_str}")
                if ai_badge:
                    st.markdown(f"<span style='background:#1565c0;color:white;padding:2px 10px;border-radius:4px;font-size:0.85em'>{ai_badge}</span>", unsafe_allow_html=True)
                st.markdown(f"📍 {row['Distance_km']} km from city centre &nbsp;|&nbsp; 🏆 Score: **{row['Recommendation Score']}**")
                st.markdown(f"🛎️ Amenities: {row['Amenities']}")
                st.markdown(f"💬 **{row['Rating']}/5** · {row['Reviews']} reviews")

            with card_c2:
                st.markdown(f"<div style='text-align:right'><span style='color:#d32f2f;font-size:1.5em;font-weight:bold'>₹{row['Price/Night']:,}</span><br><small>per night</small></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:right;color:#555'>Total: <b>₹{int(total_price):,}</b><br><small>for {nights} nights · {rooms} room(s)</small></div>", unsafe_allow_html=True)

                # ── Booking section ──
                book_key = f"book_{row['Hotel']}"
                detail_key = f"detail_{row['Hotel']}"

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("ℹ️ Details", key=detail_key, use_container_width=True):
                        st.session_state[f"show_detail_{row['Hotel']}"] = not st.session_state.get(f"show_detail_{row['Hotel']}", False)
                with bcol2:
                    if st.button("📅 Book Now", key=book_key, type="primary", use_container_width=True):
                        st.session_state[f"booking_{row['Hotel']}"] = True

            # ── Room details expand ──
            if st.session_state.get(f"show_detail_{row['Hotel']}", False):
                with st.expander("🛏️ Room Types & Details", expanded=True):
                    rt_col1, rt_col2, rt_col3 = st.columns(3)
                    with rt_col1:
                        st.markdown("**Standard Room**")
                        st.write(f"₹{row['Price/Night']:,} / night")
                        st.write("👥 2 adults · 1 bed")
                        st.write("✅ Free cancellation")
                        st.write("🍳 Breakfast optional")
                    with rt_col2:
                        st.markdown("**Deluxe Room**")
                        st.write(f"₹{int(row['Price/Night'] * 1.3):,} / night")
                        st.write("👥 2 adults · King bed")
                        st.write("✅ Free cancellation")
                        st.write("🍳 Breakfast included")
                    with rt_col3:
                        st.markdown("**Suite**")
                        st.write(f"₹{int(row['Price/Night'] * 1.85):,} / night")
                        st.write("👥 3 adults · Suite")
                        st.write("⚠️ Non-refundable")
                        st.write("🍳 All meals included")

            # ── Booking form (Agoda-style modal simulation) ──
            if st.session_state.get(f"booking_{row['Hotel']}", False):
                with st.expander(f"📋 Complete Booking — {row['Hotel']}", expanded=True):
                    st.markdown(f"**{row['Icon']} {row['Hotel']}** · {nights} nights · {rooms} room(s)")
                    st.markdown(f"Check-in: **{check_in:%d %b %Y}** → Check-out: **{check_out:%d %b %Y}**")

                    room_type = st.selectbox(
                        "Select room type",
                        ["Standard Room", "Deluxe Room", "Suite"],
                        key=f"rt_{row['Hotel']}"
                    )
                    room_prices = {"Standard Room": row["Price/Night"], "Deluxe Room": row["Price/Night"] * 1.3, "Suite": row["Price/Night"] * 1.85}
                    room_price = room_prices[room_type]
                    booking_total = room_price * nights * rooms

                    st.markdown(f"💰 **Total: ₹{int(booking_total):,}** for {nights} nights · {rooms} room(s)")

                    bc1, bc2 = st.columns(2)
                    with bc1:
                        guest_name = st.text_input("Guest name", value=profile.name, key=f"gn_{row['Hotel']}")
                        guest_email = st.text_input("Email address", key=f"ge_{row['Hotel']}")
                        guest_phone = st.text_input("Phone number", key=f"gp_{row['Hotel']}")
                    with bc2:
                        special_requests = st.text_area("Special requests (optional)", placeholder="e.g. early check-in, high floor, vegetarian meals", key=f"sr_{row['Hotel']}")
                        payment_method = st.selectbox(
                            "Payment method",
                            ["Credit Card", "Debit Card", "UPI", "Net Banking", "Pay at Hotel"],
                            key=f"pm_{row['Hotel']}"
                        )
                        cancellation = st.selectbox(
                            "Cancellation policy",
                            ["Free cancellation (until 24h before)", "Non-refundable (save 10%)"],
                            key=f"cp_{row['Hotel']}"
                        )

                    col_cancel, col_confirm = st.columns(2)
                    with col_cancel:
                        if st.button("❌ Cancel", key=f"cancel_{row['Hotel']}", use_container_width=True):
                            st.session_state[f"booking_{row['Hotel']}"] = False
                            st.rerun()
                    with col_confirm:
                        if st.button("✅ Confirm Booking", key=f"confirm_{row['Hotel']}", type="primary", use_container_width=True):
                            if not guest_name or not guest_email:
                                st.error("Please enter guest name and email to confirm booking.")
                            else:
                                import random as _rnd
                                booking_id = f"TA{_rnd.randint(100000, 999999)}"
                                st.session_state[f"booked_{row['Hotel']}"] = {
                                    "id": booking_id, "hotel": row["Hotel"], "room": room_type,
                                    "checkin": str(check_in), "checkout": str(check_out),
                                    "total": int(booking_total), "guest": guest_name, "payment": payment_method
                                }
                                st.session_state[f"booking_{row['Hotel']}"] = False
                                st.rerun()

            # ── Booking confirmation card ──
            if st.session_state.get(f"booked_{row['Hotel']}"):
                bk = st.session_state[f"booked_{row['Hotel']}"]
                st.success(f"✅ **Booking Confirmed!** ID: `{bk['id']}`")
                st.markdown(
                    f"🏨 **{bk['hotel']}** · {bk['room']} · {bk['checkin']} → {bk['checkout']} · "
                    f"₹{bk['total']:,} · {bk['payment']} · Guest: {bk['guest']}"
                )
                if st.button(f"🗑️ Cancel Booking {bk['id']}", key=f"del_{row['Hotel']}"):
                    del st.session_state[f"booked_{row['Hotel']}"]
                    st.rerun()

            st.divider()

    # ── My Bookings summary ──
    my_bookings = [v for k, v in st.session_state.items() if k.startswith("booked_")]
    if my_bookings:
        st.subheader("📜 My Bookings")
        bdf = pd.DataFrame(my_bookings)
        bdf["total"] = bdf["total"].map(rupee)
        bdf.columns = [c.title() for c in bdf.columns]
        st.dataframe(bdf, use_container_width=True, hide_index=True)

    # ── Restaurant section ──
    st.subheader("🍽️ Restaurant Finder")
    restaurants = get_restaurants(profile.dest).copy()
    restaurants["Avg Cost for 2"] = restaurants["Avg Cost for 2"].map(rupee)
    st.dataframe(restaurants, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ How hotel recommendation scoring works"):
        st.write("Hotels are ranked using a weighted scoring model: rating (×22), star category (×8), distance (×16), AI-pick bonus (×10), and a price penalty. This can be replaced with collaborative filtering, XGBoost, or a neural ranking model in production.")


def page_flights_routes() -> None:
    st.header("✈️ Flights & Optimized Routes")
    profile = require_trip()
    if not profile:
        return
    flights = generate_flights(profile.origin, profile.dest, profile.start)
    st.subheader("Simulated Flight Options")
    show = flights.copy()
    show["Price"] = show["Price"].map(rupee)
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Optimized Route Plan")
    dest_info = DESTINATIONS[profile.dest]
    route_points = [{"lat": dest_info["lat"], "lon": dest_info["lon"], "name": profile.dest}]
    atts = get_attractions(profile.dest)
    for idx, row in atts.iterrows():
        seed = stable_hash(profile.dest + row["Place"])
        route_points.append(
            {
                "lat": dest_info["lat"] + (seeded_rand(seed) - 0.5) * 0.16,
                "lon": dest_info["lon"] + (seeded_rand(seed + 1) - 0.5) * 0.16,
                "name": row["Place"],
            }
        )
    st.map(pd.DataFrame(route_points), latitude="lat", longitude="lon", zoom=11)
    st.write("Suggested order: Hotel → nearest morning attraction → lunch zone → afternoon attraction → market/food street.")
    st.write("Production upgrade: connect OpenStreetMap OSRM / Google Directions API for turn-by-turn navigation and route optimization.")


def page_weather() -> None:
    st.header("🌦️ Real-time Weather Conditions")
    profile = require_trip()
    if not profile:
        return
    weather = get_weather(profile.dest, profile.start, trip_days(profile.start, profile.end))
    st.dataframe(weather, use_container_width=True, hide_index=True)
    st.line_chart(weather.set_index("Date")[["High °C", "Low °C", "Rain mm"]])
    rainy = weather[weather["Rain mm"] > 0]
    if not rainy.empty:
        st.warning("Rain expected on: " + ", ".join(rainy["Date"].tolist()) + ". Carry umbrella/raincoat and plan indoor backups.")
    else:
        st.success("Low rain risk in the current forecast window.")


def page_attractions_map() -> None:
    st.header("🗺️ Nearby Attractions Finder & Tourist Map")
    profile = require_trip()
    if not profile:
        return
    attractions = get_attractions(profile.dest)
    show = attractions.copy()
    show["Entry Fee"] = show["Entry Fee"].map(rupee)
    st.dataframe(show, use_container_width=True, hide_index=True)

    dest_info = DESTINATIONS[profile.dest]
    points = []
    for _, row in attractions.iterrows():
        seed = stable_hash(profile.dest + row["Place"])
        points.append(
            {
                "lat": dest_info["lat"] + (seeded_rand(seed) - 0.5) * 0.12,
                "lon": dest_info["lon"] + (seeded_rand(seed + 1) - 0.5) * 0.12,
            }
        )
    st.map(pd.DataFrame(points), latitude="lat", longitude="lon", zoom=12)

    st.subheader("Population Density / Tourist Crowd Heatmap Dataset")
    month_rows = [{"Month": month_name(i), "Crowd Index": DESTINATIONS[profile.dest]["crowd"][i]} for i in range(12)]
    st.bar_chart(pd.DataFrame(month_rows).set_index("Month"))


def page_itinerary() -> None:
    st.header("🧳 LLM-powered Smart Trip Planner / Itinerary")
    profile = require_trip()
    if not profile:
        return
    itinerary = build_itinerary(profile)
    for day, group in itinerary.groupby("Day"):
        with st.expander(f"{day} — {group.iloc[0]['Date']}", expanded=day == "Day 1"):
            st.dataframe(group[["Time", "Activity", "Details"]], use_container_width=True, hide_index=True)
    st.download_button("Download itinerary as CSV", itinerary.to_csv(index=False), file_name=f"{profile.dest}_itinerary.csv", mime="text/csv")


def page_reviews_analytics() -> None:
    st.header("⭐ Reviews, Sentiment & Travel Trends")
    profile = require_trip()
    if not profile:
        return
    reviews = get_reviews(profile.dest)
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Rating", f"{reviews['Stars'].mean():.1f}/5")
    c2.metric("Positive Reviews", int((reviews["Sentiment"] == "Positive").sum()))
    c3.metric("Total Reviews Sample", len(reviews))
    st.dataframe(reviews, use_container_width=True, hide_index=True)

    sentiment_counts = reviews["Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
    st.bar_chart(sentiment_counts.set_index("Sentiment"))

    st.subheader("Travel Demand Forecasting")
    demand = forecast_demand(profile.dest)
    st.line_chart(demand.set_index("Month")[["Demand Index"]])
    st.dataframe(demand, use_container_width=True, hide_index=True)


def page_ai_assistant() -> None:
    st.header("🤖 Tourist Chatbot & AI Assistant")
    profile = get_trip()
    if "chat_history" not in st.session_state:
        dest = profile.dest if profile else "India"
        st.session_state["chat_history"] = [("assistant", f"Namaste! I am your AI travel guide for {dest}. Ask about food, hotels, transport, safety, weather, packing, or budget.")]

    quick = st.selectbox(
        "Quick questions",
        [
            "Best food options",
            "Hotels for my budget",
            "How to get around",
            "Safety tips",
            "Best attractions to visit",
            "Packing tips",
        ],
    )
    if st.button("Ask Quick Question"):
        reply = get_ai_reply(quick, profile)
        st.session_state["chat_history"].append(("user", quick))
        st.session_state["chat_history"].append(("assistant", reply))

    message = st.chat_input("Ask your travel question...")
    if message:
        st.session_state["chat_history"].append(("user", message))
        st.session_state["chat_history"].append(("assistant", get_ai_reply(message, profile)))

    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.write(text)

    with st.expander("AI Stack Notes"):
        st.write("Current implementation: Python rule-based NLP fallback.")
        st.write("Production upgrade: connect OpenAI/Gemini/Llama API, vector database RAG, BERT sentiment model, multilingual translation, and speech-to-text.")


def page_currency() -> None:
    st.header("💱 Currency Converter")
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Amount", min_value=0.0, value=10000.0)
    with c2:
        from_code = st.selectbox("From", list(RATES.keys()), index=0, format_func=lambda x: CURRENCY_NAMES[x])
    with c3:
        to_code = st.selectbox("To", list(RATES.keys()), index=1, format_func=lambda x: CURRENCY_NAMES[x])
    in_inr = amount / RATES[from_code]
    converted = in_inr * RATES[to_code]
    st.metric("Converted Amount", f"{to_code} {converted:,.2f}")
    rates_df = pd.DataFrame([{"Currency": k, "Name": v, "Rate per INR": RATES[k]} for k, v in CURRENCY_NAMES.items()])
    st.dataframe(rates_df, use_container_width=True, hide_index=True)
    st.info("Rates are simulated for offline demo. Replace RATES with ExchangeRatesAPI/Open Exchange Rates API for live conversion.")


def page_emergency() -> None:
    st.header("🚨 Emergency Contacts")
    cols = st.columns(4)
    for idx, (name, number, desc, icon) in enumerate(EMERGENCY_CONTACTS):
        with cols[idx % 4]:
            st.metric(f"{icon} {name}", number)
            st.caption(desc)
    st.subheader("Safety Checklist")
    st.write("- Save 112, 108, and 1363 before travel.")
    st.write("- Keep digital and printed copies of ID/passport/visa/insurance.")
    st.write("- Use registered taxis and avoid sharing OTP/payment details with unknown persons.")
    st.write("- Share live location with family during late-night travel.")


def page_advanced_ai() -> None:
    st.header("🧠 Advanced AI Features")
    profile = require_trip()
    if not profile:
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Landmark Recognition", "Crowd Prediction", "Fraud Detection", "ML Model Summary"])

    with tab1:
        st.subheader("Image Recognition for Landmarks")
        uploaded = st.file_uploader("Upload landmark image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        label, confidence, note = landmark_recognition_demo(uploaded.name if uploaded else None, profile.dest)
        st.metric("Predicted Landmark", label)
        st.metric("Confidence", f"{confidence * 100:.0f}%")
        st.caption(note)

    with tab2:
        st.subheader("Crowd Prediction")
        selected_date = st.date_input("Visit date", value=profile.start)
        crowd, label, suggestion = crowd_prediction(profile.dest, selected_date)
        st.metric("Crowd Index", f"{crowd}/100", label)
        st.write(suggestion)
        st.bar_chart(pd.DataFrame({"Month": [month_name(i) for i in range(12)], "Crowd": DESTINATIONS[profile.dest]["crowd"]}).set_index("Month"))

    with tab3:
        st.subheader("Travel Booking Fraud Detection")
        c1, c2, c3 = st.columns(3)
        with c1:
            amount = st.number_input("Booking amount", min_value=0, value=25000, step=1000)
            rating = st.slider("Vendor rating", 1.0, 5.0, 4.1, 0.1)
        with c2:
            advance_days = st.number_input("Advance booking days", min_value=0, value=10)
            payment = st.selectbox("Payment method", ["Credit Card", "Debit Card", "UPI", "Wallet", "UPI collect request", "Crypto"])
        with c3:
            mismatch = st.checkbox("User details mismatch detected")
        score, risk, reasons = fraud_risk_score(int(amount), float(rating), int(advance_days), payment, mismatch)
        st.metric("Fraud Risk Score", f"{score}/100", risk)
        for reason in reasons:
            st.write(f"- {reason}")

    with tab4:
        st.subheader("ML Components Used / Ready for Upgrade")
        model_rows = [
            ("Recommendation", "Content-based weighted scoring", "Collaborative Filtering / Matrix Factorization"),
            ("Forecasting", "Seasonal time-series simulation", "ARIMA / Prophet / LSTM"),
            ("Fraud Detection", "Rule-based risk score", "XGBoost / Random Forest"),
            ("NLP Assistant", "Rule-based intent matching", "LLM + RAG + BERT"),
            ("Sentiment", "Stored labels + rule-ready", "BERT / RoBERTa sentiment classifier"),
            ("Image Recognition", "Filename/context heuristic", "CNN / CLIP / Vision Transformer"),
        ]
        st.dataframe(pd.DataFrame(model_rows, columns=["Feature", "Current Python Demo", "Production ML Model"]), use_container_width=True, hide_index=True)


def page_datasets() -> None:
    st.header("🗃️ Datasets")
    st.write("This Python version embeds demo datasets and exposes tables that can later be replaced with APIs.")
    dataset_name = st.selectbox("Choose dataset", ["Destinations", "Hotels", "Attractions", "Reviews", "Weather", "Flights", "Demand Forecast"])
    profile = get_trip() or TripProfile("Demo Tourist", "Delhi", "Agra", date.today() + timedelta(days=7), date.today() + timedelta(days=10), 2, "Standard", ["Heritage"], "Balanced")
    if dataset_name == "Destinations":
        df = pd.DataFrame([{**{"Destination": k}, **v} for k, v in DESTINATIONS.items()])
    elif dataset_name == "Hotels":
        df = generate_hotels(profile.dest)
    elif dataset_name == "Attractions":
        df = get_attractions(profile.dest)
    elif dataset_name == "Reviews":
        df = get_reviews(profile.dest)
    elif dataset_name == "Weather":
        df = get_weather(profile.dest, profile.start, trip_days(profile.start, profile.end))
    elif dataset_name == "Flights":
        df = generate_flights(profile.origin, profile.dest, profile.start)
    else:
        df = forecast_demand(profile.dest)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download selected dataset", df.to_csv(index=False), file_name=f"{dataset_name.lower().replace(' ', '_')}.csv", mime="text/csv")


# =============================================================================
# SIDEBAR NAVIGATION AND APP ENTRY
# =============================================================================

def sidebar() -> str:
    st.sidebar.title("🧭 Travel Around")
    st.sidebar.caption("Smart Tourism Platform")
    name = st.sidebar.text_input("User", value=current_user_name())
    st.session_state["user_name"] = name or "Demo Tourist"

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "User Profile / Trip Setup",
            "Budget Planner",
            "Hotels & Food",
            "Flights & Routes",
            "Weather",
            "Attractions & Map",
            "Trip Planner",
            "Reviews & Analytics",
            "AI Assistant",
            "Currency Converter",
            "Emergency Contacts",
            "Advanced AI Features",
        ],
    )
    profile = get_trip()
    if profile:
        st.sidebar.success(f"Current trip: {profile.dest}\n\n{profile.start:%d %b}–{profile.end:%d %b} · {profile.people} traveller(s)")
    else:
        st.sidebar.warning("No trip saved yet")
    st.sidebar.caption("Offline demo data: TripAdvisor/Google Reviews/OpenStreetMap/Weather API placeholders")
    return page


def main() -> None:
    app_header()
    page = sidebar()
    if page == "Dashboard":
        page_dashboard()
    elif page == "User Profile / Trip Setup":
        page_profile_and_trip_setup()
    elif page == "Budget Planner":
        page_budget_planner()
    elif page == "Hotels & Food":
        page_hotels_food()
    elif page == "Flights & Routes":
        page_flights_routes()
    elif page == "Weather":
        page_weather()
    elif page == "Attractions & Map":
        page_attractions_map()
    elif page == "Trip Planner":
        page_itinerary()
    elif page == "Reviews & Analytics":
        page_reviews_analytics()
    elif page == "AI Assistant":
        page_ai_assistant()
    elif page == "Currency Converter":
        page_currency()
    elif page == "Emergency Contacts":
        page_emergency()
    elif page == "Advanced AI Features":
        page_advanced_ai()


if __name__ == "__main__":
    main()
