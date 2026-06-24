# 🧭 Travel Around — Smart Tourism Analytics Platform

> An end-to-end, AI-powered travel planning platform built entirely in Python using Streamlit. Plan trips, book hotels, estimate budgets, explore attractions, forecast weather, and get personalized recommendations — all in one place.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [AI & ML Components](#ai--ml-components)
- [Module Breakdown](#module-breakdown)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Challenges Faced](#challenges-faced)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

**Travel Around** is a comprehensive Smart Tourism Analytics Platform that replaces the need for multiple separate travel apps. Built with Python and Streamlit, it offers AI-driven destination recommendations, an Agoda-style hotel booking experience, a fully itemized budget planner, weather forecasting, itinerary generation, fraud detection, and much more — all offline-ready with demo datasets that can be swapped with live APIs.

---

## Problem Statement

Tourists today rely on 6–8 separate applications for trip planning — one for weather, one for hotels, one for navigation, one for currency conversion, and so on. This creates:

- Fragmented and time-consuming trip planning
- Poor budget visibility and unexpected expenses
- Limited personalization based on user preferences
- Difficulty identifying safe, crowd-aware, and cost-effective options

**Travel Around solves all of this by bringing every major tourism service into a single, personalized Python platform.**

---

## Features

### 👤 User Profile & Trip Setup
- Enter traveller name, origin city, destination, travel dates, number of travellers, budget category, travel style, and interests
- All selections persist across the app using Streamlit session state
- Generates personalized recommendations the moment a trip is saved

### 📊 Smart Dashboard
- Displays AI destination match score, predicted crowd level, estimated total budget, and average trip temperature at a glance
- Shows destination facts, crowd advice, and a location map
- Lists top destination recommendations ranked by AI score

### 💰 Budget Planner — Detailed & Editable
The most granular budget planner in the app, with **15 individually adjustable cost parameters**:

| Category | Parameters |
|---|---|
| 🏨 Accommodation | Hotel / stay cost per night |
| 🍽️ Food & Dining | Breakfast, Lunch, Dinner, Snacks (per person per day) |
| 🚗 Local Transport | Taxi/cab, Auto-rickshaw/metro, Fuel (per day) |
| ✈️ Intercity Travel | Total train/bus/flight cost for all travellers |
| 🎟️ Activities | Attraction entry fees, Guided tours & experiences |
| 🛍️ Extras | Shopping & souvenirs, Medicines & toiletries, Tips & donations, Photography fees |

**Live calculations show:**
- Grand total cost
- Per-person cost
- Per-day food and transport cost
- 15% emergency buffer (auto-calculated)
- Interactive bar chart of all spending categories

### 🏨 Hotel Booking (Agoda-Style)
A full hotel booking experience inspired by Agoda:

1. **Search bar** — destination, check-in date, check-out date, rooms, adults
2. **Filter panel** — max price per night, minimum guest rating, minimum star category, sort order (recommended / price / rating / distance), amenity filters (WiFi, Pool, Gym, Spa, etc.)
3. **Property cards** — show hotel name, stars, distance, amenities, guest rating, review count, AI Top Pick badge, price per night, and total cost for the trip
4. **Room type selection** — Standard Room, Deluxe Room, Suite with individual pricing
5. **Booking form** — guest name, email, phone, special requests, payment method, cancellation policy
6. **Booking confirmation** — unique Booking ID generated on confirmation
7. **My Bookings** — summary table of all confirmed bookings with cancel option

### ✈️ Flights & Routes
- Simulated flight options between origin and destination with airline, flight number, departure/arrival time, duration, stops, and price
- Optimized route plan with a live Streamlit map showing attraction locations
- Integration-ready for Google Directions API and OpenStreetMap OSRM

### 🌦️ Weather Forecast
- 14-day weather forecast for trip dates
- Shows daily condition, high/low temperature, humidity, wind speed, and rainfall
- Line chart for temperature and rain trends
- Rain alerts with packing advice

### 🗺️ Attractions & Tourist Map
- Lists top attractions with name, description, rating, and entry fee
- Interactive map with pinned attraction locations
- Monthly crowd index bar chart for the destination

### 🧳 Smart Trip Planner / Itinerary
- Generates a complete day-wise itinerary for the entire trip duration
- Covers morning attractions, lunch, afternoon visits, evening markets, and dining
- Downloadable as CSV

### ⭐ Reviews, Sentiment & Analytics
- Tourist reviews with star ratings and sentiment labels (Positive / Neutral / Negative)
- Sentiment summary bar chart
- Travel demand forecasting with 12-month trend line

### 🤖 AI Travel Chatbot
- Rule-based NLP assistant that answers questions about food, hotels, transport, safety, weather, packing, and budget
- Quick question shortcuts for common queries
- Full chat history with persistent session

### 💱 Currency Converter
- Supports 11 currencies: INR, USD, EUR, GBP, JPY, CAD, AUD, AED, SGD, CNY, THB
- Converts between any two currencies with live-rate-ready placeholders
- Full rates reference table

### 🚨 Emergency Contacts
- 8 essential helplines: Police (112), Ambulance (108), Fire Brigade (101), Tourist Helpline (1363), Women Helpline (1091), Child Helpline (1098), Disaster Management (1070), Cyber Crime (1930)
- Safety checklist for travellers

### 🧠 Advanced AI Features
- **Landmark Recognition** — image upload demo using filename heuristic (CNN/CLIP-ready)
- **Crowd Prediction** — monthly crowd index model with weekend adjustment
- **Fraud Detection** — risk scoring based on booking amount, vendor rating, advance days, payment method, and data mismatch
- **ML Model Summary** — table comparing current demo implementations with production-grade upgrade paths

---

## Technology Stack

| Technology | Role |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | UI framework — pages, widgets, session state, maps, charts, file uploader, chat |
| **Pandas** | Data manipulation — hotels, flights, weather, itinerary, reviews, budget tables |
| **NumPy** | Numerical operations and seeded random generation |
| **hashlib** | Deterministic hashing for reproducible offline demo data |
| **math** | Scoring functions, demand forecasting (sin-wave seasonality), crowd calculations |
| **datetime / calendar** | Trip date handling, itinerary generation, weather forecast windows |
| **dataclasses** | Typed `TripProfile` object for session data |

---

## AI & ML Components

| Feature | Current Implementation | Production Upgrade |
|---|---|---|
| Destination Recommendation | Content-based weighted scoring | Collaborative Filtering / Matrix Factorization |
| Hotel Ranking | Multi-factor scoring (rating, stars, distance, price) | XGBoost / Neural Ranking |
| Demand Forecasting | Seasonal time-series simulation | ARIMA / Prophet / LSTM |
| Fraud Detection | Rule-based risk scoring | XGBoost / Random Forest |
| Sentiment Analysis | Stored labels + rule-ready | BERT / RoBERTa |
| Landmark Recognition | Filename heuristic demo | CNN / CLIP / Vision Transformer |
| Chatbot | Rule-based intent matching | LLM + RAG + BERT |

---

## Module Breakdown
