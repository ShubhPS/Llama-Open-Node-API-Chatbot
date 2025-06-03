from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import json
import asyncio
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
import re
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# API Key
WEATHER_API_KEY = os.getenv('W_KEY')
LLM_API_KEY = os.getenv('L_KEY')




app = FastAPI(title="Enhanced Multi-Agent IT Support System")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Weather API functions
def detect_weather_request(user_input: str) -> bool:
    """Detect if user is asking for weather information"""
    weather_keywords = [
        'weather', 'temperature', 'forecast', 'rain', 'snow', 'sunny', 'cloudy',
        'hot', 'cold', 'humid'
    ]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in weather_keywords)


def extract_location_from_text(user_input: str) -> Optional[str]:
    """Try to extract location from user input"""
    location_patterns = [
        r'weather in (.+?)(?:\s|$|[.!?])',
        r'weather for (.+?)(?:\s|$|[.!?])',
        r'weather at (.+?)(?:\s|$|[.!?])',
        r'weather of (.+?)(?:\s|$|[.!?])',
        r'in (.+?) weather',
        r'(.+?) weather',
    ]

    for pattern in location_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            location = match.group(1).strip()
            excluded_words = ['the', 'today', 'tomorrow', 'now', 'current', 'good', 'bad']
            if location not in excluded_words and len(location) > 2:
                return location.title()
    return None


def get_coordinates_from_location(location: str, api_key: str) -> Optional[Tuple[float, float]]:
    """Get coordinates from location name using OpenWeatherMap Geocoding API"""
    geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': location,
        'limit': 1,
        'appid': api_key
    }

    try:
        response = requests.get(geocoding_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data and len(data) > 0:
            return (data[0]['lat'], data[0]['lon'])
        return None

    except requests.exceptions.RequestException as e:
        print(f"Geocoding error: {e}")
        return None


def get_weather_by_coordinates(lat: float, lon: float, api_key: str) -> dict:
    """Get weather data using lat/lon coordinates"""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to fetch weather data: {str(e)}'}


def format_weather_response(weather_data: dict, location: str) -> str:
    """Format weather data into a readable response"""
    if 'error' in weather_data:
        return f"Sorry, I couldn't get weather information for {location}. {weather_data['error']}"

    try:
        main = weather_data['main']
        weather = weather_data['weather'][0]
        wind = weather_data.get('wind', {})

        temp = main['temp']
        feels_like = main['feels_like']
        humidity = main['humidity']
        pressure = main.get('pressure', 'N/A')
        description = weather['description'].title()
        wind_speed = wind.get('speed', 'N/A')
        wind_direction = wind.get('deg', 'N/A')

        visibility = weather_data.get('visibility', 'N/A')
        if visibility != 'N/A':
            visibility = f"{visibility / 1000} km"

        response = f"""🌤️ Weather in {weather_data['name']}, {weather_data['sys']['country']}:

• Condition: {description}
• Temperature: {temp}°C (feels like {feels_like}°C)
• Humidity: {humidity}%
• Pressure: {pressure} hPa
• Wind: {wind_speed} m/s at {wind_direction}°
• Visibility: {visibility}"""

        return response

    except KeyError as e:
        return f"Sorry, I received incomplete weather data for {location}. Missing: {str(e)}"


# AQI API functions
def detect_aqi_request(user_input: str) -> bool:
    """Detect if user is asking for air quality information"""
    aqi_keywords = [
        'air quality', 'aqi', 'pollution', 'smog', 'air pollution',
        'pm2.5', 'pm10', 'ozone', 'air index', 'pollutants',
        'carbon monoxide', 'nitrogen dioxide', 'sulfur dioxide',
        'clean air', 'dirty air', 'breathable', 'air condition'
    ]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in aqi_keywords)


def extract_location_for_aqi(user_input: str) -> Optional[str]:
    """Try to extract location from user input for AQI requests"""
    location_patterns = [
        r'air quality in (.+?)(?:\s|$|[.!?])',
        r'air quality for (.+?)(?:\s|$|[.!?])',
        r'air quality at (.+?)(?:\s|$|[.!?])',
        r'air quality of (.+?)(?:\s|$|[.!?])',
        r'aqi at (.+?)(?:\s|$|[.!?])',
        r'aqi in (.+?)(?:\s|$|[.!?])',
        r'aqi for (.+?)(?:\s|$|[.!?])',
        r'pollution in (.+?)(?:\s|$|[.!?])',
        r'aqi of (.+?)(?:\s|$|[.!?])',
        r'in (.+?) air quality',
        r'(.+?) air quality'
    ]

    for pattern in location_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            location = match.group(1).strip()
            excluded_words = ['the', 'today', 'tomorrow', 'now', 'current', 'good', 'bad']
            if location not in excluded_words and len(location) > 2:
                return location.title()
    return None


def get_air_quality(lat: float, lon: float, api_key: str) -> dict:
    """Get air quality data from OpenWeatherMap Air Pollution API"""
    aqi_url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }

    try:
        response = requests.get(aqi_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': f'Failed to fetch air quality data: {str(e)}'}


def get_aqi_description(aqi_level: int) -> tuple:
    """Get AQI description and emoji based on level"""
    aqi_descriptions = {
        1: ("Good", "🟢", "Air quality is considered satisfactory, and air pollution poses little or no risk"),
        2: ("Fair", "🟡",
            "Air quality is acceptable; however, there may be a moderate health concern for a very small number of people"),
        3: ("Moderate", "🟠",
            "Members of sensitive groups may experience health effects. The general public is not likely to be affected"),
        4: ("Poor", "🔴",
            "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects"),
        5: ("Very Poor", "🟣",
            "Health warnings of emergency conditions. The entire population is more likely to be affected")
    }
    return aqi_descriptions.get(aqi_level, ("Unknown", "⚪", "AQI level unknown"))


def format_aqi_response(aqi_data: dict, location: str) -> str:
    """Format AQI data into a readable response"""
    if 'error' in aqi_data:
        return f"Sorry, I couldn't get air quality information for {location}. {aqi_data['error']}"

    try:
        aqi_info = aqi_data['list'][0]
        aqi_level = aqi_info['main']['aqi']
        components = aqi_info['components']

        description, emoji, health_info = get_aqi_description(aqi_level)

        pm2_5 = components.get('pm2_5', 0)
        pm10 = components.get('pm10', 0)
        o3 = components.get('o3', 0)
        no2 = components.get('no2', 0)
        so2 = components.get('so2', 0)
        co = components.get('co', 0)

        response = f"""{emoji} Air Quality in {location}:

• Overall AQI: {description} (Level {aqi_level}/5)
• Health Impact: {health_info}

🔬 Pollutant Levels (μg/m³):
• PM2.5: {pm2_5:.2f}
• PM10: {pm10:.2f}
• Ozone (O₃): {o3:.2f}
• Nitrogen Dioxide (NO₂): {no2:.2f}
• Sulfur Dioxide (SO₂): {so2:.2f}
• Carbon Monoxide (CO): {co:.2f}"""

        return response

    except (KeyError, IndexError) as e:
        return f"Sorry, I received incomplete air quality data for {location}. Error: {str(e)}"


# Enhanced conversation state management
class ConversationState:
    def __init__(self):
        self.waiting_for = None  # 'weather_location' or 'aqi_location'
        self.context = {}


# Global state store for each session
conversation_states = {}


def get_conversation_state(session_id: str) -> ConversationState:
    """Get or create conversation state for a session"""
    if session_id not in conversation_states:
        conversation_states[session_id] = ConversationState()
    return conversation_states[session_id]


# Initialize the LLM
llm = ChatOpenAI(
    model="meta-llama/llama-3.2-1b-instruct:free",
    openai_api_key=LLM_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that engages in interactive conversations. Ask follow-up questions when you need more information to provide a complete answer."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Store for session histories
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Create the conversation chain with message history
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def enhanced_chat_with_llm(user_input, session_id="default_session"):
    """Enhanced chat function that handles weather and AQI requests with follow-up questions"""

    state = get_conversation_state(session_id)

    # Check if we're waiting for a location response
    if state.waiting_for == 'weather_location':
        # User is providing location for weather
        location = user_input.strip()
        if location:
            coordinates = get_coordinates_from_location(location, WEATHER_API_KEY)
            if coordinates:
                lat, lon = coordinates
                weather_data = get_weather_by_coordinates(lat, lon, WEATHER_API_KEY)
                response = format_weather_response(weather_data, location)
                state.waiting_for = None  # Reset state
                return response
            else:
                return f"Sorry, I couldn't find the location '{location}'. Please try a different city name."
        else:
            return "Please provide a valid city name for the weather information."

    elif state.waiting_for == 'aqi_location':
        # User is providing location for AQI
        location = user_input.strip()
        if location:
            coordinates = get_coordinates_from_location(location, WEATHER_API_KEY)
            if coordinates:
                lat, lon = coordinates
                aqi_data = get_air_quality(lat, lon, WEATHER_API_KEY)
                response = format_aqi_response(aqi_data, location)
                state.waiting_for = None  # Reset state
                return response
            else:
                return f"Sorry, I couldn't find the location '{location}'. Please try a different city name."
        else:
            return "Please provide a valid city name for the air quality information."

    # Check for new weather requests
    elif detect_weather_request(user_input):
        location = extract_location_from_text(user_input)
        if location:
            # Location found in the message, get weather directly
            coordinates = get_coordinates_from_location(location, WEATHER_API_KEY)
            if coordinates:
                lat, lon = coordinates
                weather_data = get_weather_by_coordinates(lat, lon, WEATHER_API_KEY)
                return format_weather_response(weather_data, location)
            else:
                return f"Sorry, I couldn't find the location '{location}'. Please try a different location name."
        else:
            # No location found, ask for it
            state.waiting_for = 'weather_location'
            return "I'd be happy to help you with the weather! 🌤️ Which city would you like to know about?"

    # Check for new AQI requests
    elif detect_aqi_request(user_input):
        location = extract_location_for_aqi(user_input)
        if location:
            # Location found in the message, get AQI directly
            coordinates = get_coordinates_from_location(location, WEATHER_API_KEY)
            if coordinates:
                lat, lon = coordinates
                aqi_data = get_air_quality(lat, lon, WEATHER_API_KEY)
                return format_aqi_response(aqi_data, location)
            else:
                return f"Sorry, I couldn't find the location '{location}'. Please try a different location name."
        else:
            # No location found, ask for it
            state.waiting_for = 'aqi_location'
            return "I'd be happy to help you with air quality information! 🌬️ Which city would you like to check?"

    # Default to LLM conversation
    else:
        try:
            response = conversation.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            return response.content
        except Exception as e:
            return f"I'm having trouble processing that request. Could you please try again? Error: {str(e)}"


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.user_sessions[session_id] = {
            "connected_at": datetime.now(),
            "message_count": 0,
            "weather_requests": 0,
            "aqi_requests": 0
        }
        print(f"Client {session_id} connected")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.user_sessions:
            del self.user_sessions[session_id]
        print(f"Client {session_id} disconnected")

    async def send_personal_message(self, message: str, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(message)
            except:
                self.disconnect(session_id)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            # Update session stats
            if session_id in manager.user_sessions:
                manager.user_sessions[session_id]["message_count"] += 1

                # Count specific request types
                if detect_weather_request(user_message):
                    manager.user_sessions[session_id]["weather_requests"] += 1
                elif detect_aqi_request(user_message):
                    manager.user_sessions[session_id]["aqi_requests"] += 1

            # Process message with your chatbot function
            try:
                bot_response = enhanced_chat_with_llm(user_message, session_id)
            except Exception as e:
                bot_response = f"Sorry, I encountered an error: {str(e)}"

            # Send response back to client with custom JSON encoder
            response = {
                "type": "bot_response",
                "message": bot_response,
                "timestamp": datetime.now().isoformat(),  # Convert to ISO string directly
                "session_stats": manager.user_sessions.get(session_id, {})
            }

            # Use the custom DateTimeEncoder to handle datetime serialization
            await manager.send_personal_message(json.dumps(response, cls=DateTimeEncoder), session_id)

    except WebSocketDisconnect:
        manager.disconnect(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
