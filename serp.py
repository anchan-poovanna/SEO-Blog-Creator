# serp.py
import streamlit as st
import requests
import time

def get_search_results(query: str, num_results: int = 10) -> dict:
    # Get API key from Streamlit secrets
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
    
    if not SERPAPI_KEY:
        st.error("SERPAPI_KEY is not set in Streamlit secrets")
        return {}
    
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results,
        "hl": "en",  # language
        "gl": "us"   # country
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return {}
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2)
    return {}