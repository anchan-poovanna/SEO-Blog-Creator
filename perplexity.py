# perplexity.py
import streamlit as st
import requests

def deep_research(query: str) -> dict:
    # Get API key from Streamlit secrets
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY", "")
    
    if not PERPLEXITY_API_KEY:
        st.error("PERPLEXITY_API_KEY is not set in Streamlit secrets")
        return {}
    
    URL = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar-deep-research",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 1024,
        "return_related_questions": True,
        "stream": False
    }
    response = requests.post(URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Deep research error {response.status_code}: {response.text}")
        return {}