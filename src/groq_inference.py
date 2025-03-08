# src/groq_inference.py
import requests

def call_groq_api(api_key, model, messages, temperature=0.0, max_tokens=150, n=1):
    """
    Call the GROQ API to obtain a chat completion.
    
    :param api_key: Your GROQ API key.
    :param model: Model name (e.g., "llama-3.3-70b-versatile").
    :param messages: List of message dictionaries.
    :param temperature: Generation temperature.
    :param max_tokens: Maximum tokens to generate.
    :param n: Number of responses.
    :return: JSON response from the API.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def get_response_from_groq(prompt, api_key, model="llama-3.3-70b-versatile"):
    """
    Given a prompt, call the GROQ API and return the generated SQL text.
    
    :param prompt: The prompt string.
    :param api_key: Your GROQ API key.
    :param model: Model to use.
    :return: Generated SQL text.
    """
    messages = [{"role": "user", "content": prompt}]
    response = call_groq_api(api_key, model, messages)
    try:
        content = response["choices"][0]["message"]["content"].strip()
        return content
    except (KeyError, IndexError):
        return "Error: No valid response from GROQ API."
