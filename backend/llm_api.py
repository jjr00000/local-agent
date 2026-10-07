import requests

def call_llm(messages, temperature=0.7, max_tokens=1024):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    payload = {
        "model": "local-bionic",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    return data["choices"][0]["message"]["content"]
