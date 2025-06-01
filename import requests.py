import requests

url = "https://Sugesh23-email-api.hf.space/classify_email"

payload = {
    "sender": "user@example.com",
    "subject": "Cannot access account",
    "content": "I forgot my password and can't log in."
}

response = requests.post(url, json=payload)
print(response.json())
