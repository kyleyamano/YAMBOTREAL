import requests

BASE_URL = "https://api.topstepx.com"

def login_key(username: str, api_key: str) -> str:
    url = f"{BASE_URL}/api/Auth/loginKey"

    payload = {
        "userName": username,
        "apiKey": api_key
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise Exception(f"Login failed: {data}")

    return data["token"]
