import requests
from typing import Dict
from .broker_interface import BrokerInterface
from .order_models import Order

BASE_URL = "https://api.topstepx.com"

class TopstepBroker(BrokerInterface):

    def __init__(self, username: str, api_key: str):
        self.username = username
        self.api_key = api_key
        self.token = None

    def connect(self) -> None:
        url = f"{BASE_URL}/api/Auth/loginKey"
        payload = {
            "userName": self.username,
            "apiKey": self.api_key
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise Exception(f"Login failed: {data}")

        self.token = data["token"]

    def send_order(self, order: Order) -> Dict:
        if not self.token:
            raise Exception("Not connected")

        url = f"{BASE_URL}/api/Order/send"

        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        payload = {
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "orderType": order.order_type,
            "price": order.price,
            "stopPrice": order.stop_price
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_positions(self) -> Dict:
        raise NotImplementedError

    def get_account_info(self) -> Dict:
        raise NotImplementedError
