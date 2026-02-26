from typing import Dict
from .broker_interface import BrokerInterface
from .order_models import Order

class PaperBroker(BrokerInterface):

    def connect(self):
        print("Paper broker connected.")

    def send_order(self, order: Order) -> Dict:
        print(f"PAPER ORDER: {order}")
        return {
            "status": "FILLED",
            "price": order.price,
            "quantity": order.quantity
        }

    def get_positions(self):
        return {}

    def get_account_info(self):
        return {}
