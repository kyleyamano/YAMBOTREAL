import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.execution.order_models import Order
from src.execution.broker_paper import PaperBroker


def main():
    broker = PaperBroker()
    broker.connect()

    order = Order(
        symbol="MNQ",
        side="BUY",
        quantity=1,
        order_type="MARKET"
    )

    response = broker.send_order(order)
    print(response)


if __name__ == "__main__":
    main()
