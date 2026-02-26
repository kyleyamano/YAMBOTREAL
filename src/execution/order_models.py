from dataclasses import dataclass
from typing import Optional

@dataclass
class Order:
    symbol: str
    side: str            # "BUY" or "SELL"
    quantity: int
    order_type: str      # "MARKET", "LIMIT"
    price: Optional[float] = None
    stop_price: Optional[float] = None
