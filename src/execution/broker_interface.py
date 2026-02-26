from abc import ABC, abstractmethod
from typing import Any

class BrokerInterface(ABC):

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def send_order(self, order: Any) -> dict:
        pass

    @abstractmethod
    def get_positions(self) -> dict:
        pass

    @abstractmethod
    def get_account_info(self) -> dict:
        pass
