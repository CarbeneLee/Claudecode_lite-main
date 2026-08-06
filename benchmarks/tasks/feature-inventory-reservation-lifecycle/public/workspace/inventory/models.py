from dataclasses import dataclass
from enum import StrEnum


@dataclass
class StockItem:
    sku: str
    available: int


class ReservationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    RELEASED = "released"


@dataclass
class Reservation:
    reservation_id: str
    request_key: str
    sku: str
    quantity: int
    status: ReservationStatus = ReservationStatus.PENDING
