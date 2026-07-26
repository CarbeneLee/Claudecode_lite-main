class InventoryError(Exception):
    pass


class UnknownSku(InventoryError):
    pass


class InvalidQuantity(InventoryError):
    pass


class InsufficientStock(InventoryError):
    pass


class RequestConflict(InventoryError):
    pass


class InvalidTransition(InventoryError):
    pass


class UnknownReservation(InventoryError):
    pass
