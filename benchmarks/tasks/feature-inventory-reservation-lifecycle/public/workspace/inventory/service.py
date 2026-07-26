from inventory.models import StockItem
from inventory.store import InventoryStore


class InventoryService:
    # 组装库存查询与调整服务
    def __init__(self, store: InventoryStore) -> None:
        self._store = store

    # 查询 SKU 当前可用数量
    def available(self, sku: str) -> int:
        return self._store.get(sku).available

    # 增加已有 SKU 的可用数量
    def add_stock(self, sku: str, quantity: int) -> StockItem:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return self._store.adjust(sku, quantity)
