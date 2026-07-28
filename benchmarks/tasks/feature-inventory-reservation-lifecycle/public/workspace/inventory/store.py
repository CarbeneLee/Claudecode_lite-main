from __future__ import annotations

from inventory.errors import InvalidQuantity, UnknownSku
from inventory.models import StockItem


class InventoryStore:
    # 初始化内存库存存储
    def __init__(self) -> None:
        self._items: dict[str, StockItem] = {}

    # 新增一个库存项目
    def add_item(self, sku: str, available: int) -> StockItem:
        if available < 0:
            raise InvalidQuantity("available must not be negative")
        if sku in self._items:
            raise ValueError("sku already exists")
        item = StockItem(sku=sku, available=available)
        self._items[sku] = item
        return item

    # 按 SKU 查询库存项目
    def get(self, sku: str) -> StockItem:
        try:
            return self._items[sku]
        except KeyError as exc:
            raise UnknownSku(sku) from exc

    # 原子调整库存并拒绝负库存
    def adjust(self, sku: str, delta: int) -> StockItem:
        item = self.get(sku)
        next_available = item.available + delta
        if next_available < 0:
            raise InvalidQuantity("available must not be negative")
        item.available = next_available
        return item
