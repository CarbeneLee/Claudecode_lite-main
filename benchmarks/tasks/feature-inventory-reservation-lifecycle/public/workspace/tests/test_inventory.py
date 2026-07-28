import pytest
from inventory import InventoryService, InventoryStore, UnknownSku


# 功能：验证既有查询和补货行为保持稳定
# 设计：通过 public service 查询调整后的 exact quantity
def test_query_and_add_stock() -> None:
    store = InventoryStore()
    store.add_item("pen", 4)
    service = InventoryService(store)

    service.add_stock("pen", 3)

    assert service.available("pen") == 7


# 功能：验证未知 SKU 仍通过既有异常报告
# 设计：只调用 public service，避免绑定 store 内部字典
def test_unknown_sku() -> None:
    service = InventoryService(InventoryStore())

    with pytest.raises(UnknownSku):
        service.available("missing")
