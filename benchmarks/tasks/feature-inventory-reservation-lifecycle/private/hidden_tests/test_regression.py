import pytest
from inventory import InvalidQuantity, InventoryService, InventoryStore, UnknownSku


# 功能：验证既有库存查询和增量调整保持稳定
# 设计：通过 existing service API 保护引入 lifecycle 前的成功路径
def test_existing_query_and_adjustment_contract() -> None:
    store = InventoryStore()
    store.add_item("pen", 4)
    service = InventoryService(store)

    service.add_stock("pen", 3)

    assert service.available("pen") == 7


# 功能：验证既有未知 SKU 和负库存保护保持稳定
# 设计：分别检查异常类型和失败后的原库存，保护旧错误合同
def test_existing_error_contract() -> None:
    store = InventoryStore()
    store.add_item("pen", 1)

    with pytest.raises(UnknownSku):
        store.get("missing")
    with pytest.raises(InvalidQuantity, match="available must not be negative"):
        store.adjust("pen", -2)
    assert store.get("pen").available == 1
