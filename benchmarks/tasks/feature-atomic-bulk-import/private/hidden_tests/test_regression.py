import pytest
from orders import (
    InMemoryOrderStore,
    StoreCapacityError,
    create_order,
)


def test_single_order_create_read_and_validation_remain_available() -> None:
    store = InMemoryOrderStore(max_orders=1)
    order = create_order(
        store,
        {"order_id": "single", "sku": "pen", "quantity": "2"},
    )

    assert store.get("single") == order
    with pytest.raises(ValueError):
        create_order(
            InMemoryOrderStore(),
            {"order_id": "bad", "sku": "", "quantity": "2"},
        )
    with pytest.raises(StoreCapacityError):
        create_order(
            store,
            {"order_id": "extra", "sku": "paper", "quantity": "1"},
        )
