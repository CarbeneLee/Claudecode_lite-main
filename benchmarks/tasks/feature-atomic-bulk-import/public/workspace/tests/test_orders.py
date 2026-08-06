import pytest
from orders import InMemoryOrderStore, StoreCapacityError, create_order


def test_create_order_parses_and_persists_one_row() -> None:
    store = InMemoryOrderStore()

    order = create_order(
        store,
        {"order_id": "o-1", "sku": "pen", "quantity": "2"},
    )

    assert store.get("o-1") == order
    assert order.quantity == 2


def test_single_order_validation_does_not_mutate_store() -> None:
    store = InMemoryOrderStore()

    with pytest.raises(ValueError):
        create_order(
            store,
            {"order_id": "bad", "sku": "pen", "quantity": "0"},
        )

    assert store.all_orders() == ()


def test_store_capacity_is_enforced() -> None:
    store = InMemoryOrderStore(max_orders=1)
    create_order(
        store,
        {"order_id": "o-1", "sku": "pen", "quantity": "1"},
    )

    with pytest.raises(StoreCapacityError):
        create_order(
            store,
            {"order_id": "o-2", "sku": "paper", "quantity": "1"},
        )
