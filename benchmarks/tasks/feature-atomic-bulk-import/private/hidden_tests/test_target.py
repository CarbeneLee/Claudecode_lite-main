import pytest
from orders import (
    DuplicateOrderError,
    ImportSummary,
    InMemoryOrderStore,
    StoreCapacityError,
    create_order,
    import_orders,
)


def _row(order_id: str, *, quantity: str = "1") -> dict[str, str]:
    return {"order_id": order_id, "sku": f"sku-{order_id}", "quantity": quantity}


def test_valid_and_empty_batches_return_stable_summaries() -> None:
    store = InMemoryOrderStore()
    create_order(store, _row("existing"))

    result = import_orders(store, [_row("o-1"), _row("o-2", quantity="3")])

    assert result == ImportSummary(
        imported_count=2,
        order_ids=("o-1", "o-2"),
    )
    assert tuple(order.order_id for order in store.all_orders()) == (
        "existing",
        "o-1",
        "o-2",
    )
    assert import_orders(store, []) == ImportSummary(
        imported_count=0,
        order_ids=(),
    )


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([_row("valid"), _row("invalid", quantity="0")], ValueError),
        ([_row("same"), _row("same")], DuplicateOrderError),
    ],
)
def test_row_failure_rolls_back_the_entire_batch(
    rows: list[dict[str, str]],
    error: type[Exception],
) -> None:
    store = InMemoryOrderStore()
    create_order(store, _row("existing"))
    before = store.all_orders()

    with pytest.raises(error):
        import_orders(store, rows)

    assert store.all_orders() == before


def test_existing_duplicate_and_capacity_failure_leave_store_unchanged() -> None:
    duplicate_store = InMemoryOrderStore()
    create_order(duplicate_store, _row("existing"))
    duplicate_before = duplicate_store.all_orders()

    with pytest.raises(DuplicateOrderError):
        import_orders(duplicate_store, [_row("new"), _row("existing")])

    assert duplicate_store.all_orders() == duplicate_before

    limited_store = InMemoryOrderStore(max_orders=2)
    create_order(limited_store, _row("existing"))
    capacity_before = limited_store.all_orders()

    with pytest.raises(StoreCapacityError):
        import_orders(limited_store, [_row("new-1"), _row("new-2")])

    assert limited_store.all_orders() == capacity_before
