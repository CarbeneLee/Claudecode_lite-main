from querylang import Term, parse_query


def test_parse_query_keeps_unquoted_and_quoted_terms_in_order() -> None:
    assert parse_query('status:open owner:"Ada Lovelace"') == (
        Term(field="status", value="open"),
        Term(field="owner", value="Ada Lovelace"),
    )
