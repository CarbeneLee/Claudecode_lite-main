from querylang.errors import QuerySyntaxError


def split_terms(query: str) -> tuple[str, ...]:
    if query.endswith("\\"):
        raise QuerySyntaxError("dangling escape")
    if query.count('"') % 2:
        raise QuerySyntaxError("unclosed quote")
    return tuple(query.split())
