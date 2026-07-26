from querylang.errors import QuerySyntaxError


def split_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    current: list[str] = []
    in_quotes = False
    escaped = False
    for character in query:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\":
            current.append(character)
            escaped = True
            continue
        if character == '"':
            current.append(character)
            in_quotes = not in_quotes
            continue
        if character.isspace() and not in_quotes:
            if current:
                terms.append("".join(current))
                current = []
            continue
        current.append(character)
    if escaped:
        raise QuerySyntaxError("dangling escape")
    if current:
        terms.append("".join(current))
    return tuple(terms)
