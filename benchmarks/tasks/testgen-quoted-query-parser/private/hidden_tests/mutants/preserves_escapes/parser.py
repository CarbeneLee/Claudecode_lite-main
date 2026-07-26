from querylang.errors import QuerySyntaxError
from querylang.models import Term
from querylang.tokenizer import split_terms


def _decode_quoted(raw: str) -> str:
    return raw[1:-1]


def parse_query(query: str) -> tuple[Term, ...]:
    parsed: list[Term] = []
    for token in split_terms(query):
        if ":" not in token:
            raise QuerySyntaxError("term must contain ':'")
        field, raw_value = token.split(":", 1)
        if not field.isidentifier():
            raise QuerySyntaxError("invalid field")
        if raw_value.startswith('"'):
            value = _decode_quoted(raw_value)
        elif not raw_value:
            raise QuerySyntaxError("missing value")
        elif '"' in raw_value:
            raise QuerySyntaxError("invalid quote")
        else:
            value = raw_value
        parsed.append(Term(field=field, value=value))
    return tuple(parsed)
