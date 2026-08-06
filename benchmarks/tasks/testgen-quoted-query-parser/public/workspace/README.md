# Query Language

A query is a whitespace-separated sequence of `field:value` terms. Fields must
be non-empty identifiers. Values may be unquoted single tokens or double-quoted
strings.

Quoted values:

- may contain whitespace and `:` delimiters;
- use `\` to escape the next quote or backslash;
- may be empty.

An unclosed quote, dangling escape, missing field, or missing unquoted value is
invalid and raises `QuerySyntaxError`. `parse_query` preserves term order and
returns `Term` objects.
