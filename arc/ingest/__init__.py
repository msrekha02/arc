"""L0 and L1: get external reality in, exactly once, without trusting it.

Two boundaries live in this package and nothing else does.

THE TRUST BOUNDARY (L0, `adapters/`). A webhook is attacker-controlled input to
a money-moving system until its signature has been checked against the raw
bytes. Verification therefore happens before deserialisation, the raw payload is
archived before it is parsed, and adapters translate without deciding anything.

THE REDACTION BOUNDARY (L1, `normaliser.py`). One side is ledgerable: closed
vocabulary, structured, pseudonymous. The other is erasable: names, numbers,
bank narrations, the raw payload. A claim carries a pointer and a hash across
that line and nothing else.
"""
