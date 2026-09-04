"""L2: cause attribution.

Not "what code came back" but "whose fault, and can it be fixed without
touching the customer". The layer matters more than the label: an issuer-layer
cause requires zero customer contact, a merchant-layer cause is repaired
silently at the rail, and only a customer-layer cause justifies outreach.

The four checks run in a strict order and the first confident hit wins. That
order is the design, not a preference - see `diagnose.py`.
"""
