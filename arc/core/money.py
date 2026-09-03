"""The only monetary type in the repo.

Money is integer paise. No float, no Decimal, anywhere - not in a dataclass
field, not in a database column, not in an intermediate calculation.

WHY (GI-2): float arithmetic on money produces silent, compounding errors, and
the headline number of this system is a sum of money. An error that rounds the
wrong way on 50,000 claims is invisible in code review and fatal under audit.

Formatting to rupees happens ONLY at the presentation boundary - `format_inr`
returns a string and nothing consumes it back.
"""

from __future__ import annotations

from typing import NewType

Paise = NewType("Paise", int)

PAISE_PER_RUPEE = 100

# U+20B9 INDIAN RUPEE SIGN, written as an escape so this file stays ASCII.
RUPEE_SIGN = "\u20b9"


class NotMoney(TypeError):
    """A non-integer was offered where paise were required."""


def paise(value: int) -> Paise:
    """Construct Paise, rejecting anything that is not a plain int.

    `bool` is rejected explicitly: it is a subclass of int, so `True` would
    otherwise silently become one paise.
    """
    if isinstance(value, bool):
        raise NotMoney("bool is not a monetary amount")
    if not isinstance(value, int):
        raise NotMoney(
            f"money must be integer paise, got {type(value).__name__} ({value!r}); "
            "float and Decimal are banned for monetary values (GI-2)"
        )
    return Paise(value)


def from_rupees(rupees: int) -> Paise:
    """Convenience constructor for fixtures and rule thresholds. Integers only."""
    return paise(paise(rupees) * PAISE_PER_RUPEE)


def _group_indian(whole: int) -> str:
    """Indian digit grouping: last three digits, then pairs. 12345678 -> 1,23,45,678."""
    digits = str(whole)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups) + "," + tail


def format_inr(amount: Paise) -> str:
    """Render paise as rupees for display. PRESENTATION BOUNDARY ONLY.

    Nothing parses this back into a number. If you find yourself wanting to,
    you are carrying money as a string and should be carrying Paise.
    """
    if isinstance(amount, bool) or not isinstance(amount, int):
        raise NotMoney(f"format_inr expects Paise, got {type(amount).__name__}")
    sign = "-" if amount < 0 else ""
    whole, fraction = divmod(abs(amount), PAISE_PER_RUPEE)
    return f"{sign}{RUPEE_SIGN}{_group_indian(whole)}.{fraction:02d}"
