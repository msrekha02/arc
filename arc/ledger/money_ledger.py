"""Double-entry money movement, with a path back down.

Every movement writes two legs that sum to zero, so `SELECT sum(delta_paise)`
over the whole table is always zero. That is the invariant, and it is checkable
in one query rather than trusted.

Money enters through the EXTERNAL account, which is what makes the opening
balance balance. Without it the first credit would be a single-entry write and
the invariant would be untestable.

RECOVERY_REVERSED is not an afterthought. Chargebacks, refunds and failed
settlements un-recover money that has already been counted, and a headline
number that cannot move backwards is not a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

from arc.core.money import Paise, paise
from arc.core.time_authority import ensure_utc
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType


class MoneyAccount(StrEnum):
    """Where a claim's money currently sits.

    EXTERNAL is the counter-account that opening and closing entries balance
    against. It is not a state a claim is ever "in".
    """

    EXTERNAL = "external"
    AT_RISK = "at_risk"
    IN_TREATMENT = "in_treatment"
    RECOVERED = "recovered"
    SETTLED = "settled"
    REVERSED = "reversed"


LEGAL_MONEY_TRANSITIONS: Mapping[MoneyAccount, frozenset[MoneyAccount]] = MappingProxyType(
    {
        MoneyAccount.EXTERNAL: frozenset({MoneyAccount.AT_RISK}),
        MoneyAccount.AT_RISK: frozenset({MoneyAccount.IN_TREATMENT, MoneyAccount.RECOVERED}),
        MoneyAccount.IN_TREATMENT: frozenset({MoneyAccount.RECOVERED, MoneyAccount.AT_RISK}),
        MoneyAccount.RECOVERED: frozenset({MoneyAccount.SETTLED, MoneyAccount.REVERSED}),
        MoneyAccount.SETTLED: frozenset({MoneyAccount.REVERSED}),
        # Reversed money is at risk again, not written off.
        MoneyAccount.REVERSED: frozenset({MoneyAccount.AT_RISK, MoneyAccount.IN_TREATMENT}),
    }
)

RECOVERY_REVERSED = (MoneyAccount.RECOVERED, MoneyAccount.REVERSED)


class IllegalMoneyTransition(Exception):
    """A movement between accounts that the ledger does not allow."""


class InsufficientBalance(Exception):
    """More money was moved out of an account than the claim ever put in it."""


@dataclass(frozen=True)
class MoneyMovement:
    group_id: UUID
    claim_id: UUID
    frm: MoneyAccount
    to: MoneyAccount
    amount_paise: Paise
    occurred_at: datetime


class MoneyLedger:
    """Balances are derived from entries. Nothing stores a running total.

    A stored total is a second source of truth that drifts. Every number this
    reports is a sum over the legs, so a reversal moves it without anything
    having to remember to.
    """

    def __init__(self, ledger: DecisionLedger | None = None) -> None:
        self._ledger = ledger

    async def open_claim(
        self, conn: Any, claim_id: UUID, amount: Paise, *, at: datetime
    ) -> MoneyMovement:
        """Bring a claim's failed amount into the books, at risk."""
        return await self.transition(
            conn, claim_id, MoneyAccount.EXTERNAL, MoneyAccount.AT_RISK, amount, at=at
        )

    async def transition(
        self,
        conn: Any,
        claim_id: UUID,
        frm: MoneyAccount,
        to: MoneyAccount,
        amount: Paise,
        *,
        at: datetime,
    ) -> MoneyMovement:
        """Move `amount` from one account to another as two balancing legs."""
        ensure_utc(at)
        amount = paise(amount)
        if amount <= 0:
            raise ValueError(f"a movement must be positive, got {amount}")
        if to not in LEGAL_MONEY_TRANSITIONS[frm]:
            legal = ", ".join(sorted(LEGAL_MONEY_TRANSITIONS[frm])) or "nothing"
            raise IllegalMoneyTransition(f"{frm} -> {to} is not legal; legal: {legal}")

        group_id = uuid4()

        async with conn.transaction():
            if frm is not MoneyAccount.EXTERNAL:
                available = await self.claim_balance(conn, claim_id, frm)
                if amount > available:
                    raise InsufficientBalance(
                        f"claim {claim_id} holds {available} paise in {frm}, cannot move {amount}"
                    )

            await conn.executemany(
                """
                INSERT INTO money_entries
                    (group_id, claim_id, account, delta_paise, occurred_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                [
                    (group_id, claim_id, frm.value, -amount, at),
                    (group_id, claim_id, to.value, amount, at),
                ],
            )

            if self._ledger is not None:
                await self._ledger.append(
                    conn,
                    LedgerEntry(
                        event_type=LedgerEventType.MONEY_TRANSITION,
                        occurred_at=at,
                        claim_id=claim_id,
                        payload={
                            "group_id": str(group_id),
                            "from_account": frm.value,
                            "to_account": to.value,
                            "amount_paise": int(amount),
                            "is_recovery_reversal": (frm, to) == RECOVERY_REVERSED,
                        },
                    ),
                )

        return MoneyMovement(group_id, claim_id, frm, to, amount, at)

    async def reverse_recovery(
        self, conn: Any, claim_id: UUID, amount: Paise, *, at: datetime
    ) -> MoneyMovement:
        """The RECOVERY_REVERSED path, named so it is greppable in the ledger."""
        return await self.transition(
            conn, claim_id, MoneyAccount.RECOVERED, MoneyAccount.REVERSED, amount, at=at
        )

    async def balance(self, conn: Any, account: MoneyAccount) -> Paise:
        """Portfolio balance of one account."""
        total = await conn.fetchval(
            "SELECT coalesce(sum(delta_paise), 0) FROM money_entries WHERE account = $1",
            account.value,
        )
        return paise(int(total))

    async def claim_balance(self, conn: Any, claim_id: UUID, account: MoneyAccount) -> Paise:
        total = await conn.fetchval(
            """
            SELECT coalesce(sum(delta_paise), 0) FROM money_entries
             WHERE claim_id = $1 AND account = $2
            """,
            claim_id,
            account.value,
        )
        return paise(int(total))

    async def claim_balances(self, conn: Any, claim_id: UUID) -> dict[MoneyAccount, Paise]:
        rows = await conn.fetch(
            """
            SELECT account, sum(delta_paise) AS total FROM money_entries
             WHERE claim_id = $1 GROUP BY account
            """,
            claim_id,
        )
        return {MoneyAccount(row["account"]): paise(int(row["total"])) for row in rows}

    async def recovered_total(self, conn: Any) -> Paise:
        """The headline figure. A reversal moves it down with no extra step."""
        return await self.balance(conn, MoneyAccount.RECOVERED)

    async def is_balanced(self, conn: Any) -> bool:
        """The double-entry invariant, as one query over every leg ever written."""
        total = await conn.fetchval("SELECT coalesce(sum(delta_paise), 0) FROM money_entries")
        return int(total) == 0

    async def unbalanced_groups(self, conn: Any) -> list[UUID]:
        """Any movement whose two legs do not cancel. Should always be empty."""
        rows = await conn.fetch(
            """
            SELECT group_id FROM money_entries
             GROUP BY group_id HAVING sum(delta_paise) <> 0
            """
        )
        return [row["group_id"] for row in rows]
