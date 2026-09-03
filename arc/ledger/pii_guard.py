"""The write-guard on the Decision Ledger. It refuses; it does not clean up.

Every string that reaches the immutable ledger passes through here first. A hit
raises and the write does not happen. There is no bypass flag, no redact-and-
continue, no warn-and-log, because a guard with a bypass is not a guard.

WHY it has to be at this boundary: free-text bank narrations flow downstream
attached to the claim. Scrubbing only the LLM path leaves the ledger exposed,
and once a name is hash-chained into an append-only log, erasure is impossible
without breaking the chain. M1 caps evidence strings at 128 characters, which
is a shape rule and not a content rule: "RAJESH KUMAR AC 50100234567890 INSUF
FUNDS" is 43 characters and has to be rejected on what it contains.

The guard is deliberately tuned to over-trigger. A false positive costs a
developer a field rename. A false negative is unerasable.

It never puts what it found into its own exception message. A guard that leaks
the value it caught has simply moved the leak into the log.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PIIKind(StrEnum):
    EMAIL = "email"
    PHONE_IN = "phone_in"
    AADHAAR = "aadhaar"
    PAN_CARD = "pan_card"
    IFSC = "ifsc"
    CARD_PAN = "card_pan"
    BANK_ACCOUNT = "bank_account"
    NAME_TOKEN = "name_token"


@dataclass(frozen=True)
class PIIHit:
    """What was found and where, never the value itself.

    `fingerprint` is a truncated digest, so two reports can be correlated
    without either of them disclosing anything.
    """

    kind: PIIKind
    path: str
    length: int
    fingerprint: str

    def __str__(self) -> str:
        return f"{self.kind} at {self.path or '<root>'} (len={self.length}, fp={self.fingerprint})"


class PIIDetected(Exception):
    """Raised by the guard. Carries the hits, never the matched text."""

    def __init__(self, hits: Sequence[PIIHit]) -> None:
        self.hits: tuple[PIIHit, ...] = tuple(hits)
        detail = "\n".join(f"  {hit}" for hit in self.hits)
        super().__init__(f"PII write-guard refused the write, {len(self.hits)} hit(s):\n{detail}")


# ---------------------------------------------------------------------------
# System identifiers, masked before the numeric detectors run.
#
# A subject token is 32 hex characters and will sometimes contain a run of nine
# or more digits purely by chance, which the bank-account detector would then
# report. These are already pseudonymous, so removing them first buys precision
# without weakening the guard.
# ---------------------------------------------------------------------------
_IDENTIFIER_PATTERNS = (
    re.compile(r"\bsub_[0-9a-f]{32}\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,128}(?![0-9a-fA-F])"),
)


def _mask_identifiers(text: str) -> str:
    """Blank out system identifiers, preserving offsets and length."""
    for pattern in _IDENTIFIER_PATTERNS:
        text = pattern.sub(lambda m: "#" * len(m.group(0)), text)
    return text


# ---------------------------------------------------------------------------
# Detectors, in precedence order. The first match on a span wins, so a ten
# digit mobile number is reported as a phone and not as a bank account.
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PAN_CARD_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])")
IFSC_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])")
CARD_CANDIDATE_RE = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[ \-]?\d{4}[ \-]?\d{4}(?!\d)")
PHONE_IN_RE = re.compile(r"(?<!\d)(?:\+?91[ \-]?)?[6-9]\d{9}(?!\d)")
BANK_ACCOUNT_RE = re.compile(r"(?<!\d)\d{9,18}(?!\d)")


def luhn_valid(digits: str) -> bool:
    """Standard mod-10 checksum. Keeps arbitrary long numbers out of CARD_PAN."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Name-token heuristic, applied only to free text.
#
# A value containing whitespace is the only thing that can plausibly be a
# narration. Rule ids (TIME-WINDOW), enum values (block_permanent) and hashes
# never contain a space, so they are outside the heuristic entirely rather than
# relying on the vocabulary below to spare them.
# ---------------------------------------------------------------------------
NAME_CANDIDATE_RE = re.compile(r"\b(?:[A-Z][a-z]{2,19}|[A-Z]{3,20})\b")

# Vocabulary that legitimately appears capitalised in a payment narration.
_NARRATION_WORDS = frozenset(
    {
        "ACC",
        "ACCOUNT",
        "ACCT",
        "ACH",
        "AND",
        "AUTO",
        "AUTOPAY",
        "AVAILABLE",
        "BAL",
        "BALANCE",
        "BANK",
        "BLOCKED",
        "CANCELLED",
        "CARD",
        "CHARGE",
        "CHEQUE",
        "CLOSED",
        "COLLECT",
        "CRD",
        "CREDIT",
        "DATE",
        "DEBIT",
        "DECLINE",
        "DECLINED",
        "DISHONOUR",
        "DISHONOURED",
        "ECS",
        "EMI",
        "EXCEEDED",
        "EXPIRED",
        "FAILED",
        "FOR",
        "FROM",
        "FUND",
        "FUNDS",
        "IMPS",
        "INR",
        "INSUF",
        "INSUFF",
        "INSUFFICIENT",
        "LIMIT",
        "LOST",
        "MANDATE",
        "MAX",
        "MIN",
        "NACH",
        "NEFT",
        "NOT",
        "PAY",
        "PAYMENT",
        "PAYMENTS",
        "REF",
        "REFUND",
        "RETURN",
        "RETURNED",
        "REVERSAL",
        "RTGS",
        "STOLEN",
        "STOP",
        "SUCCESS",
        "THE",
        "TRAN",
        "TRANSACTION",
        "TRANSFER",
        "TXN",
        "UPI",
        "VIA",
        "WITH",
    }
)

# This system's own vocabulary, which also reaches the ledger in upper case.
_SYSTEM_WORDS = frozenset(
    {
        "ABANDONED",
        "ALLOW",
        "BLOCK",
        "CERT",
        "CERTIFICATE",
        "CLAIM",
        "CUSTOMER",
        "DEFER",
        "DEFERRED",
        "DEGRADED",
        "DETECTED",
        "DIAGNOSED",
        "DISPUTED",
        "DRAIN",
        "ESCALATED",
        "FORBORNE",
        "FREEZE",
        "GATE",
        "HEALING",
        "ISSUER",
        "KILL",
        "MERCHANT",
        "NORMAL",
        "OFF",
        "PERMANENT",
        "PLANNED",
        "POWER",
        "PROMISED",
        "RECOVERED",
        "RETRY",
        "REVERSED",
        "SELF",
        "SETTLED",
        "SHADOW",
        "SUBJECT",
        "SUPPRESSED",
        "SWITCH",
        "TOMBSTONE",
        "TREATMENT",
        "UNEXECUTED",
        "UNKNOWN",
        "VETO",
        "WRITTEN",
    }
)

NON_NAME_WORDS = _NARRATION_WORDS | _SYSTEM_WORDS

MIN_NAME_RUN = 2


def _name_runs(text: str) -> list[re.Match[str]]:
    """Matches forming a run of at least two adjacent non-vocabulary words."""
    if not any(char.isspace() for char in text):
        return []

    matches = list(NAME_CANDIDATE_RE.finditer(text))
    hits: list[re.Match[str]] = []
    run: list[re.Match[str]] = []

    for match in matches:
        adjacent = bool(run) and text[run[-1].end() : match.start()].strip() == ""
        if not adjacent:
            if len(run) >= MIN_NAME_RUN:
                hits.extend(run)
            run = []
        if match.group(0).upper() in NON_NAME_WORDS:
            if len(run) >= MIN_NAME_RUN:
                hits.extend(run)
            run = []
            continue
        run.append(match)

    if len(run) >= MIN_NAME_RUN:
        hits.extend(run)
    return hits


# Keys whose numeric values are money or metrics, not identifiers. Only these
# skip the numeric detectors, and only when the value is actually a number: a
# nine digit rupee total must not read as a bank account.
NUMERIC_SAFE_SUFFIXES = (
    "_paise",
    "_count",
    "_attempts",
    "_days",
    "_hours",
    "_minutes",
    "_seconds",
    "_ms",
    "_bps",
    "_index",
    "_version",
    "_size",
    "_bytes",
    "_seq",
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


class PIIGuard:
    """Scans a payload and refuses it. The only public entry point is `scan`.

    There is intentionally no non-raising inspection method: a second way in is
    the shape a bypass takes.
    """

    def scan(self, payload: Any, *, path: str = "") -> list[PIIHit]:
        """Return an empty list, or raise `PIIDetected`. Never returns hits."""
        hits: list[PIIHit] = []
        self._walk(payload, path, hits)
        if hits:
            raise PIIDetected(hits)
        return hits

    # -- traversal ---------------------------------------------------------
    def _walk(self, value: Any, path: str, hits: list[PIIHit]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                self._scan_text(str(key), child, hits, numeric=True)
                self._walk(item, child, hits)
            return

        if isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                self._walk(item, f"{path}[{index}]", hits)
            return

        if isinstance(value, bytes):
            # Digests will not decode; a smuggled narration will.
            with contextlib.suppress(UnicodeDecodeError):
                self._scan_text(value.decode("utf-8"), path, hits, numeric=True)
            return

        if value is None or isinstance(value, bool):
            return

        if isinstance(value, str):
            self._scan_text(value, path, hits, numeric=True)
            return

        if isinstance(value, (int, float)):
            leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
            numeric = not leaf.endswith(NUMERIC_SAFE_SUFFIXES)
            self._scan_text(str(value), path, hits, numeric=numeric)
            return

        self._scan_text(str(value), path, hits, numeric=True)

    # -- detection ---------------------------------------------------------
    def _scan_text(self, text: str, path: str, hits: list[PIIHit], *, numeric: bool) -> None:
        if not text:
            return

        masked = _mask_identifiers(text)
        claimed: list[tuple[int, int]] = []

        def take(start: int, end: int) -> bool:
            if any(start < seen_end and seen_start < end for seen_start, seen_end in claimed):
                return False
            claimed.append((start, end))
            return True

        def record(match: re.Match[str], kind: PIIKind) -> None:
            if not take(match.start(), match.end()):
                return
            found = match.group(0)
            hits.append(
                PIIHit(
                    kind=kind,
                    path=path or "<root>",
                    length=len(found),
                    fingerprint=_fingerprint(found),
                )
            )

        for match in EMAIL_RE.finditer(masked):
            record(match, PIIKind.EMAIL)
        for match in PAN_CARD_RE.finditer(masked):
            record(match, PIIKind.PAN_CARD)
        for match in IFSC_RE.finditer(masked):
            record(match, PIIKind.IFSC)

        if numeric:
            for match in CARD_CANDIDATE_RE.finditer(masked):
                digits = re.sub(r"\D", "", match.group(0))
                if 13 <= len(digits) <= 19 and luhn_valid(digits):
                    record(match, PIIKind.CARD_PAN)
            for match in AADHAAR_RE.finditer(masked):
                record(match, PIIKind.AADHAAR)
            for match in PHONE_IN_RE.finditer(masked):
                record(match, PIIKind.PHONE_IN)
            for match in BANK_ACCOUNT_RE.finditer(masked):
                record(match, PIIKind.BANK_ACCOUNT)

        for match in _name_runs(masked):
            record(match, PIIKind.NAME_TOKEN)
