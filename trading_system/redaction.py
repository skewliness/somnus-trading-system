from __future__ import annotations

import re
from urllib import parse


SENSITIVE_QUERY_KEYS = {
    "apikey",
    "api_key",
    "signature",
    "timestamp",
    "recvwindow",
    "secret",
    "api_secret",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
}

_KEY_VALUE_RE = re.compile(
    r"(?i)(?P<prefix>\b(?:"
    + "|".join(re.escape(key) for key in sorted(SENSITIVE_QUERY_KEYS, key=len, reverse=True))
    + r")\s*=\s*)(?P<value>[^&\s,;\"'<>]+)"
)


def _redact_query(query: str) -> str:
    pairs = parse.parse_qsl(query, keep_blank_values=True)
    if not pairs:
        return _KEY_VALUE_RE.sub(r"\g<prefix><redacted>", query)
    redacted: list[tuple[str, str]] = []
    for key, value in pairs:
        if key.lower() in SENSITIVE_QUERY_KEYS:
            redacted.append((key, "<redacted>"))
        else:
            redacted.append((key, value))
    return parse.urlencode(redacted)


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        parsed = parse.urlsplit(raw)
    except ValueError:
        return _KEY_VALUE_RE.sub(r"\g<prefix><redacted>", raw)
    if not parsed.query:
        return raw
    return parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            _redact_query(parsed.query),
            parsed.fragment,
        )
    )


_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def redact_sensitive_text(value: object) -> str:
    text = str(value)
    text = _URL_RE.sub(_redact_url, text)
    return _KEY_VALUE_RE.sub(r"\g<prefix><redacted>", text)
