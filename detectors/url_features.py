"""URL normalisation shared by training and inference (string analysis only, the URL is never fetched)."""
import re
from urllib.parse import urlsplit

_SCHEME = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://')


def normalize_url(raw: str) -> str:
    """Return the form used in the training data: no scheme, no credentials, no 'www.', no fragment,
    and always a path separator after the host (every 'good' URL in the data has one)."""
    s = raw.strip()
    if _SCHEME.match(s):
        s = _SCHEME.sub('', s, count=1)
    s = s.split('#', 1)[0]
    if '@' in s.split('/', 1)[0]:
        s = s.split('@', 1)[1]
    if s.lower().startswith('www.'):
        s = s[4:]
    if '/' not in s:
        s += '/'
    return s


def host_of(normalized: str) -> str:
    return normalized.split('/', 1)[0].split(':', 1)[0].lower()


def registered_domain(host: str) -> str:
    parts = host.split('.')
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {'co', 'com', 'org', 'net', 'ac', 'gov', 'edu'}:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])
