"""Request signing for authenticated Kalshi endpoints.

Each request carries KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP (ms) and
KALSHI-ACCESS-SIGNATURE = base64(sign(timestamp + METHOD + path)), where path is
the full URL path from the host (e.g. /trade-api/v2/portfolio/balance) without
the query string. RSA keys sign with RSA-PSS/SHA-256 (salt = digest length);
Ed25519 keys sign the message directly.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def load_private_key(path: str | Path):
    data = Path(path).expanduser().read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    if not isinstance(key, rsa.RSAPrivateKey | Ed25519PrivateKey):
        raise TypeError(f"unsupported key type {type(key).__name__}")
    return key


def sign(private_key, timestamp_ms: str, method: str, url_or_path: str) -> str:
    path = urlparse(url_or_path).path  # drops scheme/host and the query string
    message = f"{timestamp_ms}{method.upper()}{path}".encode()
    if isinstance(private_key, Ed25519PrivateKey):
        sig = private_key.sign(message)
    else:
        sig = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    return base64.b64encode(sig).decode()


def auth_headers(key_id: str, private_key, method: str, url: str) -> dict[str, str]:
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sign(private_key, ts, method, url),
    }
