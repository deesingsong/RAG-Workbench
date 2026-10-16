"""A minimal software WebAuthn authenticator for deterministic offline tests.

This is **not** a security tool and must never be used outside tests. It exists so
the suite can exercise the real py_webauthn verification path -- CBOR attestation
objects, COSE public keys and ECDSA signatures over the authenticator data --
without a browser, a hardware key or the network. It mirrors what a platform
authenticator would return to the JSON-serialisation layer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _cose_key(x: bytes, y: bytes) -> bytes:
    """A COSE_Key for ES256 / P-256 (RFC 8152 §13.1.1)."""

    return cbor2.dumps({1: 2, 3: -7, -1: 1, -2: x, -3: y})


class SoftAuthenticator:
    """An in-memory passkey that honours the create/get ceremonies."""

    def __init__(self, *, rp_id: str, origin: str, sign_count: int = 0) -> None:
        self.rp_id = rp_id
        self.origin = origin
        self.sign_count = sign_count
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.credential_id = os.urandom(32)

    def _client_data(self, ctype: str, challenge_b64: str) -> bytes:
        return json.dumps(
            {
                "type": ctype,
                "challenge": challenge_b64,
                "origin": self.origin,
                "crossOrigin": False,
            }
        ).encode("utf-8")

    def _cose_public_key(self) -> bytes:
        numbers = self.private_key.public_key().public_numbers()
        return _cose_key(numbers.x.to_bytes(32, "big"), numbers.y.to_bytes(32, "big"))

    def register(self, options_json: str) -> dict:
        options = json.loads(options_json)
        client_data = self._client_data("webauthn.create", options["challenge"])
        flags = 0x45  # UP (0x01) | UV (0x04) | AT (0x40)
        auth_data = (
            hashlib.sha256(self.rp_id.encode("utf-8")).digest()
            + bytes([flags])
            + self.sign_count.to_bytes(4, "big")
            + b"\x00" * 16  # AAGUID
            + len(self.credential_id).to_bytes(2, "big")
            + self.credential_id
            + self._cose_public_key()
        )
        attestation = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
        return {
            "id": b64url(self.credential_id),
            "rawId": b64url(self.credential_id),
            "type": "public-key",
            "authenticatorAttachment": "platform",
            "clientExtensionResults": {},
            "response": {
                "clientDataJSON": b64url(client_data),
                "attestationObject": b64url(attestation),
                "transports": ["internal"],
            },
        }

    def authenticate(self, options_json: str) -> dict:
        options = json.loads(options_json)
        self.sign_count += 1
        client_data = self._client_data("webauthn.get", options["challenge"])
        flags = 0x05  # UP (0x01) | UV (0x04)
        auth_data = (
            hashlib.sha256(self.rp_id.encode("utf-8")).digest()
            + bytes([flags])
            + self.sign_count.to_bytes(4, "big")
        )
        message = auth_data + hashlib.sha256(client_data).digest()
        signature = self.private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return {
            "id": b64url(self.credential_id),
            "rawId": b64url(self.credential_id),
            "type": "public-key",
            "authenticatorAttachment": "platform",
            "clientExtensionResults": {},
            "response": {
                "clientDataJSON": b64url(client_data),
                "authenticatorData": b64url(auth_data),
                "signature": b64url(signature),
                "userHandle": None,
            },
        }
