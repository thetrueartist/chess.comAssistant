#!/usr/bin/env python3
"""Sign a chessAssistant.py release with your OFFLINE Ed25519 signing key.

Produces `chessAssistant.py.sig` (base64) to commit alongside the code, so the
app's auto-update checker can verify the release really came from you.

Usage:
    python3 sign_release.py [path/to/chessAssistant.py]

Requirements:
    - cryptography  (already a dependency of the app)
    - Your private key at:  ~/.config/chessassistant/release_signing_key.pem
      Keep it SECRET and BACKED UP. It never goes in the repo.

Workflow for a release:
    1) Bump __version__ in chessAssistant.py
    2) python3 sign_release.py          # writes chessAssistant.py.sig
    3) git add chessAssistant.py chessAssistant.py.sig && git commit && git push
"""
import os
import sys
import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

KEY_PATH = os.path.expanduser("~/.config/chessassistant/release_signing_key.pem")


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "chessAssistant.py")

    if not os.path.exists(KEY_PATH):
        sys.exit(f"Signing key not found at {KEY_PATH}\n"
                 f"Generate one, then update RELEASE_SIGNING_PUBKEY_B64 in chessAssistant.py.")
    if not os.path.exists(target):
        sys.exit(f"Target file not found: {target}")

    with open(KEY_PATH, "rb") as f:
        priv = serialization.load_pem_private_key(f.read(), password=None)
    with open(target, "rb") as f:
        data = f.read()

    sig = priv.sign(data)

    # Self-check before writing (fail loudly rather than ship a bad signature)
    pub = priv.public_key()
    pub.verify(sig, data)

    out = target + ".sig"
    with open(out, "w") as f:
        f.write(base64.b64encode(sig).decode() + "\n")

    pub_b64 = base64.b64encode(
        pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).decode()
    print(f"Signed {os.path.basename(target)}  ->  {os.path.basename(out)}")
    print(f"Public key (must match RELEASE_SIGNING_PUBKEY_B64 in the app):\n  {pub_b64}")


if __name__ == "__main__":
    main()
