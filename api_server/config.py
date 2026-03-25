import os
from pathlib import Path

# Load .env from api_server directory or project root
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path, override=True)
else:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    except Exception:
        pass


def _norm_gemini_key(raw: str) -> str:
    """Strip BOM/spaces; keys must not be modified beyond that."""
    if not raw:
        return ""
    return str(raw).replace("\ufeff", "").strip()


def _parse_gemini_props_file(path: Path) -> list:
    """Read GEMINI_API_KEYS (comma-separated) and GEMINI_API_KEY from a .properties file."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    props = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        props[k.strip()] = v.strip()
    out = []
    seen = set()
    primary = _norm_gemini_key(props.get("GEMINI_API_KEY") or "")
    if primary:
        seen.add(primary)
        out.append(primary)
    raw = (props.get("GEMINI_API_KEYS") or "").strip()
    for part in raw.split(",") if raw else []:
        kk = _norm_gemini_key(part)
        if kk and kk not in seen:
            seen.add(kk)
            out.append(kk)
    if out:
        return out
    return [primary] if primary else []


def _root_gemini_keys_file_list():
    """Optional project root gemini_keys.properties — same source as Android when configured."""
    root = Path(__file__).resolve().parent.parent
    return _parse_gemini_props_file(root / "gemini_keys.properties")


def _gradle_gemini_key_list():
    """gradle.properties GEMINI_* (fallback if gemini_keys.properties missing)."""
    gradle_file = Path(__file__).resolve().parent.parent / "gradle.properties"
    return _parse_gemini_props_file(gradle_file)


def _local_properties_gemini_list():
    """Android local.properties — optional GEMINI_API_KEY / GEMINI_API_KEYS (same format)."""
    lp = Path(__file__).resolve().parent.parent / "local.properties"
    return _parse_gemini_props_file(lp)


def _env_gemini_key_list():
    """Env: primary key first, then comma list, then GOOGLE_API_KEY — all merged, deduped."""
    seen = set()
    out = []

    def add(k: str):
        kk = _norm_gemini_key(k)
        if kk and kk not in seen:
            seen.add(kk)
            out.append(kk)

    for name in ("GEMINI_API_KEY", "GEMINI_AI_API_KEY", "GOOGLE_API_KEY"):
        add(os.environ.get(name) or "")
    raw = (os.environ.get("GEMINI_API_KEYS") or "").strip()
    for part in raw.split(",") if raw else []:
        add(part)
    return out


def gemini_api_key_candidates():
    """
    Keys for the Python Gemini client (website /ai/* routes).

    Order (first wins for new/updated keys — edit api_server/.env for website quick fixes):
      1) Environment (.env loaded at startup)
      2) local.properties (GEMINI_* — Android Studio–friendly)
      3) gemini_keys.properties (optional root file)
      4) gradle.properties
    Deduped while preserving order.
    """
    seen = set()
    out = []
    for k in (
        _env_gemini_key_list()
        + _local_properties_gemini_list()
        + _root_gemini_keys_file_list()
        + _gradle_gemini_key_list()
    ):
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class Config:
    # Secret key for session security
    SECRET_KEY = os.environ.get("SECRET_KEY") or "supersecretkey"

    # -----------------------------
    # Gemini AI (Google)
    # -----------------------------
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_AI_API_KEY") or ""

    # -----------------------------
    # MySQL Configuration
    # -----------------------------
    MYSQL_HOST = os.environ.get("MYSQL_HOST") or "localhost"
    MYSQL_USER = os.environ.get("MYSQL_USER") or "root"
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD") or ""
    MYSQL_DB = os.environ.get("MYSQL_DB") or "unimind"
    MYSQL_CURSORCLASS = "DictCursor"

    # -----------------------------
    # SMTP / Email for OTP and notifications
    # -----------------------------
    SMTP_HOST = os.environ.get("SMTP_HOST") or "smtp.gmail.com"
    SMTP_PORT = int(os.environ.get("SMTP_PORT") or 587)
    SMTP_USER = os.environ.get("SMTP_USER") or "unimind.fphl@gmail.com"
    # IMPORTANT: this must be your 16‑character Gmail app password with NO spaces
    SMTP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD") or "jylklxhsjsbndmcp"
    SMTP_SENDER = os.environ.get("UNIMIND_SMTP_SENDER") or (
        f"UniMind <{SMTP_USER}>" if SMTP_USER else "UniMind <no-reply@unimind.local>"
    )

    # Optional (recommended for production later)
    DEBUG = True
