"""
UniMind API Server – Signup data is stored in MySQL (XAMPP).

Run:  python app.py
Or:   flask run --host=0.0.0.0 --port=5000

Before running:
  1. Start XAMPP and start MySQL.
  2. Create database (one-time): open http://localhost/phpmyadmin → New → database name "unimind" → Create.
  3. Or run the app once; it will try to create the database and table.

Set your MySQL password below in MYSQL_CONFIG (default XAMPP: user root, password "").
"""
import os
import json
import re
import traceback
import uuid
import random
import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, g, send_from_directory, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config, gemini_api_key_candidates

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:
    print("Install PyMySQL: pip install PyMySQL")
    raise

app = Flask(__name__)
# Allow both mobile app and website frontends (running on different ports / origins)
CORS(app, resources={r"/*": {"origins": "*"}})

_GEMINI_KEY_COUNT = len(gemini_api_key_candidates())
print(
    f"[UniMind] Gemini: {_GEMINI_KEY_COUNT} API key candidate(s). "
    "Diagnostics: GET /ai/gemini-sources and GET /ai/test-gemini"
)

# ---------- Gemini AI client (shared for app + website) ----------
# Website pages call /ai/* on this server; they never embed keys. Keys: api_server/.env

# Same REST surface as Android GeminiApi.kt — try several IDs (AI Studio availability varies by account).
_GEMINI_MODEL_NAMES = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
)

_GEMINI_KEY_HELP = (
    "Gemini API key is invalid or expired (Google rejected every key we tried). "
    "Create a new key at https://aistudio.google.com/apikey — set GEMINI_API_KEY and/or GEMINI_API_KEYS in "
    "api_server/.env (website reads .env first), save, restart python app.py. "
    "Open http://127.0.0.1:5000/web/ai-chat.html (not file://)."
)


def _is_gemini_api_key_rejection(err) -> bool:
    t = str(err).lower()
    return any(
        s in t
        for s in (
            "api_key_invalid",
            "api key expired",
            "invalid api key",
            "please renew the api key",
            "expired api key",
        )
    )


def _gemini_rest_generate_text(prompt: str) -> str:
    """Call Generative Language API via HTTPS (same as mobile app) when SDK path fails."""
    keys = gemini_api_key_candidates()
    if not keys:
        raise RuntimeError("No Gemini API key configured.")
    payload = json.dumps(
        {"contents": [{"parts": [{"text": prompt}]}]}
    ).encode("utf-8")
    last_err = None
    for api_key in keys:
        for model in _GEMINI_MODEL_NAMES:
            qkey = urllib.parse.quote(api_key, safe="")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={qkey}"
            )
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
                err = data.get("error")
                if err:
                    last_err = err.get("message", err)
                    continue
                c_list = data.get("candidates") or []
                if not c_list:
                    continue
                parts = (c_list[0].get("content") or {}).get("parts") or []
                if parts and parts[0].get("text"):
                    return str(parts[0]["text"]).strip()
            except urllib.error.HTTPError as e:
                try:
                    body = json.loads(e.read().decode())
                    last_err = (body.get("error") or {}).get("message", str(e))
                except Exception:
                    last_err = str(e)
                continue
            except Exception as e:
                last_err = e
                continue
    if _is_gemini_api_key_rejection(last_err):
        raise RuntimeError(_GEMINI_KEY_HELP)
    raise RuntimeError(str(last_err) if last_err else "REST Gemini failed")


def _with_gemini_model(func):
    """
    Run func(GenerativeModel) with the first working key × model combination.
    """
    if genai is None:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
    keys = gemini_api_key_candidates()
    if not keys:
        raise RuntimeError(
            "No Gemini API key configured. Set GEMINI_API_KEY / GEMINI_API_KEYS in "
            "api_server/.env and/or gradle.properties, then restart the server."
        )
    last_exc = None
    for api_key in keys:
        for model_name in _GEMINI_MODEL_NAMES:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                return func(model)
            except Exception as e:
                last_exc = e
                continue

    if last_exc is None:
        raise RuntimeError("All Gemini API keys failed.")

    if _is_gemini_api_key_rejection(last_exc):
        raise RuntimeError(_GEMINI_KEY_HELP) from last_exc
    raise RuntimeError(
        f"Gemini failed after trying {len(keys)} key(s). Last error: {last_exc}"
    ) from last_exc


def _gemini_chat_history_for_start_chat(history, current_prompt: str) -> list:
    """
    Build history for model.start_chat().

    The website sends `history` that already includes the latest user message.
    Gemini expects prior turns only here; the latest user text is sent via send_message().
    Trailing user turns are stripped so history does not end with user (invalid for start_chat).
    """
    turns = []
    for item in history or []:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        role_raw = (item.get("role") or "user").lower()
        if role_raw in ("assistant", "model"):
            role = "model"
        else:
            role = "user"
        turns.append({"role": role, "parts": [content]})
    while turns and turns[-1]["role"] == "user":
        turns.pop()
    return turns


def _gemini_chat_fallback_prompt(history, current_prompt: str) -> str:
    """Single-string conversation if start_chat fails (SDK / format edge cases)."""
    lines = [
        "You are UniMind, a helpful, concise assistant for a wellness and productivity app.",
        "Continue the conversation naturally.",
        "",
    ]
    turns = []
    for item in history or []:
        c = (item.get("content") or "").strip()
        if not c:
            continue
        r = (item.get("role") or "user").lower()
        tag = "User" if r == "user" else "Assistant"
        turns.append((tag, c))
    cur = (current_prompt or "").strip()
    if turns and turns[-1][0] == "User" and turns[-1][1] == cur:
        turns = turns[:-1]
    for tag, c in turns:
        lines.append(f"{tag}: {c}")
    lines.append(f"User: {cur}")
    lines.append("Assistant:")
    return "\n".join(lines)


def gemini_generate_plain_text(prompt: str) -> str:
    """Single-shot Gemini reply for website /ai/* helpers (finance, productivity, lifestyle)."""

    def go(model):
        result = model.generate_content(prompt)
        return (getattr(result, "text", None) or "").strip()

    try:
        out = _with_gemini_model(go)
        if out:
            return out
    except Exception:
        traceback.print_exc()
    return _gemini_rest_generate_text(prompt)

# Profile photo uploads (created on first upload)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static website (open in browser via HTTP — not file://)
WEBSITE_DIR = (Path(__file__).resolve().parent.parent / "website").resolve()

# Sample notification prompt templates used by the mobile app.
# These are NOT shown directly to the user from the backend, but are useful
# for debugging and for future server-side AI / summary features.
NOTIFICATION_TEMPLATES = {
    "finance": [
        "You’re close to your savings target!",
        "Spending slightly high today — adjust tomorrow.",
        "You managed money wisely this week.",
        "Dining expenses increased — review if necessary.",
        "Savings streak: 5 days in a row!",
        "Emergency fund growing steadily. Great job!",
        "You saved more than last month!",
        "Weekend spending alert — stay mindful.",
        "Budget discipline improving.",
        "Financial health score improved this week!",
    ],
    "health": [
        "Only 1,000 steps left to hit your goal!",
        "You’re more active than yesterday!",
        "Morning walk can boost today’s energy.",
        "Hydration reminder: Drink water.",
        "You burned more calories than usual!",
        "Weekend challenge: 8,000 steps today!",
        "Great improvement in activity levels.",
        "Your consistency is improving stamina.",
        "Fitness score rising steadily!",
    ],
    "productivity": [
        "Focus time! Your next task starts soon.",
        "You completed all planned tasks today!",
        "Only 2 tasks left — finish strong.",
        "Deep work session suggested now.",
        "You were 15% more productive this week.",
        "Task completion streak: 4 days!",
        "Time block reminder: Stay focused.",
        "Procrastination detected — start small.",
        "Your productivity peak time is afternoon.",
        "Review tomorrow’s plan tonight.",
    ],
    "lifestyle": [
        "Sleep less than 6 hours — prioritize rest.",
        "Stress slightly high — try breathing exercises.",
        "Great sleep quality last night!",
        "Mood improving compared to yesterday.",
        "Relaxation time recommended.",
        "Your sleep consistency is improving.",
        "High stress pattern detected this week.",
    ],
    "smart_weekly": [
        "Your finance is strong, but sleep needs attention.",
        "Health improved 20% this week — amazing!",
        "Balanced week overall. Keep consistency.",
        "Productivity rising, but stress also increased.",
        "Savings improved but spending fluctuates.",
        "You’re building strong discipline habits.",
        "Activity levels dipped mid-week.",
        "Best performance day: Wednesday!",
        "Most productive time: 3 PM – 6 PM.",
        "Overall life score improved this week.",
    ],
    "overall_premium": [
        "Your Life Score is now 82/100. Impressive.",
        "Elite consistency unlocked.",
        "You’re outperforming last week’s version of yourself.",
        "Momentum building. Stay in control.",
        "Life Balance: Stable and improving.",
        "Performance Mode activated.",
        "You’re entering a growth phase.",
        "Next milestone within reach.",
        "Progress curve trending upward.",
        "Keep stacking wins.",
    ],
    "gamified": [
        "🔥 5-Day Discipline Streak!",
        "💎 Financial Control Level Up!",
        "🚀 Productivity Tier Upgraded!",
        "🏅 Health Consistency Badge Earned!",
        "🌙 Sleep Master Achievement!",
        "⚡ Focus Warrior Unlocked!",
        "🎯 Goal Crusher Status!",
        "📈 Growth Mode Active!",
        "🏆 Balanced Life Badge!",
        "🔥 Momentum Streak Continues!",
    ],
    "assistant_tone": [
        "I’ve analyzed your week. You’re progressing steadily.",
        "Let’s optimize tomorrow for better balance.",
        "I recommend a lighter workload today.",
        "Today is ideal for financial planning.",
        "You’re building long-term discipline.",
        "Recovery mode suggested tonight.",
        "Strong financial stability detected.",
        "Health and productivity synergy improving.",
        "You’re becoming more consistent.",
        "Your future self will thank you.",
    ],
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- MySQL (XAMPP) config ----------
MYSQL_CONFIG = {
    "host": Config.MYSQL_HOST,
    "user": Config.MYSQL_USER,
    "password": Config.MYSQL_PASSWORD,
    "database": Config.MYSQL_DB,
    "charset": "utf8mb4",
    "cursorclass": DictCursor,
}


def get_mysql_connection(use_db=True):
    """Open a MySQL connection. If use_db=False, connect without database (to create it)."""
    conn = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        charset=MYSQL_CONFIG.get("charset", "utf8mb4"),
        cursorclass=DictCursor,
        autocommit=True,  # avoid long-running implicit transactions / locks
    )
    if use_db and MYSQL_CONFIG.get("database"):
        conn.select_db(MYSQL_CONFIG["database"])
    return conn


def init_db():
    """Create database and users table if they don't exist."""
    try:
        conn = get_mysql_connection(use_db=False)
        try:
            db_name = MYSQL_CONFIG.get("database", "unimind")
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            conn.commit()
            conn.select_db(db_name)
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        full_name VARCHAR(100) NOT NULL,
                        email VARCHAR(255) NOT NULL UNIQUE,
                        password VARCHAR(255) NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id)
                    )
                """)
                # Profile table: basic profile + goals text + contact details
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS profiles (
                        user_id INT UNSIGNED NOT NULL,
                        full_name VARCHAR(100) NULL,
                        age SMALLINT UNSIGNED NULL,
                        gender VARCHAR(20) NULL,
                        avatar_url VARCHAR(512) NULL,
                        email VARCHAR(255) NULL,
                        goals TEXT NULL,
                        dob DATE NULL,
                        phone VARCHAR(32) NULL,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                # Permissions table: what the user allowed on the device
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_permissions (
                        user_id INT UNSIGNED NOT NULL,
                        allow_notifications TINYINT(1) NOT NULL DEFAULT 0,
                        allow_location TINYINT(1) NOT NULL DEFAULT 0,
                        allow_calendar TINYINT(1) NOT NULL DEFAULT 0,
                        allow_health TINYINT(1) NOT NULL DEFAULT 0,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                # Domain tables: one per life domain, storing user data + AI text
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS domain_health (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        user_id INT UNSIGNED NOT NULL,
                        entry_date VARCHAR(32) NOT NULL,
                        user_data TEXT NULL,
                        ai_text TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                # Notifications table: app and domain notifications per user
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        user_id INT UNSIGNED NOT NULL,
                        domain VARCHAR(32) NOT NULL,
                        title VARCHAR(255) NOT NULL,
                        body TEXT NOT NULL,
                        is_read TINYINT(1) NOT NULL DEFAULT 0,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        KEY idx_notifications_user (user_id, created_at),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS domain_productivity (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        user_id INT UNSIGNED NOT NULL,
                        entry_date VARCHAR(32) NOT NULL,
                        user_data TEXT NULL,
                        ai_text TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS domain_finance (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        user_id INT UNSIGNED NOT NULL,
                        entry_date VARCHAR(32) NOT NULL,
                        user_data TEXT NULL,
                        ai_text TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS domain_lifestyle (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        user_id INT UNSIGNED NOT NULL,
                        entry_date VARCHAR(32) NOT NULL,
                        user_data TEXT NULL,
                        ai_text TEXT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS password_otps (
                        id INT UNSIGNED NOT NULL AUTO_INCREMENT,
                        user_id INT UNSIGNED NOT NULL,
                        email VARCHAR(255) NOT NULL,
                        otp_code VARCHAR(16) NOT NULL,
                        expires_at DATETIME NOT NULL,
                        used TINYINT(1) NOT NULL DEFAULT 0,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        KEY idx_password_otps_email (email),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    )
                """)
            conn.commit()
            # Ensure existing tables have all profile columns
            with conn.cursor() as cursor:
                for col in [
                    "ADD COLUMN full_name VARCHAR(100) NULL",
                    "ADD COLUMN age SMALLINT UNSIGNED NULL",
                    "ADD COLUMN gender VARCHAR(20) NULL",
                    "ADD COLUMN avatar_url VARCHAR(512) NULL",
                    "ADD COLUMN email VARCHAR(255) NULL",
                    "ADD COLUMN goals TEXT NULL",
                    "ADD COLUMN dob DATE NULL",
                    "ADD COLUMN phone VARCHAR(32) NULL",
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE profiles {col}")
                        conn.commit()
                    except pymysql.OperationalError:
                        pass  # column already exists
            conn.commit()
            print(f"[UniMind] MySQL database ready: {db_name} @ {MYSQL_CONFIG['host']}")
        finally:
            conn.close()
    except Exception as e:
        print(f"[UniMind] MySQL init failed (is XAMPP MySQL running?): {e}")
        traceback.print_exc()


def get_db():
    """Per-request MySQL connection (for read operations)."""
    if "db" not in g:
        g.db = get_mysql_connection()
    return g.db


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------- Signup: write to MySQL ----------

def store_signup_in_db(full_name, email, password):
    """Insert one user into MySQL. Returns (id, created_at)."""
    conn = get_mysql_connection()
    try:
        created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (full_name, email, password, created_at) VALUES (%s, %s, %s, %s)",
                (full_name, email, password, created_at),
            )
            new_id = cursor.lastrowid
        conn.commit()
        return new_id, created_at
    finally:
        conn.close()


@app.route("/signup", methods=["POST"])
def signup():
    """Store signup in MySQL (XAMPP database)."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        full_name = (data.get("full_name") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "")
    else:
        full_name = (request.form.get("full_name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "")

    if not full_name:
        return jsonify({"success": False, "message": "Full name is required"}), 400
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400
    if not password:
        return jsonify({"success": False, "message": "Password is required"}), 400
    # Strong password policy: min 8, uppercase, lowercase, number, special char.
    if not re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$", password):
        return jsonify({
            "success": False,
            "message": "Password must be at least 8 characters and include uppercase, lowercase, number, and special character",
        }), 400

    try:
        new_id, created_at = store_signup_in_db(full_name, email, password)
        print(f"[UniMind] Signup saved to MySQL: id={new_id} email={email}")
        return jsonify({
            "success": True,
            "message": "Account created",
            "user": {"id": new_id, "full_name": full_name, "email": email, "created_at": created_at}
        }), 201
    except pymysql.IntegrityError:
        return jsonify({"success": False, "message": "Email already registered"}), 409
    except Exception as e:
        traceback.print_exc()
        err_msg = str(e)
        if "Access denied" in err_msg:
            err_msg = "MySQL access denied. Check user/password in app.py MYSQL_CONFIG (XAMPP default: root, no password)."
        elif "Can't connect" in err_msg or "Connection refused" in err_msg:
            err_msg = "Cannot connect to MySQL. Start XAMPP and turn ON MySQL."
        elif "Unknown database" in err_msg:
            err_msg = "Database 'unimind' not found. Create it in phpMyAdmin or restart the server to auto-create."
        return jsonify({"success": False, "message": f"Server error: {err_msg}"}), 500


@app.route("/login", methods=["POST"])
def login():
    """Login: reads from MySQL."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = (data.get("password") or "")
    else:
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "")

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required"}), 400

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT id, full_name, password FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
    except pymysql.OperationalError as e:
        err_msg = str(e)
        if "Can't connect" in err_msg or "Connection refused" in err_msg or "10061" in err_msg:
            return jsonify({
                "success": False,
                "message": "Database unavailable. Start MySQL (e.g. open XAMPP and click Start for MySQL)."
            }), 503
        if "Access denied" in err_msg:
            return jsonify({"success": False, "message": "Database config error. Check MYSQL_CONFIG in app.py."}), 503
        raise
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Server error: {e}"}), 500

    if row is None:
        return jsonify({"success": False, "message": "Invalid email or password"}), 401
    if row["password"] != password:
        return jsonify({"success": False, "message": "Invalid email or password"}), 401

    return jsonify({
        "success": True,
        "message": "Login successful",
        "user": {"id": row["id"], "email": email, "full_name": row["full_name"]}
    }), 200


# ---------- Forgot password with OTP ----------

SMTP_CONFIG = {
    "host": Config.SMTP_HOST,
    "port": Config.SMTP_PORT,
    "user": Config.SMTP_USER,
    "password": Config.SMTP_PASSWORD,
    "sender": Config.SMTP_SENDER,
}


def generate_otp(length: int = 6) -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def send_otp_email(to_email: str, otp: str) -> bool:
    """Send OTP email; returns True on best-effort success."""
    if not SMTP_CONFIG["user"] or not SMTP_CONFIG["password"]:
        print(f"[UniMind] SMTP not configured. OTP for {to_email}: {otp}")
        return False
    try:
        msg = f"""From: {SMTP_CONFIG['sender']}
To: {to_email}
Subject: UniMind password reset OTP

Your UniMind one-time password (OTP) is: {otp}
It is valid for 15 minutes. If you did not request this, you can ignore this email.
"""
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"]) as server:
            server.starttls(context=context)
            server.login(SMTP_CONFIG["user"], SMTP_CONFIG["password"])
            server.sendmail(SMTP_CONFIG["sender"], [to_email], msg)
        print(f"[UniMind] OTP email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[UniMind] Failed to send OTP email to {to_email}: {e}")
        print(f"[UniMind] OTP for fallback is: {otp}")
        return False


@app.route("/auth/forgot/send_otp", methods=["POST"])
def forgot_send_otp():
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    if row is None:
        return jsonify({"success": False, "message": "Email not registered"}), 404
    user_id = row["id"]

    otp = generate_otp(6)
    expires_at = datetime.utcnow() + timedelta(minutes=15)

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO password_otps (user_id, email, otp_code, expires_at, used)
                VALUES (%s, %s, %s, %s, 0)
                """,
                (user_id, email, otp, expires_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
        conn.commit()
    finally:
        conn.close()

    send_otp_email(email, otp)
    return jsonify({"success": True, "message": "OTP sent to your email if it is registered."}), 200


@app.route("/auth/forgot/verify_otp", methods=["POST"])
def forgot_verify_otp():
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}
    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()
    if not email or not otp:
        return jsonify({"success": False, "message": "Email and OTP are required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    if row is None:
        return jsonify({"success": False, "message": "Email not registered"}), 404
    user_id = row["id"]

    cursor.execute(
        """
        SELECT id, otp_code, expires_at, used
        FROM password_otps
        WHERE user_id = %s AND email = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id, email),
    )
    otp_row = cursor.fetchone()
    if otp_row is None:
        return jsonify({"success": False, "message": "No OTP found. Request a new one."}), 400

    expires_at = otp_row["expires_at"]
    now = datetime.utcnow()
    if otp_row["used"]:
        return jsonify({"success": False, "message": "OTP already used. Request a new one."}), 400
    if hasattr(expires_at, "strftime"):
        expired = now > expires_at
    else:
        try:
            parsed = datetime.strptime(str(expires_at), "%Y-%m-%d %H:%M:%S")
            expired = now > parsed
        except Exception:
            expired = False
    if expired or otp_row["otp_code"] != otp:
        return jsonify({"success": False, "message": "Invalid or expired OTP"}), 400

    return jsonify({"success": True, "message": "OTP verified. Please create a new password."}), 200


@app.route("/auth/forgot/reset_password", methods=["POST"])
def forgot_reset_password():
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}
    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not email or not otp or not new_password:
        return jsonify({"success": False, "message": "Email, OTP, and new_password are required"}), 400

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    if row is None:
        return jsonify({"success": False, "message": "Email not registered"}), 404
    user_id = row["id"]

    cursor.execute(
        """
        SELECT id, otp_code, expires_at, used
        FROM password_otps
        WHERE user_id = %s AND email = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id, email),
    )
    otp_row = cursor.fetchone()
    if otp_row is None:
        return jsonify({"success": False, "message": "No OTP found. Request a new one."}), 400

    expires_at = otp_row["expires_at"]
    now = datetime.utcnow()
    if otp_row["used"]:
        return jsonify({"success": False, "message": "OTP already used. Request a new one."}), 400
    if hasattr(expires_at, "strftime"):
        expired = now > expires_at
    else:
        try:
            parsed = datetime.strptime(str(expires_at), "%Y-%m-%d %H:%M:%S")
            expired = now > parsed
        except Exception:
            expired = False
    if expired or otp_row["otp_code"] != otp:
        return jsonify({"success": False, "message": "Invalid or expired OTP"}), 400

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET password = %s WHERE id = %s",
                (new_password, user_id),
            )
            cur.execute(
                "UPDATE password_otps SET used = 1 WHERE id = %s",
                (otp_row["id"],),
            )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"success": True, "message": "Password updated successfully."}), 200


@app.route("/users", methods=["GET"])
def list_users():
    """List all users from MySQL (no passwords)."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, full_name, email, created_at FROM users ORDER BY id")
    rows = cursor.fetchall()
    # Convert datetime to string for JSON
    users_list = []
    for r in rows:
        created = r["created_at"]
        if hasattr(created, "isoformat"):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        users_list.append({
            "id": r["id"],
            "full_name": r["full_name"],
            "email": r["email"],
            "created_at": created,
        })
    return jsonify({"success": True, "users": users_list, "count": len(users_list)}), 200


def _save_domain_entry(table_name, user_id, entry_date, user_data, ai_text):
    """Generic helper to insert one domain entry row."""
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if cursor.fetchone() is None:
                return False, "User not found"
            if user_data is not None and not isinstance(user_data, dict):
                user_data = {"data": user_data}
            try:
                serialized = json.dumps(user_data) if user_data is not None else None
            except (TypeError, ValueError) as e:
                return False, "user_data not JSON-serializable: " + str(e)
            try:
                cursor.execute(
                    "INSERT INTO {} (user_id, entry_date, user_data, ai_text) VALUES (%s, %s, %s, %s)".format(table_name),
                    (user_id, entry_date, serialized, ai_text or None),
                )
            except Exception as e:
                return False, "Database error: " + str(e)
        conn.commit()
        return True, None
    except Exception as e:
        return False, "Save failed: " + str(e)
    finally:
        conn.close()


# ---------- Profile setup ----------
# DB table 'profiles' = Profile Setup screen only:
#   Full Name → full_name, Age → age, Gender → gender, Profile picture → avatar_url

def _parse_profile_data(data):
    """Extract profile fields: full_name, age, gender, avatar_url, dob, phone."""
    full_name = (data.get("full_name") or "").strip() or None
    age_val = data.get("age")
    if age_val is not None and age_val != "":
        try:
            age_val = int(age_val)
            if age_val < 0 or age_val > 150:
                age_val = None
        except (TypeError, ValueError):
            age_val = None
    else:
        age_val = None
    dob_val = (data.get("dob") or "").strip() or None
    phone_val = (data.get("phone") or "").strip() or None
    return {
        "full_name": full_name,
        "age": age_val,
        "gender": (data.get("gender") or "").strip() or None,
        "avatar_url": (data.get("avatar_url") or "").strip() or None,
        "dob": dob_val,
        "phone": phone_val,
    }


def _profile_row_to_json(row):
    """Profile row to JSON for API response."""
    if not row:
        return None
    out = {
        "user_id": row["user_id"],
        "full_name": row.get("full_name"),
        "email": row.get("email"),
        "age": row.get("age"),
        "gender": row.get("gender"),
        "avatar_url": row.get("avatar_url"),
        "goals": row.get("goals"),
        "dob": row.get("dob"),
        "phone": row.get("phone"),
    }
    updated = row.get("updated_at")
    out["updated_at"] = updated.strftime("%Y-%m-%d %H:%M:%S") if updated and hasattr(updated, "strftime") else str(updated)
    return out


def _parse_bool(data, *keys):
    """Parse a boolean-like field from JSON/form data (supports true/false, 1/0, 'yes'/'no', 'on'/'off')."""
    for key in keys:
        if key in data:
            val = data.get(key)
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            s = str(val).strip().lower()
            return s in ("1", "true", "yes", "on")
    return False


def _notification_row_to_json(row):
    if not row:
        return None
    created = row.get("created_at")
    created_str = created.strftime("%Y-%m-%d %H:%M:%S") if hasattr(created, "strftime") else str(created)
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "domain": row.get("domain"),
        "title": row.get("title"),
        "body": row.get("body"),
        "is_read": bool(row.get("is_read", 0)),
        "created_at": created_str,
    }


@app.route("/profile", methods=["POST", "PUT", "PATCH"])
def profile_save():
    """Create or update profile.

    Body (JSON or form):
      user_id (required, int),
      full_name (optional),
      email (optional),
      age (optional, int),
      gender (optional),
      avatar_url (optional),
      dob (optional, YYYY-MM-DD),
      phone (optional)
    """
    # Accept JSON or form; also be forgiving if front end forgets Content-Type
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}
        # If no form fields but raw body exists, try JSON-parse it
        if not data and request.data:
            try:
                data = json.loads(request.data.decode("utf-8"))
            except Exception:
                data = {}

    # Support multiple possible key names from the app: user_id / id / userId
    user_id = (
        data.get("user_id")
        or data.get("id")
        or data.get("userId")
        or request.args.get("user_id")
    )
    if user_id is None:
        return jsonify({"success": False, "message": "user_id is required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400

    profile = _parse_profile_data(data)
    email = (data.get("email") or "").strip().lower() or None
    # Retry once on MySQL lock timeout / deadlock to avoid "network error" in the app
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            # First, handle common connection problems (MySQL not running, bad password, etc.)
            try:
                conn = get_mysql_connection()
            except pymysql.OperationalError as e:
                msg = str(e)
                # 2003 / connection refused / 10061 → MySQL server not running or wrong host/port
                if "2003" in msg or "Can't connect" in msg or "Connection refused" in msg or "10061" in msg:
                    return jsonify({
                        "success": False,
                        "message": "Cannot connect to MySQL. Start XAMPP and turn ON MySQL (or check MYSQL_CONFIG in app.py)."
                    }), 503
                # Access denied → bad user/password
                if "Access denied" in msg:
                    return jsonify({
                        "success": False,
                        "message": "MySQL access denied. Check MYSQL_CONFIG user/password in app.py."
                    }), 503
                raise

            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
                if cursor.fetchone() is None:
                    return jsonify({"success": False, "message": "User not found"}), 404
                # Optionally update users table (name/email)
                update_fields = []
                params = []
                if profile["full_name"]:
                    update_fields.append("full_name = %s")
                    params.append(profile["full_name"])
                if email:
                    update_fields.append("email = %s")
                    params.append(email)
                if update_fields:
                    params.append(user_id)
                    cursor.execute(
                        f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s",
                        tuple(params),
                    )
                cursor.execute(
                    """
                    INSERT INTO profiles (user_id, full_name, age, gender, avatar_url, email, dob, phone)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        full_name = COALESCE(VALUES(full_name), full_name),
                        age = COALESCE(VALUES(age), age),
                        gender = COALESCE(VALUES(gender), gender),
                        avatar_url = COALESCE(VALUES(avatar_url), avatar_url),
                        email = COALESCE(VALUES(email), email),
                        dob = COALESCE(VALUES(dob), dob),
                        phone = COALESCE(VALUES(phone), phone)
                    """,
                    (
                        user_id,
                        profile["full_name"],
                        profile["age"],
                        profile["gender"],
                        profile["avatar_url"],
                        email,
                        profile["dob"],
                        profile["phone"],
                    ),
                )
            conn.commit()
            print(f"[UniMind] Profile saved for user_id={user_id}")
            # Return current profile
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        u.id AS user_id,
                        COALESCE(p.full_name, u.full_name) AS full_name,
                        COALESCE(p.email, u.email) AS email,
                        p.age,
                        p.gender,
                        p.avatar_url,
                        p.goals,
                        p.dob,
                        p.phone,
                        p.updated_at
                    FROM users u
                    LEFT JOIN profiles p ON p.user_id = u.id
                    WHERE u.id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
            return jsonify({
                "success": True,
                "message": "Profile saved",
                "profile": _profile_row_to_json(row),
            }), 200
        except pymysql.OperationalError as e:
            # 1205 = lock wait timeout, 1213 = deadlock
            code = e.args[0] if e.args else None
            if code in (1205, 1213) and attempt < max_attempts:
                try:
                    if conn is not None:
                        conn.rollback()
                except Exception:
                    pass
                continue
            # If still failing after retry, return a clean error to the client
            if code in (1205, 1213):
                return jsonify({
                    "success": False,
                    "message": "Database is busy, please try saving your profile again."
                }), 503
            raise
        finally:
            if conn is not None:
                conn.close()


@app.route("/notifications", methods=["POST"])
def notifications_create():
    """
    Create one notification row for a user.

    Mobile app chooses the exact text based on the
    user's live data (steps, tasks, savings, lifestyle).
    Example prompts live in NOTIFICATION_TEMPLATES above.

    Expected JSON body:
      user_id (int, required),
      domain (string, e.g. 'health', 'productivity', 'finance', 'lifestyle', 'system'),
      title (string),
      body (string),
      is_read (optional, bool).
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    domain = (data.get("domain") or "").strip() or "system"
    title = (data.get("title") or "").strip()
    body = (data.get("body") or "").strip()
    if user_id is None:
        return jsonify({"success": False, "message": "user_id is required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400
    if not title or not body:
        return jsonify({"success": False, "message": "title and body are required"}), 400
    is_read = 1 if _parse_bool(data, "is_read") else 0

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if cursor.fetchone() is None:
                return jsonify({"success": False, "message": "User not found"}), 404
            cursor.execute(
                """
                INSERT INTO notifications (user_id, domain, title, body, is_read)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (user_id, domain[:32], title[:255], body, is_read),
            )
            new_id = cursor.lastrowid
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, domain, title, body, is_read, created_at FROM notifications WHERE id = %s",
                (new_id,),
            )
            row = cur.fetchone()
        return jsonify({"success": True, "notification": _notification_row_to_json(row)}), 201
    finally:
        conn.close()


@app.route("/notifications/<int:user_id>", methods=["GET"])
def notifications_list(user_id):
    """
    List UNREAD notifications for a user (newest first).
    """
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if cursor.fetchone() is None:
        return jsonify({"success": False, "message": "User not found"}), 404
    cursor.execute(
        """
        SELECT id, user_id, domain, title, body, is_read, created_at
        FROM notifications
        WHERE user_id = %s AND is_read = 0
        ORDER BY created_at DESC, id DESC
        """,
        (user_id,),
    )
    rows = cursor.fetchall()
    return jsonify({
        "success": True,
        "notifications": [_notification_row_to_json(r) for r in rows],
        "count": len(rows),
    }), 200


@app.route("/notifications/mark_read", methods=["POST"])
def notifications_mark_read():
    """
    Mark one or all notifications as read for a user.
    Body JSON:
      user_id (required, int),
      notification_id (optional, int),
      all (optional, bool; if true marks all for that user).
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}

    user_id = data.get("user_id")
    if user_id is None:
        return jsonify({"success": False, "message": "user_id is required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400

    mark_all = _parse_bool(data, "all")
    notif_id = data.get("notification_id")

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if cursor.fetchone() is None:
        return jsonify({"success": False, "message": "User not found"}), 404

    if mark_all:
        cursor.execute(
            "UPDATE notifications SET is_read = 1 WHERE user_id = %s AND is_read = 0",
            (user_id,),
        )
        db.commit()
        return jsonify({"success": True, "updated": cursor.rowcount}), 200

    if notif_id is None:
        return jsonify({"success": False, "message": "notification_id or all=true required"}), 400
    try:
        notif_id = int(notif_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "notification_id must be an integer"}), 400

    cursor.execute(
        "UPDATE notifications SET is_read = 1 WHERE id = %s AND user_id = %s",
        (notif_id, user_id),
    )
    db.commit()
    return jsonify({"success": True, "updated": cursor.rowcount}), 200


@app.route("/permissions", methods=["POST"])
def save_permissions():
    """
    Save app permission choices from the Grant Permissions screen.
    Body (JSON or form):
      user_id (required, int),
      allow_notifications (bool),
      allow_location (bool),
      allow_calendar (bool),
      allow_health (bool)
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}

    user_id = data.get("user_id")
    if user_id is None:
        return jsonify({"success": False, "message": "user_id is required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400

    allow_notifications = 1 if _parse_bool(data, "allow_notifications", "notifications") else 0
    allow_location = 1 if _parse_bool(data, "allow_location", "location") else 0
    allow_calendar = 1 if _parse_bool(data, "allow_calendar", "calendar") else 0
    allow_health = 1 if _parse_bool(data, "allow_health", "health") else 0

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if cursor.fetchone() is None:
                return jsonify({"success": False, "message": "User not found"}), 404
            cursor.execute(
                """
                INSERT INTO user_permissions (user_id, allow_notifications, allow_location, allow_calendar, allow_health)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    allow_notifications = VALUES(allow_notifications),
                    allow_location = VALUES(allow_location),
                    allow_calendar = VALUES(allow_calendar),
                    allow_health = VALUES(allow_health)
                """,
                (user_id, allow_notifications, allow_location, allow_calendar, allow_health),
            )
        conn.commit()
        print(f"[UniMind] Permissions saved for user_id={user_id} "
              f"(notifications={allow_notifications}, location={allow_location}, "
              f"calendar={allow_calendar}, health={allow_health})")
        return jsonify({"success": True, "message": "Permissions saved"}), 200
    finally:
        conn.close()


@app.route("/domain/health", methods=["POST"])
def domain_health_save():
    """
    Save one Health domain snapshot.
    Expected JSON body:
      user_id (int), entry_date (string, e.g. '2026-03-02'),
      user_data (object): steps_today, steps_goal, calories_burned, distance_km,
        goal_percent, last_7_days (array), active_days_this_week (optional).
      ai_text (string, optional): e.g. AI recommendation from Get recommendations.
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    entry_date = (data.get("entry_date") or "").strip()
    if user_id is None or not entry_date:
        return jsonify({"success": False, "message": "user_id and entry_date are required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400
    user_data = data.get("user_data")
    if user_data is not None and not isinstance(user_data, dict):
        user_data = {"payload": user_data}
    ok, err = _save_domain_entry(
        "domain_health",
        user_id,
        entry_date,
        user_data,
        data.get("ai_text"),
    )
    if not ok:
        code = 404 if err == "User not found" else 500
        return jsonify({"success": False, "message": err or "Save failed"}), code
    return jsonify({"success": True, "message": "Health domain entry saved"}), 201


@app.route("/domain/productivity", methods=["POST"])
def domain_productivity_save():
    """
    Save one Productivity domain snapshot.
    Expected JSON body:
      user_id (int), entry_date (string),
      user_data (object): e.g.
        {
          "tasks": [...],
          "completed_today": int,
          "total_tasks_today": int,
          "focus_minutes_today": int,
          "focus_minutes_goal": int,
          "upcoming_titles": [...],
          "distractions_today": int,
          "productivity_score_today": int,
          "timer": {
            "is_running": bool,
            "mode": "focus" | "break",
            "started_at": "YYYY-MM-DDTHH:MM:SSZ",
            "elapsed_seconds": int,
            "total_seconds": int
          }
        }
      ai_text (string, optional).
    """
    # Accept JSON or form; also be forgiving if front end forgets Content-Type
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}
        # If no form fields but raw body exists, try JSON-parse it
        if not data and request.data:
            try:
                data = json.loads(request.data.decode("utf-8"))
            except Exception:
                data = {}

    # Support multiple possible key names from the app
    user_id = (
        data.get("user_id")
        or data.get("id")
        or data.get("userId")
        or request.args.get("user_id")
    )
    entry_date = (
        (data.get("entry_date")
         or data.get("date")
         or data.get("entryDate")
         or "")
    ).strip()

    if user_id is None or not entry_date:
        return jsonify({"success": False, "message": "user_id and entry_date are required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400

    # Normalize user_data so additional fields like timer state
    # (for the notification bar focus/break timer) are safely stored as JSON.
    user_data = (
        data.get("user_data")
        or data.get("userData")
        or data.get("data")
    )
    if user_data is not None and not isinstance(user_data, dict):
        user_data = {"payload": user_data}

    ok, err = _save_domain_entry(
        "domain_productivity",
        user_id,
        entry_date,
        user_data,
        data.get("ai_text"),
    )
    if not ok:
        code = 404 if err == "User not found" else 500
        return jsonify({"success": False, "message": err or "Save failed"}), code
    print(f"[UniMind] Productivity entry saved for user_id={user_id} entry_date={entry_date}")
    return jsonify({"success": True, "message": "Productivity domain entry saved"}), 201


@app.route("/domain/finance", methods=["POST"])
def domain_finance_save():
    """
    Save one Finance domain snapshot.
    Expected JSON body:
      user_id (int), entry_date (string),
      user_data (object: salary_monthly, expenses list, totals, etc),
      ai_text (string, optional).
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    entry_date = (data.get("entry_date") or "").strip()
    if user_id is None or not entry_date:
        return jsonify({"success": False, "message": "user_id and entry_date are required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400
    # Normalize user_data to dict (backend expects object)
    user_data = data.get("user_data")
    if user_data is not None and not isinstance(user_data, dict):
        user_data = {"payload": user_data}
    ok, err = _save_domain_entry(
        "domain_finance",
        user_id,
        entry_date,
        user_data,
        data.get("ai_text"),
    )
    if not ok:
        code = 404 if err == "User not found" else 500
        return jsonify({"success": False, "message": err or "Save failed"}), code
    return jsonify({"success": True, "message": "Finance domain entry saved"}), 201


@app.route("/domain/lifestyle", methods=["POST"])
def domain_lifestyle_save():
    """
    Save one Lifestyle domain snapshot.
    Expected JSON body:
      user_id (int), entry_date (string),
      user_data (object: sleep_hours, stress_level, history, etc),
      ai_text (string, optional).
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    entry_date = (data.get("entry_date") or "").strip()
    if user_id is None or not entry_date:
        return jsonify({"success": False, "message": "user_id and entry_date are required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400
    ok, err = _save_domain_entry(
        "domain_lifestyle",
        user_id,
        entry_date,
        data.get("user_data"),
        data.get("ai_text"),
    )
    if not ok:
        return jsonify({"success": False, "message": err}), 404
    return jsonify({"success": True, "message": "Lifestyle domain entry saved"}), 201


@app.route("/profile/<int:user_id>", methods=["GET"])
def profile_get(user_id):
    """Get profile for a user, including email and contact fields."""
    db = get_db()
    cursor = db.cursor()
    # Ensure the user exists
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if cursor.fetchone() is None:
        return jsonify({"success": False, "message": "User not found"}), 404

    cursor.execute(
        """
        SELECT
            u.id AS user_id,
            COALESCE(p.full_name, u.full_name) AS full_name,
            u.email,
            p.age,
            p.gender,
            p.avatar_url,
            p.goals,
            p.dob,
            p.phone,
            p.updated_at
        FROM users u
        LEFT JOIN profiles p ON p.user_id = u.id
        WHERE u.id = %s
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    if row is None:
        # No profile row yet, but user exists
        return jsonify({"success": True, "profile": None, "message": "No profile yet"}), 200
    return jsonify({"success": True, "profile": _profile_row_to_json(row)}), 200


@app.route("/profile/goals", methods=["POST"])
def profile_goals():
    """Update only the goals field for a user (called from Goals screen)."""
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form or {}

    user_id = data.get("user_id")
    goals = (data.get("goals") or "").strip()
    if user_id is None:
        return jsonify({"success": False, "message": "user_id is required"}), 400
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "user_id must be an integer"}), 400

    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if cursor.fetchone() is None:
                return jsonify({"success": False, "message": "User not found"}), 404
            cursor.execute(
                """
                INSERT INTO profiles (user_id, goals)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE
                    goals = VALUES(goals)
                """,
                (user_id, goals or None),
            )
        conn.commit()
        return jsonify({"success": True, "message": "Goals saved"}), 200
    finally:
        conn.close()


@app.route("/profile/photo", methods=["POST"])
def profile_photo_upload():
    """Upload profile picture. Multipart form: 'photo' or 'file'. Returns avatar_url (full URL)."""
    f = request.files.get("photo") or request.files.get("file")
    if not f or f.filename == "":
        return jsonify({"success": False, "message": "No file provided"}), 400
    if not allowed_file(f.filename):
        return jsonify({"success": False, "message": "Allowed: png, jpg, jpeg, gif, webp"}), 400
    ext = f.filename.rsplit(".", 1)[1].lower()
    name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, name)
    try:
        f.save(path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500
    base = request.url_root.rstrip("/")
    avatar_url = f"{base}/uploads/{name}"
    return jsonify({"success": True, "avatar_url": avatar_url}), 200


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_upload(filename):
    """Serve uploaded profile photos."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/web", methods=["GET"])
def website_entry_redirect():
    """Use HTTP URLs so the browser can call /ai/* APIs (file:// often breaks fetch)."""
    return redirect("/web/index.html", code=302)


@app.route("/web/<path:filename>", methods=["GET"])
def serve_website(filename):
    """Serve files from project /website, e.g. http://127.0.0.1:5000/web/ai-chat.html"""
    candidate = (WEBSITE_DIR / filename).resolve()
    try:
        candidate.relative_to(WEBSITE_DIR)
    except ValueError:
        return jsonify({"success": False, "message": "Invalid path"}), 403
    if not candidate.is_file():
        return jsonify({"success": False, "message": "Not found"}), 404
    return send_from_directory(str(WEBSITE_DIR), filename)


@app.route("/health", methods=["GET", "POST"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/ai/gemini-sources", methods=["GET"])
def ai_gemini_sources():
    """How many keys each source contributes (no secrets). Use to verify website picks up your edits."""
    from config import (
        gemini_api_key_candidates,
        _env_gemini_key_list,
        _gradle_gemini_key_list,
        _local_properties_gemini_list,
        _root_gemini_keys_file_list,
    )
    return jsonify({
        "success": True,
        "sources": {
            "api_server_dot_env": len(_env_gemini_key_list()),
            "local_properties": len(_local_properties_gemini_list()),
            "gemini_keys_properties": len(_root_gemini_keys_file_list()),
            "gradle_properties": len(_gradle_gemini_key_list()),
        },
        "merged_unique_key_count": len(gemini_api_key_candidates()),
        "order": "env → local.properties → gemini_keys.properties → gradle.properties",
    }), 200


@app.route("/ai/test-gemini", methods=["GET"])
def ai_test_gemini():
    """Quick check that at least one key × model can generate text (no keys returned)."""
    try:
        reply = gemini_generate_plain_text('Reply with one word only: "pong".')
        text = (reply or "").strip().lower()
        return jsonify({
            "success": True,
            "ok": bool(text),
            "reply_preview": (reply or "")[:120],
            "hint": "If ok is false, keys may be invalid or Generative Language API not enabled for the project.",
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "ok": False,
            "message": str(e),
        }), 200


@app.route("/ai/chat", methods=["POST"])
def ai_chat():
    """
    Simple chat endpoint used by both the mobile app and the website chatbot.

    Request JSON:
      {
        "prompt": "user message here",
        "history": [ {"role": "user"|"model", "content": "..."}, ... ]  # optional
      }
    """
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    history = data.get("history") or []
    if not prompt:
        return jsonify({"success": False, "message": "prompt is required"}), 400

    try:
        def go(model):
            hist = _gemini_chat_history_for_start_chat(history, prompt)
            try:
                chat_session = model.start_chat(history=hist)
                result = chat_session.send_message(prompt)
                reply = getattr(result, "text", None) or ""
            except Exception:
                traceback.print_exc()
                fb = _gemini_chat_fallback_prompt(history, prompt)
                result = model.generate_content(fb)
                reply = getattr(result, "text", None) or ""
            return (reply or "").strip()

        try:
            reply = _with_gemini_model(go)
        except Exception:
            traceback.print_exc()
            fb = _gemini_chat_fallback_prompt(history, prompt)
            reply = _gemini_rest_generate_text(fb)
        if not reply:
            reply = "I'm here to help. Could you say that another way?"
        return jsonify({
            "success": True,
            "reply": reply,
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": f"AI error: {e}"}), 500


@app.route("/ai/finance_suggestions", methods=["POST"])
def ai_finance_suggestions():
    """Website finance.html — matches website/api.js unimindAiFinanceInsight."""
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    try:
        monthly = float(data.get("monthly_salary") or 0)
        today = float(data.get("total_spent_today") or 0)
        month = float(data.get("total_spent_month") or 0)
    except (TypeError, ValueError):
        monthly, today, month = 0.0, 0.0, 0.0
    prompt = f"""You are a friendly financial coach for the UniMind app. The user's monthly salary is {monthly}. Today they spent {today}. So far this month they have spent {month}.
Give 2-3 short, actionable suggestions (1 sentence each): budgeting tip, saving tip, or spending awareness. Be specific. Plain text, no bullets or numbers."""
    try:
        reply = gemini_generate_plain_text(prompt)
        if not reply:
            reply = "Track your daily expenses to stay within your budget."
        return jsonify({"success": True, "reply": reply}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": True,
            "reply": f"Tip: Set a weekly spending limit and review expenses each evening. (AI unavailable: {e})",
        }), 200


@app.route("/ai/productivity_suggestions", methods=["POST"])
def ai_productivity_suggestions():
    """Website productivity.html — matches website/api.js unimindAiProductivityInsight."""
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    total = int(data.get("total_tasks") or 0)
    completed = int(data.get("completed_today") or 0)
    upcoming = data.get("upcoming_titles") or []
    if not isinstance(upcoming, list):
        upcoming = []
    up_text = "No upcoming tasks." if not upcoming else "Upcoming: " + ", ".join(str(x) for x in upcoming[:5]) + "."
    prompt = f"""You are a productivity coach for the UniMind app. User has {total} tasks total, completed {completed} today. {up_text}
Give one short, practical tip to stay focused or prioritize better. Plain text only."""
    try:
        reply = gemini_generate_plain_text(prompt)
        if not reply:
            reply = "Tackle the most important task first."
        return jsonify({"success": True, "reply": reply}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": True,
            "reply": f"Try time-blocking your top task for 25 minutes. (AI unavailable: {e})",
        }), 200


@app.route("/ai/lifestyle_suggestions", methods=["POST"])
def ai_lifestyle_suggestions():
    """Website lifestyle.html — matches website/api.js unimindAiLifestyleInsight."""
    if not request.is_json:
        return jsonify({"success": False, "message": "JSON body required"}), 400
    data = request.get_json(silent=True) or {}
    try:
        sleep = float(data.get("sleep_hours") or 0)
        stress = int(data.get("stress_level") or 5)
    except (TypeError, ValueError):
        sleep, stress = 0.0, 5
    stress = max(1, min(10, stress))
    prompt = f"""You are a wellness coach for the UniMind app. User slept {sleep} hours last night. Self-reported stress level (1-10): {stress}.
Give one short, kind suggestion to improve sleep or reduce stress. Plain text only."""
    try:
        reply = gemini_generate_plain_text(prompt)
        if not reply:
            reply = "Aim for 7-8 hours of sleep and short breaks during the day."
        return jsonify({"success": True, "reply": reply}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": True,
            "reply": f"Try a 10-minute wind-down before bed and limit screens. (AI unavailable: {e})",
        }), 200


@app.route("/test-db", methods=["GET"])
def test_db():
    """Check if MySQL is reachable and unimind database exists. Use for debugging."""
    try:
        conn = get_mysql_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn.close()
        return jsonify({"success": True, "message": "MySQL connection OK"}), 200
    except Exception as e:
        traceback.print_exc()
        msg = str(e)
        if "Access denied" in msg:
            msg = "MySQL access denied. Set correct password in app.py MYSQL_CONFIG."
        elif "Can't connect" in msg or "Connection refused" in msg:
            msg = "MySQL not running. Start XAMPP and start MySQL."
        elif "Unknown database" in msg:
            msg = "Database 'unimind' missing. Create it in phpMyAdmin or let the app create it on startup."
        return jsonify({"success": False, "message": msg}), 500


# Create database and table on startup (server still starts if MySQL is down)
init_db()

if __name__ == "__main__":
    print(f"[UniMind] API + website: http://127.0.0.1:5000  |  AI chat: http://127.0.0.1:5000/web/ai-chat.html")
    app.run(host="0.0.0.0", port=5000, debug=True)