import sqlite3
import hashlib

DB_NAME = "trafficai_users.db"

# ==============================
# DATABASE INIT
# ==============================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# ==============================
# PASSWORD HASHING
# ==============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ==============================
# CREATE USER
# ==============================
def create_user(username, password):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        hashed = hash_password(password)

        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hashed))

        conn.commit()
        conn.close()
        return True
    except:
        return False

# ==============================
# VERIFY USER
# ==============================
def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    hashed = hash_password(password)

    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hashed))

    user = c.fetchone()
    conn.close()

    return user
