# auth.py - Система аутентификации с ролями
import streamlit as st
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import secrets

# Путь к базе данных пользователей
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
AUTH_DB = DATA_DIR / "users.db"

# Роли пользователей
ROLES = {
    "admin": {
        "name": {"ru": "Администратор", "en": "Administrator", "kk": "Әкімші"},
        "permissions": ["all"]
    },
    "manager": {
        "name": {"ru": "Менеджер производства", "en": "Production Manager", "kk": "Өндіріс менеджері"},
        "permissions": ["view_dashboard", "view_reports", "edit_data", "view_history"]
    },
    "operator": {
        "name": {"ru": "Оператор", "en": "Operator", "kk": "Оператор"},
        "permissions": ["view_dashboard", "edit_data"]
    },
    "analyst": {
        "name": {"ru": "Аналитик", "en": "Analyst", "kk": "Талдаушы"},
        "permissions": ["view_dashboard", "view_reports", "view_history"]
    }
}


def init_auth_db():
    """Инициализация базы данных пользователей"""
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    cur = conn.cursor()

    # Таблица пользователей
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL,
            email TEXT,
            created_at TEXT,
            last_login TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)

    # Таблица сессий
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_token TEXT UNIQUE,
            created_at TEXT,
            expires_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Таблица логов активности
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()

    # Создание админа по умолчанию (если нет пользователей)
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        create_default_users(conn)

    conn.close()


def create_default_users(conn):
    """Создание пользователей по умолчанию для демонстрации"""
    cur = conn.cursor()

    default_users = [
        {
            "username": "admin",
            "password": "admin123?",
            "full_name": "Администратор Системы",
            "role": "admin",
            "email": "admin@zhaya.kz"
        },
        {
            "username": "manager",
            "password": "manager123?",
            "full_name": "Асет Нурланов",
            "role": "manager",
            "email": "aset@zhaya.kz"
        },
        {
            "username": "operator",
            "password": "operator123?",
            "full_name": "Айгуль Сериковна",
            "role": "operator",
            "email": "aigul@zhaya.kz"
        },
        {
            "username": "analyst",
            "password": "analyst123?",
            "full_name": "Данияр Токаев",
            "role": "analyst",
            "email": "daniyal@zhaya.kz"
        }
    ]

    for user in default_users:
        password_hash = hash_password(user["password"])
        cur.execute("""
            INSERT INTO users (username, password_hash, full_name, role, email, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (
            user["username"],
            password_hash,
            user["full_name"],
            user["role"],
            user["email"],
            datetime.utcnow().isoformat()
        ))

    conn.commit()


def hash_password(password: str) -> str:
    """Хеширование пароля"""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate_user(username: str, password: str) -> dict:
    """Аутентификация пользователя"""
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    cur = conn.cursor()

    password_hash = hash_password(password)

    cur.execute("""
        SELECT id, username, full_name, role, email, is_active
        FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))

    user = cur.fetchone()

    if user and user[5] == 1:  # is_active
        user_id = user[0]

        # Обновление времени последнего входа
        cur.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.utcnow().isoformat(), user_id))

        # Создание сессии
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=8)

        cur.execute("""
            INSERT INTO sessions (user_id, session_token, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, session_token, datetime.utcnow().isoformat(), expires_at.isoformat()))

        # Лог активности
        log_activity(conn, user_id, "login", f"Успешный вход в систему")

        conn.commit()
        conn.close()

        return {
            "id": user[0],
            "username": user[1],
            "full_name": user[2],
            "role": user[3],
            "email": user[4],
            "session_token": session_token,
            "authenticated": True
        }

    conn.close()
    return {"authenticated": False}


def log_activity(conn, user_id: int, action: str, details: str = ""):
    """Логирование активности пользователя"""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO activity_logs (user_id, action, details, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_id, action, details, datetime.utcnow().isoformat()))
    conn.commit()


def check_permission(user_role: str, permission: str) -> bool:
    """Проверка прав доступа"""
    if user_role not in ROLES:
        return False

    permissions = ROLES[user_role]["permissions"]
    return "all" in permissions or permission in permissions


def logout_user():
    """Выход пользователя из системы"""
    if "user" in st.session_state:
        user_id = st.session_state.user.get("id")
        if user_id:
            conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
            log_activity(conn, user_id, "logout", "Выход из системы")
            conn.close()

    # Очистка сессии
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def get_all_users():
    """Получение списка всех пользователей (для администратора)"""
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, username, full_name, role, email, created_at, last_login, is_active
        FROM users
        ORDER BY created_at DESC
    """)

    users = cur.fetchall()
    conn.close()

    return users


def show_login_page(lang_choice="ru"):
    """Красивая страница входа"""

    # CSS для страницы входа
    st.markdown("""
    <style>
    .login-container {
        max-width: 450px;
        margin: 80px auto;
        padding: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .login-card {
        background: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .login-title {
        text-align: center;
        color: #667eea;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .login-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .demo-credentials {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 0.9em;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    # Логотип и заголовок
    st.markdown('<div class="login-title">🐎 Жая</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Цифровая платформа управления производством</div>', unsafe_allow_html=True)

    # Форма входа
    with st.form("login_form"):
        username = st.text_input("👤 Имя пользователя", placeholder="Введите логин")
        password = st.text_input("🔒 Пароль", type="password", placeholder="Введите пароль")

        col1, col2 = st.columns([2, 1])
        with col1:
            submit = st.form_submit_button("🚀 Войти в систему", use_container_width=True)
        with col2:
            if st.form_submit_button("ℹ️ Помощь"):
                st.session_state.show_help = True

        if submit:
            if not username or not password:
                st.error("⚠️ Заполните все поля")
            else:
                user = authenticate_user(username, password)

                if user["authenticated"]:
                    st.session_state.user = user
                    st.success(f"✅ Добро пожаловать, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error("❌ Неверный логин или пароль")

    # Демо-учетные данные
    if st.session_state.get("show_help", False):
        st.markdown("""
        <div class="demo-credentials">
            <strong>🔑 Демо-доступы для тестирования:</strong><br>
            <b>Администратор:</b> admin / admin123<br>
            <b>Менеджер:</b> manager / manager123<br>
            <b>Оператор:</b> operator / operator123<br>
            <b>Аналитик:</b> analyst / analyst123
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # Футер
    st.markdown("---")
    st.caption("© 2025 Жая Production Platform | Версия 2.0 | Разработано для цифровизации мясной промышленности")


# Инициализация при импорте
init_auth_db()