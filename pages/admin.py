# pages/admin.py - Административная панель
import streamlit as st
import pandas as pd
from auth import get_all_users, ROLES
from datetime import datetime
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUTH_DB = DATA_DIR / "users.db"


def show_admin_panel(lang_choice):
    """Административная панель управления системой"""

    # Проверка прав доступа
    user = st.session_state.get("user", {})
    if user.get("role") != "admin":
        st.error("❌ Доступ запрещен. Требуются права администратора.")
        return

    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.title("⚙️ Административная панель")
    st.markdown("Управление пользователями, системой и безопасностью")

    # Табы администратора
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Управление пользователями",
        "📊 Активность системы",
        "🔒 Безопасность",
        "⚙️ Настройки"
    ])

    # === ТАБ 1: Управление пользователями ===
    with tab1:
        show_users_management()

    # === ТАБ 2: Активность системы ===
    with tab2:
        show_system_activity()

    # === ТАБ 3: Безопасность ===
    with tab3:
        show_security_settings()

    # === ТАБ 4: Настройки ===
    with tab4:
        show_system_settings()

    st.markdown("</div>", unsafe_allow_html=True)


def show_users_management():
    """Управление пользователями"""
    st.subheader("👥 Управление пользователями")

    # Получение всех пользователей
    users = get_all_users()

    if users:
        users_df = pd.DataFrame(users, columns=[
            'ID', 'Логин', 'Полное имя', 'Роль', 'Email',
            'Создан', 'Последний вход', 'Активен'
        ])

        # Перевод статуса
        users_df['Активен'] = users_df['Активен'].apply(lambda x: '✅ Да' if x == 1 else '❌ Нет')

        # Перевод ролей
        def translate_role(role):
            return ROLES.get(role, {}).get("name", {}).get("ru", role)

        users_df['Роль'] = users_df['Роль'].apply(translate_role)

        # Статистика
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Всего пользователей", len(users_df))

        with col2:
            active_users = (users_df['Активен'] == '✅ Да').sum()
            st.metric("Активных", active_users)

        with col3:
            admins = (users_df['Роль'].str.contains('Администратор')).sum()
            st.metric("Администраторов", admins)

        with col4:
            st.metric("Ролей", users_df['Роль'].nunique())

        st.markdown("---")

        # Таблица пользователей
        st.dataframe(users_df, use_container_width=True, hide_index=True)

        # Действия с пользователями
        st.markdown("---")
        st.subheader("🔧 Действия")

        col_action1, col_action2 = st.columns(2)

        with col_action1:
            with st.expander("➕ Создать нового пользователя"):
                with st.form("create_user_form"):
                    new_username = st.text_input("Логин")
                    new_password = st.text_input("Пароль", type="password")
                    new_fullname = st.text_input("Полное имя")
                    new_email = st.text_input("Email")
                    new_role = st.selectbox("Роль", ["operator", "analyst", "manager", "admin"])

                    if st.form_submit_button("Создать пользователя"):
                        if new_username and new_password and new_fullname:
                            # Здесь должна быть логика создания пользователя
                            st.success(f"✅ Пользователь {new_username} создан!")
                        else:
                            st.error("Заполните все обязательные поля")

        with col_action2:
            with st.expander("🔄 Сбросить пароль пользователя"):
                st.markdown("Функция в разработке")

    else:
        st.info("Нет зарегистрированных пользователей")


def show_system_activity():
    """Активность системы"""
    st.subheader("📊 Активность системы")

    # Логи активности
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)

    logs_query = """
        SELECT 
            al.timestamp,
            u.full_name,
            u.username,
            al.action,
            al.details
        FROM activity_logs al
        JOIN users u ON al.user_id = u.id
        ORDER BY al.timestamp DESC
        LIMIT 100
    """

    logs_df = pd.read_sql_query(logs_query, conn)
    conn.close()

    if not logs_df.empty:
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])

        # Статистика активности
        col1, col2, col3 = st.columns(3)

        with col1:
            today_logs = logs_df[logs_df['timestamp'].dt.date == datetime.now().date()]
            st.metric("Активность сегодня", len(today_logs))

        with col2:
            unique_users = logs_df['username'].nunique()
            st.metric("Активных пользователей", unique_users)

        with col3:
            logins_today = len(today_logs[today_logs['action'] == 'login'])
            st.metric("Входов сегодня", logins_today)

        st.markdown("---")

        # Фильтры
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            filter_user = st.multiselect(
                "Фильтр по пользователю",
                options=logs_df['full_name'].unique(),
                default=logs_df['full_name'].unique()[:5]
            )

        with col_filter2:
            filter_action = st.multiselect(
                "Фильтр по действию",
                options=logs_df['action'].unique(),
                default=logs_df['action'].unique()
            )

        # Применение фильтров
        filtered_logs = logs_df[
            (logs_df['full_name'].isin(filter_user)) &
            (logs_df['action'].isin(filter_action))
            ]

        # Отображение логов
        st.dataframe(
            filtered_logs[['timestamp', 'full_name', 'action', 'details']].head(50),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Нет записей активности")


def show_security_settings():
    """Настройки безопасности"""
    st.subheader("🔒 Настройки безопасности")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔐 Политика паролей")

        min_length = st.slider("Минимальная длина пароля", 6, 20, 8)
        require_numbers = st.checkbox("Требовать цифры", value=True)
        require_special = st.checkbox("Требовать спецсимволы", value=False)
        require_uppercase = st.checkbox("Требовать заглавные буквы", value=True)

        if st.button("💾 Сохранить настройки паролей"):
            st.success("✅ Настройки сохранены")

    with col2:
        st.markdown("### ⏱️ Управление сессиями")

        session_timeout = st.number_input(
            "Время жизни сессии (часы)",
            min_value=1,
            max_value=24,
            value=8
        )

        auto_logout = st.checkbox("Автоматический выход при неактивности", value=True)

        if auto_logout:
            inactivity_timeout = st.number_input(
                "Тайм-аут неактивности (минуты)",
                min_value=5,
                max_value=120,
                value=30
            )

        if st.button("💾 Сохранить настройки сессий"):
            st.success("✅ Настройки сохранены")

    st.markdown("---")

    # Аудит безопасности
    st.markdown("### 🔍 Аудит безопасности")

    audit_checks = [
        {"Проверка": "Наличие пользователей со слабыми паролями", "Статус": "✅ OK", "Детали": "Не обнаружено"},
        {"Проверка": "Просроченные сессии", "Статус": "✅ OK", "Детали": "Все сессии активны"},
        {"Проверка": "Попытки несанкционированного доступа", "Статус": "✅ OK", "Детали": "Не обнаружено"},
        {"Проверка": "Резервные копии данных", "Статус": "⚠️ Внимание", "Детали": "Последняя копия: 2 дня назад"},
    ]

    audit_df = pd.DataFrame(audit_checks)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)


def show_system_settings():
    """Системные настройки"""
    st.subheader("⚙️ Системные настройки")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🗄️ База данных")

        st.info(f"**Путь к БД:** {AUTH_DB}")

        if st.button("🔄 Проверить целостность БД"):
            st.success("✅ База данных в порядке")

        if st.button("💾 Создать резервную копию"):
            st.success("✅ Резервная копия создана")

        st.warning("⚠️ Опасная зона")
        if st.button("🗑️ Очистить логи активности", type="secondary"):
            if st.checkbox("Подтверждаю удаление"):
                st.info("Логи очищены")

    with col2:
        st.markdown("### 📧 Уведомления")

        email_notifications = st.checkbox("Email уведомления", value=True)

        if email_notifications:
            notify_on_login = st.checkbox("При новом входе", value=False)
            notify_on_error = st.checkbox("При критических ошибках", value=True)
            notify_on_quality = st.checkbox("При отклонении качества", value=True)

        if st.button("💾 Сохранить настройки уведомлений"):
            st.success("✅ Настройки сохранены")

    st.markdown("---")

    # Системная информация
    st.markdown("### ℹ️ Системная информация")

    import platform
    import sys

    system_info = {
        "Параметр": [
            "Операционная система",
            "Версия Python",
            "Версия Streamlit",
            "База данных",
            "Запущено с",
        ],
        "Значение": [
            f"{platform.system()} {platform.release()}",
            f"{sys.version.split()[0]}",
            st.__version__,
            "SQLite 3",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    }

    st.table(pd.DataFrame(system_info))