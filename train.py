#!/usr/bin/env python3
"""
Этап 1: Обучение модели машинного обучения.

- Генерация синтетического датасета (2500 заявок)
- Предобработка текста
- Векторизация TF-IDF
- Обучение классификатора
- Оценка качества и сохранение модели
"""

import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

from utils import preprocess_text

# Создаём папку для моделей
os.makedirs('models', exist_ok=True)

# ============================================================================
# ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА
# ============================================================================

# Шаблоны для генерации заявок по группам и отделам
TEMPLATES = {
    'Infrastructure': {
        'Network': [
            "Не пингуется сервер {ip}, вся сеть легла",
            "Проблемы с подключением к сети в кабинете {room}",
            "VPN не подключается к офису, пишет ошибку {err_code}",
            "Медленный интернет, скорость всего {speed} Мбит/с",
            "Не работает сетевой принтер по адресу {ip}",
            "Обрыв кабеля в серверной, нет связи с {location}",
            "Маршрутизатор не раздаёт IP-адреса по DHCP",
            "Недоступен шлюз по адресу {ip}, проверка связи не проходит",
        ],
        'Servers': [
            "Сервер {server_name} не отвечает на запросы",
            "Падение службы {service_name} на производственном сервере",
            "Высокая загрузка CPU на сервере {server_name}: {load}% ",
            "Не монтируется диск на сервере баз данных",
            "Ошибка в логах Event Log: {error_msg}",
            "Сервер завис при обновлении, требуется перезагрузка",
            "Не запускается виртуальная машина {vm_name}",
            "Превышен лимит дискового пространства на {mount_point}",
        ],
        'Backup': [
            "Не работает резервное копирование баз данных уже {days} день",
            "Ошибка в задании бэкапа: {error_msg}",
            "Не восстанавливается файл из архива за {date}",
            "Бэкап не завершился, статус: {status}",
            "Требуется восстановить удалённый файл из бэкапа",
            "Не хватает места в хранилище резервных копий",
            "Расписание бэкапов сбилось после обновления",
            "Проверка целостности бэкапа завершилась с ошибкой",
        ],
        'Monitoring': [
            "Система мониторинга не присылает алерты",
            "Графики в Zabbix не обновляются уже {hours} часов",
            "Ложное срабатывание мониторинга на {metric}",
            "Не отображается статус сервиса {service_name} в дашборде",
            "Пропущены метрики с хоста {hostname} за период {period}",
            "Требуется добавить новый хост в систему мониторинга",
            "Оповещения в Telegram не приходят от бота мониторинга",
            "Некорректно считается uptime сервиса {service_name}",
        ]
    },
    'Workplace': {
        'OS': [
            "Синий экран при запуске, ошибка {bsod_code}",
            "Windows не загружается после обновления",
            "Зависает проводник при открытии папки {folder}",
            "Не обновляется система, ошибка кода {err_code}",
            "Пропадает звук после выхода из спящего режима",
            "Не работает автозапуск программ при старте системы",
            "Тормозит система, процесс {process} грузит процессор",
            "Не отображаются сетевые диски в Проводнике",
        ],
        'Periphery': [
            "Принтер не печатает, мигает красная лампочка",
            "Сканер не определяется системой, драйвер не установлен",
            "Мышь не работает, пробовал перетыкать в разные USB",
            "Клавиатура печатает не те символы, раскладка сбита",
            "Монитор не включается, индикатор не горит",
            "Веб-камера показывает чёрный экран в конференциях",
            "Наушники не определяются, звук идёт через динамики",
            "Картридж в принтере закончился, требуется замена",
        ],
        'RemoteAccess': [
            "Не подключается RDP к рабочему компьютеру",
            "Citrix вылетает при запуске приложения {app_name}",
            "Забыл пароль от VPN, нужен сброс доступа",
            "Токен для двухфакторной аутентификации не принимается",
            "Не работает удалённый рабочий стол через браузер",
            "Сессия в Terminal Server обрывается через {minutes} минут",
            "Не могу подключиться к корпоративному порталу из дома",
            "Ошибка сертификата при подключении к VPN-шлюзу",
        ],
        'PCAdmin': [
            "Требуется установить ПО {software_name} на рабочую станцию",
            "Нужно добавить пользователя в локальную группу {group}",
            "Компьютер не входит в домен, ошибка {err_msg}",
            "Требуется настроить групповую политику для отдела {dept}",
            "Не применяется политика паролей после изменения",
            "Нужно выдать права локального администратора на {pc_name}",
            "Требуется удалить старое ПО перед установкой нового",
            "Конфликт лицензий при запуске {software_name}",
        ]
    },
    'BusinessApps': {
        '1C': [
            "1С зависает при формировании отчёта по {report_type}",
            "Ошибка проведения документа в 1С: {error_msg}",
            "Не обновляется конфигурация 1С до версии {version}",
            "Блокировка записи в базе 1С, пользователь {user} удерживает",
            "Не печатается печатная форма в 1С, пустой документ",
            "Медленная работа 1С при открытии справочника {catalog}",
            "Ошибка подключения к базе 1С через COM-соединение",
            "Не выгружается обмен данными с сайтом из 1С",
        ],
        'CRM': [
            "Не могу зайти в CRM, пишет доступ запрещён",
            "Не сохраняются изменения в карточке клиента {client_id}",
            "Отчёт в CRM формируется с неверными данными за {period}",
            "Не приходят уведомления о новых лидах из формы на сайте",
            "Дублируются записи в воронке продаж после импорта",
            "Не работает интеграция CRM с телефонией, звонки не логируются",
            "Ошибка при экспорте данных из CRM в Excel",
            "Поле {field_name} не отображается в форме создания сделки",
        ],
        'DocumentFlow': [
            "Не открывается документ в СЭД, ошибка формата {format}",
            "Маршрут согласования документа {doc_id} завис на этапе {stage}",
            "Не приходит уведомление о подписании документа",
            "Ошибка цифровой подписи при отправке документа контрагенту",
            "Не отображается история версий документа {doc_name}",
            "Требуется восстановить удалённый документ из архива СЭД",
            "Не работает поиск по реквизитам в системе документооборота",
            "Превышен лимит на количество активных документов у пользователя",
        ],
        'SpecialSoftware': [
            "Не запускается специализированное ПО {app_name}, ошибка {err}",
            "Лицензия на {software_name} истекла, требуется продление",
            "Конфликт версий {app_name} и системной библиотеки {lib}",
            "Не сохраняется проект в {app_name}, ошибка доступа к диску",
            "Требуется настроить плагин для {app_name} под задачу {task}",
            "Падение {app_name} при обработке файла большого размера",
            "Не работает экспорт данных из {app_name} в формат {format}",
            "Обновление {app_name} сломало существующий функционал {feature}",
        ]
    },
    'Security': {
        'AccessControl': [
            "Забыл пароль от AD, нужно сбросить срочно",
            "Учётная запись заблокирована после {attempts} неудачных попыток",
            "Требуется выдать доступ к ресурсу {resource} для роли {role}",
            "Не работает вход по смарт-карте, ошибка считывания",
            "Пользователь уволен, нужно отозвать все доступы",
            "Требуется создать новую учётную запись для сотрудника {name}",
            "Не применяется политика блокировки экрана через {minutes} мин",
            "Ошибка аутентификации в корпоративном портале, код {code}",
        ],
        'Antivirus': [
            "Антивирус блокирует программу {app}, хотя она безопасна",
            "Обнаружена угроза {threat_name} на рабочей станции {pc}",
            "Не обновляются базы антивируса уже {days} дней",
            "Полное сканирование зависло на файле {filename}",
            "Антивирус грузит процессор на {percent}%, тормозит работу",
            "Ложное срабатывание на внутренний разработанный модуль",
            "Требуется добавить исключение для папки {path} в антивирусе",
            "Не удаётся удалить карантинный файл, доступ запрещён",
        ],
        'DLP': [
            "Пришло письмо с подозрительной ссылкой, возможно фишинг",
            "Система DLP заблокировала отправку файла {filename}",
            "Попытка несанкционированного копирования данных на USB",
            "Срабатывание правила DLP при отправке отчёта контрагенту",
            "Требуется проверить логи DLP на предмет утечки по ключу {keyword}",
            "Пользователь пытается обойти DLP через архив с паролем",
            "Некорректно сработало правило: легитимная операция заблокирована",
            "Требуется настроить новое правило DLP для защиты {data_type}",
        ]
    }
}

# Дополнительные переменные для подстановки
VARIABLES = {
    'ip': [f"10.0.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(20)],
    'room': [f"{random.choice(['А', 'Б', 'В'])}-{random.randint(100, 499)}" for _ in range(15)],
    'err_code': ["800", "619", "691", "0x80070005", "ERR_CONNECTION_TIMED_OUT"],
    'speed': ["0.5", "1.2", "5", "10"],
    'location': ["филиал Москва", "офис СПб", "ЦОД-1", "удалённая площадка"],
    'server_name': [f"srv-{name}-{random.randint(1,9)}" for name in ["db", "app", "web", "file", "backup"]],
    'service_name': ["MSSQLSERVER", "Apache2.4", "nginx", "Redis", "RabbitMQ", "Elasticsearch"],
    'error_msg': ["Access denied", "Timeout expired", "Disk full", "Connection refused", "NullReferenceException"],
    'load': ["95", "98", "100", "87"],
    'vm_name': [f"vm-{app}-{random.randint(1,5)}" for app in ["test", "stage", "prod", "dev"]],
    'mount_point': ["/data", "/backup", "/var/log", "D:", "E:"],
    'days': ["второй", "третий", "пятый", "неделю"],
    'date': ["2024-01-15", "вчера", "прошлой недели", "месячной давности"],
    'status': ["Failed", "Partial", "Cancelled", "Timeout"],
    'hours': ["3", "6", "12", "24"],
    'metric': ["CPU", "Memory", "Disk I/O", "Network latency"],
    'hostname': [f"host-{random.randint(1,50)}" for _ in range(10)],
    'period': ["последние 2 часа", "ночной интервал", "выходные"],
    'bsod_code': ["IRQL_NOT_LESS_OR_EQUAL", "PAGE_FAULT_IN_NONPAGED_AREA", "CRITICAL_PROCESS_DIED"],
    'folder': ["Документы", "Загрузки", "Сетевое окружение", "Рабочий стол"],
    'process': ["svchost.exe", "explorer.exe", "chrome.exe", "1cv8.exe"],
    'software_name': ["Adobe Reader", "7-Zip", "Notepad++", "WinSCP", "PuTTY", "Git"],
    'group': ["Administrators", "Power Users", "Remote Desktop Users"],
    'dept': ["Бухгалтерия", "Отдел продаж", "Юридический", "Разработка"],
    'pc_name': [f"PC-{random.choice(['HR', 'FIN', 'DEV', 'MKT'])}-{random.randint(1,99)}" for _ in range(10)],
    'report_type': ["зарплате", "налогам", "обороту", "остаткам"],
    'version': ["8.3.22", "8.3.23", "8.3.24"],
    'user': ["ivanov", "petrova", "sidorov", "admin"],
    'catalog': ["Номенклатура", "Контрагенты", "Склады", "Цены"],
    'client_id': [f"CL-{random.randint(1000,9999)}" for _ in range(10)],
    'period_crm': ["январь", "1 квартал", "прошлый месяц", "2024 год"],
    'field_name': ["Стадия сделки", "Ответственный", "Дата закрытия", "Бюджет"],
    'format': [".docx", ".pdf", ".xml", ".odt"],
    'doc_id': [f"DOC-{random.randint(10000,99999)}" for _ in range(10)],
    'stage': ["Согласование", "Подписание", "Архивация", "Публикация"],
    'doc_name': [f"Договор_{random.randint(1,100)}", f"Приказ_{random.randint(1,50)}"],
    'app_name': ["AutoCAD", "MATLAB", "SolidWorks", "Photoshop", "CorelDRAW"],
    'err': ["0xc000007b", "Missing DLL", "License error", "Runtime error"],
    'lib': ["MSVCR120.dll", ".NET Framework 4.8", "Visual C++ Redist"],
    'task': ["печать чертежей", "расчёт нагрузок", "рендеринг", "конвертация"],
    'feature': ["экспорт в PDF", "пакетная обработка", "интеграция с 1С"],
    'attempts': ["3", "5", "10"],
    'resource': [r"\\fileserver\docs", "CRM", "1C:Enterprise", "GitLab"],
    'role': ["Менеджер", "Аналитик", "Разработчик", "Бухгалтер"],
    'name': ["Иванов И.И.", "Петрова А.С.", "Сидоров П.К."],
    'minutes': ["5", "10", "15"],
    'code': ["AUTH_001", "AUTH_002", "CERT_ERR", "LDAP_FAIL"],
    'threat_name': ["Trojan.Generic", "AdWare.BrowserModifier", "PUP.Optional"],
    'pc': [f"WS-{random.randint(1,200)}" for _ in range(10)],
    'filename': ["отчёт_финансы.xlsx", "база_клиентов.db", "конфиденциально.pdf"],
    'percent': ["80", "90", "100"],
    'path': [r"C:\Projects", r"D:\Backups", "/home/user/dev", "/opt/custom"],
    'keyword': ["паспортные данные", "номер карты", "коммерческая тайна"],
    'data_type': ["персональные данные", "финансовая отчётность", "исходный код"],
}


def generate_synthetic_dataset(n_samples: int = 2500) -> pd.DataFrame:
    """Генерация синтетического датасета заявок."""
    
    data = []
    ticket_id = 1
    
    # Распределение по группам (примерно равное)
    groups = list(TEMPLATES.keys())
    
    for _ in range(n_samples):
        # Случайный выбор группы и отдела
        group = random.choice(groups)
        department = random.choice(list(TEMPLATES[group].keys()))
        
        # Выбор шаблона и генерация текста
        template = random.choice(TEMPLATES[group][department])
        
        # Подстановка переменных в шаблон
        text = template
        for key, values in VARIABLES.items():
            placeholder = f"{{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, random.choice(values), 1)
        
        # Добавление вариативности: случайные слова-паразиты
        noise_words = ["срочно", "помогите", "пожалуйста", "уже", "опять", "всё ещё"]
        if random.random() < 0.3:
            words = text.split()
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(noise_words))
            text = ' '.join(words)
        
        data.append({
            'id': ticket_id,
            'text': text,
            'group_label': group,
            'department_label': department,
            'created_at': datetime.now().isoformat()
        })
        ticket_id += 1
    
    return pd.DataFrame(data)


def train_model(df: pd.DataFrame, test_size: float = 0.2):
    """Обучение модели классификации заявок."""
    
    print(f"📊 Датасет: {len(df)} заявок")
    print(f"📋 Группы: {df['group_label'].value_counts().to_dict()}")
    print(f"🏢 Отделы: {df['department_label'].nunique()} уникальных значений")
    
    # Предобработка текстов
    print("\n🔄 Предобработка текстов...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    # Удаление пустых после предобработки
    df = df[df['text_clean'].str.len() > 0].copy()
    print(f"✅ После очистки: {len(df)} валидных заявок")
    
    # ========================================================================
    # МОДЕЛЬ 1: Классификация по ГРУППАМ отделов
    # ========================================================================
    print("\n🎯 Обучение модели для групп отделов...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_clean'], df['group_label'], 
        test_size=test_size, random_state=42, stratify=df['group_label']
    )
    
    # Векторизация TF-IDF
    vectorizer_group = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer_group.fit_transform(X_train)
    X_test_vec = vectorizer_group.transform(X_test)
    
    # Модель: Logistic Regression с балансировкой классов
    model_group = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        multi_class='multinomial'
    )
    
    model_group.fit(X_train_vec, y_train)
    
    # Оценка качества
    y_pred_group = model_group.predict(X_test_vec)
    
    print(f"\n📈 Метрики для ГРУПП отделов:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred_group):.4f}")
    print(f"   F1 (macro): {f1_score(y_test, y_pred_group, average='macro'):.4f}")
    print(f"   F1 (weighted): {f1_score(y_test, y_pred_group, average='weighted'):.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_group))
    
    # ========================================================================
    # МОДЕЛЬ 2: Классификация по КОНКРЕТНЫМ ОТДЕЛАМ
    # ========================================================================
    print("\n🎯 Обучение модели для конкретных отделов...")
    
    X_train_dept, X_test_dept, y_train_dept, y_test_dept = train_test_split(
        df['text_clean'], df['department_label'],
        test_size=test_size, random_state=42, stratify=df['department_label']
    )
    
    vectorizer_dept = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_dept_vec = vectorizer_dept.fit_transform(X_train_dept)
    X_test_dept_vec = vectorizer_dept.transform(X_test_dept)
    
    model_dept = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        multi_class='multinomial'
    )
    
    model_dept.fit(X_train_dept_vec, y_train_dept)
    
    y_pred_dept = model_dept.predict(X_test_dept_vec)
    
    print(f"\n📈 Метрики для КОНКРЕТНЫХ ОТДЕЛОВ:")
    print(f"   Accuracy: {accuracy_score(y_test_dept, y_pred_dept):.4f}")
    print(f"   F1 (macro): {f1_score(y_test_dept, y_pred_dept, average='macro'):.4f}")
    print(f"   F1 (weighted): {f1_score(y_test_dept, y_pred_dept, average='weighted'):.4f}")
    
    # ========================================================================
    # СОХРАНЕНИЕ МОДЕЛЕЙ
    # ========================================================================
    print("\n💾 Сохранение моделей...")
    
    # Сохраняем обе модели и векторайзеры
    joblib.dump({
        'group_model': model_group,
        'dept_model': model_dept,
        'group_vectorizer': vectorizer_group,
        'dept_vectorizer': vectorizer_dept,
        'preprocessing_func': preprocess_text,
        'departments': df['department_label'].unique().tolist(),
        'groups': df['group_label'].unique().tolist()
    }, 'models/model.pkl')
    
    print("✅ Модели сохранены в models/model.pkl")
    
    # Сохранение датасета для отчётности
    df.to_csv('models/dataset_sample.csv', index=False, encoding='utf-8-sig')
    print(f"✅ Пример датасета сохранён: {len(df)} записей")
    
    return model_group, model_dept, vectorizer_group, vectorizer_dept


if __name__ == "__main__":
    print("🚀 Запуск обучения модели интеллектуальной фильтрации заявок")
    print("=" * 70)
    
    # Генерация датасета
    print("\n📝 Генерация синтетического датасета...")
    df = generate_synthetic_dataset(n_samples=2500)
    
    # Обучение и сохранение
    train_model(df)
    
    print("\n" + "=" * 70)
    print("✅ Обучение завершено! Теперь можно запускать app.py")
