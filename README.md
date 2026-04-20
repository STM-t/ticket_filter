### Project Structure

ticket_filter_system/
├── requirements.txt          # Зависимости
├── hierarchy.json            # Иерархия пользователей
├── utils.py                  # Утилиты предобработки
├── train.py                  # Обучение модели + генерация датасета
├── app.py                    # Flask веб-приложение
├── models/                   # Папка для сохранённых моделей
│   ├── model.pkl
│   └── vectorizer.pkl
├── templates/
│   └── index.html            # Веб-форма
├── static/
│   └── style.css             # Стили (опционально)
└── README.md                 # Документация
