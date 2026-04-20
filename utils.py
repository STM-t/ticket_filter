"""
Утилиты для предобработки текста заявок.
Правила допустимы ТОЛЬКО на этапе предобработки.
"""

import re
import string
import nltk
from typing import List
from pymorphy2 import MorphAnalyzer

# Загрузка стоп-слов
try:
    from nltk.corpus import stopwords
    STOPWORDS_RU = set(stopwords.words('russian'))
    STOPWORDS_EN = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS_RU = set(stopwords.words('russian'))
    STOPWORDS_EN = set(stopwords.words('english'))

STOPWORDS = STOPWORDS_RU | STOPWORDS_EN
MORPH = MorphAnalyzer()

# Словарь для нормализации технических терминов (опционально)
TECH_NORMALIZE = {
    'принтер': 'принтер', 'сканер': 'сканер', 'монитор': 'монитор',
    'сервер': 'сервер', 'комп': 'компьютер', 'ноут': 'ноутбук',
    '1с': '1с', 'крм': 'crm', 'сэд': 'сэд',
    'вируса': 'вирус', 'антивирус': 'антивирус',
    'пароль': 'пароль', 'логин': 'логин', 'доступ': 'доступ',
}


def preprocess_text(text: str) -> str:
    """
    Полный пайплайн предобработки текста:
    1. Приведение к нижнему регистру
    2. Удаление URL, email, спецсимволов
    3. Удаление пунктуации и цифр
    4. Токенизация
    5. Удаление стоп-слов
    6. Лемматизация
    """
    # 1. Нижний регистр
    text = text.lower()
    
    # 2. Удаление URL и email (правила для предобработки — ДОПУСТИМО)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 3. Удаление пунктуации и цифр
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # 4. Токенизация (простая по пробелам)
    tokens = text.split()
    
    # 5. Удаление стоп-слов и коротких токенов
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    
    # 6. Лемматизация через pymorphy2
    lemmatized = []
    for token in tokens:
        parsed = MORPH.parse(token)[0]
        lemma = parsed.normal_form
        # Нормализация технических терминов
        lemma = TECH_NORMALIZE.get(lemma, lemma)
        lemmatized.append(lemma)
    
    return ' '.join(lemmatized)


def calculate_priority(user_email: str, group_label: str, hierarchy: List[dict]) -> dict:
    """
    Расчёт приоритета заявки на основе иерархии пользователя.
    
    Формула: priority_score = weight * group_multiplier
    group_multiplier: Infrastructure=1.2, Security=1.3, BusinessApps=1.1, Workplace=1.0
    """
    # Поиск пользователя в иерархии
    user = next((u for u in hierarchy if u['email'] == user_email), 
                {'weight': 1.0, 'position': 'Unknown', 'status': 'L5'})
    
    # Множители для групп (бизнес-логика)
    group_multipliers = {
        'Infrastructure': 1.2,
        'Security': 1.3,
        'BusinessApps': 1.1,
        'Workplace': 1.0
    }
    multiplier = group_multipliers.get(group_label, 1.0)
    
    # Расчёт скоринга
    score = round(user['weight'] * multiplier, 2)
    
    # Определение уровня приоритета
    if score >= 2.0:
        level = 'Критический'
    elif score >= 1.5:
        level = 'Высокий'
    elif score >= 1.2:
        level = 'Средний'
    else:
        level = 'Низкий'
    
    return {
        'user_position': user['position'],
        'user_level': user['status'],
        'group_multiplier': multiplier,
        'priority_score': score,
        'priority_level': level
    }
