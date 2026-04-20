"""
Утилиты для предобработки текста заявок.
Правила допустимы ТОЛЬКО на этапе предобработки.
Совместимо с Python 3.10 - 3.13
"""
import re
import string
import sys
from typing import List

# ============================================================================
# 1. БЕЗОПАСНАЯ ЗАГРУЗКА ЗАВИСИМОСТЕЙ
# ============================================================================
try:
    import nltk
    from nltk.corpus import stopwords
except ImportError:
    print("❌ Установите nltk: pip install nltk", file=sys.stderr)
    sys.exit(1)

# Автоматическая загрузка стоп-слов
try:
    STOPWORDS_RU = set(stopwords.words('russian'))
    STOPWORDS_EN = set(stopwords.words('english'))
except LookupError:
    print("⚠️  Загружаем NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS_RU = set(stopwords.words('russian'))
    STOPWORDS_EN = set(stopwords.words('english'))

STOPWORDS = STOPWORDS_RU | STOPWORDS_EN

# Инициализация лемматизатора (pymorphy3 для Python 3.12+)
try:
    from pymorphy3 import MorphAnalyzer
except ImportError:
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError:
        print("❌ Установите pymorphy3: pip install pymorphy3 pymorphy3-dicts-ru", file=sys.stderr)
        sys.exit(1)

try:
    MORPH = MorphAnalyzer()
except Exception as e:
    print(f"❌ Ошибка инициализации лемматизатора: {e}", file=sys.stderr)
    print("💡 Попробуйте: pip install --upgrade setuptools pymorphy3 pymorphy3-dicts-ru", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# 2. ПРЕДОБРАБОТКА ТЕКСТА
# ============================================================================
TECH_NORMALIZE = {
    'принтер': 'принтер', 'сканер': 'сканер', 'монитор': 'монитор',
    'сервер': 'сервер', 'комп': 'компьютер', 'ноут': 'ноутбук',
    '1с': '1с', 'крм': 'crm', 'сэд': 'сэд',
    'вируса': 'вирус', 'антивирус': 'антивирус',
    'пароль': 'пароль', 'логин': 'логин', 'доступ': 'доступ',
}

def preprocess_text(text: str) -> str:
    """Полный пайплайн предобработки текста."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    
    lemmatized = []
    for token in tokens:
        try:
            parsed = MORPH.parse(token)[0]
            lemma = parsed.normal_form
            lemma = TECH_NORMALIZE.get(lemma, lemma)
            lemmatized.append(lemma)
        except Exception:
            lemmatized.append(token)  # fallback
            
    return ' '.join(lemmatized)

def calculate_priority(user_email: str, group_label: str, hierarchy: List[dict]) -> dict:
    """Расчёт приоритета заявки на основе иерархии пользователя."""
    user = next((u for u in hierarchy if u['email'] == user_email), 
                {'weight': 1.0, 'position': 'Unknown', 'status': 'L5'})
    
    group_multipliers = {'Infrastructure': 1.2, 'Security': 1.3, 'BusinessApps': 1.1, 'Workplace': 1.0}
    multiplier = group_multipliers.get(group_label, 1.0)
    score = round(user['weight'] * multiplier, 2)
    
    if score >= 2.0: level = 'Критический'
    elif score >= 1.5: level = 'Высокий'
    elif score >= 1.2: level = 'Средний'
    else: level = 'Низкий'
        
    return {
        'user_position': user['position'],
        'user_level': user['status'],
        'group_multiplier': multiplier,
        'priority_score': score,
        'priority_level': level
    }