#!/usr/bin/env python3
"""
Этап 2: Веб-приложение для классификации заявок.

- Загружает обученную модель
- Принимает заявку через веб-форму
- Предобрабатывает, векторизует, предсказывает
- Выводит результат с приоритетом
"""

import os
import json
import joblib
from flask import Flask, render_template, request, jsonify
from utils import preprocess_text, calculate_priority

app = Flask(__name__)

# Глобальные переменные для модели
MODEL = None
HIERARCHY = None


def load_model():
    """Загрузка обученной модели и векторайзера."""
    global MODEL, HIERARCHY
    
    model_path = 'models/model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}\n"
            "Сначала запустите: python train.py"
        )
    
    MODEL = joblib.load(model_path)
    print("✅ Модель загружена:", list(MODEL.keys()))
    
    # Загрузка иерархии
    with open('hierarchy.json', 'r', encoding='utf-8') as f:
        HIERARCHY = json.load(f)
    print(f"✅ Иерархия загружена: {len(HIERARCHY)} пользователей")


@app.route('/')
def index():
    """Главная страница с формой."""
    return render_template('index.html', 
                         departments=MODEL['departments'],
                         groups=MODEL['groups'])


@app.route('/predict', methods=['POST'])
def predict():
    """API эндпоинт для предсказания."""
    try:
        data = request.get_json() or request.form
        
        ticket_text = data.get('text', '').strip()
        user_email = data.get('email', '').strip().lower()
        
        if not ticket_text:
            return jsonify({'error': 'Текст заявки не может быть пустым'}), 400
        
        # Предобработка текста
        text_clean = preprocess_text(ticket_text)
        
        # Векторизация и предсказание ГРУППЫ
        X_vec_group = MODEL['group_vectorizer'].transform([text_clean])
        group_pred = MODEL['group_model'].predict(X_vec_group)[0]
        group_proba = MODEL['group_model'].predict_proba(X_vec_group)[0]
        
        # Векторизация и предсказание ОТДЕЛА
        X_vec_dept = MODEL['dept_vectorizer'].transform([text_clean])
        dept_pred = MODEL['dept_model'].predict(X_vec_dept)[0]
        dept_proba = MODEL['dept_model'].predict_proba(X_vec_dept)[0]
        
        # Получение вероятностей для топ-3 предсказаний
        def get_top_predictions(labels, probas, top_n=3):
            import numpy as np
            indices = probas.argsort()[-top_n:][::-1]
            return [
                {'label': labels[i], 'confidence': round(float(probas[i]) * 100, 2)}
                for i in indices
            ]
        
        # Расчёт приоритета
        priority = calculate_priority(user_email, group_pred, HIERARCHY)
        
        result = {
            'success': True,
            'original_text': ticket_text,
            'cleaned_text': text_clean,
            'prediction': {
                'group': {
                    'label': group_pred,
                    'confidence': round(float(max(group_proba)) * 100, 2),
                    'top_3': get_top_predictions(MODEL['groups'], group_proba)
                },
                'department': {
                    'label': dept_pred,
                    'confidence': round(float(max(dept_proba)) * 100, 2),
                    'top_3': get_top_predictions(MODEL['departments'], dept_proba)
                }
            },
            'priority': priority
        }
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/status')
def status():
    """Проверка статуса сервиса."""
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'departments_count': len(MODEL['departments']) if MODEL else 0,
        'groups_count': len(MODEL['groups']) if MODEL else 0
    })


if __name__ == '__main__':
    print("🚀 Запуск веб-приложения системы фильтрации заявок")
    print("=" * 60)
    
    # Загрузка модели при старте
    load_model()
    
    # Запуск Flask
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Чтобы не перезагружать модель при каждом изменении
    )
