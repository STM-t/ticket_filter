#!/bin/bash
# ============================================================================
# 🎫 Интеллектуальная система фильтрации заявок техподдержки
# Автоматизированный скрипт запуска проекта
# ============================================================================
# Использование:
#   ./run.sh              # Полный запуск: установка → обучение → сервер
#   ./run.sh --train-only # Только обучение модели
#   ./run.sh --serve-only # Только запуск сервера (модель должна быть обучена)
#   ./run.sh --install    # Только установка зависимостей
#   ./run.sh --help       # Показать справку
# ============================================================================

set -e  # Остановиться при первой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Пути
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
MODELS_DIR="$PROJECT_DIR/models"
MODEL_FILE="$MODELS_DIR/model.pkl"

# Флаги
MODE="full"  # full, train-only, serve-only, install

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_step()    { echo -e "\n${CYAN}➜ $1${NC}"; }

print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  🎫 Интеллектуальная фильтрация заявок техподдержки  ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}  Дипломный проект | ML + Fullstack                   ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_help() {
    cat << EOF
${CYAN}Использование:${NC}
  ./run.sh [ОПЦИЯ]

${CYAN}Опции:${NC}
  (без опции)   Полный запуск: установка → обучение → веб-сервер
  --train-only  Только обучение модели (пропустить установку и сервер)
  --serve-only  Только запуск веб-сервера (модель должна быть обучена)
  --install     Только установка зависимостей (пропустить обучение)
  --retrain     Переобучить модель, даже если она уже существует
  --no-venv     Не использовать виртуальное окружение (использовать системный Python)
  --help        Показать эту справку

${CYAN}Примеры:${NC}
  ./run.sh                      # Первый запуск проекта
  ./run.sh --serve-only         # Запуск после перезагрузки компьютера
  ./run.sh --retrain --no-venv  # Переобучение на сервере без venv

${CYAN}Требования:${NC}
  • Python 3.8+
  • pip
  • Интернет-соединение (для установки пакетов и скачивания NLTK-данных)

EOF
}

check_python() {
    log_step "Проверка окружения"
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python не найден! Установите Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_info "Обнаружен: $PYTHON_CMD $PYTHON_VERSION"
    
    # Проверка версии (минимум 3.8)
    if [[ ! "$PYTHON_VERSION" =~ ^3\.[8-9]|[0-9]{2,} ]]; then
        log_warn "Рекомендуется Python 3.8+, текущая версия: $PYTHON_VERSION"
    fi
    
    # Проверка pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        log_error "pip не найден! Установите pip для $PYTHON_CMD"
        exit 1
    fi
    log_success "pip доступен"
}

setup_venv() {
    if [[ "$NO_VENV" == "true" ]]; then
        log_warn "Пропускаем создание виртуального окружения (флаг --no-venv)"
        return 0
    fi
    
    log_step "Настройка виртуального окружения"
    
    if [[ -d "$VENV_DIR" ]]; then
        log_info "Виртуальное окружение уже существует: $VENV_DIR"
    else
        log_info "Создаём виртуальное окружение..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        log_success "Виртуальное окружение создано"
    fi
    
    # Активация
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        source "$VENV_DIR/bin/activate"
        log_success "Активировано: venv (Linux/macOS)"
    elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
        source "$VENV_DIR/Scripts/activate"
        log_success "Активировано: venv (Windows)"
    else
        log_error "Не удалось активировать виртуальное окружение"
        exit 1
    fi
    
    # Обновление pip внутри venv
    log_info "Обновляем pip..."
    python -m pip install --upgrade pip -q
}

install_dependencies() {
    log_step "Установка зависимостей"
    
    if [[ ! -f "requirements.txt" ]]; then
        log_error "Файл requirements.txt не найден в $PROJECT_DIR"
        exit 1
    fi
    
    log_info "Устанавливаем пакеты из requirements.txt..."
    echo ""
    
    # Установка с прогрессом
    if pip install -r requirements.txt; then
        log_success "Зависимости установлены"
    else
        log_error "Ошибка при установке зависимостей"
        exit 1
    fi
    
    # Проверка критических пакетов
    for pkg in flask scikit-learn pymorphy2 nltk joblib; do
        if ! python -c "import $pkg" &> /dev/null; then
            log_error "Пакет $pkg не установлен корректно!"
            exit 1
        fi
    done
    log_success "Все критические пакеты импортируются"
}

download_nltk_data() {
    log_step "Загрузка данных NLTK"
    
    # Скачиваем stopwords (основное для предобработки)
    log_info "Скачиваем стоп-слова..."
    if python -c "import nltk; nltk.download('stopwords', quiet=True)" 2>/dev/null; then
        log_success "Stopwords загружены"
    else
        log_warn "Не удалось скачать stopwords через nltk.download()"
        log_info "Пробуем альтернативный метод..."
        python -c "
import nltk, os, sys
try:
    nltk.download('stopwords', download_dir=os.path.expanduser('~/nltk_data'), quiet=True)
    print('OK')
except:
    sys.exit(1)
" && log_success "Stopwords загружены (альтернативный путь)" || log_warn "⚠️ Stopwords могут быть недоступны"
    fi
    
    # Проверка доступности
    if python -c "from nltk.corpus import stopwords; assert len(stopwords.words('russian')) > 0" &> /dev/null; then
        log_success "Русские стоп-слова доступны ($(python -c "from nltk.corpus import stopwords; print(len(stopwords.words('russian')))") слов)"
    else
        log_error "Не удалось загрузить русские стоп-слова! Классификация может работать некорректно."
        log_info "Попробуйте запустить вручную: python -c \"import nltk; nltk.download('stopwords')\""
        if [[ "$MODE" != "serve-only" ]]; then
            read -p "Продолжить без стоп-слов? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    # Инициализация pymorphy2 (скачивает словари при первом запуске)
    log_info "Инициализация pymorphy2 (может занять время при первом запуске)..."
    python -c "
import pymorphy2
import sys
try:
    morph = pymorphy2.MorphAnalyzer()
    # Тестовая лемматизация
    result = morph.parse('сервер')[0]
    print('OK')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" && log_success "pymorphy2 готов к работе" || log_warn "⚠️ pymorphy2 может работать некорректно"
}

train_model() {
    log_step "Обучение модели машинного обучения"
    
    # Если модель уже существует и не запрошено переобучение
    if [[ -f "$MODEL_FILE" && "$RETRAIN" != "true" ]]; then
        log_warn "Модель уже существует: $MODEL_FILE"
        read -p "Переобучить модель? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Пропускаем обучение, используем существующую модель"
            return 0
        fi
    fi
    
    # Создаём папку для моделей
    mkdir -p "$MODELS_DIR"
    
    log_info "Запускаем train.py..."
    log_info "Это займёт 30-90 секунд в зависимости от мощности компьютера"
    echo ""
    
    if python train.py; then
        if [[ -f "$MODEL_FILE" ]]; then
            MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
            log_success "Модель обучена и сохранена: $MODEL_FILE ($MODEL_SIZE)"
        else
            log_error "Файл модели не создан! Проверьте вывод train.py выше."
            exit 1
        fi
    else
        log_error "Ошибка при обучении модели!"
        exit 1
    fi
}

start_server() {
    log_step "Запуск веб-сервера"
    
    if [[ ! -f "$MODEL_FILE" ]]; then
        log_error "Модель не найдена: $MODEL_FILE"
        log_info "Сначала обучите модель: ./run.sh --train-only"
        exit 1
    fi
    
    log_info "Загружаем модель и запускаем Flask на порту 5000..."
    echo ""
    log_info "${GREEN}Веб-приложение доступно по адресу:${NC}"
    echo -e "  ${CYAN}👉 http://localhost:5000${NC}"
    echo ""
    log_info "Для остановки сервера нажмите: ${YELLOW}Ctrl+C${NC}"
    echo ""
    
    # Запуск Flask
    # use_reloader=False в app.py предотвращает двойную загрузку модели
    python app.py
}

# ============================================================================
# ПАРСИНГ АРГУМЕНТОВ
# ============================================================================

NO_VENV="false"
RETRAIN="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only)
            MODE="train-only"
            shift
            ;;
        --serve-only)
            MODE="serve-only"
            shift
            ;;
        --install)
            MODE="install"
            shift
            ;;
        --retrain)
            RETRAIN="true"
            shift
            ;;
        --no-venv)
            NO_VENV="true"
            shift
            ;;
        --help|-h)
            print_header
            print_help
            exit 0
            ;;
        *)
            log_error "Неизвестная опция: $1"
            print_help
            exit 1
            ;;
    esac
done

# ============================================================================
# ОСНОВНОЙ ПОТОК ВЫПОЛНЕНИЯ
# ============================================================================

print_header

cd "$PROJECT_DIR" || exit 1

case $MODE in
    full)
        log_info "Режим: ${CYAN}ПОЛНЫЙ ЗАПУСК${NC}"
        echo "  1. Проверка окружения"
        echo "  2. Установка зависимостей"
        echo "  3. Загрузка NLTK-данных"
        echo "  4. Обучение модели"
        echo "  5. Запуск веб-сервера"
        echo ""
        
        check_python
        setup_venv
        install_dependencies
        download_nltk_data
        train_model
        start_server
        ;;
        
    train-only)
        log_info "Режим: ${CYAN}ТОЛЬКО ОБУЧЕНИЕ${NC}"
        check_python
        setup_venv
        install_dependencies
        download_nltk_data
        train_model
        log_success "Обучение завершено! Теперь запустите: ./run.sh --serve-only"
        ;;
        
    serve-only)
        log_info "Режим: ${CYAN}ТОЛЬКО СЕРВЕР${NC}"
        check_python
        setup_venv
        install_dependencies
        # Пропускаем download_nltk_data - должно быть уже скачано
        start_server
        ;;
        
    install)
        log_info "Режим: ${CYAN}ТОЛЬКО УСТАНОВКА${NC}"
        check_python
        setup_venv
        install_dependencies
        download_nltk_data
        log_success "Готово! Теперь запустите: ./run.sh --train-only"
        ;;
esac
