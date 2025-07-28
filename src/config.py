# src/config.py
import torch

# Основные настройки модели
MODEL_NAME = 'intfloat/multilingual-e5-small'  # базовая модель
MAX_LENGTH = 512  # Максимальная длина последовательности токенов
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # девайс

# Настройки подготовки данных
MIN_TEXT_LENGTH = 15  # Минимальная длина текста для включения в датасет
MAX_TEXT_LENGTH = MAX_LENGTH * 3  # Максимальная длина текста (примерно 3 символа на токен)
TEST_SIZE = 0.1  # Доля валидационной выборки

# Настройки QLoRA
QUANTIZATION_CONFIG = {
    'load_in_4bit': True,
    'bnb_4bit_compute_dtype': torch.bfloat16,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_quant_type': "nf4"
}

LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'target_modules': ["query", "key", "value", "dense"],
    'lora_dropout': 0.1,
    'bias': "none",
    'task_type': "FEATURE_EXTRACTION",
    'inference_mode': False
}

# Настройки обучения
OUTPUT_DIR = './tsdae_qlora_model'  # Директория для сохранения модели
NUM_EPOCHS = 2  # Количество эпох обучения
TRAIN_BATCH_SIZE = 4  # Размер батча для обучения
EVAL_BATCH_SIZE = 2  # Размер батча для оценки
GRAD_ACCUM_STEPS = 4  # Шаги градиентной аккумуляции (эффективный batch size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)
LEARNING_RATE = 2e-5  # Скорость обучения
WARMUP_RATIO = 0.1  # Доля шагов для warmup
WEIGHT_DECAY = 0.01  # Регуляризация весов
LOGGING_STEPS = 50  # Шаги для логирования
SAVE_STEPS = 600  # Шаги для сохранения чекпоинтов
EVAL_STEPS = 200  # Шаги для оценки
SAVE_TOTAL_LIMIT = 3  # Максимальное количество сохраненных чекпоинтов

# Настройки оценки
EVAL_SAMPLE_SIZE = 200  # Размер выборки для evaluator'а
TEST_SAMPLE_SIZE = 600  # Размер выборки для финальной оценки
NUM_RANDOM_PAIRS = 100  # Количество случайных пар для вычисления средней схожести
TOP_K_SIMILAR = 5  # Топ-K похожих товаров для оценки кластеризации

# Флаги
USE_EVALUATOR = True  # Использовать evaluator во время обучения (True/False)
USE_BF16 = True  # Использовать bfloat16 для обучения