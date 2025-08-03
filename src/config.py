# src/config.py

MODEL_NAME = 'intfloat/multilingual-e5-small'  # base модель
MAX_LENGTH = 512  # макс токенов

# данные
MIN_TEXT_LENGTH = 15  # мин длина текста
TEST_SIZE = 0.1  # размер валидации

# QLoRA
import torch
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

# параметры обучения
OUTPUT_DIR = './tsdae_qlora_model'
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4  # шаги градиентной аккумуляции batch size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
# логирование/сохранеие/валидация
LOGGING_STEPS = 50
SAVE_STEPS = 600
EVAL_STEPS = 200
SAVE_TOTAL_LIMIT = 3  # кол-во чекпоинтов
# оценка
EVAL_SAMPLE_SIZE = 200
TEST_SAMPLE_SIZE = 600
NUM_RANDOM_PAIRS = 100  # кол-во случайных пар для вычисления средней схожести
TOP_K_SIMILAR = 5  # топ-K похожих товаров для оценки кластеризации