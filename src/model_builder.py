# src/model.py
import torch
import gc
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from sentence_transformers import SentenceTransformer, models

from config import (
    MODEL_NAME, MAX_LENGTH, DEVICE,
    QUANTIZATION_CONFIG, LORA_CONFIG
)


def create_peft_model():
    """
    Создание базовой модели с QLoRA.

    :return: PEFT модель.
    """
    torch.cuda.empty_cache()
    gc.collect()

    quantization_config = BitsAndBytesConfig(**QUANTIZATION_CONFIG)
    lora_config = LoraConfig(**LORA_CONFIG)

    print(f"Загрузка базовой модели {MODEL_NAME}")
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("Применение LoRA адаптеров...")
    peft_model = get_peft_model(base_model, lora_config)

    # Включаем градиенты для LoRA параметров
    for name, param in peft_model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    peft_model.print_trainable_parameters()
    return peft_model


def create_sentence_transformer(peft_model):
    """
    Создание SentenceTransformer на основе PEFT модели.

    :param peft_model: PEFT модель.
    :return: SentenceTransformer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    word_embedding_model = models.Transformer(
        MODEL_NAME,
        max_seq_length=MAX_LENGTH,
        tokenizer_args={
            'padding': True,
            'truncation': True,
            'max_length': MAX_LENGTH,
            'return_attention_mask': True
        }
    )

    # Замена базовой модели на PEFT
    word_embedding_model.auto_model = peft_model

    # Настройка конфигурации
    if hasattr(word_embedding_model.auto_model, 'config'):
        word_embedding_model.auto_model.config.output_hidden_states = False
        word_embedding_model.auto_model.config.output_attentions = False
        if hasattr(word_embedding_model.auto_model.config, 'use_cache'):
            word_embedding_model.auto_model.config.use_cache = False

    pooling_model = models.Pooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        pooling_mode='mean'
    )

    normalize_model = models.Normalize()

    sentence_model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, normalize_model]
    )

    print("SentenceTransformer создан успешно")
    return sentence_model