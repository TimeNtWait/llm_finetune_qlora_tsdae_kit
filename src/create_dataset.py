# src/create_dataset.py
"""
Модуль для подготовки данных. Предполагается, что данные уже с колонкой 'text'.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm.auto import tqdm
import json
import os

from config import (
    MIN_TEXT_LENGTH, MAX_TEXT_LENGTH, TEST_SIZE, RANDOM_STATE
)

def load_dataset(file_path, file_format=None, **kwargs):
    """
    Загрузка датафрейма. Ожидается колонка 'text'.
    Если file_format не указан, определяется по расширению файла.

    :param file_path: Путь к файлу с данными.
    :param file_format: Формат файла ('parquet', 'csv', 'json'). Если None, определяется автоматически.
    :param kwargs: Дополнительные аргументы для pd.read_* функций.
    :return: Pandas DataFrame с загруженными данными.
    """
    if file_format is None:
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if ext in ['parquet', 'csv', 'json']:
            file_format = ext
        else:
            raise ValueError(f"Неизвестное расширение файла: {ext}. Укажите file_format явно.")

    if file_format == 'parquet':
        df = pd.read_parquet(file_path, **kwargs)
    elif file_format == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_format == 'json':
        df = pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Неподдерживаемый формат: {file_format}")

    if 'text' not in df.columns:
        raise ValueError("Датасет должен содержать колонку 'text'")

    print(f"Загружен датасет из {file_path} ({len(df)} записей)")
    return df


def preprocess_text(df):
    """
    Очистка уже подготовленного текста.
    :param df: DataFrame с колонкой 'text'.
    :return: Очищенный DataFrame.
    """
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    df = df[df['text'].str.len() > MIN_TEXT_LENGTH]
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str[:MAX_TEXT_LENGTH]

    print(f"После очистки: {len(df)} записей")
    return df


def create_datasets(texts, limit=None):
    """
    Создание обучающего и валидационного датасетов.
    :param texts: Список текстов.
    :param limit: Ограничение размера датасета (для тестирования).
    :return: train_texts, val_texts.
    """
    if limit:
        texts = texts[:limit]
    train_texts, val_texts = train_test_split(texts, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Подготовлено: {len(train_texts)} обучающих, {len(val_texts)} валидационных примеров")
    return train_texts, val_texts


def generate_noisy_datasets(train_texts, val_texts, batch_size=200):
    """
    Генерация датасетов с зашумленными текстами для TSDAE.

    :param train_texts: Обучающие тексты.
    :param val_texts: Валидационные тексты.
    :param batch_size: Размер батча для map.
    :return: train_dataset, val_dataset (Dataset от HuggingFace).
    """
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset

    def generate_corrupted_batch(batch):
        """Генерация зашумленных текстов в батче."""
        corrupted = []
        original = []
        for text in batch['text']:
            try:
                rand_val = np.random.random()
                if rand_val < 0.3:
                    noisy = DenoisingAutoEncoderDataset.delete(text, delete_ratio=0.1)
                elif rand_val < 0.6:
                    noisy = DenoisingAutoEncoderDataset.replace(text, replace_ratio=0.1)
                else:
                    noisy = DenoisingAutoEncoderDataset.insert(text, insert_ratio=0.1)
                corrupted.append(noisy)
                original.append(text)
            except Exception:
                corrupted.append(text)
                original.append(text)
        return {'sentence1': corrupted, 'sentence2': original}

    train_dataset = Dataset.from_dict({'text': train_texts})
    val_dataset = Dataset.from_dict({'text': val_texts})

    train_dataset = train_dataset.map(generate_corrupted_batch, batched=True, batch_size=batch_size,
                                      remove_columns=['text'])
    val_dataset = val_dataset.map(generate_corrupted_batch, batched=True, batch_size=batch_size,
                                  remove_columns=['text'])

    print(f"Датасеты созданы: train={len(train_dataset)}, val={len(val_dataset)}")
    return train_dataset, val_dataset