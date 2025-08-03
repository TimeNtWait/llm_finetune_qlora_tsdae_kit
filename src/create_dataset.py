# src/create_dataset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm.auto import tqdm
import json
import os

from config import MAX_LENGTH, MIN_TEXT_LENGTH, TEST_SIZE

def load_dataset(file_path, file_format=None, **kwargs):
    """
    В датафрейме обязательно должна быть колонка 'text'!
    """
    if file_format is None:
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if ext in ['parquet', 'csv']:
            file_format = ext
        else:
            raise ValueError(f"Неизвестный формат файла, можно parquet, csv: {ext}")

    if file_format == 'parquet':
        df = pd.read_parquet(file_path, **kwargs)
    elif file_format == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    else:
        raise ValueError(f"Неподдерживаемый формат, можно parquet, csv: {file_format}")

    if 'text' not in df.columns:
        raise ValueError("Датафрейм должен содержать колонку 'text'")

    print(f"Загружен датафрейм из {file_path}: {len(df)}")
    return df


def preprocess_text(df):
    df = df.dropna(subset=['text'])
    df = df[(df['text'].str.strip() != '')&(df['text'].str.len() > MIN_TEXT_LENGTH)]
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str[:MAX_LENGTH * 3]  # макс длина текста (примерно 3 символа на токен)

    print(f"После очистки: {len(df)} записей")
    return df


def create_datasets(texts, limit=None):
    """
    Создание обучающего и валидационного датасетов.
    """
    if limit:
        texts = texts[:limit]
    train_texts, val_texts = train_test_split(texts, test_size=TEST_SIZE, random_state=53)
    print(f"Подготовлено train/val: {len(train_texts)} / {len(val_texts)}")
    return train_texts, val_texts


def generate_noisy_datasets(train_texts, val_texts, batch_size=200):
    """
    Генерация датасета с зашумленными текстами для TSDAE.
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