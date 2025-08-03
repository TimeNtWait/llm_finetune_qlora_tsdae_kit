# src/main.py
import torch
import os

from create_dataset import load_dataset, preprocess_text, create_datasets, generate_noisy_datasets
from model_builder import create_peft_model, create_sentence_transformer
from train import train_model
from evaluate import evaluate_model
import gc

def main(data_path, limit=None):
    print("Начинаем fine-tune TSDAE + QLoRA")

    # Подготовка данных
    df = load_dataset(data_path)
    df = preprocess_text(df)
    texts = df['text'].tolist()
    train_texts, val_texts = create_datasets(texts, limit=limit)
    train_dataset, val_dataset = generate_noisy_datasets(train_texts, val_texts)

    # Создание модели
    peft_model = create_peft_model()
    sentence_model = create_sentence_transformer(peft_model)

    # Обучение
    train_model(sentence_model, train_dataset, val_dataset)

    # Оценка
    evaluate_model(sentence_model, val_texts)

    # Освобождение памяти
    torch.cuda.empty_cache()
    gc.collect()

    print(" Процесс завершен успешно!")

if __name__ == "__main__":
    # Путь к данным (важно чтобы в ДФ была колонка 'text')
    DEFAULT_DATA_PATH = '../datasets/processed_texts.parquet'
    main(data_path=DEFAULT_DATA_PATH, limit=200) # Лимит для тетсирования, если limit=None используются все данные
    # main(data_path=DEFAULT_DATA_PATH)