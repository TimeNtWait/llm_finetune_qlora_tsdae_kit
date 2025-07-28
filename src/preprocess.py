# preprocess.py
"""
Пример скрипта для предобработки исходных данных.
Этот скрипт загружает исходный датасет и производит предобработку (например если датасет это описания товаров, то объединяем названия товаров и характеристики в один текст).
Итоговый датасет это parquet-файл с единственной колонкой 'text' который уже буедт использоваться при fine tune модели.
"""

import argparse
import pandas as pd
import re

MIN_TEXT_LENGTH = 15  # Минимальная длина текста
MAX_TEXT_LENGTH = 512 * 3  # Максимальная длина (примерно 3 символа на токен)

def join_characteristic(x):
    """
    Объединение характеристик в строку.

    :param x: Значение характеристики (строка или None).
    :return: Очищенная строка.
    """
    if x is None or pd.isna(x):
        return ""
    x = str(x).replace(':', ': ').replace(',', ', ')
    return re.sub('[{}\[\]\(\)\"]', '', x.strip().replace('\\n', '; '))


def preprocess_data(input_path, output_path):
    """
    Основная функция предобработки.

    :param input_path: Путь к исходному файлу (parquet).
    :param output_path: Путь для сохранения результата.
    """
    print(f"Загрузка данных из {input_path}...")
    df = pd.read_parquet(input_path)

    print("Обработка данных...")
    # Обработка характеристик
    df['join_characteristic'] = df['characteristic_attributes_mapping'].apply(join_characteristic)
    df['text'] = df['name'].astype(str) + ". " + df['join_characteristic']

    # Очистка данных
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    df = df[df['text'].str.len() > MIN_TEXT_LENGTH]
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str[:MAX_TEXT_LENGTH]

    # Сохраняем только колонку 'text'
    df = df[['text']]

    print(f"Подготовлено {len(df)} текстов. Сохранили в {output_path}...")
    df.to_parquet(output_path, index=False)

    print("Предобработка завершена!")

if __name__ == "__main__":
    INPUT_PATH = '../datasets/test_data.parquet'
    OUTPUT_PATH = '../datasets/processed_texts.parquet'

    preprocess_data(INPUT_PATH, OUTPUT_PATH)

