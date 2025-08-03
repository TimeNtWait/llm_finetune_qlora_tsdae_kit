# src/preprocess.py
"""
Пример скрипта для предобработки исходных данных.
Итог - датасет с единственной колонкой 'text' который уже буедт использоваться при fine tune модели.
"""

import argparse
import pandas as pd
import re

def join_characteristic(x):
    if x is None or pd.isna(x):
        return ""
    x = str(x).replace(':', ': ').replace(',', ', ')
    return re.sub('[{}\[\]\(\)\"]', '', x.strip().replace('\\n', '; '))


def preprocess_data(input_path: str, output_path: str):
    print(f"Загрузка данных из {input_path}...")
    df = pd.read_parquet(input_path)

    print("Обработка данных...")
    # Обработка характеристик
    df['join_characteristic'] = df['characteristic_attributes_mapping'].apply(join_characteristic)
    df['text'] = df['name'].astype(str) + ". " + df['join_characteristic']

    # Очистка данных
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    df = df[df['text'].str.len() > 15]
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str[:512 * 3] # Максимальная длина (примерно 3 символа на токен)

    # Сохраняем только колонку 'text'
    df = df[['text']]

    print(f"Подготовлено {len(df)} текстов. Сохранили в {output_path}...")
    df.to_parquet(output_path, index=False)

    print("Предобработка завершена!")

if __name__ == "__main__":
    INPUT_PATH = '../datasets/test_data.parquet'
    OUTPUT_PATH = '../datasets/processed_texts.parquet'

    preprocess_data(INPUT_PATH, OUTPUT_PATH)

