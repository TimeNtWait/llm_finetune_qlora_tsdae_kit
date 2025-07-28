# src/evaluate.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
import os

from config import MODEL_NAME, OUTPUT_DIR, TEST_SAMPLE_SIZE, NUM_RANDOM_PAIRS, TOP_K_SIMILAR


def evaluate_model(model, val_texts):
    """
    Оценка качества обученной модели.

    :param model: SentenceTransformer модель.
    :param val_texts: Валидационные тексты.
    """
    test_sample = np.random.choice(val_texts, min(TEST_SAMPLE_SIZE, len(val_texts)), replace=False)
    embeddings = model.encode(test_sample, convert_to_tensor=True, show_progress_bar=True)
    n_embeddings = len(embeddings)

    # Средняя косинусная схожесть между случайными парами
    if n_embeddings < 2:
        print("Недостаточно данных для вычисления средней схожести (меньше 2 эмбеддингов)")
        avg_similarity = 0.0
        std_similarity = 0.0
    else:
        similarities = []
        for _ in range(NUM_RANDOM_PAIRS):
            idx1, idx2 = np.random.choice(n_embeddings, 2, replace=False)
            sim = torch.cosine_similarity(embeddings[idx1:idx1 + 1], embeddings[idx2:idx2 + 1])
            similarities.append(sim.item())
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
    print(f"Средняя косинусная схожесть: {avg_similarity:.4f} ± {std_similarity:.4f}")

    # Кластеризация: средняя схожесть с топ-K
    sample_size = min(50, n_embeddings)
    if sample_size < 2:
        print("Недостаточно данных для вычисления топ-K схожести (меньше 2 эмбеддингов)")
        avg_topk_similarity = 0.0
    else:
        sample_embeddings = embeddings[:sample_size].cpu().numpy()
        cosine_matrix = cosine_similarity(sample_embeddings)
        topk_similarities = []
        for i in range(sample_size):
            k = min(TOP_K_SIMILAR, sample_size - 1)
            if k > 0:
                topk_indices = np.argsort(cosine_matrix[i])[-k - 1:-1]  # Исключаем себя
                topk_similarities.append(np.mean(cosine_matrix[i][topk_indices]))
            else:
                topk_similarities.append(0.0)
        avg_topk_similarity = np.mean(topk_similarities) if topk_similarities else 0.0
    print(f"Средняя схожесть с топ-{TOP_K_SIMILAR} похожими товарами: {avg_topk_similarity:.4f}")

    # Нормы embeddings
    if n_embeddings > 0:
        norms = torch.norm(embeddings, dim=1).cpu().numpy()
        print(f"Средняя норма embedding: {np.mean(norms):.4f} ± {np.std(norms):.4f}")
    else:
        print("Нет данных для вычисления норм embeddings")


def load_and_test_model(model_path=os.path.join(OUTPUT_DIR, 'final')):
    """
    Загрузка и тестирование модели.

    :param model_path: Путь к сохраненной модели.
    :return: Загруженная модель.
    """
    print(f" Загрузка модели из {model_path}")
    loaded_model = SentenceTransformer(model_path)

    test_texts = [
        "Смартфон Apple iPhone 14 128GB черный",
        "Телефон Apple iPhone 14 128 ГБ черного цвета",
        "Ноутбук ASUS VivoBook 15.6 дюймов",
        "Кофемашина Nespresso автоматическая"
    ]

    embeddings = loaded_model.encode(test_texts)
    print(f"Размерность embeddings: {embeddings.shape}")

    similarity_matrix = cosine_similarity(embeddings)
    print("Матрица косинусной схожести:")
    for i, text in enumerate(test_texts):
        print(f"{i}: {text[:50]}...")
    print(similarity_matrix)

    return loaded_model


def compare_with_base_model(trained_model_path=os.path.join(OUTPUT_DIR, 'final')):
    """
    Сравнение обученной модели с базовой.

    :param trained_model_path: Путь к обученной модели.
    """
    test_texts = [
        "Смартфон Apple iPhone 14 128GB черный",
        "Телефон Apple iPhone 14 128 ГБ черного цвета",
        "Ноутбук ASUS VivoBook 15.6 дюймов",
        "Кофемашина Nespresso автоматическая"
    ]

    base_model = SentenceTransformer(MODEL_NAME)
    trained_model = SentenceTransformer(trained_model_path)

    base_embeddings = base_model.encode(test_texts)
    trained_embeddings = trained_model.encode(test_texts)

    base_similarities = cosine_similarity(base_embeddings)
    trained_similarities = cosine_similarity(trained_embeddings)

    print("Базовая модель - матрица схожости:")
    print(base_similarities)
    print("\nОбученная модель - матрица схожости:")
    print(trained_similarities)

    diff = trained_similarities - base_similarities
    print(f"\nСредняя разность в схожести: {np.mean(np.abs(diff)):.4f}")