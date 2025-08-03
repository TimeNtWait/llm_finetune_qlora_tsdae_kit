# src/train.py

from sentence_transformers import losses, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from transformers import AutoTokenizer
import numpy as np
import torch
import gc
import os

from config import (
    MODEL_NAME, OUTPUT_DIR, NUM_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    GRAD_ACCUM_STEPS, LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY,
    LOGGING_STEPS, SAVE_STEPS, EVAL_STEPS, SAVE_TOTAL_LIMIT, EVAL_SAMPLE_SIZE
)


def setup_loss(model):
    """
    Настройка TSDAE loss.

    :param model: SentenceTransformer модель.
    :return: Loss объект.
    """
    train_loss = losses.DenoisingAutoEncoderLoss(
        model=model,
        decoder_name_or_path=MODEL_NAME,
        tie_encoder_decoder=False
    )
    train_loss.sentence_embedding_dimension = model.get_sentence_embedding_dimension()
    return train_loss


def create_evaluator(val_dataset):
    """
    Создание evaluator'а для валидации.

    :param val_dataset: Валидационный датасет.
    :return: Evaluator объект.
    """
    val_sample = val_dataset.select(range(min(EVAL_SAMPLE_SIZE, len(val_dataset))))
    eval_examples = []

    for i, example in enumerate(val_sample):
        if i < len(val_sample) // 3:
            # Идентичные пары
            eval_examples.append(InputExample(
                texts=[example['sentence2'], example['sentence2']],
                label=1.0,
                guid=f"identical_{i}"
            ))
        elif i < 2 * len(val_sample) // 3:
            # Зашумленные-оригинальные
            eval_examples.append(InputExample(
                texts=[example['sentence1'], example['sentence2']],
                label=0.7,
                guid=f"noisy_{i}"
            ))
        else:
            # Случайные пары
            if i + 1 < len(val_sample):
                other_example = val_sample[i + 1]
                eval_examples.append(InputExample(
                    texts=[example['sentence2'], other_example['sentence2']],
                    label=0.3,
                    guid=f"random_{i}"
                ))

    print(f"Создано {len(eval_examples)} примеров для оценки")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_examples,
        name='val_evaluator'
    )
    return evaluator


def train_model(model, train_dataset, val_dataset):
    """
    Запуск обучения модели.

    :param model: SentenceTransformer модель.
    :param train_dataset: Обучающий датасет.
    :param val_dataset: Валидационный датасет.
    """
    train_loss = setup_loss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model='eval_val_evaluator_spearman_cosine',
        greater_is_better=True,
        bf16=True,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
        dataloader_drop_last=True,
        ignore_data_skip=True,
        save_total_limit=SAVE_TOTAL_LIMIT
    )

    evaluator = create_evaluator(val_dataset)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset ,
        loss=train_loss,
        evaluator=evaluator
    )

    print("Запуск обучения...")
    trainer.train()

    print("Обучение завершено успешно!")
    final_dir = os.path.join(OUTPUT_DIR, 'final')
    model.save(final_dir)
    model[0].auto_model.save_pretrained(os.path.join(final_dir, 'peft'))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(os.path.join(final_dir, 'peft'))