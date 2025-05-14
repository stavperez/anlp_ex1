import argparse
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np
import os


def load_and_prepare_data(args):
    dataset = load_dataset("glue", "mrpc")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # for the tokenizer
    def preprocess(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

    encoded_dataset = dataset.map(preprocess, batched=True)

    if args.max_train_samples != -1:
        encoded_dataset['train'] = encoded_dataset['train'].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        encoded_dataset['validation'] = encoded_dataset['validation'].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        encoded_dataset['test'] = encoded_dataset['test'].select(range(args.max_predict_samples))

    return encoded_dataset, tokenizer


def init_model():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return model


def get_training_args(args, output_dir="results"):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch", # run evaluation once at the end of an epoch
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_dir="logs",
        report_to="wandb",
        logging_steps=1,
        save_strategy="no",
    )


def compute_metrics(eval_pred):
    # metric = evaluate.load("glue", "mrpc")
    metric = evaluate.load("glue", config_name="mrpc")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(args, model, tokenized_dataset, training_args):
    data_collator = DataCollatorWithPadding(tokenizer=tokenized_dataset["tokenizer"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenized_dataset["tokenizer"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    os.makedirs("logs", exist_ok=True)
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, "
                f"lr: {args.lr}, "
                f"batch_size: {args.batch_size}, "
                f"eval_acc: {eval_results['eval_accuracy']:.4f}\n")

    trainer.save_model("saved_model")


def predict_and_save(args, tokenized_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenized_dataset["tokenizer"])
    trainer = Trainer(
        model=model,
        tokenizer=tokenized_dataset["tokenizer"],
        data_collator=data_collator,
    )

    predictions = trainer.predict(tokenized_dataset["test"])
    predicted_labels = np.argmax(predictions.predictions, axis=-1)

    with open("predictions.txt", "w") as f:
        for ex, pred in zip(tokenized_dataset["test"], predicted_labels):
            f.write(f"{ex['sentence1']}###{ex['sentence2']}###{pred}\n")



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC for paraphrase detection")

    parser.add_argument("--max_train_samples", type=int, default=-1, help="Number of training samples to use, or -1 for all.")
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="Number of validation samples to use, or -1 for all.")
    parser.add_argument("--max_predict_samples", type=int, default=-1, help="Number of test samples to use, or -1 for all.")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model for prediction.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenized_dataset, tokenizer = load_and_prepare_data(args)
    model = init_model()
    training_args = get_training_args(args)

    # Keep tokenizer for Trainer use
    tokenized_dataset["tokenizer"] = tokenizer

    if args.do_train:
        train_model(args, model, tokenized_dataset, training_args)

    if args.do_predict:
        predict_and_save(args, tokenized_dataset)


