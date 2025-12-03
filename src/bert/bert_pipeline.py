from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import numpy as np
import os
from logger import logger


def load_bert_tokenizer():
    logger.info("Loading BERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    logger.info("BERT tokenizer loaded successfully.")
    return tokenizer


def load_bert_model(num_labels=5):
    logger.info("Loading BERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )
    logger.info("BERT model loaded successfully.")
    return model


class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts  # The review
        self.labels = labels  # The corresponding rating
        self.tokenizer = tokenizer  # BERT tokenizer
        self.max_length = max_length  # Max length for padding/truncating

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(
            self.texts[idx]
        )  # Get the review text and assure that it is a string
        label = int(
            self.labels[idx]
        )  # Get the corresponding rating and assure that it is an integer

        encoding = self.tokenizer(  # Tokenize the review text thanks to the BERT tokenizer
            text,  # Take as an argument the review text we want to tokenize
            truncation=True,  # If the lenght of the review is longer than max_length, truncate it
            padding="max_length",  # The padding will be done up to max_length. So if a comment is shorter,
            # it will be padded with zeros
            max_length=self.max_length,  # The maximum length of the tokenized review
            return_tensors="pt",  # Return PyTorch tensors, needed for the model
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(
                0
            ),  # Map the tokenized review to input_ids
            "attention_mask": encoding["attention_mask"].squeeze(
                0
            ),  # Map the tokenized review to attention_mask, so 1 if the token
            # is not padding, 0 if it is padding
            "labels": torch.tensor(
                label, dtype=torch.long
            ),  # Map the label to a tensor
        }
        return item


def create_dataloaders(
    df,
    tokenizer,
    text_col="reviewText",
    label_col="overall",
    test_size=0.2,  # 20% of the data will be used for validation
    batch_size=16,  # How many reviews will be processed in a single batch
    max_length=128,
    random_state=1,
):
    texts = (
        df[text_col].astype(str).tolist()
    )  # We convert the reviews to a list of strings
    labels = (
        df[label_col].astype(int) - 1
    ).tolist()  # We convert the ratings to a list of integers (0-4)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )  # We split the data into training and validation sets, so we'll get for example 80% of the comments associated with their ratings for training
    # and 20% for validation

    # Next, we'll use our Class in order to create the right Dataset objects for training and validation sets that PyTorch will be able to handle
    train_dataset = ReviewsDataset(X_train, y_train, tokenizer, max_length=max_length)
    # Here, instead of having just the reviews and ratings as lists, we have them tokenized, with their attention masks, and ready to be used by BERT
    val_dataset = ReviewsDataset(X_val, y_val, tokenizer, max_length=max_length)

    # Here, we'll use the DataLoader class from PyTorch to create the dataloaders for training and validation sets. This class will take care of batching the data,
    # shuffling it, and loading it in parallel using multiprocessing workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Dataloaders created successfully.")

    return train_loader, val_loader


def train_bert_model(
    model,
    train_loader,
    val_loader,
    device=None,
    epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    show_metrics=True,
    save_path=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):  # Let's iterate over the number of epochs
        logger.info(f"Starting epoch {epoch + 1}/{epochs}...")
        model.train()  # Set the model to training mode
        total_train_loss = 0  # Initialize the total training loss for this epoch

        for (
            batch
        ) in train_loader:  # Let's iterate over each batch in the training dataloader
            optimizer.zero_grad()  # Clear previously calculated gradients

            # Get the input_ids, attention_mask, and labels from the batch and move them to the selected device (CPU or GPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)

        if show_metrics:
            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Val   loss: {avg_val_loss:.4f}")
            print(f"Val   acc : {acc:.4f}")
            print(classification_report(all_labels, all_preds, digits=4))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_pretrained(save_path)
        logger.info(f"BERT model saved to {save_path}.")

    return model
