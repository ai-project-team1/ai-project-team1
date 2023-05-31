import argparse
import json
import os
import random
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from rich.table import Table
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizerFast

from dataset import MBTIDataset


def parse_args():
    parser = argparse.ArgumentParser()

    ablation = parser.add_argument_group("Ablation")
    ablation.add_argument(
        "--exclude_cl_loss", action="store_true", help="Exclude CL loss"
    )
    ablation.add_argument(
        "--exclude_yt_title", action="store_true", help="Exclude YouTube title"
    )
    ablation.add_argument("--exclude_emoji", action="store_true", help="Exclude emoji")
    ablation.add_argument(
        "--exclude_demojize", action="store_true", help="Exclude demojize"
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model_path", type=str, default="bert-base-uncased", help="Model name"
    )

    data = parser.add_argument_group("Data")
    data.add_argument("--data_path", type=str, default="mbti_1.csv", help="Data path")
    data.add_argument(
        "--urls_to_titles_keywords_path",
        type=str,
        default="urls_to_titles_keywords.json",
        help="URLs to titles path (for YouTube titles)",
    )
    data.add_argument(
        "--min_len", type=int, default=16, help="Minimum number of tokens per tweet"
    )
    data.add_argument(
        "--tweets_per_row", type=int, default=5, help="Number of tweets per row"
    )
    data.add_argument(
        "--dataset_split_seed", type=int, default=42, help="Dataset split seed"
    )

    training = parser.add_argument_group("Training")
    training.add_argument(
        "--alpha", type=float, default=0.01, help="Weight for CL loss"
    )
    training.add_argument("--batch_size", type=int, default=32, help="Batch size")
    training.add_argument(
        "--classifier_dropout", type=float, default=0.1, help="Classifier dropout"
    )
    training.add_argument(
        "--lr_model", type=float, default=1e-5, help="Model learning rate"
    )
    training.add_argument(
        "--lr_classifier", type=float, default=1e-4, help="Classifier learning rate"
    )
    training.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    training.add_argument("--seed", type=int, default=42, help="Random seed")
    training.add_argument("--device", type=torch.device, default="cuda", help="Device")
    training.add_argument(
        "--model_checkpoint_path",
        type=str,
        help="Model checkpoint directory path",
        default="checkpoints",
    )

    parsed = parser.parse_args()

    # Check
    if parsed.exclude_demojize and parsed.exclude_emoji:
        raise ValueError("Cannot exclude both demojize and emoji")

    return parsed


def seed_everything(seed: int):
    """
    Set random seed for reproducibility

    :param seed: seed
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU


def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute accuracy score

    :param y_true: ground truth labels (batch_size, 4)
    :param y_pred: predicted labels (batch_size, 4)
    :return: accuracy score (All MBTIs correct, E/I correct, S/N correct, T/F correct, J/P correct)
    """
    # Accurate if the entire row is identical
    correct = torch.eq(y_true, y_pred)
    return (
        correct.all(dim=1).tolist(),
        correct[:, 0].tolist(),
        correct[:, 1].tolist(),
        correct[:, 2].tolist(),
        correct[:, 3].tolist(),
    )


def contrastive_loss(embeddings, labels):
    """
    Compute contrastive loss
    The loss is computed by:
    * Attract CLS vector embeddings with the same MBTI
    * Repel CLS vector embeddings with different MBTI
    Attracting and repelling are computed by pairwise distance between two embeddings
    by reducing the distance between embeddings with the same MBTI and increasing the distance between embeddings
    with different MBTI

    :param embeddings: (batch_size, embedding_size)
    :param labels: (batch_size) 0 ~ 15
    :return: contrastive loss
    """
    if len(embeddings.shape) == 3:
        embeddings = embeddings.squeeze()

    batch_size = embeddings.shape[0]

    # Normalize embeddings to unit length
    embeddings = F.normalize(embeddings, p=2, dim=1)

    hidden1, hidden2 = torch.chunk(embeddings, 2, 0)
    size1 = hidden1.size(0)
    size2 = hidden2.size(0)

    loss = torch.tensor(0.0, device=embeddings.device)
    temp = 0.5
    LARGE_NUM = 1e9

    labels1 = torch.eye(size1).to(device=embeddings.device)
    masks1 = torch.eye(size1).to(device=embeddings.device) * LARGE_NUM
    labels2 = torch.eye(size2).to(device=embeddings.device)
    masks2 = torch.eye(size2).to(device=embeddings.device) * LARGE_NUM

    hidden1_large = hidden1
    hidden2_large = hidden2
    logits_aa = torch.mm(hidden1, hidden1_large.t()) / temp
    logits_aa = logits_aa - masks1
    logits_bb = torch.mm(hidden2, hidden2_large.t()) / temp
    logits_bb = logits_bb - masks2
    logits_ab = torch.mm(hidden1, hidden2_large.t()) / temp
    logits_ba = torch.mm(hidden2, hidden1_large.t()) / temp

    loss_a = F.cross_entropy(
        torch.cat([logits_ab, logits_aa], dim=1), labels1.argmax(dim=1)
    )
    loss_b = F.cross_entropy(
        torch.cat([logits_ba, logits_bb], dim=1), labels2.argmax(dim=1)
    )
    loss = loss_a + loss_b

    return loss / (batch_size * batch_size)


def contrastive_loss_multi_label(embeddings, labels):
    """
    Compute contrastive loss
    The loss is computed by:
    * Attract CLS vector embeddings with the same MBTI
    * Repel CLS vector embeddings with different MBTI
    Attracting and repelling are computed by pairwise distance between two embeddings
    by reducing the distance between embeddings with the same MBTI and increasing the distance between embeddings
    with different MBTI

    :param embeddings: (batch_size, embedding_size)
    :param labels: (batch_size, batch_size) 1 to attract, -1 to repel, 0 to ignore
    :return: contrastive loss
    """
    assert len(embeddings.shape) == 2, "Embeddings must be 2-dimensional"

    batch_size = embeddings.shape[0]

    # Normalize embeddings to unit length
    embeddings = F.normalize(embeddings, p=2, dim=1)  # (batch_size, embedding_size)

    # Compute pairwise distance between embeddings
    dist = torch.cdist(embeddings, embeddings)  # (batch_size, batch_size)

    # Set diagonal of labels to 0
    labels[torch.eye(batch_size, dtype=torch.bool)] = 0

    # Attract embeddings with the same label
    # Repel embeddings with different labels
    loss = torch.mul(dist, labels)  # (batch_size, batch_size)

    # Ignore 0 labels
    labels[labels == 0] = torch.nan

    loss = torch.nanmean(loss)

    return loss


def main():
    # Parse arguments
    args = parse_args()
    print("Arguments:")
    print(vars(args))

    timestamp = str(int(time()))
    print(f"Timestamp: {timestamp}")

    seed_everything(args.seed)
    print(f"Set seed: {args.seed}")

    if not os.path.exists(args.model_checkpoint_path):
        os.makedirs(args.model_checkpoint_path)

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=4,
        problem_type="multi_label_classification",
        classifier_dropout=args.classifier_dropout,
    ).to(args.device)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)

    # Different learning rate for BERT and classifier (classifier is a linear layer on top of BERT)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lr_model},
            {"params": model.classifier.parameters(), "lr": args.lr_classifier},
        ]
    )

    dataset = MBTIDataset(
        args.data_path,
        args.urls_to_titles_keywords_path,
        args.min_len,
        args.tweets_per_row,
        args.exclude_yt_title,
        args.exclude_emoji,
        args.exclude_demojize,
    )
    # dataset, _ = train_test_split(
    #     dataset, test_size = 0.9, random_state=args.dataset_split_seed
    # )
    # Split dataset into train, valid, test
    train_dataset, valid_test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=args.dataset_split_seed
    )

    valid_dataset, test_dataset = train_test_split(
        valid_test_dataset, test_size=0.5, random_state=args.dataset_split_seed
    )

    table = Table()
    for column in ["Epoch", "All acc.", "E/I acc.", "N/S acc.", "T/F acc.", "J/P acc."]:
        table.add_column(column)

    best_accuracy = 0
    train_losses = []
    try:
        for epoch in range(args.epochs):
            # Train
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )
            train_dataloader_iter = tqdm(
                train_dataloader, desc=f"Train | Epoch {epoch + 1}/{args.epochs}"
            )

            model.train()
            all_loss = []
            for (
                user_ids,
                mbti_labels,
                mbti_indices,
                mbtis,
                posts,
            ) in train_dataloader_iter:
                # Tokenize posts
                posts_tokenized = tokenizer(
                    posts, padding=True, truncation=True, return_tensors="pt"
                ).to(
                    args.device
                )  # (batch_size, max_seq_len)

                optimizer.zero_grad()

                # Forward pass
                output = model(
                    **posts_tokenized,
                    labels=mbti_labels.to(args.device),
                    output_hidden_states=True,
                )

                # Calculate cross-entropy loss for MBTI classification
                ce_loss = output.loss

                # Contrastive loss against different MBTI types
                if args.exclude_cl_loss:
                    # Contrastive loss is not used
                    cl_loss = None
                    loss = ce_loss
                else:
                    # Contrastive loss is used
                    # Use CLS vector of the last hidden state as the embedding
                    # output.hidden_states[-1]: (batch_size, max_seq_len, hidden_size)
                    # output.hidden_states[-1][:, 0, :]: (batch_size, hidden_size) [CLS] vector
                    cl_loss = contrastive_loss(
                        output.hidden_states[-1][:, 0, :], mbti_indices
                    )

                    # Weighted sum of cross-entropy loss and contrastive loss
                    loss = ce_loss + args.alpha * cl_loss

                loss.backward()
                optimizer.step()

                loss = loss.item()
                ce_loss = ce_loss.item()
                cl_loss = cl_loss.item() if cl_loss is not None else None
                all_loss.append(loss)
                train_dataloader_iter.set_postfix(
                    loss=loss, ce_loss=ce_loss, cl_loss=cl_loss
                )

            train_losses.append(all_loss)

            print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {np.mean(all_loss):.4f}")

            # Validation
            valid_dataloader = DataLoader(
                valid_dataset, batch_size=args.batch_size, shuffle=False
            )
            valid_dataloader_iter = tqdm(
                valid_dataloader, desc=f"Valid | Epoch {epoch + 1}/{args.epochs}"
            )

            model.eval()
            all_accuracy = []
            all_ei_accuracy = []
            all_sn_accuracy = []
            all_tf_accuracy = []
            all_jp_accuracy = []
            with torch.no_grad():
                for (
                    user_ids,
                    mbti_labels,
                    mbti_indices,
                    mbtis,
                    posts,
                ) in valid_dataloader_iter:
                    # Tokenize posts
                    posts_tokenized = tokenizer(
                        posts, padding=True, truncation=True, return_tensors="pt"
                    ).to(
                        args.device
                    )  # (batch_size, max_seq_len)

                    output = model(
                        **posts_tokenized,
                        labels=mbti_labels.to(args.device),
                        output_hidden_states=True,
                    )

                    # Calculate cross-entropy loss for MBTI classification
                    ce_loss = output.loss

                    if args.exclude_cl_loss:
                        # Contrastive loss is not used
                        cl_loss = None
                        loss = ce_loss
                    else:
                        # Contrastive loss is used
                        # Use CLS vector of the last hidden state as the embedding
                        # output.hidden_states[-1]: (batch_size, max_seq_len, hidden_size)
                        # output.hidden_states[-1][:, 0, :]: (batch_size, hidden_size) [CLS] vector
                        cl_loss = contrastive_loss(
                            output.hidden_states[-1][:, 0, :], mbti_indices
                        )
                        loss = ce_loss + args.alpha * cl_loss

                    prediction = torch.round(
                        torch.sigmoid(output.logits.to("cpu"))
                    ).type(torch.long)

                    accuracy, ei_acc, sn_acc, tf_acc, jp_acc = accuracy_score(
                        mbti_labels.type(torch.long),
                        prediction,
                    )
                    all_accuracy.extend(accuracy)
                    all_ei_accuracy.extend(ei_acc)
                    all_sn_accuracy.extend(sn_acc)
                    all_tf_accuracy.extend(tf_acc)
                    all_jp_accuracy.extend(jp_acc)

                    valid_dataloader_iter.set_postfix(
                        loss=loss.item(),
                        ce_loss=ce_loss.item(),
                        cl_loss=cl_loss.item() if cl_loss is not None else None,
                        accuracy=np.mean(all_accuracy),
                    )

            if np.mean(all_accuracy) > best_accuracy:
                # Save model with the best accuracy
                print(
                    f"New best accuracy: {best_accuracy:.4f} -> {np.mean(all_accuracy):.4f}"
                )
                best_accuracy = np.mean(all_accuracy)

                # Save model to file
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.model_checkpoint_path, f"model_{timestamp}_best.pth"
                    ),
                )

            table.add_row(
                f"Epoch {epoch + 1}",
                f"{np.mean(all_accuracy):.4f}",
                f"{np.mean(all_ei_accuracy):.4f}",
                f"{np.mean(all_sn_accuracy):.4f}",
                f"{np.mean(all_tf_accuracy):.4f}",
                f"{np.mean(all_jp_accuracy):.4f}",
            )
            print(table)

    except KeyboardInterrupt:
        print("Keyboard interrupt")

    # Test
    print(f"Best accuracy: {best_accuracy:.4f}")
    try:
        model.load_state_dict(
            torch.load(
                os.path.join(args.model_checkpoint_path, f"model_{timestamp}_best.pth")
            )
        )
    except FileNotFoundError:
        pass

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_dataloader_iter = tqdm(test_dataloader, desc=f"Test")

    model.eval()
    all_accuracy = []
    all_ei_accuracy = []
    all_sn_accuracy = []
    all_tf_accuracy = []
    all_jp_accuracy = []

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_ids, mbti_labels, mbti_indices, mbtis, posts in test_dataloader_iter:
            posts_tokenized = tokenizer(
                posts, padding=True, truncation=True, return_tensors="pt"
            ).to(args.device)

            output = model(
                **posts_tokenized,
                labels=mbti_labels.to(args.device),
                output_hidden_states=True,
            )

            # Calculate cross-entropy loss for MBTI classification
            ce_loss = output.loss

            if args.exclude_cl_loss:
                # Contrastive loss is not used
                cl_loss = None
                loss = ce_loss
            else:
                # Contrastive loss is used
                # Use CLS vector of the last hidden state as the embedding
                # output.hidden_states[-1]: (batch_size, max_seq_len, hidden_size)
                # output.hidden_states[-1][:, 0, :]: (batch_size, hidden_size) [CLS] vector
                cl_loss = contrastive_loss(
                    output.hidden_states[-1][:, 0, :], mbti_indices
                )
                loss = ce_loss + args.alpha * cl_loss

            prediction = torch.round(torch.sigmoid(output.logits.to("cpu"))).type(
                torch.long
            )

            predictions.extend(prediction.tolist())
            ground_truth.extend(mbti_labels.type(torch.long).tolist())

            # Calculate accuracy
            accuracy, ei_acc, sn_acc, tf_acc, jp_acc = accuracy_score(
                mbti_labels.type(torch.long),
                torch.round(torch.sigmoid(output.logits.to("cpu"))),
            )
            all_accuracy.extend(accuracy)
            all_ei_accuracy.extend(ei_acc)
            all_sn_accuracy.extend(sn_acc)
            all_tf_accuracy.extend(tf_acc)
            all_jp_accuracy.extend(jp_acc)

            test_dataloader_iter.set_postfix(
                loss=loss.item(),
                ce_loss=ce_loss.item(),
                cl_loss=cl_loss.item() if cl_loss is not None else None,
                accuracy=np.mean(all_accuracy),
            )

    table.add_row(
        "Test",
        f"{np.mean(all_accuracy):.4f}",
        f"{np.mean(all_ei_accuracy):.4f}",
        f"{np.mean(all_sn_accuracy):.4f}",
        f"{np.mean(all_tf_accuracy):.4f}",
        f"{np.mean(all_jp_accuracy):.4f}",
    )
    print(table)

    json.dump(
        {
            "train_losses": train_losses,
            "test_accuracy": np.mean(all_accuracy),
            "test_ei_accuracy": np.mean(all_ei_accuracy),
            "test_sn_accuracy": np.mean(all_sn_accuracy),
            "test_tf_accuracy": np.mean(all_tf_accuracy),
            "test_jp_accuracy": np.mean(all_jp_accuracy),
        },
        open(f"model_{timestamp}.json", "w"),
        indent=1,
        ensure_ascii=False,
    )

    # Save prediction
    json.dump(
        {
            "predictions": predictions,
            "ground_truth": ground_truth,
        },
        open(os.path.join(args.model_checkpoint_path, f"model_{timestamp}.json"), "w"),
        indent=1,
        ensure_ascii=False,
    )

    # Save model
    model.save_pretrained(
        os.path.join(args.model_checkpoint_path, f"model_{timestamp}")
    )
    tokenizer.save_pretrained(
        os.path.join(args.model_checkpoint_path, f"model_{timestamp}")
    )
    print(f"Saved model to model_{timestamp}")


if __name__ == "__main__":
    main()
