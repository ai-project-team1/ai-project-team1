import json
import random
import re

import pandas as pd
import torch
from emoji import demojize, replace_emoji
from rich import print
from torch.utils.data import Dataset


class MBTIDataset(Dataset):
    MBTI_INDEX_MAP = {
        "ESTJ": 0,
        "ESTP": 1,
        "ESFJ": 2,
        "ESFP": 3,
        "ENTJ": 4,
        "ENTP": 5,
        "ENFJ": 6,
        "ENFP": 7,
        "ISTJ": 8,
        "ISTP": 9,
        "ISFJ": 10,
        "ISFP": 11,
        "INTJ": 12,
        "INTP": 13,
        "INFJ": 14,
        "INFP": 15,
    }  # Maps MBTI to integer index

    MBTI_LABEL_MAP = {
        "ESTJ": torch.tensor([0, 0, 0, 0], dtype=torch.float),
        "ESTP": torch.tensor([0, 0, 0, 1], dtype=torch.float),
        "ESFJ": torch.tensor([0, 0, 1, 0], dtype=torch.float),
        "ESFP": torch.tensor([0, 0, 1, 1], dtype=torch.float),
        "ENTJ": torch.tensor([0, 1, 0, 0], dtype=torch.float),
        "ENTP": torch.tensor([0, 1, 0, 1], dtype=torch.float),
        "ENFJ": torch.tensor([0, 1, 1, 0], dtype=torch.float),
        "ENFP": torch.tensor([0, 1, 1, 1], dtype=torch.float),
        "ISTJ": torch.tensor([1, 0, 0, 0], dtype=torch.float),
        "ISTP": torch.tensor([1, 0, 0, 1], dtype=torch.float),
        "ISFJ": torch.tensor([1, 0, 1, 0], dtype=torch.float),
        "ISFP": torch.tensor([1, 0, 1, 1], dtype=torch.float),
        "INTJ": torch.tensor([1, 1, 0, 0], dtype=torch.float),
        "INTP": torch.tensor([1, 1, 0, 1], dtype=torch.float),
        "INFJ": torch.tensor([1, 1, 1, 0], dtype=torch.float),
        "INFP": torch.tensor([1, 1, 1, 1], dtype=torch.float),
    }  # Maps MBTI to one-hot vector

    def __init__(
        self,
        file_path: str,
        urls_to_titles_path: str,
        min_len: int = 16,
        tweets_per_row: int = 5,
        exclude_yt_title: bool = False,
        exclude_emoji: bool = False,
        exclude_demojize: bool = False,
    ):
        """
        MBTI Dataset

        :param file_path: path to dataset csv
        :param urls_to_titles_path: path to urls_to_titles json which maps YouTube URLs to their titles
        :param min_len: minimum length of a post
        :param tweets_per_row: number of tweets per row
        :param exclude_yt_title: whether to exclude converting YouTube URLs to their titles
        :param exclude_emoji: whether to exclude emojis
        :param exclude_demojize: whether to exclude demojizing emojis and use the raw emojis
        """

        def url_to_title(post: str):
            """
            Converts YouTube URLs to their titles

            :param post: a single post or tweet
            :return: post with YouTube URLs replaced with their titles
            """
            urls = url_pattern.findall(post)
            for url in urls:
                if url in self.urls_to_titles:
                    post = post.replace(url, self.urls_to_titles[url])
            return post

        super().__init__()
        self._exclude_emoji = exclude_emoji
        self._exclude_demojize = exclude_demojize

        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )  # Regex pattern to match URLs

        urls_to_titles_keywords = json.load(
            open(urls_to_titles_path, "r", encoding="utf-8")
        )  # Load YouTube URLs to their titles

        self.urls_to_titles = {}
        for key, value in urls_to_titles_keywords.items():
            if value is None:
                # Skip if no title
                continue
            self.urls_to_titles[key] = value["title"].replace("- YouTube", "").strip()

        # Read CSV file
        self.df = pd.read_csv(file_path)

        if not exclude_yt_title:
            # Convert YouTube URLs to their titles
            self.df["posts"] = self.df["posts"].apply(url_to_title)

        # Split posts by "|||" and remove posts with length less than min_len
        self.df["posts"] = self.df["posts"].apply(
            lambda x: list(
                filter(
                    lambda y: len(y) > min_len,
                    [self._preprocess(post) for post in x.split("|||")],
                )
            )
        )

        self.dataset = []
        for user_id, mbti, posts in self.df.itertuples():
            random.shuffle(posts)
            for i in range(0, len(posts), tweets_per_row):
                # Join posts with " " and append to dataset
                self.dataset.append(
                    (user_id, mbti, " ".join(posts[i : i + tweets_per_row]))
                )

        # Print first item
        print(self.dataset[0])

    def __getitem__(self, index):
        return (
            self.dataset[index][0],  # user_id
            self.MBTI_LABEL_MAP[self.dataset[index][1]],  # MBTI one-hot vector
            self.MBTI_INDEX_MAP[self.dataset[index][1]],  # MBTI integer index
            self.dataset[index][1],  # MBTI string
            self.dataset[index][2],  # posts
        )

    def __len__(self):
        return len(self.dataset)

    def _preprocess(self, x: str):
        """
        Preprocesses a single post or tweet

        :param x: a single post or tweet
        :return: preprocessed post or tweet
        """
        # Remove URLs
        x = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            x,
        )

        # Replace some invisible Unicode characters
        x = x.replace("\u2060", "").strip()

        # Remove mentions
        x = re.sub(r"@\w+", "", x).strip()

        # Replace curly quotes
        x = x.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

        # Replace ellipsis
        x = x.replace("…", "...")

        # Replace em dash
        x = x.replace("—", "-")

        # Replace en dash
        x = x.replace("–", "-")

        # Replace 《 》
        x = x.replace("《", "<").replace("》", ">")

        # Replace 〈 〉
        x = x.replace("〈", "<").replace("〉", ">")

        # Replace 〔 〕
        x = x.replace("〔", "[").replace("〕", "]")

        # Replace 〖 〗
        x = x.replace("〖", "[").replace("〗", "]")

        # Replace 〘 〙
        x = x.replace("〘", "[").replace("〙", "]")

        # Replace 〚 〛
        x = x.replace("〚", "[").replace("〛", "]")

        # Replace 「 」
        x = x.replace("「", '"').replace("」", '"')

        # Replace 『 』
        x = x.replace("『", '"').replace("』", '"')

        # Replace 【 】
        x = x.replace("【", "[").replace("】", "]")

        # Remove punctuations repeated twice or more
        x = re.sub(r"([@#$%^&*\-=+\\|/]){2,}", r"", x).strip()

        # Replace full stops repeated four times or more with three full stops
        x = re.sub(r"\.{4,}", "...", x).strip()

        # Replace question marks and exclamation marks repeated twice or more with one question mark
        x = re.sub(r"([?!]){2,}", r"\1", x).strip()

        # Replace \n with space
        x = re.sub(r"\n", " ", x).strip()

        # Remove emojis
        if not self._exclude_emoji:
            x = replace_emoji(x, "")

        if not self._exclude_demojize:
            x = demojize(x, delimiters=("", "")).replace("_", " ")

        # Replace multiple spaces with one space
        x = re.sub(r"\s+", " ", x).strip()

        return x
