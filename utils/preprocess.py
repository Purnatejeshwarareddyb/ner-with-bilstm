"""
Text preprocessing utilities for Legal NER BiLSTM Model
Handles data loading, tokenization, and sequence preparation
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class DataPreprocessor:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.tag2idx = {"O": 0}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.idx2tag = {0: "O"}
        self.max_len = 50

    def load_data(self, file_path):
        """Load IOB formatted data from file"""
        sentences = []
        tags = []
        current_sentence = []
        current_tags = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_sentence:
                            sentences.append(current_sentence)
                            tags.append(current_tags)
                            current_sentence = []
                            current_tags = []
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            word = parts[0]
                            tag = parts[1]
                            current_sentence.append(word)
                            current_tags.append(tag)

                # Add last sentence if exists
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)

        except FileNotFoundError:
            print(f"Error: File {file_path} not found!")
            return [], []

        return sentences, tags

    def build_vocabulary(self, sentences, tags):
        """Build word and tag vocabularies"""
        # Build word vocabulary
        word_idx = 2  # Start after PAD and UNK
        for sentence in sentences:
            for word in sentence:
                if word not in self.word2idx:
                    self.word2idx[word] = word_idx
                    self.idx2word[word_idx] = word
                    word_idx += 1

        # Build tag vocabulary
        tag_idx = 1  # Start after O
        for tag_seq in tags:
            for tag in tag_seq:
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = tag_idx
                    self.idx2tag[tag_idx] = tag
                    tag_idx += 1

        return len(self.word2idx), len(self.tag2idx)

    def encode_data(self, sentences, tags):
        """Convert words and tags to numerical indices"""
        X = [[self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence]
             for sentence in sentences]
        y = [[self.tag2idx[tag] for tag in tag_seq] for tag_seq in tags]

        return X, y

    def pad_data(self, X, y):
        """Pad sequences to max length"""
        X_padded = pad_sequences(X, maxlen=self.max_len, padding='post', value=self.word2idx["<PAD>"])
        y_padded = pad_sequences(y, maxlen=self.max_len, padding='post', value=self.tag2idx["O"])

        return X_padded, y_padded

    def prepare_data(self, train_path, test_path):
        """Complete data preparation pipeline"""
        print("Loading training data...")
        train_sentences, train_tags = self.load_data(train_path)

        print("Loading test data...")
        test_sentences, test_tags = self.load_data(test_path)

        print("Building vocabulary...")
        vocab_size, num_tags = self.build_vocabulary(train_sentences + test_sentences,
                                                     train_tags + test_tags)

        print("Encoding data...")
        X_train, y_train = self.encode_data(train_sentences, train_tags)
        X_test, y_test = self.encode_data(test_sentences, test_tags)

        print("Padding sequences...")
        X_train, y_train = self.pad_data(X_train, y_train)
        X_test, y_test = self.pad_data(X_test, y_test)

        # Convert to categorical for training
        y_train_cat = np.array([to_categorical(seq, num_classes=num_tags) for seq in y_train])
        y_test_cat = np.array([to_categorical(seq, num_classes=num_tags) for seq in y_test])

        print(f"\nData prepared successfully!")
        print(f"Vocabulary size: {vocab_size}")
        print(f"Number of tags: {num_tags}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Max sequence length: {self.max_len}")

        return (X_train, y_train_cat, X_test, y_test_cat,
                vocab_size, num_tags, train_sentences, test_sentences,
                train_tags, test_tags)

    def predict_sentence(self, sentence):
        """Prepare single sentence for prediction"""
        words = sentence.split()
        X = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        X_padded = pad_sequences([X], maxlen=self.max_len, padding='post',
                                 value=self.word2idx["<PAD>"])
        return X_padded, words