"""
BiLSTM Model Architecture for Legal NER
Bidirectional LSTM with embedding layer for sequence labeling
"""

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense,
    TimeDistributed, Dropout, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


class BiLSTMModel:
    def __init__(self, vocab_size, num_tags, max_len=50, embedding_dim=128, lstm_units=64):
        """
        Initialize BiLSTM model for NER

        Args:
            vocab_size: Size of vocabulary
            num_tags: Number of NER tags
            max_len: Maximum sequence length
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of LSTM units
        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None

    def build_model(self):
        """Build BiLSTM architecture"""
        print("\nBuilding BiLSTM model...")

        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                mask_zero=True,
                name='embedding'
            ),

            # Spatial dropout for embedding
            SpatialDropout1D(0.2, name='spatial_dropout'),

            # First BiLSTM layer
            Bidirectional(
                LSTM(
                    self.lstm_units,
                    return_sequences=True,
                    dropout=0.3,
                    recurrent_dropout=0.3,
                    name='lstm_1'
                ),
                name='bidirectional_1'
            ),

            # Second BiLSTM layer
            Bidirectional(
                LSTM(
                    self.lstm_units // 2,
                    return_sequences=True,
                    dropout=0.3,
                    recurrent_dropout=0.3,
                    name='lstm_2'
                ),
                name='bidirectional_2'
            ),

            # Output layer with softmax for each time step
            TimeDistributed(
                Dense(self.num_tags, activation='softmax', name='dense'),
                name='time_distributed'
            )
        ])

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        print("\nModel Architecture:")
        print("=" * 60)
        model.summary()
        print("=" * 60)

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32,
              model_path='outputs/bilstm_model.h5'):
        """
        Train the BiLSTM model

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_path: Path to save best model
        """
        if self.model is None:
            self.build_model()

        print(f"\nTraining BiLSTM model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}\n")

        # Create outputs directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        print(f"\nTraining completed!")
        print(f"Best model saved to: {model_path}")

        return history

    def predict(self, X):
        """
        Make predictions on input sequences

        Args:
            X: Input sequences

        Returns:
            Predicted tag indices for each token
        """
        if self.model is None:
            raise ValueError("Model not built or loaded!")

        predictions = self.model.predict(X, verbose=0)
        pred_tags = np.argmax(predictions, axis=-1)

        return pred_tags

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data

        Args:
            X_test: Test sequences
            y_test: Test labels

        Returns:
            Loss and accuracy
        """
        if self.model is None:
            raise ValueError("Model not built or loaded!")

        print("\nEvaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        return loss, accuracy

    def save_model(self, path='outputs/bilstm_model.h5'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save!")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"\nModel saved to: {path}")

    def load_model(self, path='outputs/bilstm_model.h5'):
        """Load pre-trained model"""
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"\nModel loaded from: {path}")
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    def predict_sentence(self, X, idx2tag, words):
        """
        Predict entities in a single sentence

        Args:
            X: Encoded and padded sentence
            idx2tag: Tag index to tag name mapping
            words: Original words in sentence

        Returns:
            List of (word, tag) tuples
        """
        pred_tags = self.predict(X)[0]

        result = []
        for i, word in enumerate(words):
            if i < len(pred_tags):
                tag = idx2tag[pred_tags[i]]
                result.append((word, tag))

        return result

    def extract_entities(self, predictions, words):
        """
        Extract entity spans from predictions

        Args:
            predictions: List of (word, tag) tuples
            words: Original words

        Returns:
            List of entity dictionaries
        """
        entities = []
        current_entity = []
        current_type = None

        for word, tag in predictions:
            if tag.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    entities.append({
                        'entity': ' '.join(current_entity),
                        'type': current_type
                    })

                # Start new entity
                current_entity = [word]
                current_type = tag.split('-')[1]

            elif tag.startswith('I-') and current_type:
                # Continue current entity
                entity_type = tag.split('-')[1]
                if entity_type == current_type:
                    current_entity.append(word)
                else:
                    # Type mismatch, save and start new
                    if current_entity:
                        entities.append({
                            'entity': ' '.join(current_entity),
                            'type': current_type
                        })
                    current_entity = [word]
                    current_type = entity_type

            else:
                # End of entity
                if current_entity:
                    entities.append({
                        'entity': ' '.join(current_entity),
                        'type': current_type
                    })
                current_entity = []
                current_type = None

        # Add last entity if exists
        if current_entity:
            entities.append({
                'entity': ' '.join(current_entity),
                'type': current_type
            })

        return entities