"""
Evaluation metrics for Legal NER BiLSTM Model
Calculates accuracy, precision, recall, and F1 scores
"""

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


class MetricsCalculator:
    def __init__(self, idx2tag):
        self.idx2tag = idx2tag

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        # Flatten predictions and true labels
        y_true_flat = []
        y_pred_flat = []

        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_tag, pred_tag in zip(true_seq, pred_seq):
                if true_tag != 0:  # Ignore padding
                    y_true_flat.append(true_tag)
                    y_pred_flat.append(pred_tag)

        # Calculate overall metrics
        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def per_entity_metrics(self, y_true, y_pred):
        """Calculate per-entity type metrics"""
        # Flatten and convert to tag names
        y_true_flat = []
        y_pred_flat = []

        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_tag, pred_tag in zip(true_seq, pred_seq):
                if true_tag != 0:  # Ignore padding
                    y_true_flat.append(self.idx2tag[true_tag])
                    y_pred_flat.append(self.idx2tag[pred_tag])

        # Get unique tags
        tags = sorted(list(set(y_true_flat + y_pred_flat)))

        # Calculate per-tag metrics
        entity_metrics = {}
        for tag in tags:
            if tag != 'O':
                # Extract entity type (remove B- or I- prefix)
                entity_type = tag.split('-')[1] if '-' in tag else tag

                if entity_type not in entity_metrics:
                    # Get all tags for this entity type
                    entity_tags = [t for t in tags if entity_type in t]

                    # Calculate metrics for this entity
                    y_true_binary = [1 if t in entity_tags else 0 for t in y_true_flat]
                    y_pred_binary = [1 if t in entity_tags else 0 for t in y_pred_flat]

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true_binary, y_pred_binary, average='binary', zero_division=0
                    )

                    entity_metrics[entity_type] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }

        return entity_metrics

    def print_metrics(self, metrics, entity_metrics):
        """Print formatted metrics report"""
        print("\n" + "=" * 50)
        print("Model: BiLSTM")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall:    {metrics['recall']:.2f}")
        print(f"F1-Score:  {metrics['f1']:.2f}")
        print("=" * 50)

        if entity_metrics:
            print("\nPer-Entity Performance:")
            print(f"{'Entity':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
            print("-" * 50)
            for entity, scores in entity_metrics.items():
                print(f"{entity:<12} {scores['precision']:<12.2f} "
                      f"{scores['recall']:<12.2f} {scores['f1']:<12.2f}")
        print("=" * 50 + "\n")

    def get_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        y_true_flat = []
        y_pred_flat = []

        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_tag, pred_tag in zip(true_seq, pred_seq):
                if true_tag != 0:  # Ignore padding
                    y_true_flat.append(self.idx2tag[true_tag])
                    y_pred_flat.append(self.idx2tag[pred_tag])

        return classification_report(y_true_flat, y_pred_flat, zero_division=0)