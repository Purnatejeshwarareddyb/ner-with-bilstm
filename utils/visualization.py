"""
Visualization utilities for Legal NER BiLSTM Model
Handles entity highlighting and chart generation
"""

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class Visualizer:
    def __init__(self):
        self.entity_colors = {
            'LAW': '#3498db',  # Blue
            'PERSON': '#2ecc71',  # Green
            'ORG': '#f39c12',  # Orange
            'DATE': '#e74c3c',  # Red
            'CASE': '#9b59b6'  # Purple
        }

    def get_entity_color(self, entity_type):
        """Get color for entity type"""
        return self.entity_colors.get(entity_type, '#95a5a6')

    def highlight_entities(self, text, entities):
        """Create color-coded entity highlighting for GUI"""
        highlighted = []
        last_pos = 0

        for entity in entities:
            entity_text = entity['entity']
            entity_type = entity['type']
            start = text.find(entity_text, last_pos)

            if start != -1:
                # Add text before entity
                if start > last_pos:
                    highlighted.append(('text', text[last_pos:start]))

                # Add entity with color
                highlighted.append((entity_type, entity_text))
                last_pos = start + len(entity_text)

        # Add remaining text
        if last_pos < len(text):
            highlighted.append(('text', text[last_pos:]))

        return highlighted

    def create_annotated_text(self, text, entities):
        """Create bracket-annotated text output"""
        result = text
        offset = 0

        for entity in entities:
            entity_text = entity['entity']
            entity_type = entity['type']
            start = text.find(entity_text)

            if start != -1:
                annotation = f"[{entity_type}: {entity_text}]"
                result = result[:start + offset] + annotation + result[start + offset + len(entity_text):]
                offset += len(annotation) - len(entity_text)

        return result

    def plot_entity_distribution(self, entities, save_path=None):
        """Create bar chart of entity distribution"""
        if not entities:
            return None

        entity_types = [e['type'] for e in entities]
        counter = Counter(entity_types)

        fig, ax = plt.subplots(figsize=(10, 6))

        types = list(counter.keys())
        counts = list(counter.values())
        colors = [self.entity_colors.get(t, '#95a5a6') for t in types]

        ax.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Entity Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for i, (t, c) in enumerate(zip(types, counts)):
            ax.text(i, c + 0.1, str(c), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_metrics_comparison(self, metrics, save_path=None):
        """Create bar chart of model metrics"""
        fig, ax = plt.subplots(figsize=(8, 6))

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ]

        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_confusion_matrix(self, y_true, y_pred, tag_names, save_path=None):
        """Create confusion matrix visualization"""
        from sklearn.metrics import confusion_matrix

        # Flatten arrays
        y_true_flat = []
        y_pred_flat = []

        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_tag, pred_tag in zip(true_seq, pred_seq):
                if true_tag != 0:  # Ignore padding
                    y_true_flat.append(true_tag)
                    y_pred_flat.append(pred_tag)

        cm = confusion_matrix(y_true_flat, y_pred_flat)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tag_names, yticklabels=tag_names,
                    cbar_kws={'label': 'Count'}, ax=ax)

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig