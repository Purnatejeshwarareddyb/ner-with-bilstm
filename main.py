"""
Legal Named Entity Recognition System using BiLSTM
Main application with GUI interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.bilstm_model import BiLSTMModel
from utils.preprocess import DataPreprocessor
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer


class LegalNERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚖️ Legal NER System - BiLSTM Model")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')

        # Initialize components
        self.preprocessor = None
        self.model = None
        self.visualizer = Visualizer()
        self.metrics = None
        self.current_entities = []

        # Style configuration
        self.setup_styles()

        # Create GUI
        self.create_widgets()

        # Train model on startup
        self.root.after(100, self.initialize_system)

    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Title.TLabel',
                        background='#2c3e50',
                        foreground='white',
                        font=('Arial', 16, 'bold'))

        style.configure('Subtitle.TLabel',
                        background='#34495e',
                        foreground='white',
                        font=('Arial', 12, 'bold'))

        style.configure('Info.TLabel',
                        background='#ecf0f1',
                        foreground='#2c3e50',
                        font=('Arial', 10))

        style.configure('Action.TButton',
                        font=('Arial', 11, 'bold'),
                        padding=10)

    def create_widgets(self):
        """Create main GUI components"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', pady=10)
        title_frame.pack(fill='x')

        title = ttk.Label(title_frame,
                          text="⚖️ Legal Named Entity Recognition System",
                          style='Title.TLabel')
        title.pack()

        subtitle = ttk.Label(title_frame,
                             text="BiLSTM Deep Learning Model",
                             style='Title.TLabel',
                             font=('Arial', 12))
        subtitle.pack()

        # Main container
        main_container = tk.Frame(self.root, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Left panel (Input & Controls)
        left_panel = tk.Frame(main_container, bg='#ecf0f1')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))

        # Input section
        input_frame = tk.LabelFrame(left_panel, text="📝 Input Text",
                                    bg='#34495e', fg='white',
                                    font=('Arial', 12, 'bold'), pady=10)
        input_frame.pack(fill='both', expand=True, pady=(0, 10))

        self.input_text = scrolledtext.ScrolledText(input_frame,
                                                    wrap=tk.WORD,
                                                    height=12,
                                                    font=('Arial', 11),
                                                    bg='white')
        self.input_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Sample text button
        sample_btn = tk.Button(input_frame, text="Load Sample Text",
                               command=self.load_sample_text,
                               bg='#3498db', fg='white',
                               font=('Arial', 10, 'bold'),
                               cursor='hand2')
        sample_btn.pack(pady=5)

        # Analyze button
        analyze_btn = tk.Button(left_panel, text="🔍 Analyze Text",
                                command=self.analyze_text,
                                bg='#27ae60', fg='white',
                                font=('Arial', 14, 'bold'),
                                height=2, cursor='hand2')
        analyze_btn.pack(fill='x', pady=10)

        # Output section
        output_frame = tk.LabelFrame(left_panel, text="✨ Analysis Results",
                                     bg='#34495e', fg='white',
                                     font=('Arial', 12, 'bold'), pady=10)
        output_frame.pack(fill='both', expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame,
                                                     wrap=tk.WORD,
                                                     height=12,
                                                     font=('Arial', 11),
                                                     bg='#f8f9fa')
        self.output_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Configure tags for entity colors
        self.setup_text_tags()

        # Right panel (Metrics & Info)
        right_panel = tk.Frame(main_container, bg='#ecf0f1', width=400)
        right_panel.pack(side='right', fill='both', padx=(5, 0))
        right_panel.pack_propagate(False)

        # Model status
        status_frame = tk.LabelFrame(right_panel, text="📊 Model Status",
                                     bg='#34495e', fg='white',
                                     font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', pady=(0, 10))

        self.status_label = tk.Label(status_frame, text="⏳ Initializing...",
                                     bg='#f39c12', fg='white',
                                     font=('Arial', 11, 'bold'),
                                     pady=10)
        self.status_label.pack(fill='x', padx=10, pady=10)

        # Metrics display
        metrics_frame = tk.LabelFrame(right_panel, text="📈 Performance Metrics",
                                      bg='#34495e', fg='white',
                                      font=('Arial', 12, 'bold'))
        metrics_frame.pack(fill='x', pady=(0, 10))

        self.metrics_text = tk.Text(metrics_frame, height=10,
                                    font=('Courier', 10, 'bold'),
                                    bg='#2c3e50', fg='#2ecc71',
                                    relief='flat')
        self.metrics_text.pack(fill='x', padx=10, pady=10)

        # Entity legend
        legend_frame = tk.LabelFrame(right_panel, text="🎨 Entity Types",
                                     bg='#34495e', fg='white',
                                     font=('Arial', 12, 'bold'))
        legend_frame.pack(fill='x', pady=(0, 10))

        entity_types = [
            ('LAW', '#3498db', 'Legal statutes and acts'),
            ('PERSON', '#2ecc71', 'Judges, lawyers, individuals'),
            ('ORG', '#f39c12', 'Courts and organizations'),
            ('DATE', '#e74c3c', 'Dates and time references'),
            ('CASE', '#9b59b6', 'Case names and citations')
        ]

        for entity, color, desc in entity_types:
            frame = tk.Frame(legend_frame, bg='#ecf0f1')
            frame.pack(fill='x', padx=10, pady=3)

            color_box = tk.Label(frame, text='  ', bg=color,
                                 width=3, relief='solid', borderwidth=1)
            color_box.pack(side='left', padx=(0, 10))

            label = tk.Label(frame, text=f"{entity}: {desc}",
                             bg='#ecf0f1', font=('Arial', 9),
                             anchor='w')
            label.pack(side='left', fill='x', expand=True)

        # Action buttons
        buttons_frame = tk.Frame(right_panel, bg='#ecf0f1')
        buttons_frame.pack(fill='x', pady=10)

        save_btn = tk.Button(buttons_frame, text="💾 Save Results",
                             command=self.save_results,
                             bg='#3498db', fg='white',
                             font=('Arial', 10, 'bold'),
                             cursor='hand2')
        save_btn.pack(fill='x', pady=5)

        export_btn = tk.Button(buttons_frame, text="📄 Export to JSON",
                               command=self.export_json,
                               bg='#9b59b6', fg='white',
                               font=('Arial', 10, 'bold'),
                               cursor='hand2')
        export_btn.pack(fill='x', pady=5)

        clear_btn = tk.Button(buttons_frame, text="🗑️ Clear All",
                              command=self.clear_all,
                              bg='#e74c3c', fg='white',
                              font=('Arial', 10, 'bold'),
                              cursor='hand2')
        clear_btn.pack(fill='x', pady=5)

    def setup_text_tags(self):
        """Configure text tags for entity highlighting"""
        self.output_text.tag_configure('LAW', foreground='#3498db',
                                       font=('Arial', 11, 'bold'))
        self.output_text.tag_configure('PERSON', foreground='#2ecc71',
                                       font=('Arial', 11, 'bold'))
        self.output_text.tag_configure('ORG', foreground='#f39c12',
                                       font=('Arial', 11, 'bold'))
        self.output_text.tag_configure('DATE', foreground='#e74c3c',
                                       font=('Arial', 11, 'bold'))
        self.output_text.tag_configure('CASE', foreground='#9b59b6',
                                       font=('Arial', 11, 'bold'))

    def initialize_system(self):
        """Initialize and train the model"""
        try:
            self.status_label.config(text="⏳ Loading data...", bg='#f39c12')
            self.root.update()

            # Initialize preprocessor
            self.preprocessor = DataPreprocessor()

            # Prepare data
            result = self.preprocessor.prepare_data('data/train.txt', 'data/test.txt')
            (X_train, y_train, X_test, y_test, vocab_size, num_tags,
             train_sentences, test_sentences, train_tags, test_tags) = result

            self.status_label.config(text="🔨 Building model...", bg='#f39c12')
            self.root.update()

            # Initialize model
            self.model = BiLSTMModel(vocab_size, num_tags, max_len=self.preprocessor.max_len)
            self.model.build_model()

            self.status_label.config(text="🎯 Training model...", bg='#f39c12')
            self.root.update()

            # Train model
            history = self.model.train(X_train, y_train, X_test, y_test,
                                       epochs=50, batch_size=32)

            self.status_label.config(text="📊 Evaluating...", bg='#f39c12')
            self.root.update()

            # Evaluate model
            loss, accuracy = self.model.evaluate(X_test, y_test)

            # Get predictions for metrics
            y_pred = self.model.predict(X_test)
            y_true = np.argmax(y_test, axis=-1)

            # Calculate metrics
            metrics_calc = MetricsCalculator(self.preprocessor.idx2tag)
            self.metrics = metrics_calc.calculate_metrics(y_true, y_pred)
            entity_metrics = metrics_calc.per_entity_metrics(y_true, y_pred)

            # Display metrics
            self.display_metrics(self.metrics, entity_metrics)

            # Print to console
            metrics_calc.print_metrics(self.metrics, entity_metrics)

            self.status_label.config(text="✅ Model Ready!", bg='#27ae60')

            messagebox.showinfo("Success",
                                f"Model trained successfully!\n\n"
                                f"Accuracy: {self.metrics['accuracy'] * 100:.2f}%\n"
                                f"F1 Score: {self.metrics['f1']:.2f}")

        except Exception as e:
            self.status_label.config(text="❌ Error!", bg='#e74c3c')
            messagebox.showerror("Error", f"Failed to initialize system:\n{str(e)}")
            print(f"Error: {e}")

    def display_metrics(self, metrics, entity_metrics):
        """Display metrics in the metrics panel"""
        self.metrics_text.delete('1.0', tk.END)

        text = f"""
╔════════════════════════════╗
║   MODEL PERFORMANCE        ║
╚════════════════════════════╝

Accuracy:  {metrics['accuracy'] * 100:>6.2f}%
Precision: {metrics['precision']:>6.2f}
Recall:    {metrics['recall']:>6.2f}
F1-Score:  {metrics['f1']:>6.2f}

{'=' * 30}
Per-Entity Metrics:
{'=' * 30}
"""
        self.metrics_text.insert('1.0', text)

        for entity, scores in entity_metrics.items():
            entity_text = f"\n{entity:>8}: F1={scores['f1']:.2f}"
            self.metrics_text.insert(tk.END, entity_text)

    def load_sample_text(self):
        """Load sample legal text"""
        sample = """The Supreme Court of India delivered its judgment on 15th March 2024. Justice Ramesh Kumar presided over the case. The petitioner cited Section 420 of the Indian Penal Code. The Delhi High Court had previously ruled on this matter in Case No. 12345/2023. The defendant, represented by Advocate Priya Sharma, argued that the provisions of the Constitution of India were violated."""

        self.input_text.delete('1.0', tk.END)
        self.input_text.insert('1.0', sample)

    def analyze_text(self):
        """Analyze input text for entities"""
        if self.model is None:
            messagebox.showwarning("Warning", "Model not ready! Please wait...")
            return

        text = self.input_text.get('1.0', tk.END).strip()

        if not text:
            messagebox.showwarning("Warning", "Please enter text to analyze!")
            return

        try:
            # Prepare text
            X, words = self.preprocessor.predict_sentence(text)

            # Predict
            predictions = self.model.predict_sentence(X, self.preprocessor.idx2tag, words)

            # Extract entities
            self.current_entities = self.model.extract_entities(predictions, words)

            # Display results
            self.display_results(text, self.current_entities)

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            print(f"Error: {e}")

    def display_results(self, text, entities):
        """Display annotated text with entity highlighting"""
        self.output_text.delete('1.0', tk.END)

        if not entities:
            self.output_text.insert('1.0', "No entities found in the text.")
            return

        # Highlight entities
        highlighted = self.visualizer.highlight_entities(text, entities)

        for item_type, item_text in highlighted:
            if item_type == 'text':
                self.output_text.insert(tk.END, item_text)
            else:
                self.output_text.insert(tk.END, item_text, item_type)

        # Add entity list
        self.output_text.insert(tk.END, "\n\n" + "=" * 50 + "\n")
        self.output_text.insert(tk.END, "Detected Entities:\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n\n")

        for i, entity in enumerate(entities, 1):
            entity_line = f"{i}. [{entity['type']}] {entity['entity']}\n"
            self.output_text.insert(tk.END, entity_line)

    def save_results(self):
        """Save annotated results to text file"""
        if not self.current_entities:
            messagebox.showwarning("Warning", "No results to save!")
            return

        text = self.input_text.get('1.0', tk.END).strip()
        annotated = self.visualizer.create_annotated_text(text, self.current_entities)

        os.makedirs('outputs', exist_ok=True)
        filepath = 'outputs/annotated_output.txt'

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(annotated)
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("Detected Entities:\n")
            f.write("=" * 50 + "\n\n")
            for i, entity in enumerate(self.current_entities, 1):
                f.write(f"{i}. [{entity['type']}] {entity['entity']}\n")

        messagebox.showinfo("Success", f"Results saved to:\n{filepath}")

    def export_json(self):
        """Export results to JSON format"""
        if not self.current_entities:
            messagebox.showwarning("Warning", "No results to export!")
            return

        text = self.input_text.get('1.0', tk.END).strip()

        output = {
            'text': text,
            'entities': self.current_entities,
            'metrics': self.metrics if self.metrics else {}
        }

        os.makedirs('outputs', exist_ok=True)
        filepath = 'outputs/results.json'

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        messagebox.showinfo("Success", f"Results exported to:\n{filepath}")

    def clear_all(self):
        """Clear all text fields"""
        self.input_text.delete('1.0', tk.END)
        self.output_text.delete('1.0', tk.END)
        self.current_entities = []


def main():
    """Main entry point"""
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Check if data files exist
    if not os.path.exists('data/train.txt') or not os.path.exists('data/test.txt'):
        print("\n" + "=" * 60)
        print("ERROR: Training data not found!")
        print("=" * 60)
        print("\nPlease create the following files:")
        print("  - data/train.txt (400 training samples)")
        print("  - data/test.txt (100 test samples)")
        print("\nFormat: IOB tagging (word tag per line, blank line between sentences)")
        print("=" * 60 + "\n")
        return

    # Launch GUI
    root = tk.Tk()
    app = LegalNERApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()