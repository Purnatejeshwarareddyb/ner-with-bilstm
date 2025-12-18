"""
Training Data Generator for Legal NER BiLSTM Model
Generates 400 training samples and 100 test samples in IOB format
"""

import random
import os

# Entity templates
COURTS = [
    "Supreme Court of India", "Delhi High Court", "Bombay High Court",
    "Madras High Court", "Calcutta High Court", "Karnataka High Court",
    "Allahabad High Court", "Gujarat High Court", "Rajasthan High Court",
    "Kerala High Court", "Punjab and Haryana High Court"
]

JUDGES = [
    "Justice Ramesh Kumar", "Justice Priya Sharma", "Justice Amit Patel",
    "Justice Neha Verma", "Justice Rajiv Malhotra", "Justice Sunita Singh",
    "Justice Anil Gupta", "Justice Kavita Rao", "Justice Vikram Desai",
    "Justice Meera Reddy"
]

LAWYERS = [
    "Advocate Ravi Menon", "Advocate Anjali Khanna", "Senior Advocate Suresh Kumar",
    "Advocate Deepak Singh", "Advocate Pooja Agarwal", "Advocate Karthik Iyer",
    "Senior Advocate Sanjay Dubey", "Advocate Nisha Kapoor"
]

LAWS = [
    "Section 420 of the Indian Penal Code", "Article 21 of the Constitution of India",
    "Section 302 IPC", "Article 14", "Section 498A IPC", "Article 32",
    "Section 376 IPC", "Companies Act 2013", "Motor Vehicles Act 1988",
    "Information Technology Act 2000", "Consumer Protection Act 2019"
]

DATES = [
    "15th March 2024", "22nd January 2023", "10th September 2022",
    "5th May 2024", "18th November 2023", "3rd February 2024",
    "25th August 2023", "12th December 2022", "7th April 2024",
    "20th October 2023"
]

CASES = [
    "Case No. 12345/2023", "Writ Petition No. 5678/2024",
    "Criminal Appeal No. 9876/2022", "Civil Appeal No. 3456/2023",
    "Special Leave Petition No. 7890/2024", "Case No. ABC/2023",
    "PIL No. 1234/2022", "Criminal Writ No. 4567/2024"
]

# Sentence templates
SENTENCE_TEMPLATES = [
    "{court} delivered its judgment on {date}.",
    "{judge} presided over the hearing.",
    "The petitioner cited {law} in the arguments.",
    "{lawyer} represented the defendant.",
    "The matter was listed before {court}.",
    "{judge} observed the legal provisions.",
    "Reference was made to {case}.",
    "The court examined {law} in detail.",
    "{judge} heard the matter on {date}.",
    "The counsel relied on {law}.",
    "{court} took cognizance of the petition.",
    "{lawyer} submitted written arguments.",
    "The case was filed as {case}.",
    "Proceedings commenced on {date}.",
    "{judge} reserved the judgment.",
]


def tag_entity(text, entity_type):
    """Convert entity text to IOB tagged format"""
    words = text.split()
    tagged = []

    for i, word in enumerate(words):
        if i == 0:
            tagged.append((word, f"B-{entity_type}"))
        else:
            tagged.append((word, f"I-{entity_type}"))

    return tagged


def generate_sentence():
    """Generate a random legal sentence with entities"""
    template = random.choice(SENTENCE_TEMPLATES)

    # Prepare entities
    entities = {
        'court': random.choice(COURTS),
        'judge': random.choice(JUDGES),
        'lawyer': random.choice(LAWYERS),
        'law': random.choice(LAWS),
        'date': random.choice(DATES),
        'case': random.choice(CASES)
    }

    # Find which entities are in the template
    used_entities = {}
    for key, value in entities.items():
        if f"{{{key}}}" in template:
            used_entities[key] = value

    # Generate sentence
    sentence = template.format(**entities)

    # Convert to IOB format
    result = []
    current_pos = 0

    # Process each entity in order
    entity_map = {
        'court': 'ORG',
        'judge': 'PERSON',
        'lawyer': 'PERSON',
        'law': 'LAW',
        'date': 'DATE',
        'case': 'CASE'
    }

    # Split sentence and tag
    words = sentence.split()
    i = 0
    while i < len(words):
        word = words[i]

        # Check if this starts an entity
        entity_found = False
        for ent_key, ent_text in used_entities.items():
            ent_words = ent_text.split()
            # Check if entity starts here
            if i + len(ent_words) <= len(words):
                match = True
                for j, ent_word in enumerate(ent_words):
                    # Handle punctuation
                    check_word = words[i + j].rstrip('.,;:!?')
                    if check_word != ent_word:
                        match = False
                        break

                if match:
                    # Tag this entity
                    entity_type = entity_map[ent_key]
                    for j, ent_word in enumerate(ent_words):
                        actual_word = words[i + j]
                        if j == 0:
                            result.append((actual_word, f"B-{entity_type}"))
                        else:
                            result.append((actual_word, f"I-{entity_type}"))
                    i += len(ent_words)
                    entity_found = True
                    break

        if not entity_found:
            result.append((word, "O"))
            i += 1

    return result


def generate_dataset(num_samples):
    """Generate multiple sentences"""
    dataset = []
    for _ in range(num_samples):
        sentence = generate_sentence()
        dataset.append(sentence)
    return dataset


def save_dataset(dataset, filename):
    """Save dataset to file in IOB format"""
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in dataset:
            for word, tag in sentence:
                f.write(f"{word} {tag}\n")
            f.write("\n")  # Blank line between sentences


def main():
    """Generate training and test data"""
    print("=" * 60)
    print("Legal NER Training Data Generator")
    print("=" * 60)

    # Create data directory
    os.makedirs('data', exist_ok=True)

    print("\nGenerating training data (400 samples)...")
    train_data = generate_dataset(400)
    save_dataset(train_data, 'data/train.txt')
    print(f"✅ Training data saved to: data/train.txt")

    print("\nGenerating test data (100 samples)...")
    test_data = generate_dataset(100)
    save_dataset(test_data, 'data/test.txt')
    print(f"✅ Test data saved to: data/test.txt")

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)
    print(f"\nTotal training sentences: {len(train_data)}")
    print(f"Total test sentences: {len(test_data)}")
    print("\nYou can now run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()