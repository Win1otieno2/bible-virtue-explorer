import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import numpy as np

# --- Setup ---
nltk.download('stopwords')

# --- Step 1: Load the KJV text file ---
with open('kjv.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# --- Step 2: Parse into (book, verse) pairs ---
data = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    if '\t' in line:
        parts = line.split('\t', 1)
    else:
        parts = re.split(r'\s(?=\d)', line, 1)
    if len(parts) == 2:
        ref, verse = parts
        book_match = re.match(r'^[1-3]?\s?[A-Za-z]+', ref)
        if book_match:
            book = book_match.group(0).strip()
            data.append((book, verse.strip()))

df = pd.DataFrame(data, columns=["book", "verse"])
print(f"Parsed {len(df)} verses from {df['book'].nunique()} books.")

# --- Step 3: Combine all verses by book ---
book_texts = df.groupby('book')['verse'].apply(lambda v: ' '.join(v)).to_dict()

# --- Step 4: Preprocess and count words ---
stop_words = set(stopwords.words('english'))
book_word_counts = {}

for book, text in book_texts.items():
    words = re.findall(r'\b[a-z]+\b', text.lower())
    words = [w for w in words if w not in stop_words]
    book_word_counts[book] = Counter(words)

# --- Step 5: Virtue / Fruit-of-the-Spirit definitions ---
VIRTUE_KEYWORDS = {
    "love": ["love", "loved", "loveth", "loving"],
    "joy": ["joy", "rejoice", "glad", "gladness"],
    "peace": ["peace", "peaceable", "peaceful"],
    "faith": ["faith", "believe", "belief", "faithful"],
    "hope": ["hope", "hopeful", "hopeth"],
    "patience": ["patience", "patient", "endurance", "longsuffering"],
    "kindness": ["kind", "kindness", "merciful", "mercy"],
    "goodness": ["good", "goodness", "righteous", "righteousness"],
    "gentleness": ["gentle", "meek", "meekness"],
    "self_control": ["temperance", "selfcontrol", "sober", "sobriety"],
    "fear": ["fear", "afraid", "dread"],
    "sin": ["sin", "sins", "sinner", "iniquity", "transgression"]
}

# --- Step 6: Ask user for books to compare ---
books_input = input("\nEnter books to compare (comma separated, e.g., Psalm, Romans, John): ")
books_to_compare = [b.strip().title() for b in books_input.split(',') if b.strip()]

missing = [b for b in books_to_compare if b not in book_word_counts]
if missing:
    print(f"\nBooks not found: {missing}")
    print(f"Available books: {', '.join(sorted(book_word_counts.keys()))}")
    exit()

# --- Step 7: Compute virtue frequencies per book ---
virtue_df = pd.DataFrame()

for book in books_to_compare:
    word_freq = book_word_counts[book]
    virtue_counts = {}
    for virtue, keywords in VIRTUE_KEYWORDS.items():
        virtue_counts[virtue] = sum(word_freq.get(k, 0) for k in keywords)
    total_words = sum(word_freq.values())
    virtue_freq = {k: v / total_words * 1000 for k, v in virtue_counts.items()}  # per 1000 words
    virtue_df[book] = virtue_freq

virtue_df = virtue_df.round(2)
print("\nVirtue frequencies (per 1000 words):")
print(virtue_df)

# --- Step 8: Compute Overall Spiritual Polarity across all books ---
positive_virtues = ["love", "joy", "peace", "faith", "hope", "patience",
                    "kindness", "goodness", "gentleness", "self_control"]
negative_virtues = ["fear", "sin"]

polarity_scores = {}
for book, text in book_word_counts.items():
    virtue_counts = {}
    for virtue, keywords in VIRTUE_KEYWORDS.items():
        virtue_counts[virtue] = sum(text.get(k, 0) for k in keywords)
    pos_sum = sum(virtue_counts[v] for v in positive_virtues)
    neg_sum = sum(virtue_counts[v] for v in negative_virtues)
    polarity_scores[book] = pos_sum - neg_sum

# Convert to DataFrame
polarity_df = pd.DataFrame.from_dict(polarity_scores, orient='index', columns=['Spiritual Polarity'])

# Normalize per 1000 words
for book in polarity_df.index:
    total_words = sum(book_word_counts[book].values())
    polarity_df.loc[book, 'Polarity_per_1000'] = polarity_df.loc[book, 'Spiritual Polarity'] / total_words * 1000

polarity_df = polarity_df.round(3)

# --- Chronological order ---
chronological_books = [
    "Genesis", "Job", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
    "Psalms", "Proverbs", "Ecclesiastes", "Song Of Solomon",
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel",
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "Revelation"
]

chronological_books = [b for b in chronological_books if b in polarity_df.index]
polarity_df = polarity_df.loc[chronological_books]

# --- Step 9: Plot Chronological Polarity Flow with Malachi–Matthew bridge ---
x = np.arange(len(polarity_df))
y = polarity_df['Polarity_per_1000'].values
nt_start = next(i for i, b in enumerate(polarity_df.index) if b == "Matthew")

plt.figure(figsize=(16,6))
plt.plot(x[:nt_start], y[:nt_start], color='goldenrod', lw=1.8, marker='o', markersize=5, label='Old Testament')
plt.plot(x[nt_start:], y[nt_start:], color='teal', lw=1.8, marker='o', markersize=5, label='New Testament')

# Add dotted bridge line between Malachi → Matthew
malachi_idx = polarity_df.index.get_loc("Malachi")
matthew_idx = polarity_df.index.get_loc("Matthew")
plt.plot([malachi_idx, matthew_idx], [y[malachi_idx], y[matthew_idx]],
         color='gray', linestyle='--', lw=1.2, alpha=0.7)
plt.text((malachi_idx + matthew_idx)/2, (y[malachi_idx]+y[matthew_idx])/2 + 0.05,
         "Intertestamental Bridge", ha='center', va='bottom', fontsize=9, color='gray')

plt.title("Chronological Spiritual Polarity: From Genesis ➜ Revelation", fontsize=15, pad=25)
plt.xlabel("Chronological Book Order")
plt.ylabel("Polarity (Positive - Negative, per 1000 words)")
plt.xticks(x, polarity_df.index, rotation=90)
plt.grid(alpha=0.3)
plt.legend()

# --- Interpretation box ---
plt.text(
    len(polarity_df.index)*0.35,
    max(y)*0.95,
    "📖 Interpretation:\n"
    "Positive = love, joy, peace, faith, hope, kindness, etc.\n"
    "Negative = fear, sin, iniquity, transgression.\n"
    "Dashed bridge → 400 years of silence: prophecy awaiting fulfillment.",
    fontsize=9.5,
    ha='left', va='top',
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5')
)

plt.tight_layout()
plt.show()

# --- Step 10: Virtue Word Clouds ---
for book in books_to_compare:
    virtue_freq = virtue_df[book].to_dict()
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(virtue_freq)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Virtue Word Cloud for {book}")
    plt.show()
