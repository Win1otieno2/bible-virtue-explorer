# app.py
import re
import io
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gradio as gr
from PIL import Image

# Download stopwords
nltk.download('stopwords')

# --- Load KJV text ---
def load_verses(path="kjv.txt"):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
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
    return pd.DataFrame(data, columns=["book", "verse"])

# --- Prepare data ---
DF = load_verses("kjv.txt")
BOOK_TEXTS = DF.groupby('book')['verse'].apply(lambda v: ' '.join(v)).to_dict()

STOP_WORDS = set(stopwords.words('english'))
BOOK_WORD_COUNTS = {}
for book, text in BOOK_TEXTS.items():
    words = re.findall(r'\b[a-z]+\b', text.lower())
    words = [w for w in words if w not in STOP_WORDS]
    BOOK_WORD_COUNTS[book] = Counter(words)

# --- Virtue definitions ---
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

# --- Chronological order (Genesis → Revelation) ---
CHRONOLOGICAL_BOOKS = [
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
CHRONOLOGICAL_BOOKS = [b for b in CHRONOLOGICAL_BOOKS if b in BOOK_WORD_COUNTS]

# --- Helper functions ---
def compute_virtue_freq(book):
    wc = BOOK_WORD_COUNTS.get(book)
    if not wc:
        return {v: 0 for v in VIRTUE_KEYWORDS}
    total = sum(wc.values()) or 1
    return {virtue: sum(wc.get(k, 0) for k in keys) / total * 1000 for virtue, keys in VIRTUE_KEYWORDS.items()}

def compute_polarity_all():
    positive = ["love","joy","peace","faith","hope","patience","kindness","goodness","gentleness","self_control"]
    negative = ["fear","sin"]
    scores = {}
    for b, wc in BOOK_WORD_COUNTS.items():
        total = sum(wc.values()) or 1
        pos = sum(sum(wc.get(k,0) for k in VIRTUE_KEYWORDS[v]) for v in positive)
        neg = sum(sum(wc.get(k,0) for k in VIRTUE_KEYWORDS[v]) for v in negative)
        scores[b] = (pos - neg) / total * 1000
    return [(b, scores.get(b, 0.0)) for b in CHRONOLOGICAL_BOOKS]

# --- Plotting ---
def plot_polarity_line():
    ordered = compute_polarity_all()
    books = [b for b, _ in ordered]
    vals = [v for _, v in ordered]
    x = np.arange(len(books))
    nt_idx = books.index("Matthew") if "Matthew" in books else None

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if nt_idx is None:
        ax.plot(x, vals, marker='o', color='teal')
    else:
        # Old vs New Testament lines
        ax.plot(x[:nt_idx], vals[:nt_idx], marker='o', color='goldenrod', label='Old Testament')
        ax.plot(x[nt_idx:], vals[nt_idx:], marker='o', color='teal', label='New Testament')

        # --- Add dotted horizontal connector between Malachi and Matthew ---
        if nt_idx > 0:
            ax.plot(
                [x[nt_idx - 1], x[nt_idx]],
                [vals[nt_idx - 1], vals[nt_idx]],
                linestyle='--', color='gray', linewidth=1.2
            )
            # Label for the gap
            mid_x = (x[nt_idx - 1] + x[nt_idx]) / 2
            mid_y = (vals[nt_idx - 1] + vals[nt_idx]) / 2
            ax.text(mid_x, mid_y + 0.3, "400 years of silence", fontsize=9,
                    ha='center', va='bottom', color='dimgray', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(books, rotation=90, fontsize=8)
    ax.set_ylabel("Polarity (pos - neg, per 1000 words)")
    ax.set_title("Chronological Spiritual Polarity (Genesis → Revelation)")
    ax.grid(alpha=0.25)

    ymax = max(vals) if vals else 1
    ax.text(len(x)*0.65, ymax*0.9,
            "Positive = love, joy, peace, faith, hope, kindness, etc.\n"
            "Negative = fear, sin, iniquity, transgression.",
            fontsize=9, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_virtue_bar(book):
    freq = compute_virtue_freq(book)
    labels = list(freq.keys())
    values = [freq[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(labels, values, color="skyblue")
    ax.set_title(f"Virtue Frequency (per 1000 words) — {book}")
    ax.set_ylabel("Mentions per 1000 words")
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_wordcloud(book):
    freq = compute_virtue_freq(book)
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq)
    fig = plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Virtue Word Cloud — {book}")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- Main Gradio function ---
def analyze(books_input):
    books = [b.strip().title() for b in books_input.split(',') if b.strip()]
    available = sorted(BOOK_WORD_COUNTS.keys())
    missing = [b for b in books if b not in BOOK_WORD_COUNTS]
    if missing:
        return f"Books not found: {', '.join(missing)}. Available: {', '.join(available)}", None, None, None

    virtue_table = {b: compute_virtue_freq(b) for b in books}
    virtue_df = pd.DataFrame(virtue_table).T.round(2)

    bar_imgs = [plot_virtue_bar(b) for b in books]
    wc_imgs = [plot_wordcloud(b) for b in books]
    polarity_img = plot_polarity_line()

    pol_list = compute_polarity_all()
    sorted_by_pol = sorted(pol_list, key=lambda x: -x[1])
    top = ', '.join([p[0] for p in sorted_by_pol[:5]])
    bottom = ', '.join([p[0] for p in sorted_by_pol[-5:]])
    summary = (
        f"Top uplifting books: {top}\n"
        f"Most convicting books: {bottom}\n"
    )
    return summary, virtue_df, polarity_img, bar_imgs[0]

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 📖 Bible Virtue & Polarity Explorer")
    gr.Markdown("Explore the flow of spiritual emotion and virtue from Genesis to Revelation.")
    with gr.Row():
        inp = gr.Textbox(value="John", label="Books to analyze")
        run = gr.Button("Analyze")
    summary_out = gr.Textbox(label="Summary", interactive=False)
    table_out = gr.Dataframe(headers=["Virtue"], label="Virtue frequencies (per 1000 words)")
    with gr.Row():
        pol_out = gr.Image(label="Polarity (Chronological)")
        wc_out = gr.Image(label="Virtue Word Cloud (first book)")
    run.click(fn=analyze, inputs=[inp], outputs=[summary_out, table_out, pol_out, wc_out])

if __name__ == "__main__":
    demo.launch()
