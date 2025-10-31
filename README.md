# 📖 Bible Virtue & Polarity Explorer

An interactive **Natural Language Processing (NLP)** exploration of the **Bible (KJV)** using **Gradio**.  
This app visualizes the emotional and spiritual flow across the books of Scripture — from *Genesis* to *Revelation* — through the lens of biblical virtues such as **love, peace, faith, and hope**.

---

## ✨ Features

- 🔍 **Virtue Frequency Analysis** — Quantifies mentions of virtues (e.g., *love*, *faith*, *peace*) per 1,000 words.  
- 🌈 **Spiritual Polarity Timeline** — Displays the positive vs. negative tone across all books, chronologically.  
  - The dotted connection between *Malachi → Matthew* symbolizes the *“400 years of silence.”*
- ☁️ **Virtue Word Cloud** — Highlights which virtues dominate a selected book.  
- 📊 **Virtue Table** — Shows numerical breakdowns for easy comparison.  
- 🕊️ **Interpretation Overlay** — Explains how positive and negative spiritual tones are defined.

---

## 🧠 How It Works

Each book of the **King James Bible** is processed with simple NLP techniques:

1. **Text Parsing** — Extracts and tokenizes all words.
2. **Stopword Removal** — Removes common words like “the”, “and”, “is”.
3. **Virtue Mapping** — Maps words to emotional or moral categories:
   - Positive: *love, joy, peace, faith, kindness...*  
   - Negative: *fear, sin, iniquity...*
4. **Polarity Calculation** — Computes `(positive - negative)` mentions per 1,000 words.
5. **Visualization** — Uses Matplotlib + WordCloud for clear interpretation.

---

## 🧩 Example Usage

1. Enter a few book names in the input box, e.g.:

   ```
    John
   ```

2. Click **Analyze**.
3. Explore:
   - The **Virtue frequency table**
   - The **Word Cloud**
   - The **Chronological Polarity Chart**



---

## 📈 Interpretation

- **Higher polarity** → A more uplifting and virtue-filled tone (love, joy, hope).  
- **Lower polarity** → A more convicting or sin-focused tone (fear, judgment, lament).  
- The transition from *Malachi → Matthew* marks the spiritual silence before the coming of Christ — represented with a subtle dotted connector.

---

## 🧰 Tech Stack

| Component | Library |
|------------|----------|
| UI Framework | [Gradio](https://gradio.app/) |
| NLP & Data | NLTK, Pandas |
| Visualization | Matplotlib, WordCloud |
| Deployment | Hugging Face Spaces |

---

## ⚙️ Installation (for local run)

```bash
git clone https://huggingface.co/spaces/win1otieno2/bible-virtue-explorer
cd bible-virtue-explorer
pip install -r requirements.txt
python app.py
```

Then open:  
👉 `http://127.0.0.1:7860`

---

## 🙏 Credits

- **Bible Text:** *King James Version ([openbible](https://openbible.com/textfiles/kjv.txt))*  
- **Developer:** Winstan Otieno  
- **Inspired by:** Galatians 5:22–23 — *“But the fruit of the Spirit is love, joy, peace…”*

---

## 🕊️ License

This project is released under the **MIT License**.  
You are free to fork, adapt, and reuse for educational or ministry purposes.

---
