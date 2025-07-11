
# Sentiment Classification Using a Fine-Tuned BERT Model

This project demonstrates fine-tuning a pre-trained BERT model on the [Amazon Polarity dataset](https://huggingface.co/datasets/amazon_polarity) to classify product reviews as **Positive** or **Negative**.

Built with 🤗 Hugging Face Transformers and evaluated using accuracy, F1-score, and manual predictions, this project is part of a data science coursework submission.

---

## 🔍 Project Overview

- **Model**: `bert-base-uncased`
- **Task**: Binary sentiment classification
- **Dataset**: Amazon Polarity (subset)
- **Training Samples**: 20,000
- **Evaluation Samples**: 4,000
- **Tools**: Transformers, Datasets, Scikit-learn, Matplotlib, Seaborn

---

## 📊 Example Predictions

```
Text: "This product was amazing, exceeded my expectations!"
→ Predicted Sentiment: Positive

Text: "Totally not worth the money, very disappointed."
→ Predicted Sentiment: Negative
```

---

## 📦 Installation

Set up the environment using pip:

```bash
pip install transformers datasets evaluate scikit-learn matplotlib seaborn
```

Or, run the notebook directly in **Google Colab**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/bert-sentiment-analysis/blob/main/notebook.ipynb)

---

## 📁 Project Structure

```
bert-sentiment-analysis/
│
├── notebook.ipynb              # Full Colab training notebook
├── figures/                    # Output visualizations (EDA, training loss, etc.)
│   ├── top_20_pos.png
│   ├── top_20_neg.png
│   ├── training_loss.png
│   ├── evaluation.png
│   └── review_length.png
├── sentiment_model/            # Saved fine-tuned model + tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── README.md                   # This file
└── requirements.txt            # (Optional) Package list
```

---

## 📈 Results

| Metric       | Score |
|--------------|-------|
| Accuracy     | 94%   |
| F1 (Positive)| 0.94  |
| F1 (Negative)| 0.93  |

Training and evaluation plots are saved in `/figures` for reporting and visualization.

---

## 🧠 Key Learnings

- BERT generalizes well to sentiment tasks with minimal tuning
- Data distribution and token-level insights can explain model behavior
- Simple models perform well on clean, binary classification tasks

---

## 🧪 Future Improvements

- Handle sarcasm and mixed sentiment
- Visualize attention weights for interpretability
- Explore smaller distilled models for deployment

---

## 👤 Author

**Mohammed Nihad K.**  
Student ID: `12345678`  
GitHub: [@yourusername](https://github.com/yourusername)

---

## 📚 References

- Devlin, J., et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
- Zhang, X., et al. (2015). [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).
