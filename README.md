# Fake News Detection

A modular, professional web application and machine learning pipeline for detecting fake news using multiple datasets and advanced NLP models. The project includes a Flask-based frontend, Jupyter notebooks for data science, and supporting datasets.

---

## Project Structure

```
Fake_news_Detection/
│
├── flask-app/
│   ├── app.py
│   ├── static/
│   │   └── Uploads/
│   └── templates/
│       ├── index.html
│       ├── faq.html
│       ├── 404.html
│       └── 500.html
│
├── Data_set/
│   └── news/
│       └── news.csv
│
├── Dataset/
│   ├── True.csv
│   └── Fake.csv
│
├── politifact/
│   ├── politifact_real.csv
│   └── politifact_fake.csv
│
├── data.csv
│
├── Data-set/
│   └── train.csv
│
├── Fakenewsdetection.ipynb
│
└── README.md
```

---

## Folder & File Details

### `flask-app/`
- **Purpose:** Contains the Flask web application for user interaction and prediction.
- **Key files:**
  - `app.py`: Main Flask backend, handles routing, file uploads, and prediction logic.
  - `static/Uploads/`: Stores uploaded images and documents for analysis.
  - `templates/`: HTML templates for the web UI.
    - `index.html`: Main UI for news verification.
    - `faq.html`: Frequently Asked Questions page.
    - `404.html`, `500.html`: Custom error pages.

### `Data_set/`
- **Purpose:** Contains the first dataset for fake news detection.
- **Key files:**
  - `news/news.csv`: News articles with labels for real/fake.

### `Dataset/`
- **Purpose:** Contains additional datasets for model training and evaluation.
- **Key files:**
  - `True.csv`: Real news articles.
  - `Fake.csv`: Fake news articles.

### `politifact/`
- **Purpose:** Contains Politifact-based datasets for further model robustness.
- **Key files:**
  - `politifact_real.csv`: Real news headlines/articles from Politifact.
  - `politifact_fake.csv`: Fake news headlines/articles from Politifact.

### `Data-set/`
- **Purpose:** Contains another dataset variant for training.
- **Key files:**
  - `train.csv`: News articles with labels.

### `data.csv`
- **Purpose:** Additional dataset for fake news detection, used in model training.

### `Fakenewsdetection.ipynb`
- **Purpose:** Jupyter notebook for data exploration, preprocessing, model training, and evaluation.
- **Details:** Includes EDA, data cleaning, feature engineering, model comparison, and saving the best model.

### `README.md`
- **Purpose:** Project overview and documentation (this file).

---

## How to Run

1. **Install requirements:**  
   Make sure you have Python 3.x and install required packages (Flask, scikit-learn, pandas, etc.).

2. **Run the Flask app:**  
   ```
   cd flask-app
   python app.py
   ```
   Visit `http://localhost:5000` in your browser.

3. **Jupyter Notebook:**  
   Open `Fakenewsdetection.ipynb` for data science workflow and model training.

---

## Notes

- All datasets are used for model training and evaluation to ensure robustness.
- The web app supports text, image, document, and URL-based news verification.
- The UI is modular, responsive, and uses Font Awesome icons for a professional look.

---

## License

This project is for educational and research purposes.
