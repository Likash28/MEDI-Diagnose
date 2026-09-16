# MEDI-Diagnose

A Django-based medical AI web platform that bundles multiple ML-powered diagnostic tools under a single application with user authentication and a healthcare chatbot.

---

## Features

| Service | Description |
|---------|-------------|
| **Healthcare Chatbot** | NLP-based medical Q&A bot (intent classification with PyTorch) |
| **Protein Stability Prediction** | Predicts melting temperature from amino acid sequence + pH using ProtBERT + XGBoost |
| **Pneumonia Detection** | Classifies chest X-ray images as Pneumonia Positive/Negative using a CNN |
| **Brain MRI Segmentation** | Segments brain tumor regions from MRI images using a DeepLab-style model |
| **Diabetes Prediction** | Predicts diabetes risk from 8 clinical health parameters |

---

## Tech Stack

- **Backend:** Django 4.1.7, SQLite
- **Deep Learning:** TensorFlow 2.21, Keras 3.14, PyTorch 2.11
- **NLP:** HuggingFace Transformers (ProtBERT), NLTK
- **ML:** XGBoost, scikit-learn, pandas, numpy
- **Image Processing:** OpenCV, Pillow
- **Frontend:** Bootstrap 4, jQuery, Owl Carousel, Typed.js

---

## Project Structure

```
MEDI-Diagnose-master/
├── chatbot/                  # Django project config
│   ├── settings.py           # App settings
│   ├── urls.py               # URL routing
│   ├── views.py              # All view logic and ML inference
│   ├── chat.py               # Chatbot inference engine
│   ├── train.py              # Chatbot training script (run once)
│   ├── tokens.py             # Email activation token
│   ├── info.py               # SMTP credentials (configure before use)
│   └── intents.json          # Chatbot intent patterns and responses
├── templates/                # HTML templates (14 pages)
├── static/                   # CSS, JS, images
├── media/                    # Uploaded images (auto-created)
├── NeuralNet.pth             # Trained chatbot model
├── deeplabnetown.h5          # Brain MRI segmentation model
├── lik.pkl                   # Protein stability XGBoost model
├── diabetes-model.pkl        # Diabetes prediction model
├── saved_model/              # TensorFlow SavedModel (object detection)
├── requirements.txt          # Python dependencies
└── manage.py                 # Django entry point
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or 3.11
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Likash28/MEDI-Diagnose.git
cd MEDI-Diagnose
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 5. Configure email (optional — for registration emails)

Open `chatbot/info.py` and set your Gmail credentials:

```python
EMAIL_HOST_USER = "your_email@gmail.com"
EMAIL_HOST_PASSWORD = "your_gmail_app_password"
```

> To generate a Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords.
> If you skip this step, you can manually activate accounts via the Django admin panel.

### 6. Apply database migrations

```bash
python manage.py migrate
```

### 7. Create an admin account (optional)

```bash
python manage.py createsuperuser
```

### 8. Run the development server

```bash
python manage.py runserver
```

Open your browser at: **http://127.0.0.1:8000**

> **Note:** First startup downloads the ProtBERT model (~1.6 GB) from HuggingFace. This takes 2–5 minutes on first run only.

---

## URL Reference

| URL | Page |
|-----|------|
| `/` | Home (redirects to register if not logged in) |
| `/register` | Create an account |
| `/login` | Log in |
| `/logout` | Log out |
| `/index/` | Main dashboard |
| `/predict` | Healthcare chatbot |
| `/getprediction/` | Protein stability prediction |
| `/predict_pneumonia/` | Pneumonia detection |
| `/brain_mri_segmentation/` | Brain MRI segmentation |
| `/predict_diabetes` | Diabetes prediction |
| `/admin/` | Django admin panel |

---

## Model Files

The following pre-trained model files must be present in the project root to use all features:

| File | Feature | Format |
|------|---------|--------|
| `chatbot/NeuralNet.pth` | Chatbot | PyTorch |
| `deeplabnetown.h5` | Brain MRI segmentation | Keras HDF5 |
| `saved_models.h5` | Pneumonia detection | Keras HDF5 |
| `lik.pkl` | Protein stability | joblib pickle |
| `diabetes-model.pkl` | Diabetes prediction | pickle |
| `saved_model/` | Object detection (dental) | TF SavedModel |

---

## Author

**Likash Gunisetti**
- GitHub: [Likash28](https://github.com/Likash28)
- Email: likashgunisetti@gmail.com
- Institution: Indian Institute of Technology Guwahati
  
