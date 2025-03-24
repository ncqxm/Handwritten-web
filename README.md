# ✍️ Handwriting Digit Recognizer (Streamlit App)

This is a simple but powerful web application that lets users **draw a digit (0–9)** and uses a pre-trained **CNN (Convolutional Neural Network)** model to recognize the digit in real-time.

Built using **Streamlit**, powered by **TensorFlow/Keras**, and trained on the classic **MNIST** dataset.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://draw-to-predict.streamlit.app/) 

---

## ✨ Features

- Draw digits 0–9 with your mouse
- AI model predicts your number instantly
- Visual bar chart of prediction confidence
- Canvas with adjustable brush size
- Fully responsive and interactive
- 
---

## 🧰 Tech Stack

- Python 3.8+
- Streamlit
- TensorFlow / Keras
- Pillow
- streamlit-drawable-canvas
- Jupyter Notebook (for model training)

---
## 📓 Model Training Notebook

See full training process in this Jupyter Notebook:  
👉 [train_model.ipynb](https://github.com/yourname/handwriting-digit-app/blob/main/train_model.ipynb)

---

## 📦 Installation (Local)

1. **Clone the repo**
```bash
git clone https://github.com/yourname/handwriting-digit-app.git
cd handwriting-digit-app
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

---

## 🧠 Model Info

- CNN trained on MNIST dataset (28x28 grayscale digits)
- Architecture:
    - 2x Conv2D + MaxPooling
    - Flatten → Dense → Dropout
    - Output layer with 10 classes (0–9)

---

## 📚 Dataset Source

This project uses the MNIST dataset for handwritten digit classification.  
📎 Official source: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)

---

## 📄 License

MIT License — feel free to use, modify, and share.

---

## ✍️ Author

Made with ❤️ by [ncqxm] — feel free to fork or contribute!
