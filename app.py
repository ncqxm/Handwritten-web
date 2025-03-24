import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# โหลดโมเดลที่ฝึกไว้
model = load_model("handwritting_model.keras")

st.set_page_config(page_title="MNIST Digit Recognizer",
                   page_icon= "✍️",
                   layout="centered")
st.title("✍️ Handwriting Digit Recognizer")
st.markdown("วาดเลข 0-9 แล้วให้ AI ทายดูสิว่าใช่มั้ย 🤖")

# ขนาดแคนวาส
CANVAS_SIZE = 400  # ใหญ่ก่อน เดี๋ยว resize ทีหลัง

# วาดเลข
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas"
)

# ถ้าผู้ใช้วาด
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img_pil = Image.fromarray(img)

    # เตรียมภาพให้เหมือน MNIST: ขาวดำ, 28x28, 1 channel
    img_pil = ImageOps.grayscale(img_pil)
    img_pil = img_pil.resize((28, 28))
    img_array = np.array(img_pil).reshape(1, 28, 28, 1)
    img_array = img_array.astype("float32") / 255.0

    # ทำนาย
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.subheader(f"AI ทายว่าเป็นเลข: {predicted_label}")
    st.bar_chart(prediction[0])
