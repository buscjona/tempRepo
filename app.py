import streamlit as st
from PIL import Image
from img_classification import teachable_machine_classification

st.title("Image Captioning")
st.header("Captioning on examples")
st.write("Upload a picture of a horse, dog, climber, football player, biker, child, or car. As these are the "
         "categories, that the model was trained on. Trying pictures of other categories might result in not correct"
         " captions!")

uploaded_file = st.file_uploader("Upload a .jpg of your choice ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.write("")
    st.write("Captioning...")
    label = teachable_machine_classification(image, 'keras_model.h5')

    for line in open('labels.txt'):
        if str(label) in line:
            st.write(line.split()[-1])
