import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon=":camera:")

st.title("ChromaticScan")

st.caption(
    "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
)

st.write("Try clicking a leaf image and watch how an AI Model will detect its disease.")

with st.sidebar:
    img = Image.open("./images/leaf.png")
    st.image(img)
    st.subheader("About ChromaticScan")
    st.write(
        "ChromaticScan is a state-of-the-art convolutional neural network (CNN) algorithm that is specifically designed for detecting plant diseases. It utilizes transfer learning by fine-tuning the ResNet 34 model on a large dataset of leaf images to achieve an impressive 99.2% accuracy in detecting various plant diseases. The algorithm is trained to identify specific patterns and features in the leaf images that are indicative of different types of diseases, such as leaf spots, blights, and wilts."
    )

    st.write(
        "ChromaticScan is designed to be highly robust and accurate, with the ability to detect plant diseases in a wide range of conditions and environments. It can be used to quickly and accurately diagnose plant diseases, allowing farmers and gardeners to take immediate action to prevent the spread of the disease and minimize crop losses."
    )

    st.write(
        "The application will infer one label out of 39 labels, including diseases like Apple scab, Early blight, Late blight, and more."
    )

    classes = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry___Powdery_mildew",
        "Cherry___healthy",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn___Common_rust",
        "Corn___Northern_Leaf_Blight",
        "Corn___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
        "Background_without_leaves",
    ]

    classes_and_descriptions = {
        "Apple___Apple_scab": "Apple with Apple scab disease detected.",
        "Apple___Black_rot": "Apple with Black rot disease detected.",
        "Apple___Cedar_apple_rust": "Apple with Cedar apple rust disease detected.",
        "Apple___healthy": "Healthy apple leaf detected.",
        "Blueberry___healthy": "Healthy blueberry leaf detected.",
        "Cherry___Powdery_mildew": "Cherry with Powdery mildew disease detected.",
        "Cherry___healthy": "Healthy cherry leaf detected.",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn with Cercospora leaf spot or Gray leaf spot disease detected.",
        "Corn___Common_rust": "Corn with Common rust disease detected.",
        "Corn___Northern_Leaf_Blight": "Corn with Northern Leaf Blight disease detected.",
        "Corn___healthy": "Healthy corn leaf detected.",
        "Grape___Black_rot": "Grape with Black rot disease detected.",
        "Grape___Esca_(Black_Measles)": "Grape with Esca (Black Measles) disease detected.",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape with Leaf blight (Isariopsis Leaf Spot) disease detected.",
        "Grape___healthy": "Healthy grape leaf detected.",
        "Orange___Haunglongbing_(Citrus_greening)": "Orange with Haunglongbing (Citrus greening) disease detected.",
        "Peach___Bacterial_spot": "Peach with Bacterial spot disease detected.",
        "Peach___healthy": "Healthy peach leaf detected.",
        "Pepper,_bell___Bacterial_spot": "Bell pepper with Bacterial spot disease detected.",
        "Pepper,_bell___healthy": "Healthy bell pepper leaf detected.",
        "Potato___Early_blight": "Potato with Early blight disease detected.",
        "Potato___Late_blight": "Potato with Late blight disease detected.",
        "Potato___healthy": "Healthy potato leaf detected.",
        "Raspberry___healthy": "Healthy raspberry leaf detected.",
        "Soybean___healthy": "Healthy soybean leaf detected.",
        "Squash___Powdery_mildew": "Squash with Powdery mildew disease detected.",
        "Strawberry___Leaf_scorch": "Strawberry with Leaf scorch disease detected.",
        "Strawberry___healthy": "Healthy strawberry leaf detected.",
        "Tomato___Bacterial_spot": "Tomato leaf with Bacterial spot disease detected.",
        "Tomato___Early_blight": "Tomato leaf with Early blight disease detected.",
        "Tomato___Late_blight": "Tomato leaf with Late blight disease detected.",
        "Tomato___Leaf_Mold": "Tomato leaf with Leaf Mold disease detected.",
        "Tomato___Septoria_leaf_spot": "Tomato leaf with Septoria leaf spot disease detected.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato leaf with Spider mites or Two-spotted spider mite disease detected.",
        "Tomato___Target_Spot": "Tomato leaf with Target Spot disease detected.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf with Tomato Yellow Leaf Curl Virus disease detected.",
        "Tomato___Tomato_mosaic_virus": "Tomato leaf with Tomato mosaic virus disease detected.",
        "Tomato___healthy": "Healthy tomato leaf detected.",
        "Background_without_leaves": "No plant leaf detected in the image.",
    }

    # Define remedies for each class
    remedies = {
        "Apple___Apple_scab": "Apply fungicide and remove affected leaves.",
        "Apple___Black_rot": "Use copper fungicides and ensure good air circulation.",
        "Apple___Cedar_apple_rust": "Use resistant varieties and apply fungicides.",
        "Apple___healthy": "No treatment needed.",
        "Blueberry___healthy": "No treatment needed.",
        "Cherry___Powdery_mildew": "Apply fungicides to manage powdery mildew.",
        "Cherry___healthy": "No treatment needed.",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides and improve drainage.",
        "Corn___Common_rust": "Use resistant hybrids and fungicides.",
        "Corn___Northern_Leaf_Blight": "Rotate crops and use resistant varieties.",
        "Corn___healthy": "No treatment needed.",
        "Grape___Black_rot": "Use fungicides and remove infected fruits.",
        "Grape___Esca_(Black_Measles)": "Prune affected vines and apply appropriate fungicides.",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides and improve air circulation.",
        "Grape___healthy": "No treatment needed.",
        "Orange___Haunglongbing_(Citrus_greening)": "Use disease-free plants and manage insect vectors.",
        "Peach___Bacterial_spot": "Apply copper-based fungicides and improve air circulation.",
        "Peach___healthy": "No treatment needed.",
        "Pepper,_bell___Bacterial_spot": "Use resistant varieties and apply copper fungicides.",
        "Pepper,_bell___healthy": "No treatment needed.",
        "Potato___Early_blight": "Use fungicides and rotate crops.",
        "Potato___Late_blight": "Use resistant varieties and apply fungicides.",
        "Potato___healthy": "No treatment needed.",
        "Raspberry___healthy": "No treatment needed.",
        "Soybean___healthy": "No treatment needed.",
        "Squash___Powdery_mildew": "Apply fungicides and improve air circulation.",
        "Strawberry___Leaf_scorch": "Water adequately and apply fungicides.",
        "Strawberry___healthy": "No treatment needed.",
        "Tomato___Bacterial_spot": "Use resistant varieties and apply copper fungicides.",
        "Tomato___Early_blight": "Use fungicides and rotate crops.",
        "Tomato___Late_blight": "Use resistant varieties and apply fungicides.",
        "Tomato___Leaf_Mold": "Improve ventilation and apply fungicides.",
        "Tomato___Septoria_leaf_spot": "Use fungicides and rotate crops.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticides and improve irrigation.",
        "Tomato___Target_Spot": "Use fungicides and rotate crops.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control aphids and remove infected plants.",
        "Tomato___Tomato_mosaic_virus": "Use disease-free seeds and remove infected plants.",
        "Tomato___healthy": "No treatment needed.",
        "Background_without_leaves": "No action required.",
    }

# Load the model once
export_file_path = "./models/myModel.h5"
try:
    model = load_model(export_file_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Function to preprocess the image
def load_and_preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions for prediction
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

# Function to predict disease
def Plant_Disease_Detection(img_file_path):
    try:
        # Load the image
        image = Image.open(img_file_path)
        preprocessed_image = load_and_preprocess_image(image)

        # Predict the class
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]
        confidence_score = prediction[0][predicted_class_index]

        # Prepare probabilities for display
        probabilities = pd.DataFrame(prediction, columns=classes)
        return predicted_class, confidence_score, probabilities
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None

# File uploader
img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if img_file is not None:
    st.image(img_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying... Please wait.")

    # Run prediction
    predicted_class, confidence_score, probabilities = Plant_Disease_Detection(img_file)

    if predicted_class:
        st.subheader("Prediction Result:")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence Score: **{confidence_score:.2f}**")
        st.write("Description: " + classes_and_descriptions.get(predicted_class, "No description available."))
        st.write("Recommended Action: " + remedies.get(predicted_class, "No specific action required."))

        # Bar chart of probabilities
        if probabilities is not None:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=probabilities.columns, y=probabilities.values.flatten(), palette='viridis')
            plt.title('Class Probability Distribution')
            plt.xticks(rotation=90)
            plt.ylabel('Probability')
            plt.xlabel('Classes')
            st.pyplot(plt)

# Footer
footer = """
<style>
footer {
    visibility: hidden;
}
</style>
"""
st.markdown(footer, unsafe_allow_html=True)
