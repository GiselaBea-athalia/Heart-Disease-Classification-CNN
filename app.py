import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import emoji  # Import the emoji library
from tensorflow.keras import backend as K  # Import Keras backend for clearing session

# Sidebar content
st.sidebar.title("🫀 Heart Disease Prediction Settings")  # Changed to heart organ emoji
st.sidebar.header("Choose Options")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your PCG signal (.wav)", type=["wav"])

# Reset button to clear the uploaded file
reset_button = st.sidebar.button("Reset")

# When reset button is clicked, clear the uploaded file
if reset_button:
    uploaded_file = None
    st.sidebar.write("Upload your PCG signal (.wav) again to start.")

# Toggle options to display different signal analyses
show_waveform = st.sidebar.checkbox("Show PCG Signal Waveform", value=True)
show_spectrogram = st.sidebar.checkbox("Show Spectrogram", value=True)
play_audio = st.sidebar.checkbox("Play Audio", value=True)

# Preprocess audio function to ensure input data is in the right format
def preprocess_audio(audio, sample_length=20000):
    # Ensure the audio signal is the right length (resample if necessary)
    if len(audio) < sample_length:
        audio = np.pad(audio, (0, sample_length - len(audio)), 'constant')  # Zero padding if shorter
    else:
        audio = audio[:sample_length]  # Truncate if longer

    # Reshape the input for the model (samples, 20000, 1)
    audio = audio.reshape((1, sample_length, 1))
    return audio

# Spectrogram visualization
def plot_spectrogram(audio, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    st.pyplot(plt)

# If no file is uploaded, show the welcome page
if uploaded_file is None:
    st.title("Welcome to the Heart Disease Prediction App 🫀")
    st.write(""" 
    This app helps you predict potential heart diseases based on your PCG signal.
    
    **How to Use:**
    1. Upload a PCG signal (.wav file).
    2. The app will analyze the signal and provide predictions about heart diseases.
    3. You can toggle options to visualize the waveform, spectrogram, and play the audio.
    
    **Click the button below to start the prediction!**
    """)
    
    start_button = st.button("Start Prediction")
    
    if start_button:
        st.sidebar.write("Upload your PCG signal (.wav) to proceed.")

# If a file is uploaded, proceed with the analysis
else:
    # Clear the Keras session to avoid memory issues
    K.clear_session()

    # Load the audio file
    audio, sr = librosa.load(uploaded_file, sr=8000)

    # Play the audio if the checkbox is selected
    if play_audio:
        st.audio(uploaded_file, format="audio/wav")

    # Display audio waveform if selected
    if show_waveform:
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
        plt.title("PCG Signal Waveform " + emoji.emojize(":sound:"))
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        st.pyplot(plt)

    # Preprocess audio for prediction
    audio_input = preprocess_audio(audio)

    # Load the trained model (ensure the model path is correct)
    model = tf.keras.models.load_model(r'C:/Users/gisel/Documents/TERM 6/ACADEMIC/SIGNAL PROCESSING/fas_mnist_1.keras')

    # Model prediction
    prediction = model.predict(audio_input)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Mapping the class to more descriptive names
    classes = ['AS', 'MR', 'MS', 'MVP', 'N']  # Add the classes here

    # Mapping the predicted class to descriptive names
    class_map = {
        'AS': 'Aortic Stenosis',
        'MR': 'Mitral Regurgitation',
        'MS': 'Mitral Stenosis',
        'MVP': 'Mitral Valve Prolapse',
        'N': 'Normal'
    }

    # Get the descriptive class label and disease name
    predicted_label = class_map.get(classes[predicted_class], 'Unknown')
    
    # Disease diagnosis based on prediction
    disease_diagnosis = {
        'AS': 'Aortic Stenosis occurs when the valve in the heart narrows, affecting blood flow.',
        'MR': 'Mitral Regurgitation occurs when the heart\'s mitral valve doesn’t close properly, leading to blood leakage.',
        'MS': 'Mitral Stenosis occurs when the mitral valve is narrowed, reducing blood flow from the left atrium to the left ventricle.',
        'MVP': 'Mitral Valve Prolapse occurs when the mitral valve bulges into the left atrium during contraction.',
        'N': 'No abnormal findings detected in the PCG signal.'
    }
    disease_info = disease_diagnosis.get(classes[predicted_class], 'No information available.')

    # Convert prediction probability to percentage
    predicted_prob = prediction[0][predicted_class] * 100  # Convert to percentage

    # Display the result without the green box
    st.subheader(f"Prediction: {predicted_label} 📊")
    st.write(f"{disease_info}")  # Display disease info directly without the label
    st.write(f"Prediction Probability: {predicted_prob:.2f}%")  # Show as percentage with two decimal places

    # Provide a link to more information based on the prediction
    if predicted_label == 'Aortic Stenosis':
        st.markdown("[Learn more about Aortic Stenosis](https://www.mayoclinic.org/diseases-conditions/aortic-stenosis/symptoms-causes/syc-20353139#:~:text=Aortic%20valve%20stenosis%20is%20a%20thickening%20and%20narrowing%20of%20the,the%20rest%20of%20the%20body.)")
    elif predicted_label == 'Mitral Regurgitation':
        st.markdown("[Learn more about Mitral Regurgitation](https://www.mayoclinic.org/diseases-conditions/mitral-valve-regurgitation/symptoms-causes/syc-20350178#:~:text=Mitral%20valve%20regurgitation%20is%20the,leaks%20backward%20across%20the%20valve.)")
    elif predicted_label == 'Mitral Valve Prolapse':
        st.markdown("[Learn more about Mitral Valve Prolapse](https://www.mayoclinic.org/diseases-conditions/mitral-valve-prolapse/symptoms-causes/syc-20355446)")
    elif predicted_label == 'Mitral Stenosis':
        st.markdown("[Learn more about Mitral Stenosis](https://www.mayoclinic.org/diseases-conditions/mitral-valve-stenosis/symptoms-causes/syc-20353159)")
    else:
        st.write("")  # No link for normal condition

    # Display spectrogram if selected
    if show_spectrogram:
        plot_spectrogram(audio, sr)
