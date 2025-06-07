import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display

# Load your trained model
model = tf.keras.models.load_model(r'C:/Users/gisel/Documents/TERM 6/ACADEMIC/SIGNAL PROCESSING/fas_mnist_1.keras')  # Update with your model path

# Title for the web app
st.title("Heart Disease Prediction from PCG Signal")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

# Function to preprocess audio
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
    # Compute the Mel-scaled spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    st.pyplot(plt)

# When file is uploaded
if uploaded_file is not None:
    # Load the audio file
    audio, sr = librosa.load(uploaded_file, sr=8000)
    st.audio(uploaded_file, format='audio/wav')  # Play the audio

    # Display audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
    plt.title("PCG Signal Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

    # Preprocess audio for prediction
    audio_input = preprocess_audio(audio)

    # Model prediction
    prediction = model.predict(audio_input)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Mapping the class
    classes = ['AS', 'MR', 'MS', 'MVP', 'N']  # Classes: Aortic Stenosis, Mitral Regurgitation, Mitral Stenosis, Mitral Valve Prolapse, Normal
    predicted_label = classes[predicted_class]
    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Prediction probabilities: {prediction}")

    # Display spectrogram
    plot_spectrogram(audio, sr)
