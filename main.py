# frontend script for the model predictor

import cv2
import wave
import librosa
import matplotlib.pyplot as plt

def get_model_prediction(aud_path) :
    prediction = 0 # replace with pytorch model prediction logic
    return prediction

def draw_wave(aud_path) :
    spf = wave.open("wavfile.wav", "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    plt.figure(1)
    plt.title("Signal Wave")
    plt.plot(signal)
    plt.savefig('frontend_wavelet_temp.png')
    plt.close()
    img = cv2.imread('frontend_wavelet_temp.png')
    return img

# ----------- General things
st.title('Volume prediction tool')
valid_molecule = True
loaded_molecule = None
selection = None
submit = None

# ----------- Inputs
st.markdown("Select input file of mp3 type ...")
upload_columns = st.columns([2, 1])

# File upload
file_upload = upload_columns[0].expander(label="Upload an audio file")
uploaded_file = file_upload.file_uploader("Choose an audio file", type=['mp3'])

temp_filename = "temp.mp3"
with open(temp_filename, "wb") as f:
    f.write(uploaded_file.getbuffer())

# Draw if valid
if uploaded_file is not None:
    st.info("This audio file appears to be valid :ballot_box_with_check:")
    pil_img = draw_wave(temp_filename)
    upload_columns[1].image(pil_img)
    submit = upload_columns[1].button("Get prediction")

else :
    st.error("This file appears to be invalid :no_entry_sign:")

# ----------- Submission
st.markdown("""---""")
if submit:
    with st.spinner(text="Fetching model prediction..."):
        prediction = get_model_prediction(temp_filename)

    # ----------- Ouputs
    outputs = st.columns([2, 1])
    outputs[0].markdown("Volume Prediction : " + prediction)

