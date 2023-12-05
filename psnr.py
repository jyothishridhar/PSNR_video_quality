import cv2
import streamlit as st
import numpy as np
import pandas as pd
import requests
import base64
from io import BytesIO

def download_video(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return file_name

def calculate_psnr(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_psnr_for_each_frame(distorted_video_path, good_video_path, psnr_threshold):
    # Open the videos
    distorted_video = cv2.VideoCapture(distorted_video_path)
    good_video = cv2.VideoCapture(good_video_path)

    psnr_values = []
    quality_status = []  # 'Good' or 'Distorted' based on PSNR threshold
    distorted_frame_numbers = []
    frame_timestamps = []  # To store the timestamp of each frame

    while True:
        # Read frames from both videos
        ret1, distorted_frame = distorted_video.read()
        ret2, good_frame = good_video.read()

        # If frames are not retrieved, the videos have ended
        if not ret1 or not ret2:
            break

        # Convert frames to grayscale
        distorted_frame_gray = cv2.cvtColor(distorted_frame, cv2.COLOR_BGR2GRAY)
        good_frame_gray = cv2.cvtColor(good_frame, cv2.COLOR_BGR2GRAY)

        # Calculate PSNR for the current frames
        psnr = calculate_psnr(distorted_frame_gray, good_frame_gray)

        # Append the PSNR value to the list
        psnr_values.append(psnr)

        # Check if distortion happens (e.g., PSNR is below a threshold)
        if psnr < psnr_threshold:
            quality_status.append('Distorted')
            distorted_frame_numbers.append(len(psnr_values))
        else:
            quality_status.append('Good')

        # Get the timestamp of the current frame and append to the list
        current_frame_time = distorted_video.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamps.append(current_frame_time)

    # Release the videos
    distorted_video.release()
    good_video.release()

    return psnr_values, quality_status, distorted_frame_numbers, frame_timestamps

# Define get_excel_link function
def get_excel_link(df, title):
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    b64 = base64.b64encode(excel_buffer.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{title}.xlsx">Download {title}</a>'

# Streamlit app code
st.title("PSNR Calculation Demo")

# URLs for the distorted and reference videos
distorted_video_url = "https://github.com/jyothishridhar/PSNR_video_quality/raw/main/distorted.avi"
good_video_url = "https://github.com/jyothishridhar/PSNR_video_quality/raw/main/referance.mp4"

# Download videos
distorted_video_path = download_video(distorted_video_url, 'distorted.mp4')
good_video_path = download_video(good_video_url, 'reference.mp4')

# Add download links
st.markdown(f"**Download Distorted Video**")
st.markdown(f"[Click here to download the Distorted Video]({distorted_video_url})")

st.markdown(f"**Download Reference Video**")
st.markdown(f"[Click here to download the Reference Video]({good_video_url})")

# Add PSNR threshold slider
psnr_threshold = st.slider("Select PSNR Threshold", min_value=0.0, max_value=50.0, value=25.0)

# Add button to run PSNR calculation
if st.button("Run PSNR Calculation"):
    # Calculate PSNR values for each frame in the distorted video
    psnr_values, quality_status, distorted_frame_numbers, frame_timestamps = calculate_psnr_for_each_frame(
        distorted_video_path, good_video_path, psnr_threshold
    )

    # Create a list of frame numbers for x-axis
    frame_numbers = list(range(1, len(psnr_values) + 1))

    # Plot the PSNR values in a line chart using Streamlit
    st.line_chart(pd.DataFrame({"Frame Number": frame_numbers, "PSNR Value": psnr_values}).set_index("Frame Number"))

    # Display the result on the app
    st.success("PSNR calculation completed!")

    # Display the PSNR values, quality status, and frame timestamps
    data = {
        'Frame Number': frame_numbers,
        'PSNR Value': psnr_values,
        'Video Quality Status': quality_status,
        'Timestamp (ms)': frame_timestamps
    }

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Save PSNR values, frame numbers, and timestamps to an Excel file
    st.markdown(get_excel_link(df, "Download PSNR Report"), unsafe_allow_html=True)
