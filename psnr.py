import cv2
import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
# import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt



def download_video(url):
    response = st.sidebar.file_uploader("Upload a video file", type=["mp4"])
    if response:
        return response

def calculate_psnr(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_psnr_for_each_frame(distorted_video_content, good_video_path):
    # Save distorted video content to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(distorted_video_content.getvalue())
        distorted_video_path = temp_file.name

    # Open the videos
    distorted_video = cv2.VideoCapture(distorted_video_path)
    good_video = cv2.VideoCapture(good_video_path)

    psnr_values = []
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
        if psnr < 25.0:
            distorted_frame_numbers.append(len(psnr_values))

        # Get the timestamp of the current frame and append to the list
        current_frame_time = distorted_video.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamps.append(current_frame_time)

    # Release the videos
    distorted_video.release()
    good_video.release()

    # Clean up the temporary file
    os.remove(distorted_video_path)

    return psnr_values, distorted_frame_numbers, frame_timestamps

# Streamlit app code
st.title("PSNR Calculation Demo")

# Upload distorted video
distorted_video_content = download_video("Upload Distorted Video")
if distorted_video_content is not None:
    # Git LFS URL for the reference video
    good_video_path = "https://github.com/jyothishridhar/PSNR_video_quality/raw/main/referance.mp4"

    # Calculate PSNR values for each frame in the distorted video
    psnr_values, distorted_frame_numbers, frame_timestamps = calculate_psnr_for_each_frame(distorted_video_content, good_video_path)

    # Create a list of frame numbers for x-axis
    frame_numbers = list(range(1, len(psnr_values) + 1))

    # Plot the PSNR values in a line graph
    st.line_chart(pd.DataFrame({"Frame Number": frame_numbers, "PSNR Value": psnr_values}).set_index("Frame Number"))

    # Display the result on the app
    st.success("PSNR calculation completed!")

    # Display the PSNR values and frame timestamps
    data = {
        'Frame Number': frame_numbers,
        'PSNR Value': psnr_values,
        'Timestamp (ms)': frame_timestamps
    }

    df = pd.DataFrame(data)
    st.dataframe(df)

    # Save PSNR values, frame numbers, and timestamps to an Excel file
    excel_buffer = df.to_excel(index=False)
    st.markdown(get_excel_link(excel_buffer, "Download PSNR Report"), unsafe_allow_html=True)
