# Ultralight Digital Human

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10-aff.svg"></a>
    <a href="https://github.com/anliyuan/Ultralight-Digital-Human/stargazers"><img src="https://img.shields.io/github/stars/anliyuan/Ultralight-Digital-Human?color=ccf"></a>
  <br>
    <br>
</p>

A lightweight digital human model that can run on mobile devices in real-time.

![DigitalHuman](https://github.com/user-attachments/assets/9d0b37ee-2076-4b4f-93ba-eb939a9fb427)

## Important Notes

*   **Audio Quality:** The quality of the audio in your input video is crucial for good results. Avoid audio with significant noise, echo, or unclear speech. Using an external microphone is recommended over the built-in microphone of your recording device.
*   **Streaming Inference:** For streaming inference, it's recommended to place the silent images and their corresponding landmarks in separate directories (`img_inference` and `lms_inference`).
*   **Training Video:** When recording your training video, it's a good practice to have the first 20 seconds without speech, but with some minor movements to simulate the actions of a digital human while speaking. This initial segment can be used as material for streaming inference.

## Training

### 1. Environment Setup

It's recommended to use a virtual environment to manage the project's dependencies.

**Using `venv` (Recommended):**

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install opencv-python transformers==4.37.2 numpy==1.23.5 soundfile librosa onnxruntime
```

**Using `conda`:**

```bash
conda create -n dh python=3.10
conda activate dh
# For systems with NVIDIA GPUs
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# For CPU-only systems
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
conda install mkl=2024.0
pip install opencv-python transformers==4.37.2 numpy==1.23.5 soundfile librosa onnxruntime
```

**Note:** This project was originally tested with `pytorch==1.13.1`. Other versions might also work.

### 2. Download Pre-trained Models

Download `wenet_encoder.onnx` from [this Google Drive link](https://drive.google.com/file/d/1e4Z9zS053JEWl6Mj3W9Lbc9GDtzHIg6b/view?usp=drive_link) and place it in the `data_utils/` directory.

### 3. Data Preprocessing

Prepare a 3-5 minute video with a clear, front-facing view of the speaker and high-quality audio. Place the video in a new folder.

This project uses two different audio feature extractors: `wenet` and `hubert`.

*   **`wenet`:** Faster, suitable for real-time applications on mobile devices. Requires a video frame rate of **20 fps**.
*   **`hubert`:** Generally produces better results. Requires a video frame rate of **25 fps**.

To preprocess your video, run the following command:

```bash
cd data_utils
python process.py YOUR_VIDEO_PATH --asr hubert
```

Replace `YOUR_VIDEO_PATH` with the path to your video file and choose the desired ASR model (`hubert` or `wenet`).

### 4. Training

After preprocessing, you can start training the model.

**Train the SyncNet (Optional but Recommended):**

Training a SyncNet first can lead to better results.

```bash
cd ..
python syncnet.py --save_dir ./syncnet_ckpt/ --dataset_dir ./data_dir/ --asr hubert
```

**Train the Digital Human Model:**

```bash
cd ..
python train.py --dataset_dir ./data_dir/ --save_dir ./checkpoint/ --asr hubert --use_syncnet --syncnet_checkpoint syncnet_ckpt
```

## Inference

Before running inference, you need to extract the audio features from your test audio file (16000Hz sample rate).

```bash
# For hubert
python data_utils/hubert.py --wav your_test_audio.wav

# For wenet
python data_utils/wenet_infer.py your_test_audio.wav
```

This will generate a `.npy` file containing the audio features.

Then, run the inference script:

```bash
python inference.py --asr hubert --dataset ./your_data_dir/ --audio_feat your_test_audio_hu.npy --save_path xxx.mp4 --checkpoint your_trained_ckpt.pth
```

Finally, merge the generated video with the original audio:

```bash
ffmpeg -i xxx.mp4 -i your_audio.wav -c:v libx264 -c:a aac result_test.mp4
```

## Troubleshooting

*   **`torch.cuda.is_available() is False` error:** This error occurs when trying to load a CUDA-trained model on a CPU-only machine. The scripts have been patched to include `map_location=torch.device('cpu')` when loading models. If you encounter this error in other scripts, apply the same fix.
*   **`ffmpeg` errors:** The `process.py` script has been updated to handle videos without audio streams. If you encounter other `ffmpeg` errors, ensure it is installed correctly and accessible in your system's PATH.
*   **`ImportError: cannot import name 'Wav2Vec2Processor' from 'transformers'`:** This is due to an incompatibility between the `transformers` and `pytorch` versions. The recommended versions are `transformers==4.37.2` and `pytorch==1.13.1`.

## Contributing

If you have any suggestions for improvement, feel free to open an issue or submit a pull request.

If you find this repository useful, please consider giving it a star!

## Support the Original Creator

This repository is a fork of the original [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human) repository. If you find this project helpful, please consider supporting the original author by buying them a coffee.

<table>
  <tr>
    <td><img src="demo/15bef5a6d08434c0d70f0ba39bb14fc0.JPG" width="180"/></td>
    <td><img src="demo/36d2896f13bee68247de6ccc89b17a94.JPG" width="180"/></td>
  </tr>
</table>
