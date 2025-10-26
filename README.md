# Speech Recognition Project

This project implements speech recognition using two different approaches: Wav2Vec2.0 and Whisper models. It provides a comparison between these two state-of-the-art speech recognition models.

## Project Structure

```
├── SpeechRec/
│   ├── Speech.py          # Main speech recognition implementation
│   ├── audio/            # Directory containing audio samples
│   └── wav2vec2-finetuned/ # Fine-tuned Wav2Vec2.0 model
├── wav2vec.ipynb         # Jupyter notebook for Wav2Vec2.0 implementation
└── whisper.ipynb         # Jupyter notebook for Whisper implementation
```

## Features

- Implementation of speech recognition using Wav2Vec2.0
- Implementation of speech recognition using Whisper
- Comparative analysis of both models
- Sample audio processing and recognition
- Pre-trained model usage

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Librosa
- Jupyter Notebook

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Meet78/Speech-Recognition.git
   ```

2. Install the required dependencies:
   ```bash
   pip install torch transformers librosa jupyter
   ```

3. Download the necessary model files:
   - The wav2vec2-finetuned model should be placed in the appropriate directory
   - Whisper model will be downloaded automatically when running the code

## Usage

1. For Wav2Vec2.0:
   - Open `wav2vec.ipynb` in Jupyter Notebook
   - Follow the instructions in the notebook for speech recognition

2. For Whisper:
   - Open `whisper.ipynb` in Jupyter Notebook
   - Follow the instructions in the notebook for speech recognition

3. Using the Python script:
   - Navigate to the SpeechRec directory
   - Run `Speech.py` with appropriate audio input

## Results

The project includes a comparative analysis of both Wav2Vec2.0 and Whisper models for speech recognition tasks.
