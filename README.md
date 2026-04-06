# voice-spoofing-detection

## Description
Binary audio spoofing detector classifying speech as real or AI-generated using Light CNN and Wav2Vec2/HuBERT. Evaluated on ASVspoof 2019 with a focus on generalization to unseen voice synthesis methods and robustness under noise, compression, and pitch variations.

## Installation & Setup Instructions

## Dataset Information
This project uses the **ASVspoof 2019 dataset**, a benchmark dataset for detecting synthetic (spoofed) speech in automatic speaker verification systems. It includes both bonafide (real) and spoofed audio, with spoofed samples generated from multiple text-to-speech and voice conversion systems.

The dataset contains approximately 25,000 audio samples and is pre-split into training, development, and evaluation sets. Protocol files provide labels for each audio sample, enabling supervised training and evaluation. This project primarily utilizes the Logical Access (LA) partition, which focuses on AI-generated speech.

- Primary partition used: Logical Access (LA)
- Audio format: FLAC
- Evaluation metric: Equal Error Rate (EER)

**Access**

The dataset is available on Kaggle:
- https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset


## Author and Contact Info
Name: Joshua Thomas
Email: josht8601@gmail.com
