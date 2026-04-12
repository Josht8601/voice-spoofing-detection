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

**Dataset Installation**
- Create a "data" folder within the project directory
- Run the following command to install the dataset and put it in the data folder: kaggle datasets download -d awsaf49/asvpoof-2019-dataset -p data/
- Once installed, the dataset will be installed as a zip file: asvpoof-2019-dataset.zip
- Unzip the file
- Go into the unzipped file to this location: data/asvpoof-2019-dataset/LA/LA
- Move the inner LA/ folder to sit directly in the data/ folder
- data/
   └── LA/
         ├── ASVspoof2019_LA_train/
         ├── ASVspoof2019_LA_dev/
         ├── ASVspoof2019_LA_eval/
         ├── ASVspoof2019_LA_cm_protocols/
         ├── ASVspoof2019_LA_asv_protocols/
         ├── ASVspoof2019_LA_asv_scores/
         ├── README.LA.txt           


## Author and Contact Info
Name: Joshua Thomas

Email: josht8601@gmail.com
