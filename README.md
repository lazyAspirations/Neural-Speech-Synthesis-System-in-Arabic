Neural Arabic Text-to-Speech (TTS) System 🎙️🤖
📌 Project Overview
This project involves the development and evaluation of a Neural Text-to-Speech system specifically designed for the Arabic language. Developed as part of the "Linguistic Data Processing" module (Master 1 DS&NLP), it addresses the morphological and phonetic complexities of Arabic to produce natural, high-quality speech.

Developed by: Aissat Mohamed Moncef

Academic Year: 2025-2026

🚀 Key Features
Neural Synthesis Pipeline: Uses advanced pre-trained models and a HiFi-GAN vocoder for high-fidelity audio generation.

Robust Arabic Processing: Full Unicode normalization and diacritic handling to ensure correct pronunciation.

Scientific Evaluation: Comprehensive assessment using 5 categories of metrics (SNR, THD, Pitch analysis, Spectral analysis, and Energy).

Interactive UI: A professional Streamlit dashboard for real-time synthesis and visualization.

🛠️ System Architecture
The system is divided into four main modules:

TTS Engine: Handles text normalization and spectrogram generation.

Vocoder: Converts spectrograms into high-quality audio waveforms.

TTSEvaluator: A custom class to calculate objective scientific metrics.

Streamlit Interface: Allows users to input custom Arabic text and view real-time analysis.

📊 Scientific Metrics & Results
The system was validated with the following results:

SNR (Signal-to-Noise Ratio): ~8.5 dB (indicating high neural synthesis quality).

Prosody: Stable and natural-sounding results without significant artifacts.

Extraction: Automated MFCC extraction and Mel-spectrogram distance calculation.

⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/lazyAspirations/Arabic-Neural-TTS.git
cd Arabic-Neural-TTS
Install Dependencies:

Bash
pip install streamlit torch numpy scipy librosa matplotlib
Launch the Interface:

Bash
streamlit run app.py
📂 Project Structure
app.py: Main Streamlit application.

TTSEvaluator.py: Scientific evaluation module.

Rapport_Scientifique.pdf: Detailed documentation and methodology.

output/: Generated WAV files and JSON evaluation reports.
