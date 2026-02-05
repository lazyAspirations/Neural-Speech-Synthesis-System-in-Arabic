"""
Streamlit Web Interface for Arabic Neural TTS
Part 1: Pre-loaded examples with evaluation
Part 2: Free user input for custom synthesis
"""

import streamlit as st
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import config
from model_loader import VITSModelLoader
from synthesis import ArabicTTSSynthesizer
from evaluation import TTSEvaluator
from data_processor import ArabicTextProcessor

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Arabic TTS Synthesis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL (cached)
# ============================================================
@st.cache_resource
def load_pipeline_cached():
    """Load VITS model and synthesizer once and cache it"""
    # Initialize model loader with defaults: VITS, HiFi-GAN, Male voice, Neutral
    model_loader = VITSModelLoader(
        model_name="VITS",
        vocoder_name="HiFi-GAN",
        use_prosody=True,
        voice="Male"
    )
    model_loader.load_model()
    
    # Set neutral emotion
    if model_loader.prosody:
        model_loader.set_prosody_emotion("neutral")
    
    # Initialize synthesizer
    synthesizer = ArabicTTSSynthesizer(model_loader)
    
    # Initialize evaluator
    evaluator = TTSEvaluator()
    
    return model_loader, synthesizer, evaluator

# ============================================================
# HEADER
# ============================================================
st.markdown("""
# 🎙️ Arabic Neural Text-to-Speech Synthesis
## VITS Model Interface with Quality Evaluation

---
""")

# ============================================================
# LOAD MODEL
# ============================================================
st.info("🔄 Loading VITS model (first time: ~10 seconds, then cached)...")
try:
    model_loader, synthesizer, evaluator = load_pipeline_cached()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# ============================================================
# PRE-LOADED PHRASES (3 examples)
# ============================================================
EXAMPLE_PHRASES = [
    {
        "text": "مَرْحَبًا بِكُمْ فِي هَذَا الْمَشْرُوعِ الصَّوْتِيِّ",
        "translation": "Welcome to this audio project",
        "id": "example_1"
    },
    {
        "text": "الذَّكاءُ الاصْطِنَاعِيُّ يُغَيِّرُ الْعَالَمَ مِنْ حَوْلِنا",
        "translation": "Artificial Intelligence is changing the world around us",
        "id": "example_2"
    },
    {
        "text": "الصَّوْتُ الْبَشَرِيُّ يَحْمِلُ الْكَثِيرَ مِنَ الْمَشَاعِرِ",
        "translation": "The human voice carries a lot of emotions",
        "id": "example_3"
    }
]

# ============================================================
# TABS: Part 1 vs Part 2
# ============================================================
tab1, tab2 = st.tabs(["Pre-loaded Examples", "Custom Synthesis"])

# ============================================================
# PART 1: PRE-LOADED EXAMPLES
# ============================================================
with tab1:
    st.markdown("### Pre-loaded Arabic Phrases with Evaluation")
    st.markdown("Three example sentences already prepared for you:")
    
    # Create 3 columns for the 3 examples
    cols = st.columns(3)
    
    for idx, (col, phrase) in enumerate(zip(cols, EXAMPLE_PHRASES)):
        with col:
            st.markdown(f"""
            <div style="background: #1f77b4; padding: 15px; border-radius: 8px; height: 100%; color: white;">
                <h4 style="color: white;">{idx + 1}</h4>
                <p style="font-size: 16px; font-weight: bold; direction: rtl; text-align: right; color: white;">
                    {phrase['text']}
                </p>
                <p style="font-size: 12px; color: #e0e0e0;">
                    <i>{phrase['translation']}</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Synthesis & Evaluation for each example
    st.markdown("### Synthesis Results")
    
    result_cols = st.columns(3)
    
    for idx, (col, phrase) in enumerate(zip(result_cols, EXAMPLE_PHRASES)):
        with col:
            st.markdown(f"**Example {idx + 1}**")
            
            # Check if audio already exists
            audio_file = config.output_dir / f"phrase_{idx + 1}.wav"
            
            if st.button(f"🎵 Synthesize Example {idx + 1}", key=f"synth_{idx}"):
                with st.spinner(f"Synthesizing: {phrase['text'][:20]}..."):
                    try:
                        # Synthesize
                        result = synthesizer.synthesize(phrase['text'])
                        
                        if result and result.get('audio') is not None:
                            audio_data = result['audio']
                            
                            # Debug: Show audio info
                            audio_data = np.asarray(audio_data).flatten().astype(np.float32)
                            
                            if len(audio_data) == 0:
                                st.error("❌ Audio is empty!")
                            else:
                                # Store in session
                                st.session_state[f"audio_{idx}"] = audio_data
                                st.session_state[f"sr_{idx}"] = result['sampling_rate']
                                
                                st.success(f"✅ Audio generated! ({len(audio_data)} samples)")
                                
                                # Evaluate
                                with st.spinner("Evaluating audio quality..."):
                                    try:
                                        metrics = evaluator.evaluate_synthesis(
                                            audio_data,
                                            result['sampling_rate'],
                                            phrase['text']
                                        )
                                    except Exception as e:
                                        st.warning(f"⚠️ Evaluation metrics not available: {str(e)}")
                                        # Provide basic metrics
                                        metrics = {
                                            'snr': 0.0,
                                            'thd': 0.0,
                                            'crest_factor': 0.0,
                                            'spectral_centroid': 0.0,
                                            'rms_energy': 0.0,
                                            'zcr': 0.0
                                        }
                                
                                st.session_state[f"metrics_{idx}"] = metrics
                        else:
                            st.error("❌ Synthesis failed - no audio generated")
                    except Exception as e:
                        st.error(f"❌ Error during synthesis: {str(e)}")
    
    st.markdown("---")
    
    # Display results
    st.markdown("### Audio & Metrics")
    
    audio_cols = st.columns(3)
    
    for idx, col in enumerate(audio_cols):
        with col:
            st.markdown(f"**Result {idx + 1}**")
            
            if f"audio_{idx}" in st.session_state:
                # Audio player
                audio_data = st.session_state[f"audio_{idx}"]
                sr = st.session_state[f"sr_{idx}"]
                
                st.audio(audio_data, sample_rate=sr)
                
                # Metrics
                metrics = st.session_state[f"metrics_{idx}"]
                
                st.markdown(f"""
                <div class="metric-card">
                    <b>Quality Metrics:</b><br>
                    🔊 SNR: {metrics.get('snr', 0):.2f} dB<br>
                    📈 THD: {metrics.get('thd', 0):.2f} %<br>
                    📊 Crest Factor: {metrics.get('crest_factor', 0):.2f}<br>
                    🎵 Spectral Centroid: {metrics.get('spectral_centroid', 0):.0f} Hz
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"Click 'Synthesize Example {idx + 1}' to generate audio")

# ============================================================
# PART 2: CUSTOM SYNTHESIS
# ============================================================
with tab2:
    st.markdown("### Free Text Input - Create Your Own Audio")
    st.markdown("Enter any Arabic phrase and get instant synthesis + evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        custom_text = st.text_area(
            "Enter Arabic text:",
            placeholder="اكتب نصاً عربياً هنا...",
            height=100,
            key="custom_text"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        
        synthesize_btn = st.button(
            "🎵 Synthesize",
            use_container_width=True,
            key="custom_synth_btn"
        )
    
    if synthesize_btn:
        if not custom_text.strip():
            st.error("❌ Please enter Arabic text!")
        else:
            st.markdown("---")
            
            # Synthesize
            with st.spinner("🔄 Synthesizing your text..."):
                try:
                    result = synthesizer.synthesize(custom_text)
                    
                    if not result or result.get('audio') is None:
                        st.error("❌ Synthesis failed - no audio generated")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Synthesis error: {str(e)}")
                    st.stop()
            
            # Prepare audio
            audio_data = np.asarray(result['audio']).flatten().astype(np.float32)
            sr = result['sampling_rate']
            
            if len(audio_data) == 0:
                st.error("❌ Audio is empty!")
                st.stop()
            
            st.markdown("### 🎵 Audio Output")
            st.info(f"📊 Audio: {len(audio_data)} samples @ {sr} Hz")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.audio(audio_data, sample_rate=sr)
            
            # Evaluate
            with st.spinner("📊 Evaluating audio quality..."):
                try:
                    metrics = evaluator.evaluate_synthesis(
                        audio_data,
                        sr,
                        custom_text
                    )
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"EVALUATION ERROR: {error_msg}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    # Provide basic metrics
                    metrics = {
                        'snr': 0.0,
                        'thd': 0.0,
                        'crest_factor': 0.0,
                        'spectral_centroid': 0.0,
                        'rms_energy': 0.0,
                        'zcr': 0.0
                    }
            
            with col2:
                st.markdown("### 📈 Quality Metrics")
                
                metric_cols = st.columns(2)
                
                with metric_cols[0]:
                    st.metric("SNR (dB)", f"{metrics.get('snr', 0):.2f}")
                    st.metric("Crest Factor", f"{metrics.get('crest_factor', 0):.2f}")
                
                with metric_cols[1]:
                    st.metric("THD (%)", f"{metrics.get('thd', 0):.2f}")
                    st.metric("ZCR", f"{metrics.get('zcr', 0):.6f}")
                
            
            st.markdown("---")
            
            # Detailed metrics
            st.markdown("### 📊 Detailed Analysis")
            
            details_col1, details_col2, details_col3 = st.columns(3)
            
            with details_col1:
                st.markdown(f"""
                **Spectral Features:**
                - Spectral Centroid: {metrics.get('spectral_centroid', 0):.0f} Hz
                - RMS Energy: {metrics.get('rms_energy', 0):.4f}
                """)
            
            with details_col2:
                st.markdown(f"""
                **Quality Indicators:**
                - SNR: {metrics.get('snr', 0):.2f} dB
                - THD: {metrics.get('thd', 0):.2f} %
                """)
            
            with details_col3:
                # Quality assessment
                snr = metrics.get('snr', 0)
                if snr > 10:
                    quality = "✅ Excellent"
                    color = "green"
                elif snr > 7:
                    quality = "✓ Good"
                    color = "blue"
                else:
                    quality = "◐ Acceptable"
                    color = "orange"
                
                st.markdown(f"""
                **Overall Quality:**
                :{color}[{quality}]
                
                SNR > 10 dB = Excellent
                SNR > 7 dB = Good
                SNR < 7 dB = Acceptable
                """)
            
            # Save option
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Download audio
                import io
                audio_bytes = audio_data.astype(np.float32).tobytes()
                st.download_button(
                    label="📥 Download Audio (WAV)",
                    data=audio_bytes,
                    file_name=f"tts_output_{timestamp}.wav",
                    mime="audio/wav"
                )
            
            with col2:
                # Download metrics
                metrics_json = json.dumps(metrics, indent=2)
                st.download_button(
                    label="📊 Download Metrics (JSON)",
                    data=metrics_json,
                    file_name=f"metrics_{timestamp}.json",
                    mime="application/json"
                )
            
            st.success("✅ Synthesis and evaluation complete!")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    🎙️ Arabic Neural TTS System | VITS Model | Scientific Evaluation
    <br>
    Created with PyTorch, Streamlit & HuggingFace Transformers
</div>
""", unsafe_allow_html=True)
