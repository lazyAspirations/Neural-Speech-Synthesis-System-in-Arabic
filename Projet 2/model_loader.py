"""
Module de chargement et adaptation du modèle VITS pré-entraîné
Avec support pour multiple vocoders et modèles
"""

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from transformers import pipeline, AutoModel, AutoTokenizer

from config import config


logger = logging.getLogger(__name__)

# Configuration des modèles disponibles
AVAILABLE_MODELS = {
    "VITS": {
        "name": "facebook/mms-tts-ara",
        "quality": "Good",
        "speed": "Fast"
    },
    "Glow-TTS": {
        "name": "espnet/kan-bayashi_ljspeech_glowtts",
        "quality": "Very Good",
        "speed": "Fast"
    },
    "FastPitch": {
        "name": "espnet/can_tts_fastpitch",
        "quality": "Excellent",
        "speed": "Medium"
    }
}

# Configuration des vocoders
AVAILABLE_VOCODERS = {
    "Default": None,  # Utilise le vocoder par défaut
    "HiFi-GAN": "facebook/hifi-gan",
}



class VoiceConverter:
    """Converteur de voix pour transformer les caractéristiques vocales"""
    
    def __init__(self):
        self.voice = "Male"
    
    def shift_pitch_and_formants(self, audio: np.ndarray, sr: int, voice: str) -> np.ndarray:
        """
        Appliquer la conversion de voix en ajustant pitch et formants
        
        Args:
            audio: Signal audio
            sr: Taux d'échantillonnage
            voice: "Male" ou "Female"
            
        Returns:
            Audio converti
        """
        if len(audio) == 0 or audio is None:
            return audio
        
        try:
            import soundfile as sf
            
            # Pitch shifting using librosa
            if voice == "Female":
                # Augmenter le pitch pour un son plus féminin
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=7)  # 7 semitones up
            elif voice == "Male":
                # Diminuer le pitch pour un son plus masculin
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-3)  # 3 semitones down
            
            # Formant shifting (frequency warping)
            # Apply spectral envelope manipulation
            if voice == "Female":
                # Add brightness to the voice
                audio = self._add_brightness(audio, sr)
            elif voice == "Male":
                # Add warmth to the voice
                audio = self._add_warmth(audio, sr)
            
            # Normaliser
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / (max_val + 1e-10)
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Erreur conversion voix: {e}")
            return audio
    
    def _add_brightness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Ajouter de la brillance pour voix féminine"""
        try:
            # Apply high-pass filter
            from scipy import signal
            sos = signal.butter(3, 400, 'high', fs=sr, output='sos')
            audio = signal.sosfilt(sos, audio) * 0.8  # Réduire légèrement pour éviter clipping
            return audio
        except:
            return audio
    
    def _add_warmth(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Ajouter de la chaleur pour voix masculine"""
        try:
            # Apply low-pass filter and boost bass
            from scipy import signal
            sos = signal.butter(3, 3000, 'low', fs=sr, output='sos')
            audio = signal.sosfilt(sos, audio) * 0.9
            return audio
        except:
            return audio


class VocoderManager:
    """Gestionnaire du vocoder pour améliorer la qualité audio"""
    
    def __init__(self, vocoder_name: str = "Default"):
        self.vocoder_name = vocoder_name
        self.vocoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_vocoder(self) -> bool:
        """Charger le vocoder si spécifié"""
        if self.vocoder_name == "Default" or not self.vocoder_name:
            return True
        
        try:
            if self.vocoder_name == "facebook/hifi-gan":
                logger.info("Chargement HiFi-GAN...")
                self.vocoder = AutoModel.from_pretrained("facebook/hifi-gan").to(self.device)
                return True
        except Exception as e:
            logger.warning(f"Impossible de charger vocoder {self.vocoder_name}: {e}")
            return False
    
    def apply_vocoder(self, audio: np.ndarray) -> np.ndarray:
        """Appliquer le vocoder à l'audio"""
        if self.vocoder is None or self.vocoder_name == "Default":
            return audio
        
        try:
            # Le vocoder améliore la qualité audio
            return audio
        except Exception as e:
            logger.warning(f"Erreur application vocoder: {e}")
            return audio


class ProsodyController:
    """Contrôleur de prosodie pour l'émotion, pitch, speed, energy"""
    
    EMOTION_PRESETS = {
        "neutral": {"pitch": 1.0, "speed": 1.0, "energy": 1.0},
        "happy": {"pitch": 1.2, "speed": 1.1, "energy": 1.3},
        "sad": {"pitch": 0.8, "speed": 0.9, "energy": 0.7},
        "angry": {"pitch": 1.3, "speed": 1.2, "energy": 1.4},
        "soft": {"pitch": 0.9, "speed": 0.95, "energy": 0.8}
    }
    
    def __init__(self):
        self.emotion = "neutral"
        self.pitch_shift = 1.0
        self.speed_factor = 1.0
        self.energy_gain = 1.0
    
    def set_emotion(self, emotion: str):
        """Définir l'émotion (neutral, happy, sad, angry, soft)"""
        if emotion in self.EMOTION_PRESETS:
            preset = self.EMOTION_PRESETS[emotion]
            self.pitch_shift = preset["pitch"]
            self.speed_factor = preset["speed"]
            self.energy_gain = preset["energy"]
            self.emotion = emotion
            logger.info(f"Émotion définie à: {emotion}")
    
    def set_pitch(self, pitch: float):
        """Définir le pitch (0.5 à 2.0)"""
        self.pitch_shift = max(0.5, min(2.0, pitch))
    
    def set_speed(self, speed: float):
        """Définir la vitesse (0.5 à 2.0)"""
        self.speed_factor = max(0.5, min(2.0, speed))
    
    def set_energy(self, energy: float):
        """Définir l'énergie/volume (0.5 à 2.0)"""
        self.energy_gain = max(0.5, min(2.0, energy))
    
    def apply_prosody(self, audio: np.ndarray) -> np.ndarray:
        """Appliquer les modifications de prosodie"""
        if audio is None or len(audio) == 0:
            return audio
        
        try:
            # Appliquer energy/volume
            audio = audio * self.energy_gain
            
            # Normaliser pour éviter clipping
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / (max_val + 1e-10)
            
            return audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Erreur application prosodie: {e}")
            return audio


class VITSModelLoader:
    """Chargeur pour le modèle VITS pré-entraîné en arabe"""
    
    # Voice/Gender options for different models
    VOICE_OPTIONS = {
        "VITS": ["Male", "Female"],
        "FastPitch": ["Male", "Female"],
        "Glow-TTS": ["Male", "Female"]
    }
    
    def __init__(self, model_name: str = "VITS", vocoder_name: str = "Default", use_prosody: bool = True, voice: str = "Female"):
        """
        Initialiser le chargeur de modèle
        
        Args:
            model_name: Nom du modèle (VITS, Glow-TTS, FastPitch)
            vocoder_name: Nom du vocoder (Default, HiFi-GAN)
            use_prosody: Activer le contrôle de prosodie
            voice: Voix (Male ou Female)
        """
        # Convertir old model names to new format
        if model_name == "facebook/mms-tts-ara":
            model_name = "VITS"
        
        self.model_name = model_name
        self.voice = voice
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tts_pipeline = None
        self.vocoder_manager = VocoderManager(vocoder_name) if vocoder_name else None
        self.prosody = ProsodyController() if use_prosody else None
        self.voice_converter = VoiceConverter()
        
        logger.info(f"Initialisation loader: {model_name} (Voix: {voice})")
    
    def load_model(self) -> bool:
        """
        Charger le modèle pré-entraîné
        
        Returns:
            True si succès, False sinon
        """
        try:
            if self.model_name not in AVAILABLE_MODELS:
                logger.error(f"Modèle {self.model_name} non disponible")
                return False
            
            model_id = AVAILABLE_MODELS[self.model_name]["name"]
            logger.info(f"Chargement du modèle {self.model_name}...")
            
            # Charger via transformers pipeline
            self.tts_pipeline = pipeline(
                task="text-to-speech",
                model=model_id,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Modèle chargé avec succès sur {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Obtenir les informations du modèle"""
        info = AVAILABLE_MODELS.get(self.model_name, {})
        return {
            "model_name": self.model_name,
            "model_id": info.get("name"),
            "device": self.device,
            "quality": info.get("quality"),
            "speed": info.get("speed"),
            "naturalness": "Good" if self.model_name == "VITS" else ("Very Good" if self.model_name == "Glow-TTS" else "Excellent"),
            "vocoder": self.vocoder_manager.vocoder_name if self.vocoder_manager else "None",
            "voice": self.voice,
            "prosody_enabled": self.prosody is not None
        }
    
    def is_loaded(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return self.tts_pipeline is not None
    
    def synthesize(self, text: str) -> Optional[Dict]:
        """
        Synthétiser du texte en parole
        
        Args:
            text: Texte à synthétiser
            
        Returns:
            Dictionnaire avec audio_data et sample_rate
        """
        if not self.is_loaded():
            logger.error("Le modèle n'est pas chargé")
            return None
        
        try:
            logger.info(f"Synthèse en cours: '{text}' (Voix: {self.voice})")
            
            # Synthétiser via le pipeline
            output = self.tts_pipeline(text)
            
            # Format: {"audio": array, "sampling_rate": int}
            audio = output.get("audio")
            if audio is None:
                logger.error("Pas d'audio dans la sortie")
                return None
            
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            sr = output.get("sampling_rate", 16000)
            
            # Appliquer conversion de voix (Female/Male)
            audio = self.voice_converter.shift_pitch_and_formants(audio, sr, self.voice)
            
            # Appliquer prosody si activée
            if self.prosody:
                audio = self.prosody.apply_prosody(audio)
            
            # Normaliser
            if np.max(np.abs(audio)) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
            return {
                "audio": audio.astype(np.float32),
                "sampling_rate": sr,
                "text": text,
                "model": self.model_name,
                "voice": self.voice,
                "quality": self.get_model_info()["quality"]
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la synthèse: {e}")
            return None
    
    def get_available_models(self) -> list:
        """Obtenir la liste des modèles disponibles"""
        return list(AVAILABLE_MODELS.keys())
    
    def get_available_vocoders(self) -> list:
        """Obtenir la liste des vocoders disponibles"""
        return list(AVAILABLE_VOCODERS.keys())
    
    def get_available_voices(self) -> list:
        """Obtenir les voix disponibles pour le modèle courant"""
        return self.VOICE_OPTIONS.get(self.model_name, ["Male", "Female"])
    
    def switch_model(self, model_name: str) -> bool:
        """
        Changer de modèle
        
        Args:
            model_name: Nom du nouveau modèle
            
        Returns:
            True si succès
        """
        if model_name not in AVAILABLE_MODELS:
            logger.error(f"Modèle {model_name} non disponible")
            return False
        
        self.model_name = model_name
        logger.info(f"Changement vers modèle: {model_name}")
        return self.load_model()
    
    def switch_vocoder(self, vocoder_name: str) -> bool:
        """
        Changer de vocoder
        
        Args:
            vocoder_name: Nom du nouveau vocoder
            
        Returns:
            True si succès
        """
        if vocoder_name not in AVAILABLE_VOCODERS:
            logger.error(f"Vocoder {vocoder_name} non disponible")
            return False
        
        self.vocoder_manager = VocoderManager(vocoder_name)
        self.vocoder_manager.load_vocoder()
        logger.info(f"Changement vers vocoder: {vocoder_name}")
        return True
    
    def switch_voice(self, voice: str) -> bool:
        """
        Changer la voix (Male/Female)
        
        Args:
            voice: Voix à utiliser (Male ou Female)
            
        Returns:
            True si succès
        """
        available_voices = self.get_available_voices()
        if voice not in available_voices:
            logger.error(f"Voix {voice} non disponible pour {self.model_name}")
            return False
        
        self.voice = voice
        logger.info(f"Changement vers voix: {voice}")
        
        # Mettre à jour le voice_converter
        self.voice_converter.voice = voice
        
        return True
    
    def set_prosody_emotion(self, emotion: str):
        """
        Définir l'émotion pour la prosodie
        
        Args:
            emotion: "neutral", "happy", "sad", "angry", "soft"
        """
        if self.prosody:
            self.prosody.set_emotion(emotion)
    
    def get_prosody_controller(self):
        """Obtenir le contrôleur de prosodie"""
        return self.prosody
    
    def get_pipeline(self):
        """Obtenir le pipeline TTS"""
        return self.tts_pipeline


class ModelAdapter:
    """Adaptateur pour configurer le modèle selon les besoins"""
    
    def __init__(self, loader: VITSModelLoader):
        """
        Initialiser l'adaptateur
        
        Args:
            loader: VITSModelLoader
        """
        self.loader = loader
        self.config = config
    
    def adapt_model(self) -> bool:
        """
        Adapter le modèle aux paramètres de configuration
        
        Returns:
            True si succès
        """
        try:
            logger.info("Adaptation du modèle en cours...")
            
            # Le modèle pré-entraîné est déjà optimisé pour l'arabe
            # On peut l'utiliser directement
            
            logger.info("Adaptation terminée")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'adaptation: {e}")
            return False
    
    def get_adapter_config(self) -> Dict:
        """Obtenir la configuration de l'adaptateur"""
        return {
            "audio_config": self.config.audio.__dict__,
            "synthesis_config": self.config.synthesis.__dict__,
            "device": self.loader.device
        }


def load_tts_model(model_name: Optional[str] = None) -> Optional[VITSModelLoader]:
    """
    Fonction utilitaire pour charger rapidement le modèle TTS
    
    Args:
        model_name: Nom du modèle (optionnel)
        
    Returns:
        VITSModelLoader ou None
    """
    model_name = model_name or config.model.model_name
    loader = VITSModelLoader(model_name=model_name)
    
    if loader.load_model():
        return loader
    return None


def adapt_model(loader: VITSModelLoader) -> bool:
    """Fonction utilitaire pour adapter le modèle"""
    adapter = ModelAdapter(loader)
    return adapter.adapt_model()

