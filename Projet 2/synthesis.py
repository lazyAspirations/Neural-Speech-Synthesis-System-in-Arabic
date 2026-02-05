"""
Module de synthèse vocale
Pipeline complet pour convertir du texte arabe en parole
"""

import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from model_loader import VITSModelLoader
from data_processor import ArabicTextProcessor, text_processor
from config import config


logger = logging.getLogger(__name__)


class ArabicTTSSynthesizer:
    """Synthétiseur TTS pour l'arabe"""
    
    def __init__(self, model_loader: VITSModelLoader):
        """
        Initialiser le synthétiseur
        
        Args:
            model_loader: Modèle chargé (VITSModelLoader)
        """
        self.model = model_loader
        self.text_processor = text_processor
        self.output_dir = config.output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Valider le texte d'entrée
        
        Args:
            text: Texte à valider
            
        Returns:
            (is_valid, message)
        """
        if not text or not text.strip():
            return False, "Le texte ne peut pas être vide"
        
        if len(text) > 500:
            return False, "Le texte est trop long (max 500 caractères)"
        
        if not ArabicTextProcessor.is_arabic(text):
            return False, "Le texte ne contient pas de caractères arabes"
        
        return True, "Texte valide"
    
    def preprocess_text(self, text: str) -> str:
        """
        Prétraiter le texte
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        cleaned = self.text_processor.clean_text(text)
        return cleaned
    
    def synthesize(self, text: str) -> Optional[Dict]:
        """
        Synthétiser le texte en parole
        
        Args:
            text: Texte arabe à synthétiser
            
        Returns:
            Dictionnaire avec audio, taux d'échantillonnage, etc.
        """
        # Valider
        is_valid, message = self.validate_input(text)
        if not is_valid:
            logger.error(f"Validation échouée: {message}")
            return None
        
        # Prétraiter
        cleaned_text = self.preprocess_text(text)
        logger.info(f"Texte prétraité: '{cleaned_text}'")
        
        # Synthétiser avec le modèle
        result = self.model.synthesize(cleaned_text)
        
        if result:
            result["original_text"] = text
            result["cleaned_text"] = cleaned_text
        
        return result
    
    def save_audio(self, audio_data: np.ndarray, sample_rate: int, 
                   filename: Optional[str] = None) -> Path:
        """
        Sauvegarder l'audio en fichier WAV
        
        Args:
            audio_data: Données audio
            sample_rate: Taux d'échantillonnage
            filename: Nom du fichier (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_output_{timestamp}.wav"
        
        output_path = self.output_dir / filename
        
        # Convertir en numpy array si nécessaire
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)
        
        # Assurer que c'est float32
        audio_data = audio_data.astype(np.float32)
        
        # Aplatir si nécessaire (VITS peut retourner (1, samples))
        if audio_data.ndim > 1:
            if audio_data.shape[0] == 1:
                audio_data = audio_data[0]  # Retirer la dimension canal si mono
            elif audio_data.shape[1] == 1:
                audio_data = audio_data[:, 0]  # Retirer la dimension canal si mono
        
        # Normaliser l'amplitude si nécessaire
        if audio_data.size > 0:
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
        
        # Sauvegarder
        sf.write(str(output_path), audio_data, sample_rate)
        logger.info(f"Audio sauvegardé: {output_path}")
        
        return output_path
    
    def synthesize_batch(self, texts: List[str], 
                        output_dir: Optional[Path] = None) -> Dict:
        """
        Synthétiser plusieurs textes
        
        Args:
            texts: Liste de textes à synthétiser
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Dictionnaire avec les résultats
        """
        output_dir = output_dir or self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {
            "total": len(texts),
            "successful": 0,
            "failed": 0,
            "outputs": []
        }
        
        for i, text in enumerate(texts, 1):
            logger.info(f"Synthèse {i}/{len(texts)}: '{text}'")
            
            result = self.synthesize(text)
            
            if result:
                # Sauvegarder
                filename = f"sentence_{i:03d}.wav"
                output_path = self.save_audio(
                    result["audio"],
                    result["sampling_rate"],
                    filename
                )
                
                results["outputs"].append({
                    "index": i,
                    "text": text,
                    "output_file": str(output_path),
                    "sample_rate": result["sampling_rate"],
                    "duration": len(result["audio"]) / result["sampling_rate"]
                })
                results["successful"] += 1
            else:
                results["failed"] += 1
                logger.warning(f"Synthèse échouée pour: '{text}'")
        
        logger.info(f"Synthèse batch terminée: {results['successful']}/{results['total']} succès")
        return results
    
    def get_synthesis_stats(self, result: Dict) -> Dict:
        """Obtenir des statistiques sur la synthèse"""
        if not result or "audio" not in result:
            return {}
        
        audio = np.array(result["audio"])
        sample_rate = result["sampling_rate"]
        
        return {
            "duration_seconds": len(audio) / sample_rate,
            "sample_rate": sample_rate,
            "num_samples": len(audio),
            "audio_min": float(audio.min()),
            "audio_max": float(audio.max()),
            "audio_mean": float(audio.mean()),
            "audio_std": float(audio.std())
        }


class SynthesisConfig:
    """Configuration avancée pour la synthèse"""
    
    def __init__(self):
        self.speaker_id = config.synthesis.speaker_id
        self.length_scale = config.synthesis.length_scale
        self.noise_scale = config.synthesis.noise_scale
        self.temperature = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "speaker_id": self.speaker_id,
            "length_scale": self.length_scale,
            "noise_scale": self.noise_scale,
            "temperature": self.temperature
        }
    
    def update(self, **kwargs):
        """Mettre à jour les paramètres"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


def create_synthesizer(model_loader: VITSModelLoader) -> ArabicTTSSynthesizer:
    """Fonction utilitaire pour créer rapidement un synthétiseur"""
    return ArabicTTSSynthesizer(model_loader)
