"""
Module d'évaluation scientifique de la synthèse vocale
Métriques de qualité, intelligibilité et naturalité
"""

import logging
import numpy as np
import librosa
from typing import Dict, List, Optional
from scipy import signal
from scipy.stats import pearsonr
from pathlib import Path

from config import config


logger = logging.getLogger(__name__)


class AudioQualityMetrics:
    """Classe pour calculer les métriques de qualité audio"""
    
    @staticmethod
    def calculate_snr(audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculer le rapport signal-bruit (SNR)
        
        Args:
            audio: Signal audio
            sample_rate: Taux d'échantillonnage
            
        Returns:
            SNR en dB
        """
        # Safety checks
        if audio is None or len(audio) == 0:
            return 0.0
        
        # Flatten if multidimensional
        audio = np.asarray(audio).flatten().astype(np.float32)
        
        if len(audio) < 4:
            return 0.0
        
        # Normaliser
        max_val = np.abs(audio).max() + 1e-8
        audio = audio / max_val
        
        # Use spectral approach for SNR
        try:
            # Compute power spectral density
            freqs, psd = signal.welch(audio, fs=sample_rate, nperseg=min(256, len(audio)))
            
            # Signal power (total)
            signal_power = np.sum(psd)
            
            # Estimate noise as the lowest power level
            noise_floor = np.percentile(psd, 10)
            
            # SNR calculation
            snr_db = 10 * np.log10((signal_power - noise_floor) / (noise_floor + 1e-10))
            
            # Clamp to realistic range
            snr_db = float(np.clip(snr_db, -20, 40))
            return snr_db
        except:
            return 0.0
    
    @staticmethod
    def calculate_thd(audio: np.ndarray) -> float:
        """
        Calculer la distorsion harmonique totale (THD)
        
        Args:
            audio: Signal audio
            
        Returns:
            THD en pourcentage
        """
        # Safety checks
        if audio is None or len(audio) == 0:
            return 0.0
        
        try:
            # Flatten and ensure float32
            audio = np.asarray(audio).flatten().astype(np.float32)
            
            if len(audio) < 4:
                return 0.0
            
            # Normalize audio
            max_val = np.abs(audio).max() + 1e-8
            audio = audio / max_val
            
            # Apply window to reduce spectral leakage
            window = signal.windows.hann(len(audio))
            audio_windowed = audio * window
            
            # FFT
            fft = np.fft.fft(audio_windowed)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Safety check for empty magnitude
            if len(magnitude) < 2 or magnitude.sum() == 0:
                return 0.0
            
            # Find fundamental frequency (first peak after DC)
            magnitude_smooth = signal.savgol_filter(magnitude, window_length=min(11, len(magnitude)//2*2+1), polyorder=2)
            fundamental_idx = np.argmax(magnitude_smooth[1:]) + 1
            
            if fundamental_idx < 1 or magnitude[fundamental_idx] < 1e-6:
                return 0.0
            
            # Calculate THD
            # Total harmonic power = all components - fundamental
            fundamental_power = magnitude[fundamental_idx]**2
            total_power = np.sum(magnitude**2)
            harmonic_power = total_power - fundamental_power
            
            # THD = sqrt(harmonic_power) / fundamental
            thd_percent = 100 * np.sqrt(harmonic_power) / (fundamental_power + 1e-10)
            
            # Clamp between 0-100%
            return float(np.clip(thd_percent, 0, 100))
        except Exception as e:
            return 0.0
    
    @staticmethod
    def calculate_crest_factor(audio: np.ndarray) -> float:
        """
        Calculer le facteur de crête (Crest Factor)
        Rapport entre pic et RMS
        
        Args:
            audio: Signal audio
            
        Returns:
            Facteur de crête en dB
        """
        # Safety checks
        if audio is None or len(audio) == 0:
            return 0.0
        
        audio = np.asarray(audio).flatten().astype(np.float32)
        
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        crest_factor = 20 * np.log10(peak / (rms + 1e-10))
        return float(crest_factor)


class ProsodicAnalysis:
    """Analyse de la prosodie (pitch, duration, etc.)"""
    
    @staticmethod
    def extract_pitch(audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Extraire le pitch du signal audio
        
        Args:
            audio: Signal audio
            sample_rate: Taux d'échantillonnage
            
        Returns:
            Dictionnaire avec statistiques de pitch
        """
        # Utiliser librosa pour l'extraction de pitch
        S = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Chroma features (représentation du pitch)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
        
        return {
            "chroma_mean": float(np.mean(chroma)),
            "chroma_std": float(np.std(chroma)),
            "chroma_max": float(np.max(chroma)),
            "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
        }
    
    @staticmethod
    def analyze_energy(audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Analyser l'énergie du signal
        
        Args:
            audio: Signal audio
            sample_rate: Taux d'échantillonnage
            
        Returns:
            Dictionnaire avec statistiques d'énergie
        """
        frame_length = int(sample_rate * 0.02)
        hop_length = frame_length // 2
        
        S = librosa.feature.melspectrogram(y=audio, sr=sample_rate,
                                           n_fft=frame_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Energie moyenne et variance
        energy = np.sum(S, axis=0)
        
        return {
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "energy_max": float(np.max(energy)),
            "energy_min": float(np.min(energy))
        }
    
    @staticmethod
    def calculate_duration(audio: np.ndarray, sample_rate: int) -> float:
        """Calculer la durée en secondes"""
        return float(len(audio) / sample_rate)


class TTSEvaluator:
    """Évaluateur complet pour la synthèse vocale TTS"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_synthesis(self, audio: np.ndarray, sample_rate: int,
                          text: str, ground_truth: Optional[np.ndarray] = None) -> Dict:
        """
        Évaluer une synthèse complète
        
        Args:
            audio: Signal synthétisé
            sample_rate: Taux d'échantillonnage
            text: Texte synthétisé
            ground_truth: Audio de référence (optionnel)
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        # Safety: ensure audio is valid
        if audio is None or len(audio) == 0:
            return {
                "text": text,
                "sample_rate": sample_rate,
                "timestamp": str(np.datetime64('now')),
                "snr": 0.0,
                "thd": 0.0,
                "crest_factor": 0.0,
                "spectral_centroid": 0.0,
                "rms_energy": 0.0,
                "zcr": 0.0
            }
        
        audio = np.asarray(audio).flatten().astype(np.float32)
        
        evaluation_result = {
            "text": text,
            "sample_rate": sample_rate,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Safely calculate each metric
        try:
            evaluation_result["snr"] = float(AudioQualityMetrics.calculate_snr(audio, sample_rate))
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            evaluation_result["snr"] = 0.0
        
        try:
            evaluation_result["thd"] = float(AudioQualityMetrics.calculate_thd(audio))
        except Exception as e:
            logger.warning(f"THD calculation failed: {e}")
            evaluation_result["thd"] = 0.0
        
        try:
            evaluation_result["crest_factor"] = float(AudioQualityMetrics.calculate_crest_factor(audio))
        except Exception as e:
            logger.warning(f"Crest factor calculation failed: {e}")
            evaluation_result["crest_factor"] = 0.0
        
        try:
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            evaluation_result["spectral_centroid"] = float(np.mean(spec_centroid))
        except Exception as e:
            logger.warning(f"Spectral centroid calculation failed: {e}")
            evaluation_result["spectral_centroid"] = 0.0
        
        try:
            energy_analysis = ProsodicAnalysis.analyze_energy(audio, sample_rate)
            evaluation_result["rms_energy"] = float(energy_analysis.get('energy_mean', 0.0))
        except Exception as e:
            logger.warning(f"Energy analysis failed: {e}")
            evaluation_result["rms_energy"] = 0.0
        
        try:
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
            evaluation_result["zcr"] = zcr
        except Exception as e:
            logger.warning(f"ZCR calculation failed: {e}")
            evaluation_result["zcr"] = 0.0
        
        logger.info(f"Qualité audio: SNR={evaluation_result['snr']:.2f}dB, THD={evaluation_result['thd']:.2f}%")
        
        # Analyse prosodique
        try:
            evaluation_result["prosody"] = {
                "pitch_analysis": ProsodicAnalysis.extract_pitch(audio, sample_rate),
                "energy_analysis": ProsodicAnalysis.analyze_energy(audio, sample_rate),
                "duration_seconds": ProsodicAnalysis.calculate_duration(audio, sample_rate)
            }
        except Exception as e:
            logger.warning(f"Prosody analysis failed: {e}")
            evaluation_result["prosody"] = {}
        
        # Caractéristiques spectrales
        try:
            evaluation_result["spectral_features"] = self._extract_spectral_features(audio, sample_rate)
        except Exception as e:
            logger.warning(f"Spectral features extraction failed: {e}")
            evaluation_result["spectral_features"] = {}
        
        # Comparaison avec référence si disponible
        if ground_truth is not None:
            try:
                evaluation_result["comparison_with_reference"] = \
                    self._compare_with_reference(audio, ground_truth, sample_rate)
            except Exception as e:
                logger.warning(f"Reference comparison failed: {e}")
        
        self.results.append(evaluation_result)
        return evaluation_result
    
    def _extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Extraire les features spectrales"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        return {
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "zero_crossing_rate_std": float(np.std(zcr)),
            "spectral_centroid_mean": float(np.mean(spec_centroid)),
            "spectral_centroid_std": float(np.std(spec_centroid)),
            "spectral_rolloff_mean": float(np.mean(spec_rolloff)),
            "mfcc_mean": [float(m) for m in np.mean(mfcc, axis=1)]
        }
    
    def _compare_with_reference(self, synthesized: np.ndarray,
                               reference: np.ndarray,
                               sample_rate: int) -> Dict:
        """Comparer l'audio synthétisé avec la référence"""
        # Résampler si nécessaire
        if len(synthesized) != len(reference):
            # Pad or trim
            min_len = min(len(synthesized), len(reference))
            synthesized = synthesized[:min_len]
            reference = reference[:min_len]
        
        # Calcul de distance
        mse = float(np.mean((synthesized - reference)**2))
        mae = float(np.mean(np.abs(synthesized - reference)))
        
        # Correlation
        correlation = float(np.corrcoef(synthesized, reference)[0, 1])
        
        return {
            "mse": mse,
            "mae": mae,
            "correlation": correlation,
            "similarity_percent": (correlation + 1) / 2 * 100  # Normaliser 0-100
        }
    
    def generate_report(self) -> Dict:
        """Générer un rapport d'évaluation"""
        if not self.results:
            logger.warning("Pas de résultats à évaluer")
            return {}
        
        report = {
            "total_samples": len(self.results),
            "samples": self.results,
            "summary": self._calculate_summary_stats()
        }
        
        return report
    
    def _calculate_summary_stats(self) -> Dict:
        """Calculer les statistiques résumées"""
        summary = {}
        
        if not self.results:
            return summary
        
        # Statistiques de qualité audio
        if all("audio_quality" in r for r in self.results):
            snr_values = [r["audio_quality"]["snr_db"] for r in self.results]
            thd_values = [r["audio_quality"]["thd_percent"] for r in self.results]
            
            summary["audio_quality_summary"] = {
                "snr_mean_db": float(np.mean(snr_values)),
                "snr_std_db": float(np.std(snr_values)),
                "thd_mean_percent": float(np.mean(thd_values)),
                "thd_std_percent": float(np.std(thd_values))
            }
        
        # Statistiques de durée
        durations = [r["prosody"]["duration_seconds"] for r in self.results]
        summary["duration_stats"] = {
            "mean_seconds": float(np.mean(durations)),
            "std_seconds": float(np.std(durations)),
            "min_seconds": float(np.min(durations)),
            "max_seconds": float(np.max(durations))
        }
        
        return summary
    
    def save_results(self, output_path: Path):
        """Sauvegarder les résultats en JSON"""
        import json
        
        report = self.generate_report()
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Résultats sauvegardés: {output_path}")


def create_evaluator() -> TTSEvaluator:
    """Créer rapidement un évaluateur"""
    return TTSEvaluator()
