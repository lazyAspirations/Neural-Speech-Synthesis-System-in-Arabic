"""
Configuration centralisée du projet de synthèse vocale neuronale en arabe
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Configuration audio"""
    sample_rate: int = 22050
    num_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 40.0
    f_max: float = 7600.0


@dataclass
class ModelConfig:
    """Configuration du modèle VITS"""
    model_name: str = "facebook/mms-tts-ara"  # Modèle pré-entraîné arabe
    language_code: str = "ara"
    device: str = "cuda"  # ou "cpu"
    use_fixed_length_audio: bool = False
    max_audio_length: Optional[float] = None


@dataclass
class SynthesisConfig:
    """Configuration de synthèse"""
    speaker_id: int = 0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8


@dataclass
class EvaluationConfig:
    """Configuration d'évaluation"""
    enable_speech_quality: bool = True
    enable_mel_distance: bool = True
    enable_pitch_analysis: bool = True
    num_samples: int = 5


@dataclass
class ProjectConfig:
    """Configuration principale du projet"""
    project_name: str = "Arabic Neural TTS"
    project_version: str = "1.0.0"
    
    # Chemins
    base_dir: Path = Path(__file__).parent
    data_dir: Path = Path(__file__).parent / "data"
    output_dir: Path = Path(__file__).parent / "outputs"
    models_dir: Path = Path(__file__).parent / "models"
    results_dir: Path = Path(__file__).parent / "results"
    
    # Configurations
    audio: AudioConfig = AudioConfig()
    model: ModelConfig = ModelConfig()
    synthesis: SynthesisConfig = SynthesisConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    # Test phrases en arabe
    test_sentences: list = None
    
    def __post_init__(self):
        """Initialiser les chemins et les phrases"""
        # Créer les répertoires s'ils n'existent pas
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Phrases de test en arabe
        if self.test_sentences is None:
            self.test_sentences = [
                "مَرْحَبًا بِكُمْ فِي هَذَا الْمَشْرُوعِ الصَّوْتِيِّ",  # Bonjour, comment allez-vous ?
                "الذَّكاءُ الاصْطِنَاعِيُّ يُغَيِّرُ الْعَالَمَ مِنْ حَوْلِنا",  # L'IA change le monde autour de nous
                "هذا نموذج لتحويل النصوص إلى كلام عربي",  # Ceci est un modèle de conversion texte-parole en arabe
                "الصَّوْتُ الْبَشَرِيُّ يَحْمِلُ الْكَثِيرَ مِنَ الْمَشَاعِرِ",  # L'arabe est une belle et riche langue
                "شكراً لك على استماعك إلى هذا النص",  # Merci de nous avoir écouté
            ]
    
    def to_dict(self) -> dict:
        """Convertir la configuration en dictionnaire"""
        return {
            "project_name": self.project_name,
            "project_version": self.project_version,
            "audio": self.audio.__dict__,
            "model": self.model.__dict__,
            "synthesis": self.synthesis.__dict__,
            "evaluation": self.evaluation.__dict__,
            "test_sentences": self.test_sentences,
        }


# Instance globale
config = ProjectConfig()
