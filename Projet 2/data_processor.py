"""
Module de traitement des données textuelles en arabe
Normalisation, tokenization et préparation du texte
"""

import re
from typing import List, Tuple
import unicodedata


class ArabicTextProcessor:
    """Processeur pour normaliser et préparer les textes arabes"""
    
    # Diacritiques arabes
    ARABIC_DIACRITICS = [
        '\u064B',  # FATHATAN
        '\u064C',  # DAMMATAN
        '\u064D',  # KASRATAN
        '\u064E',  # FATHA
        '\u064F',  # DAMMA
        '\u0650',  # KASRA
        '\u0651',  # SHADDA
        '\u0652',  # SUKUN
        '\u0653',  # MADDAH
        '\u0654',  # HAMZA ABOVE
        '\u0655',  # HAMZA BELOW
        '\u0656',  # SUBSCRIPT ALEF
        '\u0657',  # INVERTED DAMMA
        '\u0658',  # MARK NOON GHUNNA
        '\u0670',  # SUPERSCRIPT ALEF
    ]
    
    # Caractères arabes
    ARABIC_CHARS = re.compile(r'[\u0600-\u06FF]')
    
    def __init__(self, remove_diacritics: bool = True):
        """
        Initialiser le processeur
        
        Args:
            remove_diacritics: Supprimer les diacritiques arabes
        """
        self.remove_diacritics = remove_diacritics
    
    def remove_arabic_diacritics(self, text: str) -> str:
        """Supprimer les diacritiques arabes"""
        for diacritic in self.ARABIC_DIACRITICS:
            text = text.replace(diacritic, '')
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliser le texte arabe
        
        Args:
            text: Texte à normaliser
            
        Returns:
            Texte normalisé
        """
        # Unicode normalization (NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Supprimer les diacritiques si demandé
        if self.remove_diacritics:
            text = self.remove_arabic_diacritics(text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Supprimer les caractères spéciaux (garder l'arabe et la ponctuation)
        text = re.sub(r'[^\u0600-\u06FF\s\.,!?؛:،\-]', '', text)
        
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """Diviser le texte en phrases"""
        # Délimiteurs de phrase en arabe et français
        sentences = re.split(r'[.!?؟!٫]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def clean_text(self, text: str) -> str:
        """Nettoyer complètement le texte"""
        text = self.normalize_text(text)
        return text
    
    def process(self, text: str) -> Tuple[str, List[str]]:
        """
        Traiter un texte complet
        
        Args:
            text: Texte à traiter
            
        Returns:
            (texte_netoyé, liste_de_phrases)
        """
        cleaned_text = self.clean_text(text)
        sentences = self.split_sentences(cleaned_text)
        return cleaned_text, sentences
    
    @staticmethod
    def is_arabic(text: str) -> bool:
        """Vérifier si le texte contient du texte arabe"""
        return bool(ArabicTextProcessor.ARABIC_CHARS.search(text))
    
    @staticmethod
    def validate_arabic_text(text: str) -> bool:
        """Valider que le texte est principalement en arabe"""
        if not text:
            return False
        arabic_chars = sum(1 for c in text if ArabicTextProcessor.ARABIC_CHARS.match(c))
        return arabic_chars / len(text) > 0.7  # Au moins 70% d'arabe


class TextNormalizer:
    """Classe utilitaire pour la normalisation de texte"""
    
    @staticmethod
    def expand_abbreviations(text: str) -> str:
        """Expand common abbreviations (if needed)"""
        abbreviations = {
            'أ.د': 'أستاذ دكتور',
            'د.': 'دكتور',
            'السيد': 'السيد',
            'السيدة': 'السيدة',
        }
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        return text
    
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """Convertir les chiffres arabes en arabe parlé (optionnel)"""
        # Garder les chiffres tels quels pour cette implémentation
        return text


# Instance globale
text_processor = ArabicTextProcessor(remove_diacritics=False)
