"""
Internationalization (i18n) module for PDFMathTranslate
Supports multiple languages for the GUI interface
"""

import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class I18nManager:
    """Manages translations for the application"""

    # Supported languages
    SUPPORTED_LANGUAGES = {
        "English": "en_US",
        "简体中文 (Simplified Chinese)": "zh_CN",
        "繁體中文 (Traditional Chinese)": "zh_TW",
        "日本語 (Japanese)": "ja_JP",
        "한국어 (Korean)": "ko_KR",
        "Français (French)": "fr_FR",
        "Deutsch (German)": "de_DE",
        "Español (Spanish)": "es_ES",
        "Русский (Russian)": "ru_RU",
        "Italiano (Italian)": "it_IT",
    }

    def __init__(self, language: str = "English"):
        """
        Initialize the i18n manager
        
        Args:
            language: The default language to use
        """
        self._translations: Dict[str, Dict[str, str]] = {}
        self._current_language = language
        self._translations_dir = Path(__file__).parent / "translations"
        self._load_all_translations()

    def _load_all_translations(self):
        """Load all available translations"""
        if not self._translations_dir.exists():
            self._translations_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Translations directory created: {self._translations_dir}")

        for lang_name, lang_code in self.SUPPORTED_LANGUAGES.items():
            translation_file = self._translations_dir / f"{lang_code}.json"
            if translation_file.exists():
                try:
                    with open(translation_file, "r", encoding="utf-8") as f:
                        self._translations[lang_code] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load translation file {translation_file}: {e}")
                    self._translations[lang_code] = {}
            else:
                self._translations[lang_code] = {}

    def set_language(self, language: str):
        """
        Set the current language
        
        Args:
            language: The language name from SUPPORTED_LANGUAGES
        """
        if language in self.SUPPORTED_LANGUAGES:
            self._current_language = language
        else:
            logger.warning(f"Language {language} not supported, using English")
            self._current_language = "English"

    def get_current_language_code(self) -> str:
        """Get the current language code"""
        return self.SUPPORTED_LANGUAGES.get(self._current_language, "en_US")

    def get_current_language(self) -> str:
        """Get the current language name"""
        return self._current_language

    def translate(self, key: str, default: str = "") -> str:
        """
        Get translation for a key
        
        Args:
            key: The translation key
            default: Default text if translation not found
            
        Returns:
            The translated text or default
        """
        lang_code = self.get_current_language_code()
        translations = self._translations.get(lang_code, {})
        
        if key in translations:
            return translations[key]
        
        # Fallback to English if key not found in current language
        if lang_code != "en_US":
            en_translations = self._translations.get("en_US", {})
            if key in en_translations:
                return en_translations[key]
        
        logger.debug(f"Translation key not found: {key}")
        return default or key

    def __call__(self, key: str, default: str = "") -> str:
        """Allow using the manager as a callable"""
        return self.translate(key, default)

    def get_all_languages(self) -> list:
        """Get list of all supported languages"""
        return list(self.SUPPORTED_LANGUAGES.keys())


# Global i18n manager instance
_i18n_instance: Optional[I18nManager] = None


def get_i18n_manager(language: str = "English") -> I18nManager:
    """Get the global i18n manager instance"""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = I18nManager(language)
    return _i18n_instance


def set_language(language: str):
    """Set the global language"""
    manager = get_i18n_manager()
    manager.set_language(language)


def t(key: str, default: str = "") -> str:
    """Translation shortcut function"""
    manager = get_i18n_manager()
    return manager.translate(key, default)
