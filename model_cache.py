"""
Système de cache pour les modèles entraînés
"""

import hashlib
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

class ModelCache:
    def __init__(self, cache_dir: str = "cache_dir", cache_ttl_hours: int = 24):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_enabled = True
        self.cache_ttl_hours = cache_ttl_hours
    
    def _generate_cache_key(self, service: str, model_type: str, data_hash: str) -> str:
        cache_key = f"{service}_{model_type}_{data_hash}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "empty_data"
        data_str = f"{df['ds'].iloc[0]}_{df['ds'].iloc[-1]}_{len(df)}_{df['y'].sum():.2f}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.total_seconds() < (self.cache_ttl_hours * 3600)
    
    def get_cached_model(self, service: str, model_type: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not self.cache_enabled:
            return None
        
        try:
            data_hash = self._get_data_hash(df)
            cache_key = self._generate_cache_key(service, model_type, data_hash)
            cache_file = self.cache_dir / f"{cache_key}.joblib"
            
            if self._is_cache_valid(cache_file):
                self.logger.info(f"📦 Modèle {model_type} pour {service} chargé depuis le cache")
                return joblib.load(cache_file)
            else:
                self.logger.info(f"⏰ Cache expiré pour {model_type} - {service}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur cache {model_type} - {service}: {e}")
            return None
    
    def save_model_to_cache(self, service: str, model_type: str, df: pd.DataFrame, model_data: Dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        
        try:
            data_hash = self._get_data_hash(df)
            cache_key = self._generate_cache_key(service, model_type, data_hash)
            cache_file = self.cache_dir / f"{cache_key}.joblib"
            
            joblib.dump(model_data, cache_file)
            self.logger.info(f"💾 Modèle {model_type} pour {service} sauvegardé")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur sauvegarde cache {model_type} - {service}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        try:
            cache_files = list(self.cache_dir.glob("*.joblib"))
            total_size = sum(f.stat().st_size for f in cache_files)
            valid_files = [f for f in cache_files if self._is_cache_valid(f)]
            
            return {
                'total_files': len(cache_files),
                'valid_files': len(valid_files),
                'expired_files': len(cache_files) - len(valid_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur stats cache: {e}")
            return {}
    
    def clear_all_cache(self) -> None:
        """Supprime tout le cache."""
        try:
            cache_files = list(self.cache_dir.glob("*.joblib"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            self.logger.info(f"🗑️ Cache complètement vidé ({len(cache_files)} fichiers)")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors de la suppression du cache: {e}")
    
    def _clear_expired_cache(self) -> None:
        """Nettoie le cache expiré."""
        try:
            expired_files = []
            for cache_file in self.cache_dir.glob("*.joblib"):
                if not self._is_cache_valid(cache_file):
                    expired_files.append(cache_file)
            
            for expired_file in expired_files:
                expired_file.unlink()
            
            if expired_files:
                self.logger.info(f"🧹 {len(expired_files)} fichiers de cache expirés supprimés")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur lors du nettoyage du cache: {e}")
