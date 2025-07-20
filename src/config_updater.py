"""
Configuration Updater for Vocabulary Offsets

This script updates model configurations to properly handle extended text vocabulary
and shifts audio/vision vocab offsets accordingly.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigUpdater:
    """Updates model configurations with new vocabulary offsets"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
            
    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        
    def update_vocab_offsets(self, 
                           original_text_vocab_size: int,
                           new_text_vocab_size: int,
                           audio_vocab_size: int = 1024,
                           vision_vocab_size: int = 1024):
        """Update vocabulary offsets in config"""
        
        # Calculate new offsets
        text_vocab_offset = 0
        audio_vocab_offset = new_text_vocab_size
        vision_vocab_offset = new_text_vocab_size + audio_vocab_size
        total_vocab_size = new_text_vocab_size + audio_vocab_size + vision_vocab_size
        
        # Update text config
        if 'text_config' not in self.config:
            self.config['text_config'] = {}
            
        self.config['text_config'].update({
            'vocab_size': new_text_vocab_size,
            'vocab_offset': text_vocab_offset,
            'original_vocab_size': original_text_vocab_size
        })
        
        # Update audio config
        if 'audio_config' not in self.config:
            self.config['audio_config'] = {}
            
        self.config['audio_config'].update({
            'vocab_size': audio_vocab_size,
            'vocab_offset': audio_vocab_offset
        })
        
        # Update vision config
        if 'vision_config' not in self.config:
            self.config['vision_config'] = {}
            
        self.config['vision_config'].update({
            'vocab_size': vision_vocab_size,
            'vocab_offset': vision_vocab_offset
        })
        
        # Update global config
        self.config.update({
            'total_vocab_size': total_vocab_size,
            'text_vocab_offset': text_vocab_offset,
            'audio_vocab_offset': audio_vocab_offset,
            'vision_vocab_offset': vision_vocab_offset
        })
        
        logger.info("Updated vocabulary offsets:")
        logger.info(f"  Text: {text_vocab_offset} - {audio_vocab_offset - 1} (size: {new_text_vocab_size})")
        logger.info(f"  Audio: {audio_vocab_offset} - {vision_vocab_offset - 1} (size: {audio_vocab_size})")
        logger.info(f"  Vision: {vision_vocab_offset} - {total_vocab_size - 1} (size: {vision_vocab_size})")
        
    def save_config(self, output_path: str):
        """Save updated configuration to file"""
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved updated config to {output_path}")
        
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
        
    def validate_config(self) -> bool:
        """Validate the configuration has required vocab offset fields"""
        required_fields = [
            'text_vocab_offset',
            'audio_vocab_offset', 
            'vision_vocab_offset',
            'total_vocab_size'
        ]
        
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            logger.error(f"Missing required config fields: {missing_fields}")
            return False
            
        # Validate offsets are in correct order
        text_offset = self.config['text_vocab_offset']
        audio_offset = self.config['audio_vocab_offset']
        vision_offset = self.config['vision_vocab_offset']
        
        if not (text_offset < audio_offset < vision_offset):
            logger.error(f"Invalid offset order: text={text_offset}, audio={audio_offset}, vision={vision_offset}")
            return False
            
        logger.info("Configuration validation passed")
        return True


def update_model_config_file(config_path: str,
                           output_path: str,
                           original_text_vocab_size: int,
                           new_text_vocab_size: int,
                           audio_vocab_size: int = 1024,
                           vision_vocab_size: int = 1024):
    """Convenience function to update a config file"""
    
    updater = ConfigUpdater(config_path)
    updater.update_vocab_offsets(
        original_text_vocab_size=original_text_vocab_size,
        new_text_vocab_size=new_text_vocab_size,
        audio_vocab_size=audio_vocab_size,
        vision_vocab_size=vision_vocab_size
    )
    
    if updater.validate_config():
        updater.save_config(output_path)
        return True
    else:
        logger.error("Config validation failed, not saving")
        return False


def create_default_config(original_text_vocab_size: int,
                         new_text_vocab_size: int,
                         audio_vocab_size: int = 1024,
                         vision_vocab_size: int = 1024) -> Dict[str, Any]:
    """Create a default configuration with proper vocab offsets"""
    
    updater = ConfigUpdater()
    updater.update_vocab_offsets(
        original_text_vocab_size=original_text_vocab_size,
        new_text_vocab_size=new_text_vocab_size,
        audio_vocab_size=audio_vocab_size,
        vision_vocab_size=vision_vocab_size
    )
    
    return updater.get_config()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Update model config with new vocab offsets")
    parser.add_argument("--config", required=True, help="Input config file path")
    parser.add_argument("--output", required=True, help="Output config file path")
    parser.add_argument("--original-text-vocab", type=int, required=True, help="Original text vocab size")
    parser.add_argument("--new-text-vocab", type=int, required=True, help="New text vocab size")
    parser.add_argument("--audio-vocab", type=int, default=1024, help="Audio vocab size")
    parser.add_argument("--vision-vocab", type=int, default=1024, help="Vision vocab size")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    success = update_model_config_file(
        config_path=args.config,
        output_path=args.output,
        original_text_vocab_size=args.original_text_vocab,
        new_text_vocab_size=args.new_text_vocab,
        audio_vocab_size=args.audio_vocab,
        vision_vocab_size=args.vision_vocab
    )
    
    if success:
        print("Config updated successfully!")
    else:
        print("Config update failed!")
        exit(1)