import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    # Garantir que a pasta logs existe
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('logs/api.log', maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("steam_analysis_api")
