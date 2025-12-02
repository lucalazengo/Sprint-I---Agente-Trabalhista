"""
Analytics & Message Queue Service
=================================
Gerencia a persistência de interações e feedbacks.
Funciona como uma "Fila de Mensagens" (Log Append-Only) para:
1. Armazenar histórico de conversas.
2. Capturar feedback do usuário (Thumbs Up/Down).
3. Criar dataset para futuro Fine-Tuning.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from src.utils import config

logger = logging.getLogger(__name__)

# Arquivo de logs (Nossa "Fila")
LOG_FILE = config.DATA_DIR / "processed_data" / "interaction_logs.jsonl"

def log_interaction(question: str, response: str, retrieved_chunks: list, session_id: str):
    """Registra uma interação completa na fila."""
    interaction_id = str(uuid.uuid4())
    
    entry = {
        "type": "interaction",
        "interaction_id": interaction_id,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "response": response,
        "context": [c.get('artigo_numero', 'N/A') for c in retrieved_chunks] if retrieved_chunks else [],
        "feedback": None # Será preenchido depois
    }
    
    _append_to_log(entry)
    return interaction_id

def log_feedback(interaction_id: str, score: int, comment: str = ""):
    """Atualiza o feedback de uma interação específica."""
    # Em um sistema real (SQL/NoSQL), faríamos um UPDATE.
    # Em arquivo JSONL, adicionamos um novo evento de feedback linkado pelo ID.
    entry = {
        "type": "feedback",
        "interaction_id": interaction_id,
        "timestamp": datetime.utcnow().isoformat(),
        "score": score, # 1 (Like) ou 0 (Dislike)
        "comment": comment
    }
    _append_to_log(entry)

def _append_to_log(data: dict):
    """Escreve na 'fila' (arquivo) de forma segura."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Falha ao salvar log de analytics: {e}")




