"""
RAG Evaluation Script
=====================
Evaluates the Labor Law Agent using 'LLM-as-a-Judge' methodology.

Metrics:
- Correctness (Corretude): Accuracy against the Ground Truth.
- Faithfulness (Fidelidade): Adherence to the retrieved context (hallucination check).

Usage:
    python -m src.backend.eval_rag
"""

import json
import logging
import statistics
from typing import Dict, List
from pathlib import Path

from src.backend.agent import LaborLawAgent
from src.utils import config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG_Eval")

GABARITO_PATH = config.DATA_DIR / "gabarito_teste.json"
RESULTS_PATH = config.PROCESSED_DATA_DIR / "eval_results.json"

PROMPT_JUIZ = """
Você é um Juiz Imparcial avaliando um sistema de IA jurídica.
Analise os dados abaixo e atribua notas de 0.0 a 1.0 para:

1. **CORRETUDE (Correctness):** A resposta da IA está correta em relação à RESPOSTA ESPERADA?
   - 1.0: Totalmente correta.
   - 0.0: Totalmente incorreta.

2. **FIDELIDADE (Faithfulness):** A resposta da IA é derivada apenas do contexto fornecido (se houver busca) ou fatos gerais corretos, sem inventar leis ou artigos?
   - 1.0: Sem alucinações.
   - 0.0: Alucinação grave (inventou leis/artigos).

DADOS DA AVALIAÇÃO:
- **Pergunta:** {pergunta}
- **Resposta Esperada (Gabarito):** {gabarito}
- **Resposta da IA:** {resposta_ia}
- **Log de Contexto (RAG):** {contexto}

SAÍDA (Formato JSON obrigatório):
{{
    "corretude": <nota_float>,
    "fidelidade": <nota_float>,
    "justificativa": "<breve explicação>"
}}
"""

class RAGEvaluator:
    def __init__(self):
        self.agent = LaborLawAgent()
        self.judge_client = self.agent.client  # Reuse the same client (Maritaca)
        
    def load_gabarito(self) -> List[Dict]:
        with open(GABARITO_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_single_interaction(self, item: Dict) -> Dict:
        question = item['pergunta']
        expected = item['resposta_esperada']
        
        logger.info(f"Avaliacao da Questão {item['id']}: {question}")
        
        # 1. Run Agent
        # Empty history for independent evaluation
        response_ia, log_trace = self.agent.run(question, [])
        
        # 2. Run Judge
        prompt_formatted = PROMPT_JUIZ.format(
            pergunta=question,
            gabarito=expected,
            resposta_ia=response_ia,
            contexto=log_trace[:2000] # Truncate log to avoid context overflow if huge
        )
        
        try:
            judge_resp = self.judge_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_formatted}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            eval_json = json.loads(judge_resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Erro no Juiz LLM: {e}")
            eval_json = {"corretude": 0.0, "fidelidade": 0.0, "justificativa": "Erro no Juiz"}

        result = {
            "id": item['id'],
            "pergunta": question,
            "resposta_ia": response_ia,
            "gabarito": expected,
            "avaliacao": eval_json
        }
        
        logger.info(f" -> Notas: Corretude={eval_json.get('corretude')} | Fidelidade={eval_json.get('fidelidade')}")
        return result

    def run_evaluation(self):
        logger.info("Iniciando Avaliação RAG...")
        data = self.load_gabarito()
        results = []
        
        for item in data:
            res = self.evaluate_single_interaction(item)
            results.append(res)
            
        # Calculate Aggregates
        scores_corretude = [r['avaliacao'].get('corretude', 0) for r in results]
        scores_fidelidade = [r['avaliacao'].get('fidelidade', 0) for r in results]
        
        avg_corretude = statistics.mean(scores_corretude) if scores_corretude else 0
        avg_fidelidade = statistics.mean(scores_fidelidade) if scores_fidelidade else 0
        
        report = {
            "summary": {
                "total_questions": len(results),
                "avg_correctness": avg_corretude,
                "avg_faithfulness": avg_fidelidade
            },
            "details": results
        }
        
        # Save Results
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logger.info("--- RELATÓRIO FINAL ---")
        logger.info(f"Questões Avaliadas: {len(results)}")
        logger.info(f"Média Corretude: {avg_corretude:.2f}")
        logger.info(f"Média Fidelidade: {avg_fidelidade:.2f}")
        logger.info(f"Resultados salvos em: {RESULTS_PATH}")

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()

