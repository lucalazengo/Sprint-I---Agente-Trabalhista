"""
Agent Module
============
Implements the 'Brain' of the Labor Law Agent.
Architecture: Self-Correcting Loop (Decision -> Draft -> Critic).
"""

import json
import logging
from typing import List, Tuple, Dict, Optional
from openai import OpenAI

from src.utils import config
from src.backend.retriever import Retriever

logger = logging.getLogger(__name__)

# --- Prompts ---
SYSTEM_PROMPT_AGENT = """
Você é um agente jurídico empático e profissional. Sua tarefa é analisar a conversa com o usuário.
- Se for um cumprimento ou conversa fiada ("Olá", "Tudo bem?"), responda de forma cordial e interativa, mostrando disposição em ajudar.
- Se for uma pergunta técnica sobre leis trabalhistas, você DEVE usar a ferramenta `tool_buscar_na_clt`.
- Mantenha o tom de um especialista experiente e calmo.
"""

SYSTEM_PROMPT_DRAFTER = """
Você é um assistente jurídico júnior.
Sua tarefa é gerar um rascunho de resposta para a "PERGUNTA DO USUÁRIO" usando APENAS o "CONTEXTO" fornecido.
Seja direto e cite as fontes (Artigo e Origem) que você usou.
Se o contexto for irrelevante, apenas diga "CONTEXTO IRRELEVANTE".
"""

SYSTEM_PROMPT_CRITIC = """
Você é um Advogado Sênior Revisor. Sua tarefa é revisar o "RASCUNHO DE RESPOSTA" gerado por seu assistente júnior.

Você deve seguir 3 REGRAS CRÍTICAS de revisão:
1.  **CHEQUE A PRECISÃO FÁTICA:** O rascunho é 100% suportado pelo "CONTEXTO"? Se mencionar fatos ausentes no contexto, remova-os (Alucinação).
2.  **CHEQUE AS CITAÇÕES:** As citações (ex: "CLT, Art. X") correspondem exatamente à "Origem" do contexto?
3.  **TONALIDADE:** Garanta que a resposta seja fluida, profissional e direta ao ponto. **NÃO inclua logs, não diga "Loop 2" ou "Rascunho". Apenas a resposta final.**

**Sua Ação:**
-   **Se o Rascunho for bom:** Aprová-lo e responder ao usuário.
-   **Se tiver erros:** CORRIJA o rascunho e gere a resposta precisa.
-   **Se o "CONTEXTO" for irrelevante:** Responda APENAS: "Não encontrei informações sobre este tópico específico na CLT ou nas normas fornecidas. Poderia reformular sua dúvida?"
"""

class LaborLawAgent:
    def __init__(self):
        api_key = config.get_api_key("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://chat.maritaca.ai/api")
        self.retriever = Retriever(llm_client=self.client)
        
        # Define Tools
        self.tools = [{
            "type": "function",
            "function": {
                "name": "tool_buscar_na_clt",
                "description": "Busca artigos na CLT e normas correlatas para responder a uma pergunta técnica.",
                "parameters": {
                    "type": "object",
                    "properties": {"pergunta": {"type": "string", "description": "A pergunta original do usuário."}},
                    "required": ["pergunta"],
                },
            },
        }]
        logger.info("LaborLawAgent initialized.")

    def _format_context(self, chunks: List[Dict]) -> str:
        if not chunks:
            return "Nenhum contexto encontrado."
        
        context_str = ""
        for i, chunk in enumerate(chunks):
            context_str += f"[FONTE {i+1}]\nOrigem: {chunk.get('origem', chunk.get('documento_fonte', 'N/A'))}\n"
            if chunk.get('titulo_clt'): 
                context_str += f"Título: {chunk.get('titulo_clt')}\n"
            if chunk.get('capitulo_secao'):
                context_str += f"Capítulo/Seção: {chunk.get('capitulo_secao')}\n"
            
            artigo = chunk.get('artigo_numero', 'N/A')
            conteudo = chunk.get('conteudo_chunk', '')
            context_str += f"Artigo: {artigo}\nTexto:\n```\n{conteudo}\n```\n\n"
            
        return context_str

    def run(self, query: str, chat_history: List[Dict]) -> Tuple[str, str]:
        """
        Executes the agent loop with Memory Awareness:
        1. Decision (Use Tool vs Chat) taking history into account
        2. (If Tool) Search
        3. Draft Response
        4. Critic/Revision
        """
        log_trace = ""
        messages = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]
        
        # Add history to context (Memory)
        # Memory: Keep the last 20 messages (approx. 10 interactions) to maintain context without overflow
        # The user requested "at least 10 messages", so 20 covers 10 interactions fully.
        recent_history = chat_history[-20:] if len(chat_history) > 20 else chat_history
        messages.extend(recent_history)
        
        messages.append({"role": "user", "content": query})

        try:
            # --- Step 1: Decision ---
            logger.info("Step 1: Decision (Tool Use)")
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.0
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                logger.info("Decision: Direct Response")
                return msg.content, None # No log trace for chat

            # --- Step 2: Execution (Retrieval) ---
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            
            # If the model inferred a query from history (e.g. "E sobre férias?" -> "férias"), use it.
            # If 'pergunta' is generic, we might want to combine with history, but let's trust the model's extraction first.
            user_query = args.get("pergunta", query)
            
            logger.info(f"Decision: Retrieval for '{user_query}'")
            chunks, search_log = self.retriever.hybrid_search(user_query)
            log_trace += search_log
            
            context_str = self._format_context(chunks)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": context_str})

            # --- Step 3: Drafter ---
            logger.info("Step 3: Drafting Response")
            draft_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT_DRAFTER},
                {"role": "user", "content": f"PERGUNTA DO USUÁRIO: {query}\n(Considere o histórico da conversa se necessário)\n\nCONTEXTO RECUPERADO:\n{context_str}"}
            ]
            draft_resp = self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=draft_msgs,
                temperature=0.0
            )
            draft_text = draft_resp.choices[0].message.content

            # --- Step 4: Critic ---
            logger.info("Step 4: Critic Review")
            critic_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT_CRITIC},
                {"role": "user", "content": f"PERGUNTA: {query}\n\nCONTEXTO:\n{context_str}\n\nRASCUNHO:\n{draft_text}"}
            ]
            final_resp = self.client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=critic_msgs,
                temperature=0.0,
                max_tokens=2048
            )
            
            final_answer = final_resp.choices[0].message.content
            return final_answer, log_trace

        except Exception as e:
            logger.error(f"Agent Error: {e}")
            return f"❌ Desculpe, ocorreu um erro interno: {str(e)}", log_trace

