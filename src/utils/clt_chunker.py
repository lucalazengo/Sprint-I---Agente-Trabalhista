"""
PROCESSADOR DE CHUNKING DA CLT PARA RAG
========================================

Este script processa o documento legal da CLT e divide-o em chunks lógicos
baseados em artigos completos, preservando o contexto jurídico e estrutural.

ESTRATÉGIA DE CHUNKING:
- Unidade primária: Artigo Completo (caput + parágrafos + incisos + alíneas)
- Preservação do contexto hierárquico completo
- Metadados estruturados para recuperação eficiente

Autor: Sistema de Processamento Legal
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import PyPDF2

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clt_chunking_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CLTChunker:
    """
    Classe responsável pelo chunking estruturado da CLT.
    
    Esta classe implementa a estratégia de segmentação baseada em artigos
    completos, preservando a hierarquia jurídica e extraindo metadados
    estruturados para uso em sistemas RAG.
    """
    
    def __init__(self):
        """Inicializa o chunker com padrões regex para identificação estrutural."""
        # Padrões de identificação
        self.patterns = {
            # Documento fonte
            'documento_fonte': [
                r'(CLT|Consolidação das Leis do Trabalho)',
                r'Decreto-Lei\s+n[°º]?\s*(\d+\.?\d*/\d{4})',
            ],
            # Títulos da CLT
            'titulo': re.compile(
                r'^(?:T[ÍI]TULO|TITULO)\s+([IVXLCDM]+|[A-Z])\s*[-–—]?\s*(.+?)(?:\n|$)',
                re.MULTILINE | re.IGNORECASE
            ),
            # Capítulos
            'capitulo': re.compile(
                r'^CAP[ÍI]TULO\s+([IVXLCDM]+|[A-Z]|\d+)\s*[-–—]?\s*(.+?)(?:\n|$)',
                re.MULTILINE | re.IGNORECASE
            ),
            # Seções
            'secao': re.compile(
                r'^SE[CÇ][ÃA]O\s+([IVXLCDM]+|[A-Z]|\d+)\s*[-–—]?\s*(.+?)(?:\n|$)',
                re.MULTILINE | re.IGNORECASE
            ),
            # Artigos (principal) - padrão mais flexível
            'artigo': re.compile(
                r'^Art\.\s*(\d+[º°]?[-A-Z]?)\s*[\.:]?\s*',
                re.MULTILINE | re.IGNORECASE
            ),
            # Parágrafos
            'paragrafo': re.compile(
                r'^§\s*(\d+[º°o]?)\s*(.+?)(?=(?:^§|^[IVXLCDM]+\.|^[a-z]\)|^Art\.|$))',
                re.MULTILINE | re.DOTALL
            ),
            # Incisos
            'inciso': re.compile(
                r'^([IVXLCDM]+)\.\s*(.+?)(?=(?:^[IVXLCDM]+\.|^[a-z]\)|^§|^Art\.|$))',
                re.MULTILINE | re.DOTALL
            ),
            # Alíneas
            'alinea': re.compile(
                r'^([a-z])\)\s*(.+?)(?=(?:^[a-z]\)|^[IVXLCDM]+\.|^§|^Art\.|$))',
                re.MULTILINE | re.DOTALL
            ),
            # Status legal
            'status_revogado': re.compile(
                r'(?:revogado|revogada|revogados|revogadas|revogação)',
                re.IGNORECASE
            ),
            'status_vetado': re.compile(
                r'(?:vetado|vetada|vetados|vetadas|veto)',
                re.IGNORECASE
            ),
            # Constituição
            'constituicao': re.compile(
                r'(?:Constituição|Constituição da República Federativa do Brasil)',
                re.IGNORECASE
            ),
            # Normas correlatas
            'norma_correlata': re.compile(
                r'^Lei\s+n[°º]?\s*(\d+\.?\d*/\d{4})',
                re.MULTILINE | re.IGNORECASE
            ),
            # Anexos
            'anexo': re.compile(
                r'^ANEXO\s*(?:[-–—]?\s*(.+?))?(?:\n|$)',
                re.MULTILINE | re.IGNORECASE
            ),
        }
        
        # Estado atual do parsing
        self.current_titulo = None
        self.current_capitulo = None
        self.current_secao = None
        self.current_documento = "CLT (Decreto-Lei nº 5.452/1943)"
        self.current_tipo_estrutura = "CLT"
        
        logger.info("CLT Chunker inicializado com padrões de identificação")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrai texto completo de um arquivo PDF.
        
        Args:
            pdf_path (str): Caminho para o arquivo PDF
            
        Returns:
            str: Texto extraído do PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
        
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF protegido por senha: {pdf_path}")
                    return ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.error(f"Erro ao extrair página {page_num + 1}: {e}")
                        continue
            
            # Normalização básica
            text = text.strip()
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            # Preserva quebras de linha importantes
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            logger.info(f"Texto extraído: {len(text)} caracteres de {pdf_path}")
            return text
            
        except Exception as e:
            logger.error(f"Erro ao processar PDF {pdf_path}: {e}")
            raise
    
    def identify_structure_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """
        Identifica a hierarquia estrutural do documento.
        
        Args:
            text (str): Texto completo do documento
            
        Returns:
            List[Dict]: Lista de elementos estruturais identificados
        """
        structure = []
        lines = text.split('\n')
        
        current_titulo = None
        current_capitulo = None
        current_secao = None
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Título
            titulo_match = self.patterns['titulo'].match(line_clean)
            if titulo_match:
                current_titulo = {
                    'numero': titulo_match.group(1),
                    'nome': titulo_match.group(2).strip(),
                    'linha': i,
                    'tipo': 'titulo'
                }
                current_capitulo = None
                current_secao = None
                structure.append(current_titulo)
                continue
            
            # Capítulo
            capitulo_match = self.patterns['capitulo'].match(line_clean)
            if capitulo_match:
                current_capitulo = {
                    'numero': capitulo_match.group(1),
                    'nome': capitulo_match.group(2).strip(),
                    'linha': i,
                    'tipo': 'capitulo',
                    'titulo': current_titulo
                }
                current_secao = None
                structure.append(current_capitulo)
                continue
            
            # Seção
            secao_match = self.patterns['secao'].match(line_clean)
            if secao_match:
                current_secao = {
                    'numero': secao_match.group(1),
                    'nome': secao_match.group(2).strip(),
                    'linha': i,
                    'tipo': 'secao',
                    'capitulo': current_capitulo,
                    'titulo': current_titulo
                }
                structure.append(current_secao)
                continue
        
        return structure
    
    def extract_article_content(self, article_text: str) -> Dict[str, Any]:
        """
        Extrai conteúdo completo de um artigo incluindo subelementos.
        Preserva toda a estrutura do artigo (caput, parágrafos, incisos, alíneas).
        
        Args:
            article_text: Texto completo do artigo (incluindo "Art. X")
            
        Returns:
            Dict com conteúdo completo do artigo
        """
        # Normaliza o texto primeiro
        article_text = article_text.strip()
        
        # Tenta múltiplos padrões para encontrar o artigo
        artigo_numero = None
        artigo_conteudo = None
        
        # Padrão 1: Regex do pattern
        artigo_match = self.patterns['artigo'].search(article_text)
        if artigo_match:
            try:
                artigo_numero = artigo_match.group(1).strip()
                # Se o regex capturou o conteúdo, usa-o; senão, extrai do texto
                if artigo_match.lastindex and artigo_match.lastindex > 1:
                    artigo_conteudo = artigo_match.group(2).strip() if len(artigo_match.groups()) > 1 else article_text[artigo_match.end():].strip()
                else:
                    artigo_conteudo = article_text[artigo_match.end():].strip()
            except (IndexError, AttributeError):
                pass
        
        # Padrão 2: Regex alternativo
        if not artigo_numero:
            artigo_match_alt = re.search(r'Art\.\s*(\d+[º°]?[-A-Z]?)\s*[\.:]?\s*', article_text, re.IGNORECASE)
            if artigo_match_alt:
                try:
                    artigo_numero = artigo_match_alt.group(1).strip()
                    artigo_conteudo = article_text[artigo_match_alt.end():].strip()
                except (IndexError, AttributeError):
                    pass
        
        # Padrão 3: Simples split
        if not artigo_numero:
            # Último recurso: tenta extrair número do artigo manualmente
            match_simple = re.search(r'Art\.\s*(\d+[º°]?[-A-Za-z]?)', article_text[:50], re.IGNORECASE)
            if match_simple:
                artigo_numero = match_simple.group(1).strip()
                # Remove "Art. X. " ou "Art. X:" do início
                artigo_conteudo = re.sub(r'^Art\.\s*\d+[º°]?[-A-Za-z]?\s*[\.:]?\s*', '', article_text, count=1, flags=re.IGNORECASE).strip()
        
        if not artigo_numero:
            raise ValueError(f"Não foi possível identificar o artigo no texto: {article_text[:100]}")
        
        if not artigo_conteudo:
            artigo_conteudo = article_text
        
        # Normaliza espaços em branco (preserva quebras de linha estruturais)
        artigo_conteudo = re.sub(r' +', ' ', artigo_conteudo)  # Remove espaços múltiplos dentro de linhas
        artigo_conteudo = re.sub(r'\n\s*\n+', '\n\n', artigo_conteudo)  # Normaliza linhas vazias múltiplas
        
        # Constrói o conteúdo completo do artigo preservando a formatação original
        conteudo_completo = f"Art. {artigo_numero}. {artigo_conteudo}"
        
        # Identifica se tem parágrafos para referência
        has_paragrafos = bool(re.search(r'§\s*\d+', artigo_conteudo))
        
        return {
            'numero': artigo_numero,
            'caput': artigo_conteudo.split('§')[0].strip() if has_paragrafos else artigo_conteudo,
            'paragrafos': has_paragrafos,
            'conteudo_completo': conteudo_completo
        }
    
    def determine_status_legal(self, conteudo: str) -> str:
        """
        Determina o status legal de um dispositivo.
        
        Args:
            conteudo (str): Conteúdo do artigo/dispositivo
            
        Returns:
            str: Status legal ("Em vigor", "Revogado", "Vetado")
        """
        if self.patterns['status_revogado'].search(conteudo):
            return "Revogado"
        elif self.patterns['status_vetado'].search(conteudo):
            return "Vetado"
        else:
            return "Em vigor"
    
    def get_hierarchical_context(self, article_start_pos: int, structure: List[Dict]) -> Dict[str, Optional[str]]:
        """
        Obtém o contexto hierárquico para um artigo baseado na estrutura.
        
        Args:
            article_start_pos: Posição aproximada do artigo no texto
            structure: Lista de elementos estruturais
            
        Returns:
            Dict com contexto hierárquico
        """
        context = {
            'titulo_clt': None,
            'capitulo_secao': None
        }
        
        # Encontra o título mais recente antes deste artigo
        for struct in structure:
            if struct['tipo'] == 'titulo':
                context['titulo_clt'] = f"Título {struct['numero']} – {struct['nome']}"
            elif struct['tipo'] == 'capitulo':
                capitulo_nome = f"Capítulo {struct['numero']} – {struct['nome']}"
                if struct.get('secao'):
                    secao_nome = f", Seção {struct['secao']['numero']} – {struct['secao']['nome']}"
                    context['capitulo_secao'] = capitulo_nome + secao_nome
                else:
                    context['capitulo_secao'] = capitulo_nome
            elif struct['tipo'] == 'secao':
                if context['capitulo_secao']:
                    context['capitulo_secao'] += f", Seção {struct['numero']} – {struct['nome']}"
                else:
                    context['capitulo_secao'] = f"Seção {struct['numero']} – {struct['nome']}"
        
        return context
    
    def process_clt_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Processa o documento completo da CLT e gera chunks.
        
        Args:
            pdf_path (str): Caminho para o arquivo PDF
            
        Returns:
            List[Dict]: Lista de chunks estruturados
        """
        logger.info(f"Iniciando processamento do documento: {pdf_path}")
        
        # Extrai texto
        text = self.extract_text_from_pdf(pdf_path)
        
        # Estado hierárquico atual
        current_titulo = None
        current_capitulo = None
        current_secao = None
        
        # Encontra todos os elementos estruturais e artigos
        lines = text.split('\n')
        chunks = []
        
        # Processa sequencialmente para manter contexto
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Verifica se é título
            titulo_match = self.patterns['titulo'].match(line)
            if titulo_match:
                current_titulo = {
                    'numero': titulo_match.group(1),
                    'nome': titulo_match.group(2).strip()
                }
                current_capitulo = None
                current_secao = None
                i += 1
                continue
            
            # Verifica se é capítulo
            capitulo_match = self.patterns['capitulo'].match(line)
            if capitulo_match:
                current_capitulo = {
                    'numero': capitulo_match.group(1),
                    'nome': capitulo_match.group(2).strip()
                }
                current_secao = None
                i += 1
                continue
            
            # Verifica se é seção
            secao_match = self.patterns['secao'].match(line)
            if secao_match:
                current_secao = {
                    'numero': secao_match.group(1),
                    'nome': secao_match.group(2).strip()
                }
                i += 1
                continue
            
            # Verifica se é artigo
            artigo_match = self.patterns['artigo'].match(line)
            if artigo_match or (line.startswith('Art.') or line.startswith('ART.')):
                # Extrai o artigo completo (pode estar em múltiplas linhas)
                artigo_text = line
                j = i + 1
                
                # Continua coletando linhas até encontrar próximo artigo, título, capítulo ou seção
                while j < len(lines):
                    next_line = lines[j]
                    next_line_clean = next_line.strip()
                    
                    # Verifica se é início de novo elemento estrutural
                    if (self.patterns['artigo'].match(next_line_clean) or 
                        self.patterns['titulo'].match(next_line_clean) or
                        self.patterns['capitulo'].match(next_line_clean) or
                        self.patterns['secao'].match(next_line_clean) or
                        next_line_clean.startswith('Art.') or
                        next_line_clean.startswith('ART.') or
                        next_line_clean.startswith('TÍTULO') or
                        next_line_clean.startswith('TITULO') or
                        next_line_clean.startswith('CAPÍTULO') or
                        next_line_clean.startswith('CAPITULO') or
                        next_line_clean.startswith('SEÇÃO') or
                        next_line_clean.startswith('SECAO') or
                        next_line_clean.startswith('ANEXO')):
                        break
                    
                    # Adiciona linha ao artigo (mesmo se vazia para preservar estrutura)
                    if j < len(lines):
                        artigo_text += "\n" + next_line
                    j += 1
                
                # Processa o artigo encontrado
                try:
                    if artigo_text.strip():
                        artigo_data = self.extract_article_content(artigo_text)
                        
                        # Monta contexto hierárquico
                        titulo_str = None
                        if current_titulo:
                            titulo_str = f"Título {current_titulo['numero']} – {current_titulo['nome']}"
                        
                        capitulo_secao_str = None
                        if current_capitulo:
                            capitulo_secao_str = f"Capítulo {current_capitulo['numero']} – {current_capitulo['nome']}"
                            if current_secao:
                                capitulo_secao_str += f", Seção {current_secao['numero']} – {current_secao['nome']}"
                        elif current_secao:
                            capitulo_secao_str = f"Seção {current_secao['numero']} – {current_secao['nome']}"
                        
                        # Determina status legal
                        status = self.determine_status_legal(artigo_data['conteudo_completo'])
                        
                        # Verifica se é constituição ou norma correlata
                        tipo_estrutura = self.current_tipo_estrutura
                        documento_fonte = self.current_documento
                        
                        if self.patterns['constituicao'].search(artigo_text[:200]):
                            tipo_estrutura = "Constituição da República Federativa do Brasil"
                            documento_fonte = "Constituição da República Federativa do Brasil"
                        elif self.patterns['norma_correlata'].search(artigo_text[:200]):
                            norma_match = self.patterns['norma_correlata'].search(artigo_text[:200])
                            if norma_match:
                                documento_fonte = f"Lei nº {norma_match.group(1)}"
                                tipo_estrutura = documento_fonte
                        
                        # Cria chunk
                        chunk = {
                            "documento_fonte": documento_fonte,
                            "tipo_estrutura": tipo_estrutura,
                            "titulo_clt": titulo_str,
                            "capitulo_secao": capitulo_secao_str,
                            "artigo_numero": f"Art. {artigo_data['numero']}",
                            "status_legal": status,
                            "conteudo_chunk": artigo_data['conteudo_completo']
                        }
                        
                        chunks.append(chunk)
                        
                        if len(chunks) % 50 == 0:
                            logger.info(f"Processados {len(chunks)} artigos")
                
                except Exception as e:
                    logger.error(f"Erro ao processar artigo na linha {i + 1}: {e}")
                
                i = j
                continue
            
            i += 1
        
        logger.info(f"Processamento concluído. Gerados {len(chunks)} chunks")
        return chunks
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], output_path: str):
        """
        Salva chunks em arquivo JSON.
        
        Args:
            chunks: Lista de chunks
            output_path: Caminho do arquivo de saída
        """
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Chunks salvos em: {output_path}")
        logger.info(f"Total de chunks: {len(chunks)}")


def main():
    """Função principal para execução standalone."""
    import sys
    
    # Caminhos
    pdf_path = r"C:\Users\mlzengo\Documents\Garden Solutions\Adicao_Contabilidade\Sprint I\data\raw_data\clt_e_normas_correlatas_1ed - 16-192.pdf"
    output_path = r"C:\Users\mlzengo\Documents\Garden Solutions\Adicao_Contabilidade\Sprint I\data\processed_data\clt_chunks.json"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    try:
        chunker = CLTChunker()
        chunks = chunker.process_clt_document(pdf_path)
        chunker.save_chunks_to_json(chunks, output_path)
        
        print(f"\n=== PROCESSAMENTO CONCLUÍDO ===")
        print(f"Total de chunks gerados: {len(chunks)}")
        print(f"Arquivo salvo em: {output_path}")
        
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

