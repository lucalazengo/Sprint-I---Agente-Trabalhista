# pdf_extrator.py
"""
EXTRATOR DE DADOS DE NOTAS TÉCNICAS DO TJGO
==========================================

Este script é responsável por extrair dados estruturados do documentos PDF.
de processamento de texto e expressões regulares para identificar e extrair
informações específicas dos documentos jurídicos.

FUNCIONALIDADES PRINCIPAIS:
- Extração de texto de arquivos PDF
- Identificação de campos estruturados via regex
- Processamento de metadados de documentos
- Geração de dataset estruturado em CSV
- Validação e limpeza de dados extraídos

CAMPOS EXTRAÍDOS:


PIPELINE DE EXTRAÇÃO:
1. Carrega arquivos PDF
2. Extrai texto de cada documento
3. Aplica padrões regex para campos específicos
4. Valida e normaliza dados extraídos
5. Compila dataset final em CSV

GESTÃO DE DADOS:


DEPENDÊNCIAS:


Autor: 
"""

import os
import re
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import PyPDF2
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NotasExtractor:
    """
    Classe responsável pela extração estruturada de dados de notas técnicas.
    
    Esta classe implementa métodos especializados para extrair informações
    específicas de documentos jurídicos do TJGO, utilizando padrões regex
    otimizados para o formato e estrutura destes documentos.
    
    CARACTERÍSTICAS:
    - Extração robusta com múltiplos padrões regex
    - Validação e normalização automática de dados
    - Tratamento de variações de formato
    - Logging detalhado para auditoria
    - Recuperação de erros e dados parciais
    """
    
    def __init__(self):
        """
        Inicializa o extrator com padrões regex pré-compilados.
        
        Os padrões são otimizados para documentos do TJGO e incluem
        variações comuns de formatação encontradas nos documentos.
        """
        # Padrões regex para extração de campos específicos
        self.patterns = {
            'id': [
                r'(?:ID|Identificador|Número)[\s:]*(\d{5,})',
                r'Nota\s+Técnica\s+n[°º]?\s*(\d+)',
                r'NT\s*[-:]?\s*(\d+)'
            ],
            'data_conclusao': [
                r'(?:Data|Concluído|Finalizado)[\s:]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})',
                r'(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
                r'Goiânia[,\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})'
            ],
            'idade': [
                r'(?:idade|anos?)[\s:]*(\d{1,3})\s*(?:anos?|a\.?)',
                r'(\d{1,3})\s*(?:anos?|a\.?)\s*(?:de\s*idade)?',
                r'paciente.*?(\d{1,3})\s*anos?'
            ],
            'sexo': [
                r'(?:sexo|gênero)[\s:]*([MFmf]|masculino|feminino|masc|fem)',
                r'\b([MF])\b.*?(?:anos?|idade)',
                r'(?:paciente|pessoa).*?(?:sexo\s*)?([MFmf]|masculino|feminino)'
            ],
            'cidade': [
                r'(?:cidade|município|natural\s+de|residente\s+em)[\s:]*([A-ZÁÊÇÕ][a-záêçõ\s]+?)(?:\s*[-,.]|\s*\n)',
                r'(?:de|em)\s+([A-ZÁÊÇÕ][a-záêçõ\s]+?)[,\s]*(?:GO|Goiás)',
                r'Município[\s:]*([A-ZÁÊÇÕ][a-záêçõ\s]+)'
            ],
            'cid': [
                r'CID[\s:-]*([A-Z]\d{2}(?:\.\d)?)',
                r'Código\s+Internacional[\s:]*([A-Z]\d{2}(?:\.\d)?)',
                r'\b([A-Z]\d{2}(?:\.\d)?)\b.*?(?:CID|diagnóstico)'
            ],
            'procedimento': [
                r'(?:procedimento|cirurgia|tratamento)[\s:]*([^.\n]+?)(?:\.|$)',
                r'(?:indicado|recomendado|solicitado)[\s:]*([^.\n]+?)(?:\.|$)',
                r'(?:realizar|executar)[\s:]*([^.\n]+?)(?:\.|$)'
            ]
        }
        
        # Compilar padrões para melhor performance
        self.compiled_patterns = {}
        for field, patterns in self.patterns.items():
            self.compiled_patterns[field] = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                           for pattern in patterns]
        
        logger.info("Extrator inicializado com padrões regex compilados")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai texto completo de um arquivo PDF.
    
    Esta função utiliza PyPDF2 para extrair texto de documentos PDF,
    implementando tratamento de erros e fallbacks para diferentes
    tipos de codificação e estruturas de PDF.
    
    PROCESSO DE EXTRAÇÃO:
    1. Abre arquivo PDF com PyPDF2
    2. Itera por todas as páginas
    3. Extrai texto de cada página
    4. Concatena texto completo
    5. Aplica limpeza básica
    
    TRATAMENTO DE ERROS:
    - PDFs corrompidos ou ilegíveis
    - Problemas de codificação
    - Arquivos protegidos por senha
    - Estruturas PDF não padrão
    
    Args:
        pdf_path (str): Caminho para o arquivo PDF
        
    Returns:
        str: Texto extraído do PDF ou string vazia em caso de erro
        
    Raises:
        FileNotFoundError: Se o arquivo PDF não existir
        
    Example:
        >>> texto = extract_text_from_pdf('nota_tecnica_123.pdf')
        >>> print(f"Extraídos {len(texto)} caracteres")
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")
    
    try:
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Verifica se o PDF está protegido
            if pdf_reader.is_encrypted:
                logger.warning(f"PDF protegido por senha: {pdf_path}")
                return ""
            
            # Extrai texto de todas as páginas
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logger.warning(f"Página {page_num + 1} sem texto extraível: {pdf_path}")
                except Exception as e:
                    logger.error(f"Erro ao extrair página {page_num + 1} de {pdf_path}: {e}")
                    continue
        
        # Limpeza básica do texto
        text = text.strip()
        
        # Remove caracteres de controle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normaliza espaços em branco
        text = re.sub(r'\s+', ' ', text)
        
        logger.info(f"Texto extraído: {len(text)} caracteres de {pdf_path}")
        return text
        
    except Exception as e:
        logger.error(f"Erro ao processar PDF {pdf_path}: {e}")
        return ""

def extract_field_with_patterns(text: str, patterns: List[re.Pattern], field_name: str) -> Optional[str]:
    """
    Extrai campo específico usando múltiplos padrões regex.
    
    Esta função tenta extrair um campo específico do texto usando
    uma lista de padrões regex ordenados por prioridade. Retorna
    o primeiro match encontrado após validação.
    
    ESTRATÉGIA DE EXTRAÇÃO:
    1. Testa padrões em ordem de prioridade
    2. Valida match encontrado
    3. Aplica normalização específica do campo
    4. Retorna valor limpo e validado
    
    Args:
        text (str): Texto onde buscar o campo
        patterns (List[re.Pattern]): Lista de padrões compilados
        field_name (str): Nome do campo para logging
        
    Returns:
        Optional[str]: Valor extraído ou None se não encontrado
        
    Example:
        >>> patterns = [re.compile(r'idade[\\s:]*(\\d+)', re.I)]
        >>> idade = extract_field_with_patterns(texto, patterns, 'idade')
    """
    for i, pattern in enumerate(patterns):
        try:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                
                # Validação básica
                if value and len(value) > 0:
                    logger.debug(f"Campo '{field_name}' extraído com padrão {i + 1}: {value}")
                    return value
                    
        except Exception as e:
            logger.warning(f"Erro ao aplicar padrão {i + 1} para '{field_name}': {e}")
            continue
    
    logger.debug(f"Campo '{field_name}' não encontrado no texto")
    return None

def normalize_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normaliza e valida dados extraídos.
    
    Esta função aplica normalizações específicas para cada tipo
    de campo extraído, garantindo consistência e qualidade dos dados.
    
    NORMALIZAÇÕES APLICADAS:
    - Datas: Conversão para formato padrão
    - Sexo: Padronização M/F
    - Idade: Validação de faixa etária
    - Texto: Limpeza e formatação
    - CID: Validação de formato
    
    Args:
        data (Dict[str, Any]): Dados brutos extraídos
        
    Returns:
        Dict[str, Any]: Dados normalizados e validados
    """
    normalized = data.copy()
    
    # Normalização de data
    if 'data_conclusao' in normalized and normalized['data_conclusao']:
        try:
            date_str = normalized['data_conclusao']
            
            # Padrões de data comuns
            date_patterns = [
                r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})',
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    if 'de' in date_str:  # Formato "dd de mês de aaaa"
                        day, month_name, year = match.groups()
                        month_map = {
                            'janeiro': '01', 'fevereiro': '02', 'março': '03',
                            'abril': '04', 'maio': '05', 'junho': '06',
                            'julho': '07', 'agosto': '08', 'setembro': '09',
                            'outubro': '10', 'novembro': '11', 'dezembro': '12'
                        }
                        month = month_map.get(month_name.lower(), '01')
                        normalized['data_conclusao'] = f"{day.zfill(2)}/{month}/{year}"
                    else:  # Formato dd/mm/aaaa
                        day, month, year = match.groups()
                        normalized['data_conclusao'] = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
                    break
        except Exception as e:
            logger.warning(f"Erro ao normalizar data: {e}")
    
    # Normalização de sexo
    if 'sexo' in normalized and normalized['sexo']:
        sexo = normalized['sexo'].lower()
        if sexo in ['m', 'masculino', 'masc']:
            normalized['sexo'] = 'M'
        elif sexo in ['f', 'feminino', 'fem']:
            normalized['sexo'] = 'F'
        else:
            normalized['sexo'] = None
    
    # Validação de idade
    if 'idade' in normalized and normalized['idade']:
        try:
            idade = int(normalized['idade'])
            if not (0 <= idade <= 120):  # Faixa etária válida
                logger.warning(f"Idade fora da faixa válida: {idade}")
                normalized['idade'] = None
            else:
                normalized['idade'] = idade
        except ValueError:
            logger.warning(f"Idade inválida: {normalized['idade']}")
            normalized['idade'] = None
    
    # Normalização de cidade
    if 'cidade' in normalized and normalized['cidade']:
        cidade = normalized['cidade'].strip()
        # Remove "GO" ou "Goiás" do final
        cidade = re.sub(r'\s*,?\s*(?:GO|Goiás)\s*$', '', cidade, flags=re.IGNORECASE)
        normalized['cidade'] = cidade.title() if cidade else None
    
    # Validação de CID
    if 'cid' in normalized and normalized['cid']:
        cid = normalized['cid'].upper()
        if not re.match(r'^[A-Z]\d{2}(?:\.\d)?$', cid):
            logger.warning(f"Formato de CID inválido: {cid}")
            normalized['cid'] = None
        else:
            normalized['cid'] = cid
    
    return normalized

def extract_data_from_pdfs(pdf_directory: str, output_csv: str = "extracted_data.csv") -> pd.DataFrame:
    """
    Extrai dados de todos os PDFs em um diretório.
    
    Esta função processa todos os arquivos PDF em um diretório,
    extraindo dados estruturados de cada documento e compilando
    os resultados em um DataFrame pandas.
    
    PROCESSO COMPLETO:
    1. Lista todos os arquivos PDF no diretório
    2. Para cada PDF:
       - Extrai texto completo
       - Aplica padrões de extração
       - Normaliza dados extraídos
       - Adiciona ao dataset
    3. Salva resultado final em CSV
    4. Gera relatório de estatísticas
    
    TRATAMENTO DE ERROS:
    - PDFs corrompidos são registrados mas não interrompem o processo
    - Dados parciais são preservados
    - Estatísticas detalhadas de sucesso/falha
    
    Args:
        pdf_directory (str): Diretório contendo os arquivos PDF
        output_csv (str): Caminho para salvar o CSV de saída
        
    Returns:
        pd.DataFrame: DataFrame com dados extraídos de todos os PDFs
        
    Raises:
        FileNotFoundError: Se o diretório não existir
        
    Example:
        >>> df = extract_data_from_pdfs('data/pdfs/', 'extracted_data.csv')
        >>> print(f"Extraídos dados de {len(df)} documentos")
    """
    if not os.path.exists(pdf_directory):
        raise FileNotFoundError(f"Diretório não encontrado: {pdf_directory}")
    
    logger.info(f"Iniciando extração de dados do diretório: {pdf_directory}")
    
    # Lista arquivos PDF
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"Nenhum arquivo PDF encontrado em: {pdf_directory}")
        return pd.DataFrame()
    
    logger.info(f"Encontrados {len(pdf_files)} arquivos PDF para processar")
    
    # Inicializa extrator
    extractor = NotasExtractor()
    
    # Lista para armazenar dados extraídos
    extracted_data = []
    
    # Estatísticas do processo
    stats = {
        'total_files': len(pdf_files),
        'successful': 0,
        'failed': 0,
        'partial': 0
    }
    
    # Processa cada arquivo PDF
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        logger.info(f"Processando ({i}/{len(pdf_files)}): {pdf_file}")
        
        try:
            # Extrai texto do PDF
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                logger.warning(f"Nenhum texto extraído de: {pdf_file}")
                stats['failed'] += 1
                continue
            
            # Inicializa dados do documento
            doc_data = {
                'filename': pdf_file,
                'filepath': pdf_path,
                'text_length': len(text),
                'extraction_date': datetime.now().isoformat()
            }
            
            # Extrai campos específicos
            fields_extracted = 0
            
            for field_name, patterns in extractor.compiled_patterns.items():
                value = extract_field_with_patterns(text, patterns, field_name)
                doc_data[field_name] = value
                if value:
                    fields_extracted += 1
            
            # Adiciona texto completo
            doc_data['full_text'] = text
            
            # Normaliza dados extraídos
            doc_data = normalize_extracted_data(doc_data)
            
            # Adiciona à lista
            extracted_data.append(doc_data)
            
            # Atualiza estatísticas
            if fields_extracted >= 3:  # Pelo menos 3 campos extraídos
                stats['successful'] += 1
            else:
                stats['partial'] += 1
            
            logger.info(f"Extraídos {fields_extracted} campos de: {pdf_file}")
            
        except Exception as e:
            logger.error(f"Erro ao processar {pdf_file}: {e}")
            stats['failed'] += 1
            continue
    
    # Cria DataFrame
    df = pd.DataFrame(extracted_data)
    
    # Salva em CSV se especificado
    if output_csv and not df.empty:
        try:
            df.to_csv(output_csv, index=False, encoding='utf-8')
            logger.info(f"Dados salvos em: {output_csv}")
        except Exception as e:
            logger.error(f"Erro ao salvar CSV: {e}")
    
    # Relatório final
    logger.info("=== RELATÓRIO DE EXTRAÇÃO ===")
    logger.info(f"Total de arquivos: {stats['total_files']}")
    logger.info(f"Extrações completas: {stats['successful']}")
    logger.info(f"Extrações parciais: {stats['partial']}")
    logger.info(f"Falhas: {stats['failed']}")
    logger.info(f"Taxa de sucesso: {((stats['successful'] + stats['partial']) / stats['total_files'] * 100):.1f}%")
    
    return df

# Função principal para execução standalone
if __name__ == "__main__":
    """
    Execução principal do script de extração.
    
    Este bloco é executado quando o script é chamado diretamente,
    permitindo extração automatizada com configurações padrão.
    """
    import sys
    
    # Configurações padrão
    pdf_directory = "data/pdfs"
    output_csv = "data/extracted_data.csv"
    
    # Permite especificar diretório via linha de comando
    if len(sys.argv) > 1:
        pdf_directory = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    try:
        # Executa extração
        df = extract_data_from_pdfs(pdf_directory, output_csv)
        
        if not df.empty:
            print(f"\n=== RESUMO DA EXTRAÇÃO ===")
            print(f"Documentos processados: {len(df)}")
            print(f"Campos extraídos por documento:")
            
            # Estatísticas por campo
            for col in df.columns:
                if col not in ['filename', 'filepath', 'full_text', 'extraction_date', 'text_length']:
                    non_null = df[col].notna().sum()
                    percentage = (non_null / len(df)) * 100
                    print(f"  {col}: {non_null}/{len(df)} ({percentage:.1f}%)")
            
            print(f"\nDados salvos em: {output_csv}")
        else:
            print("Nenhum dado foi extraído.")
            
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        sys.exit(1)
