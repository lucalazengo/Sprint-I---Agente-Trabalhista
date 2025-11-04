"""Script para verificar a qualidade dos chunks gerados e salvar logs em .txt e .md"""
import json

def gerar_logs(data):
    linhas_txt = []
    linhas_md = []

    # Estatísticas
    stats = [
        ('Total de chunks', len(data)),
        ('Com título CLT', sum(1 for d in data if d.get("titulo_clt"))),
        ('Com capítulo/seção', sum(1 for d in data if d.get("capitulo_secao"))),
        ('Status Em vigor', sum(1 for d in data if d.get("status_legal") == "Em vigor")),
        ('Status Revogado', sum(1 for d in data if d.get("status_legal") == "Revogado")),
        ('Status Vetado', sum(1 for d in data if d.get("status_legal") == "Vetado")),
    ]

    # TXT
    linhas_txt.append('=== ESTATÍSTICAS DOS CHUNKS ===')
    for label, v in stats:
        linhas_txt.append(f'{label}: {v}')
    linhas_txt.append('\n=== EXEMPLO 1: Primeiro chunk ===')
    linhas_txt.append(json.dumps(data[0], indent=2, ensure_ascii=False))

    linhas_txt.append('\n=== EXEMPLO 2: Chunk com parágrafos ===')
    examples_with_para = [d for d in data if '§' in d.get('conteudo_chunk', '')]
    if examples_with_para:
        linhas_txt.append(json.dumps(examples_with_para[0], indent=2, ensure_ascii=False))

    linhas_txt.append('\n=== EXEMPLO 3: Chunk com capítulo ===')
    examples_with_chapter = [d for d in data if d.get('capitulo_secao')]
    if examples_with_chapter:
        linhas_txt.append(json.dumps(examples_with_chapter[0], indent=2, ensure_ascii=False))
    
    # MD
    linhas_md.append('# Estatísticas dos Chunks\n')
    for label, v in stats:
        linhas_md.append(f'- **{label}**: {v}')
    linhas_md.append('\n## Exemplo 1: Primeiro chunk\n')
    linhas_md.append('```json\n' + json.dumps(data[0], indent=2, ensure_ascii=False) + '\n```')

    linhas_md.append('\n## Exemplo 2: Chunk com parágrafos\n')
    if examples_with_para:
        linhas_md.append('```json\n' + json.dumps(examples_with_para[0], indent=2, ensure_ascii=False) + '\n```')

    linhas_md.append('\n## Exemplo 3: Chunk com capítulo\n')
    if examples_with_chapter:
        linhas_md.append('```json\n' + json.dumps(examples_with_chapter[0], indent=2, ensure_ascii=False) + '\n```')

    return '\n'.join(linhas_txt), '\n'.join(linhas_md)

def salvar_logs(txt_contents, md_contents, txt_path='data/processed_data/clt_chunks_log.txt', md_path='data/processed_data/clt_chunks_log.md'):
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(txt_contents)
    with open(md_path, 'w', encoding='utf-8') as md_file:
        md_file.write(md_contents)

if __name__ == '__main__':
    with open('data/processed_data/clt_chunks.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    txt_log, md_log = gerar_logs(data)

    # Print as antes:
    print(txt_log)

    # Salva logs
    salvar_logs(txt_log, md_log)
