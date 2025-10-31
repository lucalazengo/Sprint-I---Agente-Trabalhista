# Estatísticas dos Chunks

- **Total de chunks**: 1008
- **Com título CLT**: 1008
- **Com capítulo/seção**: 797
- **Status Em vigor**: 891
- **Status Revogado**: 100
- **Status Vetado**: 17

## Exemplo 1: Primeiro chunk

```json
{
  "documento_fonte": "CLT (Decreto-Lei nº 5.452/1943)",
  "tipo_estrutura": "CLT",
  "titulo_clt": "Título I – Dos Princípios Fundamentais",
  "capitulo_secao": null,
  "artigo_numero": "Art. 1o",
  "status_legal": "Em vigor",
  "conteudo_chunk": "Art. 1o. A República Federativa do Brasil,\nformada pela união indissolúvel dos Estados e \nMunicípios e do Distrito Federal, constitui-se \nem Estado democrático de direito e tem como \nfundamentos:\nI – a soberania;\nII – a cidadania;\nIII – a dignidade da pessoa humana;\nIV – os valores sociais do trabalho e da livre \niniciativa;\nV – o pluralismo político \nParágrafo único Todo o poder emana do \npovo, que o exerce por meio de representan -\ntes eleitos ou diretamente, nos termos desta \nConstituição"
}
```

## Exemplo 2: Chunk com parágrafos

```json
{
  "documento_fonte": "CLT (Decreto-Lei nº 5.452/1943)",
  "tipo_estrutura": "CLT",
  "titulo_clt": "Título II – Dos Direitos e Garantias",
  "capitulo_secao": "Capítulo II – Dos Direitos Sociais",
  "artigo_numero": "Art. 9o",
  "status_legal": "Em vigor",
  "conteudo_chunk": "Art. 9o. É assegurado o direito de greve,\ncompetindo aos trabalhadores decidir sobre a \noportunidade de exercê-lo e sobre os interesses \nque devam por meio dele defender \n§ 1o A lei definirá os serviços ou atividades \nessenciais e disporá sobre o atendimento das \nnecessidades inadiáveis da comunidade \n§ 2o Os abusos cometidos sujeitam os res -\nponsáveis às penas da lei"
}
```

## Exemplo 3: Chunk com capítulo

```json
{
  "documento_fonte": "CLT (Decreto-Lei nº 5.452/1943)",
  "tipo_estrutura": "CLT",
  "titulo_clt": "Título II – Dos Direitos e Garantias",
  "capitulo_secao": "Capítulo II – Dos Direitos Sociais",
  "artigo_numero": "Art. 6o",
  "status_legal": "Em vigor",
  "conteudo_chunk": "Art. 6o. São direitos sociais a educação, a\nsaúde, a alimentação, o trabalho, a moradia, o \ntransporte, o lazer, a segurança, a previdência \nsocial, a proteção à maternidade e à infância, a \nassistência aos desamparados, na forma desta \nConstituição"
}
```