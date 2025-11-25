# Agente Trabalhista CLT - Guia de Deploy

Este projeto utiliza **Streamlit**, **Torch** e **ChromaDB**, o que o torna uma aplicação "stateful" e pesada (>1GB).
Devido a essas características, **o deploy na Vercel (Serverless) NÃO é recomendado**, pois excederá os limites de tamanho (250MB) e tempo de execução.

A arquitetura correta para este MVP é o uso de **Containers (Docker)**.

## 🚀 Opção 1: Deploy no Render (Recomendado)

O Render é a alternativa mais simples e robusta para rodar este container.

1.  Faça o Push deste código para o GitHub.
2.  Crie uma conta no [Render.com](https://render.com).
3.  Clique em **"New +"** -> **"Web Service"**.
4.  Conecte seu repositório do GitHub.
5.  Selecione o plano **"Free"** (ou Starter se precisar de mais RAM).
6.  Em **Environment Variables**, adicione:
    *   `OPENAI_API_KEY`: Sua chave da Maritaca/OpenAI.
7.  O Render detectará o `Dockerfile` automaticamente e iniciará o build.

## 🚀 Opção 2: Streamlit Community Cloud

Se quiser uma opção gratuita e específica para Streamlit:

1.  Suba o código no GitHub.
2.  Acesse [share.streamlit.io](https://share.streamlit.io).
3.  Conecte o repositório.
4.  Em "Main file path", coloque: `src/frontend/app.py`.
5.  Em "Advanced Settings" -> "Secrets", adicione sua API Key:
    ```toml
    OPENAI_API_KEY = "sua-chave-aqui"
    ```

## ⚠️ Nota sobre a Vercel

Se insistir em usar Vercel, você precisará reescrever o backend para usar apenas APIs externas (sem Torch/Chroma local) e usar o Streamlit apenas como frontend estático, o que descaracterizaria a arquitetura atual de RAG Local.

