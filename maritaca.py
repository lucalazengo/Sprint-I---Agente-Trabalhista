import openai

client = openai.OpenAI(
    api_key="100464884795113056205_7ca4c23f2c798e6e",
    base_url="https://chat.maritaca.ai/api",
)

response = client.chat.completions.create(
  model="sabia-3",
  messages=[
    {"role": "user", "content": "O que sao Leis de Trabalho brasileiras?"},
  ],
  max_tokens=8000
)
answer = response.choices[0].message.content

print(f"Resposta: {answer}")   # Deve imprimir algo como "25 + 27 é igual a 52."