# DevDocs RAG — assistente técnico com recuperação de fontes

Um MVP de **Retrieval-Augmented Generation (RAG)** que indexa arquivos Markdown/texto locais, recupera trechos relevantes com embeddings e consulta um LLM local via Ollama. As respostas devem citar IDs dos trechos usados.

> Status: MVP inicial. Métricas reais de recuperação e qualidade das respostas devem ser medidas após executar a avaliação; este README não apresenta resultados ainda não medidos.

## Objetivos de engenharia

- Separar ingestão, chunking, recuperação e geração.
- Manter documentos e inferência no ambiente local.
- Retornar fontes para inspeção das respostas.
- Ter testes automatizados e perguntas de avaliação.
- Tratar falta de contexto instruindo o modelo a não inventar respostas.

## Arquitetura

    Markdown/TXT -> chunking -> Sentence Transformers -> busca por similaridade
                                                          |
    Pergunta -> embedding --------------------------------+
                                                          v
                                       trechos recuperados + pergunta
                                                          |
                                                Ollama / LLM local
                                                          |
                                           resposta + referências

## Stack

- Python 3.11+
- FastAPI e Uvicorn
- Sentence Transformers: all-MiniLM-L6-v2
- NumPy para similaridade cosseno
- Ollama para geração local com LLM
- Pytest para testes

## Requisitos

1. Python 3.11 ou superior.
2. [Ollama](https://ollama.com/) instalado e em execução.
3. Baixe o modelo de geração escolhido com o comando: ollama pull qwen2.5:3b

Na primeira execução, Sentence Transformers baixará o modelo de embeddings. Depois disso, ele pode ser reutilizado localmente.

## Executar no Windows (PowerShell)

A partir da pasta projects/devdocs-rag:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    $env:OLLAMA_BASE_URL="http://localhost:11434"
    $env:OLLAMA_MODEL="qwen2.5:3b"
    uvicorn app.main:app --reload

Abra http://127.0.0.1:8000/docs para experimentar a API.

## Endpoints

- GET /health: estado da API e número de trechos carregados.
- GET /documents: lista os documentos indexados.
- POST /ask: consulta a base e retorna resposta com fontes.

Exemplo de corpo JSON para POST /ask:

    {
      "question": "Como o projeto testa a API?",
      "top_k": 3
    }

A resposta inclui sources com arquivo, ID do trecho, similaridade e excerto para auditoria.

## Avaliação

O arquivo data/evaluation/questions.jsonl contém perguntas e fontes esperadas. Execute o script para medir Recall@3 na recuperação. Esse indicador verifica se a fonte esperada aparece entre os três primeiros resultados; não mede, sozinho, a qualidade factual da resposta gerada.

    python -m scripts.evaluate_retrieval
    pytest -q

Registre no README os resultados que você realmente obtiver, incluindo número de perguntas, modelo de embedding, hardware e limitações observadas.

## Estrutura

    projects/devdocs-rag/
    ├── app/
    │   ├── main.py
    │   └── rag.py
    ├── data/
    │   ├── docs/
    │   └── evaluation/questions.jsonl
    ├── scripts/evaluate_retrieval.py
    ├── tests/test_rag.py
    ├── .env.example
    ├── requirements.txt
    └── README.md

## Limitações e próximos passos

- A qualidade depende da recuperação e do modelo local; avalie perguntas difíceis.
- O projeto é uma demonstração técnica, não uma ferramenta de produção.
- Próximas melhorias: reranking, avaliação das respostas, logging sem dados sensíveis, cache e integração contínua.

## Sobre este projeto

Projeto de portfólio independente para demonstrar componentes de engenharia de LLMs. Não representa um sistema oficial do iFood nem utiliza documentação privada de empresas.
