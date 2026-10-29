# Bruno Lopes de Souza

### AI Engineering • Machine Learning • Software Engineering

Tecnólogo em Desenvolvimento de Software Multiplataforma pela FATEC Registro, formado em 2026. Meu foco é construir software que integra IA de forma útil e mensurável, explorando pipelines de dados, redes neurais, inferência local, APIs e avaliação de modelos.

[LinkedIn](https://www.linkedin.com/in/bruno-souza-506523323/) · [E-mail](mailto:brunolopes.bl54@gmail.com)

## Projetos em destaque

### 1. CalmWave — filtragem de áudio com IA
Projeto acadêmico em equipe de tecnologia assistiva para apoiar pessoas com Transtorno do Processamento Auditivo Central (TPAC).

- **IA/ML:** rede neural U-Net aplicada ao aprimoramento de fala, treinada em PyTorch com o dataset VoiceBank-DEMAND.
- **Inferência:** modelo exportado para ONNX e integrado a um aplicativo Android com ONNX Runtime para processamento local, sem enviar o áudio de voz para a nuvem.
- **Pipeline:** áudio PCM → STFT/espectrograma → inferência da máscara espectral → reconstrução do áudio.
- **Resultados reportados no artigo:** latência ponta a ponta estimada em aproximadamente 200 ms; nos testes em diferentes smartphones, STOI médio de 0,89 e SDR médio de 11,94 dB.
- **Minha atuação:** desenvolvimento do aplicativo mobile e dos fluxos de interação/áudio no contexto do projeto em equipe.

[Leia o estudo de caso técnico](./projects/calmwave-case-study.md)

### 2. CalmWave Audio Benchmark API
API em FastAPI para comparar áudio limpo e áudio processado, retornando métricas de qualidade PESQ, STOI e SDR. Inclui execução com Docker.

**Tecnologias:** Python, FastAPI, processamento de áudio, Docker.

[Repositório](https://github.com/bruno-souza013/metricas-audio)

### 3. Pipeline de preparação de avaliações em português
Pipeline de preparação de texto com limpeza, remoção de stopwords em português, tokenização e separação estratificada entre treino e teste.

**Tecnologias:** Python, pandas, NLTK, scikit-learn.

[Repositório](https://github.com/bruno-souza013/classificador_avaliacoes_PLN)

### 4. Experimento de classificação com KNN
Experimento educacional de classificação baseado em atributos numéricos, com comparação de diferentes valores de K e relatórios de métricas e matrizes de confusão. Não é uma ferramenta de diagnóstico.

**Tecnologias:** Python, KNN, métricas de classificação.

[Repositório](https://github.com/bruno-souza013/knn_gripe)

### 5. DevDocs RAG — assistente técnico com fontes

MVP de Retrieval-Augmented Generation com chunking, embeddings locais, busca semântica, FastAPI e geração via Ollama. Inclui documentos de demonstração, testes unitários e um conjunto curado para avaliar Recall@3; os resultados ainda devem ser executados e registrados.

**Tecnologias:** Python, FastAPI, Sentence Transformers, NumPy, Ollama, Pytest.

[Ver código e instruções](https://github.com/bruno-souza013/bruno-souza013/tree/main/projects/devdocs-rag)

### 6. Experimento NLP — Disaster Tweets

Notebook planejado para comparar baseline e classificação TF-IDF + Logistic Regression com divisão estratificada, F1, análise de erros e geração de submissão da competição.

**Status:** notebook preparado no GitHub; precisa ser executado e publicado no Kaggle para registrar métricas reais.

[Ver notebook e instruções](https://github.com/bruno-souza013/bruno-souza013/tree/main/projects/kaggle-nlp-disaster-tweets)

## Tecnologias que utilizo em projetos

- **IA/ML:** Python, PyTorch, scikit-learn, NLTK, avaliação de modelos
- **IA em produção e integração:** ONNX Runtime, FastAPI, APIs REST, Docker
- **Desenvolvimento:** Kotlin, Java, JavaScript/TypeScript, Node.js
- **Dados:** pandas, SQL, MySQL, PostgreSQL, MongoDB
- **Ferramentas:** Git e GitHub

## Foco atual

Aprofundamento em engenharia de Machine Learning, avaliação experimental, inferência eficiente e integração de modelos de IA a aplicações de software. Estou também desenvolvendo projetos com LLMs/RAG e organizando a publicação de artefatos autorizados no Hugging Face.

> Construir, medir, documentar e melhorar — da experimentação à aplicação.
