# HealthSearch — Motor de Busca Híbrido BM25 + Semântico

Projeto acadêmico da disciplina **Tendências em Ciência da Computação — Recuperação de Informação / Processamento de Linguagem Natural**, do UNIPÊ.

O objetivo é construir um protótipo em **Streamlit** capaz de combinar busca léxica com **Okapi BM25** e busca semântica baseada em **embeddings**, utilizando **Reciprocal Rank Fusion (RRF)** para gerar um ranking híbrido de documentos médicos.

## Contexto do problema

A aplicação simula um cenário de uma HealthTech com um repositório de protocolos clínicos e guias médicos.

O problema central é que:

- buscas puramente léxicas podem falhar quando a consulta usa sinônimos ou linguagem diferente da diretriz cadastrada;
- buscas puramente semânticas podem perder precisão em códigos, termos e valores muito específicos;
- uma abordagem híbrida pode combinar a precisão do BM25 com a cobertura semântica dos embeddings.

Exemplos esperados:

- uma busca por `infarto` deve conseguir recuperar documentos relacionados a `síndrome coronariana` ou `infarto agudo do miocárdio`;
- uma busca por `CÓD-ECG-12D` deve preservar alta precisão na recuperação dos documentos que possuem exatamente esse código.

## Objetivo principal

Implementar o **HealthSearch**, um motor de busca híbrido que permita comparar visualmente três estratégias:

1. Busca léxica com BM25;
2. Busca semântica vetorial;
3. Busca híbrida utilizando Reciprocal Rank Fusion.

A aplicação deverá apresentar os resultados em uma interface interativa e permitir observar em quais situações cada estratégia funciona melhor.

## Tecnologias previstas

- Python
- Streamlit
- Pandas
- `rank-bm25`
- `sentence-transformers`
- Similaridade de Cosseno
- Reciprocal Rank Fusion (RRF)

Também é permitida uma simulação vetorial documentada caso não seja possível executar diretamente um modelo de embeddings.

## Estrutura obrigatória da aplicação

O projeto deverá ser entregue principalmente através de um arquivo:

```text
healthsearch_app.py
```

A aplicação deve implementar quatro fases principais.

### 1. Ingestão e pré-processamento

O corpus médico será carregado diretamente no código.

O pré-processamento deverá incluir:

- conversão para minúsculas;
- remoção de caracteres especiais;
- tokenização;
- remoção de stopwords em português.

### 2. Busca léxica — BM25

Utilizar o algoritmo **Okapi BM25** para calcular a relevância léxica entre a consulta e os documentos.

A interface deverá possuir dois parâmetros ajustáveis na barra lateral:

| Parâmetro | Intervalo | Padrão | Função |
|---|---:|---:|---|
| `k1` | 0.0 – 3.0 | 1.2 | Controla a saturação da frequência dos termos |
| `b` | 0.0 – 1.0 | 0.75 | Controla a normalização pelo tamanho do documento |

Os resultados deverão ser recalculados sempre que esses parâmetros forem alterados.

### 3. Busca semântica vetorial

Cada documento e a consulta deverão ser representados por embeddings densos.

A relevância será calculada através da **similaridade de cosseno**.

Essa etapa será responsável principalmente por capturar relações semânticas, sinônimos e consultas formuladas com vocabulário diferente daquele utilizado nos documentos.

### 4. Busca híbrida — Reciprocal Rank Fusion

Os rankings BM25 e semântico serão combinados através do algoritmo **Reciprocal Rank Fusion (RRF)**.

A fórmula definida para o projeto é:

```text
Score_RRF(D) = α * [1 / (k_rrf + Rank_BM25)]
             + (1 - α) * [1 / (k_rrf + Rank_Semantico)]
```

Onde:

- `α` representa o peso de balanceamento entre busca léxica e semântica;
- `k_rrf = 60` é a constante de suavização;
- `Rank_BM25` representa a posição do documento no ranking BM25;
- `Rank_Semantico` representa sua posição no ranking semântico.

A interface deverá permitir alterar o valor de `α` para visualizar o impacto no ranking final.

## Corpus médico obrigatório

O projeto deverá utilizar os seis documentos abaixo diretamente no código.

| ID | Título | Conteúdo |
|---|---|---|
| Doc 1 | Protocolo Emergência ECG | Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos. |
| Doc 2 | Guia de Farmacologia Cardíaca | O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio. |
| Doc 3 | Diretriz de Hipertensão Arterial | A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI. |
| Doc 4 | Manual de AVC Isquêmico | O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas. |
| Doc 5 | Protocolo de Reanimação RCR | Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul. |
| Doc 6 | Procedimentos de UTI Geral | Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria. |

## Interface Streamlit

A aplicação deverá possuir uma interface organizada em abas, permitindo comparar diretamente os rankings.

Estrutura sugerida:

```text
HealthSearch
│
├── Sidebar
│   ├── Consulta
│   ├── k1
│   ├── b
│   ├── α do RRF
│   └── Cross-Encoder Re-Ranking
│
└── Abas
    ├── Busca Léxica — BM25
    ├── Busca Semântica
    ├── Busca Híbrida — RRF
    └── Matriz Comparativa
```

A matriz comparativa deverá facilitar a visualização da posição de cada documento nos diferentes rankings.

## Critérios de avaliação

| Critério | Peso |
|---|---:|
| Implementação BM25 e parâmetros interativos | 25% |
| Busca semântica vetorial | 20% |
| Algoritmo híbrido RRF | 25% |
| Interface e abas em Streamlit | 20% |
| Qualidade do código e relatório | 10% |

Além do funcionamento correto, será importante manter o código organizado, modular e de fácil compreensão.

## Ponto extra — Cross-Encoder Re-Ranking

Além da implementação obrigatória, será desenvolvido o desafio bônus de **Cross-Encoder Re-Ranking**, valendo até **+0,3 ponto**.

A proposta é adicionar um checkbox na interface:

```text
☑ Aplicar Cross-Encoder Re-Ranking
```

Quando habilitado:

1. executamos normalmente BM25 + busca semântica + RRF;
2. selecionamos os **Top-3 documentos do ranking híbrido**;
3. enviamos os pares `consulta + documento` para um Cross-Encoder;
4. recalculamos a relevância dos três candidatos;
5. apresentamos o ranking antes e depois do re-ranking.

Modelo sugerido pelo enunciado:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

A interface deverá deixar clara a diferença entre:

```text
Ranking Híbrido RRF
        ↓
     Top-3
        ↓
Cross-Encoder Re-Ranking
        ↓
Ranking Final Refinado
```

Esse recurso deverá ser opcional para que seja possível comparar os resultados com e sem a camada adicional.

## Entregáveis

O trabalho deverá conter:

### Código

```text
healthsearch_app.py
```

Arquivo único contendo a aplicação Streamlit completa.

### Relatório técnico

Um relatório em PDF de no máximo **2 páginas**, contendo:

- arquitetura da solução;
- explicação resumida dos motores BM25 e semântico;
- funcionamento do RRF;
- gráfico ou comparação entre os rankings;
- análise do Cross-Encoder, caso implementado;
- divisão das tarefas da equipe.

## Organização sugerida do código

Mesmo sendo solicitado um único arquivo Python, podemos manter uma estrutura lógica interna por funções:

```python
load_corpus()
preprocess_text()
create_bm25_index()
search_bm25()
generate_embeddings()
semantic_search()
reciprocal_rank_fusion()
cross_encoder_rerank()
render_results()
main()
```

Isso mantém o arquivo simples para entrega sem comprometer a organização do projeto.

## Fluxo esperado

```text
Consulta do usuário
        │
        ├───────────────┐
        │               │
        ▼               ▼
      BM25          Embeddings
        │               │
        ▼               ▼
 Ranking Léxico   Ranking Semântico
        │               │
        └───────┬───────┘
                ▼
              RRF
                │
                ▼
        Ranking Híbrido
                │
          [Opcional]
                ▼
          Cross-Encoder
                │
                ▼
          Ranking Final
```

## Execução esperada

Após instalar as dependências:

```bash
pip install streamlit pandas rank-bm25 sentence-transformers
```

Executar:

```bash
streamlit run healthsearch_app.py
```

## Estratégia de desenvolvimento

Para evitar implementar tudo de uma vez, o desenvolvimento pode seguir esta sequência:

- [ ] Montar corpus médico hardcoded;
- [ ] implementar pré-processamento;
- [ ] implementar e validar BM25;
- [ ] adicionar sliders `k1` e `b`;
- [ ] implementar embeddings e similaridade de cosseno;
- [ ] criar rankings individuais;
- [ ] implementar RRF;
- [ ] adicionar controle de `α`;
- [ ] construir abas e matriz comparativa;
- [ ] implementar Cross-Encoder nos Top-3;
- [ ] adicionar comparação antes/depois do re-ranking;
- [ ] testar consultas léxicas, semânticas e códigos médicos;
- [ ] gerar gráfico comparativo;
- [ ] preparar relatório técnico em PDF.

## Resultado esperado

Ao final, o projeto deverá demonstrar de forma prática que diferentes estratégias de recuperação atendem a necessidades distintas e que a combinação de **BM25 + embeddings + RRF**, complementada opcionalmente por **Cross-Encoder Re-Ranking**, produz uma solução de recuperação de informação mais robusta para o cenário proposto.

---

**Instituição:** UNIPÊ — Centro Universitário de João Pessoa  
**Disciplina:** Tendências em Ciência da Computação — Recuperação de Informação / Processamento de Linguagem Natural  
**Projeto:** Desafio Integrador HealthSearch
