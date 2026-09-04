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

## Execução

Instalar as dependências:

```bash
python -m pip install -r requirements.txt
```

Executar:

```bash
python -m streamlit run healthsearch_app.py
```

> Use `python -m streamlit` em vez de `streamlit` direto: quando o pip instala no modo
> `--user`, o executável `streamlit.exe` fica fora do `PATH` no Windows.

Na primeira execução os modelos são baixados do Hugging Face (~600 MB no total) e
ficam em cache local; a partir daí a aplicação roda offline.

## Estratégia de desenvolvimento

Para evitar implementar tudo de uma vez, o desenvolvimento pode seguir esta sequência:

- [x] Montar corpus médico hardcoded;
- [x] implementar pré-processamento;
- [x] implementar e validar BM25;
- [x] adicionar sliders `k1` e `b`;
- [x] implementar embeddings e similaridade de cosseno;
- [x] criar rankings individuais;
- [x] implementar RRF;
- [x] adicionar controle de `α`;
- [x] construir abas e matriz comparativa;
- [x] implementar Cross-Encoder nos Top-3;
- [x] adicionar comparação antes/depois do re-ranking;
- [x] testar consultas léxicas, semânticas e códigos médicos;
- [x] gerar gráfico comparativo;
- [ ] preparar relatório técnico em PDF.

## Decisões de implementação

### Escolha do modelo de embeddings

Três modelos multilíngues foram comparados no próprio corpus, medindo a **posição do
documento-alvo** para consultas que o BM25 não consegue resolver (sinônimos puros):

| Modelo | Rank médio do alvo | Doc 1 em "infarto" | Separação dos scores |
|---|---:|---:|---:|
| `paraphrase-multilingual-MiniLM-L12-v2` | 2.57 | 5º | 0.280 |
| `multilingual-e5-small` | 2.57 | 6º | 0.040 |
| **`distiluse-base-multilingual-cased-v1`** | **2.14** | **2º** | 0.235 |

O `distiluse` foi escolhido por ser o único que recupera o Doc 1 (*síndrome coronariana*)
no Top-3 para `infarto`, `ataque cardíaco` e `isquemia miocárdica` — exatamente a falha
dos sistemas léxicos descrita no estudo de caso. O `e5-small` acertava mais o Top-1, mas
comprimia todas as similaridades entre 0.82 e 0.88, o que tornaria os gráficos ilegíveis.

### Pré-processamento aplicado só ao motor léxico

A tokenização (minúsculas, remoção de acentos, de caracteres especiais e de stopwords)
alimenta **apenas** o BM25. O motor semântico recebe o texto original, porque o modelo de
embeddings já trata acentuação e palavras funcionais e perderia contexto com o texto
mutilado. A remoção de acentos faz `CÓD-ECG-12D` e `COD-ECG-12D` colidirem no mesmo token.

### Contingência sem internet

Se o modelo de embeddings não puder ser carregado, a aplicação cai automaticamente na
**simulação vetorial TF-IDF** permitida pelo enunciado, exibindo um aviso na interface.
Vale registrar que nesse modo os sinônimos deixam de ser recuperados: TF-IDF é um método
léxico e retorna similaridade 0.000 para `ataque cardíaco` em todos os seis documentos.

### Limitação conhecida do RRF

A fórmula exige um `Rank` para **todo** documento, inclusive os que o motor não recuperou.
Documentos com score BM25 = 0 recebem posição por desempate alfabético, então a
contribuição deles na fusão é arbitrária e não mede relevância. A fórmula foi mantida
exatamente como especificada no enunciado, e a interface sinaliza o caso na aba
*Matriz Comparativa*.

### Cross-Encoder

O modelo `cross-encoder/ms-marco-MiniLM-L-6-v2` indicado no enunciado é treinado em
inglês, o que produz scores muito negativos sobre textos em português (ex.: −11.1 para o
Doc 1). A ordenação relativa continua útil, mas os valores absolutos não devem ser lidos
como probabilidade de relevância.

## Resultado esperado

Ao final, o projeto deverá demonstrar de forma prática que diferentes estratégias de recuperação atendem a necessidades distintas e que a combinação de **BM25 + embeddings + RRF**, complementada opcionalmente por **Cross-Encoder Re-Ranking**, produz uma solução de recuperação de informação mais robusta para o cenário proposto.

---

**Instituição:** UNIPÊ — Centro Universitário de João Pessoa  
**Disciplina:** Tendências em Ciência da Computação — Recuperação de Informação / Processamento de Linguagem Natural  
**Projeto:** Desafio Integrador HealthSearch
