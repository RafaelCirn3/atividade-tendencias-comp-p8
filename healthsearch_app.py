"""HealthSearch - Motor de Busca Hibrido (BM25 + Semantico + RRF).

Disciplina: Tendencias em Ciencia da Computacao - RI / PLN (UNIPE)

Arquitetura em 4 fases, na ordem em que aparecem neste arquivo:

    Fase 1  Ingestao do corpus medico e pre-processamento (tokenizacao,
            normalizacao e remocao de stopwords em portugues).
    Fase 2  Motor lexico Okapi BM25 com k1 e b calibraveis pela sidebar.
    Fase 3  Motor semantico vetorial (embeddings densos + similaridade
            de cosseno), com simulacao vetorial TF-IDF como contingencia.
    Fase 4  Fusao dos rankings via Reciprocal Rank Fusion (RRF).

    Bonus   Re-ranking com Cross-Encoder sobre o Top-3 do RRF.

    Alunos: Ewellyn Maria de França O. Andrade, Gustavo Palmeira de A. Martinez e Rafael Cirne Medeiros.
"""

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="HealthSearch",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constante de suavizacao de posicao exigida pelo enunciado.
K_RRF = 60

MODELO_EMBEDDINGS = "sentence-transformers/distiluse-base-multilingual-cased-v1"
MODELO_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Fase 1 - Ingestao do corpus medico
# ---------------------------------------------------------------------------

CORPUS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": (
            "Pacientes com dor precordial aguda e suspeita de síndrome coronariana "
            "devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários "
            "reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": (
            "A crise hipertensiva severa requer administração de anti-hipertensivos "
            "venosos e monitoramento contínuo da pressão arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado com "
            "trombolíticos venosos em até quatro horas e meia do início dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": (
            "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas "
            "de alta qualidade e desfibrilação precoce no código azul."
        ),
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, "
            "recomenda-se a monitorização cardíaca contínua por telemetria."
        ),
    },
]

IDS = [doc["id"] for doc in CORPUS]
DOCS_POR_ID = {doc["id"]: doc for doc in CORPUS}
TEXTOS = [f"{doc['titulo']}. {doc['conteudo']}" for doc in CORPUS]


# ---------------------------------------------------------------------------
# Fase 1 - Pre-processamento
# ---------------------------------------------------------------------------

# Lista embutida no proprio script para manter a entrega em arquivo unico e
# permitir execucao totalmente offline (sem download de corpora do NLTK).
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "isso", "isto", "ja", "la", "lhe", "lhes",
    "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito",
    "na", "nas", "nao", "nem", "no", "nos", "nossa", "nossas", "nosso",
    "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo",
    "pelos", "por", "qual", "quando", "que", "quem", "sao", "se", "seja",
    "sem", "ser", "seu", "seus", "so", "sobre", "sua", "suas", "tambem", "te",
    "tem", "teu", "teus", "tua", "tuas", "um", "uma", "voce", "voces",
}


def remover_acentos(texto):
    """Reduz o texto a ASCII para que 'CÓD-ECG-12D' e 'COD-ECG-12D' colidam."""
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def normalizar(texto):
    """Minusculas, sem acentos e sem caracteres especiais."""
    texto = remover_acentos(texto.lower())
    return re.sub(r"[^a-z0-9]+", " ", texto).strip()


def tokenizar(texto):
    """Aplica a normalizacao e elimina stopwords em portugues.

    Usado APENAS no motor lexico. O motor semantico recebe o texto original,
    porque o modelo de embeddings ja lida com acentuacao e palavras funcionais
    e perderia contexto se o texto fosse mutilado antes.
    """
    return [t for t in normalizar(texto).split() if t and t not in STOPWORDS_PT]


CORPUS_TOKENIZADO = [tokenizar(texto) for texto in TEXTOS]


# ---------------------------------------------------------------------------
# Fase 2 - Motor lexico Okapi BM25
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def construir_bm25(k1, b):
    """Instancia o BM25Okapi para um par (k1, b).

    O rank_bm25 congela k1 e b no construtor, entao cada combinacao escolhida
    nos sliders exige um indice novo - por isso o cache e chaveado por eles.
    """
    return BM25Okapi(CORPUS_TOKENIZADO, k1=k1, b=b)


def buscar_bm25(consulta, k1, b):
    """Retorna {doc_id: score BM25} para a consulta."""
    tokens = tokenizar(consulta)
    if not tokens:
        return {doc_id: 0.0 for doc_id in IDS}

    indice = construir_bm25(k1, b)
    scores = np.asarray(indice.get_scores(tokens), dtype=float)
    # Com k1 = 0 a razao tf/(tf + 0) vira 0/0 para termos ausentes do documento.
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return {doc_id: float(score) for doc_id, score in zip(IDS, scores)}


# ---------------------------------------------------------------------------
# Fase 3 - Motor semantico vetorial
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_bi_encoder():
    """Carrega o modelo de embeddings densos.

    Retorna (modelo, None) em caso de sucesso e (None, motivo) quando o modelo
    nao pode ser obtido - por exemplo, primeira execucao sem internet. Nesse
    caso a aplicacao cai na simulacao vetorial TF-IDF permitida pelo enunciado.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODELO_EMBEDDINGS), None
    except Exception as erro:  # download indisponivel, modelo ausente etc.
        return None, str(erro)


@st.cache_resource(show_spinner=False)
def vetorizar_corpus():
    """Pre-calcula os vetores do corpus uma unica vez por sessao."""
    modelo, erro = carregar_bi_encoder()
    if modelo is not None:
        vetores = modelo.encode(TEXTOS, normalize_embeddings=True)
        return "embeddings", vetores, None

    # Contingencia documentada: simulacao vetorial com TF-IDF.
    vetorizador = TfidfVectorizer(tokenizer=tokenizar, token_pattern=None)
    matriz = vetorizador.fit_transform(TEXTOS)
    return "tfidf", (vetorizador, matriz), erro


def buscar_semantico(consulta):
    """Retorna ({doc_id: similaridade de cosseno}, modo, erro)."""
    modo, artefato, erro = vetorizar_corpus()

    if not consulta.strip():
        return {doc_id: 0.0 for doc_id in IDS}, modo, erro

    if modo == "embeddings":
        modelo, _ = carregar_bi_encoder()
        vetor_consulta = modelo.encode([consulta], normalize_embeddings=True)
        # Vetores normalizados: o produto interno ja e a similaridade de cosseno.
        similaridades = cosine_similarity(vetor_consulta, artefato)[0]
    else:
        vetorizador, matriz = artefato
        similaridades = cosine_similarity(vetorizador.transform([consulta]), matriz)[0]

    scores = {doc_id: float(s) for doc_id, s in zip(IDS, similaridades)}
    return scores, modo, erro


def dimensao_vetorial():
    """Numero de dimensoes do espaco vetorial em uso."""
    _, artefato, _ = vetorizar_corpus()
    return artefato.shape[1] if hasattr(artefato, "shape") else artefato[1].shape[1]


# ---------------------------------------------------------------------------
# Fase 4 - Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def posicoes(scores):
    """Converte scores em posicoes de ranking (1 = melhor)."""
    ordenado = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {doc_id: posicao for posicao, (doc_id, _) in enumerate(ordenado, start=1)}


def fundir_rrf(scores_lexicos, scores_semanticos, alpha, k=K_RRF):
    """Score_RRF(D) = a * 1/(k + Rank_BM25) + (1 - a) * 1/(k + Rank_Semantico).

    A fusao opera sobre POSICOES, nao sobre os scores brutos: e isso que
    dispensa normalizar escalas heterogeneas (BM25 ilimitado x cosseno em [-1, 1]).
    """
    rank_lexico = posicoes(scores_lexicos)
    rank_semantico = posicoes(scores_semanticos)

    return {
        doc_id: alpha * (1.0 / (k + rank_lexico[doc_id]))
        + (1.0 - alpha) * (1.0 / (k + rank_semantico[doc_id]))
        for doc_id in IDS
    }


# ---------------------------------------------------------------------------
# Bonus - Cross-Encoder Re-Ranking
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando Cross-Encoder...")
def carregar_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(MODELO_CROSS_ENCODER), None
    except Exception as erro:
        return None, str(erro)


def reranquear_cross_encoder(consulta, doc_ids):
    """Pontua cada par (consulta, documento) com o Cross-Encoder."""
    modelo, erro = carregar_cross_encoder()
    if modelo is None:
        return None, erro

    pares = [[consulta, f"{DOCS_POR_ID[d]['titulo']}. {DOCS_POR_ID[d]['conteudo']}"] for d in doc_ids]
    notas = modelo.predict(pares)
    return {doc_id: float(nota) for doc_id, nota in zip(doc_ids, notas)}, None


# ---------------------------------------------------------------------------
# Camada de apresentacao
# ---------------------------------------------------------------------------

def formatar(valor):
    return f"{valor:.5f}" if abs(valor) < 1 else f"{valor:.3f}"


def render_ranking(scores, rotulo, top_n=6, ocultar_zeros=False):
    ordenado = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    if ocultar_zeros:
        descartados = [d for d, v in ordenado if v <= 0]
        ordenado = [(d, v) for d, v in ordenado if v > 0]
        if descartados:
            st.caption(
                f"{len(descartados)} documento(s) sem nenhuma correspondência léxica "
                f"foram omitidos: {', '.join(descartados)}."
            )

    for posicao, (doc_id, score) in enumerate(ordenado, start=1):
        doc = DOCS_POR_ID[doc_id]
        with st.container(border=True):
            esquerda, direita = st.columns([8, 2])
            with esquerda:
                st.markdown(f"### {posicao}º · {doc['titulo']}")
                st.caption(doc_id)
            with direita:
                st.metric(rotulo, formatar(score))
            st.write(doc["conteudo"])


def render_matriz(scores_lexicos, scores_semanticos, scores_rrf):
    rank_lexico = posicoes(scores_lexicos)
    rank_semantico = posicoes(scores_semanticos)
    rank_rrf = posicoes(scores_rrf)

    df = pd.DataFrame(
        [
            {
                "ID": doc_id,
                "Diretriz": DOCS_POR_ID[doc_id]["titulo"],
                "Rank BM25": rank_lexico[doc_id],
                "Rank Semântico": rank_semantico[doc_id],
                "Rank RRF": rank_rrf[doc_id],
                "Score BM25": round(scores_lexicos[doc_id], 4),
                "Score Semântico": round(scores_semanticos[doc_id], 4),
                "Score RRF": round(scores_rrf[doc_id], 5),
                "Δ Rank (RRF − BM25)": rank_lexico[doc_id] - rank_rrf[doc_id],
            }
            for doc_id in IDS
        ]
    ).sort_values("Rank RRF")

    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(
        "Δ Rank positivo = o documento subiu ao entrar na fusão híbrida. "
        "Nos gráficos, quanto menor a barra, melhor a posição."
    )

    # O RRF exige uma posicao para TODO documento, inclusive os que o motor nao
    # recuperou. Documentos com score 0 recebem posicao por desempate alfabetico,
    # entao a contribuicao deles na fusao e arbitraria - nao mede relevancia.
    zerados = [d for d in IDS if scores_lexicos[d] <= 0]
    if zerados:
        st.warning(
            f"⚠️ {len(zerados)} documento(s) têm score BM25 = 0 "
            f"({', '.join(zerados)}) mas ainda recebem uma posição no ranking léxico, "
            "porque a fórmula do RRF exige um Rank para cada documento. "
            "Esse desempate é alfabético e **não** reflete relevância — leia o "
            "Rank BM25 desses documentos como \"não recuperado\"."
        )
    st.bar_chart(df.set_index("ID")[["Rank BM25", "Rank Semântico", "Rank RRF"]])
    return df


# ---------------------------------------------------------------------------
# Sidebar - controles de calibracao
# ---------------------------------------------------------------------------

EXEMPLOS = [
    "infarto",
    "ataque cardíaco",
    "CÓD-ECG-12D",
    "AAS 100mg",
    "parada do coração",
    "derrame cerebral",
]

with st.sidebar:
    st.title("⚙️ Configuração")

    consulta = st.text_input(
        "Consulta médica",
        value="infarto",
        placeholder="Ex.: infarto, CÓD-ECG-12D...",
    )
    st.caption("Exemplos: " + " · ".join(f"`{e}`" for e in EXEMPLOS))

    st.divider()
    st.subheader("Fase 2 · BM25")
    k1 = st.slider(
        "k1 · Saturação de frequência",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="Controla o quanto repetições de um termo continuam somando pontos.",
    )
    b = st.slider(
        "b · Normalização por tamanho",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="0 ignora o tamanho do documento; 1 penaliza totalmente documentos longos.",
    )

    st.divider()
    st.subheader("Fase 4 · Fusão RRF")
    alpha = st.slider(
        "α · Peso do BM25 no RRF",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 privilegia o ranking semântico; 1 privilegia o BM25.",
    )
    st.caption(f"Constante de suavização k_RRF = {K_RRF}")

    st.divider()
    st.subheader("Bônus +0,3")
    usar_cross_encoder = st.checkbox("Aplicar Cross-Encoder Re-Ranking", value=False)
    st.caption("Re-ordena o Top-3 do RRF com um modelo de relevância par-a-par.")


# ---------------------------------------------------------------------------
# Execucao do pipeline
# ---------------------------------------------------------------------------

st.title("🩺 HealthSearch")
st.subheader("Motor de Busca Híbrido · BM25 + Semântico + RRF")

# Aquece os artefatos (indice BM25 e vetores do corpus) fora do cronometro:
# a latencia exibida deve refletir a BUSCA, nao o carregamento do modelo.
construir_bm25(k1, b)
vetorizar_corpus()

inicio = time.perf_counter()
scores_lexicos = buscar_bm25(consulta, k1, b)
tempo_lexico = (time.perf_counter() - inicio) * 1000

inicio = time.perf_counter()
scores_semanticos, modo_semantico, erro_semantico = buscar_semantico(consulta)
tempo_semantico = (time.perf_counter() - inicio) * 1000

inicio = time.perf_counter()
scores_rrf = fundir_rrf(scores_lexicos, scores_semanticos, alpha)
tempo_rrf = (time.perf_counter() - inicio) * 1000

if modo_semantico == "tfidf":
    st.warning(
        "⚠️ O modelo de embeddings não pôde ser carregado, então a busca semântica "
        "está rodando na **simulação vetorial TF-IDF** prevista no enunciado. "
        "Sem o modelo denso, sinônimos como *ataque cardíaco* → *síndrome coronariana* "
        f"não são recuperados. Motivo: `{erro_semantico}`"
    )

if not consulta.strip():
    st.info("Digite uma consulta na barra lateral para executar os três motores.")

cabecalho_esq, cabecalho_dir = st.columns([3, 1])
with cabecalho_esq:
    st.markdown(f"**Consulta atual:** `{consulta or '—'}`")
    st.caption(f"Tokens após pré-processamento: `{tokenizar(consulta) or '—'}`")
with cabecalho_dir:
    st.caption(f"Corpus carregado: {len(CORPUS)} documentos")
    st.caption(
        "Motor semântico: "
        + ("embeddings densos" if modo_semantico == "embeddings" else "TF-IDF (contingência)")
    )

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("k1", f"{k1:.2f}")
m2.metric("b", f"{b:.2f}")
m3.metric("α RRF", f"{alpha:.2f}")
m4.metric("Latência BM25", f"{tempo_lexico:.1f} ms")
m5.metric("Latência semântica", f"{tempo_semantico:.1f} ms")

aba_lexica, aba_semantica, aba_hibrida, aba_matriz, aba_corpus = st.tabs(
    [
        "🔤 Busca Léxica · BM25",
        "🧠 Busca Semântica",
        "🔀 Híbrida · RRF",
        "📊 Matriz Comparativa",
        "📚 Corpus",
    ]
)

with aba_lexica:
    st.header("Ranking Léxico · Okapi BM25")
    st.write(
        "Correspondência exata de termos após tokenização, remoção de acentos e de "
        "stopwords. É o motor que acerta códigos e dosagens, e o que falha em sinônimos."
    )
    st.latex(
        r"\text{BM25}(D,Q)=\sum_{q\in Q}\text{IDF}(q)\cdot"
        r"\frac{f(q,D)\cdot(k_1+1)}{f(q,D)+k_1\cdot\left(1-b+b\cdot\frac{|D|}{\text{avgdl}}\right)}"
    )
    if max(scores_lexicos.values()) == 0:
        st.error(
            "Nenhum documento contém os termos da consulta — o ponto cego do BM25. "
            "Compare com a aba Semântica."
        )
    render_ranking(scores_lexicos, "Score BM25", ocultar_zeros=True)

with aba_semantica:
    st.header("Ranking Semântico · Similaridade de Cosseno")
    st.write(
        "Consulta e documentos são projetados no mesmo espaço vetorial. Recupera "
        "sinônimos e paráfrases médicas, mas dilui a precisão de códigos exatos."
    )
    st.latex(
        r"\text{sim}(Q,D)=\cos(\theta)="
        r"\frac{\vec{Q}\cdot\vec{D}}{\|\vec{Q}\|\,\|\vec{D}\|}"
    )
    if modo_semantico == "embeddings":
        st.caption(f"Modelo: `{MODELO_EMBEDDINGS}` · {dimensao_vetorial()} dimensões")
    else:
        st.caption("Simulação vetorial TF-IDF sobre o corpus tokenizado.")
    render_ranking(scores_semanticos, "Similaridade")

with aba_hibrida:
    st.header("Ranking Híbrido · Reciprocal Rank Fusion")
    st.latex(
        r"\text{Score}_{RRF}(D)=\alpha\cdot\frac{1}{k_{RRF}+\text{Rank}_{BM25}(D)}"
        r"+(1-\alpha)\cdot\frac{1}{k_{RRF}+\text{Rank}_{Sem}(D)}"
    )
    st.caption(
        f"α = {alpha:.2f} · k_RRF = {K_RRF} · a fusão combina **posições**, não scores, "
        f"o que dispensa normalizar escalas diferentes. Fusão calculada em {tempo_rrf:.2f} ms."
    )
    render_ranking(scores_rrf, "Score RRF")

    if usar_cross_encoder:
        st.divider()
        st.subheader("🎯 Cross-Encoder Re-Ranking · Top-3")

        top3 = [doc_id for doc_id, _ in sorted(scores_rrf.items(), key=lambda i: (-i[1], i[0]))[:3]]
        notas, erro_ce = reranquear_cross_encoder(consulta, top3)

        if notas is None:
            st.error(f"Cross-Encoder indisponível: `{erro_ce}`")
        else:
            rank_rrf = {doc_id: pos for pos, doc_id in enumerate(top3, start=1)}
            ordem_ce = sorted(notas.items(), key=lambda i: (-i[1], i[0]))
            rank_ce = {doc_id: pos for pos, (doc_id, _) in enumerate(ordem_ce, start=1)}

            tabela = pd.DataFrame(
                [
                    {
                        "ID": doc_id,
                        "Diretriz": DOCS_POR_ID[doc_id]["titulo"],
                        "Rank RRF": rank_rrf[doc_id],
                        "Score RRF": round(scores_rrf[doc_id], 5),
                        "Rank Cross-Encoder": rank_ce[doc_id],
                        "Score Cross-Encoder": round(notas[doc_id], 4),
                        "Variação": rank_rrf[doc_id] - rank_ce[doc_id],
                    }
                    for doc_id in top3
                ]
            ).sort_values("Rank Cross-Encoder")

            st.dataframe(tabela, hide_index=True, width="stretch")
            trocas = int((tabela["Variação"] != 0).sum())
            st.caption(
                f"Modelo: `{MODELO_CROSS_ENCODER}` · "
                + (
                    f"{trocas} documento(s) mudaram de posição após o re-ranking."
                    if trocas
                    else "o Cross-Encoder confirmou a ordem produzida pelo RRF."
                )
            )
    else:
        st.info("Ative ‘Aplicar Cross-Encoder Re-Ranking’ na sidebar para o desafio bônus.")

with aba_matriz:
    st.header("Matriz Comparativa de Rankings")
    st.write("Posição de cada documento nos três motores, para a consulta atual.")
    render_matriz(scores_lexicos, scores_semanticos, scores_rrf)

with aba_corpus:
    st.header("Corpus Médico")
    st.dataframe(pd.DataFrame(CORPUS), hide_index=True, width="stretch")
    st.subheader("Pré-processamento (Fase 1)")
    st.dataframe(
        pd.DataFrame(
            [
                {"ID": doc["id"], "Tokens indexados": " ".join(tokens)}
                for doc, tokens in zip(CORPUS, CORPUS_TOKENIZADO)
            ]
        ),
        hide_index=True,
        width="stretch",
    )

st.divider()
st.caption("HealthSearch · UNIPÊ · Tendências em Ciência da Computação")
