import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="HealthSearch",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


# Dados exclusivamente demonstrativos para validar a interface antes da
# implementação de BM25, embeddings, RRF e Cross-Encoder.
MOCK_BM25 = {
    "Doc 1": 7.82,
    "Doc 2": 5.41,
    "Doc 6": 3.18,
    "Doc 3": 1.22,
    "Doc 5": 0.74,
    "Doc 4": 0.31,
}

MOCK_SEMANTIC = {
    "Doc 2": 0.91,
    "Doc 1": 0.87,
    "Doc 6": 0.73,
    "Doc 3": 0.44,
    "Doc 5": 0.39,
    "Doc 4": 0.25,
}

MOCK_RRF = {
    "Doc 1": 0.01626,
    "Doc 2": 0.01613,
    "Doc 6": 0.01587,
    "Doc 3": 0.01538,
    "Doc 5": 0.01515,
    "Doc 4": 0.01493,
}

MOCK_CROSS_ENCODER = {
    "Doc 2": 8.74,
    "Doc 1": 8.29,
    "Doc 6": 5.31,
}


def corpus_by_id():
    return {doc["id"]: doc for doc in CORPUS}


def ranking_dataframe(scores, score_name):
    docs = corpus_by_id()
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    rows = []
    for position, (doc_id, score) in enumerate(ordered, start=1):
        doc = docs[doc_id]
        rows.append(
            {
                "Posição": position,
                "ID": doc_id,
                "Diretriz": doc["titulo"],
                score_name: score,
            }
        )

    return pd.DataFrame(rows)


def render_ranking_cards(scores, score_label, top_n=6):
    docs = corpus_by_id()
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

    for position, (doc_id, score) in enumerate(ordered, start=1):
        doc = docs[doc_id]
        with st.container(border=True):
            left, right = st.columns([8, 2])
            with left:
                st.markdown(f"### {position}º · {doc['titulo']}")
                st.caption(doc_id)
            with right:
                st.metric(score_label, f"{score:.5f}" if score < 1 else f"{score:.2f}")

            st.write(doc["conteudo"])


def rank_map(scores):
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {doc_id: position for position, (doc_id, _) in enumerate(ordered, start=1)}


def render_comparison_matrix():
    docs = corpus_by_id()
    bm25_rank = rank_map(MOCK_BM25)
    semantic_rank = rank_map(MOCK_SEMANTIC)
    rrf_rank = rank_map(MOCK_RRF)

    rows = []
    for doc_id in docs:
        rows.append(
            {
                "ID": doc_id,
                "Diretriz": docs[doc_id]["titulo"],
                "Rank BM25": bm25_rank[doc_id],
                "Rank Semântico": semantic_rank[doc_id],
                "Rank RRF": rrf_rank[doc_id],
            }
        )

    df = pd.DataFrame(rows).sort_values("Rank RRF")
    st.dataframe(df, hide_index=True, use_container_width=True)

    chart_df = df.set_index("ID")[["Rank BM25", "Rank Semântico", "Rank RRF"]]
    st.caption("Quanto menor a posição, melhor o ranking.")
    st.bar_chart(chart_df)


def render_cross_encoder_comparison():
    rrf_top3 = sorted(MOCK_RRF.items(), key=lambda item: item[1], reverse=True)[:3]
    rrf_rank = {doc_id: pos for pos, (doc_id, _) in enumerate(rrf_top3, start=1)}

    cross_order = sorted(
        MOCK_CROSS_ENCODER.items(), key=lambda item: item[1], reverse=True
    )
    cross_rank = {
        doc_id: pos for pos, (doc_id, _) in enumerate(cross_order, start=1)
    }

    docs = corpus_by_id()
    rows = []
    for doc_id, rrf_score in rrf_top3:
        rows.append(
            {
                "ID": doc_id,
                "Diretriz": docs[doc_id]["titulo"],
                "Rank RRF": rrf_rank[doc_id],
                "Score RRF": rrf_score,
                "Rank Cross-Encoder": cross_rank[doc_id],
                "Score Cross-Encoder": MOCK_CROSS_ENCODER[doc_id],
            }
        )

    st.dataframe(
        pd.DataFrame(rows).sort_values("Rank Cross-Encoder"),
        hide_index=True,
        use_container_width=True,
    )


with st.sidebar:
    st.title("⚙️ Configuração")
    st.caption("Controles visuais do protótipo")

    query = st.text_input(
        "Consulta médica",
        value="infarto",
        placeholder="Ex.: infarto, CÓD-ECG-12D...",
    )

    st.divider()
    st.subheader("BM25")
    k1 = st.slider(
        "k1 · Saturação de frequência",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
    )
    b = st.slider(
        "b · Normalização por tamanho",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )

    st.divider()
    st.subheader("Fusão híbrida")
    alpha = st.slider(
        "α · Peso BM25 no RRF",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 privilegia o ranking semântico; 1 privilegia o BM25.",
    )
    st.caption("Constante k_RRF = 60")

    st.divider()
    st.subheader("Bônus +0,3")
    use_cross_encoder = st.checkbox(
        "Aplicar Cross-Encoder Re-Ranking",
        value=False,
    )
    st.caption("Protótipo visual sobre os Top-3 do RRF.")


st.title("🩺 HealthSearch")
st.subheader("Motor de Busca Híbrido · BM25 + Semântico + RRF")

st.info(
    "🚧 Etapa atual: interface em desenvolvimento. Todos os scores, rankings e "
    "resultados exibidos nesta versão são DADOS MOCKADOS para validação visual. "
    "BM25, embeddings, RRF e Cross-Encoder ainda não estão implementados."
)

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(f"**Consulta atual:** `{query or '—'}`")
with header_right:
    st.caption(f"Corpus carregado: {len(CORPUS)} documentos")

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Documentos", len(CORPUS))
metric2.metric("k1", f"{k1:.2f}")
metric3.metric("b", f"{b:.2f}")
metric4.metric("α RRF", f"{alpha:.2f}")

st.caption(
    "Nesta versão os controles alteram apenas a interface. A ligação dos parâmetros "
    "com os cálculos reais será feita nas próximas etapas."
)

bm25_tab, semantic_tab, hybrid_tab, matrix_tab, corpus_tab = st.tabs(
    [
        "🔤 Busca Léxica · BM25",
        "🧠 Busca Semântica",
        "🔀 Híbrida · RRF",
        "📊 Matriz Comparativa",
        "📚 Corpus",
    ]
)

with bm25_tab:
    st.header("Ranking Léxico · BM25")
    st.write(
        "Visualização planejada para a busca baseada em correspondência de termos, "
        "com calibração pelos parâmetros k1 e b."
    )
    st.warning("Scores demonstrativos — BM25 real ainda não calculado.")
    render_ranking_cards(MOCK_BM25, "Score BM25")

with semantic_tab:
    st.header("Ranking Semântico · Embeddings")
    st.write(
        "Visualização planejada para similaridade contextual entre a consulta e as "
        "diretrizes médicas."
    )
    st.warning("Similaridades demonstrativas — embeddings ainda não gerados.")
    render_ranking_cards(MOCK_SEMANTIC, "Similaridade")

with hybrid_tab:
    st.header("Ranking Híbrido · Reciprocal Rank Fusion")
    st.write(
        "Combinação visual dos rankings léxico e semântico. O valor de α exibido na "
        "sidebar será usado posteriormente no cálculo real da fusão."
    )
    st.warning("Scores RRF demonstrativos — fusão real ainda não calculada.")
    render_ranking_cards(MOCK_RRF, "Score RRF")

    if use_cross_encoder:
        st.divider()
        st.subheader("🎯 Cross-Encoder Re-Ranking · Top-3")
        st.success(
            "Modo de demonstração habilitado. Os scores abaixo são mockados e servem "
            "apenas para validar a apresentação do desafio bônus."
        )
        render_cross_encoder_comparison()
    else:
        st.info(
            "Ative ‘Aplicar Cross-Encoder Re-Ranking’ na sidebar para visualizar a "
            "comparação mockada do bônus de +0,3 ponto."
        )

with matrix_tab:
    st.header("Matriz Comparativa de Rankings")
    st.write(
        "Comparação visual da posição de cada documento nos três mecanismos previstos."
    )
    st.warning("Posições demonstrativas — dados mockados para desenvolvimento da UI.")
    render_comparison_matrix()

with corpus_tab:
    st.header("Corpus Médico Obrigatório")
    st.write(
        "Os seis documentos definidos no enunciado já estão carregados na interface."
    )
    st.dataframe(pd.DataFrame(CORPUS), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "HealthSearch · Protótipo acadêmico — versão atual dedicada exclusivamente à "
    "interface e aos fluxos de navegação com dados mockados."
)
