import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from manual_landmarks import parse_landmarks
from landmarks import (
    LandmarkExtractor,
    draw_landmarks,
    landmarks_usados,
)
from registry import list_models, load_model

st.set_page_config(page_title="Classificador facial", layout="wide")


@st.cache_resource
def get_extractor():
    return LandmarkExtractor()


@st.cache_resource
def get_model(name: str):
    return load_model(name)


def render_model_page(modelos: list[str]) -> None:
    st.title("Conheça os modelos")
    st.markdown(
        "Veja como cada modelo transforma uma imagem em uma classificação. "
        "As etapas abaixo são lidas do pipeline salvo; os detalhes de treino "
        "são exibidos quando foram documentados junto ao modelo."
    )

    nome = st.selectbox("Modelo", modelos, key="modelo_documentacao")
    try:
        bundle = get_model(nome)
    except Exception as e:
        st.error(f"Falha ao carregar '{nome}': {e}")
        st.stop()

    config = bundle.config
    with st.container(border=True):
        st.subheader(nome.replace("_", " ").title())
        descricao = config.get("descricao", "").strip()
        if descricao:
            st.write(descricao)
        else:
            st.info(
                "Este modelo não tem uma ficha de treinamento. "
                "As etapas do pipeline salvo estão listadas abaixo, mas "
                "a origem dos dados e a validação não foram documentadas."
            )

        cols = st.columns(3)
        cols[0].metric(
            "Coordenadas de entrada",
            "Normalizadas" if config.get("coords") == "normalized" else "Pixels",
        )
        indices = config.get("keep_indices")
        if indices is not None:
            landmark_count = len(indices)
        else:
            selector = getattr(bundle.pipeline, "named_steps", {}).get("select")
            if selector is not None and (
                hasattr(selector, "get_support")
                or hasattr(selector, "transformers_")
            ):
                landmark_count = len(landmarks_usados(bundle.pipeline))
            else:
                landmark_count = 478
        cols[1].metric(
            "Landmarks usados",
            landmark_count,
        )
        resultado = config.get("results")
        cols[2].metric("Resultado registrado", resultado or "Não informado")

    st.subheader("Como a imagem percorre o modelo")
    st.caption(
        "O detector facial encontra 478 pontos. O pipeline aplica as "
        "transformações abaixo e então produz a classe prevista."
    )
    steps = getattr(bundle.pipeline, "steps", None)
    if steps:
        stages = ["Landmarks faciais"]
        for step_name, transformer in steps:
            class_name = type(transformer).__name__
            if class_name == "PrinceGPA":
                description = "Alinhamento Procrustes (GPA)"
            elif class_name == "GenericUnivariateSelect":
                description = "Seleção estatística de atributos"
            elif class_name.endswith("Classifier") or class_name == "SVC":
                description = f"Classificador {class_name}"
            else:
                description = class_name
            stages.append(f"{step_name}: {description}")
        stages.append("Classe prevista")
        st.markdown(" → ".join(stages))
    else:
        st.write(f"Pipeline: {type(bundle.pipeline).__name__}")

    parametros = config.get("parametros")
    if parametros:
        with st.expander("Parâmetros registrados"):
            st.write(parametros)
    st.info(
        "A classificação é experimental e não substitui avaliação ou "
        "diagnóstico profissional. Métricas registradas dependem do "
        "procedimento de avaliação usado no treinamento."
    )


# ---------------- Navegação e escolha do modelo ----------------
modelos = list_models()
if not modelos:
    st.sidebar.error("Nenhum modelo encontrado em models/")
    st.sidebar.info(
        "Adicione ao repositório pelo menos um arquivo .joblib em models/. "
        "Veja o README para instruções de exportação e configuração."
    )
    st.stop()

st.sidebar.header("Navegação")
pagina = st.sidebar.radio("Página", ["Classificação", "Sobre os modelos"])
if pagina == "Sobre os modelos":
    render_model_page(modelos)
    st.stop()

st.sidebar.header("Modelo")
nome_modelo = st.sidebar.selectbox("Selecione", modelos)
try:
    bundle = get_model(nome_modelo)
except Exception as e:
    st.sidebar.error(f"Falha ao carregar '{nome_modelo}': {e}")
    st.sidebar.info(
        "Se o erro cita uma classe (ex.: Procrustes), mova a definição dela "
        "para custom.py e re-salve o modelo a partir desse módulo."
    )
    st.stop()

if bundle.config["descricao"]:
    st.sidebar.caption(bundle.config["descricao"])
with st.sidebar.expander("Configuração"):
    st.json(bundle.config)

mostrar_pontos = st.sidebar.checkbox("Mostrar landmarks na imagem", value=True)

# ---------------- Corpo ----------------
st.title("Classificação a partir de landmarks faciais")
origem = st.radio(
    "Como deseja informar os dados?",
    ["Enviar imagem", "Colar landmarks manualmente"],
    horizontal=True,
)
img_rgb = None
pts = None
dimensoes_imagem = None

if origem == "Enviar imagem":
    arquivo = st.file_uploader(
        "Envie uma imagem", type=["jpg", "jpeg", "png", "webp"]
    )
    if arquivo is None:
        st.info("Envie uma foto de rosto para começar.")
        st.stop()

    img = ImageOps.exif_transpose(Image.open(arquivo)).convert("RGB")
    img_rgb = np.array(img)
    h, w = img_rgb.shape[:2]
    dimensoes_imagem = f"{w}×{h} px"

    pts = get_extractor().extract(img_rgb)
    if pts is None:
        st.error("Nenhum rosto detectado. Tente outra imagem.")
        st.image(img_rgb, use_container_width=True)
        st.stop()

    X = bundle.prepare(pts, w, h)
else:
    unidade = (
        "normalizadas (x e y divididos pela largura e altura)"
        if bundle.config["coords"] == "normalized"
        else "em pixels"
    )
    st.caption(
        f"Cole o conteúdo da coluna `landmarks` como 478 linhas com `x, y` "
        f"ou como uma lista de 956 números. Informe coordenadas {unidade}, "
        "na ordem "
        "`x0, y0, x1, y1, ...`."
    )
    texto_landmarks = st.text_area(
        "Landmarks (478 pares x, y)",
        height=220,
        placeholder="0.45, 0.32\n0.46, 0.33\n...",
    )
    if not texto_landmarks.strip():
        st.info("Cole os landmarks para começar.")
        st.stop()

    try:
        pts = parse_landmarks(texto_landmarks)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if bundle.config["keep_indices"] is not None:
        pts_modelo = pts[bundle.config["keep_indices"]]
    else:
        pts_modelo = pts
    X = pts_modelo.reshape(1, -1)

col1, col2 = st.columns([3, 2])
with col1:
    if img_rgb is not None:
        if mostrar_pontos:
            usados = landmarks_usados(bundle.pipeline, n_landmarks=len(pts))
            img_show = draw_landmarks(img_rgb, pts, usados)
            st.image(img_show, caption="Verde = usados; vermelho = não usados",
                     use_container_width=True)
        else:
            st.image(img_rgb, use_container_width=True)
    else:
        st.write(f"Landmarks informados: {len(pts)}")

with col2:
    try:
        classe, probs, score = bundle.predict(X)
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        st.caption(f"Shape enviado ao modelo: {X.shape}")
        st.stop()

    st.subheader("Resultado")
    st.metric("Categoria prevista", str(classe))
    if probs:
        st.bar_chart(probs)
    elif score is not None:
        classes = list(getattr(bundle.pipeline, "classes_", []))
        if len(classes) == 2:
            st.caption(
                f"Score de decisão: {score:+.3f} "
                f"(positivo → {classes[1]}, negativo → {classes[0]}). "
                "Quanto mais longe de 0, mais confiante."
            )
    with st.expander("Detalhes técnicos"):
        st.write(f"Entrada do modelo: {X.shape}")
        st.write(f"Origem dos dados: {origem}")
        if dimensoes_imagem:
            st.write(f"Imagem: {dimensoes_imagem}")