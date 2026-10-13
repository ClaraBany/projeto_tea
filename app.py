import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from landmarks import LandmarkExtractor, draw_landmarks, landmarks_usados
from registry import list_models, load_model

st.set_page_config(page_title="Classificador facial", layout="wide")


@st.cache_resource
def get_extractor():
    return LandmarkExtractor()


@st.cache_resource
def get_model(name: str):
    return load_model(name)


# ---------------- Sidebar: escolha do modelo ----------------
st.sidebar.header("Modelo")
modelos = list_models()
if not modelos:
    st.sidebar.error("Nenhum modelo encontrado em models/")
    st.sidebar.info(
        "Adicione ao repositório pelo menos um arquivo .joblib em models/. "
        "Veja o README para instruções de exportação e configuração."
    )
    st.stop()

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
arquivo = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png", "webp"])

if arquivo is None:
    st.info("Envie uma foto de rosto para começar.")
    st.stop()

img = ImageOps.exif_transpose(Image.open(arquivo)).convert("RGB")
img_rgb = np.array(img)
h, w = img_rgb.shape[:2]

pts = get_extractor().extract(img_rgb)
if pts is None:
    st.error("Nenhum rosto detectado. Tente outra imagem.")
    st.image(img_rgb, use_container_width=True)
    st.stop()

X = bundle.prepare(pts, w, h)

col1, col2 = st.columns([3, 2])
with col1:
    if mostrar_pontos:
        usados = landmarks_usados(bundle.pipeline, n_landmarks=len(pts))
        img_show = draw_landmarks(img_rgb, pts, usados)
        st.image(img_show, caption="Verde = usados; vermelho = não usados",
                 use_container_width=True)
    else:
        st.image(img_rgb, use_container_width=True)

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
        st.write(f"Imagem: {w}×{h} px")