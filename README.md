# Classificador de landmarks faciais

Aplicação Streamlit para enviar uma foto, escolher um modelo treinado e visualizar
a classificação **TEA vs. NEURO**. A aplicação extrai 478 landmarks faciais com
MediaPipe e envia as coordenadas ao pipeline selecionado.

> Os modelos treinados não são incluídos neste repositório. Adicione pelo menos
> um arquivo `.joblib` em `models/` antes de usar a predição ou publicar o app.

## Executar localmente

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
```

Ative o ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Adicionar um modelo treinado

Exporte o pipeline já treinado no Colab com `joblib` e coloque o arquivo no
diretório `models/`. Por exemplo:

```python
import joblib

joblib.dump(pipeline_treinado, "svc_v1.joblib")
```

No repositório, os arquivos devem ficar assim:

```text
models/
├── svc_v1.joblib
└── svc_v1.json       # opcional
```

Cada arquivo `.joblib` aparece automaticamente como opção no seletor do app.
O objeto deve ser um pipeline compatível com scikit-learn que aceite um vetor
com as coordenadas `(x0, y0, x1, y1, ...)` dos landmarks. Por padrão, são usados
os 478 pontos em pixels. Transformadores ou classes customizadas usados no
pipeline devem estar definidos em `custom.py`, para que o `joblib` consiga
carregá-los no deploy. No Colab, importe a classe desse módulo antes de montar
e exportar o pipeline (por exemplo, `from custom import PrinceGPA`); não
redefina a classe em uma célula do notebook, pois ela será salva como
`__main__.PrinceGPA`. O carregador mantém compatibilidade com esse caminho
antigo para o modelo `svc_gpa`.

O JSON opcional, com o mesmo nome-base do modelo, pode configurar:

| Chave | Valores | Padrão | Descrição |
| --- | --- | --- | --- |
| `descricao` | texto | `""` | Descrição exibida na barra lateral |
| `coords` | `"pixel"` ou `"normalized"` | `"pixel"` | Coordenadas em pixels ou normalizadas pela largura/altura |
| `keep_indices` | lista de inteiros ou `null` | `null` | Índices dos landmarks usados; `null` usa todos |
| `labels` | lista de nomes ou `null` | `null` | Nomes das classes na ordem dos índices previstos pelo modelo |

Exemplo de `models/svc_v1.json`:

```json
{
  "descricao": "SVC treinado com 478 landmarks",
  "coords": "pixel",
  "keep_indices": null,
  "labels": ["TEA", "NEURO"]
}
```

Use apenas arquivos de modelo confiáveis: carregar um arquivo `joblib` pode
executar código serializado.

## Publicar no Streamlit Community Cloud

1. Envie este projeto e os arquivos de modelo para um repositório GitHub.
2. No [Streamlit Community Cloud](https://share.streamlit.io/), crie um app
   selecionando o repositório, a branch e `app.py` como arquivo principal.
3. O serviço instalará as dependências Python listadas em `requirements.txt`
   e as dependências de sistema listadas em `packages.txt` (`libgl1` e
   `libglib2.0-0`, necessárias pelo OpenCV no ambiente Linux).

O repositório precisa conter ao menos um `models/*.joblib` para habilitar o
seletor e a predição. Verifique também os limites de tamanho de arquivo do
GitHub e do serviço de deploy antes de adicionar modelos grandes.