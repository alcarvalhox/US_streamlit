
import os
import sys
import io
import math
import zipfile
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# =====================================================================
# BLOQUEIO DE LOGS, TELEMETRIA E "PHONE HOME" DO YOLO
# =====================================================================
TMP_DIR = tempfile.gettempdir()
YOLO_CFG_DIR = os.path.join(TMP_DIR, 'Ultralytics_Config')
os.makedirs(YOLO_CFG_DIR, exist_ok=True)

os.environ['YOLO_CONFIG_DIR'] = YOLO_CFG_DIR
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLO_TELEMETRY'] = 'False'
os.environ['YOLO_UPDATE_CHECK'] = 'False'
os.environ['YOLO_SYNC'] = 'False'

# =====================================================================
# 0. AUTO-INSTALAÇÃO DE DEPENDÊNCIAS (opcional)
# =====================================================================
def install_dependencies():
    if os.path.exists("requirements.txt"):
        try:
            if 'dependencies_installed_hybrid_modelo' not in os.environ:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                os.environ['dependencies_installed_hybrid_modelo'] = '1'
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")

if __name__ == "__main__":
    install_dependencies()

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from scipy.spatial import cKDTree

# Permite importar o vetor tabular já criado anteriormente, assumindo que o arquivo
# pipeline_tabular_us_2026_revisado.py está no mesmo diretório deste app.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline_tabular_us_2026_revisado import TableVectorizer

# =====================================================================
# 1. CONFIGURAÇÕES GLOBAIS
# =====================================================================
CONFIG_MODELOS = {
    "Alma": ["best_alma_3.pt", "best_patim_1.pt"],
    "Boleto": ["best_boleto_1.pt", "best_patim_1.pt"],
    "Patim": ["best_patim_1.pt"]
}

LIMITES_PROFUNDIDADE = {
    "Boleto": (0.0, 52.0),
    "Alma": (53.0, 179.0),
    "Patim": (180.0, 223.0)
}

SONDAS_ESQUERDA = [0, 6, 8, 4, 10]
SONDAS_DIREITA = [1, 7, 9, 5, 11]

# ---------------------------------------------------------------------
# AJUSTE PEDIDO PELO USUÁRIO:
# TODOS os artefatos ficam na pasta `modelo` do GitHub.
# ---------------------------------------------------------------------
MODEL_DIR = "modelo"
ARTIFACTS_DIR = MODEL_DIR

# Tamanho da janela física por imagem B-scan
WINDOW_MM = 2400  # 2,4 m em mm
STEP_MM = 2400

# Tamanho da imagem sintética para o YOLO
BSCAN_WIDTH = 1500
BSCAN_HEIGHT = 500

# Cores visuais por probe
PROBE_TO_BGR = {
    0: (0, 255, 255), 1: (0, 255, 255),
    6: (0, 128, 0),   7: (0, 128, 0),
    8: (128, 0, 128), 9: (128, 0, 128),
    4: (0, 0, 255),   5: (0, 0, 255),
    10: (255, 0, 0),  11: (255, 0, 0)
}

# =====================================================================
# 2. FUNÇÕES AUXILIARES
# =====================================================================
def load_image_b64(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        import base64
        return base64.b64encode(f.read()).decode()


def remove_pontos_isolados(df: pd.DataFrame, raio: float = 15) -> pd.DataFrame:
    if df.empty or not {'odo', 'depth'}.issubset(df.columns):
        return df
    coords = df[['odo', 'depth']].values
    tree = cKDTree(coords)
    contagem = tree.query_ball_point(coords, r=raio, return_length=True)
    return df[contagem > 1].copy()


def normalize_odo_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'odo' in out.columns:
        odo_series = pd.to_numeric(out['odo'], errors='coerce')
        if odo_series.dropna().median() < 10000:
            out['odo'] = (odo_series * 1_000_000).round().astype(int)
        else:
            out['odo'] = odo_series.round().astype(int)
    elif 'odo_km' in out.columns:
        out['odo'] = (pd.to_numeric(out['odo_km'], errors='coerce') * 1_000_000).round().astype(int)
    elif 'x_m' in out.columns:
        out['odo'] = (pd.to_numeric(out['x_m'], errors='coerce') * 1000).round().astype(int)
    return out


def preprocessar_raw(files: List[io.BytesIO]) -> pd.DataFrame:
    lista = []
    for f in files:
        f.seek(0)
        df = pd.read_csv(f)
        df['source_upload'] = f.name
        lista.append(df)
    df_raw = pd.concat(lista, ignore_index=True)
    df_raw = normalize_odo_to_mm(df_raw)

    if 'depth_mm' not in df_raw.columns and 'depth' in df_raw.columns:
        df_raw['depth_mm'] = pd.to_numeric(df_raw['depth'], errors='coerce')
    if 'depth' not in df_raw.columns and 'depth_mm' in df_raw.columns:
        df_raw['depth'] = pd.to_numeric(df_raw['depth_mm'], errors='coerce')

    if 'side' not in df_raw.columns and 'probe' in df_raw.columns:
        probe = pd.to_numeric(df_raw['probe'], errors='coerce')
        df_raw['side'] = np.where(probe.isin(SONDAS_ESQUERDA), 'LEFT', np.where(probe.isin(SONDAS_DIREITA), 'RIGHT', 'UNKNOWN'))

    if 'level' in df_raw.columns:
        df_raw = df_raw[pd.to_numeric(df_raw['level'], errors='coerce') > 450].copy()

    df_raw = remove_pontos_isolados(df_raw)
    df_raw = df_raw.sort_values(by='odo').reset_index(drop=True)
    return df_raw


def separar_lados(df_local: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    if 'probe' in df_local.columns:
        df_esq = df_local[pd.to_numeric(df_local['probe'], errors='coerce').isin(SONDAS_ESQUERDA)].copy()
        df_dir = df_local[pd.to_numeric(df_local['probe'], errors='coerce').isin(SONDAS_DIREITA)].copy()
    else:
        side_upper = df_local['side'].astype(str).str.upper() if 'side' in df_local.columns else pd.Series(index=df_local.index, dtype=str)
        df_esq = df_local[side_upper.eq('LEFT')].copy()
        df_dir = df_local[side_upper.eq('RIGHT')].copy()
    return [("Trilho_Esq", df_esq), ("Trilho_Dir", df_dir)]


@st.cache_resource
def load_yolo_model(nome_pt: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path_pt = os.path.join(MODEL_DIR, nome_pt)
    if os.path.exists(path_pt):
        return YOLO(path_pt, task='segment', verbose=False)
    return None


@st.cache_resource
def load_tabular_artifacts(artifacts_dir: str = ARTIFACTS_DIR):
    """
    Carrega artefatos do modelo tabular a partir da pasta `modelo/`.
    Esperado em `modelo/`:
      - best_model.joblib
      - vectorizer.json
      - label_encoder.joblib
    """
    vectorizer_path = os.path.join(artifacts_dir, 'vectorizer.json')
    model_path = os.path.join(artifacts_dir, 'best_model.joblib')
    label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.joblib')

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {vectorizer_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {label_encoder_path}")

    vectorizer = TableVectorizer.load(vectorizer_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    try:
        model.n_jobs = 1
    except Exception:
        pass
    return vectorizer, model, label_encoder


def generate_bscan_buffer(df_win: pd.DataFrame, start: int, end: int, min_depth: float, max_depth: float) -> np.ndarray:
    width, height = BSCAN_WIDTH, BSCAN_HEIGHT
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    if df_win.empty:
        return img

    odos = pd.to_numeric(df_win['odo'], errors='coerce').values
    depths = pd.to_numeric(df_win['depth'], errors='coerce').values
    probes = pd.to_numeric(df_win['probe'], errors='coerce').fillna(-1).astype(int).values if 'probe' in df_win.columns else np.full(len(df_win), -1, dtype=int)

    delta_depth = max_depth - min_depth if (max_depth - min_depth) > 0 else 1.0
    x_coords = ((odos - start) / float(WINDOW_MM) * width).astype(int)
    y_coords = ((depths - min_depth) / float(delta_depth) * height).astype(int)

    size_x = 10
    size_y = 5
    base_triangle = np.array([[0, -size_y], [-size_x, size_y], [size_x, size_y]], dtype=np.int32)

    for x, y, p in zip(x_coords, y_coords, probes):
        if 0 <= x < width and 0 <= y < height:
            cv2.fillPoly(img, [base_triangle + [x, y]], PROBE_TO_BGR.get(int(p), (0, 255, 255)))
    return img


def map_bbox_to_physical(box_xyxy: np.ndarray, start_odo_mm: int, min_depth: float, max_depth: float) -> Dict[str, float]:
    x1, y1, x2, y2 = box_xyxy.astype(float)
    width = BSCAN_WIDTH
    height = BSCAN_HEIGHT

    odo1 = start_odo_mm + (x1 / width) * WINDOW_MM
    odo2 = start_odo_mm + (x2 / width) * WINDOW_MM
    delta_depth = max_depth - min_depth if (max_depth - min_depth) > 0 else 1.0
    depth1 = min_depth + (y1 / height) * delta_depth
    depth2 = min_depth + (y2 / height) * delta_depth

    return {
        'odo_min_mm': float(min(odo1, odo2)),
        'odo_max_mm': float(max(odo1, odo2)),
        'depth_min_mm': float(min(depth1, depth2)),
        'depth_max_mm': float(max(depth1, depth2)),
    }


def region_df_from_bbox(df_win: pd.DataFrame, bbox_phys: Dict[str, float]) -> pd.DataFrame:
    if df_win.empty:
        return df_win.copy()
    odo = pd.to_numeric(df_win['odo'], errors='coerce')
    depth = pd.to_numeric(df_win['depth'], errors='coerce')
    region = df_win[(odo >= bbox_phys['odo_min_mm']) & (odo <= bbox_phys['odo_max_mm']) &
                    (depth >= bbox_phys['depth_min_mm']) & (depth <= bbox_phys['depth_max_mm'])].copy()
    return region


def tabular_validate_region(region_df: pd.DataFrame, vectorizer: TableVectorizer, model, label_encoder) -> Dict[str, object]:
    if region_df.empty:
        return {'tabular_class': None, 'tabular_conf': None, 'status': 'regiao_vazia'}

    df_tab = region_df.copy()
    if 'depth_mm' not in df_tab.columns and 'depth' in df_tab.columns:
        df_tab['depth_mm'] = pd.to_numeric(df_tab['depth'], errors='coerce')
    if 'side' not in df_tab.columns and 'probe' in df_tab.columns:
        probe = pd.to_numeric(df_tab['probe'], errors='coerce')
        df_tab['side'] = np.where(probe.isin(SONDAS_ESQUERDA), 'LEFT', np.where(probe.isin(SONDAS_DIREITA), 'RIGHT', 'UNKNOWN'))
    if 'angle' not in df_tab.columns:
        df_tab['angle'] = np.nan

    feats = vectorizer.transform_one_df(df_tab).reshape(1, -1)
    pred_num = model.predict(feats)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    conf = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(feats)[0]
        conf = float(np.max(proba))
    return {'tabular_class': str(pred_label), 'tabular_conf': conf, 'status': 'ok'}


def annotate_detection_on_image(img_draw: np.ndarray, det: Dict[str, object], largura_vis: int, altura_vis: int):
    box = det['box'].astype(int)
    x1, y1, x2, y2 = box.tolist()
    sx = largura_vis / float(BSCAN_WIDTH)
    sy = altura_vis / float(BSCAN_HEIGHT)
    vx1, vy1, vx2, vy2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

    tab_class = det.get('tabular_class') or 'Sem_Validação'
    tab_conf = det.get('tabular_conf')
    yolo_class = det.get('yolo_class', 'N/A')

    cor_bbox = (0, 0, 255) if tab_class == 'Trinca' else (255, 0, 255) if tab_class == 'BHC' else (0, 255, 255)
    cv2.rectangle(img_draw, (vx1, vy1), (vx2, vy2), cor_bbox, 2)

    if tab_conf is not None:
        lbl = f"TAB: {tab_class} ({tab_conf:.2f}) | YOLO: {yolo_class}"
    else:
        lbl = f"TAB: {tab_class} | YOLO: {yolo_class}"

    cv2.putText(img_draw, lbl, (vx1 + 4, max(18, vy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(img_draw, lbl, (vx1 + 4, max(18, vy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_bbox, 1)


def detectar_e_validar(df_raw: pd.DataFrame, modelos_ativos: Dict[str, List[YOLO]], vectorizer: TableVectorizer, model_tab, label_encoder):
    progress_bar = st.progress(0.0, text="Gerando imagens B-scan, executando YOLO e validando BBs com o modelo tabular...")
    found = []
    gallery = []

    total_steps = 0
    for local_nome in modelos_ativos.keys():
        min_depth, max_depth = LIMITES_PROFUNDIDADE[local_nome]
        df_local = df_raw[(pd.to_numeric(df_raw['depth'], errors='coerce') >= min_depth) & (pd.to_numeric(df_raw['depth'], errors='coerce') < max_depth)]
        for _, df_side in separar_lados(df_local):
            if not df_side.empty:
                total_steps += max(1, math.ceil((int(df_side['odo'].max()) - int(df_side['odo'].min()) + 1) / STEP_MM))

    passo_atual = 0

    for local_nome, model_list in modelos_ativos.items():
        min_depth, max_depth = LIMITES_PROFUNDIDADE.get(local_nome, (0.0, 223.0))
        df_local = df_raw[(pd.to_numeric(df_raw['depth'], errors='coerce') >= min_depth) & (pd.to_numeric(df_raw['depth'], errors='coerce') < max_depth)].copy()
        if df_local.empty:
            continue

        for lado_nome, df_side in separar_lados(df_local):
            if df_side.empty:
                continue

            odo_min = int(df_side['odo'].min())
            odo_max = int(df_side['odo'].max())

            for start in range(odo_min, odo_max + 1, STEP_MM):
                passo_atual += 1
                if total_steps > 0:
                    progress_bar.progress(min(0.02 + 0.98 * passo_atual / total_steps, 1.0),
                                          text=f"Analisando {local_nome} ({lado_nome}) em ODO {start} mm...")

                end = start + WINDOW_MM
                df_win = df_side[(df_side['odo'] >= start) & (df_side['odo'] < end)].copy()
                if len(df_win) < 5:
                    continue

                img_base = generate_bscan_buffer(df_win, start, end, min_depth, max_depth)
                img_clean = img_base.copy()
                img_draw = cv2.resize(img_clean, (2400, 400), interpolation=cv2.INTER_LINEAR)

                raw_dets = []
                for model in model_list:
                    results = model.predict(img_clean, verbose=False, conf=0.05, iou=0.85)
                    if len(results[0].boxes) == 0:
                        continue
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        cls_nome = model.names[cls_id]
                        raw_dets.append({
                            'box': box.xyxy[0].cpu().numpy().astype(int),
                            'conf': float(box.conf),
                            'yolo_class': cls_nome,
                            'yolo_cls_id': cls_id,
                        })

                if not raw_dets:
                    continue

                raw_dets.sort(key=lambda x: x['conf'], reverse=True)
                final_dets = []
                for d in raw_dets:
                    suprimido = False
                    bx1 = d['box']
                    for f in final_dets:
                        bx2 = f['box']
                        xl, yt = max(bx1[0], bx2[0]), max(bx1[1], bx2[1])
                        xr, yb = min(bx1[2], bx2[2]), min(bx1[3], bx2[3])
                        if xr > xl and yb > yt:
                            inter = (xr - xl) * (yb - yt)
                            area1 = max(1, (bx1[2] - bx1[0]) * (bx1[3] - bx1[1]))
                            area2 = max(1, (bx2[2] - bx2[0]) * (bx2[3] - bx2[1]))
                            area_min = min(area1, area2)
                            if (inter / area_min) > 0.4:
                                suprimido = True
                                break
                    if not suprimido:
                        final_dets.append(d)

                houve_evento = False
                for local_id, det in enumerate(final_dets, 1):
                    bbox_phys = map_bbox_to_physical(det['box'], start_odo_mm=start, min_depth=min_depth, max_depth=max_depth)
                    region_df = region_df_from_bbox(df_win, bbox_phys)
                    tab = tabular_validate_region(region_df, vectorizer, model_tab, label_encoder)

                    det['tabular_class'] = tab['tabular_class']
                    det['tabular_conf'] = tab['tabular_conf']
                    det['status_tabular'] = tab['status']
                    det['bbox_phys'] = bbox_phys

                    if det['tabular_class'] is None:
                        continue

                    houve_evento = True
                    annotate_detection_on_image(img_draw, det, largura_vis=2400, altura_vis=400)

                    center_odo_mm = (bbox_phys['odo_min_mm'] + bbox_phys['odo_max_mm']) / 2.0
                    center_depth_mm = (bbox_phys['depth_min_mm'] + bbox_phys['depth_max_mm']) / 2.0
                    largura_mm = int(abs(bbox_phys['odo_max_mm'] - bbox_phys['odo_min_mm']))
                    altura_mm = int(abs(bbox_phys['depth_max_mm'] - bbox_phys['depth_min_mm']))

                    found.append({
                        'ID_Global': len(found),
                        'ID_Img': f"#{local_id}",
                        'Local': local_nome,
                        'Lado': lado_nome,
                        'Classe_YOLO': det['yolo_class'],
                        'Classe_Tabular': det['tabular_class'],
                        'ODO': int(center_odo_mm),
                        'ODO_Ref': start,
                        'Coordenada Depth(mm)': int(center_depth_mm),
                        'Largura(mm)': largura_mm,
                        'Altura(mm)': altura_mm,
                        'Confiança_YOLO': f"{det['conf']:.2%}",
                        'Confiança_Tabular': f"{det['tabular_conf']:.2%}" if det['tabular_conf'] is not None else None,
                        'Aprovado': True,
                        'BBox_ODO_Min_mm': int(bbox_phys['odo_min_mm']),
                        'BBox_ODO_Max_mm': int(bbox_phys['odo_max_mm']),
                        'BBox_Depth_Min_mm': float(bbox_phys['depth_min_mm']),
                        'BBox_Depth_Max_mm': float(bbox_phys['depth_max_mm']),
                        'Rows_Região_Tabular': int(len(region_df))
                    })

                if houve_evento:
                    gallery.append({
                        'img': img_draw,
                        'img_clean': img_clean,
                        'label': f"{lado_nome} @ {start}",
                        'odo_ref': start,
                        'lado': lado_nome,
                        'local': local_nome,
                    })

    progress_bar.progress(1.0, text="✅ Processamento concluído!")
    return found, gallery


def generate_excel_report(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Validacao_Hibrida')
    return output.getvalue()


def generate_zip_of_annotated_images(img_gallery: List[Dict[str, object]]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for item in img_gallery:
            nome = f"{item['local']}/{item['lado']}_{item['odo_ref']}.jpg"
            ok, buffer_img = cv2.imencode('.jpg', item['img'])
            if ok:
                zip_file.writestr(nome, buffer_img.tobytes())
    return zip_buffer.getvalue()

# =====================================================================
# 3. INTERFACE STREAMLIT
# =====================================================================
def main():
    st.set_page_config(
        page_title="Validação Híbrida YOLO + Modelo Tabular (US)",
        page_icon="logo.png",
        layout="wide"
    )

    css = """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stApp { background-color: #003865; }
    h1, h2, h3, h4, h5, p, label, .stMarkdown, .stText { color: white !important; }
    .stButton>button, [data-testid="stDownloadButton"] button {
        background-color: #FFC600; color: #003865 !important; font-weight: bold; border: none; border-radius: 5px;
    }
    .stButton>button:hover, [data-testid="stDownloadButton"] button:hover { background-color: #e6b300; color: white !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent !important; color: #FFC600 !important; }
    .stTabs [aria-selected="true"] { color: white !important; border-bottom-color: #FFC600 !important; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    col_logo, col_titulo = st.columns([1, 5])
    with col_logo:
        img_logo = load_image_b64('logo.png')
        if img_logo:
            st.markdown(f'<img src="data:image/png;base64,{img_logo}" style="width:140px; height:95px; object-fit:contain; border-radius:12px;"/>', unsafe_allow_html=True)
    with col_titulo:
        st.markdown("<h1 style='text-align:center;'>Validação Híbrida de Defeitos US: YOLO + Modelo Tabular</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Os arquivos da pasta <code>modelo/</code> são usados tanto pelo YOLO quanto pelo classificador tabular.</p>", unsafe_allow_html=True)

    with st.expander("ℹ️ Estrutura esperada da pasta modelo/"):
        st.code("""modelo/
  best_alma_3.pt
  best_boleto_1.pt
  best_patim_1.pt
  best_model.joblib
  vectorizer.json
  label_encoder.joblib
""")

    if 'hyb_deteccoes' not in st.session_state:
        st.session_state.hyb_deteccoes = []
    if 'hyb_gallery' not in st.session_state:
        st.session_state.hyb_gallery = []
    if 'hyb_page' not in st.session_state:
        st.session_state.hyb_page = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
    if 'hyb_uploader_key' not in st.session_state:
        st.session_state.hyb_uploader_key = 0

    col_upload, col_botoes = st.columns([3, 1])
    with col_upload:
        files = st.file_uploader(
            "Faça o upload dos arquivos CSV brutos de inspeção US",
            type=['csv'],
            accept_multiple_files=True,
            key=f"hyb_uploader_{st.session_state.hyb_uploader_key}"
        )

        arquivos_prontos = False
        if files:
            colunas_esperadas = {'frame', 'probe', 'level'}
            arquivos_validos = True
            for f in files:
                try:
                    header = pd.read_csv(f, nrows=0)
                    f.seek(0)
                    if not colunas_esperadas.issubset(set(header.columns)):
                        st.error(f"⚠️ O arquivo `{f.name}` não possui o conjunto mínimo de colunas esperado: {colunas_esperadas}")
                        arquivos_validos = False
                        break
                except Exception:
                    st.error(f"⚠️ Não foi possível ler o arquivo `{f.name}`.")
                    arquivos_validos = False
                    break
            if arquivos_validos:
                st.success(f"✅ {len(files)} arquivo(s) validados. Pronto para inferência híbrida.")
                arquivos_prontos = True

    with col_botoes:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_run = st.button("🚀 Iniciar inferência híbrida", type='primary', use_container_width=True, disabled=not arquivos_prontos)
        if st.button("🧹 Limpar upload", use_container_width=True):
            st.session_state.hyb_uploader_key += 1
            st.rerun()
        if st.button("🗑️ Resetar sistema", use_container_width=True):
            st.session_state.hyb_deteccoes = []
            st.session_state.hyb_gallery = []
            st.session_state.hyb_page = {"Alma": 0, "Boleto": 0, "Patim": 0, "🌐 Visão Global": 0}
            st.session_state.hyb_uploader_key += 1
            st.cache_resource.clear()
            st.rerun()

    if btn_run and arquivos_prontos:
        modelos_ativos = {}
        for local_nome, pts in CONFIG_MODELOS.items():
            lista_modelos = []
            for pt_file in pts:
                m = load_yolo_model(pt_file)
                if m is not None:
                    lista_modelos.append(m)
            if lista_modelos:
                modelos_ativos[local_nome] = lista_modelos

        if not modelos_ativos:
            st.error("Nenhum modelo YOLO (.pt) foi encontrado na pasta 'modelo'.")
            st.stop()

        try:
            vectorizer, model_tab, label_encoder = load_tabular_artifacts()
        except Exception as e:
            st.error(f"Erro ao carregar artefatos tabulares de `modelo/`: {e}")
            st.stop()

        try:
            df_raw = preprocessar_raw(files)
        except Exception as e:
            st.error(f"Erro no pré-processamento dos CSVs: {e}")
            st.stop()

        found, gallery = detectar_e_validar(df_raw, modelos_ativos, vectorizer, model_tab, label_encoder)
        if found:
            st.session_state.hyb_deteccoes = found
            st.session_state.hyb_gallery = gallery
            st.rerun()
        else:
            st.info("A inferência híbrida foi concluída, mas nenhuma detecção foi validada pelo modelo tabular.")

    if st.session_state.hyb_deteccoes or st.session_state.hyb_gallery:
        st.markdown("<hr>", unsafe_allow_html=True)
        df_raw = pd.DataFrame(st.session_state.hyb_deteccoes) if st.session_state.hyb_deteccoes else pd.DataFrame()
        if not df_raw.empty:
            if 'Aprovado' not in df_raw.columns:
                df_raw['Aprovado'] = True
            if 'Local' not in df_raw.columns:
                df_raw['Local'] = 'Alma'

        st.markdown("<h4 style='color:#FFC600; text-align:center;'>Alternar visualização:</h4>", unsafe_allow_html=True)
        locais_fixos = ["Alma", "Boleto", "Patim", "🌐 Visão Global"]
        local_selecionado = st.radio("Selecione:", locais_fixos, horizontal=True, label_visibility="collapsed")

        if local_selecionado == "🌐 Visão Global":
            tab_global = st.tabs(["📊 Relatório Global", "📦 Downloads"])
            with tab_global[0]:
                if not df_raw.empty:
                    df_aprov = df_raw[df_raw['Aprovado'] == True].copy()
                    st.markdown("##### Resumo Global (aprovados)")
                    contagem = df_aprov['Classe_Tabular'].value_counts().reset_index()
                    contagem.columns = ['Classe tabular validada', 'Quantidade']
                    st.dataframe(contagem, hide_index=True, use_container_width=True)

                    st.markdown("##### Relatório consolidado")
                    colunas = ['Local','Lado','Classe_YOLO','Classe_Tabular','ODO','Coordenada Depth(mm)','Largura(mm)','Altura(mm)','Confiança_YOLO','Confiança_Tabular','Rows_Região_Tabular']
                    st.dataframe(df_aprov[colunas], hide_index=True, use_container_width=True)
                else:
                    st.info('Nenhuma detecção aprovada disponível.')

            with tab_global[1]:
                if not df_raw.empty:
                    excel_data = generate_excel_report(df_raw)
                    st.download_button(
                        label="📥 Baixar relatório híbrido (Excel)",
                        data=excel_data,
                        file_name=f"relatorio_hibrido_yolo_tabular_{datetime.now().strftime('%d%m%Y_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                if st.session_state.hyb_gallery:
                    zip_imgs = generate_zip_of_annotated_images(st.session_state.hyb_gallery)
                    st.download_button(
                        label="🖼️ Baixar imagens anotadas (ZIP)",
                        data=zip_imgs,
                        file_name=f"imagens_anotadas_hibridas_{datetime.now().strftime('%d%m%Y_%H%M')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
        else:
            df_local = df_raw[df_raw['Local'] == local_selecionado].copy() if not df_raw.empty else pd.DataFrame()
            gal_local = [g for g in st.session_state.hyb_gallery if g.get('local') == local_selecionado]

            tabs = st.tabs(["📊 Tabela", "🖼️ Galeria"])
            with tabs[0]:
                if not df_local.empty:
                    st.markdown(f"##### Detecções validadas - {local_selecionado}")
                    colunas = ['Lado','Classe_YOLO','Classe_Tabular','ODO','Coordenada Depth(mm)','Largura(mm)','Altura(mm)','Confiança_YOLO','Confiança_Tabular','Rows_Região_Tabular']
                    st.dataframe(df_local[colunas], hide_index=True, use_container_width=True)
                else:
                    st.info(f"Nenhuma detecção validada em {local_selecionado}.")

            with tabs[1]:
                st.markdown("##### Galeria de imagens anotadas")
                if not gal_local:
                    st.info(f"Nenhuma imagem disponível em {local_selecionado}.")
                else:
                    itens_por_pagina = 12
                    total_pag = max(1, math.ceil(len(gal_local) / itens_por_pagina))
                    if st.session_state.hyb_page[local_selecionado] >= total_pag:
                        st.session_state.hyb_page[local_selecionado] = 0
                    ini = st.session_state.hyb_page[local_selecionado] * itens_por_pagina
                    fim = ini + itens_por_pagina
                    subset = gal_local[ini:fim]
                    cols = st.columns(3)
                    for idx, item in enumerate(subset):
                        with cols[idx % 3]:
                            st.image(item['img'], channels='BGR')
                            st.caption(item['label'])
                    if total_pag > 1:
                        c1, c2, c3 = st.columns([1, 2, 1])
                        with c1:
                            if st.session_state.hyb_page[local_selecionado] > 0:
                                if st.button('⬅️ Anterior', key=f'prev_{local_selecionado}', use_container_width=True):
                                    st.session_state.hyb_page[local_selecionado] -= 1
                                    st.rerun()
                        with c2:
                            st.markdown(f"<p style='text-align:center;'>Página {st.session_state.hyb_page[local_selecionado] + 1} de {total_pag}</p>", unsafe_allow_html=True)
                        with c3:
                            if st.session_state.hyb_page[local_selecionado] < total_pag - 1:
                                if st.button('Próxima ➡️', key=f'next_{local_selecionado}', use_container_width=True):
                                    st.session_state.hyb_page[local_selecionado] += 1
                                    st.rerun()

if __name__ == "__main__":
    main()
