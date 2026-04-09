import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageOps
import math
import pandas as pd

st.set_page_config(
    page_title="코 수술 전후 분석기",
    page_icon="👃",
    layout="wide"
)

# ─── MediaPipe Face Mesh 랜드마크 인덱스 ──────────────────────────
NOSE_TIP     = 4    # Pronasale (코끝)
NASION       = 168  # 비근부 (Bridge root)
SUBNASALE    = 2    # 코기둥 기저부 (Columellar base)
LEFT_ALAR    = 129  # 좌측 alar base (코날개 기저)
RIGHT_ALAR   = 358  # 우측 alar base
L_EYE_INNER = 133  # 좌측 내안각
R_EYE_INNER = 362  # 우측 내안각
GLABELLA     = 9   # 미간 (비각 계산 기준)
UPPER_LIP    = 0   # 상순 중앙 (nasolabial angle)
CHIN         = 152  # 턱 (face height 기준)

FRONTAL_KEYS = [
    (NOSE_TIP,     'nose_tip',    (255, 220, 0)),
    (NASION,       'nasion',      (0,   200, 255)),
    (SUBNASALE,    'subnasale',   (200, 100, 255)),
    (LEFT_ALAR,    'left_alar',   (0,   255, 120)),
    (RIGHT_ALAR,   'right_alar',  (255, 100, 0)),
    (L_EYE_INNER,  'l_eye_inner', (120, 120, 255)),
    (R_EYE_INNER,  'r_eye_inner', (120, 120, 255)),
]

LATERAL_KEYS = [
    (NOSE_TIP,   'nose_tip',   (255, 220, 0)),
    (NASION,     'nasion',     (0,   200, 255)),
    (SUBNASALE,  'subnasale',  (200, 100, 255)),
    (GLABELLA,   'glabella',   (150, 255, 150)),
    (UPPER_LIP,  'upper_lip',  (255, 150, 100)),
    (CHIN,       'chin',       (200, 200, 200)),
]

LABEL_MAP = {
    'nose_tip':   'Tip',
    'nasion':     'Nasion',
    'subnasale':  'Subnasale',
    'left_alar':  'L.Alar',
    'right_alar': 'R.Alar',
    'l_eye_inner': '',
    'r_eye_inner': '',
    'glabella':   'Glabella',
    'upper_lip':  'Upper Lip',
    'chin':       'Chin',
}


# ─── 유틸 함수 ────────────────────────────────────────────────────
def fix_orientation(pil_img):
    """EXIF 회전 정보 적용"""
    return ImageOps.exif_transpose(pil_img)

def pil_to_rgb(pil_img):
    return np.array(pil_img.convert('RGB'))

def calc_dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def calc_angle(p1, vertex, p2):
    """vertex에서 p1-vertex-p2 사이 각도 (도)"""
    v1 = np.array([p1[0]-vertex[0], p1[1]-vertex[1]], dtype=float)
    v2 = np.array([p2[0]-vertex[0], p2[1]-vertex[1]], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)))


# ─── 랜드마크 감지 ────────────────────────────────────────────────
def get_landmarks(image_rgb, keys):
    h, w = image_rgb.shape[:2]
    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    )
    res = fm.process(image_rgb)
    fm.close()

    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    return {
        name: (int(lm[idx].x * w), int(lm[idx].y * h))
        for idx, name, _ in keys
    }


# ─── 측정 ─────────────────────────────────────────────────────────
def measure_frontal(pts):
    alar_w  = calc_dist(pts['left_alar'], pts['right_alar'])
    eye_w   = calc_dist(pts['l_eye_inner'], pts['r_eye_inner'])
    ratio   = (alar_w / eye_w * 100) if eye_w else 0
    return {
        '코 너비 (Alar width, px)':          round(alar_w, 1),
        '양안 내측 간격 (Inter-eye, px)':     round(eye_w,  1),
        '코너비 / 양안간격 비율 (%)':          round(ratio,  1),
    }

def measure_lateral(pts):
    nose_h  = calc_dist(pts['nasion'], pts['nose_tip'])
    face_h  = calc_dist(pts['nasion'], pts['chin'])
    h_ratio = (nose_h / face_h * 100) if face_h else 0
    nasion_angle     = calc_angle(pts['glabella'],  pts['nasion'],    pts['nose_tip'])
    nasolabial_angle = calc_angle(pts['nose_tip'],  pts['subnasale'], pts['upper_lip'])
    return {
        '비량 길이 (Nasion→Tip, px)':         round(nose_h,          1),
        '안면 높이 (Nasion→Chin, px)':        round(face_h,          1),
        '코높이 / 안면높이 비율 (%)':           round(h_ratio,         1),
        'Nasion 각도 (Nasal-dorsal angle, °)': round(nasion_angle,    1),
        'Nasolabial 각도 (°)':                 round(nasolabial_angle,1),
    }


# ─── 이미지 오버레이 ──────────────────────────────────────────────
def annotate(img_rgb, pts, keys, is_frontal):
    img = img_rgb.copy()
    h, w = img.shape[:2]
    scale = max(1, w // 400)   # 고해상도 이미지 대응

    if is_frontal:
        # 코 너비 선
        cv2.line(img, pts['left_alar'], pts['right_alar'], (0, 255, 200), scale*2)
        # 양안 기준선
        cv2.line(img, pts['l_eye_inner'], pts['r_eye_inner'], (120, 120, 255), scale)
    else:
        # 비배 (nasion → tip)
        cv2.line(img, pts['nasion'],    pts['nose_tip'],  (0, 200, 255), scale*2)
        # Nasion 각도 (glabella → nasion)
        cv2.line(img, pts['glabella'],  pts['nasion'],    (150, 255, 150), scale)
        # Nasolabial (tip → subnasale → upper_lip)
        cv2.line(img, pts['nose_tip'],  pts['subnasale'], (255, 220, 0),   scale*2)
        cv2.line(img, pts['subnasale'], pts['upper_lip'], (255, 150, 100), scale*2)

    for _, name, color in keys:
        if name not in pts:
            continue
        p = pts[name]
        cv2.circle(img, p, scale*5, color, -1)
        cv2.circle(img, p, scale*6, (255, 255, 255), scale)
        label = LABEL_MAP.get(name, name)
        if label:
            ox, oy = p[0]+scale*8, p[1]-scale*5
            cv2.putText(img, label, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45*scale, (30,30,30), scale*3)
            cv2.putText(img, label, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45*scale, color, scale)
    return img


# ─── 비교 테이블 생성 ─────────────────────────────────────────────
def build_table(before_m, after_m):
    rows = []
    for key in before_m:
        b, a = before_m[key], after_m[key]
        diff = a - b
        pct  = (diff / b * 100) if b else 0
        arrow = "▼" if diff < 0 else ("▲" if diff > 0 else "─")
        rows.append({
            '측정 항목':   key,
            '수술 전':     b,
            '수술 후':     a,
            '변화량':      round(diff, 1),
            '변화율':      f"{arrow} {abs(round(pct,1))}%",
        })
    return pd.DataFrame(rows)


# ─── 메인 UI ──────────────────────────────────────────────────────
st.title("👃 코 수술 전후 분석기")
st.caption("MediaPipe FaceMesh 기반 · 해부학적 랜드마크 자동 감지 · 수치 비교")

view = st.radio(
    "분석 유형",
    ["📷 정면 — 코 너비 (Alar width)", "👤 측면 — 높이 · Nasion 각도 · Nasolabial 각도"],
    horizontal=True,
)
is_frontal = "정면" in view
keys = FRONTAL_KEYS if is_frontal else LATERAL_KEYS

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader("수술 전 (Before)")
    before_file = st.file_uploader("업로드", type=['jpg','jpeg','png','webp'], key='before',
                                   label_visibility='collapsed')
with col2:
    st.subheader("수술 후 (After)")
    after_file  = st.file_uploader("업로드", type=['jpg','jpeg','png','webp'], key='after',
                                   label_visibility='collapsed')

if not (before_file and after_file):
    st.info("수술 전/후 사진을 모두 업로드하면 자동 분석합니다.")

    with st.expander("📌 측정 항목 설명 보기"):
        st.markdown("""
| 뷰 | 항목 | 설명 | 정상 범위 |
|---|---|---|---|
| 정면 | 코 너비 (Alar width) | 좌우 alar base 간 거리 | — |
| 정면 | 양안 내측 간격 | 좌우 내안각 간 거리 (정규화 기준) | — |
| 정면 | 코너비/양안간격 비율 | 코 너비의 상대적 크기 | **70~80 %** |
| 측면 | 비량 길이 | Nasion → Tip 거리 | — |
| 측면 | 코높이/안면높이 비율 | — | ~28~30 % |
| 측면 | Nasion 각도 | Glabella-Nasion-Tip (비배각) | 130~140 ° |
| 측면 | **Nasolabial 각도** | Tip-Subnasale-Upper Lip | **90~110 °** |
        """)
    st.stop()

# ─── 처리 ─────────────────────────────────────────────────────────
before_pil = fix_orientation(Image.open(before_file))
after_pil  = fix_orientation(Image.open(after_file))
before_rgb = pil_to_rgb(before_pil)
after_rgb  = pil_to_rgb(after_pil)

with st.spinner("랜드마크 감지 중…"):
    before_pts = get_landmarks(before_rgb, keys)
    after_pts  = get_landmarks(after_rgb,  keys)

err = []
if before_pts is None: err.append("수술 **전** 사진")
if after_pts  is None: err.append("수술 **후** 사진")
if err:
    st.error(f"얼굴 랜드마크 감지 실패: {', '.join(err)}\n\n"
             "정면/측면이 명확하게 나온 사진으로 다시 시도해 주세요.")
    st.stop()

before_ann = annotate(before_rgb, before_pts, keys, is_frontal)
after_ann  = annotate(after_rgb,  after_pts,  keys, is_frontal)

# ─── 결과 표시 ────────────────────────────────────────────────────
st.divider()
c1, c2 = st.columns(2)
with c1:
    st.image(before_ann, caption="수술 전 — 랜드마크 오버레이", use_container_width=True)
with c2:
    st.image(after_ann,  caption="수술 후 — 랜드마크 오버레이", use_container_width=True)

before_m = measure_frontal(before_pts) if is_frontal else measure_lateral(before_pts)
after_m  = measure_frontal(after_pts)  if is_frontal else measure_lateral(after_pts)

st.divider()
st.subheader("📊 수치 비교표")
st.dataframe(build_table(before_m, after_m), hide_index=True, use_container_width=True)

st.divider()
st.subheader("📋 핵심 요약")

if is_frontal:
    rb = before_m['코너비 / 양안간격 비율 (%)']
    ra = after_m ['코너비 / 양안간격 비율 (%)']
    delta = ra - rb
    st.metric("코너비/양안간격 비율", f"{ra} %",
              f"{'+' if delta>0 else ''}{round(delta,1)} %")
    if ra <= 80:
        st.success(f"✅ 수술 후 비율 {ra}% — 이상적인 범위(≤80%) 달성")
    elif ra <= 90:
        st.info(f"ℹ️ 수술 후 비율 {ra}% — 80~90% 범위")
    else:
        st.warning(f"⚠️ 수술 후 비율 {ra}% — 여전히 넓은 편(>90%)")

    aw_b = before_m['코 너비 (Alar width, px)']
    aw_a = after_m ['코 너비 (Alar width, px)']
    red  = aw_b - aw_a
    if red > 0:
        st.info(f"코 너비 **{round(red,1)} px ({round(red/aw_b*100,1)}%) 감소**")
else:
    nl_b  = before_m['Nasolabial 각도 (°)']
    nl_a  = after_m ['Nasolabial 각도 (°)']
    nas_b = before_m['Nasion 각도 (Nasal-dorsal angle, °)']
    nas_a = after_m ['Nasion 각도 (Nasal-dorsal angle, °)']

    c1, c2 = st.columns(2)
    with c1:
        d = nl_a - nl_b
        st.metric("Nasolabial 각도", f"{nl_a}°",
                  f"{'+' if d>0 else ''}{round(d,1)}°")
        if 90 <= nl_a <= 110:
            st.success(f"✅ {nl_a}° — 이상 범위(90~110°) 내")
        elif nl_a < 90:
            st.info(f"ℹ️ {nl_a}° — 90° 미만 (코끝이 아래를 향하는 경향)")
        else:
            st.info(f"ℹ️ {nl_a}° — 110° 초과 (코끝이 위를 향하는 경향)")
    with c2:
        d = nas_a - nas_b
        st.metric("Nasion 각도", f"{nas_a}°",
                  f"{'+' if d>0 else ''}{round(d,1)}°")
        if 130 <= nas_a <= 140:
            st.success(f"✅ {nas_a}° — 이상 범위(130~140°) 내")
        else:
            st.info(f"ℹ️ {nas_a}° — 기준 범위(130~140°) 참고")

    hb = before_m['비량 길이 (Nasion→Tip, px)']
    ha = after_m ['비량 길이 (Nasion→Tip, px)']
    if ha != hb:
        diff = ha - hb
        st.info(f"코 높이(비량): **{'+' if diff>0 else ''}{round(diff,1)} px ({'+' if diff>0 else ''}{round(diff/hb*100,1)}%)**")
