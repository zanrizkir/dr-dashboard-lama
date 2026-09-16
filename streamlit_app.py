import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import base64
import io
import json
import re
import logging
from datetime import datetime
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Retina | Deteksi Retinopati Diabetik",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Design system
DESIGN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #09111f;
  --surface: #101a2b;
  --surface-2: #141f32;
  --surface-3: #18253a;
  --border: #24344d;
  --border-strong: #31445f;
  --text: #f1f5f9;
  --text-soft: #aab7c8;
  --muted: #6f8096;
  --accent: #35d0b5;
  --accent-soft: rgba(53, 208, 181, .10);
  --accent-border: rgba(53, 208, 181, .32);
  --warning: #f0b34a;
  --danger: #ef6b73;
  --info: #6ca9ff;
  --radius: 14px;
  --radius-sm: 10px;
  --shadow: 0 16px 40px rgba(0, 0, 0, .18);
  --font: 'Manrope', system-ui, sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;

  /* Alias kompatibilitas untuk markup lama agar styling tetap konsisten. */
  --bg-base: var(--bg);
  --bg-surface: var(--surface);
  --bg-elevated: var(--surface-2);
  --bg-card: var(--surface);
  --text-primary: var(--text);
  --text-secondary: var(--text-soft);
  --text-muted: var(--muted);
  --font-sans: var(--font);
  --font-mono: var(--mono);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

[data-testid="stMain"] { background: var(--bg) !important; }
.block-container {
  width: min(100%, 1540px) !important;
  margin: 0 auto !important;
  padding: 0 28px 36px !important;
}

h1, h2, h3, h4, h5, h6, p, label, button, input, select, textarea {
  font-family: var(--font) !important;
}

/* Streamlit memakai font ligature untuk ikon. Jangan timpa font ikon dengan font aplikasi. */
[data-testid="stIconMaterial"],
[data-testid$="Icon"],
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
  font-weight: normal !important;
  font-style: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap !important;
  word-wrap: normal !important;
  direction: ltr !important;
  -webkit-font-feature-settings: 'liga' !important;
  -webkit-font-smoothing: antialiased !important;
  font-feature-settings: 'liga' !important;
}

/* Span teks biasa tetap mengikuti font aplikasi, kecuali elemen ikon. */
span:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not(.material-icons) {
  font-family: var(--font) !important;
}

[data-testid="stElementContainer"] { min-width: 0 !important; }
[data-testid="column"] { min-width: 0 !important; }

/* Kurangi celah bawaan Streamlit agar judul panel dan kontrol terasa sebagai satu kelompok. */
[data-testid="column"] > div[data-testid="stVerticalBlock"] {
  gap: .45rem !important;
}

[data-testid="stButton"] > button {
  width: 100%;
  min-height: 40px;
  border: 1px solid var(--border-strong) !important;
  border-radius: var(--radius-sm) !important;
  background: var(--surface-2) !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  box-shadow: none !important;
  transition: border-color .2s ease, background .2s ease, transform .2s ease !important;
}
[data-testid="stButton"] > button:hover {
  border-color: var(--accent-border) !important;
  background: var(--surface-3) !important;
  transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
  border: 1px dashed var(--border-strong) !important;
  border-radius: var(--radius-sm) !important;
  background: #0d1727 !important;
  overflow: hidden !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
  background: #0d1727 !important;
  border: 0 !important;
  box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
  display: none !important;
}
[data-testid="stFileUploaderDropzone"] {
  min-height: 118px !important;
  padding: 18px 20px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 18px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span:first-of-type {
  position: relative !important;
  display: block !important;
  height: 20px !important;
  overflow: hidden !important;
  color: transparent !important;
  text-indent: -9999px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span:first-of-type::after {
  content: 'Seret dan lepas citra retina di sini';
  position: absolute;
  inset: 0;
  color: var(--text-soft);
  text-indent: 0;
  font-size: 14px;
  line-height: 20px;
}
[data-testid="stFileUploaderDropzoneInstructions"] span:last-of-type {
  position: relative !important;
  display: block !important;
  height: 16px !important;
  overflow: hidden !important;
  color: transparent !important;
  text-indent: -9999px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span:last-of-type::after {
  content: 'Format PNG, JPG, atau JPEG';
  position: absolute;
  inset: 0;
  color: var(--muted);
  text-indent: 0;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 16px;
}
[data-testid="stFileUploaderDropzone"] > button {
  position: relative !important;
  min-width: 108px !important;
  height: 38px !important;
  padding: 0 16px !important;
  overflow: hidden !important;
  color: transparent !important;
  text-indent: -9999px !important;
  border: 1px solid var(--accent-border) !important;
  border-radius: 9px !important;
  background: var(--accent) !important;
}
[data-testid="stFileUploaderDropzone"] > button > * {
  visibility: hidden !important;
}
[data-testid="stFileUploaderDropzone"] > button::after {
  content: 'Pilih file';
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #06201c;
  text-indent: 0;
  font-size: 12px;
  font-weight: 800;
}
[data-testid="stFileUploaderFile"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 9px !important;
  margin: 12px 14px 14px !important;
}

[data-testid="stSelectbox"] > div > div {
  min-height: 42px !important;
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--accent-border) !important;
}
[data-testid="stSelectbox"] label {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: .12em !important;
  text-transform: uppercase !important;
}

[data-testid="stImage"] { display: flex !important; justify-content: center !important; }
[data-testid="stImage"] img {
  width: 100% !important;
  max-width: 560px !important;
  aspect-ratio: 1 / 1 !important;
  object-fit: contain !important;
  background: #060b14 !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  margin: 0 auto !important;
}
[data-testid="stImageCaption"] {
  color: var(--muted) !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  text-align: center !important;
}

[data-testid="stDataFrame"], [data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--surface) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-weight: 700 !important; }
[data-testid="stExpander"] summary {
  min-height: 42px !important;
  align-items: center !important;
  gap: 8px !important;
}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
  flex: 0 0 auto !important;
  line-height: 1 !important;
}

.stMarkdown p { color: var(--text-soft) !important; line-height: 1.7 !important; }
.stMarkdown a { color: var(--accent) !important; }

.ri-hero {
  position: relative;
  overflow: hidden;
  padding: 64px 36px 52px;
  border-bottom: 1px solid var(--border);
  background:
    radial-gradient(circle at 88% 18%, rgba(53,208,181,.10), transparent 28%),
    linear-gradient(180deg, #0d1727 0%, #09111f 100%);
}
.ri-hero::before {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.015));
}
.ri-hero > * { position: relative; z-index: 1; }
.ri-hero-label, .ri-section-label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
}
.ri-hero-label {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 18px;
  color: var(--accent);
}
.ri-hero-label::before { content: ''; width: 22px; height: 1px; background: currentColor; }
.ri-hero h1 {
  margin: 0 !important;
  color: var(--text) !important;
  font-size: clamp(34px, 4.5vw, 54px) !important;
  font-weight: 800 !important;
  line-height: 1.05 !important;
  letter-spacing: -.045em !important;
}
.ri-hero h1 .accent { color: var(--accent); }
.ri-hero-sub {
  max-width: 660px;
  margin: 16px 0 0 !important;
  color: var(--text-soft) !important;
  font-size: 15px;
  line-height: 1.75;
}
.ri-hero-tags, .ri-empty-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.ri-hero-tags { margin-top: 24px; }
.ri-tag {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted);
  background: rgba(255,255,255,.015);
  font-family: var(--mono) !important;
  font-size: 10px !important;
}

.ri-section {
  width: 100%;
  max-width: 1440px;
  margin: 0 auto;
  padding: 36px 0;
}
.ri-section-label { margin-bottom: 8px; }
.ri-section-title {
  margin: 0 0 22px;
  color: var(--text);
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -.03em;
}

.ri-disclaimer {
  padding: 14px 18px;
  border: 1px solid rgba(53,208,181,.20);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  background: rgba(25, 75, 64, .22);
  color: #a9d9cf;
  font-size: 13px;
  line-height: 1.65;
}
.ri-disclaimer strong { color: #d4f5ed; }

.ri-upload-card, .ri-result-card, .ri-intel-card, .ri-stat, .ri-scan-card {
  height: 100%;
  box-sizing: border-box;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.ri-upload-card { padding: 18px 20px 12px; margin: 0; min-height: 74px; box-shadow: none; }
.ri-upload-card h4 {
  margin: 0 !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 800 !important;
}
.ri-result-card { padding: 24px; }

.ri-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 6px 11px;
  border-radius: 999px;
  font-family: var(--mono) !important;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .08em;
  white-space: nowrap;
}
.ri-badge-dot { width: 6px; height: 6px; border-radius: 50%; }
.ri-badge-monitor { color: #8dc0ff; background: rgba(65,128,210,.12); border: 1px solid rgba(108,169,255,.25); }
.ri-badge-monitor .ri-badge-dot { background: #8dc0ff; }
.ri-badge-engage { color: #f4c873; background: rgba(240,179,74,.10); border: 1px solid rgba(240,179,74,.25); }
.ri-badge-engage .ri-badge-dot { background: #f4c873; }
.ri-badge-act { color: #ff9aa1; background: rgba(239,107,115,.10); border: 1px solid rgba(239,107,115,.25); }
.ri-badge-act .ri-badge-dot { background: #ff9aa1; }

.ri-conf-bar-wrap { margin-top: 12px; }
.ri-conf-bar-label { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 8px; color: var(--text-soft); font-size: 12px; }
.ri-conf-bar-track, .ri-score-track { overflow: hidden; background: #0a1322; border-radius: 999px; }
.ri-conf-bar-track { height: 7px; }
.ri-conf-bar-fill { height: 100%; border-radius: inherit; background: var(--accent); }

.ri-score-row { display: grid; grid-template-columns: minmax(100px, 132px) minmax(0, 1fr) 48px; gap: 12px; align-items: center; padding: 9px 0; border-top: 1px solid rgba(49,68,95,.55); }
.ri-score-name { min-width: 0; color: var(--text-soft); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ri-score-track { height: 5px; }
.ri-score-fill { height: 100%; border-radius: inherit; background: rgba(53,208,181,.45); }
.ri-score-fill.active { background: var(--accent); }
.ri-score-pct { color: var(--muted); text-align: right; font-family: var(--mono) !important; font-size: 10px; }
.ri-score-pct.active { color: var(--accent); font-weight: 700; }

.ri-legend-row { display: flex; align-items: center; gap: 10px; padding: 8px 0; color: var(--text-soft); font-size: 13px; }
.ri-legend-swatch { width: 11px; height: 11px; border-radius: 3px; flex: 0 0 auto; }

.ri-stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.ri-stat { padding: 20px; }
.ri-stat-label, .ri-intel-card-label {
  color: var(--muted);
  font-family: var(--mono) !important;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.ri-stat-value { margin-top: 10px; color: var(--text); font-size: 28px; font-weight: 800; letter-spacing: -.03em; }
.ri-stat-sub { margin-top: 6px; color: var(--accent); font-size: 10px; }

.ri-intel-card { padding: 20px; }
.ri-intel-card-body { margin-top: 12px; color: var(--text-soft); font-size: 13px; line-height: 1.75; }
.ri-intel-card-channel { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); color: var(--muted); font-size: 11px; }
.ri-intel-card-channel span { color: var(--accent); }

.ri-ai-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 10px;
  border: 1px solid var(--accent-border);
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-family: var(--mono) !important;
  font-size: 10px;
}
.ri-ai-badge.offline { color: var(--text-soft); border-color: var(--border); background: var(--surface-2); }
.ri-ai-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

.ri-severity-band {
  padding: 15px 18px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-soft);
  font-size: 13px;
  line-height: 1.6;
}
.ri-severity-band strong { color: var(--text); }
.ri-severity-band.engage { border-left-color: var(--warning); }
.ri-severity-band.act { border-left-color: var(--danger); }

.ri-warn-banner, .ri-info-banner {
  margin-bottom: 18px;
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  line-height: 1.6;
}
.ri-warn-banner { color: #f5cb7b; background: rgba(240,179,74,.08); border: 1px solid rgba(240,179,74,.25); border-left: 3px solid var(--warning); }
.ri-info-banner { color: #a8cbff; background: rgba(108,169,255,.08); border: 1px solid rgba(108,169,255,.25); border-left: 3px solid var(--info); }
.ri-scan-note { margin-top: 12px; padding: 10px 14px; border-radius: var(--radius-sm); font-size: 12px; }
.ri-scan-note.ok { color: #9ee6d7; background: rgba(53,208,181,.08); border: 1px solid var(--accent-border); }

.ri-empty {
  display: grid;
  place-items: center;
  min-height: 360px;
  padding: 48px 24px;
  text-align: center;
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface);
}
.ri-empty-icon { width: 46px; height: 46px; border: 1px solid var(--accent-border); border-radius: 50%; margin-bottom: 18px; background: var(--accent-soft); }
.ri-empty-title { color: var(--text); font-size: 20px; font-weight: 800; }
.ri-empty-body { max-width: 520px; margin-top: 10px; color: var(--text-soft); font-size: 14px; line-height: 1.7; }
.ri-empty-tags { justify-content: center; margin-top: 22px; }

.ri-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 26px 0 0;
  border-top: 1px solid var(--border);
}
.ri-footer-text { color: var(--muted); font-family: var(--mono) !important; font-size: 10px; line-height: 1.7; }
.ri-footer-badge { padding: 5px 9px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); font-family: var(--mono) !important; font-size: 10px; white-space: nowrap; }

@media (max-width: 1100px) {
  .ri-stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 820px) {
  .block-container { padding: 0 18px 28px !important; }
  .ri-hero { padding: 48px 8px 38px; }
  .ri-section { padding: 28px 0; }
  .ri-stat-grid { grid-template-columns: 1fr; }
  .ri-score-row { grid-template-columns: 96px minmax(0, 1fr) 42px; gap: 8px; }
  [data-testid="stFileUploaderDropzone"] { align-items: flex-start !important; flex-direction: column !important; }
  [data-testid="stFileUploaderDropzone"] > button { width: 100% !important; }
  .ri-footer { align-items: flex-start; flex-direction: column; }
}
</style>
"""

st.markdown(DESIGN_CSS, unsafe_allow_html=True)

# Struktur data klasifikasi
CLASS_LABELS = ["Tanpa RD", "RD Ringan", "RD Sedang", "RD Berat", "RD Proliferatif"]
RISK_TIERS   = ["PANTAU", "PANTAU", "TINDAK LANJUT", "SEGERA", "SEGERA"]

CLINICAL_DESC = [
    "Tidak terdeteksi tanda retinopati diabetik. Struktur vaskular retina tampak sehat.",
    "Ditemukan mikroaneurisma tahap awal. Belum ada temuan yang mengancam penglihatan.",
    "Terdeteksi perubahan non-proliferatif sedang. Rujukan ke dokter mata dalam 30 hari.",
    "Retinopati non-proliferatif berat. Rujukan mendesak diperlukan dalam 1 minggu.",
    "Retinopati diabetik proliferatif terkonfirmasi. Perlu intervensi spesialis segera.",
]

# Template fallback klinis Bahasa Indonesia - JANGAN DIHAPUS
AI_SUMMARIES = [
    {
        "summary": "Pemeriksaan citra retina tidak menunjukkan tanda retinopati diabetik. Struktur vaskular tampak normal tanpa mikroaneurisma, eksudat, maupun neovaskularisasi.",
        "action":  "Lanjutkan kontrol diabetes rutin. Jadwalkan penapisan retina berkala berikutnya dalam rentang 12 bulan.",
        "hcp":     "Kirim pengingat digital berkala melalui portal layanan pasien. Belum memerlukan intervensi rujukan darurat.",
        "channel": "Portal Pasien Digital",
    },
    {
        "summary": "Ditemukan mikroaneurisma tahap awal pada area retina perifer, konsisten dengan retinopati diabetik non-proliferatif ringan. Belum ada temuan yang membahayakan tajam penglihatan.",
        "action":  "Optimalkan kontrol glikemik serta pemantauan tekanan darah. Lakukan evaluasi penapisan retina ulang dalam 6-12 bulan.",
        "hcp":     "Tandai berkas untuk konsultasi dokter umum / faskes primer. Berikan konseling target kontrol gula darah.",
        "channel": "Layanan Primer + Digital",
    },
    {
        "summary": "Terdeteksi retinopati diabetik non-proliferatif derajat sedang. Terdapat mikroaneurisma dan perdarahan intraretina ringan. Risiko progresi meningkat tanpa manajemen yang tepat.",
        "action":  "Rujuk pasien ke dokter spesialis mata (oftalmologis) dalam 30 hari untuk pemeriksaan fundus mendalam.",
        "hcp":     "Tandai di sistem rujukan faskes primer untuk konsultasi spesialis mata. Tingkatkan intensitas pemantauan berkala.",
        "channel": "Sistem Rujukan Faskes",
    },
    {
        "summary": "Terdeteksi retinopati diabetik non-proliferatif derajat berat. Ditemukan perdarahan retina menyeluruh dan kelainan mikrovaskular dengan risiko tinggi transisi ke tahap proliferatif.",
        "action":  "Rujukan segera ke spesialis mata subspesialis retina dalam 1 minggu. Pertimbangkan evaluasi terapi pan-retinal photocoagulation.",
        "hcp":     "Kirim notifikasi urgensi tinggi ke tim spesialis mata. Prioritaskan jalur rujukan cepat.",
        "channel": "Rujukan Segera Spesialis Mata",
    },
    {
        "summary": "Retinopati diabetik proliferatif terkonfirmasi. Terdapat tanda neovaskularisasi aktif yang berisiko tinggi mengancam penglihatan dalam waktu dekat.",
        "action":  "Intervensi spesialis mata darurat dalam pekan yang sama. Segera evaluasi kebutuhan terapi anti-VEGF atau tindakan bedah retina.",
        "hcp":     "Koordinasikan segera dengan fasilitas rujukan oftalmologi komprehensif. Tangani sebagai kasus prioritas darurat mata.",
        "channel": "Koordinasi Darurat Spesialis",
    },
]

# Konfigurasi Ollama (LLM lokal)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT  = 12

SYSTEM_PROMPT = """Anda adalah asisten penulisan teks klinis untuk dashboard skrining diabetic retinopathy (DR).
ATURAN KETAT - WAJIB DIPATUHI:
1. Anda HANYA menulis ulang/merangkai fakta yang diberikan di bawah. DILARANG menambah diagnosis, angka, atau detail klinis baru yang tidak ada di input.
2. DILARANG mengubah kelas diagnosis atau tingkat keparahan yang sudah ditentukan.
3. Jawaban HARUS dalam format JSON PERSIS seperti ini, tanpa teks tambahan apa pun di luar JSON:
{"summary": "...", "action": "...", "hcp": "...", "channel": "..."}
4. "summary": maksimal 2 kalimat, jelaskan temuan klinis untuk kelas diagnosis ini secara umum (bukan detail visual spesifik yang tidak diberikan).
5. "action": maksimal 1-2 kalimat, rekomendasi tindak lanjut standar sesuai tingkat keparahan.
6. "hcp": maksimal 1-2 kalimat, saran engagement untuk tenaga kesehatan.
7. "channel": maksimal 3-5 kata, nama kanal komunikasi singkat.
8. DILARANG memberi saran obat/dosis spesifik, DILARANG mengklaim kepastian diagnosis 100%, DILARANG bahasa yang menakut-nakuti berlebihan.
9. Jika ragu, gunakan bahasa netral dan rujuk ke "evaluasi profesional medis lebih lanjut"."""

USER_PROMPT_TEMPLATE = """Data hasil klasifikasi model AI (JANGAN diubah, HANYA dirangkai jadi kalimat):
- Diagnosis: {diagnosis}
- Confidence: {confidence}%
- Skor per kelas: {class_scores}

Tulis output sesuai format JSON yang diwajibkan di atas."""


def validate_and_sanitize_intelligence(raw_text: str, expected_diagnosis: str) -> dict | None:
    """Validasi ketat output LLM untuk mencegah kegagalan format dan halusinasi."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    required_keys = ["summary", "action", "hcp", "channel"]
    for key in required_keys:
        if key not in data or not isinstance(data[key], str) or not data[key].strip():
            return None

    if len(data["summary"]) > 450 or len(data["action"]) > 450:
        return None
    if len(data["hcp"]) > 450 or len(data["channel"]) > 60:
        return None

    summary_lower = data["summary"].lower()
    if expected_diagnosis == "Tanpa RD":
        if any(term in summary_lower for term in ["proliferative", "proliferatif", "severe", "berat", "neovaskularisasi", "haemorrhage", "perdarahan"]):
            logging.warning("Output LLM ditolak: kontradiksi diagnosis Tanpa RD.")
            return None
    elif "Proliferatif" in expected_diagnosis:
        if any(term in summary_lower for term in ["tanpa retinopati", "tidak ada retinopati", "no diabetic retinopathy", "sehat", "normal"]):
            logging.warning("Output LLM ditolak: kontradiksi diagnosis RD Proliferatif.")
            return None

    return {k: data[k].strip() for k in required_keys}


def generate_clinical_intelligence(
    diagnosis: str,
    confidence: float,
    class_scores: dict[str, str],
    predicted_class: int,
    endpoint: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    timeout: int = OLLAMA_TIMEOUT,
) -> tuple[dict, dict]:
    """Panggil LLM lokal via Ollama REST API dengan timeout dan verifikasi output.
    Jika offline, timeout, atau format salah, otomatis fallback ke AI_SUMMARIES tanpa crash."""
    fallback_data = AI_SUMMARIES[predicted_class]

    formatted_scores = ", ".join(f"{k}: {v}" for k, v in class_scores.items())
    user_prompt = USER_PROMPT_TEMPLATE.format(
        diagnosis=diagnosis,
        confidence=f"{confidence:.1f}",
        class_scores=formatted_scores,
    )

    url = f"{endpoint.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }

    try:
        logging.info(f"Mengirim permintaan clinical intelligence ke Ollama ({model})...")
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            logging.warning(f"Ollama mengembalikan HTTP status {resp.status_code}")
            return fallback_data, {
                "source": "fallback",
                "reason": f"HTTP {resp.status_code}",
                "badge": f"Template klinis standar (Ollama HTTP {resp.status_code})",
                "ollama": False,
            }

        raw_output = resp.json().get("response", "")
        validated = validate_and_sanitize_intelligence(raw_output, diagnosis)

        if validated is not None:
            logging.info("Clinical intelligence dari LLM lokal berhasil divalidasi.")
            return validated, {
                "source": "ollama",
                "model": model,
                "badge": f"Dihasilkan oleh AI Lokal (Ollama: {model})",
                "ollama": True,
            }
        else:
            logging.warning("Output LLM tidak lolos validasi. Menggunakan fallback.")
            return fallback_data, {
                "source": "fallback",
                "reason": "Format LLM tidak lolos validasi",
                "badge": "Template klinis standar (validasi output aktif)",
                "ollama": False,
            }

    except requests.exceptions.Timeout:
        logging.warning(f"Koneksi ke Ollama timeout (> {timeout}s). Menggunakan fallback.")
        return fallback_data, {
            "source": "fallback",
            "reason": f"Timeout (> {timeout}s)",
            "badge": "Template klinis standar (Ollama tidak merespons)",
            "ollama": False,
        }
    except requests.exceptions.ConnectionError:
        logging.info("Ollama tidak aktif di localhost:11434. Menggunakan template fallback.")
        return fallback_data, {
            "source": "fallback",
            "reason": "Layanan Ollama tidak aktif",
            "badge": "Template klinis standar (AI lokal tidak aktif)",
            "ollama": False,
        }
    except Exception as e:
        logging.warning(f"Kendala saat komunikasi dengan Ollama: {e}. Menggunakan fallback.")
        return fallback_data, {
            "source": "fallback",
            "reason": str(e),
            "badge": "Template klinis standar (cadangan)",
            "ollama": False,
        }


def crop_retina(img_np: np.ndarray, save_debug: bool = True, debug_prefix: str = "upload") -> tuple[np.ndarray, dict]:
    """Deteksi fundus retina dan crop ke bounding box persegi 1:1, dengan fallback ke frame penuh."""
    if img_np is None or img_np.size == 0:
        return img_np, {"success": False, "reason": "Citra kosong", "bbox": None, "aspect_ratio": 1.0}

    if img_np.dtype != np.uint8:
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    else:
        img_uint8 = img_np

    h_orig, w_orig = img_uint8.shape[:2]
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    success = False
    bbox = None
    aspect_ratio = 1.0
    reason = ""

    if not contours:
        reason = "Tidak ditemukan kontur, menggunakan frame penuh"
        crop = img_np
    else:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = float(h_orig * w_orig)

        if area < 0.05 * img_area:
            reason = f"Kontur terlalu kecil ({(area / img_area) * 100:.1f}% dari citra, menggunakan frame penuh)"
            crop = img_np
        else:
            x, y, w, h = cv2.boundingRect(largest)
            bbox = (x, y, w, h)
            aspect_ratio = w / float(max(1, h))

            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                reason = f"Rasio aspek ekstrem ({aspect_ratio:.2f}), menggunakan frame penuh"
                crop = img_np
            else:
                center_x = x + w / 2.0
                center_y = y + h / 2.0
                side = max(w, h)

                x1 = int(center_x - side / 2.0)
                y1 = int(center_y - side / 2.0)
                x2 = x1 + side
                y2 = y1 + side

                if x1 < 0:
                    x2 -= x1; x1 = 0
                if y1 < 0:
                    y2 -= y1; y1 = 0
                if x2 > w_orig:
                    x1 -= (x2 - w_orig); x2 = w_orig
                if y2 > h_orig:
                    y1 -= (y2 - h_orig); y2 = h_orig

                x1 = max(0, x1)
                y1 = max(0, y1)

                raw_crop = img_np[y1:y2, x1:x2]
                ch, cw = raw_crop.shape[:2]

                if ch != cw:
                    pad_side = max(ch, cw)
                    crop = np.zeros((pad_side, pad_side, 3), dtype=img_np.dtype)
                    oy = (pad_side - ch) // 2
                    ox = (pad_side - cw) // 2
                    crop[oy:oy+ch, ox:ox+cw] = raw_crop
                else:
                    crop = raw_crop

                crop_h, crop_w = crop.shape[:2]
                crop_ratio = crop_w / float(max(1, crop_h))
                bbox = (x, y, w, h)
                aspect_ratio = crop_ratio
                success = True
                reason = f"Retina terdeteksi dan dicrop persegi ({crop_w}x{crop_h}px, rasio {crop_ratio:.2f})"

    status_str = f"SUCCESS bbox=({bbox}), rasio_akhir={aspect_ratio:.2f}" if success else f"FALLBACK ({reason})"
    logging.info(f"[crop_retina] {status_str}")

    if save_debug:
        os.makedirs("debug_crops", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        debug_fname = f"debug_crops/{debug_prefix}_{timestamp}"
        try:
            crop_uint8 = (crop * 255).astype(np.uint8) if crop.dtype != np.uint8 and crop.max() <= 1.0 else crop.astype(np.uint8)
            cv2.imwrite(f"{debug_fname}_crop.png", cv2.cvtColor(crop_uint8, cv2.COLOR_RGB2BGR))
            debug_viz = img_uint8.copy()
            if bbox is not None:
                bx, by, bw, bh = bbox
                cv2.rectangle(debug_viz, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            color = (0, 255, 0) if success else (0, 165, 255)
            cv2.putText(debug_viz, status_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imwrite(f"{debug_fname}_input.png", cv2.cvtColor(debug_viz, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logging.warning(f"Gagal menyimpan debug crop: {e}")

    crop_info = {
        "success": success,
        "bbox": bbox,
        "aspect_ratio": aspect_ratio,
        "reason": reason,
    }
    return crop, crop_info


@st.cache_resource(show_spinner="Memuat model EfficientNetB3...")
def load_model():
    m = tf.keras.models.load_model('models/efficientnetb3_dr.keras', compile=False)
    logging.info("Model EfficientNetB3 berhasil dimuat.")
    return m


def make_gradcam(img_input: np.ndarray, raw_rgb_uint8: np.ndarray, model: tf.keras.Model, colormap: str = "Jet (Standar)") -> str | None:
    """Menghasilkan heatmap Grad-CAM yang dioverlay di atas citra input."""
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break

        if last_conv is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(img_input)
            if isinstance(conv_out, list):
                conv_out = conv_out[0]
            if isinstance(predictions, list):
                predictions = predictions[0]
            pred_class = tf.argmax(predictions[0])
            class_score = predictions[0][pred_class]

        grads = tape.gradient(class_score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_out[0]), axis=-1).numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        cam_resized = cv2.resize(cam, (300, 300), interpolation=cv2.INTER_LINEAR)
        cam_uint8 = (cam_resized * 255).astype(np.uint8)

        if "Viridis" in colormap:
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_VIRIDIS)
        elif "Turbo" in colormap:
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_TURBO)
        else:
            heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        orig = raw_rgb_uint8.astype(np.uint8)
        blended = (0.55 * orig + 0.45 * heatmap_rgb).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        logging.error(f"Kesalahan Grad-CAM: {e}")
        return None


def b64_to_pil(b64_str: str) -> Image.Image:
    """Mendekode string PNG base64 menjadi citra PIL."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


if "history" not in st.session_state:
    st.session_state["history"] = []

# Hero
st.markdown("""
<div class="ri-hero anim-fadein">
  <div class="ri-hero-label">EfficientNetB3 · APTOS 2019 · Grad-CAM</div>
  <h1>Klasifikasi <span class="accent">AI</span> Retina</h1>
  <p class="ri-hero-sub">
    Penilaian keparahan retinopati diabetik secara otomatis dari citra fundus.
    Klasifikasi lima kelas dengan peta perhatian berbobot gradien.
  </p>
  <div class="ri-hero-tags">
    <span class="ri-tag">EfficientNetB3</span>
    <span class="ri-tag">APTOS 2019</span>
    <span class="ri-tag">Grad-CAM</span>
    <span class="ri-tag">AI Lokal (Ollama)</span>
    <span class="ri-tag">5 Kelas Keparahan</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="ri-section anim-fadein-d1" style="padding-bottom:0;">
  <div class="ri-disclaimer">
    <strong>Hanya untuk Riset dan Demonstrasi.</strong>
    Alat ini dibuat untuk keperluan riset dan evaluasi.
    Seluruh hasil wajib diverifikasi oleh dokter spesialis mata sebelum digunakan untuk pengambilan keputusan klinis.
  </div>
</div>
""", unsafe_allow_html=True)

model = load_model()

# Panel unggah dan pengaturan
st.markdown('<div class="ri-section" style="padding-bottom:8px;">', unsafe_allow_html=True)

upload_col, settings_col = st.columns([3, 1], gap="medium")

with upload_col:
    st.markdown('<div class="ri-upload-card"><h4>Unggah Citra Fundus Retina</h4>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Seret dan lepas atau klik untuk memilih file — PNG / JPG / JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with settings_col:
    st.markdown('<div class="ri-upload-card"><h4>Pengaturan Grad-CAM</h4>', unsafe_allow_html=True)
    colormap_choice = st.selectbox(
        "Palet Peta Panas",
        options=["Jet (Standar)", "Viridis (Ramah Buta Warna)", "Turbo (Kontras Tinggi)"],
        index=0,
        help="Pilih skema warna untuk peta perhatian model. Viridis disarankan untuk pengguna dengan buta warna merah-hijau.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Inferensi dan render hasil
if uploaded_file is not None:

    raw_pil = Image.open(uploaded_file).convert('RGB')
    raw_np = np.array(raw_pil)

    cropped_np, crop_info = crop_retina(raw_np, save_debug=True, debug_prefix="web_upload")

    resized_pil = Image.fromarray(cropped_np).resize((300, 300), Image.BILINEAR)
    resized_np_uint8 = np.array(resized_pil)

    img_float = resized_np_uint8.astype('float32')
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(np.copy(img_float))
    img_input = np.expand_dims(img_preprocessed, axis=0)

    with st.spinner("Menganalisis pindaian retina dan menghitung peta perhatian Grad-CAM..."):
        probs           = model.predict(img_input, verbose=0)[0]
        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class])
        gradcam_b64     = make_gradcam(img_input, resized_np_uint8, model, colormap=colormap_choice)

    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])
    top1_prob = float(probs[top1_idx])
    top2_prob = float(probs[top2_idx])
    prob_margin = top1_prob - top2_prob

    label = CLASS_LABELS[predicted_class]
    tier  = RISK_TIERS[predicted_class]
    desc  = CLINICAL_DESC[predicted_class]

    class_score_dict = {lbl: f"{p * 100:.1f}%" for lbl, p in zip(CLASS_LABELS, probs)}
    cache_key = f"intel_{uploaded_file.name}_{uploaded_file.size}_{predicted_class}"
    if cache_key not in st.session_state:
        with st.spinner("Menghasilkan ringkasan klinis terstruktur melalui AI lokal (Ollama)..."):
            summary, intel_meta = generate_clinical_intelligence(
                diagnosis=label,
                confidence=confidence * 100,
                class_scores=class_score_dict,
                predicted_class=predicted_class,
            )
            st.session_state[cache_key] = (summary, intel_meta)
    else:
        summary, intel_meta = st.session_state[cache_key]

    scan_record = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "filename": uploaded_file.name,
        "diagnosis": label,
        "confidence": f"{confidence * 100:.1f}%",
        "risk_tier": tier,
        "crop_status": "Pemotongan Persegi 1:1" if crop_info["success"] else "Cadangan (Mentah)",
    }
    if not any(h["filename"] == uploaded_file.name and h["timestamp"] == scan_record["timestamp"] for h in st.session_state["history"]):
        st.session_state["history"].insert(0, scan_record)

    st.markdown('<div class="ri-section">', unsafe_allow_html=True)

    if confidence < 0.40:
        st.markdown(f"""
        <div class="ri-warn-banner">
          <strong>Keyakinan Rendah ({confidence * 100:.1f}%).</strong>
          Citra mungkin memiliki pencahayaan tidak lazim, framing sebagian, atau fitur ambigu.
          Verifikasi manual oleh tenaga ahli disarankan.
        </div>
        """, unsafe_allow_html=True)
    elif prob_margin < 0.10:
        st.markdown(f"""
        <div class="ri-info-banner">
          <strong>Klasifikasi Ambigu.</strong>
          Prediksi berdekatan antara <strong>{CLASS_LABELS[top1_idx]}</strong> ({top1_prob * 100:.1f}%)
          dan <strong>{CLASS_LABELS[top2_idx]}</strong> ({top2_prob * 100:.1f}%).
          Selisih: {prob_margin * 100:.1f}%.
        </div>
        """, unsafe_allow_html=True)

    col_scan, col_result = st.columns(2, gap="large")

    with col_scan:
        badge_class = "ri-badge-monitor" if predicted_class <= 1 else ("ri-badge-engage" if predicted_class == 2 else "ri-badge-act")
        st.markdown('<div class="ri-section-label">Pindaian Retina</div>', unsafe_allow_html=True)
        st.image(resized_pil, caption="Citra fundus terproses - pemotongan persegi 1:1 terpusat, 300 × 300 piksel", use_container_width=True)

        if crop_info["success"]:
            st.markdown(f"""
            <div class="ri-scan-note ok">
              Pemotongan retina diterapkan - rasio aspek {crop_info['aspect_ratio']:.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ri-info-banner" style="margin-top:10px; margin-bottom:0; padding:10px 16px; font-size:12px;">
              Framing: {crop_info['reason']}
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        conf_pct = confidence * 100
        st.markdown(f"""
        <div class="ri-section-label">Hasil Klasifikasi</div>
        <div class="ri-result-card">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:18px;">
            <div>
              <div style="font-size:30px; font-weight:700; letter-spacing:-0.03em; color:var(--text-primary); line-height:1;">{label}</div>
              <div style="font-size:12px; color:var(--text-secondary); font-family:var(--font-mono); margin-top:6px;">{desc}</div>
            </div>
            <span class="ri-badge {badge_class}"><span class="ri-badge-dot"></span>{tier}</span>
          </div>
          <div class="ri-conf-bar-wrap">
            <div class="ri-conf-bar-label">
              <span>Tingkat Keyakinan Model</span>
              <span style="color:var(--accent); font-weight:500;">{conf_pct:.1f}%</span>
            </div>
            <div class="ri-conf-bar-track">
              <div class="ri-conf-bar-fill" style="width:{conf_pct:.1f}%;"></div>
            </div>
          </div>
          <div style="margin-top:24px;">
            <div style="font-family:var(--font-mono); font-size:10px; text-transform:uppercase; letter-spacing:0.12em; color:var(--text-muted); margin-bottom:10px;">Skor per Kelas</div>
        """, unsafe_allow_html=True)

        for i, (lbl, p) in enumerate(zip(CLASS_LABELS, probs)):
            is_active = i == predicted_class
            fill_cls = "active" if is_active else ""
            pct_cls = "active" if is_active else ""
            st.markdown(f"""
            <div class="ri-score-row">
              <span class="ri-score-name">{lbl}</span>
              <div class="ri-score-track">
                <div class="ri-score-fill {fill_cls}" style="width:{p*100:.1f}%;"></div>
              </div>
              <span class="ri-score-pct {pct_cls}">{p*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Grad-CAM
    st.markdown('<div class="ri-section" style="padding-top:8px;">', unsafe_allow_html=True)
    st.markdown('<div class="ri-section-label">Interpretasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="ri-section-title">Peta Panas Grad-CAM</div>', unsafe_allow_html=True)

    if gradcam_b64:
        gradcam_img = b64_to_pil(gradcam_b64)
        g_col_img, g_col_info = st.columns([3, 2], gap="large")

        with g_col_img:
            st.image(
                gradcam_img,
                caption=f"Peta aktivasi berbobot gradien ({colormap_choice})",
                use_container_width=True,
            )

        with g_col_info:
            if "Viridis" in colormap_choice:
                legend_items = [
                    ("#fde68a", "Kuning / Terang: aktivasi tertinggi (patologi utama)"),
                    ("#34d399", "Toska / Hijau: aktivasi sedang"),
                    ("#6d28d9", "Ungu Gelap: baseline / pengaruh rendah"),
                ]
            else:
                legend_items = [
                    ("#ef4444", "Merah / Oranye: aktivasi tertinggi (patologi utama)"),
                    ("#fbbf24", "Kuning / Hijau: aktivasi sedang"),
                    ("#3b82f6", "Biru / Sejuk: dasar / pengaruh rendah"),
                ]

            st.markdown("""
            <div style="font-family:var(--font-mono); font-size:10px; text-transform:uppercase;
                        letter-spacing:0.12em; color:var(--text-muted); margin-bottom:12px;">
              Legenda Peta Panas
            </div>
            """, unsafe_allow_html=True)

            for color, label_text in legend_items:
                st.markdown(f"""
                <div class="ri-legend-row">
                  <div class="ri-legend-swatch" style="background:{color};"></div>
                  <span>{label_text}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:16px; padding:14px 16px; background:var(--bg-elevated);
                        border:1px solid var(--border); border-radius:var(--radius-card);
                        font-size:13px; color:var(--text-secondary); line-height:1.65;">
              Peta panas menyoroti struktur anatomi yang mempengaruhi skor klasifikasi,
              termasuk diskus optikus, makula, arkade vaskular, dan kluster mikroaneurisma.
              Pindaian normal berfokus pada pola pembuluh darah sentral, sedangkan pindaian RD
              berpusat pada area hemoragik dan eksudatif.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ri-warn-banner">Visualisasi Grad-CAM tidak dapat dihasilkan untuk citra ini.</div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Performa model
    st.markdown('<div class="ri-section" style="padding-top:8px;">', unsafe_allow_html=True)
    st.markdown('<div class="ri-section-label">Performa Model</div>', unsafe_allow_html=True)

    metrics = {}
    metrics_path = os.path.join('models', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    val_acc = metrics.get('accuracy')
    kappa = metrics.get('kappa')
    train_samples = metrics.get('train_samples')
    val_acc_str = f"{val_acc*100:.1f}%" if isinstance(val_acc, (float, int)) else "Tidak tersedia"
    kappa_str = f"{kappa:.2f}" if isinstance(kappa, (float, int)) else "Tidak tersedia"
    train_samples_str = f"{train_samples:,}" if isinstance(train_samples, int) else "Tidak tersedia"

    st.markdown(f"""
    <div class="ri-stat-grid">
      <div class="ri-stat">
        <div class="ri-stat-label">Akurasi Validasi</div>
        <div class="ri-stat-value">{val_acc_str}</div>
        <div class="ri-stat-sub">Validasi APTOS 2019</div>
      </div>
      <div class="ri-stat">
        <div class="ri-stat-label">Kappa Cohen</div>
        <div class="ri-stat-value">{kappa_str}</div>
        <div class="ri-stat-sub">Kesepakatan antar penilai</div>
      </div>
      <div class="ri-stat">
        <div class="ri-stat-label">Data Latih</div>
        <div class="ri-stat-value">{train_samples_str}</div>
        <div class="ri-stat-sub">Citra fundus</div>
      </div>
      <div class="ri-stat">
        <div class="ri-stat-label">Kelas Keparahan</div>
        <div class="ri-stat-value">5</div>
        <div class="ri-stat-sub">Tanpa RD hingga Proliferatif</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Wawasan klinis AI
    st.markdown('<div class="ri-section" style="padding-top:8px;">', unsafe_allow_html=True)

    ai_is_ollama = intel_meta.get("ollama", False)
    ai_badge_cls = "ri-ai-badge" if ai_is_ollama else "ri-ai-badge offline"
    ai_badge_text = intel_meta.get("badge", "")

    header_col, action_col = st.columns([4, 1])
    with header_col:
        st.markdown(f"""
        <div class="ri-section-label">Wawasan Klinis Berbasis AI</div>
        <div class="ri-section-title" style="margin-bottom:8px;">Ringkasan Klinis Terstruktur</div>
        <span class="{ai_badge_cls}"><span class="dot"></span>{ai_badge_text}</span>
        """, unsafe_allow_html=True)
    with action_col:
        if st.button("Buat Ulang", help="Panggil Ollama kembali untuk ringkasan baru"):
            with st.spinner("Membuat ulang melalui Ollama..."):
                summary, intel_meta = generate_clinical_intelligence(
                    diagnosis=label,
                    confidence=confidence * 100,
                    class_scores=class_score_dict,
                    predicted_class=predicted_class,
                )
                st.session_state[cache_key] = (summary, intel_meta)
                st.rerun()

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    ai_col1, ai_col2, ai_col3 = st.columns(3, gap="medium")

    with ai_col1:
        st.markdown(f"""
        <div class="ri-intel-card">
          <div class="ri-intel-card-label">Ringkasan Klinis</div>
          <div class="ri-intel-card-body">{summary["summary"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with ai_col2:
        st.markdown(f"""
        <div class="ri-intel-card">
          <div class="ri-intel-card-label">Tindakan yang Disarankan</div>
          <div class="ri-intel-card-body">{summary["action"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with ai_col3:
        st.markdown(f"""
        <div class="ri-intel-card">
          <div class="ri-intel-card-label">Keterlibatan Tenaga Kesehatan</div>
          <div class="ri-intel-card-body">{summary["hcp"]}</div>
          <div class="ri-intel-card-channel">Kanal: <span>{summary["channel"]}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    band_cls = "monitor" if predicted_class <= 1 else ("engage" if predicted_class == 2 else "act")
    st.markdown(f"""
    <div class="ri-severity-band {band_cls}">
      <strong>{tier} | {label}</strong> - {summary['action']}
    </div>
    """, unsafe_allow_html=True)

    if intel_meta.get("source") == "fallback":
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        with st.expander("Aktifkan AI generatif lokal (Ollama)", expanded=False):
            st.markdown(
                "Untuk mengaktifkan pembuatan narasi klinis dinamis via LLM lokal tanpa biaya API:\n\n"
                "1. **Pasang Ollama**: unduh dari [ollama.com](https://ollama.com)\n"
                "2. **Tarik model**: buka terminal dan jalankan:\n"
                "   ```bash\n"
                "   ollama run llama3.2:3b\n"
                "   ```\n"
                "3. **Muat ulang**: klik **Buat Ulang** di atas untuk langsung menguji output LLM."
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # Riwayat sesi
    if len(st.session_state["history"]) > 1:
        st.markdown('<div class="ri-section" style="padding-top:4px;">', unsafe_allow_html=True)
        st.markdown('<div class="ri-section-label">Sesi</div>', unsafe_allow_html=True)
        st.markdown('<div class="ri-section-title">Riwayat Pemindaian</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state["history"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="ri-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="ri-empty anim-fadein-d2">
      <div class="ri-empty-icon"></div>
      <div class="ri-empty-title">Unggah citra fundus retina untuk memulai</div>
      <div class="ri-empty-body">
        Model akan menilai keparahan retinopati diabetik dalam lima kelas
        dan menghasilkan peta perhatian Grad-CAM yang menunjukkan area yang mendasari prediksi.
      </div>
      <div class="ri-empty-tags">
        <span class="ri-tag">PNG</span>
        <span class="ri-tag">JPG</span>
        <span class="ri-tag">JPEG</span>
        <span class="ri-tag">EfficientNetB3</span>
        <span class="ri-tag">Grad-CAM</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="ri-footer">
  <div class="ri-footer-text">
    Dibangun dengan EfficientNetB3 · Dilatih menggunakan Dataset Deteksi Kebutaan APTOS 2019 ·
    Untuk keperluan demonstrasi dan riset · Bukan untuk diagnosis klinis ·
    <a href="https://github.com/priyankaraghunathan15/diabetic-retinopathy-detection" target="_blank">GitHub</a> ·
    <a href="https://www.kaggle.com/competitions/aptos2019-blindness-detection" target="_blank">Dataset APTOS 2019</a>
  </div>
  <span class="ri-footer-badge">Hanya Riset</span>
</div>
""", unsafe_allow_html=True)