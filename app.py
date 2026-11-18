"""Dashboard Streamlit: Perbandingan Fungsi Konsensus Tanpa Bobot untuk Data UMKM Jawa Timur.
Jalankan:  streamlit run app.py
Hasil bawaan dibaca dari folder data/ (dihitung oleh notebook dengan seed 42); mode unggah menjalankan ulang
alur utama (praproses → 3 base → 6 fungsi konsensus → evaluasi) pada berkas pengguna."""
from pathlib import Path
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

import core

st.set_page_config(page_title="Fungsi Konsensus UMKM Jawa Timur", page_icon=":material/hub:", layout="wide")
DATA = Path(__file__).parent / "data"

# ============================================================== tampilan
WARNA_KLASTER = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#8C6D31", "#555555", "#882255", "#44AA99"]
WARNA_FUNGSI = {"AHC average": "#2F5BEA", "AHC complete": "#F28E2B", "Spectral (C)": "#59A14F",
                "K-Medoids (1-C)": "#B07AA1", "MCLA": "#E15759", "K-Modes (label)": "#76B7B2",
                "K-Means": "#9CA3AF", "K-Prototypes": "#6B7280", "K-Medoids": "#374151"}
FONT = "Plus Jakarta Sans, Inter, sans-serif"

st.markdown("""
<style>
@font-face{font-family:'Plus Jakarta Sans';src:url('/app/static/pjs-400.woff2') format('woff2');font-weight:400}
@font-face{font-family:'Plus Jakarta Sans';src:url('/app/static/pjs-500.woff2') format('woff2');font-weight:500}
@font-face{font-family:'Plus Jakarta Sans';src:url('/app/static/pjs-600.woff2') format('woff2');font-weight:600}
@font-face{font-family:'Plus Jakarta Sans';src:url('/app/static/pjs-700.woff2') format('woff2');font-weight:700}
@font-face{font-family:'Plus Jakarta Sans';src:url('/app/static/pjs-800.woff2') format('woff2');font-weight:800}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, p, li, label, h1, h2, h3, h4, h5, h6 {
  font-family:'Plus Jakarta Sans', Inter, sans-serif !important;}
[data-testid="stAppViewContainer"]{background:#F6F8FC}
[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #E6EAF2}
.block-container{padding-top:2.2rem;max-width:1320px}
h1,h2,h3{color:#0F1B3D;letter-spacing:-0.01em}
[data-testid="stElementToolbar"]{display:none}
.hero{background:linear-gradient(120deg,#EEF2FF 0%,#F5F3FF 55%,#ECFDF5 100%);border:1px solid #E3E8F5;border-radius:26px;
  padding:34px 40px 30px;margin-bottom:22px}
.hero .lab{font-size:.78rem;font-weight:700;letter-spacing:.06em;color:#3B5BDB;text-transform:uppercase}
.hero h1{font-size:2.35rem;line-height:1.15;margin:.35rem 0 .6rem;font-weight:800;color:#0F1B3D !important}
.hero h1 span{color:#3B5BDB !important}
.hero p{color:#3F4A66;font-size:1.02rem;max-width:980px;margin:0 0 1rem}
.chip{display:inline-block;background:#fff;border:1px solid #DCE3F3;border-radius:999px;padding:6px 14px;margin:0 8px 8px 0;
  font-size:.88rem;color:#1F2A44}
.chip b{color:#3B5BDB;font-weight:700}
.kartu{background:#fff;border:1px solid #E6EAF2;border-radius:20px;padding:18px 22px;box-shadow:0 2px 10px rgba(15,27,61,.04);height:100%}
.kartu .judul{font-size:.86rem;color:#5B6683;font-weight:500}
.kartu .nilai{font-size:1.85rem;font-weight:800;color:#0F1B3D;line-height:1.2;margin-top:4px}
.kartu .ket{font-size:.82rem;color:#6B7590;margin-top:4px}
.seg{background:#fff;border:1px solid #E6EAF2;border-radius:20px;padding:16px 20px;border-top:6px solid var(--c);height:100%}
.seg .atas{display:flex;justify-content:space-between;align-items:center}
.seg .nama{font-weight:700;font-size:1.05rem;color:#0F1B3D}
.seg .pct{background:var(--bg);color:#0F1B3D;border-radius:999px;padding:2px 10px;font-size:.82rem;font-weight:600}
.seg .n{font-size:1.7rem;font-weight:800;color:#0F1B3D;margin:6px 0 4px}
.seg .n small{font-size:.8rem;font-weight:500;color:#6B7590}
.bar2{margin:7px 0;font-size:.84rem;color:#3F4A66}.bar2 .top{display:flex;justify-content:space-between}.bar2 .top b{color:#0F1B3D}
.bar2 .trk{height:7px;background:#EEF1F7;border-radius:6px;margin-top:4px;overflow:hidden}.bar2 .fil{height:100%;background:var(--c);border-radius:6px}
.bar{display:flex;align-items:center;font-size:.84rem;color:#3F4A66;margin:4px 0}
.bar .lbl{width:46%;white-space:nowrap}.bar .trk{display:block;flex:1;height:7px;background:#EEF1F7;border-radius:6px;margin:0 10px;overflow:hidden}
.bar .fil{display:block;height:100%;background:var(--c);border-radius:6px}
.hero *, .kartu *, .seg *, .langkah *, .catat, .info, .usul, .chip{font-family:'Plus Jakarta Sans', Inter, sans-serif !important}.bar .v{width:54px;text-align:right;font-weight:600;color:#0F1B3D}
.usul{background:#F4F6FB;border-radius:12px;padding:9px 12px;font-size:.84rem;color:#3F4A66;margin-top:10px}
.catat{background:#FFFBEB;border-left:5px solid #F2B705;border-radius:10px;padding:10px 14px;color:#4A3B00;font-size:.9rem;margin:10px 0}
.info{background:#EEF4FF;border-left:5px solid #3B5BDB;border-radius:10px;padding:10px 14px;color:#1F2A44;font-size:.9rem;margin:10px 0}
.langkah{background:#fff;border:1px solid #E6EAF2;border-radius:16px;padding:14px 16px;height:100%}
.langkah b{color:#0F1B3D}.langkah div{font-size:.83rem;color:#5B6683;margin-top:4px}
.pill{display:inline-block;background:#EEF2FF;color:#3B5BDB;border-radius:999px;padding:2px 9px;font-size:.78rem;margin:3px 3px 0 0;font-weight:600}
</style>
""", unsafe_allow_html=True)


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kartu(judul, nilai, ket=""):
    return f'<div class="kartu"><div class="judul">{judul}</div><div class="nilai">{nilai}</div><div class="ket">{ket}</div></div>'


def baris_kartu(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(kartu(*it), unsafe_allow_html=True)


def catatan(teks, jenis="catat"):
    st.markdown(f'<div class="{jenis}">{teks}</div>', unsafe_allow_html=True)


def gaya_plot(fig, h=380):
    fig.update_layout(font_family=FONT, font_color="#1F2A44", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF", legend_title_text="",
                      hoverlabel=dict(font_family=FONT))
    fig.update_xaxes(gridcolor="#EEF1F7", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F7", zeroline=False)
    return fig


def tampil(fig, h=380):
    st.plotly_chart(gaya_plot(fig, h), use_container_width=True, config={"displayModeBar": False})


def tabel(df, **kw):
    st.dataframe(df, hide_index=True, use_container_width=True, **kw)


# ============================================================== data
def _sheet(wb, akhiran):
    return next(v for k, v in wb.items() if k[3:] == akhiran)


@st.cache_data(show_spinner="Memuat hasil bawaan…")
def muat_bawaan():
    raw = pd.read_csv(DATA / "umkm_anonim.csv")
    bersih, _ = core.praproses(raw)
    wb = pd.read_excel(DATA / "hasil_konsensus.xlsx", sheet_name=None)
    miss = _sheet(wb, "Missing value")
    ring = {"baris awal": 2000, "duplikat dihapus": 632, "UMKM unik": len(bersih), "sel hilang": int(miss["jumlah missing"].sum()),
            "hilang per variabel": miss.set_index("Variabel")["jumlah missing"],
            "log imputasi": _sheet(wb, "Log imputasi").rename(columns={"variabel": "Variabel", "metode": "Metode",
                                                                     "nilai imputasi": "Nilai imputasi", "jumlah diimputasi": "Sel diimputasi"})}
    prot = _sheet(wb, "Protokol K")
    return {"id": "bawaan", "nama": "data bawaan (2.000 baris → 1.368 UMKM, seed 42; identitas disamarkan)",
            "bersih": bersih, "ringkas": ring, "labels": pd.read_csv(DATA / "label_semua_K.csv"),
            "EV": _sheet(wb, "Eval utama semua K"), "protokol": prot,
            "lifetime": dict(zip(prot.K, prot["lifetime AHC average"])), "wb": wb}


def sumber():
    return st.session_state.get("unggah") or muat_bawaan()


@st.cache_data(show_spinner="Menghitung jarak Gower dan proyeksi MDS…")
def jarak_mds(kunci, _Xd):
    Dm = core.gower_of(_Xd)
    emb, pos = core.classical_mds(Dm, 10)
    return Dm, emb, pos


def label(S, K, metode):
    return S["labels"][f"K{K}|{metode}"].to_numpy()


def sama_partisi(S, K, metode):
    ref = label(S, K, metode)
    return [c for c in core.CONS6 if c != metode and adjusted_rand_score(ref, label(S, K, c)) == 1.0]


def usulan(r):
    u = []
    if r["NIB_STATUS"] < 50:
        u.append("Pendampingan perizinan usaha (NIB)")
    if r["PEMASARAN_ONLINE"] < 50:
        u.append("Pelatihan pemasaran digital")
    if r["KESULITAN_USAHA"] >= 50:
        u.append("Pendampingan penanganan kendala usaha")
    return u or ["Penguatan skala dan akses pasar"]


def profil_biner(bersih, lab):
    d = bersih.assign(KL=lab)
    t = d.groupby("KL").agg(n=("ID_UMKM", "size"), **{c: (c, lambda s: (s == "YA").mean() * 100) for c in core.BINER})
    t["%"] = t.n / t.n.sum() * 100
    return t


def kartu_segmen(bersih, lab, kolom_per_baris=4):
    t = profil_biner(bersih, lab)
    K = len(t)
    for awal in range(0, K, kolom_per_baris):
        cols = st.columns(kolom_per_baris)
        for j, g in enumerate(range(awal, min(K, awal + kolom_per_baris))):
            r = t.loc[g]
            c = WARNA_KLASTER[g % 10]
            bars = "".join(
                f'<div class="bar2"><div class="top"><span>{core.LABEL_VAR[v]}</span><b>{fmt(r[v], 1)}%</b></div>'
                f'<div class="trk"><div class="fil" style="width:{r[v]:.1f}%"></div></div></div>' for v in ["NIB_STATUS", "PEMASARAN_ONLINE", "KESULITAN_USAHA"])
            cols[j].markdown(
                f'<div class="seg" style="--c:{c};--bg:{c}22"><div class="atas"><span class="nama">Klaster {g + 1}</span>'
                f'<span class="pct">{fmt(r["%"], 1)}%</span></div><div class="n">{fmt(float(r.n), 0)} <small>UMKM</small></div>{bars}'
                f'<div class="usul"><b>Usulan awal:</b> {"; ".join(usulan(r))}</div></div>',
                unsafe_allow_html=True)


# ============================================================== sidebar
def sidebar():
    S = sumber()
    with st.sidebar:
        st.markdown("#### :material/tune: Pengaturan tampilan")
        st.selectbox("Fungsi konsensus untuk profil", core.CONS6, key="fungsi", format_func=lambda c: core.NAMA[c])
        st.slider("Jumlah klaster (K)", 2, 10, key="K")
        st.button("Pakai AHC-average (EAC), K = 4", use_container_width=True,
                  on_click=lambda: st.session_state.update(fungsi="AHC average", K=4))
        st.caption(f":material/database: Sumber hasil: **{'data unggahan' if S['id'] != 'bawaan' else 'data bawaan'}** — {S['nama']}")


st.session_state.setdefault("fungsi", "AHC average")
st.session_state.setdefault("K", 4)


# ============================================================== halaman 1: beranda
def beranda():
    S = sumber()
    wb = muat_bawaan()["wb"]
    tal = _sheet(wb, "Rekap kelompok teratas")
    sim = _sheet(wb, "Sim Friedman")
    semua = tal[tal["Cakupan uji"] == "semua"].set_index("Metode")
    s0 = sim.iloc[0]
    st.markdown("""<div class="hero"><div class="lab">Asistensi Hibah Penelitian BIMA · Universitas Trunodjoyo Madura</div>
    <h1>Fungsi konsensus mana yang paling sesuai untuk <span>data UMKM campuran?</span></h1>
    <p>Enam fungsi konsensus tanpa bobot dibandingkan pada ensemble yang sama (K-Means, K-Prototypes, K-Medoids Gower) untuk
    1.368 UMKM Jawa Timur, lalu diuji ketahanannya dan divalidasi pada 180 dataset semi-sintetis yang kelompoknya diketahui.</p>
    <span class="chip"><b>1.368</b> UMKM</span><span class="chip"><b>9</b> variabel (6 numerik, 3 biner)</span>
    <span class="chip"><b>6</b> fungsi konsensus</span><span class="chip"><b>42</b> uji Friedman–Nemenyi</span>
    <span class="chip"><b>180</b> dataset semi-sintetis</span></div>""", unsafe_allow_html=True)
    a = semua.loc["AHC average"]
    baris_kartu([("Fungsi paling sesuai", "EAC", "AHC-average pada matriks co-association"),
                 ("Kelompok teratas (data asli)", f"{int(a['masuk kelompok teratas'])} dari 42", "uji Friedman–Nemenyi berbasis replikasi"),
                 ("Peringkat rata-rata simulasi", fmt(s0["peringkat AHC avg"], 2), "terbaik dari enam fungsi (180 dataset)"),
                 ("Stabilitas seed, K = 4", "0,9962", "ARI rata-rata 30 seed (435 pasangan)")])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Peringkat rata-rata pada 42 uji data asli (makin kecil makin baik)")
        d = semua.reset_index()
        d["Fungsi"] = d.Metode.map(core.NAMA)
        fig = px.bar(d.sort_values("peringkat rata-rata antar uji"), x="peringkat rata-rata antar uji", y="Fungsi", orientation="h",
                     color="Metode", color_discrete_map=WARNA_FUNGSI, text_auto=".2f")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="peringkat rata-rata")
        fig.update_yaxes(autorange="reversed")
        tampil(fig, 320)
    with c2:
        st.markdown("##### Peringkat rata-rata pada 180 dataset semi-sintetis")
        d = pd.DataFrame({"Metode": core.CONS6, "peringkat": [s0[f"peringkat {core.SINGKAT[c]}"] for c in core.CONS6]})
        d["Fungsi"] = d.Metode.map(core.NAMA)
        fig = px.bar(d.sort_values("peringkat"), x="peringkat", y="Fungsi", orientation="h", color="Metode",
                     color_discrete_map=WARNA_FUNGSI, text_auto=".2f")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="peringkat rata-rata")
        fig.update_yaxes(autorange="reversed")
        tampil(fig, 320)
    st.markdown("### Jawaban singkat")
    c = st.columns(4)
    teks = [("Data asli", "Pada K = 4, AHC-average, AHC-complete, dan MCLA menghasilkan partisi yang sama (SC 0,5612). Data asli saja tidak bisa memisahkan ketiganya."),
            ("Ketahanan", "AHC-average paling stabil terhadap seed dan termasuk dua fungsi paling tahan gangguan data; MCLA dan K-Modes-label paling rapuh."),
            ("Simulasi", "AHC-average unggul jelas bila kelompok dipisahkan variabel biner; bila dipisahkan variabel numerik, tidak ada fungsi yang berbeda nyata."),
            ("Batas", "Bila variabel numerik yang membedakan kelompok, semua konsensus jauh di bawah K-Prototypes sendiri (0,45 vs 0,77): konsensus tanpa bobot mengikuti mayoritas base.")]
    for col, (j, t) in zip(c, teks):
        col.markdown(f'<div class="langkah"><b>{j}</b><div>{t}</div></div>', unsafe_allow_html=True)
    K, f = st.session_state.K, st.session_state.fungsi
    st.markdown(f"### Sekilas segmen: {core.NAMA[f]}, K = {K}")
    kartu_segmen(S["bersih"], label(S, K, f))
    catatan("Segmen terutama dibedakan oleh formalitas (NIB) dan digitalisasi (pemasaran daring), bukan skala usaha. "
            "Usulan pembinaan bersifat awal dan perlu divalidasi pakar. Rincian ada di menu Profil dan Interpretasi.", "info")
    st.markdown("### Alur penelitian")
    langkah = [("1 · Data", "2.000 baris → 1.368 UMKM; 9 variabel; log1p + Min-Max"),
               ("2 · Ensemble", "K-Means, K-Prototypes, K-Medoids (Gower); K = 2–10"),
               ("3 · Enam konsensus", "AHC-average, AHC-complete, Spectral-CA, PAM-CA, MCLA, K-Modes-label"),
               ("4 · Evaluasi", "SC (Gower), DBI dan CHI (MDS 10 dimensi), ANMI; K tetap 3 dan 4"),
               ("5 · Ketahanan", "L1 seed, L2 ensemble, L3 data; Friedman–Nemenyi"),
               ("6 · Simulasi", "180 dataset semi-sintetis dengan kelompok diketahui")]
    for col, (j, t) in zip(st.columns(6), langkah):
        col.markdown(f'<div class="langkah"><b>{j}</b><div>{t}</div></div>', unsafe_allow_html=True)


# ============================================================== halaman 2: data
def data_praproses():
    S = sumber()
    b, r = S["bersih"], S["ringkas"]
    st.title("Data dan Praproses")
    baris_kartu([("Data awal", f"{r['baris awal']:,}".replace(",", "."), "baris"),
                 ("Duplikat persis dihapus", f"{r['duplikat dihapus']:,}".replace(",", "."), "baris"),
                 ("UMKM dianalisis", f"{r['UMKM unik']:,}".replace(",", "."), "setelah pembersihan"),
                 ("Sel hilang diimputasi", str(r["sel hilang"]), "median (numerik), modus (kategorikal)")])
    t1, t2, t3, t4, t5 = st.tabs(["Data", "Kualitas dan imputasi", "Perlakuan variabel", "Versi data (Skenario 1)", "Unggah data baru"])
    with t1:
        jenis = st.multiselect("Saring jenis lapangan usaha", sorted(b.JENIS_LAPANGAN_USAHA.unique()))
        d = b[b.JENIS_LAPANGAN_USAHA.isin(jenis)] if jenis else b
        tabel(d, height=420)
        catatan("Identitas usaha (nama, pemilik, NIB) tidak dimuat dan unduhan baris dinonaktifkan karena data bersifat rahasia.", "info")
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Sel hilang per variabel")
            h = r["hilang per variabel"]
            tabel(pd.DataFrame({"Variabel": h.index, "Sel hilang": h.values}))
        with c2:
            st.markdown("##### Log imputasi")
            tabel(r["log imputasi"].astype(str))
        st.markdown("##### Sebaran variabel biner (% YA)")
        p = pd.DataFrame({"Variabel": [core.LABEL_VAR[c] for c in core.BINER], "% YA": [(b[c] == "YA").mean() * 100 for c in core.BINER]})
        fig = px.bar(p, x="% YA", y="Variabel", orientation="h", text_auto=".1f", color_discrete_sequence=["#3B5BDB"])
        fig.update_layout(xaxis_range=[0, 100], yaxis_title="")
        tampil(fig, 240)
        st.markdown("##### Efek log1p pada variabel keuangan")
        v = st.selectbox("Variabel", core.LOG1P, format_func=lambda c: core.LABEL_VAR[c])
        c1, c2 = st.columns(2)
        with c1:
            tampil(px.histogram(b, x=v, nbins=60, title="Sebelum log1p", color_discrete_sequence=["#94A3B8"]), 300)
        with c2:
            tampil(px.histogram(np.log1p(b[v].astype(float)), nbins=60, title="Sesudah log1p", color_discrete_sequence=["#3B5BDB"]), 300)
    with t3:
        tabel(pd.DataFrame({
            "Variabel": [core.LABEL_VAR[c] for c in core.VARIABEL],
            "Jenis": ["Numerik"] * 6 + ["Biner"] * 3,
            "Perlakuan": ["Min-Max", "Min-Max", "log1p, lalu Min-Max", "log1p, lalu Min-Max", "log1p, lalu Min-Max", "Min-Max",
                          "YA = 1, TIDAK = 0", "YA = 1, TIDAK = 0", "YA = 1 (nomor NIB sah), TIDAK = 0"],
            "Median [Q1–Q3] atau % YA": [f"{fmt(b[c].median(), 0)} [{fmt(b[c].quantile(.25), 0)}–{fmt(b[c].quantile(.75), 0)}]" for c in core.NUMERIK]
                                        + [f"{fmt((b[c] == 'YA').mean() * 100, 1)}%" for c in core.BINER]}))
        catatan("Jarak Gower menggabungkan selisih absolut terskala (numerik) dan ketidakcocokan (biner) dengan bobot sama. "
                "SC dihitung langsung pada jarak Gower; DBI dan CHI pada proyeksi MDS klasik 10 dimensi dari jarak yang sama.", "info")
    with t4:
        st.markdown("SC rata-rata partisi dasar (K = 2–10) pada dua versi data, dihitung pada jarak Gower yang sama (hasil data bawaan).")
        tabel(pd.DataFrame({"Algoritma": ["K-Means", "K-Prototypes", "K-Medoids", "Rata-rata seluruh partisi"],
                            "Dengan normalisasi": ["0,4865", "0,2092", "0,4062", "0,3673"],
                            "Tanpa normalisasi": ["0,1346", "0,1173", "0,4062", "0,2194"],
                            "Selisih": ["0,3519", "0,0919", "0,0000", "0,1479"]}))
        catatan("Versi dengan normalisasi dipilih dan dipakai pada semua skenario berikutnya.", "info")
    with t5:
        st.markdown("Unggah data UMKM dalam **format mentah dinas** (kolom TAHUN_MULAI_USAHA, JUMLAH_TENAGA_KERJA, OMSET, BIAYA_PRODUKSI, "
                    "GAJI_KARYAWAN, SERTIFIKASI, PEMASARAN_ONLINE, KESULITAN_USAHA, NIB) atau **format bersih** seperti data bawaan "
                    "(LAMA_USAHA, …, SERTIFIKASI_JUMLAH, NIB_STATUS). Alur utama (praproses, 3 base, 6 fungsi konsensus, evaluasi K = 2–10) "
                    "dijalankan ulang; uji ketahanan dan simulasi tetap memakai hasil data bawaan karena memerlukan ribuan pengulangan.")
        if core.pymetis is None:
            catatan("Paket pymetis tidak terpasang (umumnya di Windows), sehingga MCLA pada data unggahan memakai partisi spektral "
                    "sebagai pengganti METIS dan hasilnya bisa sedikit berbeda dari notebook. Hasil data bawaan tidak terpengaruh.", "info")
        f = st.file_uploader("Berkas .xlsx atau .csv", type=["xlsx", "csv"])
        c1, c2 = st.columns([1, 1])
        if f is not None and c1.button("Jalankan seluruh alur", type="primary", use_container_width=True):
            isi = f.getvalue()
            raw = pd.read_csv(io.BytesIO(isi)) if f.name.endswith(".csv") else pd.read_excel(io.BytesIO(isi), dtype=object)
            try:
                bersih, ring = core.praproses(raw)
            except ValueError as e:
                st.error(str(e))
                return
            if len(bersih) > 4000:
                st.error("Mode unggah dibatasi 4.000 UMKM agar matriks jarak muat di memori.")
                return
            if len(bersih) < 30:
                st.error("Data terlalu sedikit (minimal 30 UMKM).")
                return
            bar = st.progress(0.0, "Menyiapkan…")
            res = core.jalankan_semua(bersih, progres=lambda p, t: bar.progress(p, t))
            st.session_state.unggah = {"id": hashlib.md5(isi).hexdigest()[:10], "nama": f"{f.name} ({len(bersih):,} UMKM)".replace(",", "."),
                                       "bersih": bersih, "ringkas": ring, "labels": res["labels"], "EV": res["EV"],
                                       "protokol": core.protokol_k(res["EV"], res["lifetime"]), "lifetime": res["lifetime"]}
            st.success("Selesai. Semua menu kini memakai data unggahan.")
            st.rerun()
        if st.session_state.get("unggah") and c2.button("Kembali ke data bawaan", use_container_width=True):
            del st.session_state["unggah"]
            st.rerun()


# ============================================================== halaman 3: perbandingan
def perbandingan():
    S = sumber()
    EV = S["EV"]
    st.title("Perbandingan Fungsi Konsensus")
    t1, t2, t3, t4, t5 = st.tabs(["Ensemble dan fungsi", "K = 3 dan K = 4", "Semua K (2–10)", "Protokol K", "Kepekaan geometri"])
    with t1:
        tabel(pd.DataFrame({
            "Keluarga": ["Co-association"] * 4 + ["Graf", "Label"],
            "Fungsi": [core.NAMA[c] for c in core.CONS6],
            "Masukan": ["1 − C", "1 − C", "C", "1 − C", "Indikator klaster", "Matriks label n × 3"],
            "Aturan": ["Average linkage, dipotong pada K", "Complete linkage, dipotong pada K", "Spectral clustering, C sebagai afinitas",
                       "K-Medoids (PAM)", "Meta-graf klaster (Jaccard), METIS", "K-Modes pada label base"]}))
        catatan("C = matriks co-association, yaitu proporsi partisi dasar yang menempatkan dua UMKM pada klaster yang sama. "
                "Ketiga partisi dasar berbobot sama untuk semua fungsi; dengan tiga base, C hanya bernilai 0, 1/3, 2/3, atau 1.", "info")
        st.markdown("##### Konfigurasi base clustering")
        tabel(pd.DataFrame({"Algoritma": ["K-Means", "K-Prototypes", "K-Medoids (PAM)"],
                            "Masukan": ["Sembilan fitur terskala (biner 0/1)", "Enam numerik + tiga kategorikal", "Matriks jarak Gower"],
                            "Parameter": ["k-means++, 10 inisialisasi, seed 42", "Inisialisasi Huang, 5 inisialisasi, seed 42", "seed 42"]}))
    with t2:
        K = st.segmented_control("K", [3, 4], default=4, key="k34")
        e = EV[EV.K == K].copy()
        e["Fungsi"] = e.Metode.map(core.NAMA)
        e = e.set_index("Metode").loc[core.CONS6 + core.B3].reset_index()
        tampilkan = e[["Fungsi", "Jenis", "SC", "DBI", "CHI", "ANMI", "klaster terkecil (n)"]].rename(columns={"klaster terkecil (n)": "Min n"})
        kons = tampilkan[tampilkan.Jenis == "Konsensus"]
        best = {"SC": kons.SC.max(), "DBI": kons.DBI.min(), "CHI": kons.CHI.max(), "ANMI": kons.ANMI.max()}

        def tebal(row):
            return ["font-weight:700;background:#EEF2FF" if (row.Jenis == "Konsensus" and k in best and np.isclose(row[k], best[k])) else ""
                    for k in row.index]
        st.dataframe(tampilkan.style.apply(tebal, axis=1).format({"SC": "{:.4f}", "DBI": "{:.4f}", "CHI": "{:.2f}", "ANMI": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        grup, sudah = [], set()
        for c in core.CONS6:
            if c in sudah:
                continue
            s = [c] + sama_partisi(S, K, c)
            sudah |= set(s)
            if len(s) > 1:
                grup.append(", ".join(core.NAMA[x] for x in s))
        if grup:
            catatan(f"Pada K = {K}, fungsi berikut menghasilkan partisi yang <b>identik</b>: {' | '.join(grup)}. "
                    "Data asli saja tidak dapat memisahkan mereka; pembedanya ada di menu Uji Ketahanan dan Simulasi.")
        catatan("Sorotan biru: nilai terbaik per kolom di antara enam fungsi konsensus. SC dan ANMI makin besar makin baik, DBI makin kecil, CHI makin besar.", "info")
    with t3:
        idx = st.segmented_control("Indeks", ["SC", "DBI", "CHI", "ANMI", "SC Euclid"], default="SC")
        base = st.toggle("Tampilkan partisi dasar", value=False)
        d = EV if base else EV[EV.Jenis == "Konsensus"]
        d = d.assign(Fungsi=d.Metode.map(core.NAMA))
        fig = px.line(d, x="K", y=idx, color="Metode", markers=True, color_discrete_map=WARNA_FUNGSI,
                      category_orders={"Metode": core.CONS6 + core.B3}, hover_data={"Fungsi": True, "Metode": False})
        fig.for_each_trace(lambda tr: tr.update(name=core.NAMA[tr.name], line_dash="dot" if tr.name in core.B3 else "solid"))
        fig.add_vline(x=4, line_dash="dot", line_color="#94A3B8")
        tampil(fig, 420)
    with t4:
        P = S["protokol"].copy()
        st.markdown("Skor komposit tiap fungsi = jumlah peringkat SC, DBI, dan CHI antar-K (makin kecil makin baik).")
        fig = px.bar(P, x="K", y="Jumlah (6 fungsi)", text_auto=True, color_discrete_sequence=["#3B5BDB"])
        tampil(fig, 300)
        tabel(P.rename(columns={c: core.NAMA[c] for c in core.CONS6}).round(4))
        catatan("Fungsi dibandingkan pada K tetap = 3 dan 4: jumlah skor komposit terendah pada K = 3, K optimal setiap fungsi hanya 3 atau 4, "
                "dan lifetime dendrogram AHC-average paling panjang pada K = 4. Profil segmen memakai K = 4.", "info")
    with t5:
        rows = []
        for K in core.KS:
            e = EV[(EV.K == K) & (EV.Jenis == "Konsensus")].set_index("Metode").loc[core.CONS6]
            rho = spearmanr(e.SC, e["SC Euclid"]).statistic if e.SC.nunique() > 1 else np.nan
            rows.append({"K": K, "Spearman ρ (SC Gower vs SC Euclid)": rho, "Terbaik SC Gower": core.NAMA[e.SC.idxmax()],
                         "Terbaik SC Euclid": core.NAMA[e["SC Euclid"].idxmax()]})
        tabel(pd.DataFrame(rows).round(3))
        catatan("Urutan fungsi hampir tidak bergantung pada geometri evaluasi (Gower vs Euclid). Pada K = 2 semua fungsi menghasilkan partisi yang sama.", "info")


# ============================================================== halaman 4: ketahanan
def ketahanan():
    wb = muat_bawaan()["wb"]
    st.title("Uji Ketahanan")
    if st.session_state.get("unggah"):
        catatan("Menu ini memakai hasil <b>data bawaan</b>: tiga lapis ketahanan memerlukan ratusan pengulangan alur penuh.")
    l1, l2, l3 = _sheet(wb, "Lapis1 stabilitas seed"), _sheet(wb, "Lapis2 stabilitas"), _sheet(wb, "Lapis3 data")
    fr, tal, rep = _sheet(wb, "Friedman per kriteria"), _sheet(wb, "Rekap kelompok teratas"), _sheet(wb, "Nilai per replikasi")
    t0, t1, t2, t3, t4 = st.tabs(["Ringkasan", "Lapis 1: seed", "Lapis 2: ensemble", "Lapis 3: data", "Friedman–Nemenyi"])
    with t0:
        K = st.segmented_control("K", [3, 4], default=4, key="kh")
        kol = {"L1 seed": lambda c: l1[(l1.K == K) & (l1.Metode == c)]["ARI rata-rata (stabilitas)"].iloc[0]}
        for sk, lb in [("Seed saja", "L2a seed"), ("Subsampel objek", "L2b subsampel"), ("Subruang variabel", "L2c subruang"),
                       ("Subruang + subsampel", "L2d keduanya")]:
            kol[lb] = lambda c, sk=sk: l2[(l2.K == K) & (l2["Skema perturbasi"] == sk) & (l2.Metode == c)]["ARI rata-rata (stabilitas)"].iloc[0]
        for g, lb in [("Subsampel 80%", "L3a subsampel 80%"), ("Derau biner 5%", "L3b derau biner")]:
            kol[lb] = lambda c, g=g: l3[(l3.K == K) & (l3["Gangguan data"] == g) & (l3.Metode == c)]["ARI rata-rata vs data penuh"].iloc[0]
        M = np.array([[f(c) for f in kol.values()] for c in core.CONS6])
        fig = go.Figure(go.Heatmap(z=M, x=list(kol), y=[core.NAMA[c] for c in core.CONS6], zmin=0.3, zmax=1,
                                   colorscale=[[0, "#F4F8FD"], [0.4, "#9EC5F4"], [0.75, "#2A78D6"], [1, "#104281"]],
                                   text=np.round(M, 2), texttemplate="%{text:.2f}", hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
                                   colorbar=dict(title="ARI")))
        fig.update_yaxes(autorange="reversed")
        tampil(fig, 400)
        catatan("L1 dan L2: rata-rata ARI antar-pasangan replikasi (stabilitas). L3: rata-rata ARI terhadap hasil data penuh. "
                "L2c dan L2d adalah uji tekanan (ensemble dari subset variabel).", "info")
    with t1:
        d = l1.assign(Fungsi=l1.Metode.map(core.NAMA))
        tabel(d[["K", "Fungsi", "ARI rata-rata (stabilitas)", "ARI terkecil", "ARI terbesar", "ARI rata-rata vs hasil utama", "SC rata-rata", "ANMI rata-rata"]].round(4))
        catatan("Seluruh alur (3 base + 6 fungsi) diulang dengan seed 0–29; stabilitas = rata-rata ARI 435 pasangan seed.", "info")
    with t2:
        d = l2.assign(Fungsi=l2.Metode.map(core.NAMA))
        sk = st.segmented_control("Skema", list(d["Skema perturbasi"].unique()), default="Seed saja")
        tabel(d[d["Skema perturbasi"] == sk][["K", "Peran", "Fungsi", "ARI rata-rata (stabilitas)", "ARI terkecil",
                                               "ARI rata-rata vs hasil utama 3 base", "SC rata-rata", "ANMI rata-rata"]].round(4))
        catatan("Ensemble 33 partisi (3 model × 11 replikasi) dengan seed berbeda, subsampel objek 80%, subruang variabel, atau keduanya; 30 replikasi.", "info")
    with t3:
        d = l3[l3.Metode.isin(core.CONS6 + ["(base) K-Means", "(base) K-Prototypes", "(base) K-Medoids"])]
        g = st.segmented_control("Gangguan", list(d["Gangguan data"].unique()), default="Subsampel 80%")
        dd = d[d["Gangguan data"] == g].copy()
        dd["Fungsi"] = dd.Metode.map(lambda m: core.NAMA.get(m, m))
        fig = px.bar(dd, x="Fungsi", y="ARI rata-rata vs data penuh", color="K", barmode="group", text_auto=".3f",
                     color_discrete_sequence=["#9EC5F4", "#2A78D6"])
        fig.update_layout(xaxis_title="", yaxis_range=[0, 1.05])
        tampil(fig, 380)
        tabel(dd[["K", "Fungsi", "ARI rata-rata vs data penuh", "simpangan baku", "ARI terkecil", "replikasi dengan ARI ≥ 0,90"]].round(4))
    with t4:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("##### Rekap kelompok teratas")
            cak = st.segmented_control("Cakupan", ["semua", "utama", "uji tekanan"], default="semua")
            tt = tal[tal["Cakupan uji"] == cak].assign(Fungsi=lambda x: x.Metode.map(core.NAMA))
            tabel(tt[["Fungsi", "masuk kelompok teratas", "uji signifikan", "peringkat rata-rata antar uji"]].round(2))
        with c2:
            st.markdown("##### Diagram peringkat satu uji")
            fr["Uji"] = fr.Lapis + " · " + fr.Kondisi + " · K = " + fr.K.astype(str) + " · " + fr.Kriteria
            u = st.selectbox("Pilih uji", fr.Uji.tolist())
            r = fr[fr.Uji == u].iloc[0]
            rk = pd.DataFrame({"Fungsi": [core.NAMA[c] for c in core.CONS6], "Metode": core.CONS6,
                               "Peringkat": [r[f"peringkat {core.SINGKAT[c]}"] for c in core.CONS6]}).sort_values("Peringkat")
            best, cd = rk.Peringkat.min(), r["CD Nemenyi (α = 0,05)"]
            fig = px.scatter(rk, x="Peringkat", y="Fungsi", color="Metode", color_discrete_map=WARNA_FUNGSI)
            fig.update_traces(marker_size=16)
            fig.add_vrect(x0=best, x1=best + cd, fillcolor="#3B5BDB", opacity=.10, line_width=0)
            fig.update_layout(showlegend=False, xaxis_range=[0.8, 6.2], yaxis_title="", xaxis_title="peringkat rata-rata (1 = terbaik)")
            fig.update_yaxes(autorange="reversed")
            tampil(fig, 330)
            catatan(f"p = {r['p-value']:.2e}; CD Nemenyi = {fmt(cd, 3)} (N = {int(r['Blok (N replikasi)'])}). Area biru = kelompok teratas, "
                    f"yaitu fungsi yang tidak berbeda nyata dari yang terbaik: <b>{r['Kelompok teratas (tidak berbeda nyata dari terbaik)']}</b>.", "info")
            nilai = rep[(rep.Lapis == r.Lapis) & (rep.Kondisi == r.Kondisi) & (rep.K == r.K) & (rep.Kriteria == r.Kriteria)]
            with st.expander("Nilai per replikasi"):
                tabel(nilai.drop(columns=["Lapis", "Kondisi", "K", "Kriteria"]).round(4))


# ============================================================== halaman 5: simulasi
def simulasi():
    wb = muat_bawaan()["wb"]
    st.title("Simulasi Semi-Sintetis")
    ds, per, fr, vb = _sheet(wb, "Sim desain"), _sheet(wb, "Sim hasil per dataset"), _sheet(wb, "Sim Friedman"), _sheet(wb, "Sim konsensus vs base")
    t1, t2, t3, t4, t5 = st.tabs(["Rancangan", "Hasil per kondisi", "Friedman–Nemenyi", "Konsensus vs base", "Unduh dataset"])
    with t1:
        st.markdown("""Data asli tidak punya label kelompok, sehingga tidak bisa diketahui fungsi mana yang **benar**. Simulasi membuat 180 dataset
        berukuran sama (1.368 UMKM, 9 variabel) yang meniru sebaran tiap variabel data asli, tetapi kelompoknya diketahui. Kelompok yang
        ditemukan tiap fungsi lalu dibandingkan dengan kelompok sebenarnya memakai ARI.""")
        baris_kartu([("Sumber pemisah", "3", "campuran, terutama biner, terutama numerik"), ("Tingkat pemisahan", "3", "rendah, sedang, tinggi"),
                     ("K benar", "3 dan 4", "proporsi 66/20/14% dan 60/20/13/7%"), ("Replikasi", "10", "180 dataset total")])
        st.write("")
        tabel(ds.rename(columns={"a_num": "a numerik", "a_bin": "a biner"}))
        catatan("Generator Gaussian copula: laten L = a · Z[kelompok] + E, lalu tiap kolom dipetakan ke kuantil empiris variabel asli (numerik) "
                "atau dipotong pada proporsi asli (biner). Label kelompok hanya dipakai untuk menilai hasil, tidak pernah diberikan ke algoritma.", "info")
    with t2:
        Kb = st.segmented_control("K benar", ["semua", 3, 4], default="semua")
        d = per if Kb == "semua" else per[per["K benar"] == Kb]
        m = d.groupby(["Sumber pemisah", "Pemisahan"])[[f"ARI {c}" for c in core.CONS6] + ["ARI base K-Means", "ARI base K-Prototypes", "ARI base K-Medoids"]].mean().reset_index()
        L = m.melt(id_vars=["Sumber pemisah", "Pemisahan"], var_name="Metode", value_name="ARI rata-rata")
        L["Metode"] = L.Metode.str.replace("ARI base ", "").str.replace("ARI ", "")
        L["Fungsi"] = L.Metode.map(core.NAMA)
        L["Pemisahan"] = pd.Categorical(L.Pemisahan, ["rendah", "sedang", "tinggi"])
        fig = px.bar(L.sort_values("Pemisahan"), x="Pemisahan", y="ARI rata-rata", color="Metode", barmode="group",
                     facet_col="Sumber pemisah", color_discrete_map=WARNA_FUNGSI,
                     category_orders={"Metode": core.CONS6 + core.B3, "Sumber pemisah": ["campuran", "terutama biner", "terutama numerik"]})
        fig.for_each_trace(lambda tr: tr.update(name=core.NAMA[tr.name]))
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        tampil(fig, 430)
        catatan("Batang abu-abu = partisi dasar. Bila kelompok dipisahkan variabel numerik, K-Prototypes sendirian jauh di atas semua konsensus: "
                "dua dari tiga base bekerja pada jarak yang didominasi variabel biner, dan konsensus tanpa bobot mengikuti mayoritas.", "info")
    with t3:
        f = fr[fr.Kriteria == "ARI"].copy()
        tabel(f[["Kelompok dataset", "Blok (N dataset)", "p-value", "CD Nemenyi (α = 0,05)", "Peringkat terbaik",
                 "Kelompok teratas (tidak berbeda nyata dari terbaik)"] + [f"peringkat {core.SINGKAT[c]}" for c in core.CONS6]]
              .rename(columns={f"peringkat {core.SINGKAT[c]}": core.NAMA[c] for c in core.CONS6}).round(3))
        catatan("Angka per fungsi = peringkat rata-rata (1 = terbaik) dari ARI terhadap kelompok sebenarnya.", "info")
    with t4:
        d = vb.assign(Fungsi=vb.Metode.map(core.NAMA))
        tabel(d[["Sumber pemisah", "Fungsi", "ARI rata-rata konsensus", "ARI rata-rata base terbaik per dataset",
                 "dataset dengan konsensus ≥ base terbaik − 0,01", "jumlah dataset"]].round(3))
    with t5:
        st.markdown("Ke-180 dataset (CSV dalam ZIP) beserta `daftar_dataset.csv` dan `README.txt`. Tiap file memuat 9 variabel dalam skala asli "
                    "dan kolom `KELOMPOK_BENAR`. Data ini sintetis; tidak ada baris yang berasal langsung dari UMKM asli.")
        st.download_button("Unduh Dataset_Semi_Sintetis_180.zip", (DATA / "Dataset_Semi_Sintetis_180.zip").read_bytes(),
                           file_name="Dataset_Semi_Sintetis_180.zip", mime="application/zip", type="primary")


# ============================================================== halaman 6: profil
def profil():
    S = sumber()
    b = S["bersih"]
    K, f = st.session_state.K, st.session_state.fungsi
    lab = label(S, K, f)
    Xd = core.normalisasi(b)
    st.title("Profil dan Interpretasi")
    st.caption(f"Partisi: **{core.NAMA[f]}**, K = {K} (ubah di panel kiri).")
    sama = sama_partisi(S, K, f)
    if sama:
        catatan(f"Partisi ini identik dengan {', '.join(core.NAMA[c] for c in sama)}; profilnya tidak bergantung pada pilihan di antara fungsi tersebut.", "info")
    t1, t2, t3, t4 = st.tabs(["Kartu profil", "Sebaran (MDS)", "Peran variabel", "Jenis usaha"])
    with t1:
        kartu_segmen(b, lab)
        st.markdown("##### Median variabel numerik per klaster")
        d = b.assign(Klaster=lab + 1).groupby("Klaster")[core.NUMERIK].median()
        d = d.apply(lambda col: col.map(lambda v: fmt(float(v), 0))).reset_index()
        tabel(d.rename(columns=core.LABEL_VAR))
        catatan("Usulan awal disusun dari pola tiga variabel biner (NIB < 50% → perizinan; pemasaran daring < 50% → pemasaran digital; "
                "kesulitan ≥ 50% → penanganan kendala) dan perlu divalidasi pakar pembinaan UMKM.")
    with t2:
        Dm, emb, pos = jarak_mds(S["id"], Xd)
        d = pd.DataFrame({"Dimensi 1": emb[:, 0], "Dimensi 2": emb[:, 1], "Klaster": [f"Klaster {g + 1}" for g in lab]})
        fig = px.scatter(d, x="Dimensi 1", y="Dimensi 2", color="Klaster", opacity=.7,
                         color_discrete_sequence=WARNA_KLASTER, category_orders={"Klaster": [f"Klaster {g + 1}" for g in range(K)]})
        fig.update_traces(marker_size=7)
        tampil(fig, 480)
        catatan(f"Proyeksi MDS klasik dari jarak Gower; dua dimensi menjelaskan {fmt(pos[:2].sum() / pos.sum() * 100, 1)}% dari total variansi positif. "
                f"SC partisi ini {fmt(S['EV'].query('K == @K and Metode == @f').SC.iloc[0])}.", "info")
    with t3:
        eta = core.eta_kuadrat(Xd, lab)
        por = core.porsi_variansi(Xd)
        d = pd.DataFrame({"Variabel": [core.LABEL_VAR[c] for c in core.VARIABEL], "Jenis": ["Numerik"] * 6 + ["Biner"] * 3,
                          "Porsi variansi pada jarak (%)": por.values, "η² terhadap klaster": eta.values}).sort_values("η² terhadap klaster")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(d, x="η² terhadap klaster", y="Variabel", orientation="h", color="Jenis", text_auto=".3f",
                         color_discrete_map={"Biner": "#3B5BDB", "Numerik": "#A5B4CF"}, title="η²: variansi variabel yang dijelaskan klaster")
            fig.update_layout(xaxis_range=[0, 1], yaxis_title="")
            tampil(fig, 400)
        with c2:
            fig = px.bar(d.sort_values("Porsi variansi pada jarak (%)"), x="Porsi variansi pada jarak (%)", y="Variabel", orientation="h",
                         color="Jenis", text_auto=".1f", color_discrete_map={"Biner": "#3B5BDB", "Numerik": "#A5B4CF"},
                         title="Porsi variansi tiap variabel pada fitur terskala")
            fig.update_layout(yaxis_title="")
            tampil(fig, 400)
        catatan(f"Tiga variabel biner menyumbang {fmt(por[core.BINER].sum(), 1)}% variansi; klaster karenanya menggambarkan formalitas (NIB) "
                "dan digitalisasi (pemasaran daring), bukan skala usaha.")
    with t4:
        d = pd.crosstab(b.JENIS_LAPANGAN_USAHA, lab + 1, normalize="columns") * 100
        L = d.reset_index().melt(id_vars="JENIS_LAPANGAN_USAHA", var_name="Klaster", value_name="%")
        L["Klaster"] = "Klaster " + L.Klaster.astype(str)
        fig = px.bar(L, x="Klaster", y="%", color="JENIS_LAPANGAN_USAHA", text_auto=".0f",
                     color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(legend=dict(orientation="h", y=-0.25), yaxis_title="% UMKM dalam klaster")
        tampil(fig, 520)


# ============================================================== halaman 7: cek UMKM baru
@st.cache_resource(show_spinner="Melatih ulang base clustering untuk penempatan…")
def penempat(kunci, _bersih, K, _lab):
    p = core.Penempat(_bersih, K, _lab)
    return p, p.uji_penempatan_ulang()


def cek_umkm():
    S = sumber()
    K, f = st.session_state.K, st.session_state.fungsi
    lab = label(S, K, f)
    st.title("Cek UMKM Baru")
    st.caption(f"UMKM baru ditempatkan ke partisi **{core.NAMA[f]}**, K = {K} (ubah di panel kiri).")
    with st.form("baru"):
        c1, c2, c3 = st.columns(3)
        tahun = c1.number_input("Tahun mulai usaha", 1950, core.TAHUN_REFERENSI, 2018)
        tk = c1.number_input("Jumlah tenaga kerja (orang)", 0, 10000, 3)
        sert = c1.number_input("Jumlah sertifikasi (0–5)", 0, 5, 1)
        omset = c2.number_input("Omset (Rp)", 0, value=60_000_000, step=1_000_000, format="%d")
        biaya = c2.number_input("Biaya produksi (Rp)", 0, value=20_000_000, step=1_000_000, format="%d")
        gaji = c2.number_input("Gaji karyawan (Rp)", 0, value=10_000_000, step=1_000_000, format="%d")
        online = c3.radio("Pemasaran daring", ["YA", "TIDAK"], horizontal=True)
        sulit = c3.radio("Kesulitan usaha", ["YA", "TIDAK"], horizontal=True)
        nib = c3.radio("Memiliki NIB", ["YA", "TIDAK"], index=1, horizontal=True)
        kirim = st.form_submit_button("Tempatkan ke klaster", type="primary")
    p, akurasi = penempat(f"{S['id']}|{K}|{f}", S["bersih"], K, lab)
    if kirim:
        baru = pd.DataFrame([{"ID_UMKM": "BARU", "JENIS_LAPANGAN_USAHA": "-", "LAMA_USAHA": core.TAHUN_REFERENSI - tahun,
                              "JUMLAH_TENAGA_KERJA": tk, "OMSET": omset, "BIAYA_PRODUKSI": biaya, "GAJI_KARYAWAN": gaji,
                              "SERTIFIKASI_JUMLAH": sert, "PEMASARAN_ONLINE": online, "KESULITAN_USAHA": sulit, "NIB_STATUS": nib}])
        h, _ = p.tempatkan(baru)
        h = h[0]
        g = h["klaster"]
        t = profil_biner(S["bersih"], lab).loc[g]
        c = WARNA_KLASTER[g % 10]
        c1, c2 = st.columns([1, 1.3])
        c1.markdown(f'<div class="seg" style="--c:{c};--bg:{c}22"><div class="atas"><span class="nama">Masuk Klaster {g + 1}</span>'
                    f'<span class="pct">{int(t.n)} UMKM</span></div><div class="usul"><b>Usulan awal:</b> {"; ".join(usulan(t))}</div>'
                    f'<div class="usul">Label base: K-Means {h["base"]["K-Means"] + 1} · K-Prototypes {h["base"]["K-Prototypes"] + 1} · '
                    f'K-Medoids {h["base"]["K-Medoids"] + 1}</div></div>', unsafe_allow_html=True)
        d = pd.DataFrame({"Klaster": [f"Klaster {i + 1}" for i in range(K)], "Rata-rata co-association": h["skor"], "Jarak Gower rata-rata": h["jarak"]})
        fig = px.bar(d, x="Klaster", y="Rata-rata co-association", text_auto=".3f", color="Klaster", color_discrete_sequence=WARNA_KLASTER)
        fig.update_layout(showlegend=False, yaxis_range=[0, 1.05])
        with c2:
            tampil(fig, 300)
    catatan(f"Aturan penempatan mengikuti EAC: ketiga base memberi label untuk UMKM baru, lalu UMKM baru masuk ke klaster dengan rata-rata "
            f"co-association tertinggi (seri dipecah dengan jarak Gower rata-rata). Uji penempatan ulang: {fmt(akurasi * 100, 2)}% UMKM lama kembali "
            "ke klasternya sendiri bila ditempatkan tanpa dirinya.", "info")


# ============================================================== navigasi
nav = st.navigation([
    st.Page(beranda, title="Beranda", icon=":material/home:", url_path="beranda", default=True),
    st.Page(data_praproses, title="Data dan Praproses", icon=":material/table_chart:", url_path="data"),
    st.Page(perbandingan, title="Perbandingan Fungsi Konsensus", icon=":material/hub:", url_path="perbandingan"),
    st.Page(ketahanan, title="Uji Ketahanan", icon=":material/shield:", url_path="ketahanan"),
    st.Page(simulasi, title="Simulasi Semi-Sintetis", icon=":material/science:", url_path="simulasi"),
    st.Page(profil, title="Profil dan Interpretasi", icon=":material/groups:", url_path="profil"),
    st.Page(cek_umkm, title="Cek UMKM Baru", icon=":material/person_search:", url_path="cek-umkm"),
])
st.logo(str(Path(__file__).parent / "static" / "logo.png"), size="large")
sidebar()
nav.run()