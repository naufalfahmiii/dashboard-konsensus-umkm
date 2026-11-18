"""Inti perhitungan dashboard: praproses, base clustering, enam fungsi konsensus tanpa bobot, evaluasi,
peran variabel, dan penempatan UMKM baru. Kode mengikuti notebook
Notebook_Perbandingan_Fungsi_Konsensus_UMKM.ipynb (seed 42) sehingga hasilnya sama."""
import warnings
import numpy as np
import pandas as pd
import gower
import kmedoids
try:                                   # pymetis tidak menyediakan paket siap pakai untuk Windows
    import pymetis
except ImportError:
    pymetis = None
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             normalized_mutual_info_score)
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
KS = list(range(2, 11))
TAHUN_REFERENSI = 2026
NUMERIK = ["LAMA_USAHA", "JUMLAH_TENAGA_KERJA", "OMSET", "BIAYA_PRODUKSI", "GAJI_KARYAWAN", "SERTIFIKASI_JUMLAH"]
BINER = ["PEMASARAN_ONLINE", "KESULITAN_USAHA", "NIB_STATUS"]
LOG1P = ["OMSET", "BIAYA_PRODUKSI", "GAJI_KARYAWAN"]
VARIABEL = NUMERIK + BINER
B3 = ["K-Means", "K-Prototypes", "K-Medoids"]
CONS6 = ["AHC average", "AHC complete", "Spectral (C)", "K-Medoids (1-C)", "MCLA", "K-Modes (label)"]
NAMA = {"AHC average": "AHC-average (EAC)", "AHC complete": "AHC-complete", "Spectral (C)": "Spectral-CA",
        "K-Medoids (1-C)": "PAM-CA", "MCLA": "MCLA", "K-Modes (label)": "K-Modes-label",
        "K-Means": "K-Means", "K-Prototypes": "K-Prototypes", "K-Medoids": "K-Medoids (Gower)"}
SINGKAT = {"AHC average": "AHC avg", "AHC complete": "AHC comp", "Spectral (C)": "Spectral C",
           "K-Medoids (1-C)": "PAM 1−C", "MCLA": "MCLA", "K-Modes (label)": "K-Modes label"}
LABEL_VAR = {"LAMA_USAHA": "Lama usaha (tahun)", "JUMLAH_TENAGA_KERJA": "Jumlah tenaga kerja",
             "OMSET": "Omset (Rp)", "BIAYA_PRODUKSI": "Biaya produksi (Rp)", "GAJI_KARYAWAN": "Gaji karyawan (Rp)",
             "SERTIFIKASI_JUMLAH": "Sertifikasi (jumlah)", "PEMASARAN_ONLINE": "Pemasaran daring",
             "KESULITAN_USAHA": "Kesulitan usaha", "NIB_STATUS": "NIB"}


# ------------------------------------------------------------------ praproses
def praproses(raw: pd.DataFrame):
    """Terima data mentah (format dinas) atau data bersih (format data bawaan).
    Kembalikan (data bersih untuk tampilan, ringkasan praproses)."""
    raw = raw.copy()
    for c in raw.columns:
        raw[c] = raw[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    n_awal = len(raw)
    mentah = {"TAHUN_MULAI_USAHA", "SERTIFIKASI", "NIB"}.issubset(raw.columns)
    bersih_siap = set(VARIABEL).issubset(raw.columns)
    if not (mentah or bersih_siap):
        perlu = ("TAHUN_MULAI_USAHA, JUMLAH_TENAGA_KERJA, OMSET, BIAYA_PRODUKSI, GAJI_KARYAWAN, SERTIFIKASI, "
                 "PEMASARAN_ONLINE, KESULITAN_USAHA, NIB")
        raise ValueError(f"Kolom tidak dikenali. Format mentah memerlukan: {perlu}.")
    # duplikat persis hanya dihapus pada data mentah (seluruh kolom, seperti notebook); data bersih sudah bebas duplikat
    df = raw.drop_duplicates(keep="first").reset_index(drop=True) if mentah else raw.reset_index(drop=True)
    n_dup = n_awal - len(df)
    d = pd.DataFrame(index=df.index)
    d["ID_UMKM"] = df["ID_UMKM"] if "ID_UMKM" in df else [f"UMKM-{i:04d}" for i in range(1, len(df) + 1)]
    d["JENIS_LAPANGAN_USAHA"] = df["JENIS_LAPANGAN_USAHA"] if "JENIS_LAPANGAN_USAHA" in df else "-"
    if mentah:
        for c in ["TAHUN_MULAI_USAHA", "JUMLAH_TENAGA_KERJA", "OMSET", "BIAYA_PRODUKSI", "GAJI_KARYAWAN"]:
            d[c] = pd.to_numeric(df[c], errors="coerce")
        d.loc[d.TAHUN_MULAI_USAHA == 0, "TAHUN_MULAI_USAHA"] = np.nan
        d["SERTIFIKASI_JUMLAH"] = df.SERTIFIKASI.map(
            lambda t: 0 if str(t).strip() in ("-", "", "nan") else len([x for x in str(t).split(",") if x.strip()])).astype(int)
        d["NIB_STATUS"] = df.NIB.map(lambda s: "TIDAK" if (str(s).strip() == "-" or set(str(s).strip()) == {"0"}) else "YA")
        d["PEMASARAN_ONLINE"] = df.PEMASARAN_ONLINE.replace("-", np.nan)
        d["KESULITAN_USAHA"] = df.KESULITAN_USAHA
        awal = ["TAHUN_MULAI_USAHA", "JUMLAH_TENAGA_KERJA", "OMSET", "BIAYA_PRODUKSI", "GAJI_KARYAWAN",
                "SERTIFIKASI_JUMLAH", "PEMASARAN_ONLINE", "KESULITAN_USAHA", "NIB_STATUS"]
    else:
        for c in NUMERIK:
            d[c] = pd.to_numeric(df[c], errors="coerce")
        for c in BINER:
            d[c] = df[c].map(lambda v: {"1": "YA", "0": "TIDAK", "1.0": "YA", "0.0": "TIDAK"}.get(str(v), v))
        awal = NUMERIK + BINER
    hilang = d[awal].isna().sum()
    log = []
    for c in awal:
        m = int(d[c].isna().sum())
        if m == 0:
            continue
        num = pd.api.types.is_numeric_dtype(d[c])
        nilai = d[c].median() if num else d[c].mode().iloc[0]
        d[c] = d[c].fillna(nilai)
        log.append({"Variabel": c, "Metode": "median" if num else "modus", "Nilai imputasi": nilai, "Sel diimputasi": m})
    if mentah:
        d["LAMA_USAHA"] = TAHUN_REFERENSI - d.TAHUN_MULAI_USAHA
    for c in BINER:
        if not set(d[c].unique()).issubset({"YA", "TIDAK"}):
            raise ValueError(f"Kolom {c} harus berisi YA/TIDAK (atau 1/0).")
    ringkas = {"baris awal": n_awal, "duplikat dihapus": n_dup, "UMKM unik": len(d),
               "sel hilang": int(hilang.sum()), "hilang per variabel": hilang, "log imputasi": pd.DataFrame(log)}
    kolom = ["ID_UMKM", "JENIS_LAPANGAN_USAHA"] + (["TAHUN_MULAI_USAHA"] if "TAHUN_MULAI_USAHA" in d else []) + VARIABEL
    return d[kolom].reset_index(drop=True), ringkas


def parameter_skala(bersih):
    """Batas Min-Max (setelah log1p) dari data latih; dipakai ulang untuk UMKM baru."""
    par = {}
    for c in NUMERIK:
        s = bersih[c].astype(float)
        s = np.log1p(s) if c in LOG1P else s
        par[c] = (float(s.min()), float(s.max()))
    return par


def normalisasi(bersih, par=None):
    par = par or parameter_skala(bersih)
    X = pd.DataFrame(index=bersih.index)
    for c in NUMERIK:
        s = bersih[c].astype(float)
        s = np.log1p(s) if c in LOG1P else s
        lo, hi = par[c]
        X[c] = ((s - lo) / (hi - lo)).clip(0, 1) if hi > lo else 0.0
    for c in BINER:
        X[c] = (bersih[c] == "YA").astype(float).values
    return X


# ------------------------------------------------------------------ jarak
def gower_of(Xd):
    g = Xd.copy()
    for c in BINER:
        g[c] = np.where(g[c] == 1, "YA", "TIDAK").astype(object)
    Dm = gower.gower_matrix(g, cat_features=[False] * 6 + [True] * 3).astype(float)
    np.fill_diagonal(Dm, 0.0)
    return (Dm + Dm.T) / 2


def classical_mds(Dm, k=10):
    n = Dm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (Dm ** 2) @ J
    w, v = np.linalg.eigh(B)
    i = np.argsort(w)[::-1][:k]
    emb = v[:, i] * np.sqrt(np.clip(w[i], 0, None))
    pos = np.sort(w[w > 0])[::-1]
    return emb, pos


# ------------------------------------------------------------------ base dan konsensus
def _kproto(Xd, K, seed):
    Xp = np.hstack([Xd[NUMERIK].to_numpy(float), np.where(Xd[BINER].to_numpy() == 1, "YA", "TIDAK")])
    for init in ("Huang", "Cao", "random"):
        try:
            m = KPrototypes(n_clusters=K, init=init, n_init=5, random_state=seed, verbose=0)
            lab = m.fit_predict(Xp, categorical=[6, 7, 8])
            return m, np.asarray(lab).astype(int)
        except ValueError:
            continue
    raise RuntimeError("K-Prototypes gagal")


def latih_base3(Xd, Dm, K, seed=RANDOM_STATE):
    km = KMeans(K, n_init=10, random_state=seed).fit(Xd.to_numpy(float))
    kp, kp_lab = _kproto(Xd, K, seed)
    pam = kmedoids.pam(Dm, K, random_state=seed)
    labels = {"K-Means": np.asarray(km.labels_).astype(int), "K-Prototypes": kp_lab,
              "K-Medoids": np.asarray(pam.labels).astype(int)}
    return labels, {"K-Means": km, "K-Prototypes": kp, "medoid": np.asarray(pam.medoids)}


def coassoc(parts, names=B3):
    n = len(parts[names[0]])
    Cm = np.zeros((n, n), np.float32)
    for a in names:
        Cm += (parts[a][:, None] == parts[a][None, :]).astype(np.float32)
    return Cm / len(names)


def _rapikan(lab):
    return np.unique(lab, return_inverse=True)[1].astype(int)


def mcla(parts, names, K, seed=0):
    n = len(parts[names[0]])
    H = []
    for a in names:
        for k in np.unique(parts[a]):
            H.append((parts[a] == k).astype(np.float32))
    H = np.array(H)
    m = len(H)
    inter = H @ H.T
    sz = H.sum(1)
    W = inter / (sz[:, None] + sz[None, :] - inter)
    np.fill_diagonal(W, 0)
    xadj, adjncy, ew = [0], [], []
    for i in range(m):
        nb = list(np.nonzero(W[i] > 0)[0])
        adjncy += nb
        ew += [int(round(W[i, j] * 1000)) or 1 for j in nb]
        xadj.append(len(adjncy))
    if pymetis is not None:
        _, mem = pymetis.part_graph(K, xadj=xadj, adjncy=adjncy, eweights=ew)
        mem = np.asarray(mem)
    else:                              # cadangan bila pymetis tidak terpasang: partisi spektral pada meta-graf
        from sklearn.cluster import SpectralClustering
        A = W + 1e-6
        np.fill_diagonal(A, 0)
        mem = SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed).fit_predict(A)
    assoc = np.zeros((n, K), np.float32)
    for g in range(K):
        idx = np.where(mem == g)[0]
        if len(idx):
            assoc[:, g] = H[idx].mean(0)
    rng = np.random.default_rng(seed)
    mx = assoc.max(1)
    return np.array([rng.choice(np.where(assoc[i] == mx[i])[0]) for i in range(n)])


def konsensus6(parts, K, seed=RANDOM_STATE, names=B3):
    Cm = coassoc(parts, names)
    Dj = 1 - Cm
    np.fill_diagonal(Dj, 0.0)
    Dj = ((Dj + Dj.T) / 2).astype(float)
    out = {"AHC average": AgglomerativeClustering(K, metric="precomputed", linkage="average").fit_predict(Dj),
           "AHC complete": AgglomerativeClustering(K, metric="precomputed", linkage="complete").fit_predict(Dj),
           "Spectral (C)": SpectralClustering(K, affinity="precomputed", assign_labels="kmeans", random_state=seed,
                                              n_init=10).fit_predict(Cm.astype(float)),
           "K-Medoids (1-C)": np.asarray(kmedoids.pam(Dj, K, random_state=seed).labels),
           "MCLA": mcla(parts, names, K, seed)}
    L = np.column_stack([parts[a] for a in names]).astype(str)
    lk = None
    for init in ("Huang", "Cao"):
        try:
            lk = KModes(n_clusters=K, init=init, n_init=10, random_state=seed, verbose=0).fit_predict(L)
            break
        except ValueError:
            continue
    out["K-Modes (label)"] = lk
    return {k: _rapikan(np.asarray(v)) for k, v in out.items()}


def lifetime_ahc(parts, Ks=KS):
    Dj = 1 - coassoc(parts)
    np.fill_diagonal(Dj, 0)
    Dj = ((Dj + Dj.T) / 2).astype(float)
    h = linkage(squareform(Dj, checks=False), "average")[:, 2]
    n = len(Dj)
    return {K: float(h[n - K] - h[n - K - 1]) for K in Ks}


# ------------------------------------------------------------------ evaluasi
def evaluasi(lab, Dm, emb, Xd, parts):
    if len(np.unique(lab)) < 2:
        return {"SC": np.nan, "DBI": np.nan, "CHI": np.nan, "ANMI": np.nan, "SC Euclid": np.nan}
    sz = np.bincount(lab)
    return {"klaster aktif": int((sz > 0).sum()), "klaster terkecil (n)": int(sz[sz > 0].min()),
            "klaster terbesar (n)": int(sz.max()),
            "SC": silhouette_score(Dm, lab, metric="precomputed"),
            "DBI": davies_bouldin_score(emb, lab), "CHI": calinski_harabasz_score(emb, lab),
            "ANMI": float(np.mean([normalized_mutual_info_score(lab, parts[a]) for a in B3])),
            "SC Euclid": silhouette_score(Xd.to_numpy(float), lab)}


def peringkat_komposit(t):
    t = t.copy()
    t["R_SC"] = t.SC.rank(ascending=False, method="min")
    t["R_DBI"] = t.DBI.rank(method="min")
    t["R_CHI"] = t.CHI.rank(ascending=False, method="min")
    t["Skor komposit"] = t.R_SC + t.R_DBI + t.R_CHI
    return t


def protokol_k(EV, lifetime):
    KC = pd.DataFrame({c: peringkat_komposit(EV[EV.Metode == c][["K", "SC", "DBI", "CHI"]]).set_index("K")["Skor komposit"]
                       for c in CONS6}).loc[KS]
    KC["Jumlah (6 fungsi)"] = KC.sum(1)
    kopt = {c: int(KC[c].astype(float).idxmin()) for c in CONS6}
    KC["lifetime AHC average"] = [lifetime[K] for K in KS]
    KC["K optimal untuk fungsi"] = [", ".join(c for c, k in kopt.items() if k == K) for K in KS]
    return KC.reset_index().rename(columns={"index": "K"})


def jalankan_semua(bersih, Ks=KS, seed=RANDOM_STATE, progres=None):
    """Seluruh alur pada satu dataset: normalisasi → Gower → MDS → 3 base + 6 fungsi untuk tiap K → evaluasi."""
    Xd = normalisasi(bersih)
    Dm = gower_of(Xd)
    emb, pos = classical_mds(Dm, 10)
    labels, rows = {}, []
    for i, K in enumerate(Ks):
        if progres:
            progres(i / len(Ks), f"K = {K}: base clustering dan enam fungsi konsensus")
        P, _ = latih_base3(Xd, Dm, K, seed)
        out = konsensus6(P, K, seed)
        for k, v in {**P, **out}.items():
            labels[f"K{K}|{k}"] = v
            rows.append({"K": K, "Jenis": "Base" if k in B3 else "Konsensus", "Metode": k,
                         **evaluasi(v, Dm, emb, Xd, P)})
    if progres:
        progres(1.0, "Selesai")
    EV = pd.DataFrame(rows)
    P4 = {a: labels[f"K4|{a}"] for a in B3} if 4 in Ks else None
    life = lifetime_ahc(P4) if P4 is not None else {K: np.nan for K in KS}
    return {"EV": EV, "labels": pd.DataFrame(labels), "lifetime": life, "mds_pos": pos}


# ------------------------------------------------------------------ peran variabel
def porsi_variansi(Xd):
    v = Xd.var(ddof=0)
    return v / v.sum() * 100


def eta_kuadrat(Xd, lab):
    out = {}
    for c in Xd.columns:
        x = Xd[c].to_numpy(float)
        sst = ((x - x.mean()) ** 2).sum()
        ssb = sum((lab == g).sum() * (x[lab == g].mean() - x.mean()) ** 2 for g in np.unique(lab))
        out[c] = ssb / sst if sst > 0 else np.nan
    return pd.Series(out)


# ------------------------------------------------------------------ penempatan UMKM baru
class Penempat:
    """Menempatkan UMKM baru dengan logika EAC: tiap base memberi label bagi UMKM baru, co-association UMKM baru
    dengan tiap UMKM lama dihitung, lalu UMKM baru masuk ke klaster dengan rata-rata co-association tertinggi
    (setara jarak rata-rata 1 − C terkecil, seperti average linkage). Seri dipecah dengan jarak Gower rata-rata."""

    def __init__(self, bersih, K, final_lab, seed=RANDOM_STATE):
        self.par = parameter_skala(bersih)
        self.Xd = normalisasi(bersih, self.par)
        Dm = gower_of(self.Xd)
        self.P, self.model = latih_base3(self.Xd, Dm, K, seed)
        self.final = np.asarray(final_lab)
        self.K = K

    def _label_base(self, Xn):
        km = self.model["K-Means"].predict(Xn.to_numpy(float))
        Xp = np.hstack([Xn[NUMERIK].to_numpy(float), np.where(Xn[BINER].to_numpy() == 1, "YA", "TIDAK")])
        kp = self.model["K-Prototypes"].predict(Xp, categorical=[6, 7, 8])
        med = self.Xd.to_numpy(float)[self.model["medoid"]]
        dmed = np.abs(Xn.to_numpy(float)[:, None, :] - med[None, :, :]).mean(2)
        kd = self.P["K-Medoids"][self.model["medoid"]][dmed.argmin(1)]
        return {"K-Means": np.asarray(km), "K-Prototypes": np.asarray(kp).astype(int), "K-Medoids": kd}

    def tempatkan(self, bersih_baru):
        Xn = normalisasi(bersih_baru, self.par)
        lb = self._label_base(Xn)
        Xa = self.Xd.to_numpy(float)
        hasil = []
        for i in range(len(Xn)):
            c = np.mean([(self.P[a] == lb[a][i]).astype(float) for a in B3], axis=0)
            dg = np.abs(Xa - Xn.to_numpy(float)[i]).mean(1)
            skor = np.array([c[self.final == g].mean() for g in range(self.K)])
            jarak = np.array([dg[self.final == g].mean() for g in range(self.K)])
            kand = np.where(np.isclose(skor, skor.max()))[0]
            g = int(kand[np.argmin(jarak[kand])])
            hasil.append({"klaster": g, "skor": skor, "jarak": jarak, "base": {a: int(lb[a][i]) for a in B3}})
        return hasil, Xn

    def uji_penempatan_ulang(self):
        """Tiap UMKM lama ditempatkan ulang tanpa dirinya sendiri; persentase yang kembali ke klasternya."""
        C = coassoc(self.P).astype(float)
        np.fill_diagonal(C, 0)
        Xa = self.Xd.to_numpy(float)
        oh = np.eye(self.K)[self.final]
        cnt = oh.sum(0)[None, :] - oh
        skor = (C @ oh) / np.maximum(cnt, 1)
        cocok = []
        for i in range(len(Xa)):
            kand = np.where(np.isclose(skor[i], skor[i].max()))[0]
            if len(kand) > 1:
                dg = np.abs(Xa - Xa[i]).mean(1)
                dg[i] = np.nan
                jarak = np.array([np.nanmean(dg[self.final == g]) for g in kand])
                cocok.append(kand[np.argmin(jarak)] == self.final[i])
            else:
                cocok.append(kand[0] == self.final[i])
        return float(np.mean(cocok))