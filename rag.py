#!/usr/bin/env python3
"""DataHub RAG — the index box of a layered-context framework for AI agents.

Part of a three-layer context architecture that lets an agent read a large
memory tree / operations register WITHOUT spending model tokens on search:
Layer 1 = small fixed files loaded every session (pointers only),
Layer 2 = the memory tree on disk (searched, never loaded whole),
Layer 3 = this box: a fully local RAG index over your markdown corpus.
Embedding, indexing and searching all run on your own CPU — zero model
tokens; a scheduler (cron/systemd timer) re-indexes with plain scripts,
and the agent only ever reads the few cited chunks a query returns.
The box is a FINDER, not a source of truth: decisive facts are always
verified in the primary file a citation points to.

One script, two commands:
  python rag.py index            build/update the index (incremental, by mtime+size)
  python rag.py query "question" ask; prints at most 4 chunks from the top --k pool

Hybrid search: base score = w_sem*semantic (e5 cosine) + w_kw*exact-word,
both min-max normalised per query; weights 0.6/0.4, or 0.5/0.5 for short
queries (<=6 content words). The word component is idf-weighted.
Final score = base score x recency factor: register entries 1.00-1.60
(1-day half-life); other chunks 1.00-1.25 (3-day half-life).
A synonym map (sinonim.json) expands query phrases into corpus vocabulary
for the word component; a same-topic demotion pass pushes stale register
entries (cosine >= 0.92 with a newer entry) just below the newer one.
Index = SQLite (index.db, in this folder — YOUR data, never publish it).

Configuration is by environment variable + two small files (see README):
  RAG_CORPUS_DIR   main folder of .md files (default: ./corpus)
  RAG_EXTRA_BASE   base for extra-sources.txt paths (default: this folder)
  RAG_LOG_NAME     optional register file whose dated entries stay atomic
  RAG_ENTRY_PREFIX register entry prefix (default: DH)
  RAG_MODEL        embedding model (default: intfloat/multilingual-e5-small)
  RAG_EXCLUDE      comma-separated fnmatch patterns to skip (e.g. answer keys)
"""
import argparse, fnmatch, json, math, os, re, sqlite3, sys, time
from datetime import date, datetime, timezone, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "index.db")
WM = os.path.join(BASE, "watermark.json")
# Corpus sources are configuration, never hard-coded paths (framework rule).
CORPUS_DIR = os.path.expanduser(
    os.environ.get("RAG_CORPUS_DIR", os.path.join(BASE, "corpus")))
# Extra sources: explicit files listed one per line in extra-sources.txt
# (relative to BASIS_TAMBAHAN, or absolute). Their source label is the
# listed relative path — unique across folders (plain names can collide:
# two TASKS.md in different places stay distinguishable).
BASIS_TAMBAHAN = os.path.expanduser(
    os.environ.get("RAG_EXTRA_BASE", BASE))
BERKAS_SUMBER_TAMBAHAN = os.path.join(BASE, "extra-sources.txt")
LOG_NAME = os.environ.get("RAG_LOG_NAME", "")
ENTRY_PREFIX = os.environ.get("RAG_ENTRY_PREFIX", "DH")
MODEL_NAME = os.environ.get("RAG_MODEL", "intfloat/multilingual-e5-small")
EMBED_CHARS = 2200          # teks yang di-embedding per potongan (e5 maks 512 token)
WIB = datetime.now().astimezone().tzinfo  # cap waktu memakai zona lokal mesin
# Berkas yang tidak boleh masuk indeks (mis. kunci jawaban uji — anti-contekan);
# atur lewat RAG_EXCLUDE, pola fnmatch dipisah koma. Bawaan: tidak ada.
POLA_KECUALI = tuple(p.strip() for p in
                     os.environ.get("RAG_EXCLUDE", "").split(",") if p.strip())
STOPKATA = {"berapa", "apa", "bagaimana", "mengapa", "kapan", "siapa", "yang",
            "dan", "di", "ke", "dari", "saat", "ini", "itu", "adalah", "the"}
CAP_DH = 2400     # entri register DH dicetak nyaris utuh (satuan atomik)
CAP_LAIN = 650    # potongan non-register tetap ringkas
CETAK_MAKS = 4    # jumlah potongan tercetak per tanya

# ---------- pemotongan ----------

_EP = re.escape(ENTRY_PREFIX)
DH_HEAD = re.compile(r"^## (" + _EP + r"-\d{4}-\d{2}-\d{2}-\d+)\b.*$", re.M)
TBL_HEAD = re.compile(r"^\|\s*(" + _EP + r"-\d{4}-\d{2}-\d{2}-\d+)\s*\|")
HEAD = re.compile(r"^#{1,3} .*$", re.M)
BULAN = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "mei": 5, "jun": 6, "jul": 7,
         "agu": 8, "ags": 8, "sep": 9, "okt": 10, "nov": 11, "des": 12,
         "januari": 1, "februari": 2, "maret": 3, "april": 4, "juni": 6,
         "juli": 7, "agustus": 8, "september": 9, "oktober": 10,
         "november": 11, "desember": 12}
DATE_ID = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})")


def cari_tanggal(teks):
    m = DATE_ID.search(teks)
    if m and m.group(2).lower() in BULAN:
        return "%04d-%02d-%02d" % (int(m.group(3)), BULAN[m.group(2).lower()],
                                   int(m.group(1)))
    return None


def potong_log(lines):
    """Satu entri DH = satu potongan; jangan membelah entri."""
    # potongan berjalan dari header DH ke header DH berikutnya;
    # bagian log yang lebih baru memakai baris tabel "| DH-... |" — itu pun entri
    out, cur = [], None
    for i, ln in enumerate(lines):
        m = DH_HEAD.match(ln) or TBL_HEAD.match(ln)
        if m:
            if cur:
                out.append(cur)
            cur = {"start": i, "end": i, "dh": m.group(1),
                   "date": m.group(1)[len(ENTRY_PREFIX) + 1:
                                      len(ENTRY_PREFIX) + 11],
                   "lines": [ln]}
        elif cur is not None:
            cur["lines"].append(ln)
            cur["end"] = i
        # teks sebelum entri DH pertama (pembuka log) diabaikan kecuali ada isi
    if cur:
        out.append(cur)
    return [{"start": c["start"] + 1, "end": c["end"] + 1, "dh": c["dh"],
             "date": c["date"], "text": "\n".join(c["lines"]).strip()}
            for c in out if "\n".join(c["lines"]).strip()]


def potong_biasa(lines):
    """Per judul; bagian panjang dipecah per paragraf (maks ±1600 char)."""
    sections, cur = [], {"start": 0, "head": "", "lines": []}
    for i, ln in enumerate(lines):
        if HEAD.match(ln) and cur["lines"]:
            sections.append(cur)
            cur = {"start": i, "head": ln.strip("# ").strip(), "lines": []}
        elif HEAD.match(ln):
            cur["start"] = i
            cur["head"] = ln.strip("# ").strip()
        cur["lines"].append(ln)
    if cur["lines"]:
        sections.append(cur)
    out = []
    for s in sections:
        paras, buf, bstart = [], [], s["start"]
        for j, ln in enumerate(s["lines"]):
            buf.append(ln)
            if ln.strip() == "" and len("\n".join(buf)) > 500:
                paras.append((bstart, s["start"] + j, "\n".join(buf).strip()))
                buf, bstart = [], s["start"] + j + 1
        if buf:
            paras.append((bstart, s["start"] + len(s["lines"]) - 1,
                          "\n".join(buf).strip()))
        for (a, b, teks) in paras:
            if not teks:
                continue
            # pecah lagi bila masih terlalu panjang
            while len(teks) > 1600:
                cut = teks.rfind("\n", 0, 1600)
                if cut < 400:
                    cut = 1600
                out.append({"start": a + 1, "end": b + 1, "dh": None,
                            "date": cari_tanggal(teks), "text": teks[:cut].strip()})
                teks = teks[cut:].strip()
            if teks:
                out.append({"start": a + 1, "end": b + 1, "dh": None,
                            "date": cari_tanggal(teks), "text": teks})
    return out


def daftar_sumber_tambahan():
    try:
        with open(BERKAS_SUMBER_TAMBAHAN, encoding="utf-8") as fh:
            return [ln.strip() for ln in fh
                    if ln.strip() and not ln.strip().startswith("#")]
    except FileNotFoundError:
        return []


def label_dari_path(path):
    """Label kolom source untuk sebuah path: berkas korpus utama = nama
    berkas; berkas tambahan = path relatif ke BASIS_TAMBAHAN (persis label
    yang disimpan baca_korpus) — dipakai penghapusan baris berkas hilang
    agar konsisten dengan label saat penyisipan."""
    if os.path.dirname(path) == CORPUS_DIR:
        return os.path.basename(path)
    if path.startswith(BASIS_TAMBAHAN + "/"):
        return path[len(BASIS_TAMBAHAN) + 1:]
    return os.path.basename(path)


def baca_korpus():
    files = sorted(f for f in (os.listdir(CORPUS_DIR)
                               if os.path.isdir(CORPUS_DIR) else [])
                   if f.endswith(".md")
                   and not any(fnmatch.fnmatch(f, p) for p in POLA_KECUALI))
    hasil = []
    for f in files:
        path = os.path.join(CORPUS_DIR, f)
        st = os.stat(path)
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().split("\n")
        chunks = potong_log(lines) if f == LOG_NAME else potong_biasa(lines)
        hasil.append((f, path, st.st_mtime, st.st_size, chunks))
    for rel in daftar_sumber_tambahan():
        if any(fnmatch.fnmatch(os.path.basename(rel), p)
               for p in POLA_KECUALI):
            continue
        path = os.path.join(BASIS_TAMBAHAN, rel)
        if not os.path.isfile(path):
            continue
        st = os.stat(path)
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().split("\n")
        hasil.append((rel, path, st.st_mtime, st.st_size,
                      potong_biasa(lines)))
    return hasil

# ---------- indeks ----------


def buka_db():
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime REAL, size INTEGER);
    CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, source TEXT,
        line_start INTEGER, line_end INTEGER, dh TEXT, entry_date TEXT,
        text TEXT, vector BLOB);
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
    """)
    return con


def muat_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def cmd_index():
    import numpy as np
    t0 = time.time()
    korpus = baca_korpus()
    con = buka_db()
    tersimpan = {r[0]: (r[1], r[2]) for r in
                 con.execute("SELECT path, mtime, size FROM files")}
    berubah = [k for k in korpus
               if tersimpan.get(k[1]) != (k[2], k[3])]
    hilang = set(tersimpan) - {k[1] for k in korpus}
    for p in hilang:
        con.execute("DELETE FROM chunks WHERE source=?", (label_dari_path(p),))
        con.execute("DELETE FROM files WHERE path=?", (p,))
    if not berubah and not hilang:
        total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"Index up to date: {total} chunks, no files changed.")
        con.close()
        return
    model = muat_model()
    for (nama, path, mtime, size, chunks) in berubah:
        con.execute("DELETE FROM chunks WHERE source=?", (nama,))
        teks = [("passage: " + c["text"][:EMBED_CHARS]) for c in chunks]
        vecs = model.encode(teks, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=False)
        for c, v in zip(chunks, vecs):
            con.execute(
                "INSERT INTO chunks(source,line_start,line_end,dh,entry_date,"
                "text,vector) VALUES (?,?,?,?,?,?,?)",
                (nama, c["start"], c["end"], c["dh"], c["date"], c["text"],
                 np.asarray(v, dtype=np.float32).tobytes()))
        con.execute("INSERT OR REPLACE INTO files VALUES (?,?,?)",
                    (path, mtime, size))
        con.commit()  # commit per berkas: bangun panjang aman dari putus di tengah
        print(f"  re-indexed: {nama} ({len(chunks)} chunks)")
    con.execute("INSERT OR REPLACE INTO meta VALUES ('model',?)", (MODEL_NAME,))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('built_at',?)",
                (datetime.now(WIB).isoformat(timespec="seconds"),))
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    terbaru = max(k[2] for k in korpus)
    wm = {"built_at": datetime.now(WIB).isoformat(timespec="seconds"),
          "newest_source": datetime.fromtimestamp(terbaru, WIB)
          .isoformat(timespec="seconds"),
          "files": len(korpus), "chunks": total, "model": MODEL_NAME}
    with open(WM, "w") as fh:
        json.dump(wm, fh, indent=1)
    con.close()
    print(f"Done: {total} chunks from {len(korpus)} files, "
          f"{time.time()-t0:.1f}s.")

# ---------- tanya (hibrida) ----------

TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenisasi(s):
    return TOKEN.findall(s.lower())


def cmd_query(pertanyaan, k):
    import numpy as np
    t0 = time.time()
    con = buka_db()
    rows = con.execute("SELECT source,line_start,line_end,dh,entry_date,text,"
                       "vector FROM chunks").fetchall()
    con.close()
    if not rows:
        print("Index is empty — run: python rag.py index")
        return
    try:
        wm = json.load(open(WM))
        kepala = f"Index as of {wm['built_at']}"
    except Exception:
        kepala = "Index as of (watermark unreadable)"
    model = muat_model()
    t_model = time.time()
    qv = model.encode(["query: " + pertanyaan], normalize_embeddings=True,
                      show_progress_bar=False)[0]
    t_emb = time.time()
    mat = np.frombuffer(b"".join(r[6] for r in rows),
                        dtype=np.float32).reshape(len(rows), -1)
    sem = mat @ np.asarray(qv, dtype=np.float32)
    qtok = set(tokenisasi(pertanyaan))
    # Putaran 4: peta sinonim — frasa pertanyaan yang cocok peta (sinonim.json)
    # menambah istilah kosakata register ke komponen kata (idf berlaku);
    # sisi makna tidak diubah.
    tambahan = set()
    try:
        with open(os.path.join(BASE, "sinonim.json")) as fh:
            peta_sinonim = json.load(fh)
    except Exception:
        peta_sinonim = {}
    tanya_norm = " " + " ".join(tokenisasi(pertanyaan)) + " "
    for frasa, padanan in peta_sinonim.items():
        if frasa.startswith("_"):
            continue
        if " " + " ".join(tokenisasi(frasa)) + " " in tanya_norm:
            for p in padanan:
                tambahan |= set(tokenisasi(p))
    if tambahan:
        print(f"[synonyms active: {sorted(tambahan)}]", file=sys.stderr)
    istilah = qtok | tambahan
    # Putaran 3: bobot kata dengan kelangkaan (idf) — kata yang muncul di
    # banyak potongan (mis. "kas") sinyalnya menipis; kata langka berbobot
    # besar. idf = log(1 + total_potongan / potongan_mengandung_kata).
    hitung, df = [], {t: 0 for t in istilah}
    for r in rows:
        teks = r[5].lower()
        c = {}
        for t in istilah:
            n = teks.count(t)
            if n:
                c[t] = n
                df[t] += 1
        hitung.append(c)
    idf = {t: math.log(1 + len(rows) / df[t]) for t in istilah if df[t]}
    kw = np.zeros(len(rows))
    for i, r in enumerate(rows):
        skor = 0.0
        for t, n in hitung[i].items():
            skor += idf[t] * math.log1p(n)
        if r[3] and r[3].lower() in qtok:      # nomor DH persis = dorongan kuat
            skor += 5.0
        kw[i] = skor

    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi > lo else a * 0

    # Putaran 2: kueri pendek (<=6 kata isi, di luar stopkata) mudah tenggelam
    # sinyal kata-persisnya (kasus Q1 "kas") -> bobot kata dinaikkan ke 0,5
    isi = qtok - STOPKATA
    if len(isi) <= 6:
        w_makna, w_kata = 0.5, 0.5
    else:
        w_makna, w_kata = 0.6, 0.4
    dasar = w_makna * norm(sem) + w_kata * norm(kw)

    # Putaran 2: bobot kebaruan — faktor 1,00–1,25, paruh-waktu 3 hari,
    # dihitung dari tanggal entri terbaru di indeks; tanpa tanggal = 1,00
    def ke_tanggal(s):
        try:
            y, m, d = (int(x) for x in s.split("-"))
            return date(y, m, d)
        except Exception:
            return None

    tanggal = [ke_tanggal(r[4]) if r[4] else None for r in rows]
    bertanggal = [t for t in tanggal if t]
    acuan = max(bertanggal) if bertanggal else None
    faktor = np.ones(len(rows))
    if acuan:
        for i, t in enumerate(tanggal):
            if t:
                if rows[i][3]:
                    # Putaran 3: entri register DH — kebaruan tajam
                    # (1,00–1,60, paruh-waktu 1 hari) agar di dalam satu
                    # topik entri terbaru menang jelas atas entri lama
                    faktor[i] = 1.0 + 0.60 * (0.5 ** ((acuan - t).days / 1.0))
                else:
                    faktor[i] = 1.0 + 0.25 * (0.5 ** ((acuan - t).days / 3.0))
    final = dasar * faktor
    # Putaran 4: demosi entri basi dalam gugus setopik — potongan ber-DH
    # yang maknanya sangat mirip (kosinus >= 0,92; vektor tersimpan sudah
    # ternormalisasi) dengan potongan ber-DH lain yang tanggalnya lebih
    # baru diturunkan tepat ke bawah entri yang lebih baru itu.
    idx_dh = [i for i in range(len(rows)) if rows[i][3] and tanggal[i]]
    if len(idx_dh) > 1:
        mirip = mat[idx_dh] @ mat[idx_dh].T
        turun = {}
        for a, i in enumerate(idx_dh):
            for b, j in enumerate(idx_dh):
                if tanggal[j] > tanggal[i] and mirip[a, b] >= 0.92 \
                        and final[i] >= final[j]:
                    turun[i] = min(turun.get(i, final[i]), final[j] * 0.999)
        for i, v in turun.items():
            final[i] = v
    peringkat = np.argsort(-final)
    urut = peringkat[:k]
    for pos, i in enumerate(peringkat[:10], 1):  # diagnostik peringkat (stderr)
        r = rows[int(i)]
        print(f"  #{pos} skor={final[int(i)]:.3f} {r[0]}"
              f"{' ' + r[3] if r[3] else ''}", file=sys.stderr)

    # Putaran 2: cetak maks CETAK_MAKS dari kolam k teratas, utamakan ber-DH;
    # tampilkan kembali urut peringkat. Entri DH dicetak nyaris utuh.
    kolam = [int(i) for i in urut]
    ber_dh = [i for i in kolam if rows[i][3]]
    tak_dh = [i for i in kolam if not rows[i][3]]
    terpilih = sorted((ber_dh + tak_dh)[:CETAK_MAKS], key=kolam.index)
    keluar = [kepala]
    for i in terpilih:
        r = rows[i]
        label = f"[source: {r[0]}, lines {r[1]}–{r[2]}"
        if r[3]:
            label += f", {r[3]}"
        if r[4]:
            label += f", {r[4]}"
        label += "]"
        teks = r[5]
        maks_teks = CAP_DH if r[3] else CAP_LAIN
        if len(teks) > maks_teks:
            potong = teks[:maks_teks]
            spasi = potong.rfind(" ")
            if spasi > maks_teks - 120:  # potong rapi di batas kata terdekat
                potong = potong[:spasi]
            teks = potong.rstrip() + " …"
        keluar.append(label + "\n" + teks)
    print("\n\n".join(keluar))
    print(f"[time: total {time.time()-t0:.2f}s | model load "
          f"{t_model-t0:.2f}s | query embedding {t_emb-t_model:.3f}s | "
          f"semantic/word weights {w_makna}/{w_kata}]",
          file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    q = sub.add_parser("query")
    q.add_argument("pertanyaan")
    q.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "index":
        cmd_index()
    else:
        cmd_query(a.pertanyaan, a.k)
