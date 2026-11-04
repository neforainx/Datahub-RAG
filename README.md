# DataHub RAG — a layered-context framework for AI agents

**Let your agent read a large memory tree and operations register without
spending model tokens on search.**

An agent that carries its whole memory in context pays for it on every
single message: the same files are re-sent, re-read and re-summarised
forever. This framework replaces that habit with three layers and one
local tool, so *searching* memory costs **zero model tokens** — the agent
only ever reads the handful of cited chunks a query returns.

## The architecture

```
Layer 1 — Fixed files      Loaded into context every session.
          (tiny)           Identity, rules, standing decisions — and
                           POINTERS to the layers below. Nothing else.

Layer 2 — Memory tree      Your notes, journals, project files on disk.
          (on disk)        Searched on demand, never loaded whole.

Layer 3 — Index box        THIS tool: a local RAG index over Layer 2 and
          (this repo)      your operations register. Finds the right
                           chunk + citation; the agent then opens the
                           primary file to verify decisive facts.
```

Reading order is always 1 → 2 → 3, one layer at a time.

### Rules that make it work

- **The index is a finder, not a source of truth.** Money figures,
  statuses and decisions are read from the primary file a citation
  points to — never trusted from a search snippet alone.
- **Indexing and searching are scripts, not agent work.** A timer
  re-indexes changed files; a query is one local command. Neither
  wakes a model.
- **Layer 1 stays tiny.** If a fact can be found by search, it does
  not belong in the always-loaded files — a pointer does.
- **Newest is not automatically truest.** Recency is a ranking signal
  (see below), not a verdict; corrections are written into the primary
  files so the record itself converges.

## What the box does

`rag.py` is a single-file, fully local search engine over markdown:

- **Chunking** — by heading, then by paragraph (~1,600 chars max).
  A register file (below) is chunked per entry instead, so one entry
  is never split.
- **Embeddings** — `intfloat/multilingual-e5-small` by default
  (384-dim, CPU-friendly, strong on non-English text too), stored as
  blobs in one SQLite file (`index.db`).
- **Hybrid scoring** — semantic cosine + exact-word score, the word
  side weighted by idf so rare terms dominate common ones; short
  queries get a stronger word weight.
- **Recency weighting** — scores are multiplied by a freshness factor
  (register entries: up to ×1.6, 1-day half-life; other chunks:
  up to ×1.25, 3-day half-life), and a demotion pass pushes a stale
  register entry just below a newer entry on the same topic
  (cosine ≥ 0.92), so yesterday's correction outranks last week's
  superseded note without hard-deleting history.
- **Synonym expansion** — `sinonim.json` maps question phrases to the
  vocabulary your corpus actually uses, feeding the word component.
- **Incremental indexing** — only files whose mtime/size changed are
  re-embedded; the build commits per file, so an interrupted run is
  safe to repeat.

## Quickstart

```bash
git clone <this repo> datahub-rag && cd datahub-rag
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

# try it on the bundled fictional sample corpus:
.venv/bin/python rag.py index
.venv/bin/python rag.py query "what is the refund policy"
.venv/bin/python rag.py query "when is the weekly meeting"
```

The first run downloads the embedding model (~470 MB). Query output
prints cited chunks like:

```
[source: sample-journal.md, lines 9–15]
### Refund policy
Decided 14 October 2026: customers can request a full refund within …
```

Then point it at your own corpus (next section), delete `corpus/`,
and re-index.

## Configuration

Corpus sources are configuration — never edit paths into the code.

| Setting | Default | Meaning |
|---|---|---|
| `RAG_CORPUS_DIR` | `./corpus` | Main folder; every `.md` in it is indexed (label = file name) |
| `extra-sources.txt` | — | Extra files, one per line; label = the listed path, so same-named files in different folders stay distinct |
| `RAG_EXTRA_BASE` | this folder | Base that relative `extra-sources.txt` paths resolve against |
| `RAG_MODEL` | `intfloat/multilingual-e5-small` | Any sentence-transformers model |
| `RAG_EXCLUDE` | *(none)* | Comma-separated fnmatch patterns to skip — e.g. your evaluation answer keys, so the agent can't "cheat" from them |
| `sinonim.json` | sample included | Question phrase → corpus vocabulary map |

### Register mode (optional)

If you keep an append-only operations log whose entries carry headers
like `## OP-2026-11-02-002` (prefix + ISO date + sequence, or table rows
`| OP-2026-11-02-002 |`), enable register mode for that file:

```bash
RAG_LOG_NAME=sample-register.md RAG_ENTRY_PREFIX=OP \
  .venv/bin/python rag.py index
.venv/bin/python rag.py query "what is the current balance"
```

Entries stay atomic, carry their date from the header, and get the
sharper recency weighting + stale-entry demotion described above.

## Keeping it fresh

Re-indexing is a plain script run — schedule it; never spend agent
time on it. Ready-made systemd units live in `units/` (refresh
2 minutes after boot, then every 3 hours, `Persistent=true` so a
missed run is caught up). A cron line calling `rag.py index` works
just as well.

## Data safety

`index.db` and `watermark.json` are built from **your** corpus and
contain **your** text. They are git-ignored here on purpose. Publish
the framework, never your index.

## Limitations

- Single machine, single writer; the query path scans all chunk
  vectors in memory — comfortable up to roughly 100k chunks.
- Dates are recognised in entry headers and in `12 October 2026` /
  `12 Okt 2026` style text; undated chunks get no recency boost.
- The box finds and cites; it does not summarise or judge. That part
  stays with the agent — on the primary source, not the snippet.

## License

MIT — see `LICENSE`.
