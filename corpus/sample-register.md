# Operations Register (SAMPLE)

Register-style file: dated entries with `## OP-YYYY-MM-DD-NNN` headers.
Used with register mode (RAG_LOG_NAME + RAG_ENTRY_PREFIX=OP) so every
entry stays one atomic chunk and newer entries outrank older ones on
the same topic.

## OP-2026-10-28-001
Payment received from Client A: 1,500 credits for the October package.
Balance after this entry: 1,500 credits.

## OP-2026-11-02-002
Refund paid to Client B: 200 credits (see the refund policy in the
team journal). Balance after this entry: 1,300 credits.

## OP-2026-11-03-003
Payment received from Client C: 700 credits for a rush job.
Balance after this entry: 2,000 credits.
