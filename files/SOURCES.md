# Knowledge documents: synthetic invoices

Source: **Synthetic Shopify Invoice Test Pack v1.0.0**, by **SalorWorks**.

- Repository: https://github.com/SalorWorks/shopify-invoice-test-pack
- Revision: `bb2e531cfa504e6f7200243620121c523f9f7622`
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Upstream permission/notice: `LICENSE`; citation metadata: `CITATION.cff`.
- Original files are copied without modification. No real supplier records are used.

| Local files | Upstream fixture | Purpose |
| --- | --- | --- |
| `pdf/invoice-clean.pdf`, `png/invoice-clean.png` | `fixtures/01-simple-english/` | Native PDF text, table and clean image OCR |
| `pdf/invoice-scan.pdf`, `png/invoice-scan.png` | `fixtures/14-poor-scan/` | Rotated, uneven-contrast OCR; PDF contains the raster |
| `pdf/invoice-multipage.pdf`, `png/invoice-multipage.png` | `fixtures/02-multipage/` | Multi-page PDF; PNG is page 1 |
| `pdf/invoice-discount.pdf`, `png/invoice-discount.png` | `fixtures/09-line-discount/` | Line discount and VAT |
| `pdf/invoice-case-pack.pdf`, `png/invoice-case-pack.png` | `fixtures/12-case-pack/` | Cases and units per case |

The paired `*.expected.json` files in `evaluation/dataset/references/` contain
the author's expected field values. Our five QA references live in
`evaluation/dataset/questions.json`.
`sources.json` records exact download URLs and SHA-256 hashes for reproducibility.
Redistribution of these samples must retain appropriate credit and the license
link; indicate changes if you modify them.

All five PDFs and five PNGs are indexed in the agent's default document session.
To index them again (unchanged files are skipped), use PowerShell:

```powershell
Get-ChildItem files/pdf,files/png -File |
  ForEach-Object { python -m retrieval ingest $_.FullName }
```

These are the agent's knowledge documents; the small evaluation reuses them.
`data/` holds the generated index, vectors and working snapshots and is ignored
by Git. Moving these originals does not require re-embedding: the indexed
content, file names and pages remain the same.

For an isolated index, use `python -m retrieval --session invoices ingest ...`.
Set `DOCUMENT_SESSION=invoices` in `.env` and restart the agent to search that session.
