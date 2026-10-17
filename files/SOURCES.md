# Croatian Legal Document QA - project knowledge files

Source: the user-provided `Croatian_Legal_Document_QA_Dataset.zip`.
The archive's original README is preserved verbatim in `ORIGIN.md`; its original
13 questions and answers are in `../evaluation/dataset/references/questions.json`.
`sources.json` records the original archive member, archive SHA-256, file SHA-256
and source links. File contents were not modified; only folders were renamed.

## What the corpus contains

- `pdf/kazneni_zakon_izvadak.pdf`: criminal-law test excerpt (one page).
- `pdf/prometni_zakon_izvadak.pdf`: road-traffic test excerpt (one page).
- `pdf/*_scan.pdf`: two image-only PDFs for the OCR fallback.
- `png/*_stranica_1.png`: two original images for image OCR.

There are **two topics in six files**, not six independent legal documents.
These are simplified TEST EXCERPTS. They omit conditions and exceptions and are
not official consolidated laws or verified current legal advice. The archive
states a review date of 17 October 2026; this project does not certify that date
or independently establish that every provision is current.

## Permission and attribution

The archive's reuse statement explicitly permits its original test layouts and
excerpts in a demonstration repository with attribution to the listed sources.
This is the supplied permission, not a CC0 or Creative Commons licence invented
by this project. Preserve `ORIGIN.md` and the sources below when redistributing.
Published official legislative texts are addressed by Article 18(3) of the
Croatian Copyright and Related Rights Act:
https://narodne-novine.nn.hr/clanci/sluzbeni/full/2021_10_111_1941.html

Sources supplied by the archive author (links are references, not copied sites):

1. https://narodne-novine.nn.hr/clanci/sluzbeni/2011_11_125_2498.html
2. https://narodne-novine.nn.hr/clanci/sluzbeni/full/2025_11_136_2018.html
3. https://zakoni.lexium.hr/kazneni-zakon
4. https://cvh.hr/gradani/propisi-i-upute/zakoni/zakon-o-sigurnosti-prometa-na-cestama/

No third-party site layout, logos, commentary or commercial consolidated
publication has been copied into this project.

## Build the local index

From the project root, with Ollama and `qwen3-embedding:0.6b` available:

```powershell
.\.venv\Scripts\Activate.ps1
$documents = Get-ChildItem files/pdf/*.pdf, files/png/*.png
python -m retrieval ingest @($documents.FullName)
python -m retrieval list
python agent.py
```

All six formats are deliberately indexed for this demo. Paired files contain
repeated content, so they are not independent evaluation samples. New web
sessions receive only manifest-listed public files, reusing their embeddings.
`data/` holds the generated, Git-ignored index and document snapshots.
