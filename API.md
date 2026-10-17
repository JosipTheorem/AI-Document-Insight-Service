# Pokretanje

Ollama mora biti pokrenuta. Iz korijena projekta pokreni backend u jednom terminalu:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

U drugom terminalu pokreni sučelje:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Streamlit: http://localhost:8501 · Interaktivna API dokumentacija: http://localhost:8000/docs

Streamlit koristi isključivo HTTP za razgovor i upload. `API_BASE_URL` može
promijeniti adresu backenda; zadana je `http://127.0.0.1:8000`.
Terminalski agent i evaluacijski notebook nastavljaju raditi kao prije.

## Tijek

1. Sučelje automatski poziva `POST /sessions` i dobiva nasumičan ID sesije.
   Već indeksirani javni demo računi kopiraju se u njezin opseg bez ponovnog
   embeddinga. API klijent može poslati `{"include_examples": false}` za praznu sesiju.
2. **Obradi dokument** šalje PDF/PNG/JPEG kao multipart na `POST /upload`, uz
   `session_id`. Backend provjeri datoteku, spremi radnu kopiju, izdvoji tekst/OCR,
   očisti ga, napravi chunkove i embeddinge te ih spremi u SQLite indeks.
   Uspješan odgovor znači da je dokument odmah dostupan pretrazi te sesije.
3. Enter ili gumb slanja poziva `POST /ask` s `session_id` i `message`.
   Isti 2B agent, security hook i hybrid search vraćaju odgovor. HTTP odgovor
   ima polja `session_id`, `answer` i `sources`; izvori su citati koje je agent
   naveo i koji se pojavljuju u rezultatima njegova alata.

`GET /documents?session_id=...` vraća dokumente sesije. `GET /health` provjerava
je li backend dostupan, bez pozivanja modela.

Datoteke su ograničene na 30 MB, slike na 40 megapiksela, PDF-ovi na 200 stranica.
Isti sadržaj u istoj sesiji ne obrađuje se ponovno. Novi upload dodaje dokument;
ne briše prethodne. Pogreška obrade ne uklanja već indeksirane dokumente.

`data/` ostaje lokalna baza i spremište radnih kopija. ID-jevi sesija i dokumenti
ostaju nakon restarta; povijest razgovora trenutačno je u memoriji backenda i
resetira se njegovim restartom. **Novi razgovor** stvara zasebnu sesiju.
Ovaj lokalni demo koristi ID sesije kao ključ pristupa, bez korisničkih računa/JWT-a.
