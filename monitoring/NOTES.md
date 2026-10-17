# Logging i monitoring

Backend i terminal pišu događaje u `app_logging/logs/application.txt`.
Jedan red = jedan JSON događaj, s UTC vremenom. Datoteka se ne sprema u Git.
`APP_LOG_FILE` po potrebi mijenja putanju; backend i dashboard moraju koristiti
istu vrijednost. Lokalni primjer namijenjen je jednom backend procesu.

Dashboard pokreni iz korijena projekta:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run monitoring/dashboard.py --server.port 8502
```

Otvori http://localhost:8502. Filter uključuje oba odabrana datuma u zoni
Europe/Zagreb. Prikaz se osvježava svakih pet sekundi. Nema zasebne baze metrika:
sve se računa iz tog tekstualnog loga. Velika produkcijska količina logova
zahtijevala bi rotaciju i centralizirani servis; ovdje pokazujemo osnovnu ideju.

## Što se računa

- Tokeni: stvarni `usage_metadata` / `prompt_eval_count` / `eval_count` iz Ollame,
  jednom po dovršenom LLM pozivu, uključujući security i sve agentove pozive
  poslije alata. Output uključuje tokene koje model prijavi za generiranje,
  uključujući reasoning ako ih provider tako broji; sam reasoning ne zapisujemo.
  Embedding pozivi mjere se vremenski, ali njihovi tokeni nisu u LLM zbroju.
- Minutni i sekundni intervali: tokeni se pripisuju završetku poziva. To nije
  rekonstrukcija token streaminga niti stvarna brzina dekodiranja modela.
  Sekundni graf prikazuje posljednjih 15 minuta koje sadrže podatke u filtru.
- Medijan/p95/p99: trajanje cijelog uspješnog `/ask` HTTP zahtjeva, uključujući
  čekanje, security, pretragu i generiranje. Uključuju i uspješne terminalske
  poruke ako je terminal korišten. Sigurnosno odbijanje je obrađena poruka.
- Greške: broj različitih request ID-jeva s ERROR događajem; isti kvar ne
  broji se ponovno kad ga zabilježe callback, agent i HTTP sloj.
  HTTP 4xx, uključujući potrošenu kvotu, prikazuju se odvojeno.
- Faze: ekstrakcija/OCR, embedding, pretraga, model i cijeli upload. Neke faze
  se preklapaju, pa se njihova trajanja ne zbrajaju.

Nema umjetnih ili rekonstruiranih starih podataka. Ako token metapodaci nedostaju,
dashboard to navodi. Nedovršen poziv/greška može potrošiti tokene za koje provider
nije vratio podatke; tada ih ne možemo pouzdano prebrojati. Nepotpun završni red
loga preskače se i ponovno čita pri osvježavanju.

Ne zapisujemo pitanja, odgovore, tekst dokumenata ni pune sesijske pristupne ID-jeve.
`session_ref` je njihov kratki hash, a `request_id` povezuje faze istog zahtjeva.
Mjesta koja zapisuju događaje u Pythonu označena su komentarom `#logging`.

## Kvota razgovora

API dopušta 10 valjanih poruka po sesiji. Security odbijanja i neuspjeli pokušaji
nakon prihvaćenog `/ask` ulaza troše poruku; neispravan zahtjev ne troši kvotu.
Deseti odgovor prikazuje obavijest za novi razgovor. Jedanaesti zahtjev vraća
HTTP 429 bez poziva security ili glavnog modela. Brojač je u SQLiteu i ostaje
nakon restarta. Novi razgovor dobiva zasebnu kvotu; terminal ima 10 poruka po
pokretanju. Evaluacija nije interaktivna sesija pa nije ograničena tom kvotom.
