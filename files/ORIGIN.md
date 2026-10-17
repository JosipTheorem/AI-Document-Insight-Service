# Croatian Legal Document QA - mini test dataset

Mala demonstracijska zbirka za **AI Backend Developer / Document Insight Service**.

## Što paket sadrži

- `pdf/kazneni_zakon_izvadak.pdf`: kratki PDF s odabranim kaznenim djelima i kaznama.
- `pdf/prometni_zakon_izvadak.pdf`: kratki PDF s odabranim prometnim kaznama.
- `images/*.png`: slike PDF stranica za testiranje OCR-a nad slikama.
- `scanned_pdf/*_scan.pdf`: *image-only* PDF-ovi bez tekstualnog sloja, za testiranje OCR fallbacka.
- `questions.json`: 13 gotovih pitanja i referentnih odgovora (uključujući pitanje bez odgovora u tekstu).

## Brzi test

1. Na `POST /upload` pošalji `pdf/kazneni_zakon_izvadak.pdf`.
2. Na `POST /ask` postavi: **Kolika je kazna zatvora za krađu?**
3. Očekivani odgovor: **Do tri godine**, članak 228. stavak 1.
4. Ponovi test s `images/kazneni_zakon_izvadak_stranica_1.png` ili `scanned_pdf/kazneni_zakon_izvadak_scan.pdf`. Ti ulazi trebaju proći OCR.
5. Učitaj oba PDF-a i usporedi pitanja o kaznenim djelima s pitanjima o prometnim prekršajima.

## Izvori

1. Kazneni zakon - Narodne novine, NN 125/2011 (početna objava): https://narodne-novine.nn.hr/clanci/sluzbeni/2011_11_125_2498.html
2. Izmjene i dopune - NN 136/2025: https://narodne-novine.nn.hr/clanci/sluzbeni/full/2025_11_136_2018.html
3. Referentni pročišćeni tekst Kaznenog zakona: https://zakoni.lexium.hr/kazneni-zakon
4. Zakon o sigurnosti prometa na cestama - Centar za vozila Hrvatske: https://cvh.hr/gradani/propisi-i-upute/zakoni/zakon-o-sigurnosti-prometa-na-cestama/

Dokumente za testiranje sam sam složio kao **sažete izvatke** zakonskih odredbi; **nisu službeni pročišćeni tekstovi**. Neke stavke i iznimke izostavljene su radi jednostavnog testiranja. Za stvarna pravna pitanja uvijek provjeri aktualne službene izvore. Testni sadržaj pregledan 17. 10. 2026. godine.

## Pravo ponovne uporabe

Prema članku 18. stavku 3. Zakona o autorskom pravu i srodnim pravima (NN 111/2021), objavljeni službeni zakonodavni tekstovi prestaju biti zaštićeni autorskim pravom pod navedenim uvjetima. Dodatni oblici i raspored ovih vlastitih testnih izvadaka mogu se koristiti za demonstracijski repozitorij, uz navođenje gore navedenih izvora. To ne znači da su logo, izgled tuđih internetskih stranica ili njihovi autorski komentari slobodni za preslikavanje.

Zakon o autorskom pravu: https://narodne-novine.nn.hr/clanci/sluzbeni/full/2021_10_111_1941.html
