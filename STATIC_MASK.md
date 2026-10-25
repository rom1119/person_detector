# Czasowe maskowanie nieruchomych obiektów

Nowy moduł `lib/static_object_mask.py` jest podłączony w `main.py` bezpośrednio
przed wywołaniem YOLO i po otrzymaniu jego wyników. Dotychczasowy filtr statyczny,
obsługa kamery i wysyłanie powiadomień pozostają w obecnym układzie. Bufor alarmowy
otrzymał dodatkowe wykluczanie wcześniejszych wykryć rozpoznanych jako statyczne.

Domyślne ustawienia w konstruktorze `StaticObjectMasker` w `main.py`:

- **4 wykrycia w przesuwającym się oknie 20 s** tworzą maskę. Nie trzeba czekać
  pełnych 20 s, jeśli cztery zgodne wykrycia pojawią się wcześniej. Skrócenie okna
  z 30 do 20 s wymaga częstszych wykryć; samo nie przyspiesza czwartej obserwacji.
- **120 s** to czas życia maski, liczony od jej utworzenia. Kolejne wykrycia,
  także o większej pewności, nie przedłużają tego czasu.
- **15×15 px** to czarna łata w środku zapamiętanej ramki, w rozdzielczości
  oryginalnej klatki. Dla mniejszej ramki łata jest zmniejszana.
- Każdy obiekt ma osobną historię. Dopasowanie uwzględnia położenie i rozmiar
  ramki, toleruje drobne drgania, a jednostajny ruch nie powinien tworzyć maski.
  Domyślnie dopuszczalne rozrzuty środka wynoszą 12 px; spójne przesunięcie
  od 4 px jest traktowane jako ruch, a nie nieruchomość.
- Już zapamiętany obiekt jest rozpoznawany także po nakładaniu się ramek:
  `mask_iou_threshold=0.45`. IoU to pole części wspólnej podzielone przez pole
  sumy dwóch ramek. Pozwala to odrzucać skróconą lub przesuniętą ramkę tego samego
  obiektu, nawet gdy po nałożeniu łaty jej środek przesunął się o ponad 12 px.
  Ta luźniejsza reguła nie służy do uczenia nieruchomości. Sama maska nie podąża
  za kolejnymi ramkami i nie zwiększa swojego obszaru.
- Co **15 s** model dostaje kontrolną klatkę bez łat. Znany nieruchomy obiekt
  nadal nie trafia wtedy do bufora alarmowego. Dwie kontrole bez jego wykrycia
  usuwają maskę wcześniej. Wyraźnie inna, nakładająca się ramka na kontroli
  również ją usuwa.
- `log_events=True` wypisuje komunikat przy utworzeniu nowej maski: środek,
  ramkę, rozmiar łaty, liczbę obserwacji, czas ich zbierania i czas maskowania.
  Ten komunikat nie jest powtarzany co klatkę.

Przykład komunikatu:

```text
[STATIC_MASK] Dodano maskę: środek=(1456.5, 310.0), bbox=(1422.0, 243.0, 1491.0, 377.0), łata do 15x15 px, trafienia=4 w 1.5s (okno 20s), czas=120s
```

Przed sprawdzeniem `MIN_DETECTIONS` bufor jest czyszczony z dowodów statyczności:

- Każda zapisana klatka pamięta pozycje i pewność wykryć oraz czas obserwacji.
- Dopiero po uznaniu obiektu za statyczny jego wcześniejsze pasujące ramki,
  od początku zebranej historii tego obiektu, są wykluczane z licznika alarmu.
- Klatka zawierająca wyłącznie taki obiekt przestaje się liczyć. Jeżeli była na
  niej również inna osoba, nadal liczy się raz; pewność odnosi się wtedy do
  pozostałych wykryć. Nie odejmujemy jednego globalnego licznika obiektów.
- Czyszczenie następuje po każdej predykcji, również gdy po filtrowaniu nie
  zostało nic do zapisania. Usunięte z licznika wykrycia nie wracają po wygaśnięciu
  maski. `MIN_DETECTIONS=7` dotyczy już tylko pozostałych klatek.
- Zdjęcia wykluczonych klatek pozostają w katalogach bufora do przeglądu.
  Nie są przekazywane na Telegram. Ta zmiana nie dodaje automatycznego usuwania
  zdjęć z dysku. Istniejące adnotacje starszych klatek nie są rysowane ponownie;
  mieszana klatka może nadal pokazywać ramkę później uznaną za statyczną.

Po wykluczeniu wcześniejszych klatek pojawi się również komunikat
`[STATIC_MASK] Wykluczono z alarmu ... wcześniejszych klatek; pozostało .../7.`

Łaty powstają na kopii obrazu podawanej do modelu. Przy obecnym zapisie wykryć:

- `files/buffer` → `files/clear`: oryginał bez maski i bez ramek YOLO.
- `files/buff_annot` → `files/annotated`: obraz podany do YOLO, z zastosowaną
  czarną maską i ramkami zachowanych wykryć.
- Telegram otrzymuje zdjęcie z `files/annotated`, czyli również z widoczną maską.

Klatka kontrolna nie ma łat, bo model dostaje wtedy oryginał. Jej zapis z ramkami
i zdjęcie na Telegramie również nie mają maski. Nowa maska utworzona na podstawie
bieżącego wyniku pojawia się dopiero w kolejnej maskowanej predykcji. W ten sposób
zapis zawsze pokazuje obraz faktycznie użyty przez YOLO. Gdy wszystkie wykrycia
zostaną odrzucone, obecna logika nie zapisuje klatki do bufora ani nie wysyła alarmu.

Ponieważ mała czarna łata nie gwarantuje zniknięcia detekcji YOLO,
moduł dodatkowo odrzuca ramkę zgodną
z zapamiętanym obiektem przez czas działania maski. Detekcje o innym położeniu
lub rozmiarze mogą przejść dalej, jeżeli nie spełniają żadnej z reguł dopasowania
do aktywnej maski.

To heurystyka nieruchomości, nie rozpoznawanie, czy obiekt naprawdę jest człowiekiem.
Nieruchoma osoba może również zostać zamaskowana. Kontrole i ograniczony czas
zmniejszają ten problem, ale go nie eliminują. Zbyt duże tolerancje położenia
mogą łączyć sąsiednie obiekty. Parametry wymagają oceny na nagraniu z tej kamery;
sam test syntetyczny nie potwierdza skuteczności na deszczu lub owadach.

Wszystkie parametry można zmienić w jednym miejscu w `main.py`,
w wywołaniu `StaticObjectMasker(...)`. Każdy ma komentarz z opisem i jednostką.
Wartości podane tutaj mają pierwszeństwo przed domyślnymi wartościami w klasie.
Komentarze są też bezpośrednio przy parametrach konstruktora w module.

Większe `center_tolerance` i `size_tolerance` pozwalają na silniejsze drgania
ramki, ale zwiększają ryzyko połączenia różnych obiektów. Większe
`movement_threshold` ułatwia uznanie powolnego ruchu za nieruchomość.
Większe `required_hits` wymaga więcej obserwacji przed zamaskowaniem.
Mniejsze `mask_iou_threshold` toleruje większe różnice ramki już zapamiętanego
obiektu, ale zwiększa ryzyko wyciszenia innej osoby wchodzącej w ten sam obszar.
Dlatego próg ruchu podczas uczenia pozostaje bez zmian. Nieruchomości nie
ustalamy na podstawie samej pewności YOLO, która może wzrosnąć przez owada.

Liczba wywołań YOLO pozostaje taka sama: kontrola bez maski zastępuje zwykłą
predykcję, a analiza nakładania ramek i bufora nie uruchamia modelu ponownie.

Testy bez kamery i Telegrama:

```bash
cd /workspace/person_detector
venv/bin/python -m unittest discover -s tests -v
```

Testy obejmują krótsze/przesunięte ramki, wzrost pewności wykrycia, cofanie
wcześniejszych zliczeń, klatkę ze statycznym obiektem i inną osobą, rzeczywisty
próg siedmiu klatek w funkcji bufora, zapis czystego i maskowanego JPEG-a oraz
wybór pliku do kolejki Telegrama. Kamera i wysyłka sieciowa nie są uruchamiane.
