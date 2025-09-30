# Matematika-DAS 🚀

Ovaj repozitorijum predstavlja istraživački projekat sa doktorskih studija, Matematika 1 i 2, koji proučava **destilaciju znanja** iz modela mašinskog učenja u **pravila prvog reda** koristeći **Aleph ILP** sistem.  
Repozitorijum sadrži:

* 🐍 Python skripte za treniranje, izvoz i evaluaciju modela *(Decision Tree, Random Forest, XGBoost)*
* 🧠 Prolog/Aleph tokove koji indukuju logička pravila iz treniranih modela
* 📄 LaTeX izveštaj (`seminarski/`) sa svim poglavljima, slikama i bibliografijom
* 🐳 Dockerfile za reproduktibilno okruženje *(Python 3.12, SWI-Prolog, Aleph, ML biblioteke)*

> **Napomena:** Većina koda, dokumentacije i izveštaja je napisana uz pomoć LLM-ova (ChatGPT & GitHub Copilot).

---

## Struktura repozitorijuma 🗂️

```text
.
├── .gitignore
├── Dockerfile
├── README.md                ← **ovde ste**
├── requirements.txt
├── scripts/
│   ├── full_table.py
│   ├── main.py
│   ├── obrada.py
│   ├── presets.py
│   ├── rule_check.py
│   ├── rules.py
│   └── train_and_export.py
└── seminarski/
        ├── biblist.bib
        ├── images/
        │   ├── UCT.jpg
        │   └── charts/… (PNG fajlovi)
        ├── kodovi/
        │   ├── helloworld.c
        │   ├── helloworld.cpp
        │   └── helloworld.py
        ├── main.tex
        ├── poglavlja/
        │   ├── abstrakt.tex
        │   ├── dodatci.tex
        │   ├── listinzi.tex
        │   ├── naslovna_strana.tex
        │   ├── preamble.tex
        │   ├── sadrzaj.tex
        │   ├── telo/
        │   │   ├── poglavlje1-teorija.tex
        │   │   ├── poglavlje2-tehnologije.tex
        │   │   └── poglavlje3-implementacija.tex
        │   ├── uvod.tex
        │   └── zakljucak.tex
        ├── prezentacija.md
        └── rsvp.sty
```

> **Napomena:** *`seminarski/`* sadrži kompletan LaTeX izvor sa slikama i bibliografijom, a *`scripts/`* implementira end-to-end tok rada.

---

## Preduslovi ⚙️

* **Python 3.12** (ili noviji)
* **SWI-Prolog** (≥ 8.0) sa instaliranim **Aleph** paketom
* 📦 Standardne ML biblioteke iz `requirements.txt`

Ako želite izolovano okruženje, koristite priloženu Docker sliku, odnosno `Dockerfile`.

---

## Instalacija 🛠️

1. **Klonirajte repozitorijum**

     ```bash
     git clone https://github.com/ristic-ac/Matematika-DAS.git
     cd Matematika-DAS
     ```

2. **Izgradite Docker sliku**

    ```bash
    docker build -t das-matematika:latest .
    ```

3. **Pokrenite kontejner sa bind-mount direktorijumima**

    ```bash
    docker run -v "$(pwd)/scripts:/app/scripts" \
             -v "$(pwd)/outputs:/app/outputs" \
             -v "$(pwd)/cache:/app/cache" \
             -it das-matematika:latest
    ```

4. **Pokrenite main.py unutar kontejnera**

    ```bash
    python scripts/main.py [model] [akcija] [dataset] [mode_index]
    ```

---

## Upotreba 🏃

Sve glavne akcije pokreću se preko `scripts/main.py`.  
Forma komande:

```bash
python scripts/main.py <model> <akcija> <dataset> [mode_index]
```

| Argument       | Značenje                                                               |
| -------------- | ---------------------------------------------------------------------- |
| `<model>`      | `dt`, `rf`, `xgb` ili `all`                                            |
| `<akcija>`     | `train` – treniraj i izvezi; `aleph` – samo Aleph; `both` – oba koraka |
| `<dataset>`    | `mushroom` ili `adult` (OpenML)                                        |
| `[mode_index]` | Opciono: konkretan Aleph režim (npr. `sniper`)                         |

### Treniranje i izvoz modela 🏗️

```bash
# Treniraj sva tri modela na Mushroom skupu i izvezi Aleph fajlove
python scripts/main.py all both mushroom
```

Kreira se hijerarhija:

```text
outputs/
└── mushroom/
        ├── dt/
        │   └── <mode_index>/   # npr. sniper, sweet_spot, sweeper
        ├── rf/
        └── xgb/
```

U svakom `<mode_index>` direktorijumu nalazi se:

* `<dataset>.pl` – Prolog program (pozadinsko znanje + mode deklaracije)
* `<dataset>_test.f` / `<dataset>_test.n` – pozitivni/negativni test primeri
* `<dataset>_hypothesis.pl` – indukovan skup pravila

### Pokretanje Aleph-a 🧩

```bash
python scripts/main.py dt aleph mushroom sniper
```

Indukovana pravila se ispisuju u konzoli i čuvaju kao `<dataset>_hypothesis.pl`.  
Po završetku se automatski pokreće evaluacija (`rule_check.py`) – *fidelity*, tačnost i dr.

### Evaluacija i grafikoni 📊

Za objedinjeni izveštaj preko svih eksperimenata:

```bash
python scripts/obrada.py
```

Rezultat: `all_results.json`.

### Generisanje LaTeX izveštaja 📝

1. **Kompilacija**

     ```bash
     cd seminarski
     latexmk -xelatex --shell-escape -synctex=1 -interaction=nonstopmode -file-line-error main.tex
     ```

     Dobija se `main.pdf` (finalni izveštaj). Sve slike su u `seminarski/images/`.

---

## Doprinos 🤝

1. Forkujte repozitorijum
2. Napravite feature granu: `git checkout -b feat/awesome-feature`
3. Komitujte promene sa jasnim porukama
4. Otvorite Pull Request ka `main`

> **Molba:** zadržite postojeću strukturu i ažurirajte LaTeX izveštaj ako dodate nove eksperimente ili figure.

---

## Licenca 📄

Ovaj projekat trenutno nema zvaničnu licencu.  
Za korišćenje ili distribuciju, kontaktirajte autora.

---

*💡 Srećno sa eksperimentima! 🍀*
