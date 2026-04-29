# Piano Di Modularizzazione cyPredict

Data: 2026-04-29

## Stato Attuale

- Fase 1 completata: `cyPredict/__init__.py` e' un compatibility layer e la classe legacy vive in `cyPredict/cypredict.py`.
- Fase 2 completata in questo ciclo di lavoro: `core/state.py` contiene `StateMixin`, enum legacy, `__init__`, `track_time` e `set_start_time`.
- Fase 3.1 completata in questo ciclo di lavoro: `core/data.py` contiene `DataMixin` e `download_finance_data`.
- Fase 3.2 completata in questo ciclo di lavoro: `core/dates.py` contiene `DatesMixin`, `find_next_valid_datetime` e `datetime_dateset_extend`.
- Fase 4 completata in questo ciclo di lavoro: `core/detrending.py` contiene `DetrendingMixin`, `hp_filter`, `jh_filter`, `linear_detrend` e `detrend_lowess`.
- Fase 5 parziale completata in questo ciclo di lavoro: `core/spectral.py` contiene `SpectralMixin` e `get_bartels_score`.
- Fase 5/8 parziale completata in questo ciclo di lavoro: `core/diagnostics.py`, `core/reconstruction.py` e `core/scoring.py` contengono diagnostica, padding/ricostruzione segnali e scoring globale.
- Fase 8.1 completata in questo ciclo di lavoro: `core/indicators.py` contiene `IndicatorsMixin` e gli helper `indict_*`.
- Fase 7 parziale completata in questo ciclo di lavoro: `core/optimization.py` contiene `OptimizationMixin` e `custom_crossover`.
- Fase 8.2 parziale completata in questo ciclo di lavoro: `core/extrema.py` contiene `ExtremaMixin`, helper trade/extrema, correlazione CDC/detrended e `MultiAn_cyclesAlignKPI`.
- Fase 9.1 completata in questo ciclo di lavoro: `core/persistence.py` contiene `PersistenceMixin`, `save_dataframe` e `get_most_updated_optimization_pars`.
- La classe pubblica resta `cyPredict.cyPredict` e ora eredita da `StateMixin`, `DataMixin`, `DatesMixin`, `DetrendingMixin`, `SpectralMixin`, `DiagnosticsMixin`, `ExtremaMixin`, `IndicatorsMixin`, `OptimizationMixin`, `PersistenceMixin`, `ReconstructionMixin` e `ScoringMixin`.
- Gli import legacy sono stati mantenuti prima dell'import dei mixin: questa regola e' importante per evitare cambiamenti indiretti nell'ordine di inizializzazione delle librerie scientifiche/native. Unica eccezione verificata: l'import `yfinance` e' stato rimosso dal monolite perche' ora e' locale a `core/data.py` e il golden QQQ resta stabile.
- Gli import calendario storici (`pytz`, `timezone`, `USFederalHolidayCalendar`, `BDay`, `timedelta`, `date`) restano temporaneamente nel monolite anche se non sono referenziati direttamente: la loro rimozione ha prodotto drift golden, quindi vanno trattati solo in un commit dedicato con analisi dell'ordine di import.
- Gli import nuovi pesanti non presenti nel percorso golden, come `LinearRegression` usato da `jh_filter`, vanno preferibilmente caricati in modo lazy dentro il metodo; l'import top-level ha prodotto drift golden.
- Prossima fase consigliata: estrarre min/max, poi spostare `analyze_and_plot` intero in un mixin dedicato quando il numero di dipendenze laterali sara' piu' basso, sempre con commit separati e golden QQQ dopo ogni spostamento.

## Obiettivo

Suddividere progressivamente il file monolitico `cyPredict/__init__.py` in moduli strutturati, mantenendo invariata la logica di calcolo e lasciando stabile l'API legacy finche' i notebook applicativi e GammaSignalForge non saranno migrati alla nuova struttura.

Il principio guida e':

- refactor meccanico, non funzionale;
- piccoli commit verificabili;
- compatibilita' legacy sempre disponibile;
- golden test prima e dopo ogni spostamento rilevante;
- nessun cambio alle procedure di calcolo, ai default significativi o alla semantica dei ritorni.

## Vincoli

1. Usare sempre l'ambiente Anaconda `cyenv` per compilazione e test.
2. Non modificare i calcoli durante la modularizzazione.
3. Ogni spostamento di codice deve essere isolato in commit piccoli.
4. `from cyPredict import cyPredict` deve continuare a funzionare.
5. `import cyPredict; cyPredict.cyPredict(...)` deve continuare a funzionare.
6. I notebook devono continuare a girare con l'API legacy fino alla migrazione esplicita.
7. GammaSignalForge deve poter usare inizialmente l'API legacy e, solo in una fase successiva, l'API modulare nuova.
8. Le librerie native gia' spostate sotto `native/` non devono essere duplicate.

## Strategia Raccomandata

La strategia piu' sicura non e' spezzare subito la classe in funzioni pure. La classe contiene molto stato condiviso su `self`, quindi la prima fase deve essere una separazione meccanica tramite mixin o moduli di metodi, mantenendo il comportamento identico.

Sequenza raccomandata:

1. Creare una classe pubblica legacy che resta `cyPredict.cyPredict`.
2. Spostare gruppi di metodi in mixin tematici.
3. Far ereditare la classe pubblica dai mixin.
4. Solo dopo la stabilizzazione, estrarre funzioni pure dai mixin dove ha senso.
5. Introdurre una nuova API piu' pulita per GammaSignalForge senza rompere l'API storica.

## Struttura Target Iniziale

```text
cyPredict/
  __init__.py
  cypredict.py
  logging_utils.py
  core/
    __init__.py
    state.py
    data.py
    detrending.py
    dates.py
    spectral.py
    reconstruction.py
    multiperiod.py
    optimization.py
    indicators.py
    extrema.py
    minmax.py
    persistence.py
    plotting.py
    scoring.py
  compat/
    __init__.py
    legacy_imports.py
  services/
    __init__.py
    cycle_projection_service.py
```

Ruolo dei file:

- `cyPredict/__init__.py`: re-export legacy, piccolo e stabile.
- `cyPredict/cypredict.py`: definizione della classe pubblica `cyPredict`, costruita tramite mixin.
- `core/state.py`: `__init__`, enum ancora necessari, inizializzazione attributi e stato condiviso.
- `core/data.py`: download dati, normalizzazione dati, gestione yfinance.
- `core/dates.py`: estensione datetime e calendari.
- `core/detrending.py`: HP filter, JH filter, linear detrend, LOWESS.
- `core/spectral.py`: Goertzel, analisi frequenze, ampiezze e fasi.
- `core/reconstruction.py`: ricostruzione segnali, padding, proiezioni.
- `core/multiperiod.py`: orchestrazione multiperiod.
- `core/optimization.py`: DEAP, C++ genetic optimizer, NLopt, Hyperopt.
- `core/indicators.py`: MACD, RSI, medie centrate e derivate.
- `core/extrema.py`: minimi, massimi, allineamenti, KPI.
- `core/minmax.py`: dataset wide per min/max analysis.
- `core/persistence.py`: salvataggio CSV, recupero parametri da file.
- `core/plotting.py`: grafici Plotly e diagnostica visuale.
- `core/scoring.py`: score righe, global score, fitness wrapper.
- `services/cycle_projection_service.py`: API futura pensata per worker periodici e GammaSignalForge.

## Struttura Della Classe Durante La Prima Fase

Esempio desiderato:

```python
class cyPredict(
    StateMixin,
    DataMixin,
    DatesMixin,
    DetrendingMixin,
    SpectralMixin,
    DiagnosticsMixin,
    ReconstructionMixin,
    MultiperiodMixin,
    OptimizationMixin,
    IndicatorsMixin,
    ExtremaMixin,
    MinMaxMixin,
    PersistenceMixin,
    PlottingMixin,
    ScoringMixin,
):
    pass
```

Questa forma permette di spostare metodi senza cambiare le chiamate esistenti. E' meno elegante di una riprogettazione completa, ma riduce molto il rischio di cambiare involontariamente il calcolo.

## API Legacy Da Preservare

Fino a migrazione completata, questi contratti devono restare validi:

```python
from cyPredict import cyPredict
cp = cyPredict(...)
```

```python
import cyPredict
cp = cyPredict.cyPredict(...)
```

Metodi legacy principali da preservare:

- `download_finance_data`
- `analyze_and_plot`
- `multiperiod_analysis`
- `rebuilt_signal_zeros`
- `min_max_analysis_concatenated_dataframe`
- `get_min_max_analysis_df`
- `genOpt_cycleParsGenOptimization`
- `get_most_updated_optimization_pars`

Le firme gia' pulite non devono reintrodurre parametri rimossi.

## API Nuova Per GammaSignalForge

Solo dopo la separazione meccanica, creare un livello service-oriented. Obiettivo: un worker deve poter chiedere una proiezione periodica senza conoscere dettagli interni della classe legacy.

API candidata:

```python
from cyPredict.services import CycleProjectionService

service = CycleProjectionService(config)
result = service.run_projection(
    symbol="QQQ",
    start="2022-01-01",
    end="2024-01-01",
    timeframe="1d",
    current_date="2023-12-29",
    periods=periods_df,
)
```

Output candidato:

- dataframe elaborato;
- segnali scalati;
- prossimi minimi e massimi locali;
- metadata di configurazione;
- timestamp elaborazione;
- versione modello/codice;
- eventuale payload serializzabile per storage o message queue.

Questa API deve essere aggiunta, non sostituire subito i metodi legacy.

## Sequenza Operativa A Piccoli Commit

### Fase 0 - Baseline E Protezioni

Commit 0.1:

- verificare repo pulito;
- eseguire `py_compile`;
- eseguire `pytest`;
- eseguire golden QQQ;
- annotare commit baseline.

Comandi:

```powershell
& 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m py_compile cyPredict\__init__.py
& 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m py_compile cyPredict\cypredict.py
& 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m pytest
$env:CYPREDICT_RUN_GOLDEN='1'; & 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m pytest tests\test_golden_cypredict.py -q
```

### Fase 1 - Preparare Il Package Shell

Commit 1.1:

- creare `cyPredict/cypredict.py`;
- spostare solo la definizione pubblica finale in modo minimale;
- lasciare `cyPredict/__init__.py` come compatibility layer;
- assicurare re-export di `cyPredict`.

Verifica:

- import legacy;
- test rapidi;
- golden QQQ.

### Fase 2 - Estrarre State E Costruttore

Commit 2.1:

- creare `core/state.py`;
- spostare enum ancora necessari, inizializzazione attributi, `__init__`;
- mantenere stessi attributi e stessi default;
- nessuna modifica a nomi di attributi.

Rischio:

- gli attributi su `self` sono usati da molti metodi. Non rinominare nulla.

Verifica:

- `py_compile`;
- `pytest`;
- golden QQQ.

### Fase 3 - Estrarre Data E Date Utilities

Commit 3.1:

- creare `core/data.py`;
- spostare `download_finance_data`;
- mantenere gestione yfinance invariata.

Commit 3.2:

- creare `core/dates.py`;
- spostare `find_next_valid_datetime`;
- spostare `datetime_dateset_extend`.

Verifica:

- test import;
- golden QQQ;
- eventuale smoke notebook su QQQ o ES=F EOD.

### Fase 4 - Estrarre Detrending

Commit 4.1:

- creare `core/detrending.py`;
- spostare `hp_filter`, `jh_filter`, `linear_detrend`, `detrend_lowess`.

Vincolo:

- non toccare formule, dtype, indici o fill NaN.

Verifica:

- golden QQQ;
- aggiungere se possibile test unitario minimale su output shape e indice.

### Fase 5 - Estrarre Analisi Spettrale E Ricostruzione

Commit 5.1:

- creare `core/spectral.py`;
- spostare Goertzel wrappers e funzioni legate ad ampiezze/frequenze/fasi.

Commit 5.2:

- creare `core/reconstruction.py`;
- spostare `rebuilt_signal_zeros`;
- spostare funzioni di composizione segnali.

Vincolo:

- non cambiare ordine dei cicli, ordine colonne, padding o indici.

Verifica:

- golden QQQ;
- confronto hash dataframe e segnali.

### Fase 6 - Estrarre `analyze_and_plot`

Commit 6.1:

- creare `core/spectral_analysis.py` oppure usare `core/spectral.py`;
- spostare `analyze_and_plot` intero senza spezzarlo internamente;
- aggiornare import interni;
- lasciare firma e ritorni invariati.

Nota:

- questa funzione e' troppo centrale per un refactor funzionale immediato. Prima va solo spostata.

Verifica obbligatoria:

- `py_compile`;
- `pytest`;
- golden QQQ;
- smoke notebook QQQ/ES=F se disponibile.

### Fase 7 - Estrarre Multiperiod E Optimization

Commit 7.1:

- creare `core/multiperiod.py`;
- spostare `multiperiod_analysis` intero.

Commit 7.2:

- creare `core/optimization.py`;
- spostare funzioni DEAP, C++ GA, NLopt, Hyperopt;
- mantenere callback NLopt con parametro `grad` anche se inutilizzato.

Vincolo:

- non cambiare `multiprocessing`;
- non cambiare il modo in cui vengono chiamate DLL/native extension.

Verifica:

- test rapidi;
- scenario multiperiod piccolo con QQQ o ES=F;
- golden multiperiod da aggiungere se non ancora presente.

### Fase 8 - Estrarre Indicatori, Estremi E Min/Max

Commit 8.1:

- creare `core/indicators.py`;
- spostare MACD, RSI, medie centrate.

Commit 8.2:

- creare `core/extrema.py`;
- spostare extrema helpers e KPI allineamenti.

Commit 8.3:

- creare `core/minmax.py`;
- spostare `min_max_analysis`;
- spostare `min_max_analysis_concatenated_dataframe`;
- spostare `get_min_max_analysis_df`.

Verifica:

- aggiungere test di smoke min/max con dataset ridotto;
- non cambiare nomi colonne output.

### Fase 9 - Estrarre Persistenza E Report

Commit 9.1:

- creare `core/persistence.py`;
- spostare `save_dataframe`;
- spostare `get_most_updated_optimization_pars`.

Commit 9.2:

- creare `core/plotting.py`;
- spostare solo parti chiaramente dedicate a Plotly se gia' isolate;
- se non sono isolate, lasciare temporaneamente nel modulo chiamante.

Vincolo:

- evitare refactor invasivi dei blocchi Plotly nella prima passata.

### Fase 10 - Introdurre Service Per Worker

Commit 10.1:

- creare `services/cycle_projection_service.py`;
- wrapper sottile che usa ancora la classe legacy;
- definire input/output serializzabili.

Commit 10.2:

- aggiungere test service con QQQ EOD;
- documentare esempio worker periodico.

Output service consigliato:

```python
{
    "symbol": "QQQ",
    "timeframe": "1d",
    "current_date": "...",
    "generated_at": "...",
    "signals": ...,
    "next_extrema": ...,
    "configuration": ...,
    "diagnostics": ...,
}
```

## Aggiornamento Notebook

Fase iniziale:

- non cambiare import nei notebook;
- continuare a usare API legacy.

Fase intermedia:

- aggiornare solo notebook nuovi o di test per usare il service;
- lasciare i notebook storici su API legacy.

Fase finale:

- aggiornare notebook principali;
- marcare i notebook storici come legacy;
- rimuovere eventuali alias solo dopo conferma che GammaSignalForge non li usa.

## Compatibilita' GammaSignalForge

GammaSignalForge dovrebbe migrare in tre step:

1. usare `cyPredict.cyPredict` esattamente come oggi;
2. usare `CycleProjectionService` quando stabile;
3. usare moduli interni solo se serve davvero un controllo fine.

Non esporre subito i moduli `core` come API pubblica stabile. Considerarli interni finche' il service non e' consolidato.

## Verifiche Obbligatorie Per Ogni Commit

Minimo:

```powershell
& 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m py_compile cyPredict\__init__.py
& 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m py_compile cyPredict\cypredict.py
& 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m pytest
```

Per commit che spostano calcolo o workflow centrali:

```powershell
$env:CYPREDICT_RUN_GOLDEN='1'; & 'C:\Users\Federico\anaconda3\envs\cyenv\python.exe' -m pytest tests\test_golden_cypredict.py -q
```

Da aggiungere:

- golden multiperiod QQQ o ES=F;
- golden min/max ridotto;
- smoke test service GammaSignalForge-like.

## Regole Di Stop

Fermare il refactor e correggere prima di proseguire se:

- cambia un hash golden senza causa spiegata;
- cambia una colonna di output non prevista;
- cambia il numero di righe proiettate;
- cambia il tipo dell'indice;
- cambia un default usato dai notebook;
- una DLL/native extension non viene trovata;
- i notebook principali non importano piu' la classe.

## Rischi Principali

| Rischio | Mitigazione |
| --- | --- |
| Stato condiviso su `self` difficile da tracciare | Prima fase con mixin, senza rinominare attributi. |
| Modifica involontaria di calcolo | Spostamenti meccanici, diff piccoli, golden test. |
| Notebook rotti da import o firme | `__init__.py` come compatibility layer e API legacy stabile. |
| GammaSignalForge dipende da dettagli non documentati | Introdurre service stabile prima di cambiare API pubblica. |
| Native extension non trovate dopo spostamento | Centralizzare path setup e test `test_native_imports`. |
| Plotting mescolato a calcolo | Spostarlo solo quando il blocco e' isolabile senza modifiche funzionali. |

## Definition Of Done

La modularizzazione puo' considerarsi conclusa quando:

- `cyPredict/__init__.py` contiene solo re-export e compatibility setup minimo;
- la classe legacy resta importabile e funzionante;
- i metodi principali sono distribuiti in moduli tematici;
- i golden QQQ, multiperiod e min/max passano;
- almeno un notebook QQQ o ES=F funziona con API legacy;
- GammaSignalForge puo' usare `CycleProjectionService`;
- la documentazione indica chiaramente API legacy, API service e moduli interni.
