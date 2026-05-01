# Piano di pulizia e consolidamento cyPredict

Data: 2026-04-29  
Baseline Git verificata prima del piano: `427250c57f813dc80987c407211f69f867ac6710` su `origin/main`  
Ambito: pulizia, documentazione, logging, struttura progetto, import delle librerie native e preparazione all'uso periodico da worker in GammaSignalForge.

## Obiettivo

Rendere `cyPredict` mantenibile, documentato e integrabile in processi automatici senza alterare le procedure di calcolo dei cicli dominanti, della trasformata di Goertzel, del detrending, dei filtri, dell'ottimizzazione delle ampiezze/frequenze/fasi e della ricostruzione dei segnali.

La regola operativa principale e':

> ogni modifica deve preservare l'output numerico rispetto alla baseline, salvo modifiche esplicitamente approvate.

## Contesto osservato

Repository attuale:

- `cyPredict/cypredict.py`: modulo monolitico con la classe `cyPredict`, separato dal re-export legacy in `cyPredict/__init__.py`.
- `README.md`: vuoto.
- `.gitingore`: file vuoto e con nome errato; dovrebbe essere `.gitignore`.
- `examples/`: presente ma non popolata con esempi eseguibili nella ricognizione iniziale.
- Cartelle generate non tracciate: `cyPredict/__pycache__`, `cyPredict/.ipynb_checkpoints`.

Uso reale osservato nei notebook in:

- `D:\Dropbox\TRADING\STUDIES DEVELOPMENT\CYCLES ANALYSIS\cyPredict - V3 - ES=F - Intraday.ipynb`
- `D:\Dropbox\TRADING\STUDIES DEVELOPMENT\CYCLES ANALYSIS\cyPredict - V3 - NQ=F.ipynb`
- `D:\Dropbox\TRADING\STUDIES DEVELOPMENT\CYCLES ANALYSIS\cyPredict - V3 - GC=F - Intraday.ipynb`
- `D:\Dropbox\TRADING\STUDIES DEVELOPMENT\CYCLES ANALYSIS\cyPredict - V2 - SPY.ipynb`
- report e test storici nella stessa cartella.

Funzioni piu' rilevanti usate dai notebook:

- `cyPredict(...)`
- `analyze_and_plot(...)`
- `multiperiod_analysis(...)`
- `get_most_updated_optimization_pars(...)`
- `get_min_max_analysis_df(...)`
- `min_max_analysis_concatenated_dataframe(...)`

Problema gia' visibile nei notebook: alcune celle passano parametri non piu' accettati da firme aggiornate, ad esempio `CDC_bb_analysis`, `CDC_RSI_analysis`, `CDC_MACD_analysis` su `multiperiod_analysis`. Questo indica che la pulizia delle firme deve partire da una matrice "firma attuale vs uso reale", non da rimozioni automatiche.

Librerie native/custom osservate fuori dalla repo:

- `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\GOERTZEL TRANSFORM C`
- `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\cyFitness`
- `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\cyGAopt`
- `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\cyGAoptMulticore`
- `D:\Dropbox\TRADING\TRADING PROPERTY LIBRARIES\GeneticOptimization`

Moduli importati da `cyPredict`:

- `goertzel`
- `cyfitness`
- `cyGAopt`
- `cyGAoptMultiCore`

## Vincoli

1. Non modificare le formule, le finestre, il flusso di ottimizzazione o le logiche di selezione dei cicli senza test di equivalenza e approvazione.
2. Separare pulizia strutturale da variazioni funzionali.
3. Ogni rimozione di parametro deve essere motivata da:
   - assenza di uso nel corpo funzione,
   - assenza di uso nei notebook/applicazioni,
   - assenza di dipendenza implicita via attributi di istanza,
   - test di non regressione.
4. Le funzioni pubbliche usate dai notebook devono avere una fase transitoria se la firma cambia.
5. Gli output usati a valle devono restare disponibili, anche se affiancati da strutture piu' leggibili.

## Strategia generale

Procedere a milestone piccole, ciascuna verificabile:

1. Congelare comportamento e dati di riferimento.
2. Pulire packaging, ignore file e struttura minima.
3. Mappare API, parametri, ritorni e side effect.
4. Centralizzare logging senza cambiare calcoli.
5. Documentare ogni funzione.
6. Rimuovere codice morto e parametri non usati solo dopo verifica.
7. Spostare librerie native/custom dentro una struttura di progetto riproducibile.
8. Esporre un'interfaccia pulita per worker periodici e GammaSignalForge.

## Milestone 0 - Baseline numerica e sicurezza

### Scopo

Creare una base oggettiva per dimostrare che le modifiche di pulizia non cambiano i risultati.

### Attivita'

1. Creare un branch dedicato, ad esempio `cleanup/cypredict-structure-docs`.
2. Salvare la baseline commit usata per i confronti: `427250c57f813dc80987c407211f69f867ac6710`.
3. Identificare 4 scenari minimi:
   - daily su ETF/equity, ad esempio SPY.
   - intraday su future, ad esempio ES=F.
   - intraday su future con dati da file CSV.
   - multirange con ottimizzazione C++ delle ampiezze/frequenze/fasi.
4. Estrarre dai notebook celle riproducibili e convertirle in script di test controllati.
5. Per ogni scenario salvare:
   - input dati o riferimento a file dati stabile;
   - parametri completi;
   - hash/versione dei moduli nativi;
   - output chiave.
6. Definire output chiave da confrontare:
   - `current_date`;
   - `index_of_max_time_for_cd`;
   - periodi dominanti selezionati;
   - ampiezze/frequenze/fasi ottimizzate;
   - `best_fitness_value`;
   - `composite_signal`;
   - `scaled_signals`;
   - dataframe generati da `min_max_analysis_concatenated_dataframe`;
   - CSV incrementale generato da `get_min_max_analysis_df`.
7. Definire tolleranze:
   - uguaglianza esatta per date, colonne, dimensioni, tipi e ordinamenti;
   - tolleranza numerica stretta per float, ad esempio `rtol=1e-10`, `atol=1e-12`, da adeguare solo se C++/multiprocessing introduce non determinismo;
   - tolleranza separata per ottimizzazioni genetiche se non deterministiche, con seed obbligatorio.

### Deliverable

- `tests/golden/` con fixture e output baseline.
- `tests/test_golden_cypredict.py`.
- Documento `docs/baseline_scenarios.md`.

### Criterio di uscita

Almeno uno scenario daily e uno intraday devono passare prima di qualunque rimozione di codice.

## Milestone 1 - Igiene repository e packaging minimo

### Scopo

Rendere la repo clonabile, importabile e pulita senza toccare il comportamento.

### Attivita'

1. Rinominare `.gitingore` in `.gitignore`.
2. Aggiungere esclusioni:
   - `__pycache__/`
   - `.ipynb_checkpoints/`
   - `*.pyc`
   - `build/`
   - `dist/`
   - `*.egg-info/`
   - log e output generati.
3. Aggiungere `pyproject.toml` o `setup.cfg` minimo per installare il pacchetto.
4. Spostare codice Python sotto una struttura standard:
   - opzione conservativa iniziale: mantenere `cyPredict/__init__.py` ma aggiungere file modulari nuovi;
   - opzione finale: `src/cypredict/`.
5. Aggiungere `README.md` con:
   - scopo della libreria;
   - installazione;
   - dipendenze native;
   - esempio minimo;
   - avviso che non e' consulenza finanziaria.
6. Aggiungere `requirements.txt` o gruppi dipendenze in `pyproject.toml`.
7. Separare esempi da notebook di ricerca:
   - `examples/` per script brevi e riproducibili;
   - `notebooks/` solo se si decide di versionare notebook puliti;
   - notebook sperimentali grandi fuori dalla libreria o gestiti con DVC/Git LFS se necessario.

### Deliverable

- `.gitignore` corretto.
- `pyproject.toml`.
- `README.md` iniziale.
- Struttura cartelle documentata.

### Criterio di uscita

`python -m pip install -e .` deve permettere `import cypredict` o l'import legacy deciso.

### Stato aggiornato

Milestone 1 chiusa sul criterio legacy deciso: `python -m pip install -e .`
in ambiente `cyenv` completa correttamente e permette `from cyPredict import
cyPredict`. La repo contiene `.gitignore`, `pyproject.toml`, README, cartelle
`docs/`, `scripts/`, `tests/`, `examples/` e `native/`. Gli esempi minimi sono
separati dai notebook di ricerca: `examples/minimal_import.py` non scarica dati
e verifica solo import e path native. Lo script `scripts/verify_packaging.ps1`
rende ripetibile il controllo install/editable/import.

## Milestone 2 - Mappa API, parametri e uso reale

### Scopo

Capire cosa e' pubblico, cosa e' interno e quali parametri sono realmente attivi.

### Attivita'

1. Generare una tabella per ogni funzione:
   - nome;
   - riga attuale;
   - tipo: pubblica, interna, debug, legacy;
   - parametri;
   - default;
   - parametri letti;
   - parametri passati ad altre funzioni;
   - attributi `self.*` letti/scritti;
   - ritorni;
   - side effect: modifica `self.data`, stampa, file, grafici, multiprocessing.
2. Incrociare la tabella con i notebook.
3. Marcare ogni parametro come:
   - `active`: usato nel calcolo;
   - `routing`: passato ad altra funzione;
   - `mode-specific`: usato solo per un algoritmo/modalita';
   - `legacy`: presente per vecchi notebook;
   - `unused-candidate`: non usato ma da verificare;
   - `remove-approved`: eliminabile dopo test.
4. Identificare parametri che diventano non significativi in base ad altri:
   - `kaiser_beta` significativo solo se `windowing == "kaiser"`;
   - `lowess_k` significativo solo se `detrend_type == "lowess"`;
   - `linear_filter_window_size_multiplier` significativo solo con detrend lineare o modalita' che lo usa;
   - `period_related_rebuild_multiplier` significativo solo se `period_related_rebuild_range == True`;
   - `frequencies_ft` e `phases_ft` significativi solo per algoritmi che ottimizzano frequenze/fasi;
   - `show_charts` significativo solo nei percorsi con plotting;
   - `enabled_multiprocessing` significativo solo per algoritmi che hanno ramo parallelizzato.
5. Decidere se mantenere compatibilita' con i parametri visti nei notebook ma rimossi/commentati:
   - `time_zone`;
   - `CDC_bb_analysis`;
   - `CDC_RSI_analysis`;
   - `CDC_MACD_analysis`;
   - eventuali alias storici.

### Parametri verificati e rimossi

Analisi statica e controllo dei notebook hanno confermato la rimozione dei seguenti parametri senza modificare le procedure di calcolo:

- `analyze_and_plot`: `include_calibrated_MACD`, `include_calibrated_RSI`, `indicators_signal_calcualtion`, `enabled_multiprocessing`.
- `multiperiod_analysis`: `pars_from_opt_file`, `files_path_name`, `bb_delta_fixed_periods`, `bb_delta_sg_filter_window`, `RSI_cycles_analysis_type`.
- `indict_MACD_SGMACD`: `signals_results`.
- `indict_RSI_SG_smooth_RSI`: `signals_results`.
- `indict_centered_average_deltas`: `signals_results`.
- `rebuilt_signal_zeros`: `debug`.
- `CDC_vs_detrended_correlation`: `data`, `lowess_k`, `best_fit_start_back_period`.
- `CDC_vs_detrended_correlation_sum`: `best_fit_start_back_period`.
- `trade_predicted_dominant_cicles_peaks`: `data`.
- `min_max_analysis_concatenated_dataframe`: `pars_from_opt_file`, `files_path_name`, `bb_delta_fixed_periods`, `bb_delta_sg_filter_window`, `RSI_cycles_analysis_type`, `show_charts`.
- `get_min_max_analysis_df`: `source_type`, `data_column_name`, `GoogleDriveMountPoint`, `index_column_name`.

`enabled_multiprocessing` resta attivo nelle funzioni dove controlla davvero rami parallelizzati.

### Deliverable

- `docs/api_inventory.md`.
- `docs/parameter_matrix.md`.
- Issue/lista di rimozioni approvate.

### Criterio di uscita

Nessun parametro viene rimosso senza una riga nella matrice e senza test passati.

### Stato aggiornato

Milestone 2 chiusa: `docs/api_inventory.md` descrive la superficie corrente
per moduli/mixin, `docs/parameter_matrix.md` non contiene piu' candidati
aperti e distingue i parametri rimossi da quelli mode-specific mantenuti. Il
test `tests/test_api_parameter_contracts.py` blocca la reintroduzione dei
parametri legacy rimossi e segnala qualunque parametro core non letto, con la
sola eccezione ammessa del parametro `grad` richiesto dalla callback NLopt.

## Milestone 3 - Stabilizzazione dell'interfaccia pubblica

### Scopo

Ridurre firme e ritorni senza rompere notebook e integrazioni operative.

### Attivita'

1. Definire API pubblica supportata:
   - costruttore;
   - `analyze_and_plot`;
   - `multiperiod_analysis`;
   - `get_min_max_analysis_df`;
   - `get_most_updated_optimization_pars`;
   - eventuale nuova API worker-oriented.
2. Separare funzioni interne con prefisso `_` solo dopo aver verificato che non siano usate nei notebook.
3. Per le firme grandi, introdurre configurazioni strutturate:
   - `DataConfig`;
   - `DetrendConfig`;
   - `GoertzelConfig`;
   - `OptimizationConfig`;
   - `ProjectionConfig`;
   - `OutputConfig`.
4. Mantenere temporaneamente wrapper legacy:
   - accettano la firma vecchia;
   - costruiscono i config object;
   - emettono warning controllati;
   - chiamano la nuova funzione interna.
5. Rimuovere i parametri inutilizzati solo dopo una release interna di transizione.
6. Correggere typo solo con alias:
   - `cicles` -> `cycles`;
   - `indict` -> `indicator`;
   - `calcualtion` -> `calculation`;
   - `weigthed` -> `weighted`;
   - `gloabl` -> `global`.

### Deliverable

- API pubblica documentata.
- Config dataclass o TypedDict.
- Wrapper legacy con warning.

### Criterio di uscita

I notebook principali devono poter essere aggiornati con sostituzioni minime e tracciabili.

### Stato aggiornato

Milestone 3 chiusa come fase di transizione: la superficie legacy resta
supportata, ma sono disponibili i config object in `cyPredict.config` e i metodi
`analyze_and_plot_from_config` e `multiperiod_analysis_from_config`. Questi
metodi espandono i config nelle firme legacy senza modificare la procedura di
calcolo. La documentazione operativa e' in `docs/public_api.md`; i test di
equivalenza della traduzione config -> kwargs legacy sono in
`tests/test_config_objects.py`.

## Milestone 4 - Ritorni e oggetti risultato

### Scopo

Rendere chiaro cosa viene restituito, cosa e' necessario e cosa e' solo diagnostico.

### Stato attuale da verificare

Ritorni principali osservati:

- `analyze_and_plot`: `current_date`, `index_of_max_time_for_cd`, `original_data`, `signals_results`, `configuration`; in errore ritorna tuple di `None`.
- `multiperiod_analysis`: `elaborated_data_series`, `signals_results_series`, `composite_signal`, `configurations_series`, `None`, `None`, `index_of_max_time_for_cd`, `scaled_signals`, `best_fitness_value`.
- `min_max_analysis_concatenated_dataframe`: dataframe con dati base, KPI, min/max e fitness.
- `get_min_max_analysis_df`: dataframe incrementale salvato anche su CSV.

### Attivita'

1. Documentare i ritorni reali per ogni funzione, inclusi casi di errore.
2. Identificare elementi sempre `None` o non piu' utili.
3. Introdurre oggetti risultato leggibili:
   - `AnalysisResult`;
   - `MultiPeriodResult`;
   - `MinMaxAnalysisResult`.
4. Mantenere compatibilita' con tuple legacy:
   - metodo `.as_legacy_tuple()`;
   - oppure parametro `return_format="legacy" | "object"`.
5. Separare output operativi da diagnostici:
   - operativi: segnali, date, massimi/minimi futuri, fitness;
   - diagnostici: grafici, tabelle intermedie, timing, debug.
6. Definire output minimo per GammaSignalForge:
   - asset/ticker;
   - timeframe;
   - timestamp elaborazione;
   - current bar timestamp;
   - lista proiezioni future;
   - tipo evento previsto: massimo/minimo;
   - score/confidenza;
   - parametri usati;
   - versione modello/codice;
   - eventuali warning.

### Deliverable

- `docs/result_contracts.md`.
- Classi risultato o schema TypedDict.
- Test che confrontano tuple legacy e nuovi oggetti.

### Criterio di uscita

Ogni funzione pubblica ha ritorno documentato e stabile.

### Stato aggiornato

Milestone 4 chiusa come fase di transizione: i ritorni legacy non cambiano, ma
sono disponibili `AnalysisResult`, `MultiPeriodResult` e `MinMaxAnalysisResult`
in `cyPredict.results` e dal package root. I wrapper
`analyze_and_plot_result`, `multiperiod_analysis_result`,
`min_max_analysis_concatenated_dataframe_result` e
`get_min_max_analysis_df_result` chiamano i metodi legacy e impacchettano il
risultato. I test in `tests/test_result_objects.py` verificano la conversione
da/a tuple o dataframe legacy, incluso il ramo di errore a sei valori di
`multiperiod_analysis`.

## Milestone 5 - Logging centralizzato

### Scopo

Sostituire `print(...)` diffusi con un logging strutturato, filtrabile e salvabile su file.

### Attivita'

1. Creare un modulo, ad esempio `cypredict/logging.py`.
2. Definire una funzione o classe centrale:
   - `log_event(message, category, function, level="INFO", **context)`;
   - oppure `CyPredictLogger`.
3. Campi minimi:
   - `timestamp`;
   - `run_id`;
   - `ticker`;
   - `timeframe`;
   - `function`;
   - `category`;
   - `level`;
   - `message`;
   - `elapsed_seconds`;
   - `context`.
4. Categorie consigliate:
   - `processing`;
   - `debug`;
   - `timing`;
   - `calculation`;
   - `data`;
   - `io`;
   - `optimization`;
   - `warning`;
   - `error`.
5. Output:
   - console;
   - file `.log` human-readable;
   - file `.jsonl` per ingestion automatica;
   - opzionale CSV per analisi rapida.
6. Naming file:
   - `logs/cypredict_{ticker}_{timeframe}_{YYYYMMDD_HHMMSS}_{run_id}.log`;
   - `logs/cypredict_{ticker}_{timeframe}_{YYYYMMDD_HHMMSS}_{run_id}.jsonl`.
7. Parametri di controllo:
   - `log_level`;
   - `log_to_console`;
   - `log_to_file`;
   - `log_dir`;
   - `run_id`;
   - rimozione dei flag legacy `time_tracking` e `print_activity_remarks` dopo migrazione al logger strutturato.
8. Sostituire progressivamente `print`:
   - prima messaggi di errore/timing;
   - poi messaggi di processo;
   - infine debug.
9. Per i notebook, mantenere output console compatibile ma governato dal log level.

### Deliverable

- Modulo logging.
- Test su formato log.
- Documentazione esempi.

### Criterio di uscita

Nessuna stampa operativa non controllata nelle funzioni pubbliche, salvo output richiesto esplicitamente dall'utente.

## Milestone 6 - Docstring esaustive

### Scopo

Rendere ogni funzione comprensibile senza leggere l'intero corpo.

### Standard

Usare stile NumPy docstring o Google docstring in modo coerente. Ogni funzione deve includere:

- descrizione sintetica;
- contesto d'uso;
- parametri;
- interazioni tra parametri;
- parametri ignorati o significativi solo in certe modalita';
- ritorni;
- eccezioni;
- side effect;
- note numeriche;
- esempi minimi;
- esempi per casi avanzati se funzione pubblica.

### Priorita'

1. Funzioni pubbliche:
   - `__init__`;
   - `download_finance_data`;
   - `analyze_and_plot`;
   - `multiperiod_analysis`;
   - `get_min_max_analysis_df`;
   - `get_most_updated_optimization_pars`;
   - `genOpt_cycleParsGenOptimization`.
2. Funzioni di calcolo:
   - `hp_filter`;
   - `jh_filter`;
   - `linear_detrend`;
   - `detrend_lowess`;
   - `get_bartels_score`;
   - `cicles_composite_signals`;
   - `rebuilt_signal_zeros`;
   - `MultiAn_evaluateFitness`;
   - `MultiAn_evaluateFitness_py`;
   - `MultiAn_optimize_NLOPT`;
   - `MultiAn_cyclesAlignKPI`.
3. Funzioni di analisi trading/KPI:
   - `min_max_analysis`;
   - `trade_predicted_dominant_cicles_peaks`;
   - `CDC_vs_detrended_correlation`;
   - `CDC_vs_detrended_correlation_sum`.
4. Funzioni debug/legacy.

### Esempio di contenuto richiesto per parametri condizionali

Per `windowing`:

- `None`: nessuna finestratura prima della trasformata.
- `"kaiser"`: applica finestra Kaiser; rende `kaiser_beta` significativo.
- altri valori: definire comportamento o validazione.

Per `period_related_rebuild_range`:

- `False`: il segnale ricostruito usa l'intera finestra prevista.
- `True`: il contributo di ogni ciclo viene limitato in base al suo periodo e a `period_related_rebuild_multiplier`.
- se `False`, `period_related_rebuild_multiplier` non ha effetto e la docstring deve dirlo.

### Deliverable

- Docstring complete.
- `docs/api_reference.md` generato o scritto.

### Criterio di uscita

Ogni funzione ha docstring non vuota; le funzioni pubbliche hanno esempi eseguibili.

### Stato aggiornato

Milestone 6 chiusa a livello di copertura documentale: tutte le classi e
funzioni nei moduli mantenuti `cyPredict/core/*.py`, `cyPredict/config.py`,
`cyPredict/results.py` e `cyPredict/logging_utils.py` hanno una docstring non
vuota. Le funzioni pubbliche principali hanno esempi in `docs/public_api.md` e
`docs/api_reference.md`; i dettagli di ritorni, parametri condizionali,
logging e result object sono documentati nei file `docs/*`. Il test
`tests/test_docstring_contracts.py` impedisce nuove funzioni/classi core prive
di docstring.

## Milestone 7 - Rimozione codice morto e ridondanze

### Scopo

Ridurre rumore e rischio senza cambiare procedure.

### Attivita'

1. Rimuovere import duplicati o non usati dopo verifica:
   - import duplicati osservati: `random`, `StandardScaler`, `tabulate`, `euclidean`;
   - import commentati e vecchi, se non utili a documentare fallback.
2. Rimuovere blocchi di codice commentato lunghi dopo:
   - salvataggio in Git;
   - verifica che non siano documentazione utile;
   - eventuale spostamento in `docs/legacy_notes.md` se contengono logica storica importante.
3. Separare debug sperimentale da codice produttivo.
4. Eliminare parametri non usati solo se approvati dalla matrice.
5. Convertire costanti magiche in configurazioni nominate.
6. Rimuovere rami irraggiungibili o sostituirli con errori espliciti.
7. Validare input all'inizio delle funzioni pubbliche.
8. Isolare gestione notebook/Plotly/IPython dal calcolo.

### Deliverable

- PR/commit piccoli per categoria:
  - import;
  - commenti storici;
  - parametri;
  - logging;
  - docstring.

### Criterio di uscita

Test golden invariati dopo ogni gruppo di rimozioni.

### Stato aggiornato

Milestone 7 chiusa per la prima passata sicura: rimossi import duplicati/non
usati rimasti dal file monolitico `cyPredict/cypredict.py` e candidati non
referenziati in `core/analysis.py` e `core/multiperiod.py`, rimossa la doppia
assegnazione legacy `cyPredict.cyPredict = cyPredict`, eliminati commenti
inline di vecchio codice in `core/analysis.py` e `core/multiperiod.py`, e
rinominato un commento debug in `core/optimization.py` senza modificare i
calcoli. Verifiche eseguite: controllo AST sugli import, compileall, test
mirati import/native/docstring, suite completa standard e golden test QQQ EOD
con `CYPREDICT_RUN_GOLDEN=1`.

## Milestone 8 - Modularizzazione senza cambiare calcoli

### Scopo

Spezzare il file monolitico in moduli coerenti.

### Struttura proposta

```text
cyPredict/
  pyproject.toml
  README.md
  docs/
    api_inventory.md
    parameter_matrix.md
    result_contracts.md
    baseline_scenarios.md
    planned_development/
  examples/
    daily_basic.py
    intraday_from_file.py
    multiperiod_projection.py
  src/
    cypredict/
      __init__.py
      core.py
      data.py
      detrending.py
      goertzel.py
      optimization.py
      signals.py
      extrema.py
      indicators.py
      plotting.py
      logging.py
      config.py
      results.py
      native/
        README.md
        goertzel/
        cyfitness/
        cygaopt/
        cygaopt_multicore/
  tests/
    golden/
    test_detrending.py
    test_goertzel_contract.py
    test_multiperiod_golden.py
    test_min_max_outputs.py
  scripts/
    build_native.ps1
    run_golden_tests.ps1
  logs/
  outputs/
```

### Sequenza consigliata

1. Creare moduli vuoti e spostare solo helper non critici.
2. Spostare logging/config/result object.
3. Spostare data loading.
4. Spostare detrending.
5. Spostare plotting.
6. Lasciare inizialmente `analyze_and_plot` e `multiperiod_analysis` in `core.py`.
7. Solo dopo test, separare ottimizzazione e Goertzel.

### Stato aggiornato

Milestone 8 chiusa come modularizzazione compatibile: la logica e' spezzata in
mixin sotto `cyPredict/core/`, il plotting notebook resta isolato in
`core/plotting.py` mantenendo l'output legacy, e il file
`cyPredict/cypredict.py` contiene solo composizione della classe e guardie
native. Per l'import pubblico pulito e' stata aggiunta la facade lowercase
`cypredict.py`, che espone `CyPredict` come alias della classe legacy senza
spostare la cartella `cyPredict/` su Windows. La futura visualizzazione web di
GammaSignalForge non e' implementata qui: dovra' riusare gli stessi concetti
operativi, cioe' data corrente selezionata, separazione tra tratto passato e
proiezione futura, e sovrapposizione al grafico a candele gestito dal livello
applicativo web. Verifiche eseguite: parsing AST senza bytecode, esempio
`examples/minimal_import.py`, import legacy/lowercase in entrambi gli ordini,
editable install in `cyenv`, test mirati import/native/docstring, suite
completa standard e golden test QQQ EOD con `CYPREDICT_RUN_GOLDEN=1`.

### Criterio di uscita

L'import pubblico deve restare stabile:

```python
from cypredict import CyPredict
```

e, se necessario in transizione:

```python
import cyPredict
model = cyPredict.cyPredict(...)
```

## Milestone 9 - Librerie native/custom dentro progetto

### Scopo

Rendere riproducibile build e import dei moduli C/C++.

### Attivita'

1. Inventariare versioni e sorgenti:
   - `goertzel.cpp`;
   - `cyfitness.cpp`;
   - `cyGAopt.cpp`;
   - `cyGAoptMulticore.cpp`;
   - setup/CMake usati;
   - dipendenze compiler e Python ABI.
2. Decidere se includere:
   - sorgenti soltanto;
   - sorgenti + wheel buildata;
   - sorgenti + `.pyd` per uso locale.
3. Preferenza consigliata:
   - versionare sorgenti e script build;
   - non versionare `build/`;
   - versionare wheel in release/artifact, non nella repo principale;
   - documentare `.pyd` locale come artifact generato.
4. Creare struttura:
   - `src/cypredict/native/goertzel`;
   - `src/cypredict/native/cyfitness`;
   - `src/cypredict/native/cygaopt`;
   - `src/cypredict/native/cygaopt_multicore`.
5. Aggiungere script:
   - `scripts/build_native.ps1`;
   - `scripts/clean_native_build.ps1`;
   - eventuale `cibuildwheel` in futuro.
6. Aggiungere test di import:
   - import moduli nativi;
   - chiamata minima Goertzel;
   - chiamata minima fitness;
   - fallback o errore chiaro se non disponibili.
7. Gestire ABI:
   - Python 3.10 osservato nei `.pyd`;
   - documentare incompatibilita' con altre versioni Python;
   - pianificare wheel per versioni supportate.

### Deliverable

- Sorgenti native dentro repo o submodule.
- Script build.
- Test di smoke import.
- Documentazione build Windows.

### Criterio di uscita

Un ambiente pulito deve poter costruire o installare le estensioni senza copiare manualmente file da cartelle di studio.

### Stato aggiornato

Milestone 9 chiusa: sorgenti native, setup e build script sono versionati sotto
`native/`; `scripts/build_native.ps1` ricompila `goertzel`, `cyfitness`,
`cyGAopt`, `cyGAoptMultiCore` e il modulo legacy `genetic_optimization` in
ambiente `cyenv`; `scripts/clean_native_build.ps1` pulisce in modo vincolato i
build artifact sotto `native/`. I test nativi ora non si limitano all'import:
eseguono una chiamata minima Goertzel, una chiamata minima
`cyfitness.evaluate_fitness` e una chiamata minima
`cyGAoptMultiCore.evaluate_cycle_loss`. Verifiche eseguite: build native
completa con MSVC 14.44 e Windows SDK 10.0.26100.0, parsing dello script clean,
test nativi, suite completa standard e golden test QQQ EOD con
`CYPREDICT_RUN_GOLDEN=1`.

## Milestone 10 - Worker periodici e GammaSignalForge

### Scopo

Preparare un contratto stabile per calcolo periodico delle proiezioni da worker.

### Requisiti suggeriti

1. Funzione senza grafici e senza dipendenza notebook:

```python
result = predictor.project_cycles(
    data=data,
    as_of=current_datetime,
    config=projection_config,
)
```

2. Output serializzabile:

```json
{
  "ticker": "ES=F",
  "timeframe": "5m",
  "as_of": "2024-06-24T23:50:00+00:00",
  "generated_at": "2026-04-29T16:00:00+02:00",
  "code_version": "427250c",
  "dominant_periods": [],
  "future_turning_points": [],
  "fitness": null,
  "warnings": []
}
```

3. Worker:
   - legge ultimo dato disponibile;
   - salta se non ci sono nuove barre;
   - calcola proiezione;
   - salva output con chiave `ticker/timeframe/as_of`;
   - registra log strutturato;
   - segnala errori senza bloccare altri ticker.
4. Persistenza:
   - CSV per compatibilita';
   - Parquet o database per uso applicativo;
   - JSON per integrazione rapida.
5. Idempotenza:
   - stesso input e stesso config producono stesso output;
   - se esiste gia' output per `ticker/timeframe/as_of/config_hash`, non ricalcola salvo `force=True`.
6. Scheduling:
   - worker per timeframe;
   - trigger dopo chiusura barra;
   - margine configurabile per data provider delay.
7. Config hash:
   - serializzare tutti i parametri;
   - includere versione librerie native;
   - salvare hash nell'output.

### Deliverable

- `project_cycles(...)` o equivalente.
- Schema output per GammaSignalForge.
- Esempio worker in `examples/worker_projection.py`.
- Test idempotenza.

### Criterio di uscita

GammaSignalForge deve poter chiamare una API senza dipendere da notebook, grafici o stampe libere su console.

## Milestone 11 - Notebook e esempi applicativi

### Scopo

Allineare esempi reali alla API pulita.

### Attivita'

1. Creare notebook/script piccoli derivati dagli esempi V3:
   - daily;
   - intraday;
   - multirange;
   - min/max dataframe incrementale.
2. Sostituire celle con parametri legacy con config object o firme aggiornate.
3. Rimuovere output pesanti salvati nel notebook.
4. Tenere i notebook di ricerca storici come archivio, non come test primario.
5. Aggiungere script riproducibili in `examples/`.

### Deliverable

- `examples/daily_basic.py`.
- `examples/intraday_file_projection.py`.
- `examples/min_max_incremental.py`.
- Eventuale notebook pulito `notebooks/cypredict_quickstart.ipynb`.

## Ordine operativo consigliato

1. Baseline golden.
2. `.gitignore`, packaging e README.
3. Inventario API e parametri.
4. Logging centralizzato.
5. Docstring per funzioni pubbliche.
6. Oggetti risultato/wrapper legacy.
7. Rimozione parametri approvati.
8. Modularizzazione.
9. Native libs e build.
10. API worker e GammaSignalForge.
11. Pulizia notebook/esempi.

## Rischi principali

1. Ottimizzazioni genetiche non deterministiche.
   - Mitigazione: seed obbligatorio, logging seed, test su intervalli/tolleranze.
2. Differenze tra Python e C++.
   - Mitigazione: test dedicati su funzioni native e confronto baseline.
3. Timezone e date intraday.
   - Mitigazione: test specifici su date timezone-aware e timezone-naive.
4. Rimozione parametri ancora usati nei notebook.
   - Mitigazione: scansione notebook e wrapper legacy.
5. Plotting mischiato al calcolo.
   - Mitigazione: separare `show_charts` e moduli plotting.
6. Import da cartelle locali non riproducibili.
   - Mitigazione: native folder, build scripts, smoke tests.

## Checklist di accettazione finale

- Repo installabile in editable mode.
- Nessuna cartella `__pycache__` o `.ipynb_checkpoints` tracciata.
- Ogni funzione ha docstring.
- Le funzioni pubbliche hanno esempi.
- Le firme pubbliche sono documentate.
- I parametri rimossi sono tracciati in `parameter_matrix.md`.
- I log sono strutturati e opzionalmente salvabili su file.
- Le librerie native sono dentro struttura controllata o buildabili da script.
- I test golden passano.
- GammaSignalForge puo' usare un'API senza notebook e senza output grafico obbligatorio.
