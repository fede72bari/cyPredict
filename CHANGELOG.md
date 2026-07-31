# Changelog

## [0.3.0] - 2026-07-31

### Fixed
- **Il passa-banda non veniva MAI applicato.** All'ingresso di `multiperiod_analysis` c'era
  `if(detrend_type != 'linear' and detrend_type != 'lowess'): detrend_type = 'hp_filter'`, che
  riscriveva in passa-alto qualsiasi altro valore. Il detrend telescopico in `analyze_and_plot`
  era corretto ma irraggiungibile: chiedendo `band_pass` si ottenevano cicli identici a quelli
  del passa-alto (stesse frequenze e stesse ampiezze, verificato su dati reali). Ora si
  normalizza solo cio' che non e' implementato.
- **Riferimento del fit globale non assegnato col passa-banda.** La scelta della serie di
  riferimento copriva solo `hp_filter` e `lowess`: con il passa-banda `index_detrended_data`
  restava non assegnata (`UnboundLocalError`). Il passa-banda usa ora lo stesso criterio del
  passa-alto (la banda col lambda piu' grande), con una rete di sicurezza sulla serie piu' lunga
  per qualsiasi altro detrend.

### Changed
- **Separati la FAMIGLIA di filtro e il MODO.** Il passa-banda non e' un'alternativa al filtro HP:
  e' lo stesso filtro usato due volte, uno in sottrazione dell'altro, quindi e' un modo di usare
  una famiglia e si puo' realizzare anche con altre famiglie.
  - `detrend_type` resta la famiglia: `hp_filter`, `lowess`, `jh_filter`, `linear`, `quadratic`.
  - nuovo `filter_band_type`: `high_pass` (tiene tutto cio' che e' piu' veloce del taglio) oppure
    `band_pass` (toglie anche la parte piu' veloce, lasciando la sola banda).
  - il modo e' applicato in fondo alla catena di detrend, dopo che la famiglia ha prodotto la
    serie: vale per qualsiasi famiglia. Taglio veloce implementato per `hp_filter`
    (`hp_filter_lambda_min`) e per `lowess` (`lowess_k_min`); per le altre famiglie il modo
    passa-banda registra a log che manca il taglio veloce e resta passa-alto.
  - il modo si puo' decidere anche banda per banda, con una colonna `filter_band_type` nelle
    righe dei range.
- **Compatibilita' totale**: `detrend_type='band_pass'` continua a funzionare e viene tradotto
  nella coppia (famiglia `hp_filter`, modo `band_pass`). Verificato su dati reali: vecchia forma e
  nuova forma danno risultati identici; passa-banda e passa-alto danno risultati diversi.

## [0.2.0] - 2026-07-30

### Added
- **Ottimizzazione algebrica** come alternativa alle euristiche, in `OptimizationMixin`:
  - `MultiAn_fit_algebraic()` — ampiezze **e fasi** ottimali in forma chiusa. Riscrivendo
    `a·sin(ωt+φ) = A·sin(ωt) + B·cos(ωt)`, a frequenze fissate il modello è lineare nei
    coefficienti: l'errore quadratico è convesso e il minimo si calcola con le equazioni
    normali. Deterministico, senza griglie di discretizzazione né seed.
  - `MultiAn_fit_varpro()` — variable projection: le ampiezze vengono eliminate
    analiticamente e la ricerca resta solo sulle frequenze (n dimensioni invece di 3n),
    partendo dai picchi della trasformata.
  - Misurato su 1.019 confronti reali contro il fit genetico: correlazione mediana
    0,474 → 0,680 (migliore nel 99% dei casi), NMSE 0,775 → 0,538, in un terzo del tempo.
    Il guadagno viene soprattutto dalle fasi, che il genetico non ottimizza.
- **Metriche di qualità del fit** (prima calcolate fuori dalla libreria):
  - `MultiAn_frequency_bin_width()` — risoluzione spettrale della finestra di analisi.
  - `MultiAn_scale_composite()` — scala del composite: `lsq` (regressione, minimizza
    l'errore) o `minmax` (storica). Misurato: la riscalatura min/max peggiora l'NMSE del 96%.
  - `MultiAn_fit_fidelity()` — correlazione e NMSE sulla finestra di ottimizzazione.
- **Detrend `band_pass`** (passa-banda telescopico): `HP(λ) − HP(λ_min)`, con `λ_min` derivato
  automaticamente dalla banda precedente o dal bordo teorico. Purezza in banda misurata
  65-74% contro 24-38% del solo passa-alto.

### Changed
- `MultiAn_fit_varpro` esprime il vincolo sulle frequenze in **frazione di bin**
  (`freq_bound_bins`, default mezzo bin) invece che in percentuale fissa: la stessa
  percentuale vale mezzo bin su una finestra corta e diversi bin su una lunga, e solo la
  risoluzione spettrale definisce quanto sia sensato spostarsi. Il parametro storico
  `freq_bounds_pct` resta accettato per compatibilità.
