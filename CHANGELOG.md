# Changelog

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
