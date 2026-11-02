# Paris bike forecast

Hourly forecasts of bike traffic at the City of Paris permanent bike counters, evaluated with
rolling-origin backtests against seasonal-naive baselines, with quantile intervals and drift
monitoring.

This is my own rewrite of a bike-demand forecasting project I did as coursework. The original
course repo lives under a classmate's account and nothing here is copied from it: the data
source, pipeline, models and evaluation were rebuilt from scratch.

Work in progress. Results, architecture and run instructions are added phase by phase.

## Data

- **Paris bike counters** ("Comptage vélo", Ville de Paris, opendata.paris.fr): hourly counts
  per counter. Licence: Open Database License (ODbL).
- **Weather**: Open-Meteo historical weather API. Licence: CC BY 4.0. Weather data by
  [Open-Meteo.com](https://open-meteo.com/).
- **Public holidays**: calendrier.api.gouv.fr (data.gouv.fr). Licence: Licence Ouverte 2.0.

History length and counter counts are generated from the sources and added in Phase 1.
