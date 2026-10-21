# LatAm Stock Signals (versión en español)

Proyecto final del [Stock Markets Analytics Zoomcamp 2026](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp). La versión completa en inglés está en [README.md](README.md).

**Todos los días, después del cierre, el sistema elige las acciones latinoamericanas con más probabilidad de ganarle al ETF de su país en el próximo mes, simula la estrategia y publica las operaciones para la rueda siguiente.**

## Resultados (período de prueba ene-2022 → sep-2026, en USD, con comisiones)

| Cartera ($100k) | CAGR | Volatilidad | Sharpe | Máxima caída |
|---|---|---|---|---|
| **Estrategia: XGBoost top-5 diario, stop-loss 10%, tamaño según probabilidad** | **17,9%** | **14,9%** | **0,89** | **-22,4%** |
| ILF (ETF Latinoamérica 40) | 16,1% | 22,6% | 0,58 | -29,7% |
| EWZ (ETF Brasil) | 14,3% | 26,6% | 0,47 | -32,2% |
| S&P 500 | 10,6% | 16,9% | 0,43 | -25,4% |

## Resumen

- **Problema:** predecir si cada acción le gana al ETF de su país (EWZ, EWW, ECH, ARGT, ILF) en USD durante los próximos 20 días hábiles.
- **Universo:** 239 acciones de Brasil (IBrA), México (IPC), Chile (IPSA), ADRs argentinos y ADRs andinos (Colombia y Perú).
- **Datos:** Yahoo Finance (precios, ETFs, tipos de cambio, commodities, tasas de EE.UU., VIX), API del sitio de B3 (composición del IBrA y sector de cada acción), Banco Central do Brasil (Selic, IPCA) y BCRA (reservas). Más de 1 millón de registros diarios desde 2004.
- **Variables:** ~190 numéricas (momentum, técnicas, TA-Lib, fuerza relativa vs el ETF y el S&P 500, beta, rankings dentro de cada mercado y día, mercado y macro) + 10 categóricas (mercado, sector, mes y variables discretizadas).
- **Limpieza:** días sin volumen, splits mal ajustados en Yahoo (56 mil filas eliminadas), picos erróneos del USD/CLP y rezago de publicación en los datos macro. Solo se opera con acciones que mueven ≥ US$1M por día.
- **Modelos:** árbol de decisión, Random Forest y XGBoost, con split temporal y embargo de 20 días, selección del conjunto de variables y ajuste de hiperparámetros en validación (2019-2021). Las variables de mercado y macro empeoraban la validación (el modelo memorizaba períodos), así que se usan como filtros y no como entradas.
- **Simulación:** 4 estrategias vectorizadas por modelo (umbral, top-5, top-5 solo en mercado alcista, long-short) y una simulación exacta día por día con reinversión, stop-loss/take-profit y tamaño por probabilidad. La estrategia se elige en validación y se evalúa una sola vez en prueba.
- **Automatización:** `main.py` con carga incremental, dos regímenes (`--fast` desde disco o snapshot de Google Drive; completo desde las APIs), reentrenamiento opcional (`--retrain`), cron / Programador de tareas, Docker, reporte diario con monitoreo del modelo y envío opcional por Telegram.

## Cómo correrlo

```bash
uv sync
uv run python main.py          # descarga incremental + transformaciones + predicciones + reporte
uv run python main.py --fast   # sin descargas (datos y modelos en data/)
uv run jupyter lab notebooks/  # notebooks de análisis
```

Más detalles (variables, tablas por modelo y estrategia, análisis por mercado, sector y año, limitaciones): [README.md](README.md).

Proyecto educativo. No es una recomendación de inversión.
