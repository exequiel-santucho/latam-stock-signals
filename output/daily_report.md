# LatAm stock signals - 2026-10-21

Model: `pred_xgb` | horizon: 20 trading days | generated 2026-10-21 20:03

## Buy candidates (beat the country ETF in the next 20 days)

| Ticker       | Market   | Sector           |   Close |   prob | market_regime   |
|:-------------|:---------|:-----------------|--------:|-------:|:----------------|
| FUNO11.MX    | MX       | Real Estate      |   32.96 |  0.608 | bear            |
| KOFUBL.MX    | MX       | Consumer Staples |  220.2  |  0.606 | bear            |
| CBAV3.SA     | BR       | Materials        |   11.22 |  0.606 | bull            |
| VESTA.MX     | MX       | Real Estate      |   67.55 |  0.591 | bear            |
| FIBRAMQ12.MX | MX       | Real Estate      |   46.03 |  0.59  | bear            |

## Model health (last 120 days with known outcome)

- Top-5 precision: 42.8% (base rate 45.8%)
- Average excess return of the top-5 vs the country ETF: -0.93%

## Strategy (test period, exact simulation)

| portfolio              |   cagr |   annual_vol |   sharpe |   max_drawdown |   calmar |   total_return |
|:-----------------------|-------:|-------------:|---------:|---------------:|---------:|---------------:|
| strategy               |  1.179 |        0.149 |    0.885 |         -0.224 |    0.8   |          1.174 |
| ILF (Latin America 40) |  1.161 |        0.226 |    0.581 |         -0.297 |    0.54  |          1.019 |
| EWZ (Brazil)           |  1.143 |        0.266 |    0.472 |         -0.322 |    0.442 |          0.875 |
| S&P 500                |  1.106 |        0.169 |    0.429 |         -0.254 |    0.416 |          0.607 |
