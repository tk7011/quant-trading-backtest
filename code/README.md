quant_backtest/

├── pipeline.py              ← runs everything end-to-end

├── data/
│   ├── download.py          ← Binance OHLC download

│   └── merge.py             ← feature CSV merger
├── strategy/
│   ├── chandelier_exit.py   ← CE indicator + signals

│   └── optimise_parameters.py
├── ml/
│   ├── markov_regimes.py    ← MRS model + plots

│   ├── feature_engineering.py  ← EMA + target

│   └── train_random_forest.py
└── backtest/
    ├── metrics.py           ← raw CE trade metrics
    
    └── signal_filter.py     ← ML-filtered backtest
    
    
