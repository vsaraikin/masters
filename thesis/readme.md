# Resources



## Ранжирование

-	An Automated Trading Strategy Grounded in Machine Learning and On‑Chain Analytics (Glassnode Insights, Feb 12 2024) – Исследовала, как on‑chain данные могут предсказывать динамику цены биткоина; использовала supervised ML‑модель с feature‑importance (SHAP) для отбора ключевых метрик (процент адресов в прибыли, STH‑SOPR) и формирования «Bitcoin Sharpe Signal»; вывод: composite on‑chain signals дают значимый альфа‑сигнал для автоматизированных стратегий и пригодятся для построения риск‑ориентированных торговых моделей  ￼
-	Building Cross‑Sectional Systematic Strategies by Learning to Rank (Poh et al., JFDS Apr 2021) – Изучала применение LTR‑алгоритмов вместо регрессии для кросс‑секционного momentum в акциях; применяла RankNet, LambdaMART, ListNet, ListMLE для парного и спискового ранжирования акций; вывод: LTR повышает Sharpe Ratio почти в 3 раза по сравнению с традиционными методами, что делает его эффективным инструментом для отбора активов D_Poh_DPhil_Thesis_final.pdf](file-service://file-3mqDrdkSVDrZHZc84JhiP5)
•	Enhancing Cross‑Sectional Currency Strategies by Context‑Aware Learning to Rank with Self‑Attention (Poh et al., JFDS Jul 2022) – Исследовала учёт контекста при перекомпоновке начального ранжирования валют с помощью Transformer‑энкодера; обучала self‑attention на топ/боттом списках для ре‑ранжирования; вывод: модель даёт ~30 % прироста Sharpe и устойчивость в risk‑off фазах, пригодна для адаптивного FX‑ранжирования D_Poh_DPhil_Thesis_final.pdf](file-service://file-3mqDrdkSVDrZHZc84JhiP5)
•	Transfer Ranking in Finance: Applications to Cross‑Sectional Momentum with Data Scarcity (Poh, Zohren, Roberts; JSI 2023) – Рассмотрела проблему ранжирования при малом объёме данных (крипто); предложила Fused Encoder Networks с parameter sharing и self‑attention; вывод: модель превосходит бенчмарки по Sharpe и robustness для топ‑10 крипто‑момен­тум‑стратегий, полезна для активов с короткой историей D_Poh_DPhil_Thesis_final.pdf](file-service://file-3mqDrdkSVDrZHZc84JhiP5)
•	Shuffle and Learn: Unsupervised Learning Using Temporal Order Verification (Misra, Zitnick, Hebert; ECCV/CVPR 2016) – Исследовала self‑supervised pretext‑задачу восстановления порядка кадров в видео; использовала CNN для классификации «правильного» порядка пар/триплетов кадров; вывод: модель учит сильные темпоральные представления, применимые для предобучения и эмбеддинга временных окон он‑чейн‑метрик  ￼
•	Self‑Supervised Spatial‑Temporal Normality Learning for Time Series Anomaly Detection (STEN) (Chen et al.; arXiv Jun 28 2024) – Изучала unsupervised anomaly detection в multivariate time series через два pretext‑таска: предсказание порядка (OTN) и предсказание расстояний (DSN); использовала Jensen‑Shannon loss и MSE для обучения; вывод: STEN значительно превосходит SOTA на benchmark’ах TSAD, пригоден для ранжирования аномальных дней в он‑чейн данных  ￼
•	Rank Pooling for Action Recognition (Fernando, Gavves, Oramas, Ghodrati, Tuytelaars; TPAMI 2017) – Предложила функциональное temporal pooling, обучая линейную RankSVM на фичах кадров для их хронологического ранжирования; параметры RankSVM стали представлением видео; вывод: такое представление захватывает динамику лучше, чем простое усреднение, и может использоваться для кодирования динамики метрик скользящих окон перед LTR  ￼
•	Cross‑Sectional Momentum Strategy (Jegadeesh & Titman; JF 1993) – Исследовала прибыльность стратегии сортировки акций по прошлым 3–12‑месячным доходностям; использовала простое ранжирование и торговлю top/bottom децилей; вывод: кросс‑секционный моментум приносит статистически значимую прибыль, служит классическим базисом и вдохновением для применения LTR к он‑чейн данным  ￼


Ресурсы
1.	Building Cross‑Sectional Systematic Strategies by Learning to Rank (Poh et al. 2021)
– чисто LTR‑постановка: обучали RankNet, LambdaMART, ListNet, ListMLE — ранжировали акции по будущей доходности.
2.	Enhancing Cross‑Sectional Currency Strategies with Context‑Aware Learning to Rank (Poh et al. 2022)
– LTR + self‑attention: первоначальный список валют ранжируется, затем контекстно ре‑ранжируется Transformer‑моделью.
3.	Transfer Ranking in Finance: Applications to Cross‑Sectional Momentum with Data Scarcity (Poh et al. 2023)
– Transfer LTR: Fused Encoder Networks обучаются передавать ранжирующие признаки с одного рынка на другой.
4.	Shuffle and Learn: Unsupervised Learning Using Temporal Order Verification (Misra et al. 2016)
– self‑supervised: перемешанные фрагменты «ранжируются» по хронологии (model predicts порядок).
5.	Self‑Supervised Spatial‑Temporal Normality Learning for Time Series Anomaly Detection (STEN) (Chen et al. 2024)
– один pretext‑таск – Temporal Order Network: требует ранжировать суб‑последовательности по времени.
6.	Rank Pooling for Action Recognition (Fernando et al. 2017)
– обучают RankSVM ранжировать кадры в видео‑окне по времени и используют полученные веса как свёртку.
7.	Cross‑Sectional Momentum Strategy (Jegadeesh & Titman 1993)
– классическое ранжирование акций по прошлым доходностям для отбора «winners» и «losers».

----
- https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00643-1:
  - CNN–LSTM model consistently achieved higher accuracy
  - The Boruta + CNN–LSTM combination shows the best performance
- https://www.mdpi.com/2571-9394/6/2/16
  - LSTM-Attention with gradient- specific optimization performs well in Bitcoin forecasts, making it more appropriate for Bitcoin predictions
- https://arxiv.org/html/2405.11431v1
  - Bidirecional-LSTM provides the highest accuracy
  - multilayer perceptron and ARIMA
- бакалавр диплом
  - https://www.diva-portal.org/smash/get/diva2:1778251/FULLTEXT03.pdf
- https://github.com/Armanx200/Bitcoin_Price_Prediction
  - RF Regressor
- Weighted Ensemble
  - https://medium.com/bitgrit-data-science-publication/predict-cryptocurrency-prices-with-data-science-0ea732a0ea0f

- [ONCHAIN ANALYTICS - A New Methodology for Cryptocurrency Analysis](https://www.theseus.fi/bitstream/handle/10024/879592/Nguyen_Quyen.pdf?sequence=2&isAllowed=y)
  - Много источников
- [Constructing Cross-Sectional Trading Strategies](https://www.robots.ox.ac.uk/~sjrob/Theses/D_Poh_DPhil_Thesis_final.pdf)
- 





## 1. Определяем фазу рынка (Bull / Bear / Side)
- **Что делаем**  
  1. Сегментируем историю цены BTC на «рост / падение / боковик» (Hidden Markov Model или Bai‑Perron).  
  2. Собираем дневные on‑chain метрики: NUPL, % Supply in Profit, MVRV‑Z, Accumulation Trend Score и др.  
  3. Обучаем LightGBM предсказывать фазу рынка на следующий день.  
- **Зачем** – заранее знать, когда наращивать риск, а когда уходить в кэш.  
- **Ресурсы**  
  - Glassnode Insights: *“Dissecting Bitcoin’s Unrealised Profit/Loss”*  
    <https://medium.com/glassnode-insights/dissecting-bitcoins-unrealised-on-chain-profit-loss-73e735020c8d>  
  - Glassnode Insights: *“Finance Bridge Edition 5”*  
    <https://insights.glassnode.com/finance-bridge-edition-5/>

---

## 2. Предсказываем всплески волатильности на завтра
- **Что делаем**  
  1. Метка «спайк», если завтрашний ход цены > 1.5 × 20‑дн σ.  
  2. Фичи: whale‑tx объёмы, Miner → Exch flows, Funding‑Rate, ATM‑IV 1w, mempool pressure.  
  3. Обучаем Synthesizer‑Transformer выявлять спайки.  
- **Зачем** – заранее снижать плечо или хеджироваться опционами.  
- **Ресурсы**  
  - arXiv 2211.08281: *“Forecasting Bitcoin Volatility Spikes from Whale Transactions…”*  
    <https://ar5iv.labs.arxiv.org/html/2211.08281>

---

## 3. 10‑дневный Sharpe‑Signal (регрессия)
- **Что делаем**  
  1. Считаем будущий Sharpe Ratio на 10 дней вперёд.  
  2. CatBoost регрессирует это значение по on‑chain метрикам, включая Realized Cap Δ, Options Basis и др.  
- **Зачем** – числовой «скор» для масштабирования позиции.  
- **Ресурсы**  
  - Glassnode Insights: *“The Predictive Power of Glassnode Data”*  
    <https://insights.glassnode.com/the-predictive-power-of-glassnode-data/>

---

## 4. Индикатор «умные vs глупые» (SOPR‑spread)
- **Что делаем**  
  1. Из JSON `*_sopr_by_wallet_size` берём SOPR крупных кошельков (> 1 k BTC) и розницы (< 0.01 BTC).  
  2. Δ = SOPR<sub>киты</sub> − SOPR<sub>ретейл</sub>.  
  3. z‑score(Δ, 180 дн): z < –1 → покупаем, z > +1 → частично выходим.  
- **Зачем** – простая интерпретируемая метрика поведения «умных денег».  
- **Ресурсы**  
  - Та же JSON‑метрика Glassnode: `o_v1_metrics_breakdowns_sopr_by_wallet_size`  
  - Статья из п. 1 как методология интерпретации SOPR.

---

## 5. Комбинированная торговая стратегия
- **Что делаем**  
  Ежедневные правила:  
  - Bull prob > 0.6 и SharpeSig > 0 → 100 % long.  
  - Side & SharpeSig > 0 → 50 % long.  
  - Bear prob > 0.5 → Flat.  
  - Vol‑Spike prob > 0.6 → снижаем плечо + недельный straddle.  
  - ΔSOPR сигналы ± 0.3 позиции.  
  Backtest 2019‑2023 (vectorbt), сравнение с HODL.  
- **Зачем** – получить стратегию, которая бьёт buy‑and‑hold по Sharpe и снижает просадку.  
- **Ресурсы**  
  - Все предыдущие блоки (1‑4) + Glassnode API для котировок он‑чейн‑метрик.

---

## 6. Каузальный анализ ключевых метрик
- **Что делаем**  
  1. Берём топ‑фичи с высоким SHAP‑вкладом из блоков 1‑3.  
  2. Проверяем, предшествуют ли их изменения цене (Multiple‑Output GP causality, лаг 1‑30 дней).  
  3. Формируем таблицу лаг → p‑value.  
- **Зачем** – академически показать, что метрика действительно «ведёт» цену, а не просто коррелирует.  
- **Ресурсы**  
  - Chalkiadakis et al. 2022: *“On‑chain analytics for sentiment‑driven statistical causality in cryptocurrencies”*  
    <https://doi.org/10.1016/j.bcra.2022.100063>

## Порядок реализации в коде

1. **0_DataPrep.ipynb** — парс JSON, заполняем пропуски, группируем фичи.  
2. **Phase1_MarketRegime.ipynb** — метки Bull/Bear/Side, модель LightGBM, графики SHAP.  
3. **Phase2_VolSpike.ipynb** — бинарная модель всплесков волатильности.  
4. **Phase3_SharpeSignal.ipynb** — регрессия на 10‑дн Sharpe.  
5. **Phase4_SOPRspread.ipynb** — расчёт δSOPR и пороговых сигналов, визуализация.  
6. **Phase5_StrategyBacktest.ipynb** — соединяем все правила, бэктест в vectorbt, вывод equity curve и KPI.  
7. **Phase6_CausalityStudy.ipynb** — проверка Грейнджера / Gaussian‑process causality, таблица результатов.

Каждый ноутбук даёт самостоятельный результат (таблица метрик, картинка или кривая доходности), что делает работу и полезной, и доказательной.