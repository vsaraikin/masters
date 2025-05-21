# Resources

обязательно нужно указать, что хотим предсказывать, что за вид задачи, 

## Draft Ideas

1. Regression Price. Use multiple Y, where each Y goes from 1 to 14.
2. Use correlation between Y and X to find the best features. Choose the best features based on correlation. // This idea should be formalized.
3. Classification of market regimes. Test different number of classes. The worse the class, the more expsosure we need to have.
4. Sharpe signal prediction
5. Volatility prediction
6. Learning to rank


## Papers

1. [Blockchain Metrics and Indicators in Cryptocurrency Trading](https://arxiv.org/pdf/2403.00770). Авторы ввели «рibbon»-индикаторы на основе он-чейн-метрик и показали, что их добавление в LogReg / RF / MLP повышает Sharpe и долю прибыльных сделок по сравнению с ценовыми-only моделями.   

2. [A Scalable RL-based System Using On-Chain Data for Cryptocurrency Portfolio Management](https://arxiv.org/pdf/2307.01599). PPO-агент ежеминутно ребалансирует портфель, опираясь на он-чейн-фичи; бэктест дал +83 % годовых и рост Sortino на 2,2 пункта против buy-and-hold BTC.   

3. [Forecasting Bitcoin Volatility Spikes from Whale Transactions and CryptoQuant Data](https://ar5iv.labs.arxiv.org/html/2211.08281). Synthesizer Transformer классифицирует «будет ли завтра волатильность > Х %», используя 150+ он-чейн-фич и твиты Whale-Alert; обошёл GARCH/LSTM и снизил max drawdown в опционной стратегии.   

4. [Deep Learning for Bitcoin Price Direction Prediction: Models and Trading Strategies](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00643-1). Boruta FS + CNN-LSTM на ценовых + он-чейн-признаках дал 82 % accuracy; лонг/шорт-стратегия показала 6 654 % совокупной доходности (2017-2023).   

5. [Predicting Cryptocurrencies Market Phases through On-Chain Data](https://iris.unito.it/retrieve/2927f4b0-43f0-46fd-a840-069dd1ec61c0/6.%20ICBC23%20-%20PREDICTING%20BTC.pdf). GMM/HMM-кластеризация Glassnode-метрик выделяет 5 фаз («капитуляция»→«эйфория»); динамическая аллокация по фазам улучшает CAGR и снижает просадку.   

6. [An Exploratory Study of Cryptocurrency Price Prediction Using ML Techniques](https://www.astesj.com/publications/ASTESJ_080321.pdf). Random Forest и GB превзошли SVM/k-NN по MAE; включение он-чейн-фич увеличило R² на ~20 % против чисто ценовых.   

7. [Uncovering the Determinants of Crypto-Asset Prices via Machine Learning](https://arxiv.org/pdf/2201.12893). LASSO / XGBoost на 40 фундаментальных признаках выявили, что NUPL, SOPR и биржевые потоки объясняют до 35 % дисперсии цены; стратегия по top-feature-портфелю повысила Sharpe на 0.5-0.7.   

8. [Master Thesis: Predictive Models for Cryptocurrency Markets](https://pure.tue.nl/ws/portalfiles/portal/341640654/Master_Thesis_Nikhil_Razab-Sekh.pdf). Генетическое программирование + нечёткие правила достигли RMSE 678 и MAPE 1.7 %, обойдя RNN/LSTM и сохранив интерпретируемые правила.   

9. [Deep Learning Approaches for Cryptocurrency Price Forecasting](https://pure.hw.ac.uk/ws/portalfiles/portal/58776350/1_s2.0_S2096720922000033_main_1_.pdf). Transformer-TFT показал лучшую directional-accuracy 78 % и минимальный RMSE 1.2 %; SHAP выделил SOPR и биржевые инфлоу как ключевые признаки.   

10. [Deep Learning for Bitcoin Price Direction Prediction (PDF mirror)](https://www.newswise.com/pdf_docs/173642305592044_s40854-024-00643-1.pdf). Авторский PDF статьи из п. 4, выводы идентичны.   

11. [Hybrid Time-Series and On-Chain Indicators for Bitcoin Forecasting](https://arxiv.org/pdf/2204.12928). Gradient Boosting на TA + Glassnode; Boruta оставила 9 критичных фич, MVRV-Z и Exchange Netflow среди топ-важных; MAE снизилась до ≈360 USD.   

12. [Blockchain-Based Cryptocurrency Price Prediction with Chaos Theory, On-Chain and Sentiment](https://dergipark.org.tr/tr/download/article-file/2750466). Обзор показывает, что гибрид «он-чейн + sentiment» снижает MAPE до 3 % против 6 % у чисто теханалитических моделей.   

13. [Glassnode Finance Bridge • Edition 5](https://insights.glassnode.com/finance-bridge-edition-5/). Отчёт использует MVRV-Z, HODL Waves и долю STH-в-убытке как риск-фильтр; модель риска сокращает max-drawdown на 30 %.   

14. [Dissecting Bitcoin’s Unrealised On-Chain Profit & Loss](https://medium.com/glassnode-insights/dissecting-bitcoins-unrealised-on-chain-profit-loss-73e735020c8d). Вводится NUPL: `NUPL < 0` маркирует дно рынка, `NUPL > 0.75` — эйфорию; контрциклические стратегии по этим порогам повышают Sharpe до 2.0.   


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


Bull / Bear / Neutral
  - Glassnode Insights: *“Dissecting Bitcoin’s Unrealised Profit/Loss”*  
    <https://medium.com/glassnode-insights/dissecting-bitcoins-unrealised-on-chain-profit-loss-73e735020c8d>  
  - Glassnode Insights: *“Finance Bridge Edition 5”*  
    <https://insights.glassnode.com/finance-bridge-edition-5/>

---
Volatility Prediction
  - arXiv 2211.08281: *“Forecasting Bitcoin Volatility Spikes from Whale Transactions…”*  
    <https://ar5iv.labs.arxiv.org/html/2211.08281>
