import asyncio
from playwright.async_api import async_playwright
import os

# Папка для сохранения
os.makedirs("downloads", exist_ok=True)

# Список ссылок и названий (вставь сюда полный список)
references = [
    ("A Data Driven Approach to Market Regime Classification (Imperial MSc)", "https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/McIndoe.pdf"),
    ("A Scalable RL-based System Using On-Chain Data for Cryptocurrency Portfolio Management", "https://arxiv.org/pdf/2307.01599"),
    ("Algorithmic Factor Investing with Regime Classification (Medium)", "https://medium.com/@matthewwang_91639/algorithmic-factor-investing-with-market-regime-classification-6bc2f8c7168b"),
    ("An Exploratory Study of Cryptocurrency Price Prediction Using ML Techniques", "https://www.astesj.com/publications/ASTESJ_080321.pdf"),
    ("BSS Short Signal – Identify High-Confidence Shorting Opportunities", "https://insights.glassnode.com/bss-short-signal-identify-high-confidence-shorting-opportunities"),
    ("Blockchain Metrics and Indicators in Cryptocurrency Trading", "https://arxiv.org/pdf/2403.00770"),
    ("Blockchain-Based Cryptocurrency Price Prediction with Chaos Theory, On-Chain and Sentiment", "https://dergipark.org.tr/tr/download/article-file/2750466"),
    ("CatBoost Ranking Losses", "https://catboost.ai/en/docs/concepts/loss-functions-ranking"),
    ("Constructing Cross-Sectional Trading Strategies (Oxford DPhil Thesis)", "https://www.robots.ox.ac.uk/~sjrob/Theses/D_Poh_DPhil_Thesis_final.pdf"),
    ("Decoding Market Regimes with ML (State Street)", "https://www.ssga.com/library-content/assets/pdf/global/pc/2025/decoding-market-regimes-with-machine-learning.pdf"),
    ("Deep Learning Approaches for Cryptocurrency Price Forecasting", "https://pure.hw.ac.uk/ws/portalfiles/portal/58776350/1_s2.0_S2096720922000033_main_1_.pdf"),
    ("Deep Learning for Bitcoin Price Direction Prediction (PDF mirror)", "https://www.newswise.com/pdf_docs/173642305592044_s40854-024-00643-1.pdf"),
    ("Deep Learning for Bitcoin Price Direction Prediction: Models and Trading Strategies", "https://jfin-swufe.springeropen.com/articles/10.1186/s40854-024-00643-1"),
    ("Dissecting Bitcoin’s Unrealised On-Chain Profit & Loss", "https://medium.com/glassnode-insights/dissecting-bitcoins-unrealised-on-chain-profit-loss-73e735020c8d"),
    ("Forecasting Bitcoin Volatility Spikes from Whale Transactions and CryptoQuant Data", "https://ar5iv.labs.arxiv.org/html/2211.08281"),
    ("From On‑chain to Macro (VLDB Workshop)", "https://vldb.org/workshops/2024/proceedings/FAB/FAB-6.pdf"),
    ("Glassnode Finance Bridge • Edition 5", "https://insights.glassnode.com/finance-bridge-edition-5/"),
    ("Hybrid Time-Series and On-Chain Indicators for Bitcoin Forecasting", "https://arxiv.org/pdf/2204.12928"),
    ("Kaggle LTR Baseline", "https://github.com/Zhenye-Na/kaggle-learning-to-rank"),
    ("Learning to Rank Time-Series Data", "https://arxiv.org/abs/2006.04988"),
    ("Learning to Rank for Information Retrieval (Liu, 2009)", "https://www.nowpublishers.com/article/Details/INR-016"),
    ("LightGBM Docs – Ranking", "https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective"),
    ("Market Regime Classification Using Correlation Networks (UNC)", "https://econ.unc.edu/wp-content/uploads/sites/1423/2025/03/2019-Mayo-Zhu_Nam.pdf"),
    ("Market Regime Classification with Signatures (Bilokon et al.)", "https://arxiv.org/abs/2107.00066"),
    ("Market Regime Detection via Realized Covariances", "https://arxiv.org/abs/2104.03667"),
    ("Master Thesis: Predictive Models for Cryptocurrency Markets", "https://pure.tue.nl/ws/portalfiles/portal/341640654/Master_Thesis_Nikhil_Razab-Sekh.pdf"),
    ("ONCHAIN ANALYTICS – A New Methodology for Cryptocurrency Analysis", "https://www.theseus.fi/bitstream/handle/10024/879592/Nguyen_Quyen.pdf?sequence=2&isAllowed=y"),
    ("On the Importance of Social Signals for Bitcoin Trading", "https://arxiv.org/pdf/1506.01513"),
    ("On-Chain Metric Investing Strategy with Glassnode API", "https://alpaca.markets/learn/on-chain-metric-investing-strategy-with-glassnodeapi"),
    ("Predicting Cryptocurrencies Market Phases through On-Chain Data", "https://iris.unito.it/retrieve/2927f4b0-43f0-46fd-a840-069dd1ec61c0/6.%20ICBC23%20-%20PREDICTING%20BTC.pdf"),
    ("Portfolio Optimization with Market Regimes (ResearchGate)", "https://www.researchgate.net/publication/390055790_Quantitative_Portfolio_Optimization_Framework_with_Market_Regimes_Classification_Probabilistic_Time_Series_Forecasting_and_Hidden_Markov_Models"),
    ("Regime Switching Forecasting for Crypto (Springer, 2024)", "https://link.springer.com/article/10.1007/s42521-024-00123-2"),
    ("Sliced Wasserstein k-means Clustering for Regimes", "https://arxiv.org/abs/2310.01285"),
    ("TensorFlow Ranking (Google)", "https://github.com/tensorflow/ranking"),
    ("The Ultimate Guide to Bitcoin Whales (Glassnode)", "https://insights.glassnode.com/content/the-ultimate-guide-to-bitcoin-whales-2"),
    ("Understanding ML Algorithms in Crypto Trading (3Commas)", "https://3commas.io/blog/understanding-machine-learning-algorithms-in-crypt"),
    ("Uncovering the Determinants of Crypto-Asset Prices via Machine Learning", "https://arxiv.org/pdf/2201.12893"),
    ("Weighted Ensemble – Predicting Crypto Prices", "https://medium.com/bitgrit-data-science-publication/predict-cryptocurrency-prices-with-data-science-0ea732a0ea0f"),
    ("XGBoost LTR Tutorial", "https://xgboost.readthedocs.io/en/stable/tutorials/ranking.html"),
    ("Best Feature Selection using Correlation Analysis for Prediction of Bitcoin Transaction Count (APNOMS)", "https://nmlab.korea.ac.kr/publication/published.papers/2019/2019.09-Feature_Selection_for_Prediction_of_Bitcoin_Transaction_Count-APNOMS2019.pdf")
]

async def download_all():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        for i, (title, url) in enumerate(references, start=1):
            safe_title = title.replace(" ", "_").replace("/", "_").replace(":", "").replace("–", "-")
            path = f"downloads/{i:02d}_{safe_title}.pdf"

            print(f"Processing: {title}")
            page = await context.new_page()
            try:
                await page.goto(url, timeout=60000)
                await page.pdf(path=path, format="A4", print_background=True)
                print(f"Saved PDF: {path}")
            except Exception as e:
                print(f"Error processing {title}: {e}")
            finally:
                await page.close()

        await browser.close()

asyncio.run(download_all())