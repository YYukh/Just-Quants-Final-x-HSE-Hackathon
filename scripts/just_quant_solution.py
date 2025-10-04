# scripts/just_quant_solution.py

import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import class_weight
from xgboost import XGBClassifier

# Предполагается, что эти модули находятся в src/
from src.feature_eng import generate_features
from src.labeling import three_barrier_std, barrier_std


class JustQuantSolution:
    def __init__(self, max_period=20):
        self.max_period = max_period
        self._setup_sentiment_model()

    def _setup_sentiment_model(self):
        print("📥 Загрузка модели сентимента...")
        sentiment_model_name = "blanchefort/rubert-base-cased-sentiment"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Используется устройство: {self.device}")
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()
        print("✅ Модель сентимента готова!")

    def __call__(
        self,
        train_path: str,
        train_path_news: str,
        test_path: str,
        test_path_news: str,
        output_path: str
    ):
        # === 1. Загрузка данных ===
        train_df = pd.read_csv(train_path)
        train_df.index = pd.to_datetime(train_df['begin'])

        test_df = pd.read_csv(test_path)
        test_df.index = pd.to_datetime(test_df['begin'])

        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df = full_df.sort_values(['ticker', 'begin'])
        full_df.index = pd.to_datetime(full_df['begin'])
        full_df.drop(columns='begin', inplace=True)

        news_train = pd.read_csv(train_path_news)
        if 'Unnamed: 0' in news_train.columns:
            news_train.drop(columns=['Unnamed: 0'], inplace=True)
        news_train.index = pd.to_datetime(news_train['publish_date'])
        news_train.drop(columns=['publish_date'], inplace=True)
        news_train.index.rename('begin', inplace=True)

        news_test = pd.read_csv(test_path_news)
        news_test.index = pd.to_datetime(news_test['publish_date'])
        news_test.drop(columns=['publish_date'], inplace=True)
        news_test.index.rename('begin', inplace=True)

        news_full_df = pd.concat([news_train, news_test])

        # === 2. Конфигурация ===
        COMPANY_TO_TICKER = {
            # === Банки ===
            "сбербанк": "SBER", "сбер": "SBER", "сбербанк россии": "SBER", "пао сбербанк": "SBER",
            "sberbank": "SBER", "sber": "SBER",
            "втб": "VTBR", "банк втб": "VTBR", "пао втб": "VTBR", "vtb": "VTBR",
            "газпромбанк": "GPB", "гпб": "GPB", "gazprombank": "GPB",
            "альфа-банк": "ALBK", "альфабанк": "ALBK", "alfabank": "ALBK",
            "росбанк": "ROSB", "rosbank": "ROSB",
            "открытие": "OPMD", "банк открытие": "OPMD", "фк открытие": "OPMD",
            "сбербанк страхование": "SGZH", "сгж": "SGZH", "sgzh": "SGZH",

            # === Энергетика и нефть ===
            "газпром": "GAZP", "пао газпром": "GAZP", "gazprom": "GAZP",
            "роснефть": "ROSN", "пао роснефть": "ROSN", "rosneft": "ROSN",
            "лукойл": "LKOH", "lukoil": "LKOH",
            "новатэк": "NVTK", "пао новатэк": "NVTK", "novatek": "NVTK",
            "сургутнефтегаз": "SNGS", "снг": "SNGS", "surgutneftegas": "SNGS",
            "татнефть": "TATN", "оао татнефть": "TATN", "tatneft": "TATN",
            "транснефть": "TRNF", "transneft": "TRNF",
            "интер рао": "IRAO", "интеррао": "IRAO", "inter rao": "IRAO", "inter-rao": "IRAO",
            "мосэнерго": "MSNG", "msk энерго": "MSNG", "mosenergo": "MSNG",
            "русгидро": "HYDR", "rusgidro": "HYDR",
            "фск еэс": "FEES", "россети": "RSTI", "rosseti": "RSTI",

            # === Металлы и горнодобыча ===
            "норникель": "GMKN", "норильский никель": "GMKN", "mmk": "MAGN",
            "ммк": "MAGN", "магнитогорский металлургический комбинат": "MAGN",
            "северсталь": "CHMF", "severstal": "CHMF",
            "нлмк": "NLMK", "новолипецкий металлургический комбинат": "NLMK",
            "евраз": "EVRG", "evraz": "EVRG",
            "полюс": "PLZL", "polyus": "PLZL",
            "алроса": "ALRS", "alrosa": "ALRS",
            "уралкалий": "URKA", "uralkali": "URKA",
            "фосагро": "PHOR", "phosagro": "PHOR",
            "мечел": "MTLR", "mechel": "MTLR",

            # === Телеком и технологии ===
            "мтс": "MTSS", "mts": "MTSS", "пао мтс": "MTSS",
            "билайн": "VEON", "вымпелком": "VEON", "beeline": "VEON",
            "мегафон": "MFON", "megafon": "MFON",
            "яндекс": "YNDX", "yandex": "YNDX",
            "vk": "VKCO", "вк": "VKCO", "вконтакте": "VKCO", "mail.ru": "VKCO",
            "ozon": "OZON", "озон": "OZON",
            "wildberries": "WB", "вайлдберриз": "WB", "вб": "WB",
            "дзен": "YNDX",  # часть Яндекса

            # === Ритейл и потребительский сектор ===
            "магнит": "MGNT", "magnit": "MGNT",
            "лента": "LNTA", "lenta": "LNTA",
            "х5 ритейл": "FIVE", "пятёрочка": "FIVE", "перекрёсток": "FIVE", "x5 retail": "FIVE",
            "глобус": "GLTR", "globus": "GLTR",
            "дикси": "DIXY", "dixy": "DIXY",
            "эльдорадо": "SALM", "eldorado": "SALM",  # через Салым
            "сбермаркет": "SBER",  # часть Сбера
            "авито": "YNDX",  # принадлежит Яндексу

            # === Транспорт и логистика ===
            "аэрофлот": "AFLT", "aeroflot": "AFLT",
            "сберлогистика": "SBER",  # пока не отдельный тикер
            "дельта": "TGKA", "delta": "TGKA",  # ТГК-1
            "глобалтранс": "GLTR",  # осторожно: конфликт с Globus (ритейл), но тикер другой
            "нма": "NMTP", "нмтп": "NMTP", "новороссийский морской торговый порт": "NMTP",

            # === Финансы и биржи ===
            "мосбиржа": "MOEX", "московская биржа": "MOEX", "moex": "MOEX",
            "спб биржа": "SPBE", "spb exchange": "SPBE", "спбexchange": "SPBE",
            "афк система": "AFKS", "система": "AFKS", "afk sistema": "AFKS",
            "самолет": "SMLT", "samolet": "SMLT",
            "пионер": "PIKK", "pik": "PIKK",

            # === Холдинги и диверсифицированные группы ===
            "севергрупп": "SVCB",  # Северсталь-групп
            "роснефтегаз": "РНГ",  # не торгуется отдельно

            # === Английские и транслит-варианты ===
            "gazprom neft": "SIBN", "газпром нефть": "SIBN", "сургутнефть": "SNGS",
            "tatneft": "TATN", "lukoil": "LKOH", "novatek": "NVTK",
            "polyus gold": "PLZL", "alrosa": "ALRS", "phosagro": "PHOR",
            "severstal": "CHMF", "nlmk": "NLMK", "evraz": "EVRG",
            "mts russia": "MTSS", "veon": "VEON", "megafon": "MFON",
            "ozon holdings": "OZON", "wild berries": "WB",
            "yandex nv": "YNDX", "vk group": "VKCO",
            "magnit retail": "MGNT", "lenta group": "LNTA",
            "x5 group": "FIVE", "aeroflot group": "AFLT",
        }

        ALLOWED_TICKERS_SET = set(COMPANY_TO_TICKER.values())
        MACRO_KEYWORDS = [
            "инфляция", "ключевая ставка", "цб", "банк россии", "ввп", "безработица",
            "курс рубля", "дефицит", "бюджет", "импорт", "экспорт", "ликвидность",
            "рынок труда", "промышленность", "потребительские цены", "ипц"
        ]

        # === 3. Вспомогательные функции ===
        def normalize_text(text: str) -> str:
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def detect_ticker(title: str, text: str) -> str:
            content = normalize_text(f"{title} {text}")
            for variant, ticker in COMPANY_TO_TICKER.items():
                if variant in content:
                    return ticker
            return "none"

        def detect_macro(title: str, text: str) -> bool:
            content = normalize_text(f"{title} {text}")
            return any(kw in content for kw in MACRO_KEYWORDS)

        def get_sentiment_batch(texts, batch_size=256):
            labels = ["negative", "neutral", "positive"]
            results = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Сентимент"):
                batch = texts[i:i + batch_size]
                inputs = self.sentiment_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                with torch.no_grad():
                    logits = self.sentiment_model(**inputs).logits
                preds = torch.argmax(logits, dim=-1)
                results.extend([labels[p] for p in preds.cpu().numpy()])
            return results

        # === 4. Обработка новостей ===
        print("🔍 Анализ тикеров и макро...")
        news_analyze = news_full_df.copy()
        news_analyze['ticker'] = [
            detect_ticker(row.title, row.publication)
            for row in tqdm(news_analyze.itertuples(), total=len(news_analyze), desc="Тикеры")
        ]

        news_analyze['is_macro'] = [
            detect_macro(row.title, row.publication)
            for row in tqdm(news_analyze.itertuples(), total=len(news_analyze), desc="Макро")
        ]

        print("🧠 Анализ сентимента...")
        news_analyze['sentiment'] = get_sentiment_batch(
            [f"{row.title}. {row.publication}" for row in news_analyze.itertuples()],
            batch_size=256
        )

        news_analyze.index = news_analyze.index - pd.Timedelta(days=1)
        news_df = news_analyze.copy()
        news_df.index = pd.to_datetime(news_df.index)

        sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        news_df['sentiment_encoded'] = news_df['sentiment'].map(sentiment_map)
        news_df['is_macro_encoded'] = np.where(news_df['is_macro'], news_df['sentiment_encoded'], 0)
        news_df['date'] = news_df.index.date

        news_agg = news_df.groupby(['date', 'ticker']).agg({
            'sentiment_encoded': 'mean',
            'is_macro_encoded': 'mean'
        }).reset_index()
        
        prices_df = full_df.reset_index()
        prices_df['begin'] = pd.to_datetime(prices_df['begin']).dt.date

        merged_df = prices_df.merge(
            news_agg,
            left_on=['begin', 'ticker'],
            right_on=['date', 'ticker'],
            how='left'
        )

        merged_df['sentiment_encoded'] = merged_df['sentiment_encoded'].fillna(0)
        merged_df['is_macro_encoded'] = merged_df['is_macro_encoded'].fillna(0)
        merged_df = merged_df.drop(columns=['date'])
        merged_df.set_index('begin', inplace=True)
        merged_df.index = pd.to_datetime(merged_df.index)

        full_df_clean = merged_df.reset_index().drop_duplicates(subset=['begin', 'ticker'], keep='last').set_index('begin')
        df_pivot = full_df_clean.set_index('ticker', append=True).unstack('ticker')
        df_pivot.ffill(inplace=True)

        close = df_pivot['close']
        X = generate_features(df_pivot).stack().sort_index()
        X.dropna(inplace=True)

        # === 5. Генерация лейблов и волатильности ===
        y_dict = {}
        vol_dict = {}

        close.index = pd.to_datetime(close.index)

        for period in range(1, self.max_period+1):
            print(f"Generating labels and vol for period = {period}")
            tripple_barrier_label = close.apply(
                three_barrier_std,
                ptSl=[1, 1],
                rolling_n=50,
                base_scaling_factor=2.0,
                base_period=10,
                period=period
            )
            y_raw = tripple_barrier_label.stack()
            y = y_raw.reindex(X.index).dropna()
            y = y + 1  # -1 → 0, 0 → 1, 1 → 2
            y_dict[period] = y

            barrier = close.apply(
                barrier_std,
                ptSl=[1, 1],
                rolling_n=50,
                base_scaling_factor=2.0,
                base_period=10,
                period=period
            )
            vol_raw = barrier.stack()
            vol = vol_raw.reindex(X.index).dropna()
            vol_dict[period] = vol

        y_df = pd.DataFrame({f'{p}': y for p, y in y_dict.items()})
        vol_df = pd.DataFrame({f'{p}': v for p, v in vol_dict.items()})

        train_end = train_df.index[-1]
        test_start = test_df.index[0]
        X_train = X[:train_end]
        X_test = X[test_start:]
        y_train = y_df[:train_end]
        y_test = y_df[test_start:]
        vol_train = vol_df[:train_end]
        vol_test = vol_df[test_start:]

        # === 6. Оптимизация и обучение ===
        sample_weights = class_weight.compute_sample_weight("balanced", y_train['1'])

        def objective(trial):
            params = {
                'objective': 'multi:softmax',
                'num_class': 3,
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1),
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }

            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx]['1'], y_train.iloc[val_idx]['1']
                fold_sample_weights = class_weight.compute_sample_weight("balanced", y_tr)
                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr, sample_weight=fold_sample_weights)
                y_pred = model.predict(X_val)
                score = balanced_accuracy_score(y_val, y_pred)
                scores.append(score)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        best_params = study.best_params
        best_model_base = XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss',
            **best_params
        )

        # === 7. Предсказание для всех периодов ===
        index_common = X_test.index
        pred_classes = pd.DataFrame(index=index_common)
        pred_proba_all = pd.DataFrame(index=index_common)

        for period in range(1, self.max_period+1):
            print(f"Training on period = {period}")
            y_tr = y_train[f'{period}']
            y_te = y_test[f'{period}']

            X_tr_aligned = X_train.reindex(y_tr.index)
            weights = class_weight.compute_sample_weight("balanced", y_tr)

            model = deepcopy(best_model_base)
            model.fit(X_tr_aligned, y_tr, sample_weight=weights)

            X_te_aligned = X_test.reindex(y_te.index)
            y_pred = model.predict(X_te_aligned)
            y_pred_proba = model.predict_proba(X_te_aligned)

            pred_classes[str(period)] = pd.Series(y_pred, index=y_te.index)
            pred_proba_all[f'prob_{period}_class_0'] = pd.Series(y_pred_proba[:, 0], index=y_te.index)
            pred_proba_all[f'prob_{period}_class_1'] = pd.Series(y_pred_proba[:, 1], index=y_te.index)
            pred_proba_all[f'prob_{period}_class_2'] = pd.Series(y_pred_proba[:, 2], index=y_te.index)

        pred_classes = pred_classes.reindex(index_common)
        pred_proba_all = pred_proba_all.reindex(index_common)

        pred_returns = y_test.copy()
        for period in range(1, self.max_period+1):
            pred_returns[f'{period}'] = np.where(
                pred_classes[f'{period}'] == 2,
                vol_test[f'{period}'] * pred_proba_all[f'prob_{period}_class_2'],
                np.where(
                    pred_classes[f'{period}'] == 0,
                    -vol_test[f'{period}'] * pred_proba_all[f'prob_{period}_class_0'],
                    vol_test[f'{period}'] * (pred_proba_all[f'prob_{period}_class_2'] - pred_proba_all[f'prob_{period}_class_0'])
                )
            )

        last_date = pred_returns.index.get_level_values('begin').max()
        sub_df = pred_returns.xs(last_date, level='begin')
        sub_df.to_csv(output_path)
        print(f"✅ Результат сохранён в {output_path}")