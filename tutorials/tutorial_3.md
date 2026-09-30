# Туториал 3. Baseline: модель, валидация, первый сабмит

(~60-90 мин)

Цель этого этапа — не хороший score, а **рабочий пайплайн от сырых данных до
сабмита на Kaggle**. Пока модель максимально простая и "сырая" — без исправления
выброса в `EstimatedSalary` из туториала 2 (это будет в туториале 4). Baseline даёт
точку отсчёта: любое улучшение в следующих туториалах будет сравниваться именно с
этим числом.

Код пишется в новом ноутбуке `notebooks/02_baseline.ipynb`.

## Шаг 1. Отбор признаков

```python
feature_cols = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
target_col = "Exited"

X = train[feature_cols]
y = train[target_col]
```

`id`, `CustomerId`, `Surname` сознательно не включены: это идентификаторы, у них
нет причинной связи с оттоком, а `Surname` вдобавок текстовый и с сотнями уникальных
значений — модель может случайно "выучить" ложную корреляцию по фамилии на train и
провалиться на test (классический пример переобучения на шуме).

## Шаг 2. Train/val split со стратификацией

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
```

`stratify=y` — обязателен из-за дисбаланса классов (80/20 из туториала 2): без
него случайный сплит может дать валидационную выборку с другим соотношением классов,
и ROC AUC на ней будет не совсем тем, что вы получите на реальном лидерборде.
`random_state=42` — не магическое число, а просто фиксация, чтобы результат был
воспроизводим между перезапусками ноутбука.

## Шаг 3. Pipeline: кодирование + модель

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

categorical_cols = ["Geography", "Gender"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
```

Почему `Pipeline`, а не сделать `OneHotEncoder`/`StandardScaler` руками на всём
датасете заранее: `StandardScaler` внутри `Pipeline` вызывает `.fit()` только на
`X_train` при обучении и переиспользует эти же параметры (среднее/std) для
`X_val` и `test` через `.transform()`. Если отмасштабировать весь датасет
целиком до сплита — валидационные данные незаметно "подсмотрят" статистики,
посчитанные с их же участием. Это и есть **data leakage**: на валидации результат
будет казаться чуть лучше, чем модель покажет на реально новых данных.

`OneHotEncoder(drop="first")` — убирает одну категорию на признак (например, для
`Gender` останется только колонка "is Male"), чтобы не дублировать информацию
(если не Female и не "другое", то по остаточному принципу — Male).

`max_iter=1000` — `LogisticRegression` по умолчанию может не успеть сойтись за
100 итераций на этих данных и выдаст warning; это не ошибка, а сигнал увеличить лимит.

## Шаг 4. Оценка на валидации: ROC AUC

```python
from sklearn.metrics import roc_auc_score

val_proba = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_proba)
print(f"Validation ROC AUC: {val_auc:.4f}")
```

`predict_proba(X_val)[:, 1]` — вероятность класса `1` (`Exited`), не бинарное
предсказание. Это важно: `model.predict(X_val)` вернёт 0/1 по порогу 0.5, а ROC AUC
меряет качество ранжирования по вероятностям на **всех** порогах сразу — если
скормить ей 0/1, метрика станет куда грубее и менее информативной. Соревнование
тоже требует именно вероятности в сабмите — не случайно.

Ориентир: baseline логрегрессия на этой задаче обычно даёт ROC AUC в районе
0.75-0.85 (нет гарантий — зависит от случайного сплита), топ-10 лидерборда — это
~0.933+. Разрыв закроют туториалы 4-6 (feature engineering, градиентный бустинг,
тюнинг).

## Шаг 5. Инференс на test.csv и файл сабмита

```python
test_proba = model.predict_proba(test[feature_cols])[:, 1]

submission = pd.DataFrame({
    "id": test["id"],
    "Exited": test_proba,
})
submission.to_csv("../submissions/baseline_logreg.csv", index=False)
```

Обратите внимание: `model` обучен на `X_train` (80% train), а не на всём `train` —
для первого сабмита это нормально (важнее проверить, что пайплайн работает
end-to-end), но имейте в виду: дообучение финальной модели на 100% train перед
последним сабмитом — стандартная практика, к ней вернёмся в туториале 8.

Папка `submissions/` ещё не существует — создайте её (`mkdir submissions` в
терминале или через проводник); она уже в `.gitignore`.

## Шаг 6. Сабмит на Kaggle

```powershell
kaggle competitions submit -c predicting-customer-churn-a-real-world-binary-classification-challeng -f submissions/baseline_logreg.csv -m "baseline: logreg + onehot + standard scaler"
```

Разбор: `-f` — какой файл отправить, `-m` — комментарий к сабмиту (видно потом в
истории сабмитов на сайте). Проверить результат:

```powershell
kaggle competitions submissions -c predicting-customer-churn-a-real-world-binary-classification-challeng
```

## Шаг 7. Сверить локальный и реальный score

Сравните `val_auc` из шага 4 с score, который показал Kaggle. Они не обязаны
совпадать один в один, но должны быть близки (разница в пределах 0.01-0.02 —
нормальный шум от разных выборок). Если разница намного больше — это повод
насторожиться: возможна утечка данных в валидации, слишком маленькая/нерепрезентативная
`X_val`, или различия в распределениях train/test (но в туториале 2 мы это уже
проверили для нескольких признаков и разницы не увидели).

## Шаг 8. Зафиксировать baseline

В последней markdown-ячейке ноутбука запишите: модель, локальный ROC AUC, score на
Kaggle, дата. Например:

```
Baseline: LogisticRegression + OneHot + StandardScaler
Локальный ROC AUC (val): 0.XXXX
Kaggle score: 0.XXXX
```

Это не бюрократия — без записанной точки отсчёта сложно объективно понять, дал ли
эффект feature engineering в туториале 4, или показалось.

## Как использовать Claude Code на этом этапе

- Пишите пайплайн по частям и просите ревью: *"Я собрал Pipeline с OneHotEncoder и
  LogisticRegression — вот код, нет ли утечки данных?"*
- Если `roc_auc_score` выдаёт неожиданно низкое число (~0.5) — это частый признак,
  что в `predict_proba` перепутаны столбцы (`[:, 0]` вместо `[:, 1]`) — хороший
  повод спросить Claude Code, но сначала попробуйте найти причину сами.
- Спросите "почему": *"Почему OneHotEncoder(drop='first'), а не просто
  OneHotEncoder() без параметров?"*

## Проверка (пройдите по порядку)

1. `X`/`y` не содержат `id`, `CustomerId`, `Surname`.
2. `train_test_split` вызван с `stratify=y`.
3. `Pipeline` объединяет препроцессинг и модель в один объект.
4. Посчитан `roc_auc_score` на `predict_proba`, а не на `predict`.
5. `submissions/baseline_logreg.csv` создан и по структуре совпадает с
   `sample_submission.csv` (`id`, `Exited`).
6. Сабмит на Kaggle прошёл, score получен.
7. Локальный и реальный ROC AUC близки (или объяснена причина расхождения).
8. Baseline score записан в ноутбуке.

Следующий шаг — tutorial_4.md (feature engineering: исправляем выброс в
`EstimatedSalary`, кодируем признаки основательнее, пробуем новые фичи).
