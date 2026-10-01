# Туториал 5. Градиентный бустинг и кросс-валидация

(~90-120 мин)

Сейчас лучший результат — логрегрессия с ручным feature engineering: 0.9246
локально / 0.92871 приватный score (туториал 4). До топ-10 (≈0.933+) осталось
немного. Цель этого этапа — закрыть разрыв через смену модели на градиентный
бустинг и более надёжную схему валидации.

Сначала:

```powershell
pip install -r requirements.txt
```

(добавлен `catboost`). Код — в новом ноутбуке `notebooks/04_boosting.ipynb`.

## Шаг 1. Почему градиентный бустинг

`LogisticRegression` моделирует только линейную (после кодирования — почти
линейную) зависимость от каждого признака. В туториале 4 пришлось **вручную**
превращать `NumOfProducts` в категорию, чтобы модель увидела нелинейный провал на
значении 2. Деревья решений устроены иначе: каждый узел дерева сам находит порог
("если `NumOfProducts` >= 3, то..."), поэтому нелинейности и пороговые эффекты
им не нужно подсказывать вручную.

**Градиентный бустинг** строит деревья последовательно: каждое следующее дерево
обучается предсказывать *ошибки* (точнее, градиент функции потерь) предыдущих
деревьев, а не целевую переменную с нуля. Результат — ансамбль из сотен неглубоких
деревьев, который на практике почти всегда сильнее одного дерева и часто сильнее
линейных моделей на табличных данных вроде этой.

## Шаг 2. CatBoost и категориальные признаки

Берём `CatBoostClassifier` — он умеет работать с категориальными признаками
напрямую, без one-hot/target encoding (это то, чем в туториале 4 занимались
руками для `Geography`). Единственное требование — явно перечислить, какие
колонки категориальные, и они должны быть типа `str`/`int`, не `float`:

```python
from catboost import CatBoostClassifier

feature_cols = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
cat_features = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]

X = train[feature_cols].copy()
y = train["Exited"].copy()

for col in cat_features:
    if pd.api.types.is_float_dtype(X[col]):
        X[col] = X[col].astype(int)
```

`NumOfProducts`, `HasCrCard`, `IsActiveMember` в файле — `float` (1.0, 0.0 и т.п.),
хотя по смыслу это категории/флаги — CatBoost с `float` в `cat_features` упадёт с
ошибкой, поэтому конвертируем в `int` явно. **Осторожно с проверкой типа**:
`X[col].dtype != "object"` — плохая проверка для этого случая, в некоторых версиях
pandas строковые колонки имеют dtype `str`, а не `object`, и такая проверка
попытается привести `Geography`/`Gender` к `int` и упадёт с ошибкой вида
`invalid literal for int() with base 10: 'France'`. Надёжнее — явно проверять
числовой тип: `pd.api.types.is_float_dtype(X[col])`.

## Шаг 3. StratifiedKFold вместо одного split

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # обучение модели — шаг 4
```

Один `train_test_split` (туториалы 3-4) даёт **одну** оценку ROC AUC, которая
зависит от того, какие именно 20% строк случайно попали в валидацию — при другом
`random_state` число может ощутимо измениться. K-Fold разбивает данные на 5 частей,
и **каждая** по очереди становится валидацией, пока остальные 4 — обучением.
В сумме вы получаете **out-of-fold (OOF) предсказание** для каждой строки train
(она была в валидации ровно один раз) — это позволяет посчитать ROC AUC на всём
train целиком, а не на случайных 20%, и даёт более стабильную оценку. OOF-предсказания
пригодятся в туториале 6 для ансамблей.

`stratify` в `train_test_split` был явным параметром, у `StratifiedKFold` он в
названии — сохраняет пропорцию классов (80/20) в каждом фолде, как и раньше.

## Шаг 4. Обучение модели с early stopping внутри цикла

```python
from sklearn.metrics import roc_auc_score

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
    )
    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_scores.append(fold_auc)
    print(f"Fold {fold}: ROC AUC = {fold_auc:.4f}, best_iteration = {model.get_best_iteration()}")

print(f"\nOOF ROC AUC: {roc_auc_score(y, oof_preds):.4f}")
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
```

`iterations=1000` — верхний лимит числа деревьев, `early_stopping_rounds=50` —
если 50 итераций подряд не улучшают `eval_metric` на `eval_set`, обучение
останавливается раньше — защита от переобучения и от лишнего ожидания.
`get_best_iteration()` — на каком дереве остановились фактически, пригодится для
финального обучения в шаге 6.

`OOF ROC AUC` и `Mean fold ROC AUC` обычно близки, но не идентичны — первое
считается на объединённых предсказаниях всех фолдов сразу, второе — среднее по
отдельным оценкам каждого фолда. Большое расхождение между ними — сигнал, что
фолды сильно неоднородны.

## Шаг 5. Сравнение: сырые признаки vs признаки туториала 4

Прогоните шаги 2-4 дважды: сначала на "сырых" колонках (как выше), потом добавив
`AgeGroup` и `BalanceSalaryRatio` из туториала 4 (клип `EstimatedSalary` тоже можно
оставить — выбросам деревья менее чувствительны, чем `StandardScaler`, но не
повредит). Сравните OOF ROC AUC в обоих случаях.

Возможные исходы — оба поучительны:
- Признаки почти не изменили score → бустинг сам нашёл нужные пороги без подсказок
  (ожидаемо для `AgeGroup` — дерево само умеет резать `Age` на интервалы).
- Признаки всё ещё немного помогли → значит, в них есть информация, которую
  дереву сложнее найти самостоятельно за ограниченное число итераций.

## Шаг 6. Финальная модель и сабмит

```python
best_iterations = [m.get_best_iteration() for m in models]  # сохраните models в шаге 4
final_iterations = int(np.mean(best_iterations))

final_model = CatBoostClassifier(
    iterations=final_iterations,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=False,
)
final_model.fit(X, y, cat_features=cat_features)

test_proba = final_model.predict_proba(test[feature_cols])[:, 1]
submission = pd.DataFrame({"id": test["id"], "Exited": test_proba})
submission.to_csv("../submissions/catboost_cv.csv", index=False)
```

Финальная модель обучается на **всех** размеченных данных (100% train, не 80%) —
с фиксированным числом итераций, усреднённым по фолдам (без `eval_set`, потому что
отдельной валидации для финальной модели уже не остаётся — весь train идёт в дело).

Отправьте сабмит и сравните с текущим лучшим (0.92871 приватный score).

## Как использовать Claude Code на этом этапе

- Если CatBoost падает с ошибкой на `cat_features` — покажите полный текст ошибки,
  почти всегда дело в типе колонки (`float` вместо `int`/`str`).
- Спросите "почему": *"Почему early_stopping_rounds=50, а не 10 или 200?"* —
  хороший повод обсудить компромисс между скоростью и риском остановиться слишком
  рано на шумном фолде.
- Не просите "напиши весь пайплайн с бустингом" одним запросом — соберите цикл
  по фолдам сами, по кусочкам, тем более что здесь легко случайно перепутать
  индексы `train_idx`/`val_idx` и не заметить это без построчного разбора.

## Проверка (пройдите по порядку)

1. Категориальные колонки приведены к `int`/`str`, не `float`.
2. `StratifiedKFold(n_splits=5)` используется вместо одного `train_test_split`.
3. Для каждого фолда есть `fold_auc` и `best_iteration`, напечатанные в процессе.
4. Посчитан `OOF ROC AUC` на всех 15000 строках train.
5. Сравнены сырые признаки и признаки туториала 4 — известно, помогли они бустингу
   или нет.
6. Финальная модель обучена на 100% train с усреднённым числом итераций.
7. Сабмит отправлен, score сравнён с 0.92871.

Следующий шаг — tutorial_6.md (тюнинг гиперпараметров и ансамбли — рассчитано на то,
что после этого туториала цель топ-10 уже либо достигнута, либо очень близка).
