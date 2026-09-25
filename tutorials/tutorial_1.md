# Туториал 1. Окружение, Kaggle API и данные

(~45-75 мин)

Этот этап готовит всё, что понадобится в любом Kaggle-проекте: изолированное
окружение, доступ к данным соревнования через Kaggle API и первый коммит. Ничего
специфичного для churn-модели тут ещё нет — сами данные посмотрим "одним глазом" в
конце, полноценный EDA будет в туториале 2.

Все команды ниже — для терминала **PowerShell внутри VS Code**. Открыть:
`Ctrl + ~` (тильда) или *Terminal → New Terminal*. Важно, чтобы терминал был открыт
именно в папке `kaggle_project` — проверить можно командой `pwd`.

## Шаг 1. Проверить Python и создать venv

```powershell
python --version
```

Должна показаться версия 3.11.x. Дальше — как обычно, изолированное окружение на
проект (если делали это в KidsMap Moscow — здесь та же логика, просто отдельная
папка `.venv` для этого проекта):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

После активации в строке терминала должен появиться префикс `(.venv)`. Если
PowerShell ругается на `execution of scripts is disabled` — один раз выполните
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, подтвердите `Y` и повторите
активацию.

В VS Code отдельно выберите интерпретатор: `Ctrl+Shift+P` → *Python: Select
Interpreter* → вариант с `.venv` в пути.

## Шаг 2. requirements.txt — только то, что нужно для старта

Для этого этапа нужны три пакета: `pandas` (посмотреть на данные), `jupyter`
(ноутбуки для EDA/экспериментов), `kaggle` (официальный CLI для скачивания данных и
сабмитов). Остальное (`scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`, `shap`)
добавим на туториалах, где они реально понадобятся — не тащим лишнее заранее.

Попросите Claude Code создать `requirements.txt` с этими тремя пакетами, затем:

```powershell
pip install -r requirements.txt
```

## Шаг 3. Аккаунт на Kaggle и API-токен

Если аккаунта на kaggle.com ещё нет — зарегистрируйтесь (email или Google).
Дальше нужен персональный API-токен, чтобы качать данные из терминала, а не руками
через браузер:

1. Зайдите на kaggle.com → аватар в правом верхнем углу → **Settings**.
2. Раздел **API** → кнопка **Create New Token**. Kaggle покажет токен вида
   `KGAT_...` **один раз** — скопируйте его сразу, повторно посмотреть не выйдет
   (можно только сгенерировать новый, отозвав старый). Это текущий формат токена
   Kaggle API (более старый вариант — файл `kaggle.json` с username+key — тоже
   ещё поддерживается некоторыми версиями CLI, но новый способ проще).
3. Сохраните токен в `~/.kaggle/access_token` — именно там его ищет клиент:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "KGAT_ваш_токен" -NoNewline
```

**Важно:** этот токен — секрет уровня пароля (даёт доступ к вашему аккаунту).
Он лежит вне папки проекта (`%USERPROFILE%\.kaggle\`), так что случайно
закоммитить его сложно — но если вы, например, вставляли токен в какой-то текстовый
файл внутри проекта, чтобы не потерять, обязательно удалите этот файл после того,
как перенесли токен в `~/.kaggle/access_token`, и убедитесь, что он не попал в
`git add` (в `.gitignore` этого проекта уже есть строка `keys.txt` на всякий случай).

4. Проверка, что CLI видит токен:

```powershell
kaggle competitions list
```

Если видите список соревнований (а не ошибку авторизации) — всё настроено.

## Шаг 4. Принять правила соревнования

Соревнование хоть и завершено (режим Late Submission), но для скачивания данных и
отправки сабмитов через API обычно всё равно нужно один раз нажать **Join
Competition** / принять правила на самой странице соревнования в браузере:
https://www.kaggle.com/competitions/predicting-customer-churn-a-real-world-binary-classification-challeng/rules

Если попробуете скачать данные без этого — CLI ответит ошибкой 403.

## Шаг 5. Скачать данные

```powershell
kaggle competitions download -c predicting-customer-churn-a-real-world-binary-classification-challeng -p data/raw
```

Разбор: `-c <slug>` — какое соревнование, `-p data/raw` — куда положить файл.
Скачается архив (обычно `.zip`) — распакуйте его в ту же папку `data/raw/` (правой
кнопкой → *Extract All* в проводнике, либо `Expand-Archive` в PowerShell). После
распаковки в `data/raw/` должны появиться три файла: `train.csv`, `test.csv`,
`sample_submission.csv`.

## Шаг 6. .gitignore — что не должно попадать в git

Сырые данные соревнования не коммитятся: они большие, легко перекачиваются заново
командой из шага 5, и в некоторых соревнованиях запрещены к публикации правилами.
Попросите Claude Code создать `.gitignore` для Python-проекта, включающий как
минимум:

```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/
submissions/
```

## Шаг 7. Первый взгляд на данные (не EDA, а просто sanity check)

Откройте Jupyter (`jupyter notebook` в терминале с активным `.venv`, либо создайте
ноутбук прямо в VS Code) и создайте `notebooks/01_eda.ipynb` — пока с парой ячеек,
просто чтобы убедиться, что всё скачалось и читается:

```python
import pandas as pd

train = pd.read_csv("../data/raw/train.csv")
test = pd.read_csv("../data/raw/test.csv")

print(train.shape, test.shape)
train.head()
```

Ожидаемо увидите колонки `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`,
`Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`,
`Exited` (в `train`, но не в `test` — это и есть то, что нужно предсказать).
Полноценный разбор распределений, пропусков и баланса классов — в туториале 2.

## Шаг 8. Первый коммит

```powershell
git init
git status
```

Убедитесь, что в списке нет `.venv/` и `data/raw/` (если есть — `.gitignore` не
сработал, разберитесь прежде чем коммитить). Затем:

```powershell
git add CLAUDE.md tutorials/ requirements.txt .gitignore notebooks/01_eda.ipynb
git commit -m "Настройка окружения и данных соревнования"
```

## Как использовать Claude Code на этом этапе

- *"Создай requirements.txt для Kaggle ML-проекта, но пока только pandas, jupyter и
  kaggle"* — тренировка давать точный, ограниченный скоуп.
- *"Создай .gitignore для Python-проекта с venv, Jupyter-ноутбуками и папкой
  data/raw"*.
- Если `kaggle competitions download` вернёт ошибку — покажите текст ошибки Claude
  Code и попросите объяснить, что она значит, прежде чем гуглить.

## Проверка (пройдите по порядку)

1. `python --version` показывает 3.11.x.
2. После активации venv в терминале виден префикс `(.venv)`.
3. `kaggle competitions list` не выдаёт ошибку авторизации.
4. В `data/raw/` лежат `train.csv`, `test.csv`, `sample_submission.csv`.
5. Ноутбук `notebooks/01_eda.ipynb` открывается и `train.head()` показывает данные.
6. `git status` не показывает `.venv/` и `data/raw/` как файлы для добавления.
7. `git log --oneline` показывает хотя бы один коммит.

Следующий шаг — tutorial_2.md (появится, когда дойдём до EDA).
