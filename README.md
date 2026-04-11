## Быстрый запуск с нуля

### 1. Клонирование и окружение

```powershell
git clone https://github.com/PaulNikolaev/camera-coordinate-mapping.git
cd camera-coordinate-mapping
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` нужен для обучения, инференса и расчета метрик. В проекте используется `scikit-learn`.

### 2. Подготовка датасета

```powershell
python validate_data.py
```

Эта команда:

- скачивает ZIP-архив датасета по умолчанию;
- распаковывает его в `coord_data/`;
- приводит корневую папку к каноническому имени `coord_data/`;
- выполняет полную strict-проверку структуры и аннотаций.

Если `coord_data/` уже существует, `validate_data.py` не обучает модель и не строит признаки заново: скрипт просто
валидирует существующий датасет.

Полезные флаги:

```powershell
python validate_data.py --force-download
python validate_data.py --keep-archive
python validate_data.py --data-root coord_data
python validate_data.py --archive-path test-task.zip
```

- `--force-download` удаляет текущий `coord_data/`, заново скачивает архив и распаковывает его.
- `--keep-archive` оставляет ZIP-файл после успешной подготовки.
- `--data-root` позволяет указать нестандартный путь к распакованному датасету.
- `--archive-path` задает путь к ZIP-архиву.

### 3. Что делает `validate_data.py`, а что делает `train.py`

`validate_data.py` отвечает только за подготовку и strict-проверку датасета.  
`train.py` отвечает только за обучение baseline-моделей и ожидает, что `coord_data/` уже существует.

Это разные задачи:

- `validate_data.py` проверяет, что данные можно использовать;
- `train.py` использует уже готовые данные и сохраняет артефакты модели.

### 4. Baseline одной командой

Чтобы обучить baseline и сразу посчитать MED на `val`, используйте pipeline-скрипт:

```powershell
python run_pipeline.py --data-root coord_data --artifacts-dir artifacts
```

Эта команда:

- обучает две модели: `top -> door2` и `bottom -> door2`;
- сохраняет артефакты в `artifacts/`;
- сразу считает MED на `val`;
- не требует отдельного шага подготовки признаков.

По умолчанию метрики сохраняются в `artifacts/metrics.json`.

Если нужно только обучение без оценки:

```powershell
python train.py --data-root coord_data --artifacts-dir artifacts
```

Артефакты после обучения:

- `artifacts/top_model.pkl`
- `artifacts/bottom_model.pkl`
- `artifacts/manifest.json`
- `artifacts/training_report.json`

### 5. `predict`

После обучения можно получить предсказание координат в системе `door2`:

```powershell
python predict.py 1600 900 top --artifacts-dir artifacts
python predict.py 1600 900 bottom --artifacts-dir artifacts
```

Аргументы:

- `x` и `y` - координаты точки в исходной камере;
- `source` - источник, только `top` или `bottom`;
- `--artifacts-dir` - папка с `manifest.json` и сохраненными моделями.

Вывод - JSON с исходными координатами и предсказанными `x_door2`, `y_door2`.

### 6. Расчет MED

Для отдельного расчета метрик на `val` используйте:

```powershell
python evaluate.py --data-root coord_data --artifacts-dir artifacts
```

Если нужен нестандартный путь:

```powershell
python evaluate.py --data-root coord_data --artifacts-dir artifacts --output-metrics reports/metrics.json
```

`evaluate.py` считает `MED` (`Mean Euclidean Distance`) в пикселях между:

- предсказанными координатами в системе `door2`;
- эталонными координатами из `val`.

Результат сохраняется в JSON и включает:

- `overall.med` - общий MED по всем точкам;
- `sources.top.med` и `sources.bottom.med` - MED по каждой камере;
- количество использованных точек, записей и сессий;
- `preparation_report` с отчетом по очистке данных.

### 7. Правила очистки данных

Подготовка выборки для `train.py`, `run_pipeline.py` и `evaluate.py` использует два режима.

По умолчанию используется `allow_partial`:

- запись отбрасывается, если отсутствуют файлы, сломана JSON-схема, нет изображений, координаты выходят за границы
  кадра, не совпадают `number`, есть дубликаты точек или вообще нет точек;
- запись с количеством точек вне диапазона `17-22` не отбрасывается автоматически, если совпадающие точки корректны;
- такая проблема фиксируется в отчете как `invalid_point_count`;
- сессия считается отброшенной, если после очистки в ней не осталось ни одной usable-записи.

Строгий режим включается флагом `--strict`:

```powershell
python train.py --data-root coord_data --artifacts-dir artifacts --strict
python evaluate.py --data-root coord_data --artifacts-dir artifacts --strict
python run_pipeline.py --data-root coord_data --artifacts-dir artifacts --strict
```

В `strict` дополнительно отбрасываются все frame pair записи, у которых число точек в `image1_coordinates` или
`image2_coordinates` выходит за диапазон `17-22`.

Практически это означает:

- `validate_data.py` выполняет отдельную строгую проверку всего датасета;
- `train.py` и `evaluate.py` по умолчанию работают мягче и пытаются использовать максимум валидных совпадений;
- если нужен полностью жесткий режим подготовки выборки для обучения и оценки, используйте `--strict`.

Отдельно учитывается известная опечатка в архиве: `coodrs_top.json` принимается как допустимый fallback для
`coords_top.json`, но помечается в отчете как `annotation_filename_typo`.

### 8. Воспроизводимый baseline

Полный воспроизводимый сценарий на чистой машине:

```powershell
git clone https://github.com/PaulNikolaev/camera-coordinate-mapping.git
cd camera-coordinate-mapping
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python validate_data.py
python run_pipeline.py --data-root coord_data --artifacts-dir artifacts
python predict.py 1600 900 top --artifacts-dir artifacts
```

### 9. Тесты

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
