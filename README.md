# Camera Coordinate Mapping

Модель для маппинга координат из камер `top` и `bottom` в систему координат `door2`.

Используется две отдельные модели `ExtraTreesRegressor`:

- `top -> door2`
- `bottom -> door2`

## Состав репозитория

- `solution/` - основная логика загрузки данных, обучения, инференса и оценки
- `train.py` - обучение моделей с нуля на `train` из `split.json`
- `predict.py` - загрузка артефактов и предсказание `door2`-координат
- `evaluate.py` - расчёт MED на `val`
- `run_pipeline.py` - обучение и оценка одной командой
- `requirements.txt` - зависимости

## Подготовка окружения

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Подготовка датасета

Ожидается, что датасет находится в `coord_data/`:

```text
coord_data/
├── split.json
├── train/
└── val/
```

Если нужно автоматически скачать и провалидировать датасет:

```powershell
python validate_data.py
```

## Обучение с нуля

```powershell
python train.py --data-root coord_data --artifacts-dir artifacts
```

`train.py` использует только сессии из `split.json["train"]` и сохраняет артефакты в `artifacts/`.

После обучения создаются:

- `artifacts/top_model.pkl`
- `artifacts/bottom_model.pkl`
- `artifacts/manifest.json`
- `artifacts/training_report.json`

## Инференс

Контракт предсказания:

```python
predict(x, y, source) -> (x_door2, y_door2)
```

CLI:

```powershell
python predict.py 1600 900 top --artifacts-dir artifacts
python predict.py 1600 900 bottom --artifacts-dir artifacts
```

Python API:

```python
from solution.inference import predict

x_door2, y_door2 = predict(1600, 900, "top")
print(x_door2, y_door2)
```

Где `source` принимает значения `top` или `bottom`.

## Оценка качества

Метрика качества - `MED` (`Mean Euclidean Distance`) в пикселях на валидационном сплите.

Расчёт метрик:

```powershell
python evaluate.py --data-root coord_data --artifacts-dir artifacts
```

Результат сохраняется в:

```text
artifacts/metrics.json
```

В файле сохраняются:

- общий `overall.med`;
- отдельный `sources.top.med`;
- отдельный `sources.bottom.med`.

## Полный pipeline

```powershell
python run_pipeline.py --data-root coord_data --artifacts-dir artifacts
```

Команда обучает модели на `train`, считает MED на `val` и сохраняет артефакты вместе с `artifacts/metrics.json`.

## Воспроизведение на чистой машине

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python validate_data.py
python run_pipeline.py --data-root coord_data --artifacts-dir artifacts
python predict.py 1600 900 top --artifacts-dir artifacts
```

## Параметры baseline

- `ExtraTreesRegressor`
- `n_estimators=600`
- `min_samples_leaf=2`
- `random_state=42`

## Тесты

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
