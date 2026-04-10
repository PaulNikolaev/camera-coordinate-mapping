## Быстрый запуск с нуля

Клонирование проекта:

```powershell
git clone https://github.com/PaulNikolaev/camera-coordinate-mapping.git
cd camera-coordinate-mapping
```

Создание и активация виртуального окружения:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Подготовка датасета:

```powershell
python validate_data.py
```

Обучение baseline-модели:

```powershell
python train.py --data-root coord_data --artifacts-dir artifacts
```

Проверка инференса:

```powershell
python predict.py 1600 900 top --artifacts-dir artifacts
python predict.py 1600 900 bottom --artifacts-dir artifacts
```

Оценка качества на `val`:

```powershell
python evaluate.py --data-root coord_data --artifacts-dir artifacts
```

После оценки метрики будут сохранены в `artifacts/metrics.json`.

Запуск тестов:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
