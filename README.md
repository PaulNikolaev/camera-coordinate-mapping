## camera-coordinate-mapping

Проект: маппинг координат из камер `top/bottom` в систему координат `door2` (3200×1800).

### Быстрый старт (с нуля до METRICS)

- **Клонирование**
  - Клонируйте репозиторий и перейдите в каталог проекта.

- **Окружение (Windows PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **Данные**
  - Скачайте архив датасета по ссылке из `TASK_PLAN.md` и распакуйте так, чтобы рядом с репозиторием появился каталог `coord_data/` со структурой:
    - `coord_data/split.json`
    - `coord_data/train/`
    - `coord_data/val/`
    - `coords_top.json` / `coords_bottom.json` внутри сессий

- **Один прогон пайплайна**
  - Команда будет добавлена в рамках реализации (цель: в одном процессе сделать проверку наличия данных → одну полную валидацию → обучение на `train` → оценку на `val` → сохранить MED-файл).

### Что НЕ коммитить

- Датасет и архивы: `coord_data/`, `*.zip`
- Виртуальное окружение: `.venv/`
- Артефакты обучения: `artifacts/`, `checkpoints/`, веса моделей

Это уже настроено в `.gitignore`.

### Git (ветка и remote)

Репозиторий инициализирован с веткой по умолчанию `main`.

Пример привязки remote (замените URL на свой):

```powershell
git remote add origin <URL>
git push -u origin main
```

