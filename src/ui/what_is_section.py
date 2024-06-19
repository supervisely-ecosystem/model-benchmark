import src.globals as g
import src.ui.settings as settings
import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DatasetThumbnail,
    IFrame,
    Markdown,
    SelectDataset,
    Text,
)


markdown = Markdown(
    """## What is YOLOv8 model? (collapse)

Можно также добавить ссылку на наш блог пост, если есть

!\[blog post link\]

**О чем еще здесь можно рассказать:**

- Ключевая инфа о модели текстом: год, конференция, paper, гитхаб, какой скор на лидерборде от авторов, в каком сценарии эта модель была или есть SOTA и в каком году. Что-то ещё из того что писали про свою модель сами авторы, взять из ридми на гитхабе.
- Особенности модели, чем отличается от остальных, какую проблему решали авторы этой моделью.
- Для чего эта модель идеально подходит, какие сценарии использования? Возможно авторы проектировали модель под специальный use case, описать это. Например, YOLO подходит для real-time object detection, для real-time detection на видео.
- Историческая справка, как развивалась модель, прошлые версии.
- Краткий анализ метрик. На чем модель фейлит, а в чем хорошо предсказывает.

## Expert insights?

linkedin - ответ на вопрос когда применять когда нет, что лучше или хуже, что нужно учитывать. текст в свободной форме

## How To Use: Training, inference, evaluation loop (collapse)

Однотипная диаграмка, и небольшой текст со ссылками - Sly apps, inference notebooks, docker images, … небольшой раздел со ссылками на документацию (embeddings sampling, improvement loop, active learning, labeling jobs, model comparison, .… – стандартизован для всех моделей). какие-то модели будут частично интегрированы

Jupyter notebooks + python scripts + apps + videos + guides + …
""",
    show_border=False,
)

container = Container(
    widgets=[
        markdown,
    ]
)
