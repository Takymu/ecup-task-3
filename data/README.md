# data/

Здесь должен лежать `train.parquet` — данные организаторов (30.6 млн строк, ~180 МБ).
В репозиторий он не кладётся.

Схема:

```
event_date  user_id  search  cat  searches
has_search_to_cart  has_search_to_ord  has_cat_to_cart  has_cat_to_ord
search_to_cart  search_to_ord  cat_to_cart  cat_to_ord
to_cart  to_ord  gmv_search  gmv_cat  gmv
```

Период: 01.01.2025 – 13.02.2026. Целевое окно: 14.02 – 15.03.2026.

`sample_submit.csv.gz` — шаблон сабмита от организаторов (250 000 строк).
Распаковывать не нужно: код читает `.gz` напрямую через polars. Если какой-то из скриптов
рабочего дерева ждёт несжатый файл — `gzip -dk sample_submit.csv.gz`.
