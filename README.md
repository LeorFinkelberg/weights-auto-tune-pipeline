### _Порядок работы с проектом_
```bash
$ git clone <repo>
$ cd <repo>
$ uv sync
```
NB! Перед каждым коммитом запускается `pre-commit` с проверкой. Однако, если требуется зафиксировать изменения без `pre-commit`, то следует добавить флаг `--no-verify`
```bash
$ git commit -v --no-verify
```

Для запуска тестов выполнить
```bash
$ nox -s test
```