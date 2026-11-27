# T3. Ходьба Unitree Go1 с RL-политикой

Участник запускает два узла ROS 2 Humble. `t3_walking_policy` читает `/cmd_vel` и `/lowstate` и публикует цели для 12 суставов в `/lowcmd_bare`. `t3_safety_block` проверяет команды и публикует результат в `/lowcmd` с целевой частотой 100 Гц по `/clock`. Исполнение на CPU. `preferred_speed_mps` установлено в **0,6 м/с**.

Политика использует ориентацию IMU, состояния суставов и историю предыдущих действий. Safety Block проверяет 655 пар геометрий из таблицы организаторов с шагом до 0,1° на переходе. Блок ограничивает углы и изменение цели до 5° на публикацию, отклоняет некорректные числовые значения и через 0,4 секунды потери входного потока удерживает последнюю безопасную цель с нулевыми `dq` и `tau`.

## Собрать и запустить образ

Клонируйте ветку `t3` и перейдите в каталог решения. Для сборки нужны Docker Engine и Buildx. Для запуска образа достаточно CPU.

```bash
git clone --branch t3 https://github.com/Q1uLe/hackaton_ao_maped.git
cd hackaton_ao_maped/participant/t3
```

Дальнейшие команды запускайте из этого каталога.

```bash
REV=$(git rev-parse HEAD)
docker buildx build --platform linux/amd64 --label "org.opencontainers.image.revision=$REV" --load -t go1-participant:local ..
docker run --rm --network none go1-participant:local ping
```

`ping` возвращает один JSON-объект из [submission.json](submission.json). Judge запускает образ с аргументом `T3` и передаёт параметры DDS через окружение. Общий образ из `participant/Dockerfile` поддерживает T2, T3 и T4, запускается от пользователя `participant` и использует сообщения ROS 2 из общего рабочего пространства.

После сборки образ можно перенести на другой компьютер. Сохраните его командой `docker save -o go1-t3-linux-amd64.tar go1-participant:local`, затем загрузите архив на целевой машине командой `docker load -i go1-t3-linux-amd64.tar`. Исходники, веса и модель для сборки входят в ветку.

Запускайте локальную проверку с публичным Simulator на Linux.

```bash
docker pull registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0
EPISODES=10 bash tools/check_ros.sh go1-participant:local
```

Скрипт пересоздаёт контейнеры Simulator и Participant для каждого эпизода. Он использует `/odom/ground_truth` публичного симулятора для измерения расстояния и падений. Узлы участника подписаны только на команду скорости, телеметрию суставов и IMU, `/clock` и `/lowcmd_bare`.

При программной эмуляции QEMU используйте `SIM_REALTIME_FACTOR=0.1 WALL_LIMIT=240 bash tools/check_ros.sh go1-participant:local`. Параметр замедляет привязку цикла MuJoCo к реальному времени, сохраняя шаг физики и настройки контроллера. В сохранённом прогоне до интеграции отдельный образ T3 прошёл 3,003 м за 6,228 с симуляции без падения. Средняя частота `/lowcmd` составила 90,24 Гц, боковое смещение составило 0,469 м. В штатном темпе предыдущего прогона под QEMU частота сообщений упала ниже 50 Гц. [Новый прогон ROS 2](results/ppo-finetune-8h/ros-container.json) · [Результаты первой версии](results/t3-ros.json)

## Воспроизвести проверку физики

Установите Python-зависимости в виртуальное окружение перед проверкой.

```bash
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python -m tools.evaluate --episodes 10 --output training/runs/local-check.json
```

Для той же проверки в Windows PowerShell выполните команды из `participant/t3/`.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m tools.evaluate --episodes 10 --output training/runs/local-check.json
```

Headless-прогон использует опубликованную сцену MuJoCo 3.3.7 и закон расчёта моментов публичного Lowlevel. В нём работают те же функции политики и Safety Block, что в ROS 2. При запуске по умолчанию робот стоит во время прогрева. Флаг `--warmup-walk` подаёт команду движения с начала прогрева. Для проверки весов до восьмичасового обучения добавьте `--residual training/baseline_residual.npz`. Флаг `--pretrained` запускает оригинальную политику Walk These Ways.

## Веса и материалы дообучения

Исходная политика Walk These Ways училась методом PPO. Для T3 её основные веса заморожены. Дополнительная сеть училась методом PPO в сцене организаторов и корректирует 12 выходов исходной политики. Начальная коррекция равна нулю. Награда учитывает продольную скорость, боковое отклонение, курс, высоту корпуса, плавность действий и момент. Во время обучения варьируются начальная поза, трение, демпфирование и шум энкодеров. [Walk These Ways, оригинальная реализация](https://github.com/Improbable-AI/walk-these-ways)

- [Веса исходной политики](weights/walk_these_ways.npz) и [текущие веса коррекции](weights/residual.npz). Восьмичасовой прогон продолжил обучение из [чекпойнта первой версии](training/baseline_residual.npz), выбранного на 20 480 шагах из прогона на 61 440 шагов.
- [Конфигурация](training/config.json), [код PPO](training/train.py), [журнал обучения](results/ppo-training.jsonl) и [сравнение чекпойнтов](results/checkpoint-selection.json).
- [Десять эпизодов первой версии с прогревом стоя](results/t3-mujoco.json) и [десять эпизодов первой версии с командой во время прогрева](results/t3-mujoco-warmup-walk.json).

```bash
python -m pip install -r training/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m training.train --out training/runs/reproduction
python -m tools.evaluate --residual training/runs/reproduction/residual_020480.npz
```

Для экспорта исходных весов используйте [tools/export_pretrained.py](tools/export_pretrained.py) и папку `runs/gait-conditioned-agility/pretrain-v0/train/025417.456545/checkpoints` из ревизии `0e7236bdc81ce855cbe3d70345a7899452bdeb1c`. Модель MuJoCo и таблица пар взяты из ревизии `9de879c0a2e2e92220bd88d3613171ac08c2ba7a` публичного симулятора. [Ревизия Walk These Ways](https://github.com/Improbable-AI/walk-these-ways/tree/0e7236bdc81ce855cbe3d70345a7899452bdeb1c) · [Ревизия симулятора](https://github.com/Urika000/go1-simulator-public/tree/9de879c0a2e2e92220bd88d3613171ac08c2ba7a)

## Длительное дообучение

Замерьте время сбора опыта и обновления PPO перед запуском. Режим `hybrid` использует CPU для шести сред MuJoCo и RTX для обновлений PPO. Режимы `cpu` и `cuda` переносят вычисления самой сети на соответствующее устройство. Физический движок MuJoCo работает на CPU во всех трёх режимах. [Результаты замера](results/ppo-device-benchmark.json)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.benchmark --out training/runs/benchmark.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m training.long_run --hours 8 --mode hybrid --initial training/baseline_residual.npz --out training/runs/t3-long
```

В Windows задайте переменные окружения через `$env:OPENBLAS_NUM_THREADS='1'`, `$env:OMP_NUM_THREADS='1'` и `$env:MKL_NUM_THREADS='1'` перед запуском. Процесс временно запрашивает у Windows блокирование автоматического сна. Параметры обучения и критерии отбора сохранены в [training/long_config.json](training/long_config.json) и [training/scenarios.py](training/scenarios.py).

`status.json` показывает этап, время окончания и номер лучшего чекпойнта. В `train.jsonl` записываются пропускная способность, награда, падения и показатели PPO. При проверках с фиксированными seed робот проходит ровную площадку при двух режимах прогрева и отдельные сценарии с толчками, изменённым трением, демпфированием и задержкой одной команды. При двух последовательных ухудшениях процесс восстанавливает лучший чекпойнт и снижает скорость обучения.

`latest.zip` содержит состояние PPO для восстановления, а `best.npz` содержит веса лучшего по валидации кандидата. После остановки `final.json` сравнивает исходные и новые веса на 32 отдельных эпизодах на каждую модель. При сбое используйте `python -m training.long_run --resume --out training/runs/t3-long`, чтобы сохранить исходный срок окончания. Для досрочной остановки создайте пустой файл `STOP` в папке запуска. Рабочие веса в `weights/residual.npz` заменяются отдельно после проверки.

Восьмичасовой прогон собрал 21 583 374 перехода. Веса выбраны на отметке 5 554 176 переходов по валидации и затем проверены на других seed. [Итоговый отчёт](results/ppo-finetune-8h/final.json) · [Журнал валидации](results/ppo-finetune-8h/validation.jsonl) · [Метрики PPO](results/ppo-finetune-8h/train.jsonl)

| Независимая проверка | Первая версия | После дообучения |
| --- | ---: | ---: |
| Успешные штатные эпизоды | 20/20 | 20/20 |
| Медианное время на 3 м | 6,535 с | 5,310 с |
| Среднее боковое отклонение в штатных эпизодах | 15,4 см | 3,2 см |
| Успешные эпизоды с возмущениями | 12/12 | 12/12 |
| Среднее боковое отклонение с возмущениями | 28,1 см | 7,9 см |

На seed из первого прогона T3 новая модель прошла 10 из 10 эпизодов за 5,39–5,41 с каждый. [Повторный прогон T3](results/ppo-finetune-8h/judge-local.json)

## Опубликовать решение

Соберите и проверьте `go1-participant:local` из чистого checkout ветки `t3` по командам выше. Затем загрузите этот образ под тегом T3 в проект команды в Harbor. Учётные данные вводите при `docker login`.

```bash
REV=$(git rev-parse HEAD)
read -r -p 'Имя проекта и репозитория команды в Harbor: ' TEAM_NAME
IMAGE_REPO="registry.cookcloud.ru/${TEAM_NAME}/${TEAM_NAME}"
IMAGE="${IMAGE_REPO}:t3-${REV:0:12}"
docker login registry.cookcloud.ru
docker image inspect go1-participant:local --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
```

Убедитесь, что `docker image inspect` напечатал тот же коммит, что и `git rev-parse HEAD`. Отправьте образ и получите digest из Harbor.

```bash
docker tag go1-participant:local "$IMAGE"
docker push "$IMAGE"
docker buildx imagetools inspect "$IMAGE"
read -r -p 'Digest из Harbor (sha256:...): ' DIGEST
docker pull --platform linux/amd64 "${IMAGE_REPO}@${DIGEST}"
docker run --rm --network none "${IMAGE_REPO}@${DIGEST}" ping
printf '%s\n' "https://github.com/Q1uLe/hackaton_ao_maped/commit/${REV}" \
  "docker pull --platform linux/amd64 ${IMAGE_REPO}@${DIGEST}"
```

В личном кабинете капитан указывает выведенную ссылку на коммит и команду `docker pull` с digest из Harbor. [Инструкция организаторов](https://github.com/Urika000/go1-simulator-public/blob/main/docs/Инструкция_по_загрузке_решения.md)
