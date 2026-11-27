# T3. Ходьба Go1 с RL-политикой

Общий образ из `participant/Dockerfile` поддерживает T2, T3 и T4. Код, веса и результаты T3 находятся в `participant/t3/`. [Запуск, тесты и сдача T3](participant/t3/README.md)

# 1 Build
```bash
cd ./participant
docker build -t go1-participant:local .
```

# 2 Test ping
```bash
docker run --rm go1-participant:local ping
```

# 3 Run T2 with Simulator
From `go1-simulator-public`

```bash
export SIM_IMAGE=registry.cookcloud.ru/go1-hackathon-remote-stage/go1-sim:v1.0
python3 run_local.py --task T2 \
    --sim-image "$SIM_IMAGE" \
    --participant-image go1-participant:local \
    --gui --logs
```


