if [[ $1 == "cpu" ]]; then
    docker compose --env-file docker/.env -f docker/compose-cpu.yml down
else
    docker compose --env-file docker/.env -f docker/compose-gpu.yml down
fi
