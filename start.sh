if [[ $1 == "cpu" ]]; then
    docker compose --env-file docker/.env -f docker/compose-cpu.yml up -d
elif [[ $1 == "deploy" ]]; then
    docker compose --env-file docker/.env -f docker/compose-deploy.yml up -d
else
    docker compose --env-file docker/.env -f docker/compose-gpu.yml up -d
fi
