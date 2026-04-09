# Docker commands
.PHONY: build run up down logs shell

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

shell:
	docker exec -it llama3-fraud-guard /bin/bash

# Clean up
clean:
	docker system prune -f
