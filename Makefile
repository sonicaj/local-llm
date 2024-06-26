.PHONY: build start stop index

# Default make target
all: start

# Build the Docker images
build:
	docker-compose build

# Start all services
start:
	docker-compose up -d

# Stop all services
stop:
	docker-compose down

# Rebuild and restart all services
restart: stop start

# Index documents into the vector store
index:
	docker-compose run -v ./data:/app/data --rm local-llm-backend /usr/local/bin/process_documents

# Log output from containers
logs:
	docker-compose logs -f
