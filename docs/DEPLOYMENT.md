# Deployment Guide

## Prerequisites
- Docker & Docker Compose
- Access to production servers
- SSL certificates
- Environment variables

## Production Environment Setup

### 1. Server Requirements
- CPU: 4+ cores
- RAM: 16GB+
- Storage: 100GB+
- OS: Ubuntu 20.04 LTS

### 2. Security Setup
1. Configure firewall
```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw enable
```

2. Setup SSL certificates
```bash
certbot certonly --nginx -d yourdomain.com
```

### 3. Docker Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Deployment Steps

### 1. Database Deployment
1. Start Neo4j container:
```bash
docker-compose up -d neo4j
```

2. Initialize database:
```bash
docker exec -it neo4j neo4j-admin database create patent-db
```

### 2. Backend Deployment
1. Build Spring Boot application:
```bash
./mvnw clean package -DskipTests
```

2. Deploy container:
```bash
docker-compose up -d backend
```

### 3. Python Services Deployment
1. Build Python service:
```bash
docker build -t patent-python-service ./src
```

2. Deploy container:
```bash
docker-compose up -d python-service
```

### 4. Frontend Deployment
1. Build frontend:
```bash
cd frontend
npm run build
```

2. Deploy container:
```bash
docker-compose up -d frontend
```

## Docker Compose Configuration

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:4.4
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    ports:
      - "7474:7474"
      - "7687:7687"

  backend:
    build: ./backend
    environment:
      SPRING_PROFILES_ACTIVE: prod
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      JWT_SECRET: ${JWT_SECRET}
    depends_on:
      - neo4j
    ports:
      - "8080:8080"

  python-service:
    build: ./src
    environment:
      MODEL_PATH: /app/models
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
    depends_on:
      - neo4j
    ports:
      - "5000:5000"

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
      - python-service

volumes:
  neo4j_data:
  neo4j_logs:
```

## Monitoring Setup

### 1. Prometheus & Grafana
1. Add monitoring services to docker-compose:
```yaml
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    depends_on:
      - prometheus
    ports:
      - "3000:3000"
```

2. Configure Grafana dashboards for:
   - System metrics
   - Application metrics
   - Business metrics

### 2. Logging
1. Setup ELK Stack:
```yaml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.9.3
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.9.3
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.9.3
    depends_on:
      - elasticsearch
    ports:
      - "5601:5601"
```

## Backup Strategy

### 1. Database Backup
1. Setup daily backups:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
docker exec neo4j neo4j-admin dump --database=patent-db --to=/backups/backup-$DATE.dump
```

2. Configure backup rotation:
```bash
find /backups -name "backup-*.dump" -mtime +7 -delete
```

### 2. Application Backup
1. Backup configurations:
```bash
tar -czf configs-$DATE.tar.gz /etc/patent-platform/
```

2. Backup uploaded files:
```bash
tar -czf uploads-$DATE.tar.gz /var/patent-platform/uploads/
```

## Rollback Procedure

### 1. Database Rollback
```bash
# Stop services
docker-compose down

# Restore database
docker exec neo4j neo4j-admin load --from=/backups/backup-$DATE.dump --database=patent-db --force

# Start services
docker-compose up -d
```

### 2. Application Rollback
```bash
# Pull previous version
docker-compose pull

# Restart services
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. Database Connection Issues
```bash
# Check Neo4j logs
docker logs neo4j

# Verify network connectivity
docker network inspect patent-platform_default
```

2. Memory Issues
```bash
# Check memory usage
docker stats

# Adjust container limits in docker-compose.yml
```

3. Disk Space Issues
```bash
# Clean up old containers and images
docker system prune

# Clean up old logs
find /var/log -name "*.log" -mtime +30 -delete
```

### Health Checks

1. Backend Health
```bash
curl http://localhost:8080/actuator/health
```

2. Python Service Health
```bash
curl http://localhost:5000/health
```

3. Database Health
```bash
curl http://localhost:7474/browser/
``` 