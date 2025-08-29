# Ragify Plugin - Deployment Guide

This guide covers all deployment options for the Ragify plugin.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Redis (for caching and vector storage)
- Optional: PostgreSQL (for metadata storage)

### Installation

#### From PyPI (Recommended)
```bash
pip install ragify
```

#### From Source
```bash
git clone https://github.com/sumitnemade/ragify.git
cd ragify
pip install -e .
```

## 📦 Package Deployment

### Building the Package

1. **Install build tools:**
   ```bash
   pip install build twine
   ```

2. **Build the package:**
   ```bash
   python -m build
   ```

3. **Check the package:**
   ```bash
   twine check dist/*
   ```

4. **Upload to PyPI:**
   ```bash
   twine upload dist/*
   ```

### Using the Deployment Script

```bash
python scripts/deploy.py
```

This script will:
- Clean previous builds
- Run tests
- Build the package
- Check the package
- Provide next steps



## ☁️ Cloud Deployment

### AWS Deployment

#### Using ECS

1. **Create ECS cluster**
2. **Deploy using the provided package**
3. **Configure environment variables**

#### Using Lambda (Serverless)

```yaml
# serverless.yml
service: ragify-plugin

provider:
  name: aws
  runtime: python3.11
  region: us-east-1

functions:
  ragify:
    handler: handler.main
    events:
      - http:
          path: /context
          method: post
```

### Google Cloud Platform

#### Using Cloud Run

```bash
# Deploy directly from source
gcloud run deploy ragify --source . --platform managed
```

### Azure

#### Using Azure App Service

```bash
# Deploy to Azure App Service
az webapp up --name ragify --resource-group myResourceGroup --runtime "PYTHON:3.11"
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `POSTGRES_URL` | PostgreSQL connection URL | None |
| `LOG_LEVEL` | Logging level | `INFO` |
| `PRIVACY_LEVEL` | Default privacy level | `private` |
| `MAX_CONTEXT_SIZE` | Maximum context size in tokens | `10000` |

### Configuration File

Create a `config.yaml` file:

```yaml
vector_db_url: "redis://localhost:6379"
cache_url: "redis://localhost:6379"
privacy_level: "enterprise"
max_context_size: 10000
default_relevance_threshold: 0.5
enable_caching: true
cache_ttl: 3600
enable_analytics: true
log_level: "INFO"
```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Use HTTPS for all external communications
- [ ] Implement proper authentication and authorization
- [ ] Use environment variables for sensitive data
- [ ] Enable logging and monitoring
- [ ] Regular security updates
- [ ] Network segmentation
- [ ] Data encryption at rest and in transit

### Privacy Controls

The plugin supports multiple privacy levels:

- **Public**: No privacy controls
- **Restricted**: Basic encryption and anonymization
- **Restricted**: Full encryption and strict access controls

## 📊 Monitoring and Logging

### Health Checks

```bash
# Check plugin health
curl http://localhost:8000/health

# Check Redis connection
redis-cli ping

# Check PostgreSQL connection
psql -h localhost -U ragify -d ragify -c "SELECT 1;"
```

### Logging

The plugin uses structured logging with `structlog`. Configure log levels:

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

## 🔄 CI/CD Pipeline

### GitHub Actions

The repository includes a basic CI/CD pipeline framework:

1. **Tests**: Run on multiple Python versions
2. **Linting**: Code quality checks
3. **Build**: Package building and validation
4. **Deploy**: Automatic deployment to PyPI

### Required Secrets

Set up these secrets in your GitHub repository:

- `PYPI_API_TOKEN`: PyPI upload token

## 🚨 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check Redis is running
   redis-cli ping
   
   # Check connection URL
   echo $REDIS_URL
   ```

2. **PostgreSQL Connection Error**
   ```bash
   # Check PostgreSQL is running
   pg_isready -h localhost -p 5432
   
   # Check connection URL
   echo $POSTGRES_URL
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   ps aux | grep python
   
   # Increase memory limits
   # Configure your deployment platform's memory limits
   ```

### Performance Tuning

1. **Redis Configuration**
   ```conf
   # redis.conf
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   ```

2. **PostgreSQL Configuration**
   ```conf
   # postgresql.conf
   shared_buffers = 256MB
   effective_cache_size = 1GB
   ```

## 📞 Support

For deployment support:

- **Issues**: [GitHub Issues](https://github.com/sumitnemade/ragify/issues)
- **Documentation**: [Read the Docs](https://ragify.readthedocs.io)
- **Email**: nemadesumit@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
