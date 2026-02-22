# Configuration Management

This directory contains the configuration system for the video pipeline, using a three-layer approach for maximum flexibility and type safety.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 1: YAML Files (Version Controlled)      │
│  - Task definitions (tasks.yaml)                │
│  - Environment configs (environments/*.yaml)    │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Layer 2: Environment Variables                 │
│  - Secrets (.env - gitignored)                  │
│  - Environment-specific overrides               │
└─────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  Layer 3: Pydantic Settings (Type-Safe)         │
│  - Runtime configuration (settings.py)          │
│  - Validation and defaults                      │
└─────────────────────────────────────────────────┘
```

## Files Structure

```
config/
├── __init__.py               # Exports get_settings()
├── settings.py               # Pydantic Settings models
├── tasks.yaml                # Task configurations
├── environments/
│   └── dev.yaml             # Development environment config
└── README.md                # This file
```

## Usage

### 1. Loading Application Settings

```python
from video_pipeline.config import get_settings

# Get singleton settings instance
settings = get_settings()

# Access nested settings
print(settings.minio.endpoint)
print(settings.postgres.connection_string)
print(settings.dask.to_cluster_kwargs())
```

### 2. Loading Task Configuration

```python
from video_pipeline.task.base.base_task import TaskConfig

# Load task config from YAML
config = TaskConfig.from_yaml("video_registration")

# Use in @task decorator
@task(**config.to_task_kwargs())
async def my_task(...):
    pass
```

### 3. Environment-Specific Configuration

Set the `APP_ENV` environment variable to load different configurations:

```bash
# Load dev.yaml
export APP_ENV=dev

# Load staging.yaml (when created)
export APP_ENV=staging

# Load prod.yaml (when created)
export APP_ENV=prod
```

## Configuration Layers Explained

### Layer 1: YAML Files (Static Configuration)

**Purpose**: Version-controlled, declarative configuration

**Files**:
- `tasks.yaml`: Task definitions (name, description, retries, timeouts, etc.)
- `environments/dev.yaml`: Development environment settings

**When to use**:
- Task metadata and default behavior
- Environment-specific defaults (workers, timeouts)
- Configuration that should be tracked in git

**Example** (`tasks.yaml`):
```yaml
tasks:
  video_registration:
    name: "Video Registration"
    description: "Extract and register video metadata"
    stage: "Ingestion"
    tags: ["video", "metadata"]
    retries: 2
    retry_delay_seconds: 5
    timeout_seconds: 120
```

### Layer 2: Environment Variables (Secrets & Overrides)

**Purpose**: Secrets and environment-specific values

**File**: `.env` (gitignored, use `.env.example` as template)

**When to use**:
- Secrets (passwords, API keys)
- Hostname/endpoints that differ per environment
- Local developer overrides

**Example** (`.env`):
```bash
APP_ENV=dev

# Secrets
POSTGRES_PASSWORD=secretpassword
MINIO_SECRET_KEY=minioadmin

# Environment-specific
POSTGRES_HOST=localhost
TRITON_URL=localhost:8001
```

**Priority**: Environment variables OVERRIDE YAML values

### Layer 3: Pydantic Settings (Runtime Configuration)

**Purpose**: Type-safe, validated runtime configuration

**File**: `settings.py`

**Features**:
- Type validation (ensures integers are integers, etc.)
- Default values
- Nested configuration objects
- Helper methods (e.g., `connection_string`, `to_cluster_kwargs()`)

**When to use**:
- When you need type safety
- When you need validated configuration
- When you need computed properties

## Common Patterns

### Pattern 1: Adding a New Task

1. Add task definition to `tasks.yaml`:

```yaml
tasks:
  my_new_task:
    name: "My New Task"
    description: "Does something cool"
    stage: "Processing"
    tags: ["processing"]
    retries: 2
    retry_delay_seconds: 5
    timeout_seconds: 300
```

2. Load in your task file:

```python
from video_pipeline.task.base.base_task import TaskConfig

MY_TASK_CONFIG = TaskConfig.from_yaml("my_new_task")

@task(**MY_TASK_CONFIG.to_task_kwargs())
async def my_new_task(...):
    pass
```

### Pattern 2: Adding a New Service Configuration

1. Add Pydantic Settings model in `settings.py`:

```python
class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: str | None = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
    )
```

2. Add to `AppSettings`:

```python
class AppSettings(BaseSettings):
    # ... existing settings ...
    redis: RedisSettings = Field(default_factory=RedisSettings)
```

3. Add to environment YAML (`environments/dev.yaml`):

```yaml
redis:
  host: "localhost"
  port: 6379
```

4. Add secrets to `.env`:

```bash
REDIS_PASSWORD=mysecretpassword
```

### Pattern 3: Using Configuration in Tasks

```python
from video_pipeline.config import get_settings

@task(**VIDEO_CONFIG.to_task_kwargs())
async def video_reg_task(video_input: VideoInput, context: TaskExecutionContext):
    settings = get_settings()

    # Use configuration
    task_impl = VideoRegistryTask(
        minio_client=MinioStorageClient(
            endpoint=settings.minio.endpoint,
            access_key=settings.minio.access_key,
            secret_key=settings.minio.secret_key,
            secure=settings.minio.secure,
        ),
    )
```

## Configuration Priority (Highest to Lowest)

1. **Environment Variables** (`.env` file or system env vars)
2. **Environment YAML** (`environments/{APP_ENV}.yaml`)
3. **Pydantic Defaults** (in `settings.py`)

Example:
```bash
# .env
MINIO_ENDPOINT=override.example.com
```

```yaml
# environments/dev.yaml
minio:
  endpoint: "localhost:9000"  # This will be overridden by .env
```

```python
# settings.py
class MinioSettings(BaseSettings):
    endpoint: str = Field(default="fallback:9000")  # Only used if not in YAML or .env
```

Result: `settings.minio.endpoint == "override.example.com"`

## Best Practices

### ✅ DO

- Store secrets in `.env` file (gitignored)
- Use YAML for non-secret, version-controlled configuration
- Use type hints in Pydantic models
- Use `get_settings()` to access configuration
- Add validation with Pydantic validators
- Use environment-specific YAML files for different deployments

### ❌ DON'T

- Commit `.env` file to git (use `.env.example` instead)
- Hardcode configuration values in Python code
- Mix concerns (secrets in YAML, task config in Python)
- Create settings instances directly (use `get_settings()`)

## Environment Variables Reference

### MinIO
- `MINIO_ENDPOINT`: MinIO server endpoint
- `MINIO_ACCESS_KEY`: MinIO access key
- `MINIO_SECRET_KEY`: MinIO secret key
- `MINIO_SECURE`: Use HTTPS (true/false)
- `MINIO_BUCKET_NAME`: Default bucket name

### PostgreSQL
- `POSTGRES_HOST`: Database host
- `POSTGRES_PORT`: Database port
- `POSTGRES_DATABASE`: Database name
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password

### Qdrant
- `QDRANT_HOST`: Qdrant server host
- `QDRANT_PORT`: Qdrant server port
- `QDRANT_API_KEY`: Qdrant API key (optional)
- `QDRANT_COLLECTION_NAME`: Default collection name

### Dask
- `DASK_N_WORKERS`: Number of Dask workers
- `DASK_THREADS_PER_WORKER`: Threads per worker
- `DASK_PROCESSES`: Use processes (true/false)
- `DASK_SCHEDULER_ADDRESS`: External scheduler address (optional)

### Tasks
- `TASK_DEFAULT_RETRIES`: Default retry count for tasks
- `TASK_DEFAULT_RETRY_DELAY`: Default retry delay in seconds
- `TASK_DEFAULT_TIMEOUT`: Default timeout in seconds

### Triton
- `TRITON_URL`: Triton Inference Server URL
- `TRITON_TIMEOUT`: Request timeout in seconds

## Testing Configuration

```python
from video_pipeline.config.settings import reset_settings, AppSettings

def test_my_feature():
    # Reset singleton for clean test
    reset_settings()

    # Create test settings
    settings = AppSettings(
        minio=MinioSettings(endpoint="test:9000"),
        postgres=PostgresSettings(host="testdb"),
    )

    # Your test code here
```

## Troubleshooting

### Issue: Settings not loading from .env

**Solution**: Ensure `.env` file is in the project root (where `pyproject.toml` is located)

### Issue: YAML file not found

**Solution**: Check that `APP_ENV` is set correctly and the corresponding YAML file exists in `config/environments/`

### Issue: Type validation error

**Solution**: Check that environment variables are in the correct format (e.g., integers for ports, booleans as "true"/"false")

### Issue: Settings cached with old values

**Solution**: Call `reset_settings()` or restart the Python process

## Migration Guide

If you're migrating from the old configuration system:

### Before (Using Prefect Variables):
```python
task_init_kwargs = cast(dict, Variable.get('task_init_kwargs'))
task_impl = VideoRegistryTask(**task_init_kwargs)
```

### After (Using Settings):
```python
settings = get_settings()
task_impl = VideoRegistryTask(
    minio_client=MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    ),
)
```

### Before (Task Config in Python):
```python
VIDEO_CONFIG = TaskConfig(
    name="Video Registration",
    description="Extract metadata",
    tags=["video"],
    retries=2,
)
```

### After (Task Config in YAML):
```python
VIDEO_CONFIG = TaskConfig.from_yaml("video_registration")
```
