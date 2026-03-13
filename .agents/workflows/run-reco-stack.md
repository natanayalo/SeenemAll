---
description: Build and launch the Seen’emAll stack, run migrations, and verify connectivity.
---
// turbo-all

1. Launch Docker containers:
```bash
docker compose up -d --build
```

2. Run database migrations:
```bash
make migrate
```

3. Populate the catalog (if empty):
```bash
make etl-tmdb
```

4. Verify health and connectivity:
```bash
make health
```

5. Check live metrics:
```bash
make metrics
```
