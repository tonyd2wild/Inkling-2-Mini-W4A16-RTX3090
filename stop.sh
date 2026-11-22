#!/usr/bin/env bash
# Stop and remove the container (any lane).
NAME=${NAME:-inkling-w4a16}
docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME is not running"
