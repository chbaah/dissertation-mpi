#!/bin/sh
set -e

# Use user-provided environment variables.
# If none are supplied, use localhost defaults.

export NODE_API_URL="${NODE_API_URL:-http://localhost:3000}"
export PLUMBER_API_URL="${PLUMBER_API_URL:-http://localhost:3796}"

echo "Configuring frontend:"
echo "NODE_API_URL=${NODE_API_URL}"
echo "PLUMBER_API_URL=${PLUMBER_API_URL}"

envsubst '${NODE_API_URL} ${PLUMBER_API_URL}' \
    < /usr/share/nginx/html/js/config.template.js \
    > /usr/share/nginx/html/js/config.js

exec nginx -g 'daemon off;'