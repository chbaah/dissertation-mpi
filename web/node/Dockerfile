FROM node:24-alpine

WORKDIR /app

# Copy dependency definitions first
COPY web/node/package.json web/node/package-lock.json ./

# Install production dependencies
RUN npm ci --omit=dev

# Copy application source
COPY web/node/js/server.js ./

EXPOSE 3000

CMD ["npm", "start"]
