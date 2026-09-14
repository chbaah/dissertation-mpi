FROM node:24-alpine

WORKDIR /app

# Copy dependency definitions first
COPY package.json package-lock.json ./

# Install production dependencies
RUN npm ci --omit=dev

# Copy application source
COPY js/server.js ./

EXPOSE 3000

CMD ["npm", "start"]