FROM node:22-alpine AS builder

WORKDIR /app

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Copy package files first for better cache utilization
COPY package*.json ./
RUN npm ci --prefer-offline

# Copy the rest of the code
COPY . .

# Build for production
RUN npm run build

# Production stage
FROM node:22-alpine
WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev --prefer-offline

# Copy built assets and server
COPY --from=builder /app/dist ./dist
COPY server.js ./server.js

EXPOSE 3000
CMD ["node", "server.js"]
