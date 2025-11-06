FROM node:18-alpine as builder

WORKDIR /app

# Copy package files first for better cache utilization
COPY package*.json ./
RUN npm ci --prefer-offline

# Copy the rest of the code
COPY . .

# Build for production
RUN npm run build

# Production stage
FROM node:18-alpine
WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev --prefer-offline

# Copy built assets and server
COPY --from=builder /app/build ./build
COPY server.js ./server.js

EXPOSE 3000
CMD ["node", "server.js"]
