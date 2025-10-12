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

# Install serve to run the built app
RUN npm install -g serve

# Copy built assets from builder stage
COPY --from=builder /app/build ./build

EXPOSE 3000
CMD ["serve", "-s", "build", "-l", "3000"]
