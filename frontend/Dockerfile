FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Set environment variable for API
ENV NEXT_PUBLIC_API_URL=http://localhost:8000

# Run in development mode
CMD ["npm", "run", "dev"]
