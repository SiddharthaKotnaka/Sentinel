# Sentinel Frontend Container
FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy application source
COPY frontend/ ./

ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    PORT=3000

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
