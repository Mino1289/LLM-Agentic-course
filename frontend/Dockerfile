FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG API_PROXY_TARGET=http://127.0.0.1:8000
ENV API_PROXY_TARGET=$API_PROXY_TARGET
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ARG API_PROXY_TARGET=http://127.0.0.1:8000
ENV API_PROXY_TARGET=$API_PROXY_TARGET
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
