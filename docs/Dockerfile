FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY evidence.config.yaml ./
COPY sources/ ./sources/
COPY pages/ ./pages/

RUN npm run sources

EXPOSE 3000

CMD ["npm", "run", "dev"]
