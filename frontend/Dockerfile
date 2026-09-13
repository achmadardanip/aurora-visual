FROM node:22.22.0-alpine3.23 AS build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
COPY contracts /app/contracts
RUN npm run build
FROM nginx:1.28.2-alpine3.23
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/frontend/dist /usr/share/nginx/html
EXPOSE 5171
HEALTHCHECK --interval=20s CMD wget -q -O /dev/null http://127.0.0.1:5171/
