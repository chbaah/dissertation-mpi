FROM nginx:alpine

# envsubst is provided by gettext
RUN apk add --no-cache gettext

# Copy frontend files into nginx web directory
COPY web/frontend/index.html web/frontend/predict.html /usr/share/nginx/html/

# Copy frontend files into nginx web directory
COPY web/frontend/js/config.template.js web/frontend/js/predict.js web/frontend/js/script.js /usr/share/nginx/html/js/


# CSS
COPY web/frontend/css/ /usr/share/nginx/html/css/

# Images
COPY web/frontend/img/ /usr/share/nginx/html/img/


# Install our startup script
COPY web/frontend/docker-entrypoint.sh /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/docker-entrypoint.sh"]
