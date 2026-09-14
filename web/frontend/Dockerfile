FROM nginx:alpine

# envsubst is provided by gettext
RUN apk add --no-cache gettext

# Copy frontend files into nginx web directory
COPY index.html predict.html /usr/share/nginx/html/

# Copy frontend files into nginx web directory
COPY js/config.template.js js/predict.js js/script.js /usr/share/nginx/html/js/


# CSS
COPY css/ /usr/share/nginx/html/css/

# Images
COPY img/ /usr/share/nginx/html/img/


# Install our startup script
COPY docker-entrypoint.sh /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/docker-entrypoint.sh"]