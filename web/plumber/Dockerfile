FROM rocker/r-ver:4.4.3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++ \
        gcc \
        gfortran \
        make \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libpq-dev \
        libgit2-dev \
        libicu-dev \
        libomp-dev \
        libsodium-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libpng-dev \
        libtiff5-dev \
        libjpeg-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN R -e "install.packages(c( \
    'plumber', \
    'lubridate', \
    'DBI', \
    'RPostgreSQL', \
    'dplyr', \
    'parsnip', \
    'recipes', \
    'xgboost', \
    'bundle', \
    'jsonlite', \
    'readr' \
), repos='https://cloud.r-project.org/')"

# Fail the Docker build if any required R package is missing
RUN R -e "required <- c( \
    'plumber', \
    'lubridate', \
    'DBI', \
    'RPostgreSQL', \
    'dplyr', \
    'parsnip', \
    'recipes', \
    'xgboost', \
    'bundle', \
    'jsonlite', \
    'readr' \
); \
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]; \
if (length(missing) > 0) { \
    stop(paste('Missing R packages:', paste(missing, collapse=', '))) \
} else { \
    message('All required R packages are installed.') \
}"

WORKDIR /app

COPY web/plumber/code/plumber.R /app/plumber.R
COPY web/plumber/code/run.R /app/run.R

RUN mkdir -p /app/models

COPY models/bestmodelimages/xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds \
     /app/models/xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds


COPY models/bestmodelimages/xgboost_all_pred_notfm_recipe.rds \
     /app/models/xgboost_all_pred_notfm_recipe.rds


ENV MODEL_DIR=/app/models
ENV MODEL_FILE=xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds
ENV RECIPE_FILE=xgboost_all_pred_notfm_recipe.rds

ENV PLUMBER_HOST=0.0.0.0
ENV PLUMBER_PORT=3796
ENV PLUMBER_FILE=/app/plumber.R

EXPOSE 3796

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:3796/api/health || exit 1

CMD ["Rscript", "/app/run.R"]
