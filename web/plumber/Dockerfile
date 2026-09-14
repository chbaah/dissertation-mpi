# ============================================================================
# MPI PLUMBER API
# ============================================================================

FROM rocker/r-ver:4.4.3


# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

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
        curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================================
# R PACKAGES
# ============================================================================

RUN R -e "install.packages( \
    c( \
        'plumber', \
        'lubridate', \
        'DBI', \
        'RPostgreSQL', \
        'dplyr', \
        'tidymodels', \
        'parsnip', \
        'recipes', \
        'xgboost', \
        'bundle', \
        'jsonlite' \
    ), \
    repos='https://cloud.r-project.org/' \
)"


# ============================================================================
# APPLICATION DIRECTORY
# ============================================================================

WORKDIR /app


# ============================================================================
# COPY PLUMBER FILES
# ============================================================================

COPY web/plumber/code/plumber.R /app/plumber.R
COPY web/plumber/code/run.R /app/run.R


# ============================================================================
# COPY FINAL MODEL AND RECIPE
# ============================================================================

RUN mkdir -p /app/models


COPY models/bestmodelimages/xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds \
     /app/models/xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds


COPY models/bestmodelimages/xgboost_all_pred_notfm_recipe.rds \
     /app/models/xgboost_all_pred_notfm_recipe.rds


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

ENV MODEL_DIR=/app/models

ENV MODEL_FILE=xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds

ENV RECIPE_FILE=xgboost_all_pred_notfm_recipe.rds


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=mpi
ENV DB_USER=postgres


# ============================================================================
# PLUMBER CONFIGURATION
# ============================================================================

ENV PLUMBER_HOST=0.0.0.0
ENV PLUMBER_PORT=3796
ENV PLUMBER_FILE=/app/plumber.R


# ============================================================================
# PORT
# ============================================================================

EXPOSE 3796


# ============================================================================
# HEALTH CHECK
# ============================================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD curl --fail http://localhost:3796/api/health || exit 1


# ============================================================================
# START API
# ============================================================================

CMD ["Rscript", "/app/run.R"]
