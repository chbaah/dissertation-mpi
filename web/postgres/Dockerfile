FROM postgres:17

ENV POSTGRES_DB=mpi
ENV POSTGRES_USER=postgres

# Initialization scripts are executed automatically by the
# official PostgreSQL image when the database is first created.
COPY databasedocker/01-create-tables.sql \
     /docker-entrypoint-initdb.d/01-create-tables.sql

COPY databasedocker/02-import-data.sql \
     /docker-entrypoint-initdb.d/02-import-data.sql

# Initial MPI dataset used by 02-import-data.sql
COPY ../../analysis/finaldataframe/ntl_lcu_osm_data.west.africa.csv \
     /docker-entrypoint-initdb.d/data/ntl_lcu_osm_data.west.africa.csv

EXPOSE 5432
