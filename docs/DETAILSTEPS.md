# Multidimensional Poverty Index Prediction Application

## From Surveys to Algorithms: Machine Learning for Multidimensional Poverty Index Assessment

This project investigates predicting subnational **Multidimensional Poverty Index (MPI)** values using open-source geospatial and infrastructure data.

The web application consists of four main components:

1. **PostgreSQL database** — stores the geospatial, infrastructure, and other predictor data the application requires.
2. **R Plumber API** — loads the trained XGBoost model and preprocessing recipe and performs MPI predictions.
3. **Node.js/Express API** — provides application and database API endpoints used by the frontend.
4. **Frontend application** — provides the browser-based interface for interacting with the MPI prediction system.

The application can be deployed using Docker containers or by installing and running the individual components directly on the operating system.

This document describes the procedures for deploying the application using Docker containers or by installing the individual components directly on the operating system.

---

# 1. Repository

Clone the project repository and change into the repository root:

```bash
git clone git@github.com:chbaah/dissertation-mpi.git
cd dissertation-mpi
```

All Docker build commands in this document should be executed from the repository root because the Dockerfiles reference files located in different parts of the repository.

The relevant structure is approximately:

```text
    
dissertation-mpi/
├── analysis
│   ├── finaldataframe
│   │   ├── ntl_lcu_osm_data.west.africa.csv
│   │   └── ntl_lcu_osm_data.west.africa.rds
│   └── finalproj-data_analysis-mpi.R
├── data-acquisition
│   └── data_acquisition-GEE-OSM.R
├── docs
│   └── DETAILSTEPS.md
├── env
│   ├── frontend.env.example
│   ├── node.env.example
│   ├── plumber.env.example
│   └── postgres.env.example
├── models
│   └── bestmodelimages
│       ├── xgboost_all_pred_notfm_recipe.rds
│       └── xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds
├── README.md
└── web
    ├── frontend
    │   ├── css
    │   │   └── styles.css
    │   ├── default.conf
    │   ├── docker-entrypoint.sh
    │   ├── Dockerfile
    │   ├── img
    │   │   └── backgroundmap.jpg
    │   ├── index.html
    │   ├── js
    │   │   ├── config.js
    │   │   ├── config.template.js
    │   │   ├── predict.js
    │   │   └── script.js
    │   └── predict.html
    ├── node
    │   ├── Dockerfile
    │   ├── js
    │   │   └── server.js
    │   ├── package.json
    │   └── package-lock.json
    ├── plumber
    │   ├── code
    │   │   ├── plumber.R
    │   │   └── run.R
    │   └── Dockerfile
    └── postgres
        ├── databasedocker
        │   ├── 01-create-tables.sql
        │   └── 02-import-data.sql
        └── Dockerfile

```

---

# 2. Docker Network

The four application containers communicate over a user-defined Docker bridge network.

Create the network before starting any of the containers:

```bash
sudo docker network create --driver bridge mpi-network
```

The shorter command below produces a user-defined bridge network as well because `bridge` is the default driver:

```bash
sudo docker network create mpi-network
```

Verify the network:

```bash
sudo docker network inspect mpi-network
```

Containers connected to this network can resolve one another using their container names.

For example, the Node.js and Plumber containers can connect to PostgreSQL using:

```text
DB_HOST=mpi-postgres
DB_PORT=5432
```

rather than using the PostgreSQL container's IP address.

---

# 3. Environment Configuration

Environment variables are used to keep runtime configuration outside the container images. The repository contains example environment files under the `env/` directory:

```text
env/
├── postgres.env.example
├── plumber.env.example
├── node.env.example
└── frontend.env.example
```

These files provide templates for the configuration required by each application component. Before starting the containers, copies of the example files should be created without the `.example` extension:

```bash
cp env/postgres.env.example env/postgres.env
cp env/plumber.env.example env/plumber.env
cp env/node.env.example env/node.env
cp env/frontend.env.example env/frontend.env
```

The resulting runtime configuration is therefore:

```text
env/
├── postgres.env
├── plumber.env
├── node.env
└── frontend.env
```

The `.env` files can then be updated with the appropriate values for the deployment environment.

Environment files containing passwords or other secrets should not be committed to Git. Ensure `.gitignore` contains an appropriate rule such as:

```text
*.env
```

The `.env.example` files can remain in the Git repository because they should contain only example values and not actual passwords or other secrets.

Example environment files are described below. Passwords shown in this document are examples for local development only and should be replaced with secure values for a production deployment.

## PostgreSQL

Example `env/postgres.env`:

```text
POSTGRES_DB=mpi
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<database-password>
```

## Plumber

Example `env/plumber.env`:

```text
DB_HOST=mpi-postgres
DB_PORT=5432
DB_NAME=mpi
DB_USER=postgres
DB_PASSWORD=<database-password>

PLUMBER_HOST=0.0.0.0
PLUMBER_PORT=3796
PLUMBER_FILE=/app/plumber.R

MODEL_DIR=/app/models
MODEL_FILE=xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds
RECIPE_FILE=xgboost_all_pred_notfm_recipe.rds
```

## Node.js

Example `env/node.env`:

```text
DB_HOST=mpi-postgres
DB_PORT=5432
DB_NAME=mpi
DB_USER=postgres
DB_PASSWORD=<database-password>

PORT=3000
```

## Frontend

The Docker frontend uses Nginx as a reverse proxy for requests to the Node.js
and Plumber APIs.

Example `env/frontend.env`:

```text
NODE_API_URL=/node-api
PLUMBER_API_URL=/plumber-api
```

---

# 4. PostgreSQL Database

The PostgreSQL container stores the application data used to obtain predictor values for MPI prediction.

The image initialises the database using the same final CSV dataset used by the modelling workflow:

```text
analysis/finaldataframe/ntl_lcu_osm_data.west.africa.csv
```

The initialisation scripts create and populate the required PostgreSQL table.

## 4.1 Build the PostgreSQL image

From the repository root:

```bash
sudo docker build \
  -f web/postgres/Dockerfile \
  -t chbaah/dissertation-mpi:postgres-v20260916 \
  .
```

The versioned tag identifies a specific build and should not subsequently be reused for a different image.

## 4.2 Tag a verified build as latest

After testing the versioned image successfully:

```bash
sudo docker tag \
  chbaah/dissertation-mpi:postgres-v20260916 \
  chbaah/dissertation-mpi:postgres-latest
```

Both tags should point to the same image ID:

```bash
sudo docker images chbaah/dissertation-mpi
```

## 4.3 Create the PostgreSQL data volume

Create a named Docker volume:

```bash
sudo docker volume create mpi_postgres_data
```

The volume provides persistent database storage outside the PostgreSQL container's lifecycle.

Verify it:

```bash
sudo docker volume ls
```

### Important initialisation behaviour

PostgreSQL Docker initialisation scripts under:

```text
/docker-entrypoint-initdb.d/
```

are executed only when PostgreSQL initialises a new, empty data directory.

Therefore, rebuilding the PostgreSQL image and restarting it with an existing initialised `mpi_postgres_data` volume will **not** re-run the CSV and initialisation SQL scripts.

To perform a completely fresh database initialisation during development, the old container and volume can be removed:

```bash
sudo docker rm -f mpi-postgres
sudo docker volume rm mpi_postgres_data
sudo docker volume create mpi_postgres_data
```

**Warning:** removing the volume permanently deletes the PostgreSQL data stored in that volume.

## 4.4 Run PostgreSQL

Recommended env-file method:

```bash
sudo docker run -d \
  --name mpi-postgres \
  --network mpi-network \
  --env-file env/postgres.env \
  -p 5432:5432 \
  -v mpi_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  chbaah/dissertation-mpi:postgres-latest
```

For local development, the variables can alternatively be supplied directly:

```bash
sudo docker run -d \
  --name mpi-postgres \
  --network mpi-network \
  -e POSTGRES_DB=mpi \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -v mpi_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  chbaah/dissertation-mpi:postgres-latest
```

The direct password shown above is for local testing only.

## 4.5 Verify PostgreSQL

Check the container:

```bash
sudo docker ps
```

Check the logs:

```bash
sudo docker logs mpi-postgres
```

Check database readiness:

```bash
sudo docker exec mpi-postgres \
  pg_isready -U postgres -d mpi
```

Verify that the application table was created and populated:

```bash
sudo docker exec mpi-postgres \
  psql -U postgres -d mpi \
  -c "SELECT COUNT(*) FROM public.combined_prep_table;"
```


## 4.6 Push PostgreSQL images to Docker Hub

Authenticate:

```bash
sudo docker login
```

Push the immutable version:

```bash
sudo docker push chbaah/dissertation-mpi:postgres-v20260916
```

Then push the current/latest pointer:

```bash
sudo docker push chbaah/dissertation-mpi:postgres-latest
```

## 4.7 Pull a prebuilt PostgreSQL image

Specific version:

```bash
sudo docker pull chbaah/dissertation-mpi:postgres-v20260916
```

Current version:

```bash
sudo docker pull chbaah/dissertation-mpi:postgres-latest
```

For reproducible deployments, the version-specific tag is preferable.

## 4.8 Traditional PostgreSQL installation

Docker is not mandatory. PostgreSQL can instead be installed directly on the operating system.

The general process is:

1. Install PostgreSQL.
2. Start and enable the PostgreSQL service.
3. Create the `mpi` database.
4. Create the required tables using the SQL definition in:

```text
web/postgres/databasedocker/01-create-tables.sql
```

5. Import:

```text
analysis/finaldataframe/ntl_lcu_osm_data.west.africa.csv
```

using the corresponding import logic in:

```text
web/postgres/databasedocker/02-import-data.sql
```

6. Configure the Node.js and Plumber applications to connect to the host PostgreSQL instance.

---

# 5. R Plumber Prediction API

The R Plumber service performs MPI predictions. It loads the trained XGBoost model and the associated preprocessing recipe and exposes prediction functionality through HTTP API endpoints.

## 5.1 Build the Plumber image

```bash
sudo docker build \
  --progress=plain \
  -f web/plumber/Dockerfile \
  -t chbaah/dissertation-mpi:plumber-v20260916 \
  .
```

## 5.2 Tag and push

After successful testing:

```bash
sudo docker tag \
  chbaah/dissertation-mpi:plumber-v20260916 \
  chbaah/dissertation-mpi:plumber-latest

sudo docker push chbaah/dissertation-mpi:plumber-v20260916
sudo docker push chbaah/dissertation-mpi:plumber-latest
```

## 5.3 Run the Plumber container

Ensure PostgreSQL is running first:

```bash
sudo docker exec mpi-postgres \
  pg_isready -U postgres -d mpi
```

Then run Plumber:

```bash
sudo docker run -d \
  --name mpi-plumber \
  --network mpi-network \
  --env-file env/plumber.env \
  -p 3796:3796 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:plumber-latest
```

Alternatively:

```bash
sudo docker run -d \
  --name mpi-plumber \
  --network mpi-network \
  -e DB_HOST=mpi-postgres \
  -e DB_PORT=5432 \
  -e DB_NAME=mpi \
  -e DB_USER=postgres \
  -e DB_PASSWORD=postgres \
  -e PLUMBER_HOST=0.0.0.0 \
  -e PLUMBER_PORT=3796 \
  -e PLUMBER_FILE=/app/plumber.R \
  -e MODEL_DIR=/app/models \
  -e MODEL_FILE=xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds \
  -e RECIPE_FILE=xgboost_all_pred_notfm_recipe.rds \
  -p 3796:3796 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:plumber-latest
```

## 5.4 Verify the Plumber service

Check logs:

```bash
sudo docker logs mpi-plumber
```

Test the health endpoint:

```bash
curl http://localhost:3796/api/health
```

The response should indicate the service is running and that the model and recipe loaded successfully.

## 5.5 Pull a prebuilt image

```bash
sudo docker pull chbaah/dissertation-mpi:plumber-v20260916
```

or:

```bash
sudo docker pull chbaah/dissertation-mpi:plumber-latest
```

## 5.6 Traditional R/Plumber installation

The prediction API can also run directly on the operating system.

The general process is:

1. Install R and the required operating-system development libraries.
2. Install the required R packages.
3. Obtain:

```text
web/plumber/code/plumber.R
web/plumber/code/run.R
```

4. Ensure the trained XGBoost model and recipe are available.
5. Configure the database and model environment variables for the local environment.
6. Start the service using:

```bash
Rscript web/plumber/code/run.R
```

The Plumber service listens on port `3796` by default.

---

# 6. Node.js API

The Node.js/Express service provides application API endpoints and communicates with PostgreSQL.

## 6.1 Build the Node.js image

```bash
sudo docker build \
  -f web/node/Dockerfile \
  -t chbaah/dissertation-mpi:node-v20260917 \
  .
```

## 6.2 Tag and push

After successful testing:

```bash
sudo docker tag \
  chbaah/dissertation-mpi:node-v20260917 \
  chbaah/dissertation-mpi:node-latest

sudo docker push chbaah/dissertation-mpi:node-v20260917
sudo docker push chbaah/dissertation-mpi:node-latest
```

## 6.3 Run the Node.js container

```bash
sudo docker run -d \
  --name mpi-node \
  --network mpi-network \
  --env-file env/node.env \
  -p 3000:3000 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:node-latest
```

Alternatively:

```bash
sudo docker run -d \
  --name mpi-node \
  --network mpi-network \
  -e DB_HOST=mpi-postgres \
  -e DB_PORT=5432 \
  -e DB_NAME=mpi \
  -e DB_USER=postgres \
  -e DB_PASSWORD=postgres \
  -e PORT=3000 \
  -p 3000:3000 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:node-latest
```

## 6.4 Verify the Node.js service

Check logs:

```bash
sudo docker logs mpi-node
```

Test the health endpoint:

```bash
curl http://localhost:3000/api/health
```

A node js endpoint can be tested with:

```bash
curl http://localhost:3000/api/countries
```

## 6.5 Pull a prebuilt image

```bash
sudo docker pull chbaah/dissertation-mpi:node-v20260917
```

or:

```bash
sudo docker pull chbaah/dissertation-mpi:node-latest
```


## 6.6 Traditional Node.js installation

The Node.js/Express API can also run directly on the operating system without Docker.

The repository contains both `package.json` and `package-lock.json`, which define the Node.js dependencies required by the application. Therefore, individual packages such as Express, PostgreSQL (`pg`), CORS, and Morgan do not need to be installed manually.

The general process is:

1. Install a supported version of Node.js and npm on the operating system.

2. Change into the Node.js application directory:

```bash
cd web/node
```

3. Install the application dependencies using:

```bash
npm ci
```

`npm ci` uses `package-lock.json` to install the dependency versions recorded for the project. This provides a more reproducible installation than installing the dependencies individually.

The command creates a local `node_modules/` directory containing the installed packages. The `node_modules/` directory should not be committed to the Git repository.

4. Configure the environment variables required by the Node.js application. For example, if PostgreSQL is installed directly on the same machine:

```bash
export DB_HOST=127.0.0.1
export DB_PORT=5432
export DB_NAME=mpi
export DB_USER=postgres
export DB_PASSWORD=<database-password>
export PORT=3000
```

When PostgreSQL is running directly on the host operating system, `DB_HOST` should normally be `127.0.0.1` or another appropriate PostgreSQL hostname rather than the Docker container name `mpi-postgres`.

5. Start the Node.js API using the application's npm start command if one is defined in `package.json`:

```bash
npm start
```

Alternatively, the server can be started directly:

```bash
node js/server.js
```

6. Verify that the service is running:

```bash
curl http://localhost:3000/health
```

The Node.js API listens on port `3000` by default.

---

# 7. Frontend Application

The frontend provides the browser-based MPI prediction interface.

The Docker deployment uses Nginx to serve the static HTML, CSS, JavaScript, and image files. Nginx also acts as a reverse proxy, forwarding API requests from the frontend to the Node.js and Plumber containers over the mpi-network Docker network. The Nginx reverse-proxy configuration is stored in:

```bash
web/frontend/default.conf
```

The configuration serves the frontend files and forwards requests using the following paths:

```text
- /node-api/ forwards requests to Node.js API (Node container)
- /plumber-api/ forwards requests to R Plumber API (Plumber container)
```


## 7.1 Build the frontend image

```bash
sudo docker build \
  -f web/frontend/Dockerfile \
  -t chbaah/dissertation-mpi:frontend-v20260917 \
  .
```

## 7.2 Tag and push

After successful testing:

```bash
sudo docker tag \
  chbaah/dissertation-mpi:frontend-v20260917 \
  chbaah/dissertation-mpi:frontend-latest

sudo docker push chbaah/dissertation-mpi:frontend-v20260917
sudo docker push chbaah/dissertation-mpi:frontend-latest
```

## 7.3 Run the frontend container

```bash
sudo docker run -d \
  --name mpi-frontend \
  --network mpi-network \
  --env-file env/frontend.env \
  -p 8080:80 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:frontend-latest
```

Alternatively, for a local deployment:

```bash
sudo docker run -d \
  --name mpi-frontend \
  --network mpi-network \
  -e NODE_API_URL=/node-api \
  -e PLUMBER_API_URL=/plumber-api \
  -p 8080:80 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:frontend-latest
```

The frontend variables use relative paths rather than Docker container names or host ports. JavaScript running in the browser sends requests such as /node-api/countries to the same Nginx server that provided the frontend. Nginx then forwards the request to mpi-node:3000 over the Docker network. Similarly, requests beginning with /plumber-api/ are forwarded to mpi-plumber:3796.

This prevents the browser from having to resolve Docker-internal container names such as mpi-node and mpi-plumber.

## 7.4 Verify the frontend

Check the logs:

```bash
sudo docker logs mpi-frontend
```

Test Nginx:

```bash
curl -I http://localhost:8080
```

The frontend should then be accessible locally at:

```text
http://localhost:8080
```

The following two tests can be used to confirm that Nginx can reverse proxy requests to the Node.js and Plumber APIs:


```text
curl http://localhost:8080/node-api/api/countries
```

```text
curl http://localhost:8080/plumber-api/api/health
```

The generated runtime configuration can also be inspected inside the container:

```bash
sudo docker exec mpi-frontend \
  cat /usr/share/nginx/html/js/config.js
```

## 7.5 Pull a prebuilt image

```bash
sudo docker pull chbaah/dissertation-mpi:frontend-v20260917
```

or:

```bash
sudo docker pull chbaah/dissertation-mpi:frontend-latest
```

## 7.6 Traditional frontend installation

The frontend can also be hosted directly on the operating system without using Docker. A web server such as Nginx or Apache HTTP Server can be used to serve the HTML, CSS, JavaScript and image files stored under `web/frontend/`.

The general process is:

1. Install a web server such as Nginx or Apache HTTP Server.

2. Configure the web server to serve the files stored under:

```text
web/frontend/
```

3. Configure `js/config.js` with the addresses of the Node.js and Plumber APIs. For example, when the frontend and APIs are running on the same local machine:

```javascript
window.APP_CONFIG = {
    NODE_API_URL: "http://localhost:3000",
    PLUMBER_API_URL: "http://localhost:3796"
};
```

4. Ensure the Node.js and Plumber services are running and can be reached using the addresses configured in `config.js`.

5. Access the frontend from a web browser and test the MPI prediction process.

The exact configuration required for Nginx or Apache will depend on the operating system and the environment where the application is deployed.



---

# 8. Starting the Complete Docker Application

Once the four Docker images have been built or downloaded, the application can be started. PostgreSQL should be started first because both the Node.js and Plumber services require access to the database. The Plumber and Node.js containers can then be started, followed by the frontend container.

The containers should therefore be started in the following order:

1. PostgreSQL
2. Plumber
3. Node.js
4. Frontend

Before starting the services that depend on PostgreSQL, the following command can be used to confirm that the database is ready:

```bash
until sudo docker exec mpi-postgres \
  pg_isready -U postgres -d mpi >/dev/null 2>&1
do
  sleep 1
done
```

Once PostgreSQL is ready, the remaining containers can be started.

---

# 9. Verifying the Complete Deployment

After starting the application, the following command can be used to confirm that the four containers are running:

```bash
sudo docker ps
```

The expected containers are `mpi-postgres`, `mpi-plumber`, `mpi-node`, and `mpi-frontend`.

The Docker network can also be inspected using:

```bash
sudo docker network inspect mpi-network
```

All four containers should be attached to the `mpi-network` network.

The PostgreSQL database can be checked by querying the `combined_prep_table` table:

```bash
sudo docker exec mpi-postgres \
  psql -U postgres -d mpi \
  -c "SELECT COUNT(*) FROM public.combined_prep_table;"
```

The Plumber API can be checked using:

```bash
curl http://localhost:3796/api/health
```

The Node.js API can be checked using:

```bash
curl http://localhost:3000/api/health
```

The frontend can also be checked from the command line using:

```bash
curl -I http://localhost:8080
```

For a complete test from the frontend to the 2 backend api, the following commands can be executed:

```bash
curl http://localhost:8080/node-api/api/countries
```

```bash
curl http://localhost:8080/plumber-api/api/health
```

For a local deployment, the web application can finally be accessed from a browser using:

```text
http://localhost:8080
```

A prediction can then be performed through the web interface to confirm that the frontend, Node.js API, Plumber API and PostgreSQL database are working together.

---

# 10. Container Management

The four containers can be stopped using:

```bash
sudo docker stop mpi-frontend mpi-node mpi-plumber mpi-postgres
```

When restarting the application, PostgreSQL should be started first, followed by the other services:

```bash
sudo docker start mpi-postgres
sudo docker start mpi-plumber
sudo docker start mpi-node
sudo docker start mpi-frontend
```

The logs for each container can be viewed using:

```bash
sudo docker logs mpi-postgres
sudo docker logs mpi-plumber
sudo docker logs mpi-node
sudo docker logs mpi-frontend
```

Where continuous monitoring of a container log is required, the `-f` option can be used:

```bash
sudo docker logs -f <container-name>
```

---

# 11. Docker Image Versioning

Separate image tags are used for the four components of the application. The version number is based on the date on which the image was created. For example:

```text
chbaah/dissertation-mpi:postgres-v20260916
chbaah/dissertation-mpi:plumber-v20260916
chbaah/dissertation-mpi:node-v20260917
chbaah/dissertation-mpi:frontend-v20260917
```

A `latest` tag is also maintained for each component:

```text
chbaah/dissertation-mpi:postgres-latest
chbaah/dissertation-mpi:plumber-latest
chbaah/dissertation-mpi:node-latest
chbaah/dissertation-mpi:frontend-latest
```

The versioned image is built and tested before it is tagged as the latest version. For example, if `postgres-v20260914` has been tested successfully, the same image can be tagged as:

```text
postgres-latest
```

The version-specific tags are not intended to be overwritten. This means that `postgres-v20260914` should continue to refer to the image that was created on that date. If changes are subsequently made to the PostgreSQL image, a new version tag should be created, for example:

```text
postgres-v20261020
```

After the new image has been tested, `postgres-latest` can then be updated to refer to the new image.

Keeping both versioned and latest tags makes it possible to identify the exact image used for a particular deployment while still providing a simple way of obtaining the most recent tested version.

---

# 12. Deployment Architecture

The web application consists of four separate services: the frontend, Node.js API, Plumber prediction API and PostgreSQL database. In the Docker implementation, each service runs in its own container.

The frontend is served by Nginx and provides the interface used to select the required country, region and year, or to manually provide predictor values for a single prediction. The frontend communicates with the Node.js API and the Plumber API using HTTP requests.

The Node.js API provides the application endpoints used to retrieve information required by the frontend. It connects to PostgreSQL to obtain the required data. The Plumber API is responsible for the MPI prediction process. It loads the trained XGBoost model and its preprocessing recipe and also connects to PostgreSQL when predictor data is required.

PostgreSQL stores the data required by the application. Within the Docker implementation, the database uses the `mpi_postgres_data` named volume so that the database files are retained when the PostgreSQL container is stopped or removed.

All four containers are attached to the `mpi-network` Docker bridge network. The Node.js and Plumber containers therefore connect to PostgreSQL using the PostgreSQL container name:

```text
mpi-postgres
```

rather than using a fixed container IP address.

The frontend is served by Nginx. Although the JavaScript code executes in the user's browser, the browser does not connect directly to the Node.js or Plumber containers. Instead, the frontend uses relative API paths. Requests beginning with /node-api/ and /plumber-api/ are sent to Nginx.

Nginx operates inside the mpi-frontend container and is connected to the mpi-network Docker network. It can therefore resolve the Docker container names mpi-node and mpi-plumber using Docker's internal DNS service. Nginx forwards Node.js requests to mpi-node:3000 and Plumber requests to mpi-plumber:3796.

This allows the browser to access the application through a single frontend address, such as http://localhost:8080, without requiring direct knowledge of the backend container names or ports.


```text
NODE_API_URL=/node-api
PLUMBER_API_URL=/plumber-api
```


The current Docker deployment can be represented as:

```text

                         User Browser
                              |
                              | HTTP
                              | port 8080
                              v
                    +--------------------+
                    |     Frontend       |
                    |       Nginx        |
                    |  Container port 80 |
                    |                    |
                    |    /node-api/      |
                    |   /plumber-api/    |
                    +---------+----------+
                              |
                 Docker internal network
                      (mpi-network)
                         /         \
                        /           \
                       v             v
              +---------------+   +---------------+
              |    Node.js    |   |   R Plumber   |
              |  Express API  |   | Prediction API|
              |   port 3000   |   |   port 3796   |
              +-------+-------+   +-------+-------+
                      |                   |
                      |                   |
                      +---------+---------+
                                |
                                | Docker internal network (mpi-network)
                                v
                       +------------------+
                       |    PostgreSQL    |
                       |   mpi database   |
                       |    port 5432     |
                       +------------------+


```

The Node.js, Plumber and PostgreSQL ports are currently published to the Docker host to allow the individual services to be tested directly during development. The frontend does not require direct access to these published ports because normal application requests are routed through Nginx. In a production deployment, unnecessary host-port mappings can be removed.

---

# 13. Future Improvements

The current deployment intentionally keeps the four application components separate to demonstrate the architecture and allow each service to be tested independently.

Possible future improvements include:

* Docker Compose for managing the four containers as one application stack.
* Docker health checks and dependency readiness checks.
* TLS/HTTPS for production deployment.
* Improved secret management instead of plain-text environment files.
* Restricting Node.js, Plumber and PostgreSQL from unnecessary public host-port exposure. In the run container the host port can be removed. When using cloud infra. security groups can be configured to prevent inbound connections on those ports.
* Automated image builds and testing using CI/CD.
* Automated deployment from versioned Docker images.
