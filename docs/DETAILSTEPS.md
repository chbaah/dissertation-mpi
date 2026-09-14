# Multidimensional Poverty Index Prediction

The project investigates the prediction of subnational Multidimensional Poverty Index (MPI) values using open-source geospatial and infrastructure data.
This documentation provides guidance on creating the 4 docker images required to run the web application required to predict MPI values.

**From Surveys to Algorithms: Machine Learning for Multidimensional Poverty Index Assessment**


## Postgresql Database

1. Change directory into the dissertation-mpi directory after cloning from github
- cd ~/dissertation-mpi

2. Build the postgresql docker image. The image imports the same final dataframe (csv file)  used to build the different models. The dataframe is stored in the directory path ~/dissertation-mpi/analysis/finaldataframe. The information stored in the dataframe is required for the application to successfully work.
- sudo docker build \
  -f web/postgres/Dockerfile \
  -t chbaah/dissertation-mpi:postgres-v20260914 \
  .

3. Optional step, tag the image as latest depending on your use-case.
- sudo docker tag \
  chbaah/dissertation-mpi:postgres-v20260914 \
  chbaah/dissertation-mpi:postgres-latest

4. The same prebuild docker image file can be pulled from the chbaah docker hub using the command:
- sudo docker pull chbaah/dissertation-mpi:postgresql-v20260914 (specific image version)
- sudo docker pull chbaah/dissertation-mpi:postgresql-latest (retrieve the latest image version)

5. To run the image as a container execute the below. First create a named volume
- sudo docker volume create mpi_postgres_data
- sudo docker run -d \
  --name mpi-postgres \
  --network mpi-network \
  --env-file env/postgres.env \
  -p 5432:5432 \
  -v mpi_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  chbaah/dissertation-mpi:postgres-latest

Alternatively the docker can be run using the command:
- sudo docker run -d \
  --name mpi-postgres \
  --network mpi-network \
  -e POSTGRES_DB=mpi \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -v mpi_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  chbaah/dissertation-mpi:postgres-v20260914

6. Push the images to docker hub
- sudo docker login
- sudo docker push chbaah/dissertation-mpi:postgres-v20260914
- sudo docker push chbaah/dissertation-mpi:postgres-latest

7. An alternative to using docker is to install postgresql directly on the operating system and import the csv stored in the analysis/finaldataframe/ntl_lcu_osm_data.west.africa.csv into the mpi database.

## Plumber API

1. Change directory into the dissertation-mpi directory after cloning from github
- cd ~/dissertation-mpi

2. Build the plumber API docker image. This container performs the MPI predictions.
- sudo docker build \
  -f web/plumber/Dockerfile \
  -t chbaah/dissertation-mpi:plumber-v20260914 \
  .

3. Optional step, tag the image as latest depending on your use-case.
- sudo docker tag \
  chbaah/dissertation-mpi:plumber-v20260914 \
  chbaah/dissertation-mpi:plumber-latest

4. Push the images to docker hub
- sudo docker login
- sudo docker push chbaah/dissertation-mpi:plumber-v20260914
- sudo docker push chbaah/dissertation-mpi:plumber-latest

5. The same prebuild docker image file can be pulled from the chbaah docker hub using the command:
- sudo docker pull chbaah/dissertation-mpi:plumber-v20260914 (specific image version)
- sudo docker pull chbaah/dissertation-mpi:plumber-latest (retrieve the latest image version)

6. To run the image as a container execute the below.
- sudo docker run -d \
  --name mpi-plumber \
  --network mpi-network \
  --env-file env/plumber.env \
  -p 3796:3796 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:plumber-v20260914

7. Alternatively it can be run using the command below:
- sudo docker run -d \
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
  chbaah/dissertation-mpi:plumber-v20260914

8. An alternative installation without using docker is to setup an R environment, download the 2 scripts namely: run.R and plumber.R

## NodeJS API

1. Change directory into the dissertation-mpi directory after cloning from github
- cd ~/dissertation-mpi

2. Build the nodejs API docker image. This container performs the MPI predictions.
- sudo docker build \
  -f web/node/Dockerfile \
  -t chbaah/dissertation-mpi:node-v20260914 \
  .

3. Optional step, tag the image as latest depending on your use-case.
- sudo docker tag \
  chbaah/dissertation-mpi:node-v20260914 \
  chbaah/dissertation-mpi:node-latest

4. Push the images to docker hub
- sudo docker login
- sudo docker push chbaah/dissertation-mpi:node-v20260914
- sudo docker push chbaah/dissertation-mpi:node-latest

5. The same prebuild docker image file can be pulled from the chbaah docker hub using the command:
- sudo docker pull chbaah/dissertation-mpi:node-v20260914 (specific image version)
- sudo docker pull chbaah/dissertation-mpi:node-latest (retrieve the latest image version)

6. To run the image as a container execute the below.
- sudo docker run -d \
  --name mpi-node \
  --network mpi-network \
  --env-file env/node.env \
  -p 3000:3000 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:node-v20260914

7. Alternatively, it can be run using the command below:
- sudo docker run -d \
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
  chbaah/dissertation-mpi:node-v20260914

8. The approach of installing node js directly on the os can be also be adopted; where docker is not used. This will require standard node js installation per os type.


## Frontend App
1. Change directory into the dissertation-mpi directory after cloning from github
- cd ~/dissertation-mpi

2. Build the frontend web interface docker image.
- sudo docker build \
  -f web/frontend/Dockerfile  \  
  -t chbaah/dissertation-mpi:frontend-v20260914 \
  . 

3. Optional step, tag the image as latest depending on your use-case.
- sudo docker tag \
  chbaah/dissertation-mpi:frontend-v20260914 \
  chbaah/dissertation-mpi:frontend-latest

4. Push the images to docker hub
- sudo docker login
- sudo docker push chbaah/dissertation-mpi:frontend-v20260914
- sudo docker push chbaah/dissertation-mpi:frontend-latest

5. The same prebuild docker image file can be pulled from the chbaah docker hub using the command:
- sudo docker pull chbaah/dissertation-mpi:frontend-v20260914 (specific image version)
- sudo docker pull chbaah/dissertation-mpi:frontend-latest (retrieve the latest image version)

6. To run the image as a container execute the below.
- docker run -d \
  --name mpi-frontend \
  --network mpi-network \
  --env-file env/frontend.env \
  -p 8080:80 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:frontend-v20260914

7. Alternatively, it can be run using the command below:
- sudo docker run -d \
  --name mpi-frontend \
  --network mpi-network \
  -e NODE_API_URL=http://localhost:3000 \
  -e PLUMBER_API_URL=http://localhost:3796 \
  -p 8080:80 \
  --restart unless-stopped \
  chbaah/dissertation-mpi:frontend-v20260914

8. An alternative approach without using docker is to install nginx or apache httpd and configure as appropriate.


## Creating the docker network
This is the network where all 4 containers will be attached to.

1. The command as specified below can be used to create the bridge network
- docker network create --driver bridge mpi-network
