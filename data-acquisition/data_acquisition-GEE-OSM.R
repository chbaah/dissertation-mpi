# ===============================================================
# DATA ACQUISITION, INTEGRATION AND PRE-MODELLING EXPLORATORY ANALYSIS
# ===============================================================
# This script covers the acquisition and preparation of the data used in the
# dissertation, including Google Earth Engine, OpenStreetMap and OPHI data.
# It continues through the exploratory assessment performed before model
# splitting and fitting. The model-building section of the original script
# has been intentionally excluded because it is handled in the separate
# modelling scripts.
#
# Executable statements from the retained portion of the original script
# have not been changed in this commented version.

################################################################################
# Remove all objects
################################################################################
rm(list=ls())

#################################################################################
# Install recommended Library
#
# 1. blackmarbler: is a R  package that provides a simple way to use
# nighttime light data from NASA's Black Marble.
# Source: https://cran.r-project.org/web/packages/blackmarbler/readme/README.html
#
# 2. geodata: function for downloading of geographic data for use in spatial 
# analysis and mapping.
# Source: https://cran.r-project.org/web/packages/geodata/geodata.pdf
#
# 3. sf: Support for simple features, a standardized way to encode spatial vector
# data.
# Source: https://cran.r-project.org/web/packages/sf/sf.pdf
#
# 4. raster: Reading, writing, manipulating, analyzing and modeling of spatial data. 
# This package has been superseded by the "terra" package.
# first install r-cran-terra using the command: "sudo apt -y install r-cran-terra"
# Source: https://cran.r-project.org/web/packages/raster/raster.pdf
#
# 5. caret: for data analysis and model buidling
# Source: https://topepo.github.io/caret/
#
# 6. tidyverse: for data manipulation. consist of the following:
# ggplot2, dplyr, tidyr, readr, purr, tibble, stringr, forcats
# Source: https://www.tidyverse.org/packages/
#
# 7. doParallel: The doParallel package is a “parallel backend” for the foreach package. 
# It provides a mechanism needed to execute foreach loops in parallel.
# Source: https://cran.r-project.org/web/packages/doParallel/vignettes/gettingstartedParallel.pdf
# 
# 8. exactextractr: perform zonal statistics
# 
# 9. string4: extract specific part of a string
#
# 10. remotes - helps to install remote packages
# 
# 11. epitools - function julian2date
# convert julian date to standard date
#
# 12. readxl -Read Excel
#
# 13. GGally - correlation matrix
#
# 14. data.table 
# for sql like query
#
# 15. Compressive Sensing
# R1imagic
#
# 16. Zoo - impute missing data
#
# 17. imputeTS
#
# 18. COINr - scaling package. Normalization and its family
#
# 19. tidymodels
#
# 20. modeltime
#
# 21. timetk
#
# 22. kernlab 
# Required to build support vector regression model

# 23. moments 
# Required for skewness and kurtosis cal

# 24. tidymodel - kknn
# required for knn model in tidymodels

#25. tidymodel - ranger
# required for random forest model in tidymodels

#26. MODIStsp - MODISTSP package for Landcover/user
# rgee and reticulate
#######################Installation of packages#################################
install.packages("geodata")
# OS installation
# apt-get install libabsl-dev -y
# apt-get install cmake -y
install.packages('s2')
install.packages("sf")

# OS installation before installing terra
#sudo apt-get update
#sudo apt-get install -y libudunits2-dev libgdal-dev libgeos-dev libproj-dev
#rm -f /home/charles/R/x86_64-pc-linux-gnu-library/4.5/00LOCK-terra


install.packages("terra")
install.packages("raster", dependencies = TRUE)
install.packages("tidyverse")
install.packages("tidyterra")*
install.packages("doParallel")
install.packages("exactextractr")*
install.packages("stringr")
install.packages("remotes")
install.packages("epitools")
install.packages("readxl")
install.packages("GGally")
install.packages("data.table")
install.packages("tidymodels")
install.packages("finetune")
install.packages("kernlab")
install.packages("devtools")
install.packages("glmnet")
install.packages("ggrepel")
install.packages("future")
install.packages("doFuture")
install.packages("moments")

# GEE Land cover/user packages
remotes::install_github("r-spatial/rgee")
install.packages("reticulate")
#remotes::install_github("chbaah/rgee")

# Install the following on the OS:protobuf-compiler, libjq-dev
install.packages("jqr", dependencies = T)
install.packages("protolite")
install.packages("jsonlite")
install.packages("stars")
install.packages("googledrive")
install.packages("osmdata")
install.packages("stringi")
install.packages("janitor")


# Install postgreSQL client library for integration between r and postgres
install.packages("RPostgreSQL")
install.packages("jsonlite")

# Load the packages required for spatial processing, data manipulation,
# Google Earth Engine access, OpenStreetMap extraction and exploratory analysis.
################################################################################
# Load all packages installed
# lubridate package for the processing of date
################################################################################
#library(blackmarbler)
library(geodata)
library(sf)
library(raster)
library(terra)
library(tidyverse)
library(ggrepel)
library(tidyterra)
library(exactextractr)
library(lubridate)
library(stringr)
library(epitools)
library(readxl)
library(GGally)
library(tidymodels)
library(finetune)
library(tools)
library(glmnet)
library(kknn)
library(MASS)
library(remotes)
library(rgee)
library(reticulate)
library(stars)
library(googledrive)
library(osmdata)
library(stringi)
library(moments)
library(RPostgreSQL)
library(future)
library(doFuture)


################################################################################

# Define the main project directories and the source identifiers used to store
# and retrieve nighttime-light, land-cover, OSM, shapefile and MPI data.
############################# Variables Definition##############################

main_dir      <- "/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio"
mpi_dir       <- paste0(main_dir,"/","MPI")
shpfiledir    <- paste0(main_dir, "/", "dissertationdatasf")
shpdirgadm    <- paste0(shpfiledir, "/", "GADM")
shpdirdhs     <- paste0(shpfiledir, "/", "DHS")
ntldata_dir   <- paste0(main_dir,"/", "data", "/", "ntldata")
lcudata_dir   <- paste0(main_dir,"/", "data", "/", "lcudata")
osmdata_dir   <- paste0(main_dir,"/", "data", "/", "osmdata")
osmtmp_dir    <- paste0(main_dir,"/", "data", "/", "osmdatatemp")
log_dir       <- paste0(main_dir, "/", "log")
logfile       <- paste0(log_dir, "/", Sys.Date(),".log")
ntl_h5_dir    <- paste0(main_dir, "/", "h5", "/", "ntl")
lcu_h5_dir    <- paste0(main_dir, "/", "h5","/", "lcu")

mpi_file_name <- "countriesmpidata.xlsx" 

# Write the data frame to file
destdftofile  <- paste0(main_dir, "/", "finaldataframe")

# Google earth engine data sources
pathntl <- "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
pathmodis <- "MODIS/061/MCD12Q1"
################################################################################


# Configure the Python environment and authenticate the R session for access
# to Google Earth Engine and the associated Google Drive export workflow.
#######################Initialize google earth engine###########################

#Get the username
HOME <- Sys.getenv("HOME")
envname <- 'r-reticulate'

# Install python - using os commands
#1. /home/charles/.virtualenvs
#2. Create virtualenv
# /usr/bin/python3.12 -m venv r-reticulate
# source r-reticulate/bin/activate 
#3. Instance the required models
# pip install numpy
# pip install earthengine_api

# /home/charles/.pyenv/versions/3.10.16/bin/python3.10"
reticulate::install_python()

# create virtual env
reticulate::virtualenv_create(
  envname = envname,
  python = "/usr/bin/python3"
)


# Initialize the Python Environment
reticulate::use_virtualenv(envname)

# From the OS perform the following action to install earthengin api
# cd ~/.virtualenvs
# source r-reticulate/bin/activate
# pip3 install earthengine_api



#
rgee::ee_install_set_pyenv(
  py_path = "/home/charles/.virtualenvs/r-reticulate/bin/python", # Change it for your own Python PATH
  py_env = envname # Change it for your own Python ENV
)

# restart r session and run ee_check()
rgee::ee_check()

# check python virtualenv being used
virtualenv_list()

# authenticate
rgee::ee_Authenticate(
  user = "baah.charles@gmail.com"
)

# Initialize
rgee::ee_Initialize(
  user = "baah.charles@gmail.com",
  drive = T,
  project = "ee-baahcharles"
)


# Run this step on subsequent execution
envname <- 'r-reticulate'
reticulate::use_virtualenv(envname)

# To clean up if not working
rgee::ee_clean_user_credentials(user = "baah.charles@gmail.com")
rgee::ee_Authenticate()
#reticulate::py_run_string("import ee; ee.Authenticate()")

#This generated the rgee_sessioninfo.txt file
rgee::ee_Initialize(
  user = "baah.charles@gmail.com",
  drive = T,
  project = ee-baahcharles
)

rgee::ee_check()

# Authorize rgee to manage Earth Engine resources, Google Drive, and Google Cloud Storage
rgee::ee_clean_user_credentials(user = "baah.charles@gmail.com")
reticulate::py_run_string("import ee; ee.Authenticate()")
# will test with project = 'ee-baahchrles" if that also works - 622016578699
reticulate::py_run_string("import ee; ee.Initialize(project='622016578699')")
#ee_Initialize(project='ee-baahcharles')
#==============End==============


#######################Initialize google earth engine###########################



# Define a helper used as an alternative route for transferring exported
# Earth Engine files from Google Drive to the local project directories.
# Files transfer
transfertolocalfromdrive <- function(sourcefolder, destinationdirectory){
  
  # Get list of files in the directory
  filesindirectory <- drive_ls(
    path = paste0(sourcefolder, "/")
  )$name
  
  # Download file to destinationdirectory
  lapply(filesindirectory, drive_download(x, path = paste0(destinationdirectory), overwrite = TRUE))
  
}

# Define the main Earth Engine extraction function. For each country and
# required year, the function identifies the appropriate administrative
# boundary and downloads the nighttime-light and land-cover raster data.
# Download the westafrican ntl data
westafrican.ntl.lcu <- function(countrydata){
  
  print(paste0("Countrydata is: ", countrydata))
  print(paste0("Countrydata class is : ", class(countrydata)))
  countryname <- data.frame(countrydata) %>% dplyr::select(country) %>% unique()
  assocyears <- data.frame(countrydata) %>% dplyr::select(year)
  shapefilesourceval <- data.frame(countrydata) %>% dplyr::select('shapefile_source') %>% unlist() %>% as.vector()
  shapefilesource <- data.frame(countrydata) %>% dplyr::select('shapefile_source')
  ccode <- data.frame(countrydata) %>% dplyr::select('countrycode') %>% unique()
  
  # Define file name for shapefile
  filename = paste0(ccode, ".shp")
  
  print(paste0("Country is: ", countryname))
  print(paste0("year is: ", assocyears))
  print(paste0("class of assocyears variable is: ", class(assocyears)))
  print(paste0("shapefile source is :", shapefilesource))
  print(paste0("shapefilesource value is :", shapefilesourceval))
  print(paste0("class of shapfilesource value: ", class(shapefilesourceval)))
  print(paste0("country code :", ccode))
  
  shapefileyear = unlist(strsplit(shapefilesourceval, " "))
  print(paste0("shapefileyear from shapfile source columns : ", as.vector(shapefileyear)[1]))
  
  
  
  # Choropleth map
  #choropleth.country <- geodata::gadm(country = countryname, path = tempdir(), level = 1, version="4.1", resolution = 1)
  #plot(choropleth.country)
  
  
  
  
  
  ################################################################################
  # Make raster stack of nighttime light across multiple time periods
  # Get the bearer token from the LAADS DAAC site: https://ladsweb.modaps.eosdis.nasa.gov/search/order/1
  # Downloading the product
  ################################################################################
  
  
  #print(Bearer)
  
  
  # Verify if directory where ntl raster file will be stored exist. if not create it
  
  if (file.exists(ntldata_dir)){
    setwd(file.path(ntldata_dir))
  } else {
    dir.create(file.path(ntldata_dir))
    setwd(file.path(ntldata_dir))
  } # end of if statement
  
  
# Retain observation years from 2012 onwards because the VIIRS nighttime-light
# source used in this study is required only for the study period covered here.
  # Extract the unique vector in the tapply function
  vecyears <- assocyears %>% dplyr::select(year) %>% unlist() %>% as.vector()
  print(class(vecyears))
  print(paste0("vecyears: ", vecyears))
  
  downloadyear <- lubridate::ymd(vecyears, truncated = 2L)
  
  
  # Remove date less than 2012
  downloadyear <- downloadyear[!(downloadyear <= ymd(as.character("2011-12-31")))]
  print(paste0("downloadyear : ", downloadyear))
  
  
  
  if (all(is.na(as.vector(downloadyear))))  {
    write(paste0(Sys.time(), ": Info: ", countryname, ": No valid date for NTL data download."), file=logfile, append = TRUE)
    print(paste0(countryname, ": No valid date for NTL data download"))
    
  } else {
    
    for (i in 1:length(downloadyear)){
      print(paste0("Processing specific date:", downloadyear[i]))
      print(paste0("Confirming data type of date: ", class(downloadyear[i])))
      
      ################
      
      vallevel <- stringr::str_sub(shapefilesourceval[i], -1)
      print(paste0("vallevel is : ", vallevel))
      
      # Get shapefile year
      intermshapefileyear = unlist(strsplit(shapefilesourceval[i], " "))
      shapefileyear = as.vector(intermshapefileyear)[1]
      print(paste0("shapefileyear from shapfile source columns : ", shapefileyear))
      
# Select either the GADM or DHS administrative boundary specified for the
# current country-year observation before requesting the raster data.
      # Checking the source of the shapefile
      if (sum(str_detect(shapefilesourceval[i], 'GADM')) > 0 ){
        
        
        sf.file_path <- paste0(shpdirgadm,"/", ccode, "/", filename)
        
        if (file.exists(sf.file_path)){
          
          # Reading sf from file
          df.sf <- sf::st_read(sf.file_path)
          #df.sf
          
          # Zonal statistics - preparation
          df.sf$id <- 1:max(nrow(df.sf))
          print(df.sf)
          
          
        } else {
          
          # Choropleth map Download
          choropleth.country <- geodata::gadm(country = ccode, path = tempdir(), level = vallevel, version="3.6", resolution = 1)
          
          # if sharp file does not exist, create one and read
          choropleth.country.df <- choropleth.country %>% st_as_sf()
          sf::st_write(choropleth.country.df, sf.file_path)
          
          # Reading sf from file
          df.sf <- sf::st_read(sf.file_path)
          
          # Zonal statistics - preparation
          df.sf$id <- 1:max(nrow(df.sf))
          print(df.sf)
          
        } # end if statement for gadm sf
        
        
      } else {
        
        # Use shape file in DHS folder downloaded from the DHS site: 
        sf.file_path <- paste0(shpdirdhs,"/", ccode, "/", shapefileyear, "/", filename)
        
        # Reading sf from dhs file
        df.sf <- sf::st_read(sf.file_path)
        
        # Zonal statistics - preparation
        df.sf$id <- 1:max(nrow(df.sf))
        print(df.sf)
        
      } # End GADM and DHS statement
      
      ###############
      
      
      # raster file name and location for night time light
      #rasterfilename <- paste0(countryname, "_ntldata_", downloadyear[i], format=".tif")
      rasterfilename <- paste0(ccode, "_ntldata_", downloadyear[i], format=".tif")
      print(paste0("ntl raster file name:", rasterfilename))
      
      if (file.exists(paste0(ntldata_dir, "/", rasterfilename))){
        write(paste0(Sys.time(), ": Info: ", " Raster file : ", rasterfilename, " already exist on the disk before this execution.", " Redownloading  will not occur"), file=logfile, append = TRUE)
        print(paste0(rasterfilename, " already exist on the disk before this execution.", " Redownloading will not occur"))
      } else {
        
        # Read in the shapefile as gee object
        df.sf <- sf::st_read(sf.file_path) %>%
          sf_as_ee()
        #df.sf <- sf::st_read(sf.file_path)
        
        
        
        # Get the coordinates of the shapefile
        country_bounds <- df.sf$geometry()
        
        
# Construct the annual nighttime-light image by filtering the VIIRS collection
# to the country boundary and observation year, selecting average radiance
# and averaging the available images within that year.
        # Yearly composite nighttime light data download
        #ntldata <- bm_raster(roi_sf = df.sf,
        #                    product_id = "VNP46A4",
        #                     date = downloadyear[i],
        #                     bearer = Bearer,
        #                     output_location_type = "r_memory",
        #                     # file_dir = file.path(paste0("/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/", "ntldata")),
        #                     # file_prefix = countryname,
        #                     # file_skip_if_exists = TRUE,
        #                     h5_dir = file.path(ntl_h5_dir),
        #                     check_all_tiles_exist = FALSE
        #)
        
        
        # Yearly composite nighttime light data download
        
        ntlintermfile <- paste0(ccode, "_ntldata_", downloadyear[i])
        ntlrasterfilename <- paste0(ccode, "_ntldata_", downloadyear[i], format=".tif")
        ntlyearofdata <- year(downloadyear[i])
        ntlstartdate  <- ee$Date$fromYMD(ntlyearofdata, 01, 01)
        ntlenddate    <- ee$Date$fromYMD(ntlyearofdata, 12, 31)
        
        
        # Processing nighttime light data into a raster file
        ntldata <- rgee::ee$ImageCollection(pathntl)$
          filterBounds(country_bounds)$
          filterDate(ntlstartdate, ntlenddate)$
          select("avg_rad")$
          mean()$
          reproject(crs=country_bounds$projection(),
                    scale=country_bounds$projection()$nominalScale()
          )
        
        
# Export the annual nighttime-light raster to Google Drive and then transfer
# the completed GeoTIFF to the local nighttime-light data directory.
        # Move data to google drive
        my_ntl_task <- rgee::ee_image_to_drive(
          ntldata,
          description = "myNTLExportImageTask",
          folder = "ntl_backup",
          fileNamePrefix = ntlintermfile,
          timePrefix = FALSE,
          maxPixels = 100000000,
          skipEmptyTiles = FALSE,
          fileFormat = "GeoTIFF",
          region = country_bounds,
          crs = country_bounds$projection()
        )
        
        my_ntl_task$start()
        processstatusntl <- ee_monitoring(my_ntl_task, max_attempts = 60)
        
        
        # Copy data from google drive to local machine
        #setwd(lcudata_dir)
        print(paste0("status of ntl ee_monitoring:", processstatusntl))
        ntl.img <- ee_drive_to_local(task = my_ntl_task,
                                     dsn = paste0(ntldata_dir, "/", ntlrasterfilename),
                                     overwrite = TRUE,
                                     metadata = F)
        
        if (is.null(ntl.img)){
          #if (is.null(ntldata)){
          # write to log file
          write(paste0(Sys.time(), ": Warn: ", "image file : ", countryname, " : ", downloadyear[i], " could not be downloaded using 'ee_drive_to_local' API. Attempting alternative download."), file=logfile, append=TRUE)
          transfertolocalfromdrive(ntl_backup, ntldata_dir)
        } else {
          write(paste0(Sys.time(), ": Info: ", "image file : ", countryname, ":", downloadyear[i], " downloaded successfully to local disk"), file=logfile, append=TRUE)
          # Write the rasterstack variable to a file for further processing
          #print(paste("raster file data:", ntldata))
          #crs(ntldata) <- crs(df.sf)
          #terra::writeRaster(ntldata, filename=rasterfilename, overwrite=TRUE)
          
          write(paste0(Sys.time(), ": Info: ", "Raster file : ", ntl.img, " written to disk successfully."), file=logfile, append=TRUE)
          print(paste0(ntl.img," was successfully written to disk"))
          #write(paste0(Sys.time(), ": Info: ", "Raster file : ", ntldata, " written to disk successfully."), file=logfile, append=TRUE)
          #print(paste0(ntldata," was successfully written to disk"))
          rgee::ee_clean_container(name = "ntl_backup", type = "drive")
          
        } # end of null test
        
      } # End of if statement for rasterfilename exist
      
      
# Obtain the MODIS land-cover raster for the same country boundary and year
# so that the spatial and temporal identifiers correspond with the MPI record.
      ####### Obtain the Land Cover/Use data based on the same shape file
      
      # raster file name and location for Land Cover/Use data
      lcuintermfile <- paste0(ccode, "_lcudata_", downloadyear[i])
      lcurasterfilename <- paste0(ccode, "_lcudata_", downloadyear[i], format=".tif")
      lcuyearofdata <- year(downloadyear[i])
      lcustartdate  <- ee$Date$fromYMD(lcuyearofdata, 01, 01)
      lcuenddate    <- ee$Date$fromYMD(lcuyearofdata, 12, 31)
      
      
      
      print(paste0("land classification and use folder:", lcudata_dir))
      print(paste0("lcu raster file name:", lcurasterfilename))
      print(lcustartdate)
      print(lcuenddate)
      
      if (file.exists(paste0(lcudata_dir, "/", lcurasterfilename))){
        write(paste0(Sys.time(), ": Info: ", " Raster file : ", lcurasterfilename, " already exist on the disk before this execution.", " Redownloading  will not occur"), file=logfile, append = TRUE)
        print(paste0(lcurasterfilename, " already exist on the disk before this execution.", " Redownloading will not occur"))
      } else {
        print(paste0("Hello charles"))
        print(paste0("sf.file_path:", sf.file_path))
        
        # Read in the shapefile as gee object
        df.sf <- sf::st_read(sf.file_path) %>%
          sf_as_ee()
        
        
        # Get the coordinates of the shapefile
        country_bounds <- df.sf$geometry()
        
        
# Select the MODIS IGBP land-cover classification layer (LC_Type1) for the
# required year and prepare it for export from Google Earth Engine.
        # The data for use Land cover / use
        lcudata <- rgee::ee$ImageCollection(pathmodis)$
          filterBounds(country_bounds)$
          filterDate(lcustartdate, lcuenddate)$
          select("LC_Type1")$first()$
          reproject(crs=country_bounds$projection(), 
                    scale=country_bounds$projection()$nominalScale()
          )
        
        
        # Move data to google drive
        my_task <- rgee::ee_image_to_drive(
          lcudata,
          description = "myExportImageTask",
          folder = "lcu_backup",
          fileNamePrefix = lcuintermfile,
          timePrefix = FALSE,
          maxPixels = 100000000,
          skipEmptyTiles = FALSE,
          fileFormat = "GeoTIFF",
          region = country_bounds,
          crs = country_bounds$projection()
        )
        
        my_task$start()
        processstatus <- ee_monitoring(my_task, max_attempts = 60)
        
        
        # Copy data from google drive to local machine
        #setwd(lcudata_dir)
        print(paste0("status of ee_monitoring:", processstatus))
        img <- ee_drive_to_local(task = my_task,
                                 dsn = paste0(lcudata_dir, "/", lcurasterfilename),
                                 overwrite = TRUE,
                                 metadata = F)
        
        
        
        
        if (is.null(img)){
          # write to log file
          write(paste0(Sys.time(), ": Warn: ", "image file : ", countryname, " : ", downloadyear[i], " could not be downloaded using 'ee_drive_to_local' API. Attempting alternative download."), file=logfile, append=TRUE)
          transfertolocalfromdrive(lcu_backup, lcudata_dir)
          
          
        } else {
          write(paste0(Sys.time(), ": Info: ", "image file : ", countryname, ":", downloadyear[i], " downloaded successfully to local disk"), file=logfile, append=TRUE)
          # Write the rasterstack variable to a file for further processing
          #terra::writeRaster(lcurasterobject, filename=lcurasterfilename, overwrite=TRUE)
          
          write(paste0(Sys.time(), ": Info: ", "Raster file : ", img, " written to disk successfully."), file=logfile, append=TRUE)
          print(paste0(img," was successfully written to disk"))
          rgee::ee_clean_container(name = "lcu_backup", type = "drive")
          
        } # end of null test
      } # end of if statement for whether lcu raster file exist or not
      
      #############################################################
      
      
      ############################
      
    } # End of for loop
    
  } # if statement for NA date
  
} # End of westafrican.ntl.lcu function 



# Load the OPHI subnational MPI reference table and retain the country, region,
# year, MPI value and shapefile information required by the extraction workflow.
# Load the countries mpi dataframe
mpi_data <- readxl::read_excel(paste0(mpi_dir,"/", mpi_file_name), sheet = 1)
mpi_data <- mpi_data %>% dplyr::select(country, countrycode, region, year, mpi, datafile, shapefile_source)

# Review one of the country's data
mpi_data %>% dplyr::filter(countrycode == "BFA")

# Create the unique country-year-shapefile combinations used to control the
# Google Earth Engine downloads and avoid processing observations without
# an identified administrative-boundary source.
# Extract unique country, countrycode, shape file source and year
updated_mpi_table <- mpi_data %>% dplyr::select(country, countrycode, year, shapefile_source) %>% dplyr::group_by(country, countrycode, year, shapefile_source) %>% 
  filter(!is.na(shapefile_source)) %>% unique()

# use tapply to extract ntl and lcu data and write to raster file
tapply(X=updated_mpi_table[c("country","countrycode", "year","shapefile_source")], INDEX=updated_mpi_table$country, FUN=westafrican.ntl.lcu)


#============================temp==================================


#====================================================================

# Calculate regional nighttime-light summaries from the downloaded rasters.
# The function selects the matching GADM or DHS boundary and derives the
# weighted mean radiance and the percentage of the region covered by light.
zonalstatsfunntl <- function(rasterfilename, var2){
  
  fname <- rast(rasterfilename)
  fnamewithoutext <- file_path_sans_ext(rasterfilename)
  countrycodeonly <- str_extract(fnamewithoutext, '[^_]+')
  countryshpfilenamegadm <- paste0(shpdirgadm,"/", countrycodeonly, "/",  countrycodeonly, ".shp")
  
  
  
  # Verify source of shape file
  rasterfiledateval <- stringr::str_sub(fnamewithoutext, -10)
  rasterfiledate <- ymd(rasterfiledateval)
  sshapef <- var2 %>% filter(countrycode == countrycodeonly, year == year(rasterfiledate)) %>% dplyr::select(shapefile_source) %>% 
    unlist()
  sshapefval <- sshapef[4]
  
  #print(paste0("countryshpfilenamedhs: ", countryshpfilenamedhs))
  print(paste0("countryshpfilenamegadm: ", countryshpfilenamegadm))
  print(paste0("fname: ", fname))
  print(paste0("fnamewithoutext: ", fnamewithoutext))
  print(paste0("rasterfiledate: ", rasterfiledate))
  print(paste0("class of rasterfiledate: ", class(rasterfiledate)))
  print(paste0("sshapef: ", sshapef))
  print(paste0("sshapefval: ", sshapefval))
  
  
  # Check if shape file existed in either dhs or gadm folder
  
  if (sum(str_detect(sshapefval, 'GADM')) > 0 ) {
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamegadm)
    
    ###################
    # Zonal statistics - preparation
    countryshpfile.shpformat$id <- 1:max(nrow(countryshpfile.shpformat))
    
    print(paste("raster file name without extension: ", fname))
    print(paste("country shape file: ", countryshpfile.shpformat))
    
    exactextractr::exact_extract(fname, countryshpfile.shpformat, function(values, coverage_fraction){
      
      # Calculate total area in pixels (including cells with NaN or NA values)
      # total_area_in_px <- sum(coverage_fraction, na.rm = TRUE)
      total_area_in_px <- sum(coverage_fraction > 0)
      
      # Filter out NA values and zero coverage fractions
      valid_values <- values[!is.na(values) & coverage_fraction > 0]
      valid_coverage <- coverage_fraction[!is.na(values) & coverage_fraction > 0]
      
      if (length(valid_values) == 0 || length(valid_coverage) == 0) {
        # If no valid values, return NA for mean and other metrics
        ntl_mean_intensity <- NA
        perc_spread <- NA
      } else {
        # total pixel in terms of total nt.l intensity
        #ntl_total_intensity = sum(valid_values * valid_coverage, na.rm=TRUE)
        #print(ntl_total_intensity)
        
        # Ensure valid_values and valid_coverage are not empty before calculating weighted mean
        if (length(valid_values) > 0 && length(valid_coverage) > 0) {
          ntl_mean_intensity <- weighted.mean(valid_values, valid_coverage, na.rm = TRUE)
        } else {
          ntl_mean_intensity <- NA
        }
        print(paste("weighted mean: ", ntl_mean_intensity))
        
        # Calculating the spread.  Percentage of pixel covered by light factoring total pixel per region
        spread_of_ntl_in_px <- sum(valid_coverage[valid_values > 0], na.rm = TRUE)
        perc_spread = (spread_of_ntl_in_px / total_area_in_px) * 100
        
      }
      
      #Extract the date component from the raster layer name
      daty <- stringr::str_sub(rasterfilename, -14, -5)
      
      #ntl_px_zero_val = total_px_zero
      
      data.frame(
        collectiondate = daty,
        countrycode = countrycodeonly,
        #total_ntl_area_px = total_area_in_px,
        #ntl_total_intensity = ntl_total_intensity,
        ntl_mean_intensity = ntl_mean_intensity,
        ntl_coverage_perc = perc_spread
        
      )
    },
    append_cols = "id",
    full_colnames = TRUE,
    summarize_df = FALSE,
    stack_apply = FALSE
    ) %>% dplyr::inner_join(countryshpfile.shpformat, by = "id") %>% dplyr::select(!starts_with(c("GID", "VARNAME_", "NL_NAME", "ENGTYPE_", "CC_","HASC_" ))) %>%
      rename("region" = "NAME_1", "country" = "NAME_0", "adminarea" = contains("TYPE_")) 
    # %>% rename_if("NAME_2", region)
    
    ###################
    
  } else {
    
    intermshapefileyear = unlist(strsplit(sshapefval, " "))
    shapefileyear = as.vector(intermshapefileyear)[1]
    countryshpfilenamedhs <- paste0(shpdirdhs,"/", countrycodeonly, "/", shapefileyear, "/", countrycodeonly, ".shp")
    print(paste0("Path to DHS shape file : ", countryshpfilenamedhs))
    
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamedhs)
    print(paste0("countryshpfilenamedhs: ", countryshpfilenamedhs))
    
    #############################
    # Zonal statistics - preparation
    countryshpfile.shpformat$id <- 1:max(nrow(countryshpfile.shpformat))
    
    print(fname)
    print(countryshpfile.shpformat)
    
    exactextractr::exact_extract(fname, countryshpfile.shpformat, function(values, coverage_fraction){
      
      # Calculate total area in pixels (including cells with NaN or NA values)
      total_area_in_px <- sum(coverage_fraction > 0)
      
      # Filter out NA values and zero coverage fractions
      valid_values <- values[!is.na(values) & coverage_fraction > 0]
      valid_coverage <- coverage_fraction[!is.na(values) & coverage_fraction > 0]
      
      if (length(valid_values) == 0 || length(valid_coverage) == 0) {
        # If no valid values, return NA for mean and other metrics
        ntl_mean_intensity <- NA
        perc_spread <- NA
      } else {
        # total pixel in terms of total nt.l intensity
        # ntl_total_intensity = sum(valid_values * valid_coverage, na.rm=TRUE)
        # print(ntl_total_intensity)
        
        # Ensure valid_values and valid_coverage are not empty before calculating weighted mean
        if (length(valid_values) > 0 && length(valid_coverage) > 0) {
          ntl_mean_intensity <- weighted.mean(valid_values, valid_coverage, na.rm = TRUE)
        } else {
          ntl_mean_intensity <- NA
        }
        print(paste("weighted mean: ", ntl_mean_intensity))
        
        # Calculating the spread.  Percentage of pixel covered by light factoring total pixel per region
        spread_of_ntl_in_px <- sum(valid_coverage[valid_values > 0], na.rm = TRUE)
        perc_spread = (spread_of_ntl_in_px / total_area_in_px) * 100
        
      }
      
      
      #ntl_px_zero_val = total_px_zero
      
      #Extract the date component from the raster layer name
      daty <- stringr::str_sub(rasterfilename, -14, -5)
      
      data.frame(
        collectiondate = daty,
        countrycode = countrycodeonly,
        #total_ntl_area_px = total_area_in_px,
        #ntl_total_intensity = ntl_total_intensity,
        ntl_mean_intensity = ntl_mean_intensity,
        ntl_coverage_perc = perc_spread
      )
    },
    append_cols = "id",
    full_colnames = TRUE,
    summarize_df = FALSE,
    stack_apply = FALSE
    ) %>% dplyr::inner_join(countryshpfile.shpformat, by = "id")  %>% 
      dplyr::select(-c("ISO", "FIPS", "DHSCC", "SVYTYPE", "SVYYEAR", "CNTRYNAMEF", "CNTRYNAMES", 
                       "DHSREGSP", "SVYID", "REG_ID", "Svy_Map", "MULTLEVEL", "LEVELRNK", "REGVAR", "REGCODE", "OTHREGVAR", "OTHREGCO", "OTHREGNA", "LEVELCO", 
                       "REPALLIND", "REGNOTES", "DHSREGEN", "DHSREGFR", "SVYNOTES")) %>% 
      rename("region" = "REGNAME", "country" = "CNTRYNAMEE", "adminarea" = "LEVELNA")
    
    #############################
    
  } # End of if statement
} # generated dataframe func ending
############################################################################


# Calculate regional land-cover percentages from the downloaded MODIS raster.
# Each selected IGBP class is expressed as a percentage of the regional area
# represented by valid raster coverage.
#####################zonal statistics for Land Cover#########################

zonalstatsfunlcu <- function(rasterfilename, var2){
  
  fname <- rast(rasterfilename)
  fnamewithoutext <- file_path_sans_ext(rasterfilename)
  countrycodeonly <- str_extract(fnamewithoutext, '[^_]+')
  countryshpfilenamegadm <- paste0(shpdirgadm,"/", countrycodeonly, "/", countrycodeonly, ".shp")
  
  # Verify source of shape file
  shapefiledateval <- stringr::str_sub(fnamewithoutext, -10)
  shapefiledate <- ymd(shapefiledateval)
  sshapef <- var2 %>% filter(countrycode == countrycodeonly, year == year(shapefiledate)) %>% dplyr::select(shapefile_source) %>% 
    unlist()
  sshapefval <- sshapef[4]
  
  print(paste0("countryshpfilenamegadm: ", countryshpfilenamegadm))
  print(paste0("fname: ", fname))
  print(paste0("fnamewithoutext: ", fnamewithoutext))
  print(paste0("shapefiledate: ", shapefiledate))
  print(paste0("class of shapefiledate: ", class(shapefiledate)))
  print(paste0("sshapef: ", sshapef))
  print(paste0("sshapefval: ", sshapefval))
  print(paste("crs for raster file - ", rasterfilename, "is: ", crs(fname, describe=T)))
  
  
  # Check if shape file existed in either dhs or gadm folder
  
  
  #if (file.exists(countryshpfilenamegadm)){
  if (sum(str_detect(sshapefval, 'GADM')) > 0 ) {
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamegadm)
    
    #countryshaperaster <- st_read("/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/dissertationdatasf/DHS/BEN/BEN.shp") %>%
    #  st_zm()
    
    
    
    ###################
    # Zonal statistics - preparation
    countryshpfile.shpformat$id <- 1:max(nrow(countryshpfile.shpformat))
    
    #crsval <- crs(countryshpfile.shpformat, proj = TRUE)
    #reprojfname <- project(fname, crsval)
    
    print(fname)
    print(countryshpfile.shpformat)
    
    exactextractr::exact_extract(fname, countryshpfile.shpformat, function(values, coverage_fraction){
      
      # Calculate total area in pixels (including cells with NaN or NA values)
      #total_area_in_px <- sum(coverage_fraction, na.rm = TRUE)
      total_area_in_px <- sum(coverage_fraction > 0 )
      print(paste("lcu total area in px:", total_area_in_px))
      
      # Filter out NA values and zero coverage fractions
      valid_values <- values[!is.na(values) & coverage_fraction > 0]
      valid_coverage <- coverage_fraction[!is.na(values) & coverage_fraction > 0]
      
      if (length(valid_values) == 0 || length(valid_coverage) == 0) {
        # If no valid values, return NA for mean and other metrics
        lcu_mixedforest_perc <- NA
        lcu_closedshrublands_perc <- NA
        lcu_openshrublands_perc <- NA
        lcu_woodysavannas_perc <- NA
        lcu_savannas_perc <- NA
        lcu_grasslands_perc <- NA
        lcu_croplands_perc <- NA
        lcu_urbanbuiltup_perc <- NA
        lcu_croplandnatveg_perc <- NA
        lcu_barren_perc <- NA
      } else {
        
        # LCU Classification calculation - Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%
        pixel_count_mixedforest = sum(valid_coverage[values == 5], na.rm=TRUE)
        
        # Percentage of pixel covered by Mixed Forests
        perc_spread_mixedforest = (pixel_count_mixedforest / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover.
        pixel_count_closedshrublands = sum(valid_coverage[values == 6], na.rm=TRUE)
        
        # Percentage of pixel covered by Closed Shrublands
        perc_spread_closedshrublands = (pixel_count_closedshrublands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover.
        pixel_count_openshrublands = sum(valid_coverage[values == 7], na.rm=TRUE)
        
        # Percentage of pixel covered by Open Shrublands
        perc_spread_openshrublands = (pixel_count_openshrublands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Woody Savannas: tree cover 30-60% (canopy >2m).
        pixel_count_woodysavannas = sum(valid_coverage[values == 8], na.rm=TRUE)
        
        # Percentage of pixel covered by Woody Savannas
        perc_spread_woodysavannas = (pixel_count_woodysavannas / total_area_in_px) * 100
        
        
        # LCU Classification calculation - 	Savannas: tree cover 10-30% (canopy >2m).
        pixel_count_savannas = sum(valid_coverage[values == 9], na.rm=TRUE)
        
        # Percentage of pixel covered by Savannas: tree cover
        perc_spread_savannas = (pixel_count_savannas / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Grasslands: dominated by herbaceous annuals (<2m).
        pixel_count_grasslands = sum(valid_coverage[values == 10], na.rm=TRUE)
        
        # Percentage of pixel covered by Grasslands
        perc_spread_grasslands = (pixel_count_grasslands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Croplands: at least 60% of area is cultivated cropland.
        pixel_count_croplands = sum(valid_coverage[values == 12], na.rm=TRUE)
        
        # Percentage of pixel covered by Croplands
        perc_spread_croplands = (pixel_count_croplands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Urban and Built-up Lands: at least 30% impervious surface area including building materials, asphalt and vehicles.
        pixel_count_urbanbuiltup = sum(valid_coverage[values == 13], na.rm=TRUE)
        
        # Percentage of pixel covered by Urban and Built-up Lands
        perc_spread_urbanbuiltup = (pixel_count_urbanbuiltup / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Cropland/Natural Vegetation Mosaics: mosaics of small-scale cultivation 40-60% with natural tree, shrub, or herbaceous vegetation.
        pixel_count_croplandnatveg = sum(valid_coverage[values == 14], na.rm=TRUE)
        
        # Percentage of pixel covered by light factoring total pixel per region
        perc_spread_croplandnatveg = (pixel_count_croplandnatveg / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Cropland/Natural Vegetation Mosaics: mosaics of small-scale cultivation 40-60% with natural tree, shrub, or herbaceous vegetation.
        pixel_count_barren = sum(valid_coverage[values == 16], na.rm=TRUE)
        
        # Percentage of pixel covered by light factoring total pixel per region
        perc_spread_barren = (pixel_count_barren / total_area_in_px) * 100
        
      }
      
      #Extract the date component from the raster layer name
      daty <- stringr::str_sub(rasterfilename, -14, -5)
      
      data.frame(
        collectiondate = daty,
        countrycode = countrycodeonly,
        #total_lcu_area_px = total_area_in_px,
        lcu_mixedforest_perc = perc_spread_mixedforest,
        lcu_closedshrublands_perc = perc_spread_closedshrublands,
        lcu_openshrublands_perc = perc_spread_openshrublands,
        lcu_woodysavannas_perc = perc_spread_woodysavannas,
        lcu_savannas_perc = perc_spread_savannas,
        lcu_grasslands_perc = perc_spread_grasslands,
        lcu_croplands_perc = perc_spread_croplands,
        lcu_urbanbuiltup_perc = perc_spread_urbanbuiltup,
        lcu_croplandnatveg_perc = perc_spread_croplandnatveg,
        lcu_barren_perc = perc_spread_barren
        
      )
    },
    append_cols = "id",
    full_colnames = TRUE,
    summarize_df = FALSE,
    stack_apply = FALSE
    ) %>% dplyr::inner_join(countryshpfile.shpformat, by = "id") %>% dplyr::select(!starts_with(c("GID", "VARNAME_", "NL_NAME", "ENGTYPE_", "CC_","HASC_" ))) %>%
      rename("region" = "NAME_1", "country" = "NAME_0", "adminarea" = contains("TYPE_")) 
    # %>% rename_if("NAME_2", region)
    
    ###################
    
  } else {
    
    intermshapefileyear = unlist(strsplit(sshapefval, " "))
    shapefileyear = as.vector(intermshapefileyear)[1]
    countryshpfilenamedhs <- paste0(shpdirdhs,"/", countrycodeonly, "/", shapefileyear, "/", countrycodeonly, ".shp")
    print(paste0("Path to DHS shape file : ", countryshpfilenamedhs))
    
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamedhs)
    print(paste("countryshpfilenamedhs: ", countryshpfilenamedhs))
    
    #############################
    # Zonal statistics - preparation
    countryshpfile.shpformat$id <- 1:max(nrow(countryshpfile.shpformat))
    
    #crsval <- crs(countryshpfile.shpformat, proj = TRUE)
    #reprojfname <- project(fname, crsval)
    
    print(fname)
    print(countryshpfile.shpformat)
    
    exactextractr::exact_extract(fname, countryshpfile.shpformat, function(values, coverage_fraction){
      
      # Calculate total area in pixels (including cells with NaN or NA values)
      #total_area_in_px <- sum(coverage_fraction, na.rm = TRUE)
      total_area_in_px <- sum(coverage_fraction > 0)
      print(paste("lcu total area in px:", total_area_in_px))
      
      # Filter out NA values and zero coverage fractions
      valid_values <- values[!is.na(values) & coverage_fraction > 0]
      valid_coverage <- coverage_fraction[!is.na(values) & coverage_fraction > 0]
      
      if (length(valid_values) == 0 || length(valid_coverage) == 0) {
        # If no valid values, return NA for mean and other metrics
        lcu_mixedforest_perc <- NA
        lcu_closedshrublands_perc <- NA
        lcu_openshrublands_perc <- NA
        lcu_woodysavannas_perc <- NA
        lcu_savannas_perc <- NA
        lcu_grasslands_perc <- NA
        lcu_croplands_perc <- NA
        lcu_urbanbuiltup_perc <- NA
        lcu_croplandnatveg_perc <- NA
        lcu_barren_perc <- NA
      } else {
        
        # LCU Classification calculation - Mixed Forests: dominated by neither deciduous nor evergreen (40-60% of each) tree type (canopy >2m). Tree cover >60%
        pixel_count_mixedforest = sum(valid_coverage[values == 5], na.rm=TRUE)
        
        # Percentage of pixel covered by Mixed Forests
        perc_spread_mixedforest = (pixel_count_mixedforest / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Closed Shrublands: dominated by woody perennials (1-2m height) >60% cover.
        pixel_count_closedshrublands = sum(valid_coverage[values == 6], na.rm=TRUE)
        
        # Percentage of pixel covered by Closed Shrublands
        perc_spread_closedshrublands = (pixel_count_closedshrublands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Open Shrublands: dominated by woody perennials (1-2m height) 10-60% cover.
        pixel_count_openshrublands = sum(valid_coverage[values == 7], na.rm=TRUE)
        
        # Percentage of pixel covered by Open Shrublands
        perc_spread_openshrublands = (pixel_count_openshrublands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Woody Savannas: tree cover 30-60% (canopy >2m).
        pixel_count_woodysavannas = sum(valid_coverage[values == 8], na.rm=TRUE)
        
        # Percentage of pixel covered by Woody Savannas
        perc_spread_woodysavannas = (pixel_count_woodysavannas / total_area_in_px) * 100
        
        
        # LCU Classification calculation - 	Savannas: tree cover 10-30% (canopy >2m).
        pixel_count_savannas = sum(valid_coverage[values == 9], na.rm=TRUE)
        
        # Percentage of pixel covered by Savannas: tree cover
        perc_spread_savannas = (pixel_count_savannas / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Grasslands: dominated by herbaceous annuals (<2m).
        pixel_count_grasslands = sum(valid_coverage[values == 10], na.rm=TRUE)
        
        # Percentage of pixel covered by Grasslands
        perc_spread_grasslands = (pixel_count_grasslands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Croplands: at least 60% of area is cultivated cropland.
        pixel_count_croplands = sum(valid_coverage[values == 12], na.rm=TRUE)
        
        # Percentage of pixel covered by Croplands
        perc_spread_croplands = (pixel_count_croplands / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Urban and Built-up Lands: at least 30% impervious surface area including building materials, asphalt and vehicles.
        pixel_count_urbanbuiltup = sum(valid_coverage[values == 13], na.rm=TRUE)
        
        # Percentage of pixel covered by Urban and Built-up Lands
        perc_spread_urbanbuiltup = (pixel_count_urbanbuiltup / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Cropland/Natural Vegetation Mosaics: mosaics of small-scale cultivation 40-60% with natural tree, shrub, or herbaceous vegetation.
        pixel_count_croplandnatveg = sum(valid_coverage[values == 14], na.rm=TRUE)
        
        # Percentage of pixel covered by light factoring total pixel per region
        perc_spread_croplandnatveg = (pixel_count_croplandnatveg / total_area_in_px) * 100
        
        
        # LCU Classification calculation - Cropland/Natural Vegetation Mosaics: mosaics of small-scale cultivation 40-60% with natural tree, shrub, or herbaceous vegetation.
        pixel_count_barren = sum(valid_coverage[values == 16], na.rm=TRUE)
        
        # Percentage of pixel covered by light factoring total pixel per region
        perc_spread_barren = (pixel_count_barren / total_area_in_px) * 100
        
      }
      
      #Extract the date component from the raster layer name
      daty <- stringr::str_sub(rasterfilename, -14, -5)
      
      data.frame(
        collectiondate = daty,
        countrycode = countrycodeonly,
        #total_lcu_area_px = total_area_in_px,
        lcu_mixedforest_perc = perc_spread_mixedforest,
        lcu_closedshrublands_perc = perc_spread_closedshrublands,
        lcu_openshrublands_perc = perc_spread_openshrublands,
        lcu_woodysavannas_perc = perc_spread_woodysavannas,
        lcu_savannas_perc = perc_spread_savannas,
        lcu_grasslands_perc = perc_spread_grasslands,
        lcu_croplands_perc = perc_spread_croplands,
        lcu_urbanbuiltup_perc = perc_spread_urbanbuiltup,
        lcu_croplandnatveg_perc = perc_spread_croplandnatveg,
        lcu_barren_perc = perc_spread_barren
      )
    },
    append_cols = "id",
    full_colnames = TRUE,
    summarize_df = FALSE,
    stack_apply = FALSE
    ) %>% dplyr::inner_join(countryshpfile.shpformat, by = "id")  %>% 
      dplyr::select(-c("ISO", "FIPS", "DHSCC", "SVYTYPE", "SVYYEAR", "CNTRYNAMEF", "CNTRYNAMES", 
                       "DHSREGSP", "SVYID", "REG_ID", "Svy_Map", "MULTLEVEL", "LEVELRNK", "REGVAR", "REGCODE", "OTHREGVAR", "OTHREGCO", "OTHREGNA", "LEVELCO", 
                       "REPALLIND", "REGNOTES", "DHSREGEN", "DHSREGFR", "SVYNOTES")) %>% 
      rename("region" = "REGNAME", "country" = "CNTRYNAMEE", "adminarea" = "LEVELNA")
    
    #############################
    
  } # End of if statement
} # generated dataframe func ending
#=======================================



# Extract OpenStreetMap infrastructure for each administrative region using
# historical Overpass queries associated with the corresponding observation year.
# Education, medical facilities, major roads and residential building categories
# are counted and later combined with the GEE-derived predictors.
##############################Open Street Map Data###########################

osm.data.extract <- function(rasterfilename,timeout, df, memorytoalloc = 2684354560){
  
  
  sf_use_s2(FALSE) # fixes Loop 0 is not valid: Edge 1 crosses edge 3 error
  
  # setting up all required variable parameters
  timeoutval = timeout
  #fname <- rast(rasterfilename)
  fnamewithoutext <- file_path_sans_ext(rasterfilename)
  countrycodeonly <- str_extract(fnamewithoutext, '[^_]+')
  countryshpfilenamegadm <- paste0(shpdirgadm,"/", countrycodeonly, "/", countrycodeonly, ".shp")
  
# Identify the administrative-boundary source and construct the historical OSM
# query date used for the current country-year extraction.
  # Verify source of shape file
  rasterfiledateval <- stringr::str_sub(fnamewithoutext, -10)
  rasterfiledate <- ymd(rasterfiledateval)
  sshapef <- df %>% filter(countrycode == countrycodeonly, year == year(rasterfiledate)) %>% dplyr::select(shapefile_source) %>% 
    unlist()
  sshapefval <- sshapef[4]
  
  
  yearaftertoextract = year(rasterfiledate) + 1
  dateformat = paste0(yearaftertoextract, "-01-01 00:00:00")
  dateval = lubridate::format_ISO8601(as.POSIXct(dateformat, tz = "UTC"), usetz = "Z")
  
  
  print(paste0("countryshpfilenamegadm: ", countryshpfilenamegadm))
  print(paste0("fnamewithoutext: ", fnamewithoutext))
  print(paste0("rasterfiledate: ", rasterfiledate))
  print(paste0("class of rasterfiledate: ", class(rasterfiledate)))
  print(paste0("sshapef: ", sshapef))
  print(paste0("sshapefval: ", sshapefval))
  print(paste0("dateval: ", dateval))
  
  # set overpas url
  # default is "https://overpass.kumi.systems/api/interpreter"
  osmdata::set_overpass_url("https://overpass-api.de/api/interpreter")
  
  
# Read the administrative boundaries used for the OSM extraction. Sierra Leone
# is handled at NAME_2 in this workflow, while other GADM observations use NAME_1;
# DHS observations use the corresponding DHS regional field.
  # Check if shape file existed in either dhs or gadm folder
  
  if (sum(str_detect(sshapefval, 'GADM')) > 0 & countrycodeonly != "SLE") {
    
    columnname <- c("name_1")
    print(paste("columnname is:", columnname))
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamegadm) %>% 
      janitor::clean_names() %>% dplyr::select(all_of(columnname))
    
  } else if (sum(str_detect(sshapefval, 'GADM')) > 0 & countrycodeonly == "SLE"){
    # set the variables based on countrycodeonly variable being SLE
    columnname <- c("name_2")
    print(paste("columnname is:", columnname))
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamegadm) %>% 
      janitor::clean_names() %>% dplyr::select(all_of(columnname))
  } else {
    intermshapefileyear = unlist(strsplit(sshapefval, " "))
    shapefileyear = as.vector(intermshapefileyear)[1]
    countryshpfilenamedhs <- paste0(shpdirdhs,"/", countrycodeonly, "/", shapefileyear, "/", countrycodeonly, ".shp")
    print(paste0("Path to DHS shape file : ", countryshpfilenamedhs))
    columnname <- c("dhsregen")
    #columnname <- c("regname")
    print(paste("columnname is:", columnname))
    countryshpfile.shpformat <- sf::st_read(countryshpfilenamedhs) %>%
      janitor::clean_names() %>% dplyr::select(all_of(columnname))
    
  }
  
  
  # declare an empty dataframe
  
  resultstable <- data.frame(ccode = character(),
                             region = character(),
                             buildschools = integer(),
                             buildhospitals = integer(),
                             buildroads = integer(),
                             buildaccomodation = integer(),
                             buildhouse = integer()
  )
  
  #towns <- data.frame(countryshpfile.shpformat) %>% dplyr::select(.data[[columnname]]) %>% unlist()
  towns <- data.frame(countryshpfile.shpformat) %>% dplyr::select(all_of(columnname)) %>% unlist()
  print(paste0("towns:", towns))
  #quit(status = 1)
  
  
  if (file.exists(paste0(osmdata_dir, "/", countrycodeonly, "_osmdata_", rasterfiledate, ".csv"))){
    
    print(paste0("csv file already exist for:", countrycodeonly, "_osmdata_", rasterfiledate, ".csv"))
    
    # Write state to log file
    write(paste0(Sys.time(), ": Info: ", "csv file already exist for:", countrycodeonly, "_osmdata_", rasterfiledate, ".csv. Skip reprocessing"), file=logfile, append = TRUE)
  } else {
    
    for (town in towns){
# Process each administrative region separately. The region bounding box is used
# for the Overpass request and the regional polygon is subsequently used to
# retain only returned features that intersect the area of interest.
      print("------------")
      print(paste0("The aoi being process is:", town))
      print(paste0("fyear:", dateval))
      
      # Get the bounding box
      townbbox <- countryshpfile.shpformat %>%  filter(get(columnname) == town) %>% sf::st_bbox()
      print(paste0("townbbox:", townbbox))
      
      # Get the geometry object for the specific area of interest
      polytownboundary <- countryshpfile.shpformat %>%  filter(get(columnname) == town) %>% sf::st_geometry()
      #print(paste0("polytownboundary:", polytownboundary))
      
      # New section- get bounding box
      #bb_poly <- osmdata::getbb(townbbox, format_out = "polygon")
      #print(paste0("bb_poly:", bb_poly))
      
      # Temporary osm file name read from disk
      temposmfile <- paste0(osmtmp_dir, "/", countrycodeonly,"_", rasterfiledate, "_eduosmdata_", town, ".osm")
      
      if (file.exists(temposmfile)){
        # osm object - education
        # reading from disk
        print(paste("starting reading of osm object from disk: Education"))
        osm.edu.obj <- osmdata_sf(doc=temposmfile)
# Request education-related OSM features for the current region and historical
# query date, then remove duplicate OSM objects before spatial counting.
      } else{
        # osm object - education
        print(paste("starting creation of osm object: Education"))
        # Write data to disk and read from the disk for further processing
        osm.edu.obj <- opq(bbox = c(townbbox["xmin"], townbbox["ymin"], townbbox["xmax"], townbbox["ymax"]), datetime = dateval, timeout = timeoutval) %>%
          add_osm_feature(key = 'building', value = c('kindergarten', 'school','college', 'university', 'research_institute'), match_case = FALSE) %>%
          osmdata_xml(temposmfile)
        
        print(paste0("Extraction Date:", osm.edu.obj$timestamp))
        osm.edu.obj <- osmdata_sf(doc=temposmfile)
        
      }
      
      print(paste0("Extraction Date:", osm.edu.obj$timestamp))
      osm.edu.obj.unique <- unique_osmdata(osm.edu.obj)
      #print(paste0("unique.osm.edu.obj:", osm.edu.obj.unique))
      
# Count education features across the available OSM geometry types after
# transforming them to the regional CRS and filtering them to the region boundary.
      
      # filter for objects within the area of interest
      ptsfobject <- osm.edu.obj.unique$osm_points
      #print(paste("points sf objects for education:", ptsfobject))
      if (!is.null(ptsfobject)){
        print(paste("points crs value for ", town, ": ", sf::st_crs(polytownboundary)$epsg))
        print(paste("points crs value for bbounding box:", sf::st_crs(ptsfobject)$epsg))
        #sf::st_crs(ptsfobject) <- sf::st_crs(polytownboundary)$epsg
        ptsfobjecttransform <- sf::st_transform(ptsfobject, sf::st_crs(polytownboundary))
        print(paste("Logical Matrix:", ptsfobjecttransform))
        print(paste("Logical Matrix dimension:", dim(ptsfobjecttransform)))
        tryCatch({
          ptsfobjecttransformfiltered <- sf::st_filter(ptsfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          # default to method of get road count without intersections
          print(paste0("Error related to st_filter on points:", e))
          ptcount = ptsfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(ptsfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            ptcount = ptsfobjecttransformfiltered %>% nrow()
          } else {
            ptcount = 0
          }
        }
        )} else {
          ptcount = 0
        }
      
      
      linesfobject <- osm.edu.obj.unique$osm_lines
      if (!is.null(linesfobject)){
        #print(paste("line sf objects:", linesfobject))
        print(paste("line - crs value for bbounding box:", sf::st_crs(linesfobject)$epsg))
        #st_crs(linesfobject) <- st_crs(polytownboundary)$epsg
        linesfobjecttransform <- st_transform(linesfobject, st_crs(polytownboundary))
        tryCatch({
          linesfobjecttransformfiltered <- sf::st_filter(linesfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on linesf:", e))
          linecount = linesfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(linesfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            linecount = linesfobjecttransformfiltered %>% nrow()
          } else{
            linecount = 0
          }
        }
        )} else {
          linecount = 0
        }
      
      
      polygonsfobject <- osm.edu.obj.unique$osm_polygons
      if (!is.null(polygonsfobject)){
        #print(paste("polygon sf objects:", polygonsfobject))
        print(paste("polygon - crs value for bbounding box:", sf::st_crs(polygonsfobject)$epsg))
        #st_crs(polygonsfobject) <- st_crs(polytownboundary)$epsg
        polygonsfobjecttransform <- st_transform(polygonsfobject, st_crs(polytownboundary))
        
        tryCatch({
          polygonsfobjecttransformfiltered <- sf::st_filter(polygonsfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on polygons on education:", e))
          #polygonscount = polygonsfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(polygonsfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            polygonscount = polygonsfobjecttransformfiltered %>% nrow()
          } else{
            polygonscount = 0
          }
        }
        )} else {
          polygonscount = 0
        }
      
      
      mlinessfobject <- osm.edu.obj.unique$osm_multilines
      if (!is.null(mlinessfobject)){
        #print(paste("multilines sf objects:", mlinessfobject))
        print(paste("multilines - crs value for bbounding box:", sf::st_crs(mlinessfobject)$epsg))
        #st_crs(mlinessfobject) <- st_crs(polytownboundary)$epsg
        mlinessfobjecttransform <- st_transform(mlinessfobject, st_crs(polytownboundary))
        tryCatch({
          mlinessfobjecttransformfiltered <- sf::st_filter(mlinessfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on multiline object on education:", e))
          #polygonscount = polygonsfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(mlinessfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            mlinescount = mlinessfobjecttransformfiltered %>% nrow()
          } else {
            mlinescount = 0
          }
        }
        )} else {
          mlinescount = 0
        }
      
      
      mpolyobject <- osm.edu.obj.unique$osm_multipolygons
      if (!is.null(mpolyobject)){
        #print(paste("multipolygon sf objects:", mpolyobject))
        print(paste("multipolygon - crs value for bbounding box:", sf::st_crs(mpolyobject)$epsg))
        #st_crs(mpolyobject) <- st_crs(polytownboundary)$epsg
        mpolyobjecttransform <- st_transform(mpolyobject, st_crs(polytownboundary))
        mpolyobjecttransformfiltered <- NULL
        tryCatch({
          mpolyobjecttransformfiltered <- sf::st_filter(mpolyobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on multipolygon object on education:", e))
        }, finally = {
          if (!is.null(mpolyobjecttransformfiltered)){
            mpolycount = mpolyobjecttransformfiltered %>% nrow()
          } else {
            mpolycount = 0
          }
        })
      } else{
        mpolycount = 0
      }
      
      
      
      totedc = sum(ptcount, linecount, polygonscount, mlinescount, mpolycount)
      
      # Write state to log file
      write(paste0(Sys.time(), ": Info: ", " OSM data for education obtained for : ", countrycodeonly, ":", town, ":", rasterfiledate, " - ", totedc), file=logfile, append = TRUE)
      
      
      # osm object - Medical Facilities
      # Temporary osm file name read from disk
      temposmfile <- paste0(osmtmp_dir, "/", countrycodeonly,"_", rasterfiledate, "_medosmdata_", town, ".osm")
      
      if (file.exists(temposmfile)){
        # osm object - education
# Request medical-facility OSM features and count the returned geometries that
# intersect the current administrative region.
        # reading from disk
        print(paste("starting reading of osm object from disk: Medical"))
        osm.med.obj <- osmdata_sf(doc=temposmfile)
      } else {
        print(paste("starting creation of osm object: Medical"))
        osm.med.obj <- townbbox %>% opq(.,datetime = dateval, timeout = timeoutval) %>%
          add_osm_feature(key = 'amenity', value = c('hospital', 'clinic','dentist', 'doctors', 'nursing_home',' pharmacy', 'social_facility'), match_case = FALSE) %>%
          osmdata_xml(temposmfile)
        
        osm.med.obj <- osmdata_sf(doc=temposmfile)
      }
      
      print(paste0("Extraction Date:", osm.med.obj$timestamp))
      osm.med.obj.unique <- unique_osmdata(osm.med.obj)
      
      # filter for objects within the area of interest
      ptsfobject <- osm.med.obj.unique$osm_points
      if (!is.null(ptsfobject)){
        #print(paste("points sf objects for medical:", ptsfobject))
        print(paste("crs value for bb ", town, ": ", sf::st_crs(ptsfobject)$epsg))
        print(paste("crs value for bbounding box:", sf::st_crs(ptsfobject)$epsg))
        #st_crs(ptsfobject) <- st_crs(polytownboundary)$epsg
        ptsfobjecttransform <- st_transform(ptsfobject, st_crs(polytownboundary))
        ptsfobjecttransformfiltered <- sf::st_filter(ptsfobjecttransform, polytownboundary, .predicate = st_intersects)
        ptcount = ptsfobjecttransformfiltered %>% nrow()
      } else{
        ptcount = 0
        
      }
      
      linesfobject <- osm.med.obj.unique$osm_lines
      if (!is.null(linesfobject)){
        #print(paste("line sf objects:", linesfobject))
        print(paste("line - crs value for bb ", town, ": ", sf::st_crs(linesfobject)$epsg))
        #st_crs(linesfobject) <- st_crs(polytownboundary)$epsg
        linesfobjecttransform <- st_transform(linesfobject, st_crs(polytownboundary))
        linesfobjecttransformfiltered <- sf::st_filter(linesfobjecttransform, polytownboundary, .predicate = st_intersects)
        linecount = linesfobjecttransformfiltered %>% nrow()
      } else {
        linecount = 0
      }
      
      
      polygonsfobject <- osm.med.obj.unique$osm_polygons
      if (!is.null(polygonsfobject)){
        #print(paste("polygon sf objects:", polygonsfobject))
        print(paste("polygon - crs value for bb ", town, ": ", sf::st_crs(polygonsfobject)$epsg))
        #st_crs(polygonsfobject) <- st_crs(polytownboundary)$epsg
        polygonsfobjecttransform <- st_transform(polygonsfobject, st_crs(polytownboundary))
        polygonsfobjecttransformfiltered <- sf::st_filter(polygonsfobjecttransform, polytownboundary, .predicate = st_intersects)
        if (!is.null(polygonsfobjecttransformfiltered)){
          #print(paste0("test:", polygonsfobjecttransformfiltered))
          polygonscount = polygonsfobjecttransformfiltered %>% nrow()
        } else{
          polygonscount = 0
        }
      } else {
        polygonscount = 0
      }
      
      mlinessfobject <- osm.med.obj.unique$osm_multilines
      if (!is.null(mlinessfobject)){
        #print(paste("multiline sf objects:", mlinessfobject))
        print(paste("multiline - crs value for bb ", town, ": ", sf::st_crs(mlinessfobject)$epsg))
        #st_crs(mlinessfobject) <- st_crs(polytownboundary)$epsg
        mlinessfobjecttransform <- st_transform(mlinessfobject, st_crs(polytownboundary))
        mlinessfobjecttransformfiltered <- sf::st_filter(mlinessfobjecttransform, polytownboundary, .predicate = st_intersects)
        mlinescount = mlinessfobjecttransformfiltered %>% nrow()
      } else {
        mlinescount = 0
      }
      
      
      mpolyobject <- osm.med.obj.unique$osm_multipolygons
      if (!is.null(mpolyobject)){
        #print(paste("multipolygon sf objects:", mpolyobject))
        print(paste("line - crs value for bb ", town, ": ", sf::st_crs(linesfobject)$epsg))
        #st_crs(mpolyobject) <- st_crs(polytownboundary)$epsg
        mpolyobjecttransform <- st_transform(mpolyobject, st_crs(polytownboundary))
        mpolyobjecttransformfiltered <- sf::st_filter(mpolyobjecttransform, polytownboundary, .predicate = st_intersects)
        mpolycount = mpolyobjecttransformfiltered %>% nrow()
      } else {
        mpolycount = 0
      }
      
      tothospital = sum(ptcount, linecount, polygonscount, mlinescount, mpolycount)
      
      # Write state to log file
      write(paste0(Sys.time(), ": Info: ", " OSM data for hospital obtained for : ", countrycodeonly, ":", town, ":", rasterfiledate,  " - ", tothospital), file=logfile, append = TRUE)
      
# Request the selected major-road classes and count OSM geometries that intersect
# the current administrative region.
      # osm object - road
      temposmfile <- paste0(osmtmp_dir, "/", countrycodeonly,"_", rasterfiledate, "_roadsosmdata_", town, ".osm")
      
      if (file.exists(temposmfile)){
        # osm object - education
        # reading from disk
        print(paste("starting reading of osm object from disk: Roads"))
        osm.road.obj <- osmdata_sf(doc=temposmfile)
      } else{
        print(paste("starting creation of osm object: Road"))
        osm.road.obj <- townbbox %>% opq(., datetime = dateval, timeout = timeoutval) %>%
          add_osm_feature(key = "highway", value = c("motorway", "trunk", "primary", "secondary"), match_case = FALSE) %>%
          osmdata_xml(temposmfile)
        osm.road.obj <- osmdata_sf(doc=temposmfile)
      }
      
      
      print(paste0("Extraction Date:", osm.med.obj$timestamp))
      osm.road.obj.unique <- unique_osmdata(osm.road.obj)
      
      # filter for objects within the area of interest
      ptsfobject <- osm.road.obj.unique$osm_points
      #if (!is.null(ptsfobject) | ((ptsfobject %>% nrow()) >= 0) ){
      if (!is.null(ptsfobject)){
        #print(paste("points sf objects for roads:", ptsfobject))
        print(paste("points - crs value for bb ", town, ": ", sf::st_crs(ptsfobject)$epsg))
        #st_crs(ptsfobject) <- st_crs(polytownboundary)$epsg
        ptsfobjecttransform <- st_transform(ptsfobject, st_crs(polytownboundary))
        tryCatch({
          ptsfobjecttransformfiltered <- sf::st_filter(ptsfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on points:", e))
          ptcount = ptsfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(ptsfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            ptcount = ptsfobjecttransformfiltered %>% nrow()
          } else {
            ptcount = 0
          }
        }
        )} else {
          ptcount = 0
        }
      
      linesfobject <- osm.road.obj.unique$osm_lines
      if (!is.null(linesfobject)){
        #print(paste("line sf objects:", linesfobject))
        print(paste("line - crs value for bb ", town, ": ", sf::st_crs(linesfobject)$epsg))
        #st_crs(linesfobject) <- st_crs(polytownboundary)$epsg
        linesfobjecttransform <- st_transform(linesfobject, st_crs(polytownboundary))
        tryCatch({
          linesfobjecttransformfiltered <- sf::st_filter(linesfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on linesf:", e))
          linecount = linesfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(linesfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            linecount = linesfobjecttransformfiltered %>% nrow()
          } else{
            linecount = 0
          }
        }
        )} else {
          linecount = 0
        }
      
      polygonsfobject <- osm.road.obj.unique$osm_polygons
      
      if (!is.null(polygonsfobject)){
        #print(paste("polygon sf objects:", polygonsfobject))
        print(paste("polygon - crs value for bb ", town, ": ", sf::st_crs(polygonsfobject)$epsg))
        #st_crs(polygonsfobject) <- st_crs(polytownboundary)$epsg
        polygonsfobjecttransform <- st_transform(polygonsfobject, st_crs(polytownboundary))
        
        tryCatch({
          polygonsfobjecttransformfiltered <- sf::st_filter(polygonsfobjecttransform, polytownboundary, .predicate = st_intersects)
        }, error = function(e){
          print(paste0("Error related to st_filter on polygons:", e))
          polygonscount = polygonsfobject %>% nrow() # default to method of get road count without intersections
        }, finally = {
          if (!is.null(polygonsfobjecttransformfiltered)){
            #print(paste0("testtwo:", polygonsfobjecttransformfiltered))
            polygonscount = polygonsfobjecttransformfiltered %>% nrow()
          } else{
            polygonscount = 0
          }
        }
        )} else {
          polygonscount = 0
        }
      
      
      mlinessfobject <- osm.road.obj.unique$osm_multilines
      if (!is.null(mlinessfobject)){
        #print(paste("multiline sf objects:", mlinessfobject))
        print(paste("multilines - crs value for bb ", town, ": ", sf::st_crs(mlinessfobject)$epsg))
        #st_crs(mlinessfobject) <- st_crs(polytownboundary)$epsg
        mlinessfobjecttransform <- st_transform(mlinessfobject, st_crs(polytownboundary))
        mlinessfobjecttransformfiltered <- sf::st_filter(mlinessfobjecttransform, polytownboundary, .predicate = st_intersects)
        mlinescount = mlinessfobjecttransformfiltered %>% nrow()
      } else {
        mlinescount = 0
      }
      
      
      mpolyobject <- osm.road.obj.unique$osm_multipolygons
      if (!is.null(mpolyobject)){
        #print(paste("multipoly sf objects:", mpolyobject))
        print(paste("multipoly - crs value for ", town, ": ", sf::st_crs(mpolyobject)$epsg))
        #st_crs(mpolyobject) <- st_crs(polytownboundary)$epsg
        mpolyobjecttransform <- st_transform(mpolyobject, st_crs(polytownboundary))
        mpolyobjecttransformfiltered <- sf::st_filter(mpolyobjecttransform, polytownboundary, .predicate = st_intersects)
        mpolycount = mpolyobjecttransformfiltered %>% nrow()
      } else {
        mpolycount = 0
      }
      
      totroad = sum(ptcount, linecount, polygonscount, mlinescount, mpolycount)
      
      # Write state to log file
      write(paste0(Sys.time(), ": Info: ", " OSM data for road obtained for : ", countrycodeonly, ":", town, ":", rasterfiledate, " - ", totroad), file=logfile, append = TRUE)
      
# Request the selected accommodation/residential building categories and count
# the features that intersect the current administrative region.
      # osm object - accomodation Facilities
      temposmfile <- paste0(osmtmp_dir, "/", countrycodeonly, "_", rasterfiledate, "_accosmdata_", town, ".osm")
      
      if (file.exists(temposmfile)){
        # osm object - education
        # reading from disk
        print(paste("starting reading of osm object from disk: acoomodation"))
        osm.accomodation.obj <- osmdata_sf(doc=temposmfile)
      } else{
        print(paste("starting creation of osm object: Accomodation"))
        osm.accomodation.obj <- townbbox %>% opq(., datetime = dateval, timeout = timeoutval) %>%
          add_osm_feature(key = 'building', value = c("apartments", "terrace","detached", "semidetached_house", "bungalow"), match_case = FALSE) %>%
          osmdata_xml(temposmfile)
        
        osm.accomodation.obj <- osmdata_sf(doc=temposmfile)
      }
      
      print(paste0("Extraction Date:", osm.accomodation.obj$timestamp))
      osm.accomodation.obj.unique <- unique_osmdata(osm.accomodation.obj)
      
      # filter for objects within the area of interest
      ptsfobject <- osm.accomodation.obj.unique$osm_points
      if (!is.null(ptsfobject)){
        #print(paste("points sf objects for accomodation:", ptsfobject))
        print(paste("points - crs value for ", town, ": ", sf::st_crs(ptsfobject)$epsg))
        #st_crs(ptsfobject) <- st_crs(polytownboundary)$epsg
        ptsfobjecttransform <- st_transform(ptsfobject, st_crs(polytownboundary))
        ptsfobjecttransformfiltered <- sf::st_filter(ptsfobjecttransform, polytownboundary, .predicate = st_intersects)
        ptcount = ptsfobjecttransformfiltered %>% nrow()
      } else{
        ptcount = 0
        
      }
      
      linesfobject <- osm.accomodation.obj.unique$osm_lines
      if (!is.null(linesfobject)){
        #print(paste("line sf objects:", linesfobject))
        print(paste("multipolygon - crs value for bb ", town, ": ", sf::st_crs(linesfobject)$epsg))
        #st_crs(linesfobject) <- st_crs(polytownboundary)$epsg
        linesfobjecttransform <- st_transform(linesfobject, st_crs(polytownboundary))
        linesfobjecttransformfiltered <- sf::st_filter(linesfobjecttransform, polytownboundary, .predicate = st_intersects)
        linecount = linesfobjecttransformfiltered %>% nrow()
      } else {
        linecount = 0
      }
      
      
      polygonsfobject <- osm.accomodation.obj.unique$osm_polygons
      
      if (!is.null(polygonsfobject)){
        #print(paste("polygon sf objects:", polygonsfobject))
        #st_crs(polygonsfobject) <- st_crs(polytownboundary)$epsg
        print(paste("multipolygon - crs value for bb ", town, ": ", sf::st_crs(polygonsfobject)$epsg))
        polygonsfobjecttransform <- st_transform(polygonsfobject, st_crs(polytownboundary))
        polygonsfobjecttransformfiltered <- sf::st_filter(polygonsfobjecttransform, polytownboundary, .predicate = st_intersects)
        if (!is.null(polygonsfobjecttransformfiltered)){
          #print(paste0("test:", polygonsfobjecttransformfiltered))
          polygonscount = polygonsfobjecttransformfiltered %>% nrow()
        } else{
          polygonscount = 0
        }
      } else {
        polygonscount = 0
      }
      
      mlinessfobject <- osm.accomodation.obj.unique$osm_multilines
      if (!is.null(mlinessfobject)){
        #print(paste("multiline sf objects:", mlinessfobject))
        print(paste("multipolygon - crs value for bb ", town, ": ", sf::st_crs(mlinessfobject)$epsg))
        #st_crs(mlinessfobject) <- st_crs(polytownboundary)$epsg
        mlinessfobjecttransform <- st_transform(mlinessfobject, st_crs(polytownboundary))
        mlinessfobjecttransformfiltered <- sf::st_filter(mlinessfobjecttransform, polytownboundary, .predicate = st_intersects)
        mlinescount = mlinessfobjecttransformfiltered %>% nrow()
      } else {
        mlinescount = 0
      }
      
      
      mpolyobject <- osm.accomodation.obj.unique$osm_multipolygons
      if (!is.null(mpolyobject)){
        #print(paste("multipolygon sf objects:", mpolyobject))
        print(paste("multipolygon - crs value for bb ", town, ": ", sf::st_crs(mpolyobject)$epsg))
        #st_crs(mpolyobject) <- st_crs(polytownboundary)$epsg
        print(paste("multipolygon - crs value for ", town, ": ", sf::st_crs(mpolyobject)$epsg))
        mpolyobjecttransform <- st_transform(mpolyobject, st_crs(polytownboundary))
        mpolyobjecttransformfiltered <- sf::st_filter(mpolyobjecttransform, polytownboundary, .predicate = st_intersects)
        mpolycount = mpolyobjecttransformfiltered %>% nrow()
      } else {
        mpolycount = 0
      }
      
      totaccomodation = sum(ptcount, linecount, polygonscount, mlinescount, mpolycount)
      
      # Write state to log file
      write(paste0(Sys.time(), ": Info: ", " OSM data for accomodation obtained for : ", countrycodeonly, ":", town, ":", rasterfiledate, " - ", totaccomodation), file=logfile, append = TRUE)
      
# Request buildings tagged specifically as houses and combine their count with
# the accommodation-building count later to form the residence predictor.
      # osm object - House Facilities
      temposmfile <- paste0(osmtmp_dir, "/", countrycodeonly,"_", rasterfiledate, "_houseosmdata_", town, ".osm")
      
      if (file.exists(temposmfile)){
        # osm object - education
        # reading from disk
        print(paste("starting reading of osm object from disk: House"))
        osm.house.obj <- osmdata_sf(doc=temposmfile)
      } else{
        print(paste("starting creation of osm object: House"))
        osm.house.obj <- townbbox %>% opq(., datetime = dateval, timeout = timeoutval, memsize = memorytoalloc) %>%
          add_osm_feature(key = 'building', value = "house", match_case = FALSE) %>%
          osmdata_xml(temposmfile)
        
        osm.house.obj <- osmdata_sf(doc=temposmfile)
      }
      
      print(paste0("Extraction Date:", osm.house.obj$timestamp))
      osm.house.obj.unique <- unique_osmdata(osm.house.obj)
      
      # filter for objects within the area of interest
      ptsfobject <- osm.house.obj.unique$osm_points
      if (!is.null(ptsfobject)){
        #print(paste("points sf objects for roads:", ptsfobject))
        print(paste("points - crs value for ", town, ": ", sf::st_crs(ptsfobject)$epsg))
        #st_crs(ptsfobject) <- st_crs(polytownboundary)$epsg
        ptsfobjecttransform <- st_transform(ptsfobject, st_crs(polytownboundary))
        ptsfobjecttransformfiltered <- sf::st_filter(ptsfobjecttransform, polytownboundary, .predicate = st_intersects)
        ptcount = ptsfobjecttransformfiltered %>% nrow()
      } else{
        ptcount = 0
        
      }
      
      linesfobject <- osm.house.obj.unique$osm_lines
      if (!is.null(linesfobject)){
        #print(paste("line sf objects:", linesfobject))
        print(paste("lines - crs value for ", town, ": ", sf::st_crs(linesfobject)$epsg))
        #st_crs(linesfobject) <- st_crs(polytownboundary)$epsg
        linesfobjecttransform <- st_transform(linesfobject, st_crs(polytownboundary))
        linesfobjecttransformfiltered <- sf::st_filter(linesfobjecttransform, polytownboundary, .predicate = st_intersects)
        linecount = linesfobjecttransformfiltered %>% nrow()
      } else {
        linecount = 0
      }
      
      
      polygonsfobject <- osm.house.obj.unique$osm_polygons
      
      if (!is.null(polygonsfobject)){
        #print(paste("polygon sf objects:", polygonsfobject))
        print(paste("polygon - crs value for ", town, ": ", sf::st_crs(polygonsfobject)$epsg))
        #st_crs(polygonsfobject) <- st_crs(polytownboundary)$epsg
        polygonsfobjecttransform <- st_transform(polygonsfobject, st_crs(polytownboundary))
        polygonsfobjecttransformfiltered <- sf::st_filter(polygonsfobjecttransform, polytownboundary, .predicate = st_intersects)
        
        if (!is.null(polygonsfobjecttransformfiltered)){
          #print(paste0("test:", polygonsfobjecttransformfiltered))
          polygonscount = polygonsfobjecttransformfiltered %>% nrow()
        } else{
          polygonscount = 0
        }
        
      } else {
        polygonscount = 0
      }
      
      mlinessfobject <- osm.house.obj.unique$osm_multilines
      if (!is.null(mlinessfobject)){
        #print(paste("multiline sf objects:", mlinessfobject))
        print(paste("multilines - crs value for ", town, ": ", sf::st_crs(mlinessfobject)$epsg))
        #st_crs(mlinessfobject) <- st_crs(polytownboundary)$epsg
        mlinessfobjecttransform <- st_transform(mlinessfobject, st_crs(polytownboundary))
        mlinessfobjecttransformfiltered <- sf::st_filter(mlinessfobjecttransform, polytownboundary, .predicate = st_intersects)
        mlinescount = mlinessfobjecttransformfiltered %>% nrow()
      } else {
        mlinescount = 0
      }
      
      
      mpolyobject <- osm.house.obj.unique$osm_multipolygons
      if (!is.null(mpolyobject)){
        #print(paste("multipolygon sf objects:", mpolyobject))
        print(paste("multipolygon - crs value for ", town, ": ", sf::st_crs(mpolyobject)$epsg))
        #st_crs(mpolyobject) <- st_crs(polytownboundary)$epsg
        mpolyobjecttransform <- st_transform(mpolyobject, st_crs(polytownboundary))
        mpolyobjecttransformfiltered <- sf::st_filter(mpolyobjecttransform, polytownboundary, .predicate = st_intersects)
        mpolycount = mpolyobjecttransformfiltered %>% nrow()
      } else {
        mpolycount = 0
      }
      
      tothouse = sum(ptcount, linecount, polygonscount, mlinescount, mpolycount)
      
      # Write state to log file
      write(paste0(Sys.time(), ": Info: ", " OSM data for house obtained for : ", countrycodeonly, ":", town, ":", rasterfiledate, " - ", tothouse), file=logfile, append = TRUE)
      
# Store the regional OSM counts for schools, medical facilities, roads,
# accommodation buildings and houses before writing the country-year result.
      resulttemp <- data.frame(
        ccode = countrycodeonly,
        region = town,
        year = year(rasterfiledate),
        builtschools = totedc,
        builthospitals = tothospital,
        builtroads = totroad,
        builtaccomodation = totaccomodation,
        builthouse = tothouse
        
      )
      resultstable <- rbind(resultstable, resulttemp)
      
    } # End of for loop
    
    # Write final dataframe object to file
    tryCatch(
      {
        write.csv(resultstable, paste0(osmdata_dir, "/", countrycodeonly, "_osmdata_", rasterfiledate, ".csv"))
        
        # Write state to log file
        write(paste0(Sys.time(), ": Info: ", " OSM data written to file: ", osmdata_dir, "/", countrycodeonly, "_osmdata_", rasterfiledate, ".csv"), file=logfile, append = TRUE)
        
      },
      error = function(e){
        print(e)
      },
      error = function(e){
        print(e)
      }
    )
    
    return(resultstable)
    
  } # End of if statement
  
} # Function closing bracket



########## Combining all countries data frame into a single data frame#######

# Load the Rds files into R
setwd(file.path(ntldata_dir))
getwd()

# Load all countries raster files in location ntldata_dir
ntlallfiles <- list.files(path=ntldata_dir, pattern = ".tif$", full.names = FALSE )

length(ntlallfiles)

# Generating the list of data for each country - ntl 
zonalstatsntl <- lapply(X = ntlallfiles, FUN = zonalstatsfunntl, var2 = updated_mpi_table)

zonalstatsntl[[13]]
# Apply the nighttime-light zonal-statistics function to all downloaded NTL
# rasters, standardise selected region names and combine the country results.

# Modify the data of SLE (Sierra Leone) shapefile of GADM. Drop previous region and update NAME_2 to region to align with the other lists
zonalstatsntl[[37]] <- zonalstatsntl[[37]] %>% dplyr::select(-c("region")) %>% rename("region" = "NAME_2")
#zonalstatsntl[[40]] <- zonalstatsntl[[40]] %>% dplyr::select(-c("region")) %>% rename("region" = "NAME_2")

# Update the region name of Ghana to remove "\r\n" from the Upper East Region. This issue is corrected in the shape file for 2022 Ghana file
#zonalstatsntl[[14]]
zonalstatsntl[[11]] %>% filter(region == "Upper East\r\n")
zonalstatsntl[[12]] %>% filter(region == "Upper East\r\n")
zonalstatsntl[[13]] %>% filter(region == "Upper East\r\n")

zonalstatsntl[[11]] <- zonalstatsntl[[11]] %>% mutate(region = str_replace_all(region, "Upper East\r\n", "Upper East"))
zonalstatsntl[[12]] <- zonalstatsntl[[12]] %>% mutate(region = str_replace_all(region, "Upper East\r\n", "Upper East"))
zonalstatsntl[[13]]


# combining the list of dataframe by rows
ntl_data_w.africa <- zonalstatsntl %>% reduce(bind_rows)
ntl_data_w.africa %>% filter(countrycode == "SLE")

# Review the dataframe
str(ntl_data_w.africa)

# Convert collectiondate to date data type
ntl_data_w.africa <- ntl_data_w.africa %>% mutate(collectiondate = lubridate::ymd(collectiondate))


# Drop the Geometry and adminarea columns as not required
ntl_data_w.africa <- ntl_data_w.africa %>% dplyr::select(-c(adminarea, geometry))
str(ntl_data_w.africa)

# Replace french/special characters with english
ntl_data_w.africa <- ntl_data_w.africa %>% mutate(region = stri_trans_general(region, "Latin-ASCII"))

# Convert all letters to small letters
ntl_data_w.africa.final <- ntl_data_w.africa %>% mutate(region = tolower(region))

# view the dataframe
str(ntl_data_w.africa.final)
dim(ntl_data_w.africa.final)


# Apply the land-cover zonal-statistics function to all downloaded LCU rasters,
# standardise region names and combine the country results.
############Land Cover and Use###########################
# Generating the list of data for each country - lcu
# Load the Rds files into R
setwd(file.path(lcudata_dir))
getwd()

lcuallfiles <- list.files(path=lcudata_dir, pattern = ".tif$", full.names = FALSE )
zonalstatslcu <- lapply(X = lcuallfiles, FUN = zonalstatsfunlcu, var2 = updated_mpi_table)

# Modify the data of SLE shapefile of GADM. Drop previous region and update NAME_2 to region 
zonalstatslcu[[37]] <- zonalstatslcu[[37]] %>% dplyr::select(-c("region")) %>% rename("region" = "NAME_2")
#zonalstatslcu[[40]] <- zonalstatslcu[[40]] %>% dplyr::select(-c("region")) %>% rename("region" = "NAME_2")

# Update the region name of Ghana to remove "\r\n"
zonalstatslcu[[11]] %>% filter(region == "Upper East\r\n")
zonalstatslcu[[12]] %>% filter(region == "Upper East\r\n")
zonalstatslcu[[13]] %>% filter(region == "Upper East\r\n")

zonalstatslcu[[11]] <- zonalstatslcu[[11]] %>% mutate(region = str_replace_all(region, "Upper East\r\n", "Upper East"))
zonalstatslcu[[12]] <- zonalstatslcu[[12]] %>% mutate(region = str_replace_all(region, "Upper East\r\n", "Upper East"))

# combining the list of dataframe by rows
lcu_data_w.africa <- zonalstatslcu %>% reduce(bind_rows)
lcu_data_w.africa %>% filter(countrycode == "SLE")

# Review the dataframe
str(lcu_data_w.africa)

# Convert collectiondate to date data type
lcu_data_w.africa <- lcu_data_w.africa %>% mutate(collectiondate = lubridate::ymd(collectiondate))

# Replace french/special characters with english
lcu_data_w.africa <- lcu_data_w.africa %>% mutate(region = stri_trans_general(region, "Latin-ASCII"))

# Convert all letters to small letters
lcu_data_w.africa <- lcu_data_w.africa %>% mutate(region = tolower(region))

str(lcu_data_w.africa)
View(lcu_data_w.africa)

# Filter for relevant variables and write the data to file
lcu_data_w.africa.final <- lcu_data_w.africa %>% dplyr::select(-c(adminarea, geometry))
dim(lcu_data_w.africa.final)
str(lcu_data_w.africa.final)


# Run the OSM extraction for the available country-specific land-cover files.
# Country-specific timeout and memory settings are retained from the original
# extraction workflow to accommodate differences in Overpass query size.
############Open Street Map Data########################
# Generating the dataframe for open street map per country
#osmstatslst
lcubenfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('BEN', lcurastfiles)) %>% unlist()
lcubfafiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('BFA', lcurastfiles)) %>% unlist()
lcucivfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('CIV', lcurastfiles)) %>% unlist()
lcucmrfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('CMR', lcurastfiles)) %>% unlist()
lcughafiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('GHA', lcurastfiles)) %>% unlist()
lcuginfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('GIN', lcurastfiles)) %>% unlist()
#lcugmbfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('GMB', lcurastfiles)) %>% unlist()
lcugnbfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('GNB', lcurastfiles)) %>% unlist()
lculbrfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('LBR', lcurastfiles)) %>% unlist()
lcumlifiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('MLI', lcurastfiles)) %>% unlist()
lcumrtfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('MRT', lcurastfiles)) %>% unlist()
lcunerfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('NER', lcurastfiles)) %>% unlist()
lcungafiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('NGA', lcurastfiles)) %>% unlist()
lcusenfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('SEN', lcurastfiles)) %>% unlist()
lcuslefiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('SLE', lcurastfiles)) %>% unlist()
lcustpfiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('STP', lcurastfiles)) %>% unlist()
lcutgofiles <- data.frame(lcurastfiles = lcuallfiles) %>% filter(grepl('TGO', lcurastfiles)) %>% unlist()
lcutgofiles[1]

osmstatsobj.BEN <- lapply(X = lcubenfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.BFA <- lapply(X = lcubfafiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.CIV <- lapply(X = lcucivfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.CMR <- lapply(X = lcucmrfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.GHA <- lapply(X = lcughafiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.GIN <- lapply(X = lcuginfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table, memorytoalloc = 2147483648)
#osmstatsobj.GMB <- lapply(X = lcugmbfiles, FUN = osm.data.extract, timeout = 900, df = updated_mpi_table)
osmstatsobj.GNB <- lapply(X = lcugnbfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.LBR <- lapply(X = lculbrfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.MLI <- lapply(X = lcumlifiles, FUN = osm.data.extract, timeout = 900, df = updated_mpi_table)
osmstatsobj.MRT <- lapply(X = lcumrtfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.NER <- lapply(X = lcunerfiles, FUN = osm.data.extract, timeout = 1500, df = updated_mpi_table)
osmstatsobj.NGA <- lapply(X = lcungafiles, FUN = osm.data.extract, timeout = 2000, df = updated_mpi_table)
osmstatsobj.SEN <- lapply(X = lcusenfiles, FUN = osm.data.extract, timeout = 1200, df = updated_mpi_table, memorytoalloc = 2684354560)
osmstatsobj.SLE <- lapply(X = lcuslefiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.STP <- lapply(X = lcustpfiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)
osmstatsobj.TGO <- lapply(X = lcutgofiles, FUN = osm.data.extract, timeout = 500, df = updated_mpi_table)



# Read the country-year OSM output files, combine them into one dataframe and
# standardise the collection date and region identifiers before integration.
#####################Load all csv files for OSM to a dataframe##################
osmallfiles <- list.files(path=osmdata_dir, pattern = ".csv$", full.names = TRUE )


# Read csv into dataframe
readcsvtodf <- function(fcsv){
  df <- read.delim(fcsv, header = TRUE, sep = ",")
}

#Generate a list for the osm data consisting of all selected countries
zonalstatsosm <- lapply(osmallfiles, readcsvtodf)

osm_data_w.africa <- zonalstatsosm %>% reduce(bind_rows)

# Rows for Ghana
osm_data_w.africa %>% filter(ccode == "GHA")

# Review the dataframe
str(osm_data_w.africa)
View(osm_data_w.africa)



# Convert collectiondate to date data type
osm_data_w.africa <- osm_data_w.africa %>% mutate(collectiondate = lubridate::ymd(year, truncated = 2L)) %>%
  dplyr::select(-c(year, X))

# Convert collection date to year only
#osm_data_w.africa <- osm_data_w.africa %>% mutate(collectiondate = year(collectiondate))


# Replace french/special characters with english
osm_data_w.africa <- osm_data_w.africa %>% mutate(region = stri_trans_general(region, "Latin-ASCII"))

# Update the town Upper East\r\r to Upper East
osm_data_w.africa <- osm_data_w.africa %>% mutate(region = str_replace_all(region, "Upper East\n", "Upper East"))

# Convert all letters to small letters
osm_data_w.africa <- osm_data_w.africa %>% mutate(region = tolower(region))

# Combine the two residential OSM counts into a single residence predictor and
# rename the infrastructure variables to the names used in the modelling dataset.
# sum the column accomodation and buildhouse into buildresidence and rename column ccode to countrycode
osm_data_w.africa.final <- osm_data_w.africa %>% 
  mutate(buildresidence = rowSums(across(c(buildaccomodation, buildhouse))))  %>%
  rename(countrycode = ccode) %>%
  dplyr::select(-c(buildaccomodation, buildhouse))

str(osm_data_w.africa.final)
dim(osm_data_w.africa.final)
View(osm_data_w.africa.final)

# Rename the osm data frame columns
osm_data_w.africa.final <- osm_data_w.africa.final %>% rename("builtschools" = buildschools,
                                                              "builtmedicalfacilities"= buildhospitals,
                                                              "builtroads"= buildroads,
                                                              "builtresidence" = buildresidence
)


osm_data_w.africa.final %>% filter(countrycode == "GHA")

# Comparing to identify the dfference
osm_data_w.africa.final %>% filter(countrycode == "GHA") %>% nrow()
lcu_data_w.africa.final %>% filter(countrycode == "GHA") %>% nrow()

#############################################################################


# Integrate the nighttime-light, land-cover and OSM datasets using country code,
# standardised region name and collection date as the common identifiers.
#############Combining all the data sources##########################

#----------------- Full Join on ntl and lcu---------------------#
ntl_lcu_data_w.africa <- ntl_data_w.africa.final %>% full_join(lcu_data_w.africa.final,
                                                               by=c("countrycode", "region", "collectiondate")) %>%
  dplyr::select(-c(id.x, id.y, country.y)) %>% rename(country = country.x)



dim(ntl_lcu_data_w.africa)
View(ntl_lcu_data_w.africa)

# Full Join of ntl_lcu_data_w.africa with OSM data
ntl_lcu_osm_data_w.africa <- ntl_lcu_data_w.africa %>% full_join(osm_data_w.africa.final,
                                                                 by=c("countrycode", "region", "collectiondate"))
View(ntl_lcu_osm_data_w.africa)
dim(ntl_lcu_osm_data_w.africa)

# Review the dimension of the combined data
dim(ntl_lcu_osm_data_w.africa)

# Check for missing data
rows_with_na <- ntl_lcu_osm_data_w.africa[apply(ntl_lcu_osm_data_w.africa, 1, function(x) any(is.na(x))), ]
dim(rows_with_na)


# Prepare the OPHI MPI table using the same collection-date and region-name
# conventions before joining the observed MPI values to the proxy predictors.
# Combine "updated_mpi_table" and "ntl_data_w.africa" into one table
# Convert year column to date column: collection date
mpi_data_modwithdate <- mpi_data %>% dplyr::select(country, countrycode, region, year, mpi) %>% 
  mutate(collectiondate = lubridate::ymd(year, truncated = 2L), mpi = as.numeric(mpi))

# ensure mpi is 3 digits from the decimal point.
mpi_data_modwithdate <- mpi_data_modwithdate %>% mutate(mpi = round(mpi, 3))

# Remove mpi value for years earlier than 2012
mpi_data_modwithdate <- mpi_data_modwithdate %>% filter(collectiondate > "2011-12-31")

mpi_data_modwithdate %>% filter(countrycode == 'BFA')


# Replace french/special characters with english
mpi_data_modwithdate <- mpi_data_modwithdate %>% mutate(region = stri_trans_general(region, "Latin-ASCII"))

# Convert all letters to small letters
mpi_data_modwithdate <- mpi_data_modwithdate %>% mutate(region = tolower(region))


dim(mpi_data_modwithdate)
View(mpi_data_modwithdate)

# Check for null value in mpi - no null value
mpi_data_modwithdate %>% filter(is.null(mpi))
mpionly <- mpi_data_modwithdate %>% dplyr::select(country,region)
all <- ntl_lcu_osm_data_w.africa %>% dplyr::select(country,region)

# full_join to have the mpi value as part of the main dataframe
ntl_lcu_osm_mpi_data_w.africa <- ntl_lcu_osm_data_w.africa %>% 
  full_join(mpi_data_modwithdate, by=c("countrycode", "region", "collectiondate")) %>%
  dplyr::select(-c(country.y, year)) %>% rename(country = country.x)

str(ntl_lcu_osm_mpi_data_w.africa)
dim(ntl_lcu_osm_mpi_data_w.africa)
View(ntl_lcu_osm_mpi_data_w.africa)


# Inspect missingness after integrating all data sources and identify observations
# that do not contain a complete set of MPI and predictor values.
# check for null values
na.dataframe <- ntl_lcu_osm_mpi_data_w.africa %>% filter(is.na(mpi))
ntl_lcu_osm_mpi_data_w.africa %>% filter(country == "Mali") %>% filter(is.na(mpi))


na_counts <- ntl_lcu_osm_mpi_data_w.africa %>%
  summarise(across(everything(), ~ sum(is.na(.))))

print(na_counts)


# Select rows with any NA values
rows_with_na <- ntl_lcu_osm_mpi_data_w.africa %>%
  filter(if_any(everything(), ~ is.na(.)))

# Print the resulting dataframe
print(rows_with_na)


# Retain complete observations for the analytical dataset and standardise the
# column names used in the subsequent exploratory and modelling workflows.
# Drop any row with NA.
ntl_lcu_osm_mpi_data_w.africa.dropna <- ntl_lcu_osm_mpi_data_w.africa %>% drop_na()

View(ntl_lcu_osm_mpi_data_w.africa.dropna)
str(ntl_lcu_osm_mpi_data_w.africa.dropna)
dim(ntl_lcu_osm_mpi_data_w.africa.dropna)

ntl_lcu_osm_mpi_data_w.africa.dropna %>% filter(is.na(mpi))

# Function to merge words by removing underscores
merge_words <- function(col_name) {
  # Remove all underscores and merge the words
  str_replace_all(col_name, "_", "")
}

# Apply the function to merge column names
colnames(ntl_lcu_osm_mpi_data_w.africa.dropna) <- sapply(colnames(ntl_lcu_osm_mpi_data_w.africa.dropna), merge_words)

# Print the updated column names
print(colnames(ntl_lcu_osm_mpi_data_w.africa.dropna))

#### Reorder columns ######
#ntl_lcu_osm_mpi_data_w.africa.dropna <- ntl_lcu_osm_mpi_data_w.africa.dropna %>% mutate(collectionyear = year(ymd(collectiondate)))
ntl_lcu_osm_mpi_data_w.africa.dropna <- ntl_lcu_osm_mpi_data_w.africa.dropna %>% dplyr::select(c(collectiondate, country, countrycode, region, 
                                                                                                 everything()))

# Create the final working dataframe by rounding continuous measurements and
# storing the OSM infrastructure counts as integer variables.
# Round up all number columns to 3 digits
str(ntl_lcu_osm_mpi_data_w.africa.dropna)

df <- ntl_lcu_osm_mpi_data_w.africa.dropna %>% mutate(
  ntlmeanintensity = round(ntlmeanintensity, 3),
  ntlcoverageperc = round(ntlcoverageperc, 3),
  lcumixedforestperc = round(lcumixedforestperc,3),
  lcuclosedshrublandsperc = round(lcuclosedshrublandsperc, 3), 
  lcuopenshrublandsperc = round(lcuopenshrublandsperc, 3),
  lcuwoodysavannasperc = round(lcuwoodysavannasperc, 3),
  lcusavannasperc = round(lcusavannasperc, 3),
  lcugrasslandsperc = round(lcugrasslandsperc, 3),
  lcucroplandsperc = round(lcucroplandsperc, 3),
  lcuurbanbuiltupperc = round(lcuurbanbuiltupperc, 3),
  lcucroplandnatvegperc = round(lcucroplandnatvegperc, 3),
  lcubarrenperc = round(lcubarrenperc, 3), 
  builtschools = as.integer(builtschools),
  builtmedicalfacilities = as.integer(builtmedicalfacilities),
  builtroads = as.integer(builtroads),
  builtresidence = as.integer(builtresidence),
  mpi = round(mpi, 3)
)


#  Set countrycode and region as factor
#ntl_lcu_osm_mpi_data_w.africa.dropna.final <- ntl_lcu_osm_mpi_data_w.africa.dropna.final %>% mutate(countrycode = as.factor(countrycode), region = as.factor(region))

# Perform high-level checks of the final dataframe, including its structure,
# dimensions, summary statistics and the number of unique values by data type.
###### Understanding the data frame on a high level ###
str(df)
dim(df)
summary(df)

# Number of observations in each column
observation_counts <- sapply(df, length)
print(observation_counts)

# Get the number of unique data per column
# Sum of unique numbers for each column
# Example dataframe with mixed data types

# Function to count unique values for each data type
count_unique <- function(dfr) {
  results <- list(
    numeric = dfr %>% summarise(across(where(is.numeric), ~ length(unique(.)))),
    date = dfr %>% summarise(across(where(~ inherits(., "Date")), ~ length(unique(.)))),
    character = dfr %>% summarise(across(where(is.character), ~ length(unique(.))))
  )
  return(results)
}

# Get the counts
unique_counts <- count_unique(df)
print(unique_counts)


df %>% filter(countrycode == "GHA")
dim(df)

# select countries with mpi lower than 0.133
region0133 <- df %>% filter(mpi < 0.133)
View(region0133)

region0412 <- df %>% filter(mpi > 0.412)
View(region0412)

# Save the final integrated dataframe to PostgreSQL, CSV and RDS formats and
# reload the RDS object as a basic check that the prepared data can be restored.
########### Preparing the data ############################

# Write data to postgres databases - dataframe and table will contain null value for mpi however contains ntl lcu and osm data
conn <- dbConnect(PostgreSQL(), user="postgres", password="password", dbname="dissertation")
dbWriteTable(conn, "finalntllcuosmmpi2026", df)


# Write final dataframe object to file
write.csv(df, paste0(destdftofile, "/", "ntl_lcu_osm_data.west.africa-2026.csv"))
saveRDS(df, file = paste0(destdftofile, "/", "ntl_lcu_osm_data.west.africa-2026.rds") )

# Restore to confirm if data is restorable
df <- readRDS(paste0(destdftofile, "/", "ntl_lcu_osm_data.west.africa-2026.rds"))

# Begin the exploratory analysis of the prepared dataset. The following section
# examines distributions, relationships and skewness before any train/test split
# or model fitting is performed.
########### Exploratory Data Analysis ####################





###### Facet_wrap of the histogram

# Select numeric column Reshape the data to long format
df_long <- df %>% select_if(is.numeric) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

# Manually calculate bins and counts
data_bins <- df_long %>%
  group_by(variable) %>%
  reframe(
    bin = cut(value, breaks = 10, include.lowest = TRUE),
    value = value
  ) %>%
  group_by(variable, bin) %>%
  summarize(count = n(), .groups = "drop")

# View the resulting data
print(data_bins)

# Plot the histogram
ggplot(data_bins, aes(x = bin, y = count)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  facet_wrap(~ variable, scales = "free") +
  labs(
    x = "Value",
    y = "Count") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold", size = 12),
    axis.title = element_text(size = 14),
    strip.text = element_text(face = "bold", size = 14),
    strip.background = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    panel.border = element_rect(color = "black", fill = NA),
    text = element_text(family = "serif"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = margin(b = 10)),  # Center title and add margin
    plot.background = element_blank()  # Ensure plot background is transparent
  )

ggsave("/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/data/images/barchat.pdf", 
       width = 20, height = 15, dpi = 300, bg = "transparent")


###### Correlation matrix and Correlation Analysis


# Custom function to add correlation coefficients and p-values to the plots
cor_fun <- function(data, mapping, method = "spearman", ...) {
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)
  
  # Calculate correlation coefficient
  corr <- cor(x, y, method = method, use = "complete.obs")  # Handle missing values
  
  # Calculate p-value
  p_value <- cor.test(x, y, method = method)$p.value
  
  # Determine significance level
  significance <- ifelse(p_value < 0.001, "***",
                         ifelse(p_value < 0.01, "**",
                                ifelse(p_value < 0.05, "*", "")))
  
  # Create label with correlation coefficient and significance
  label <- paste("r =", round(corr, 2), significance)
  
  # Color coding for correlation strength
  col <- ifelse(corr > 0.6, "green", ifelse(corr < -0.6, "red", "black"))
  
  # Display the label
  ggally_text(
    label = label,
    mapping = aes(),
    color = col,
    size = 4,
    fontface = "bold",
    ...
  )
}

# Create the ggpairs plot with customizations - pearson
ggpairs(
  df, 
  columns = 5:21,  # Select columns for analysis
  upper = list(continuous = wrap(cor_fun, method = "pearson")),  # Add correlation coefficients and p-values
  lower = list(continuous = wrap("smooth", method = "lm", se = FALSE, color = "blue")),  # Add regression lines
  diag = list(continuous = wrap("densityDiag", fill = "orange", alpha = 0.5))  # Add density plots
) +
  theme_minimal() +  # Use a minimal theme
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 8),  # Rotate x-axis labels vertically
    axis.text.y = element_text(angle = 0, hjust = 1, size = 8),  # Adjust y-axis labels
    strip.text = element_text(size = 10, face = "bold", angle=45),  # Adjust facet label text
    strip.text.y = element_text(size = 10, face = "bold", angle = 0),  # Rotate y-axis facet labels
    panel.grid.major = element_line(color = "gray90"),  # Add light gray gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    panel.background = element_rect(fill = "white"),  # Set panel background to white
    plot.background = element_rect(fill = "white")  # Set plot background to white
  )


getwd()
ggsave("/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/data/images/correlation_matrix_spearman2026.png", 
       width = 20, height = 15, dpi = 300)


#############spearman
#----------------------------------------------------------
# 1. CORRELATION FUNCTION (Preserves warning)
#----------------------------------------------------------
cor_fun <- function(data, mapping, method = "spearman", ...) {
  x <- eval_data_col(data, mapping$x)
  y <- eval_data_col(data, mapping$y)
  
  # Handle constant variables
  if (sd(x, na.rm = TRUE) == 0 || sd(y, na.rm = TRUE) == 0) {
    return(ggally_text("NA\n(constant)", size = 3, color = "gray50"))
  }
  
  # Calculate correlation (warning preserved)
  corr <- cor(x, y, method = method, use = "complete.obs")
  test <- cor.test(x, y, method = method)  # Warning will appear here
  
  # Formatting
  significance <- ifelse(test$p.value < 0.001, "***",
                         ifelse(test$p.value < 0.01, "**",
                                ifelse(test$p.value < 0.05, "*", "")))
  
  label <- paste0("ρ = ", round(corr, 2), significance)
  
  ggally_text(
    label,
    size = 2.5,
    color = ifelse(abs(corr) > 0.6, "#1F77B4", "black"),  # Blue for strong correlations
    fontface = "bold"
  )
}

#----------------------------------------------------------
# 2. PLOT GENERATION 
#----------------------------------------------------------
plot <- ggpairs(
  df,
  columns = 5:21,
  upper = list(continuous = wrap(cor_fun)),
  lower = list(continuous = wrap("smooth", method = "lm", color = "#1F77B4")),
  diag = list(continuous = wrap("densityDiag", fill = "#FF7F00"))
) +
  theme_bw() +
  theme(
    # Rotate ALL axis labels (including subplot column names)
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 7),
    axis.text.y = element_text(angle = 0, hjust = 1, size = 7),
    
    # Adjust facet (strip) labels - these are the subplot titles
    strip.text.x = element_text(angle = 90, size = 8, vjust = 0.5),  # Vertical text
    strip.text.y = element_text(angle = 0, size = 8),  # Horizontal text
    
    # Layout tweaks
    strip.background = element_blank(),
    panel.spacing = unit(0.1, "lines"),
    plot.margin = margin(1, 1, 1, 1, "cm")
  )
#----------------------------------------------------------
# 3. SAVE OUTPUT
#----------------------------------------------------------
ggsave("/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/data/images/correlation_matrix.tiff", plot, 
       width = 14, height = 12,  # Larger canvas for readability
       dpi = 300, compression = "lzw")

# If you need to customize further:
ggsave(
  "/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/data/images/correlation_matrix_final.png", 
  plot, 
  width = 14, 
  height = 12,
  dpi = 300,          # High resolution (300 dots per inch)
  bg = "white",       # Transparent background if needed (use "transparent")
  type = "cairo"      # Higher quality anti-aliasing (recommended)
)

###################



####### Scatter plot of the variables

# Plot a scatter diagram of the mpi with various predictors

# Create the plot

plotmpintlcov <- ggplot(df, aes(x = ntlcoverageperc, y = mpi)) +
  geom_point(alpha = 0.6, size = 3, color = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, color = "darkred", linetype = "dashed") +
  labs(
    x = "Percentage coverage of Nighttime Light per subnational region",
    y = "Multidimensional Poverty Index (MPI)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray40"),
    axis.title.x = element_text(face = "bold", size = 16),  # Increased from 14 to 16
    axis.title.y = element_text(face = "bold", size = 16),  # Increased from 14 to 16
    axis.text = element_text(size = 12),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 10, color = "gray50", hjust = 1),
    axis.line = element_line(linewidth = 1.5)
  ) +
  scale_x_continuous(labels = scales::percent_format(scale = 1)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10))

ggsave(
  "/home/charles/Documents/study/MscDataScience/Dissertation/Rstudio/data/images/plotmpintlcov.png", 
  plotmpintlcov, 
  width = 14, 
  height = 12,
  dpi = 300,          # High resolution (300 dots per inch)
  bg = "white",       # Transparent background if needed (use "transparent")
  type = "cairo"      # Higher quality anti-aliasing (recommended)
)

dev.off()




# Create a scatter plot and linear regression line
ggplot(df, aes(x = ntlmeanintensity, y = mpi)) +
  geom_point(alpha = 0.6, size = 3, color = "steelblue") +  # Adjust point transparency, size, and color
  geom_smooth(method = "lm", se = FALSE, color = "darkred", linetype = "dashed") +  # Add a trendline
  labs(
    title = "Scatter Plot of MPI Against Nighttime Light Coverage",
    subtitle = "Relationship between Multidimensional Poverty Index (MPI) and yearly average ntl intensity",
    x = "Percentage Spread of Nighttime Light",
    y = "Multidimensional Poverty Index (MPI)",
    caption = "Data Source: [Insert Data Source Here]"
  ) +
  theme_minimal(base_size = 14) +  # Use a minimal theme with a larger base font size
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),  # Customize title
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray40"),  # Customize subtitle
    axis.title.x = element_text(face = "bold", size = 14),  # Customize x-axis title
    axis.title.y = element_text(face = "bold", size = 14),  # Customize y-axis title
    axis.text = element_text(size = 12),  # Customize axis text
    panel.grid.major = element_line(color = "gray90"),  # Customize major grid lines
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    plot.caption = element_text(size = 10, color = "gray50", hjust = 1)  # Customize caption
  ) +
  scale_x_continuous(labels = scales::percent_format(scale = 1)) +  # Format x-axis as percentages
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10))  # Add pretty breaks for y-axis



# Create the plot
Pmpintlcoverage <-  ggplot(df, aes(x = ntlcoverageperc, y = mpi)) +
  geom_point(alpha = 0.6, size = 3, color = "steelblue") +  # Adjust point transparency, size, and color
  geom_smooth(method = "lm", se = FALSE, color = "darkred", linetype = "dashed") +  # Add a trendline
  labs(
    x = "Percentage coverage of Nighttime Light per region",
    y = "Multidimensional Poverty Index (MPI)",
    caption = "Data Source: [Dataset used to build the model. Extracted from Google Earth Engine, Open Street Map and OPHI ]"
  ) +
  theme_minimal(base_size = 14) +  # Use a minimal theme with a larger base font size
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),  # Customize title
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray40"),  # Customize subtitle
    axis.title.x = element_text(face = "bold", size = 14),  # Customize x-axis title
    axis.title.y = element_text(face = "bold", size = 14),  # Customize y-axis title
    axis.text = element_text(size = 12),  # Customize axis text
    panel.grid.major = element_line(color = "gray90"),  # Customize major grid lines
    panel.grid.minor = element_blank(),  # Remove minor grid lines
    plot.caption = element_text(size = 10, color = "gray50", hjust = 1)  # Customize caption
  ) +
  scale_x_continuous(labels = scales::percent_format(scale = 1)) +  # Format x-axis as percentages
  scale_y_continuous(breaks = scales::pretty_breaks(n = 10))  # Add pretty breaks for y-axis

png("plotmpintlcoverage2026.png") 
print(Pmpintlcoverage)
dev.off()



# Check for normality of the data
boxplot(df$ntlcoverageperc)
colnames(ntl_data_w.africa)
colnames(df)
#boxplot(ntl_data_w.africa$ntl_total_intensity)
#boxplot(ntl_data_w.africa$ntl_perc_spread)
#boxplot(ntl_data_w.africa$ntl_px_zero_val)


ggplot(ntl_data_w.africa, aes(x=ntl_mean_intensity)) +
  geom_histogram(bins = 30, col= "white") + labs(title = "A histogram of mean night time light value distribution",
                                                 x = "mean night time light")

ggplot(df, aes(x=ntlmeanintensity)) +
  geom_histogram(bins = 30, col= "white") + labs(title = "A histogram of mean night time light value distribution",
                                                 x = "mean night time light")
ggplot(ntl_data_w.africa, aes(x=ntl_coverage_perc)) +
  geom_histogram(bins = 30, col= "white") 

ggplot(df, aes(x=ntlcoverageperc)) +
  geom_histogram(bins = 30, col= "white") 

ggplot(df, aes(x=mpi)) +
  geom_histogram(bins = 30, col= "white") + labs(title = "A histogram of mpi value distribution",
                                                 x = "mean night time light")

# shapiro test shows data not normally distributed
shapiro.test(df$ntlcoverageperc)
shapiro.test(df$ntlmeanintensity)
shapiro.test(df$mpi)


# skewness test
skewness(df$ntlcoverageperc)
skewness(df$lcumixedforestperc)
skewness(df$lcu_closedshrublands_perc)


str(df)

# Quantify skewness for the numeric variables and compare several candidate
# transformations to understand how strongly skewed predictors respond to each
# transformation. These checks are exploratory and do not perform model fitting.
# Log Transform the data to account for skewness
skwnessdata <- function(varname, table, pstatus){
  
  # check the skewness value
  #skew_val = table %>% dplyr::select(.data[[varname]]) %>% skewness()
  skew_val = table %>% dplyr::select(all_of(varname)) %>% skewness()
  print(paste0("skew_val: ", skew_val))
  print(paste0("colname: ", varname))
  
  result <- data.frame(
    colname = varname,
    status = pstatus,
    skewval = skew_val,
    row.names = NULL
  )
  #print(result)
  return(result)
}

collist = df %>% select_if(is.numeric) %>% colnames %>% unlist
skewtable <- lapply(X = collist, FUN = skwnessdata, table = df, pstatus = "beforetransformation")
df.skewtable <- skewtable %>% reduce(bind_rows)
df.skewtable <- df.skewtable %>% mutate(skewval = round(skewval,3))
dim(df.skewtable)

df.skewtable


# Examine square-root transformed versions of the predictors and recalculate
# their skewness for comparison with the untransformed data.
# square root -  Transform the data to account for skewness
df.sqrttransform <- df %>% mutate(ntlmeanintensity = round(sqrt(ntlmeanintensity), 3),
                                  ntlcoverageperc = round(sqrt(ntlcoverageperc), 3),
                                  lcumixedforestperc = round(sqrt(lcumixedforestperc), 3),
                                  lcuclosedshrublandsperc = round(sqrt(lcuclosedshrublandsperc), 3), 
                                  lcuopenshrublandsperc = round(sqrt(lcuopenshrublandsperc), 3),
                                  lcuwoodysavannasperc = round(sqrt(lcuwoodysavannasperc), 3),
                                  lcusavannasperc = round(sqrt(lcusavannasperc), 3),
                                  lcugrasslandsperc = round(sqrt(lcugrasslandsperc), 3),
                                  lcucroplandsperc = round(sqrt(lcucroplandsperc), 3),
                                  lcuurbanbuiltupperc = round(sqrt(lcuurbanbuiltupperc), 3),
                                  lcucroplandnatvegperc = round(sqrt(lcucroplandnatvegperc), 3),
                                  lcubarrenperc = round(sqrt(lcubarrenperc), 3),
                                  builtschools = round(sqrt(builtschools), 3),
                                  builtmedicalfacilities = round(sqrt(builtmedicalfacilities),3),
                                  builtroads = round(sqrt(builtroads), 3),
                                  builtresidence = round(sqrt(builtresidence), 3)
)

colnames(df.sqrttransform)

# drop columns not required
#df.sqrttransform <- df.sqrttransform %>% dplyr::select(-c("ntl_mean_intensity", "ntl_spread_perc", "lcu_mixedforest_perc", "lcu_closedshrublands_perc",
#                                                          "lcu_openshrublands_perc","lcu_woodysavannas_perc", "lcu_savannas_perc",
#                                                          "lcu_grasslands_perc", "lcu_croplands_perc", "lcu_urbanbuiltup_perc","lcu_croplandnatveg_perc",
#                                                          "lcu_barren_perc", "buildschools", "buildhospitals", "buildroads", "buildresidence" ))


colnames(df.sqrttransform)


###### Correlation
#GGally::ggpairs(df.sqrttransform, columns = 5:21)

############check for skewness#######
collistsqrt = df.sqrttransform %>% select_if(is.numeric) %>% colnames %>% unlist
skewtablesqrt <- lapply(X = collistsqrt, FUN = skwnessdata, table = df.sqrttransform, pstatus = "sqrttransformation")
df.skewtablesqrt <- skewtablesqrt %>% reduce(bind_rows)
df.skewtablesqrt <- df.skewtablesqrt %>% mutate(skewval = round(skewval,3))
dim(df.skewtablesqrt)

df.skewtablesqrt



# Examine cube-root transformed versions of the predictors and recalculate
# their skewness for comparison with the alternative transformations.
#####Cuberoot
df.cbrttransform <- df %>% mutate(ntlmeanintensity = round((ntlmeanintensity)^(1/3), 3),
                                  ntlcoverageperc = round((ntlcoverageperc)^(1/3), 3),
                                  lcumixedforestperc = round((lcumixedforestperc)^(1/3), 3),
                                  lcuclosedshrublandsperc = round((lcuclosedshrublandsperc)^(1/3), 3), 
                                  lcuopenshrublandsperc = round((lcuopenshrublandsperc)^(1/3), 3),
                                  lcuwoodysavannasperc = round((lcuwoodysavannasperc)^(1/3), 3),
                                  lcusavannasperc = round((lcusavannasperc)^(1/3), 3),
                                  lcugrasslandsperc = round((lcugrasslandsperc)^(1/3), 3),
                                  lcucroplandsperc = round((lcucroplandsperc)^(1/3), 3),
                                  lcuurbanbuiltupperc = round((lcuurbanbuiltupperc)^(1/3), 3),
                                  lcucroplandnatvegperc = round((lcucroplandnatvegperc)^(1/3), 3),
                                  lcubarrenperc = round((lcubarrenperc)^(1/3), 3),
                                  builtschools = round((builtschools)^(1/3), 3),
                                  builtmedicalfacilities = round((builtmedicalfacilities)^(1/3),3),
                                  builtroads = round((builtroads)^(1/3), 3),
                                  builtresidence = round((builtresidence)^(1/3), 3)
)

colnames(df.cbrttransform)

# drop columns not required
#df.cbrttransform <- df.cbrttransform %>% dplyr::select(-c("ntl_mean_intensity", "ntl_spread_perc", "lcu_mixedforest_perc", "lcu_closedshrublands_perc",
#                                                          "lcu_openshrublands_perc","lcu_woodysavannas_perc", "lcu_savannas_perc",
#                                                          "lcu_grasslands_perc", "lcu_croplands_perc", "lcu_urbanbuiltup_perc","lcu_croplandnatveg_perc",
#                                                          "lcu_barren_perc", "buildschools", "buildhospitals", "buildroads", "buildresidence" ))


colnames(df.cbrttransform)


############check for skewness#######
collistcbrt = df.cbrttransform %>% select_if(is.numeric) %>% colnames %>% unlist
skewtablecbrt <- lapply(X = collistcbrt, FUN = skwnessdata, table = df.cbrttransform, pstatus = "cbrttransformation")
df.skewtablecbrt <- skewtablecbrt %>% reduce(bind_rows)
df.skewtablecbrt <- df.skewtablecbrt %>% mutate(skewval = round(skewval,3))
dim(df.skewtablecbrt)

df.skewtablecbrt




# Examine natural-log transformed versions of the predictors using an offset of
# one so that predictors containing zero values can be transformed.
# Natural Log Transform the data to account for skewness
df.logtransform <- df %>% mutate(ntlmeanintensity = round(log(ntlmeanintensity + 1), 3),
                                 ntlcoverageperc = round(log(ntlcoverageperc + 1), 3),
                                 lcumixedforestperc = round(log(lcumixedforestperc + 1), 3),
                                 lcuclosedshrublandsperc = round(log(lcuclosedshrublandsperc + 1), 3), 
                                 lcuopenshrublandsperc = round(log(lcuopenshrublandsperc + 1), 3),
                                 lcuwoodysavannasperc = round(log(lcuwoodysavannasperc + 1), 3),
                                 lcusavannasperc = round(log(lcusavannasperc + 1), 3),
                                 lcugrasslandsperc = round(log(lcugrasslandsperc + 1), 3),
                                 lcucroplandsperc = round(log(lcucroplandsperc + 1), 3),
                                 lcuurbanbuiltupperc = round(log(lcuurbanbuiltupperc + 1), 3),
                                 lcucroplandnatvegperc = round(log(lcucroplandnatvegperc + 1), 3),
                                 lcubarrenperc = round(log(lcubarrenperc + 1), 3),
                                 builtschools = round(log(builtschools + 1), 3),
                                 builtmedicalfacilities = round(log(builtmedicalfacilities + 1),3),
                                 builtroads = round(log(builtroads + 1), 3),
                                 builtresidence = round(log(builtresidence + 1), 3)
)

colnames(df.logtransform)

############check for skewness#######
collistlog = df.logtransform %>% select_if(is.numeric) %>% colnames %>% unlist
skewtablelog <- lapply(X = collistlog, FUN = skwnessdata, table = df.logtransform, pstatus = "logtransformation")
df.skewtablelog <- skewtablelog %>% reduce(bind_rows)
df.skewtablelog <- df.skewtablelog %>% mutate(skewval = round(skewval,3))
dim(df.skewtablelog)

df.skewtablelog





# Examine base-10 logarithmic transformations as an additional exploratory
# comparison of predictor skewness.
# Log base 10 transform the data to account for skewness
df.log10transform <- df %>% mutate(ntlmeanintensity = round(log10(ntlmeanintensity + 1), 3),
                                   ntlcoverageperc = round(log10(ntlcoverageperc + 1), 3),
                                   lcumixedforestperc = round(log10(lcumixedforestperc + 1), 3),
                                   lcuclosedshrublandsperc = round(log10(lcuclosedshrublandsperc + 1), 3), 
                                   lcuopenshrublandsperc = round(log10(lcuopenshrublandsperc + 1), 3),
                                   lcuwoodysavannasperc = round(log10(lcuwoodysavannasperc + 1), 3),
                                   lcusavannasperc = round(log10(lcusavannasperc + 1), 3),
                                   lcugrasslandsperc = round(log10(lcugrasslandsperc + 1), 3),
                                   lcucroplandsperc = round(log10(lcucroplandsperc + 1), 3),
                                   lcuurbanbuiltupperc = round(log10(lcuurbanbuiltupperc + 1), 3),
                                   lcucroplandnatvegperc = round(log10(lcucroplandnatvegperc + 1), 3),
                                   lcubarrenperc = round(log10(lcubarrenperc + 1), 3),
                                   builtschools = round(log10(builtschools + 1), 3),
                                   builtmedicalfacilities = round(log10(builtmedicalfacilities + 1),3),
                                   builtroads = round(log10(builtroads + 1), 3),
                                   builtresidence = round(log10(builtresidence + 1), 3)
)

colnames(df.log10transform)

############check for skewness#######
collistlog10 = df.log10transform %>% select_if(is.numeric) %>% colnames %>% unlist
skewtablelog10 <- lapply(X = collistlog10, FUN = skwnessdata, table = df.log10transform, pstatus = "log10transformation")
df.skewtablelog10 <- skewtablelog10 %>% reduce(bind_rows)
df.skewtablelog10 <- df.skewtablelog10 %>% mutate(skewval = round(skewval,3))
dim(df.skewtablelog10)
df.skewtablelog10


# Examine inverse transformations and compare their skewness with the other
# candidate transformations considered during the exploratory assessment.
#####Inverse Transform
df.inversetransform <- df %>% mutate(ntlmeanintensity = round((ntlmeanintensity + 1)^(-1), 3),
                                     ntlcoverageperc = round((ntlcoverageperc + 1)^(-1), 3),
                                     lcumixedforestperc = round((lcumixedforestperc + 1)^(-1), 3),
                                     lcuclosedshrublandsperc = round((lcuclosedshrublandsperc + 1)^(-1), 3), 
                                     lcuopenshrublandsperc = round((lcuopenshrublandsperc + 1)^(-1), 3),
                                     lcuwoodysavannasperc = round((lcuwoodysavannasperc + 1)^(-1), 3),
                                     lcusavannasperc = round((lcusavannasperc + 1)^(-1), 3),
                                     lcugrasslandsperc = round((lcugrasslandsperc + 1)^(-1), 3),
                                     lcucroplandsperc = round((lcucroplandsperc + 1)^(-1), 3),
                                     lcuurbanbuiltupperc = round((lcuurbanbuiltupperc + 1)^(-1), 3),
                                     lcucroplandnatvegperc = round((lcucroplandnatvegperc + 1)^(-1), 3),
                                     lcubarrenperc = round((lcubarrenperc + 1)^(-1), 3),
                                     builtschools = round((builtschools + 1)^(-1), 3),
                                     builtmedicalfacilities = round((builtmedicalfacilities + 1)^(-1),3),
                                     builtroads = round((builtroads + 1)^(-1), 3),
                                     builtresidence = round((builtresidence + 1)^(-1), 3)
)

colnames(df.inversetransform)

############check for skewness#######
collistinv = df.inversetransform %>% select_if(is.numeric) %>% colnames %>% unlist
skewtableinv <- lapply(X = collistinv, FUN = skwnessdata, table = df.inversetransform, pstatus = "inversetransformation")
df.skewtableinv <- skewtableinv %>% reduce(bind_rows)
df.skewtableinv <- df.skewtableinv %>% mutate(skewval = round(skewval,3))
dim(df.skewtableinv)
df.skewtableinv

# Combine the skewness results from the original and transformed datasets so
# that the behaviour of each predictor can be reviewed across transformations.
# row_bind all the tables together
# This table helps to identify which transformation should be carried out to reduce skewness
df.skewnesscombined <- rbind(df.skewtable, df.skewtablesqrt, df.skewtablecbrt, df.skewtablelog, df.skewtablelog10,df.skewtableinv)

df.skewnesscombined %>% dplyr::filter(colname == "ntlmeanintensity")
df.skewnesscombined %>% dplyr::filter(colname == "ntlcoverageperc")

df.skewnesscombined %>% dplyr::filter(colname == "lcumixedforestperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcuclosedshrublandsperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcuopenshrublandsperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcuwoodysavannasperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcusavannasperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcugrasslandsperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcucroplandsperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcuurbanbuiltupperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcucroplandnatvegperc")
df.skewnesscombined %>% dplyr::filter(colname == "lcubarrenperc")

df.skewnesscombined %>% dplyr::filter(colname == "builtschools")
df.skewnesscombined %>% dplyr::filter(colname == "builtmedicalfacilities")
df.skewnesscombined %>% dplyr::filter(colname == "builtroads")
df.skewnesscombined %>% dplyr::filter(colname == "builtresidence")