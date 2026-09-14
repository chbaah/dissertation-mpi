# ============================================================================
# MULTIDIMENSIONAL POVERTY INDEX PLUMBER API
# ============================================================================

library(lubridate)
library(plumber)
library(parsnip)
library(tidymodels)
library(recipes)
library(RPostgreSQL)
library(DBI)
library(xgboost)
library(bundle)
library(dplyr)

# ----------------------------------------------------------------------------
# Global settings
# ----------------------------------------------------------------------------

MPI_THRESHOLD <- 0.3333

# ----------------------------------------------------------------------------
# Model and recipe configuration
# ----------------------------------------------------------------------------

LOCAL_PROJECT_DIR <- Sys.getenv(
  "MPI_PROJECT_DIR",
  unset = "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI"
)

MODEL_DIR <- Sys.getenv(
  "MODEL_DIR",
  unset = file.path(LOCAL_PROJECT_DIR, "bestmodelimages")
)

MODEL_FILE <- Sys.getenv(
  "MODEL_FILE",
  unset = "xgb_ULTIMATE_best_model_space_filling_all_pred_notfm_20260420_015733.rds"
)

RECIPE_FILE <- Sys.getenv(
  "RECIPE_FILE",
  unset = "xgboost_all_pred_notfm_recipe.rds"
)

model_path <- file.path(MODEL_DIR, MODEL_FILE)
recipe_path <- file.path(MODEL_DIR, RECIPE_FILE)

if (!file.exists(model_path)) {
  stop(paste0("Model file not found: ", model_path))
}

if (!file.exists(recipe_path)) {
  stop(paste0("Recipe file not found: ", recipe_path))
}

message("Loading MPI model: ", model_path)
fmodel_bundled <- readRDS(model_path)
fmodel <- bundle::unbundle(fmodel_bundled)

message("Loading preprocessing recipe: ", recipe_path)
frecipe <- readRDS(recipe_path)

message("Model class: ", paste(class(fmodel), collapse = ", "))
message("Recipe class: ", paste(class(frecipe), collapse = ", "))

# Raw variables expected by the fitted recipe
RAW_PREDICTORS <- frecipe$var_info %>%
  dplyr::filter(role == "predictor") %>%
  dplyr::pull(variable) %>%
  unique()

if (length(RAW_PREDICTORS) == 0) {
  stop("No predictor variables were found in the fitted recipe.")
}

message(
  "Raw predictors expected by recipe: ",
  paste(RAW_PREDICTORS, collapse = ", ")
)

# ----------------------------------------------------------------------------
# Database configuration
# ----------------------------------------------------------------------------

DB_HOST <- Sys.getenv("DB_HOST", unset = "127.0.0.1")
DB_PORT <- as.integer(Sys.getenv("DB_PORT", unset = "5432"))
DB_NAME <- Sys.getenv("DB_NAME", unset = "mpi")
DB_USER <- Sys.getenv("DB_USER", unset = "postgres")
DB_PASSWORD <- Sys.getenv("DB_PASSWORD", unset = "postgres")

get_db_connection <- function() {
  DBI::dbConnect(
    RPostgreSQL::PostgreSQL(),
    host = DB_HOST,
    port = DB_PORT,
    dbname = DB_NAME,
    user = DB_USER,
    password = DB_PASSWORD
  )
}

# ----------------------------------------------------------------------------
# Prediction helper functions
# ----------------------------------------------------------------------------

prepare_raw_predictors <- function(new_data) {
  missing_predictors <- setdiff(RAW_PREDICTORS, names(new_data))
  
  if (length(missing_predictors) > 0) {
    stop(
      paste0(
        "Missing predictor variables: ",
        paste(missing_predictors, collapse = ", ")
      )
    )
  }
  
  new_data %>%
    dplyr::select(dplyr::all_of(RAW_PREDICTORS))
}


# ============================================================================
# SHARED MODEL PREDICTION FUNCTION
# ============================================================================
#
# Every prediction passes through this same function.
#
#
#     Raw data
#        |
#        v
# prepare_raw_predictors()
#        |
#        v
#     bake()
#        |
#        v
# processed predictors
#        |
#        v
#   xgb.DMatrix
#        |
#        v
# XGBoost prediction
#
#
# This ensures bulk predictions and single-observation predictions use
# exactly the same preprocessing logic.
#
# ============================================================================


predict_mpi <- function(new_data) {
  raw_data <- prepare_raw_predictors(new_data)
  
  baked_data <- recipes::bake(
    frecipe,
    new_data = raw_data
  )
  
  predictor_cols <- setdiff(names(baked_data), "mpi")
  
  if (length(predictor_cols) == 0) {
    stop("No predictor columns remained after recipe preprocessing.")
  }
  
  X_new <- baked_data %>%
    dplyr::select(dplyr::all_of(predictor_cols)) %>%
    as.data.frame()
  
  X_new[] <- lapply(X_new, function(x) suppressWarnings(as.numeric(x)))
  
  if (anyNA(X_new)) {
    stop("NA values were detected in the processed XGBoost predictors.")
  }
  
  dmatrix <- xgboost::xgb.DMatrix(data = as.matrix(X_new))
  predictions <- predict(fmodel, dmatrix)
  
  list(
    predictions = predictions,
    baked_data = baked_data,
    predictor_cols = predictor_cols
  )
}

#* @apiTitle Multidimensional Poverty Index API

# ----------------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------------

#* @filter cors
function(req, res) {
  res$setHeader("Access-Control-Allow-Origin", "*")
  
  if (req$REQUEST_METHOD == "OPTIONS") {
    res$setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    
    if (!is.null(req$HTTP_ACCESS_CONTROL_REQUEST_HEADERS)) {
      res$setHeader(
        "Access-Control-Allow-Headers",
        req$HTTP_ACCESS_CONTROL_REQUEST_HEADERS
      )
    }
    
    res$status <- 200
    return(list())
  }
  
  plumber::forward()
}

# ----------------------------------------------------------------------------
# Test endpoint
# ----------------------------------------------------------------------------

#* @param msg Message to return
#* @get /api/test
#* @serializer json
function(msg = "Hello") {
  list(
    msg = jsonlite::unbox(
      paste0("The message is: '", msg, "'")
    )
  )
}

# ----------------------------------------------------------------------------
# Health endpoint
# ----------------------------------------------------------------------------

#* @get /api/health
#* @serializer json
function() {
  list(
    status = jsonlite::unbox("healthy"),
    model_loaded = jsonlite::unbox(exists("fmodel") && !is.null(fmodel)),
    recipe_loaded = jsonlite::unbox(exists("frecipe") && !is.null(frecipe)),
    model_path = jsonlite::unbox(model_path),
    recipe_path = jsonlite::unbox(recipe_path),
    model_file = jsonlite::unbox(MODEL_FILE),
    recipe_file = jsonlite::unbox(RECIPE_FILE),
    number_raw_predictors = jsonlite::unbox(length(RAW_PREDICTORS))
  )
}

# ----------------------------------------------------------------------------
# Bulk MPI prediction
# ----------------------------------------------------------------------------

#* @param countrycode Country code
#* @param region One or more comma-separated regions
#* @param year Collection date
#* @serializer csv
#* @get /api/mpi
function(country, region, year) {
  if (missing(country) || is.null(country) || country == "") {
    stop("country must be provided.")
  }
  
  if (missing(region) || is.null(region) || region == "") {
    stop("region must be provided.")
  }
  
  if (missing(year) || is.null(year) || year == "") {
    stop("year must be provided.")
  }
  
  region <- tolower(region)
  regions <- trimws(strsplit(region, ",")[[1]])
  regions_pg <- paste0("{", paste(regions, collapse = ","), "}")
  
  message("Country: ", country)
  message("Requested regions: ", region)
  
  conn <- get_db_connection()
  on.exit(DBI::dbDisconnect(conn), add = TRUE)
  
  query <- "
    SELECT DISTINCT ON (region) *
    FROM combined_prep_table
    WHERE country = $1
      AND region = ANY($2)
      AND collectiondate = $3
    ORDER BY region, collectiondate DESC
  "
  
  dfapi <- DBI::dbGetQuery(
    conn,
    query,
    params = list(country, regions_pg, year)
  )
  
  if (nrow(dfapi) == 0) {
    stop(
      paste0(
        "No observations found for countrycode=",
        country,
        ", region=",
        region,
        ", year=",
        year
      )
    )
  }
  
  result_data <- dfapi
  prediction_data <- dfapi
  
  if ("year" %in% RAW_PREDICTORS) {
    prediction_data <- prediction_data %>%
      dplyr::mutate(year = lubridate::year(collectiondate))
  }
  
  prediction_result <- predict_mpi(prediction_data)
  pred_value <- round(prediction_result$predictions, 3)
  
  result_data <- result_data %>%
    dplyr::mutate(
      year = lubridate::year(collectiondate),
      mpi_predicted = pred_value,
      poverty_status = ifelse(
        mpi_predicted >= MPI_THRESHOLD,
        "Poor",
        "Non-Poor"
      )
    )
  
  result_data %>%
    dplyr::select(
      year,
      country,
      countrycode,
      region,
      dplyr::any_of(RAW_PREDICTORS),
      mpi_predicted,
      poverty_status
    )
}

# ----------------------------------------------------------------------------
# Predictor variables
# ----------------------------------------------------------------------------

#* @get /api/predictor_variables
#* @serializer json
function() {
  tryCatch({
    template_data <- frecipe$template
    
    variable_list <- lapply(RAW_PREDICTORS, function(variable_name) {
      if (variable_name %in% names(template_data)) {
        values <- suppressWarnings(
          as.numeric(template_data[[variable_name]])
        )
        
        valid_values <- values[!is.na(values)]
        
        if (length(valid_values) > 0) {
          variable_min <- round(min(valid_values), 4)
          variable_max <- round(max(valid_values), 4)
          variable_mean <- round(mean(valid_values), 4)
          variable_median <- round(median(valid_values), 4)
        } else {
          variable_min <- NA_real_
          variable_max <- NA_real_
          variable_mean <- NA_real_
          variable_median <- NA_real_
        }
      } else {
        variable_min <- NA_real_
        variable_max <- NA_real_
        variable_mean <- NA_real_
        variable_median <- NA_real_
      }
      
      list(
        name = jsonlite::unbox(variable_name),
        type = jsonlite::unbox("numeric"),
        min = jsonlite::unbox(variable_min),
        max = jsonlite::unbox(variable_max),
        mean = jsonlite::unbox(variable_mean),
        median = jsonlite::unbox(variable_median)
      )
    })
    
    list(
      status = jsonlite::unbox("success"),
      n_predictors = jsonlite::unbox(length(variable_list)),
      variables = variable_list
    )
    
  }, error = function(e) {
    list(
      status = jsonlite::unbox("error"),
      message = jsonlite::unbox(conditionMessage(e))
    )
  })
}

# ----------------------------------------------------------------------------
# Single observation prediction
# ----------------------------------------------------------------------------

#* @get /api/predict_single
#* @serializer json
function(req) {
  tryCatch({
    params <- req$argsQuery
    
    input_df <- as.data.frame(
      setNames(
        lapply(RAW_PREDICTORS, function(variable_name) {
          value <- params[[variable_name]]
          
          if (is.null(value) || value == "") {
            stop(
              paste0(
                "Missing value for predictor: ",
                variable_name
              )
            )
          }
          
          numeric_value <- suppressWarnings(as.numeric(value))
          
          if (is.na(numeric_value)) {
            stop(
              paste0(
                "Invalid numeric value for predictor: ",
                variable_name
              )
            )
          }
          
          numeric_value
        }),
        RAW_PREDICTORS
      )
    )
    
    prediction_result <- predict_mpi(input_df)
    pred_value <- round(prediction_result$predictions[1], 3)
    
    poverty_status <- ifelse(
      pred_value >= MPI_THRESHOLD,
      "Poor",
      "Non-Poor"
    )
    
    list(
      mpi_predicted = jsonlite::unbox(pred_value),
      poverty_status = jsonlite::unbox(poverty_status),
      threshold = jsonlite::unbox(MPI_THRESHOLD),
      interpretation = jsonlite::unbox(
        paste0(
          "Predicted MPI of ",
          pred_value,
          " indicates this area is ",
          poverty_status,
          " (threshold: ",
          MPI_THRESHOLD,
          ")"
        )
      )
    )
    
  }, error = function(e) {
    list(
      status = jsonlite::unbox("error"),
      message = jsonlite::unbox(conditionMessage(e))
    )
  })
}

# ----------------------------------------------------------------------------
# Recipe/model debugging
# ----------------------------------------------------------------------------

#* @get /api/debug_recipe
#* @serializer json
function() {
  tryCatch({
    template_cols <- names(frecipe$template)
    var_info <- frecipe$var_info
    step_names <- sapply(frecipe$steps, function(step) class(step)[1])
    
    list(
      model_file = jsonlite::unbox(MODEL_FILE),
      model_path = jsonlite::unbox(model_path),
      model_class = class(fmodel),
      recipe_file = jsonlite::unbox(RECIPE_FILE),
      recipe_path = jsonlite::unbox(recipe_path),
      template_columns = template_cols,
      template_nrow = nrow(frecipe$template),
      raw_predictors = RAW_PREDICTORS,
      number_raw_predictors = jsonlite::unbox(length(RAW_PREDICTORS)),
      var_info_variables = var_info$variable,
      var_info_roles = var_info$role,
      recipe_steps = step_names,
      predictors_missing_from_template = setdiff(
        RAW_PREDICTORS,
        template_cols
      )
    )
    
  }, error = function(e) {
    list(
      status = jsonlite::unbox("error"),
      message = jsonlite::unbox(conditionMessage(e))
    )
  })
}