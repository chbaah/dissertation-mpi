# ============================================================================
# START MULTIDIMENSIONAL POVERTY INDEX PLUMBER API
# ============================================================================

library(plumber)


# ============================================================================
# DETERMINE LOCATION OF THIS SCRIPT
# ============================================================================

args <- commandArgs(
  trailingOnly = FALSE
)

file_arg <- grep(
  "^--file=",
  args,
  value = TRUE
)


if (length(file_arg) > 0) {
  
  RUN_SCRIPT <- normalizePath(
    sub(
      "^--file=",
      "",
      file_arg[1]
    )
  )
  
  SCRIPT_DIR <- dirname(
    RUN_SCRIPT
  )
  
} else {
  
  # Useful when running interactively from RStudio
  SCRIPT_DIR <- getwd()
  
}


# ============================================================================
# CONFIGURATION
# ============================================================================

PLUMBER_HOST <- Sys.getenv(
  "PLUMBER_HOST",
  unset = "0.0.0.0"
)


PLUMBER_PORT <- as.integer(
  Sys.getenv(
    "PLUMBER_PORT",
    unset = "3796"
  )
)


# If PLUMBER_FILE has not been explicitly provided,
# use plumber.R sitting beside run.R.

PLUMBER_FILE <- Sys.getenv(
  "PLUMBER_FILE",
  unset = file.path(
    SCRIPT_DIR,
    "plumber.R"
  )
)


# ============================================================================
# VALIDATION
# ============================================================================

if (!file.exists(PLUMBER_FILE)) {
  
  stop(
    paste0(
      "Plumber API file not found: ",
      PLUMBER_FILE
    )
  )
  
}


# ============================================================================
# STARTUP INFORMATION
# ============================================================================

message("============================================")
message("Starting MPI Plumber API")
message("============================================")

message(
  "Plumber file: ",
  PLUMBER_FILE
)

message(
  "Host: ",
  PLUMBER_HOST
)

message(
  "Port: ",
  PLUMBER_PORT
)


# ============================================================================
# LOAD API
# ============================================================================

pr <- plumber::plumb(
  PLUMBER_FILE
)


# ============================================================================
# START SERVER
# ============================================================================

pr$run(
  
  host = PLUMBER_HOST,
  
  port = PLUMBER_PORT,
  
  swagger = TRUE
  
)