###############################################################################
# Remove all objects
################################################################################
rm(list=ls())
gc()

#######################os installation#######################
# Required for tidyverse and lme4.
# Run from the OS terminal


# sudo apt update

# sudo apt install -y \
# libcurl4-openssl-dev \
# libssl-dev \
# libxml2-dev \
# libfontconfig1-dev \
# libharfbuzz-dev \
# libfribidi-dev \
# libfreetype6-dev \
# libpng-dev \
# libtiff5-dev \
# libjpeg-dev \
# libnlopt-dev


#######################Installation of packages#################################
install.packages("tidyverse")
install.packages("doParallel")
install.packages("remotes")
install.packages("tidymodels")
install.packages("kernlab")
install.packages("devtools")
install.packages("kknn")
install.packages("glmnet")
install.packages("xgboost")
install.packages("ranger")
remotes::install_github("koalaverse/vip")
install.packages("lme4") 
install.packages("bundle")
install.packages("future")
install.packages("doFuture")
install.packages("finetune")
install.packages("betareg")
install.packages("hexbin")
install.packages("reticulate")
install.packages("pdp")


###############################################################################
# Load all packages installed
################################################################################
library(tidyverse)
library(lubridate)
library(readxl)
library(tidymodels)
library(kknn)
library(glmnet)
library(ranger)
library(remotes)
library(xgboost)
library(bundle)
library(parsnip)
library(vip)
library(finetune)
library(lme4)
library(lmtest)
library(rlang)
library(dplyr)
library(recipes)
library(rsample)
library(furrr)
library(yardstick)
library(purrr)
library(tibble)
library(grid)
library(reticulate)
library(keras3)
library(patchwork)
library(hexbin)
library(tidymodels)
library(finetune)
library(dials)
library(doParallel)
library(dials)
library(tibble)
library(dplyr)
library(yardstick)
library(furrr)
library(progressr)
library(readr)
library(betareg)
library(stringr)
library(pdp)
################################################################################



############################# Variables Definition##############################
main_dir <- file.path("/home/ubuntu/Documents/study",
                      "MscDataScience/Dissertation",
                      "Rstudio/MPI")
destdftofile  <- paste0(main_dir, "/", "finaldataframe") 
model_dir     <- paste0(main_dir, "/", "modeldir")

#Plots/Diagram directories
plotsdir <- paste0(main_dir, "/", "plots")

######################Common Section across all algorithms######################

# -------------------------------
# 0) Read dataset from file
# -------------------------------

dff <- readRDS(paste0(destdftofile, "/", "ntl_lcu_osm_data.west.africa.rds"))
df <- dff %>% select(-c(collectiondate,country, countrycode, region))

# ======================================================
# 1) Reproducibility
# ======================================================
RNGkind("L'Ecuyer-CMRG")
set.seed(1238)

# -------------------------------
# 2) Data Split - training and test
# -------------------------------
cat("=== STEP 2: DATA SPLITTING ===\n")
# Data split
data_split <- initial_split(df, prop = 0.70)
ntl_train <- training(data_split)
ntl_test  <- testing(data_split)

# -------------------------------
# 3) Data preprocessing (your recipes)
# -------------------------------
recipetemplate <- recipe(
  mpi ~ ntlmeanintensity + ntlcoverageperc + lcumixedforestperc + lcuclosedshrublandsperc +
    lcuopenshrublandsperc + lcuwoodysavannasperc + lcusavannasperc + lcugrasslandsperc +
    lcucroplandsperc + lcuurbanbuiltupperc + lcucroplandnatvegperc + lcubarrenperc +
    builtschools + builtmedicalfacilities + builtroads + builtresidence +
    lcucroplandnatvegperc + lcubarrenperc,
  data = ntl_train
)

all_pred_notfm.recipe <- recipetemplate %>%
  step_zv(all_predictors(), skip = FALSE)

all_pred_tfmwithrange.recipe <- recipetemplate %>%
  step_zv(all_predictors(), skip = FALSE) %>%
  step_lincomb(all_predictors(), skip = FALSE) %>%
  step_corr(all_predictors(), threshold = 0.75, method = "spearman", skip = FALSE) %>%
  step_range(all_predictors(), min = 0, max = 1, skip = FALSE)

all_pred_tfmwithlogcuberange.recipe <-  recipetemplate %>%
  step_zv(all_predictors(), skip = FALSE) %>%
  step_lincomb(all_predictors(), skip = FALSE) %>%
  step_corr(all_predictors(), threshold = 0.75, method = "spearman", skip = FALSE) %>%
  step_log(c(builtschools, builtmedicalfacilities, builtresidence), offset = 1, base = exp(1),  skip = FALSE) %>%
  step_mutate(across(c(ntlmeanintensity, lcuwoodysavannasperc, lcusavannasperc, lcugrasslandsperc,
                       lcucroplandsperc,builtroads), ~ sign(.) * abs(.)^(1/3)),
              skip = FALSE
  ) %>%
  step_range(all_predictors(), min = 0, max = 1,  skip = FALSE)

all_pred_tfmwithnormalize.recipe <- recipetemplate %>%
  step_zv(all_predictors(), skip = FALSE) %>%
  step_lincomb(all_predictors(), skip = FALSE) %>%
  step_corr(all_predictors(), threshold = 0.75, method = "spearman", skip = FALSE) %>%
  step_normalize(all_predictors(), skip = FALSE)

all_pred_tfmwithlogcubenormalize.recipe <- recipetemplate %>%
  step_zv(all_predictors(), skip = FALSE) %>%
  step_lincomb(all_predictors(), skip = FALSE) %>%
  step_corr(all_predictors(), threshold = 0.75, method = "spearman", skip = FALSE) %>%
  step_log(c(builtschools, builtmedicalfacilities, builtresidence), offset = 1, base = exp(1),  skip = FALSE) %>%
  step_mutate(across(c(ntlmeanintensity, lcuwoodysavannasperc, lcusavannasperc, lcugrasslandsperc,
                       lcucroplandsperc,builtroads), ~ sign(.) * abs(.)^(1/3)),
              skip = FALSE
  ) %>%
  step_normalize(all_predictors(), skip = FALSE)

# -------------------------------
# 4) 10 fold Cross-validation Data Split
# -------------------------------

set.seed(1238)
ntl_vfolds_cv <- vfold_cv(ntl_train, v = 10, repeats = 5)

##################End of Common Section across all algorithms###################

# ============================================================================
# BETA REGRESSION MODEL DEVELOPMENT AND EVALUATION
# This script evaluates beta regression using three preprocessing recipes,
# repeated cross-validation, alternative link functions and optimisation methods.
# It also records model convergence, runtime, test performance and coefficients.
# ============================================================================

#####################New Beta Regression - 25th Feb. 2026#####################
# Create a reproducible seed for each preprocessing recipe.
# The recipe name is converted into an integer value so that the same recipe
# receives the same seed whenever the analysis is repeated.

seed_for_recipe <- function(recipe_name, base = 1238L) {
  as.integer((base + sum(utf8ToInt(paste0("BETAREG_", recipe_name)))) %% .Machine$integer.max)
}

# Create a unique identifier for the current run.
# This is used in the output filenames so that results from separate runs are retained.
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")

cat("========================================================================\n")
cat("BETA REGRESSION ANALYSIS - REPRODUCIBLE VERSION\n")
cat("Run ID:", run_id, "| Start:", as.character(Sys.time()), "| Seed: 1238\n")
cat("========================================================================\n\n")


# Store the preprocessing recipes evaluated for beta regression.
# This allows the same modelling procedure to be applied to each recipe.
preproc.betareg <- list(
  all_pred_notfm = all_pred_notfm.recipe,
  all_pred_tfmwithlogcubenormalize = all_pred_tfmwithlogcubenormalize.recipe,
  all_pred_tfmwithlogcuberange = all_pred_tfmwithlogcuberange.recipe
)

cat("Recipes:", paste(names(preproc.betareg), collapse = ", "), "\n\n")

# -------------------------------
# 4) Hyperparameter Grid
# -------------------------------
cat("=== STEP 4: HYPERPARAMETER GRID ===\n")
param_grid <- tidyr::expand_grid(
  link = c("logit", "probit", "cloglog", "cauchit"),
  link_phi = c("identity", "log", "sqrt"),
  optim.method = c("BFGS", "nlminb")
)
cat("Total configurations:", nrow(param_grid), "\n\n")

# ===============================================================
# 5) TIMING + HELPERS
# ===============================================================
# Create an empty table for recording runtime information.
timing_log <- tibble()

# Helper function used to measure and record execution time for a code block.
# The associated recipe, grid configuration and fold information are also retained.
time_block <- function(label, recipe = NA_character_, grid_id = NA_integer_,
                       link = NA_character_, link_phi = NA_character_,
                       optim.method = NA_character_, fold_id = NA_integer_, expr) {
  t0 <- Sys.time()
  out <- force(expr)
  t1 <- Sys.time()
  timing_log <<- bind_rows(timing_log, tibble(
    timestamp_start = t0, timestamp_end = t1,
    elapsed_sec = as.numeric(difftime(t1, t0, units = "secs")),
    label = label, recipe = recipe, grid_id = grid_id,
    link = link, link_phi = link_phi, optim.method = optim.method, fold_id = fold_id
  ))
  out
}

# ===============================================================
# 6) Safe Guard function for MPI transformation
# ===============================================================
# Beta regression requires the response variable to lie strictly between 0 and 1.
# This safeguard function moves any boundary values slightly inside that interval.
safe_beta_y <- function(y, eps = 1e-6) {
  if (any(y <= 0 | y >= 1, na.rm = TRUE)) y <- pmin(pmax(y, eps), 1 - eps)
  y
}

# Fit one beta regression model using a training fold and generate predictions
# for the corresponding validation fold.
# Convergence information and optimiser iteration counts are also returned.
fit_predict_betareg <- function(train_fold, val_fold, link, link_phi, optim.method,
                                recipe_name, grid_id, fold_id) {
  tryCatch({
  # Convert the processed folds to standard data frames before model fitting.
    train_fold <- as.data.frame(train_fold)
    val_fold   <- as.data.frame(val_fold)
    
    # Stop the fold evaluation if the MPI response column is missing.
    if (!"mpi" %in% names(train_fold) || !"mpi" %in% names(val_fold)) {
      stop("MPI column missing")
    }
    
    # Apply the safeguard function to both the training and validation MPI values.
    train_fold$mpi <- safe_beta_y(train_fold$mpi)
    val_fold$mpi   <- safe_beta_y(val_fold$mpi)
    
    # Retain numeric columns only and keep the validation columns aligned with training.
    numeric_cols <- sapply(train_fold, is.numeric)
    train_fold <- train_fold[, numeric_cols, drop = FALSE]
    val_fold <- val_fold[, names(train_fold), drop = FALSE]
    
    # Stop model fitting when the training-fold MPI has effectively no variation.
    if (sd(train_fold[["mpi"]], na.rm = TRUE) < 1e-8) stop("MPI constant")
    
    # Fit beta regression using the selected mean link, precision link and optimiser.
    # A maximum of 500 optimisation iterations is allowed. (actual iteration is also recorded)
    fit <- betareg::betareg(
      mpi ~ ., 
      data        = train_fold, 
      link        = as.character(link),
      link.phi    = as.character(link_phi), 
      method      = as.character(optim.method),
      control     = betareg::betareg.control(
        maxit    = 500,      # reduced from 5000 — prevents infinite loop
        reltol   = 1e-6,     # convergence tolerance
        start    = NULL      # let betareg choose starting values
      )
    )
    
    # capture optimizer iterations + convergence (mirrors betareg's own summary logic)
    mytail <- function(x) if (length(x) == 0) NA_real_ else as.numeric(x[length(x)])
    
# Extract the number of optimisation iterations used by the fitted model.
# The stored field differs between nlminb and BFGS, so both cases are handled.
    n_iter <- tryCatch({
      if (as.character(optim.method) == "nlminb") {
        val <- fit$optim$iterations
      } else {
        val <- fit$optim$counts        # confirmed field name for your version
        if (is.null(val)) val <- fit$optim$count   # fallback for older betareg versions
        val <- na.omit(val)
      }
      out <- mytail(val)
      if (is.null(out) || length(out) == 0) NA_real_ else out
    }, error = function(e) NA_real_)
    
# Retrieve the convergence status reported by the fitted beta regression model.
    converged <- tryCatch({
      cv <- fit$converged
      if (is.null(cv)) NA else isTRUE(cv)
    }, error = function(e) NA)
    
    
    # Record whether the optimiser reached the maximum iteration limit.    
    hit_maxit <- isTRUE(!is.na(n_iter) && n_iter >= 500)
  
    # Return observed MPI values, predictions and convergence information.
    tibble(mpi = val_fold[["mpi"]], 
           .pred = predict(fit, newdata = val_fold, type = "response"),
           n_iter = n_iter,
           converged = converged,
           hit_maxit = hit_maxit)

  # If model fitting fails, return missing values instead of stopping the full CV run.  
  }, error = function(e) {
    tibble(mpi  = rep(NA_real_, nrow(val_fold)), 
           .pred = rep(NA_real_, nrow(val_fold)),
           n_iter = NA_real_,
           converged = NA,
           hit_maxit = NA)
  })
}

# ===============================================================
# 6) DIAGNOSTICS + SINGLE TEST
# ===============================================================
cat("=== STEP 5: DIAGNOSTICS ===\n")
test_split <- ntl_vfolds_cv$splits[[1]]
cat("Sample fold training:", dim(analysis(test_split)), "\n")
cat("MPI range:", round(range(analysis(test_split)$mpi), 4), "\n\n")

# Run one configuration before starting the full cross-validation.
# This confirms that preprocessing, fitting and prediction work correctly.
cat("=== STEP 6: SINGLE CONFIG TEST ===\n")
recipe_name <- names(preproc.betareg)[1]
cfg <- as.list(param_grid[1, ])
rec <- preproc.betareg[[recipe_name]]
split <- ntl_vfolds_cv$splits[[1]]

# Prepare the selected recipe using the training fold and apply it to both
# the training and validation observations.
prep_rec <- prep(rec, training = analysis(split))
train_proc <- as.data.frame(bake(prep_rec, new_data = analysis(split)))
val_proc <- as.data.frame(bake(prep_rec, new_data = assessment(split)))
train_proc$mpi <- safe_beta_y(train_proc$mpi)
val_proc$mpi <- safe_beta_y(val_proc$mpi)

test_result <- fit_predict_betareg(train_proc, val_proc, cfg$link, cfg$link_phi,
                                   cfg$optim.method, recipe_name, 1, 1)
cat("Test predictions:", sum(!is.na(test_result$.pred)), "/", nrow(test_result), "\n\n")
print(test_result)
cat("Iterations for test config:", test_result$n_iter[1], "| Converged:", test_result$converged[1], "\n\n")

# ===============================================================
# 7) FULL CV RUN
# ===============================================================
# Start the full CV analysis only when the single-configuration test is successful.
if (sum(!is.na(test_result$.pred)) > 0) {
  cat("TEST PASSED - Starting full CV\n")
  cat("========================================================================\n\n")
  
  # Set the random seed and configure parallel processing for the CV evaluation.
  set.seed(1238)
  plan(multisession, workers = parallel::detectCores() - 4)
  
  
  # Record the total runtime for the complete beta regression analysis.
  final_results <- time_block(label = "TOTAL_RUN", expr = {
    all_results <- list()
    
  # Evaluate each preprocessing recipe in turn.
    for (r in names(preproc.betareg)) {
      cat("\n=== RECIPE:", r, "===\n")
      recipe_start <- Sys.time()
      
      # Record the total runtime required for the current recipe.
      recipe_tbl <- time_block(label = "RECIPE_TOTAL", recipe = r, expr = {
        rec <- preproc.betareg[[r]]
        recipe_results <- vector("list", nrow(param_grid))
        
        # Evaluate every hyperparameter configuration for the current recipe.
        for (g in seq_len(nrow(param_grid))) {
          cfg <- as.list(param_grid[g, ])
          cat("  [Grid ", g, "/", nrow(param_grid), "] ", cfg$link, "+", cfg$link_phi, "+", cfg$optim.method, "\n", sep="")
          
          # Record the runtime for the current hyperparameter configuration.
          recipe_results[[g]] <- time_block(
            label = "GRID_TOTAL", recipe = r, grid_id = g,
            link = cfg$link, link_phi = cfg$link_phi, optim.method = cfg$optim.method,
            expr = {
              # Evaluate the current configuration across all CV folds in parallel.
              fold_metrics <- future_map_dfr(seq_along(ntl_vfolds_cv$splits), function(i) {
                tryCatch({
                  # FIX 1: Set deterministic seed per fold
                  # fold_seed <- seed_for_recipe(r) + (g * 1000L) + i
                  # set.seed(fold_seed)
                  
                  time_block(label = "FOLD_TOTAL", recipe = r, grid_id = g, link = cfg$link,
                             link_phi = cfg$link_phi, optim.method = cfg$optim.method, fold_id = i,
                             expr = {
                  # Retrieve the current split, prepare the recipe using the analysis data,
                  # and apply the fitted recipe to the analysis and assessment observations
                               split <- ntl_vfolds_cv$splits[[i]]
                               prep_rec <- prep(rec, training = analysis(split))
                               train_proc <- as.data.frame(bake(prep_rec, new_data = analysis(split)))
                               val_proc <- as.data.frame(bake(prep_rec, new_data = assessment(split)))
                               
                               # Fit the model on the processed training fold and predict the validation fold.
                               preds_tbl <- fit_predict_betareg(train_proc, val_proc, cfg$link, cfg$link_phi,
                                                                cfg$optim.method, r, g, i)
                               
                               # Return missing metrics when no usable predictions are produced.
                               if (all(is.na(preds_tbl$.pred)) || nrow(preds_tbl) == 0) {
                                 return(tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_, accuracy=NA_real_,
                                               sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_,
                                               n_iter=NA_real_, converged=NA, hit_maxit=NA))
                               }
                               
                               # Retain observations with valid observed and predicted MPI values.
                               valid_idx <- !is.na(preds_tbl$mpi) & !is.na(preds_tbl$.pred)
                               if (sum(valid_idx) < 2) {
                                 return(tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_, accuracy=NA_real_,
                                               sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_,
                                               n_iter=NA_real_, converged=NA, hit_maxit=NA))
                               }
                               
                               # Calculate RMSE, MAE and R-squared for the current validation fold.
                               rmse_v <- rmse_vec(preds_tbl$mpi[valid_idx], preds_tbl$.pred[valid_idx])
                               mae_v <- mae_vec(preds_tbl$mpi[valid_idx], preds_tbl$.pred[valid_idx])
                               rsq_v <- rsq_vec(preds_tbl$mpi[valid_idx], preds_tbl$.pred[valid_idx])
                               
                               # Convert observed and predicted MPI values into poor and non-poor classes
                               # using the study poverty threshold of 0.3333.
                               actual <- ifelse(preds_tbl$mpi[valid_idx] >= 0.3333, "poor", "non_poor")
                               pred <- ifelse(preds_tbl$.pred[valid_idx] >= 0.3333, "poor", "non_poor")
                               actual <- factor(actual, levels = c("poor", "non_poor"))
                               pred <- factor(pred, levels = c("poor", "non_poor"))
                               cm <- table(Actual = actual, Predicted = pred)
                               
                               # FIXED - scalar safe
                               # Extract the confusion-matrix counts with the poor class treated as positive.
                               tp <- cm["poor",     "poor"]
                               fn <- cm["poor",     "non_poor"]
                               fp <- cm["non_poor", "poor"]
                               tn <- cm["non_poor", "non_poor"]
                               
                               # Calculate classification metrics while protecting against division by zero.
                               acc  <- (tp + tn) / max(1, sum(cm))
                               sen  <- if ((tp + fn) > 0) tp / (tp + fn) else 0
                               spe  <- if ((tn + fp) > 0) tn / (tn + fp) else 0
                               prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
                               f1   <- if ((prec + sen) > 0) 2 * prec * sen / (prec + sen) else 0
                               
                               # Return regression metrics, classification metrics and convergence information
                               # for the current validation fold.
                               tibble(rmse=rmse_v, mae=mae_v, rsq=rsq_v,
                                      accuracy=acc, sensitivity=sen, specificity=spe, f1=f1,
                                      n_iter = preds_tbl$n_iter[1],
                                      converged = preds_tbl$converged[1],
                                      hit_maxit = preds_tbl$hit_maxit[1])
                               
                             })
                }, error = function(e) {
                  tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_, accuracy=NA_real_,
                         sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_,
                         n_iter=NA_real_, converged=NA, hit_maxit=NA)
                })
                # Use controlled random-number generation for the parallel furrr workers.
              }, .options = furrr_options(seed = 1238L, scheduling = 1, chunk_size = 1))
              
              # Count the number of folds that produced valid RMSE values.
              n_ok <- sum(!is.na(fold_metrics$rmse))
              
              # Summarise performance across the repeated CV folds for this configuration.
              # Mean values, standard errors, fold success and convergence information are retained.
              tibble(recipe = r, grid_id = g, link = cfg$link, link_phi = cfg$link_phi,
                     optim.method = cfg$optim.method,
                     rmse = mean(fold_metrics$rmse, na.rm=T),
                     se_rmse = sd(fold_metrics$rmse, na.rm=T)/sqrt(n_ok),
                     mae = mean(fold_metrics$mae, na.rm=T),
                     se_mae = sd(fold_metrics$mae, na.rm=T)/sqrt(n_ok),
                     rsq = mean(fold_metrics$rsq, na.rm=T),
                     se_rsq = sd(fold_metrics$rsq, na.rm=T)/sqrt(n_ok),
                     accuracy = mean(fold_metrics$accuracy, na.rm=T),
                     sensitivity = mean(fold_metrics$sensitivity, na.rm=T),
                     specificity = mean(fold_metrics$specificity, na.rm=T),
                     f1 = mean(fold_metrics$f1, na.rm=T),
                     se_accuracy = sd(fold_metrics$accuracy, na.rm=T)/sqrt(n_ok),
                     se_sensitivity = sd(fold_metrics$sensitivity, na.rm=T)/sqrt(n_ok),
                     se_specificity = sd(fold_metrics$specificity, na.rm=T)/sqrt(n_ok),
                     se_f1 = sd(fold_metrics$f1, na.rm=T)/sqrt(n_ok),
                     n_successful_folds = n_ok,
                     total_folds = nrow(fold_metrics),
                     success_rate = n_ok / nrow(fold_metrics),
                     # New
                     mean_n_iter   = mean(fold_metrics$n_iter, na.rm = TRUE),
                     max_n_iter    = suppressWarnings(max(fold_metrics$n_iter, na.rm = TRUE)),
                     pct_converged = mean(fold_metrics$converged, na.rm = TRUE),
                     pct_hit_maxit = mean(fold_metrics$hit_maxit, na.rm = TRUE))
            })
        }
        bind_rows(recipe_results)
      })
      
      # Report the elapsed time for the completed preprocessing recipe.
      recipe_elapsed <- as.numeric(difftime(Sys.time(), recipe_start, units = "secs"))
      cat("  Recipe completed in", round(recipe_elapsed, 2), "sec (", round(recipe_elapsed/60, 2), "min)\n")
      all_results[[r]] <- recipe_tbl
    }
    bind_rows(all_results)
  })
  
  
  # Return processing to sequential mode after the parallel CV stage.
  plan(sequential)
  
  # Save the complete cross-validation results for all recipes and configurations.
  cv_file <- file.path(model_dir, "betareg", paste0("beta_regression_complete_results_", run_id, ".csv"))
  write_csv(final_results, cv_file)
  cat("\n CV results:", cv_file, "\n")
  
  # Save the detailed timing log collected during the analysis.
  # Save timing
  timing_file <- file.path(model_dir, "betareg", paste0("timing_log_", run_id, ".csv"))
  write_csv(timing_log, timing_file)
  cat(" Timing log:", timing_file, "\n")
  
  # Timing summary
  cat("\n=== TIMING SUMMARY ===\n")
  print(timing_log %>% group_by(label) %>%
          summarise(n=n(), total_sec=sum(elapsed_sec, na.rm=T),
                    avg_sec=mean(elapsed_sec, na.rm=T), .groups="drop") %>%
          arrange(desc(total_sec)))
  
  # Report the total number of configurations and those that completed all folds.
  cat("\n=== RESULTS SUMMARY ===\n")
  cat("Total configs:", nrow(final_results), "| Successful:", sum(final_results$success_rate == 1), "\n\n")
  
  # ===============================================
  # 8) BEST CONFIGURATIONS
  # ===============================================
  cat("========================================================================\n")
  cat("STEP 8: BEST CONFIGURATIONS\n")
  cat("========================================================================\n")
  
  # Select the overall best configuration using the lowest CV RMSE among
  # configurations that successfully completed every fold.
  best_config <- final_results %>% filter(success_rate == 1) %>% arrange(rmse) %>% slice(1)
  cat("\n=== BEST OVERALL ===\n")
  print(best_config %>% select(recipe, link, link_phi, optim.method, rmse, mae, rsq, f1))
  
  # Select the best performing configuration separately for each recipe.
  best_by_recipe <- final_results %>% filter(success_rate == 1) %>%
    group_by(recipe) %>% arrange(rmse) %>% slice(1) %>% ungroup()
  cat("\n=== BEST PER RECIPE ===\n")
  print(best_by_recipe %>% select(recipe, link, link_phi, optim.method, rmse, mae, rsq, f1))
  
  # Save the best configuration identified for each preprocessing recipe.
  best_file <- file.path(model_dir, "betareg", paste0("best_by_recipe_", run_id, ".csv"))
  write_csv(best_by_recipe, best_file)
  cat("\n Best configs:", best_file, "\n")
  
  # ===============================================
  # 9) FINAL MODELS + TEST PREDICTION
  # ===============================================
  cat("\n========================================================================\n")
  cat("STEP 9: FINAL MODELS + TEST PREDICTION\n")
  cat("========================================================================\n")
  
  # Define the output files for final test predictions and test performance metrics.
  pred_file <- file.path(model_dir, "betareg", paste0("betareg_test_predictions_", run_id, ".csv"))
  metric_file <- file.path(model_dir, "betareg", paste0("betareg_test_metrics_", run_id, ".csv"))
  if (file.exists(pred_file)) file.remove(pred_file)
  if (file.exists(metric_file)) file.remove(metric_file)
  
  # FIXED - consistent with CV fold approach
  # Helper function for calculating classification metrics on the independent test set.
  # The poor class is treated as the positive class using the 0.3333 threshold.
  calc_class_metrics <- function(truth, pred, threshold = 0.3333) {
    actual <- factor(ifelse(truth >= threshold, "poor", "non_poor"),
                     levels = c("poor", "non_poor"))  # poor = positive
    predicted <- factor(ifelse(pred >= threshold, "poor", "non_poor"),
                        levels = c("poor", "non_poor"))
    
    cm <- table(Actual = actual, Predicted = predicted)
    
    # Extract confusion-matrix values by class name.
    # Named extraction — robust to empty levels
    tp <- cm["poor",     "poor"]
    fn <- cm["poor",     "non_poor"]
    fp <- cm["non_poor", "poor"]
    tn <- cm["non_poor", "non_poor"]
    
    # Calculate accuracy, sensitivity, specificity, precision and F1-score.
    acc  <- (tp + tn) / max(1, sum(cm))
    sen  <- if ((tp + fn) > 0) tp / (tp + fn) else 0   # sensitivity: poor recall
    spe  <- if ((tn + fp) > 0) tn / (tn + fp) else 0   # specificity: non_poor recall
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0   # precision: poor PPV
    f1   <- if ((prec + sen) > 0) 2 * prec * sen / (prec + sen) else 0
    
    tibble(tp=tp, tn=tn, fp=fp, fn=fn,
           accuracy=acc, sensitivity=sen,
           specificity=spe, precision=prec, f1=f1)
  }
  
  # Store the independent test metrics from the best model under each recipe.
  all_test_metrics <- list()
  
  # Fit and evaluate the best CV configuration from each recipe using the
  # complete training dataset and the independent test dataset.
  for (k in seq_len(nrow(best_by_recipe))) {
    cfg <- as.list(best_by_recipe[k, ])
    rname <- cfg$recipe
    
    cat("\n=== RECIPE:", rname, "===\n")
    cat("Config:", cfg$link, "+", cfg$link_phi, "+", cfg$optim.method, "\n")
    
    # Record final-model training time and use the deterministic recipe-specific seed.
    final_start <- Sys.time()
    set.seed(seed_for_recipe(rname))
    
    # Prepare the selected recipe on the full training data and apply the same
    # fitted transformations to the independent test data.
    rec <- preproc.betareg[[rname]]
    prep_rec <- prep(rec, training = ntl_train)
    train_proc <- as.data.frame(bake(prep_rec, new_data = ntl_train))
    test_proc <- as.data.frame(bake(prep_rec, new_data = ntl_test))
    
    # Retain the original MPI values for evaluation before applying the boundary safeguard.
    train_proc$mpi_raw <- train_proc$mpi
    test_proc$mpi_raw <- test_proc$mpi
    train_proc$mpi <- safe_beta_y(train_proc$mpi)
    test_proc$mpi <- safe_beta_y(test_proc$mpi)
    
    cat("Training final model (seed=", seed_for_recipe(rname), ")...\n", sep="")
    # Fit the final beta regression model for the current preprocessing recipe.
    final_model <- betareg::betareg(mpi ~ . - mpi_raw, data = train_proc,
                                    link = as.character(cfg$link),
                                    link.phi = as.character(cfg$link_phi),
                                    method = as.character(cfg$optim.method),
                                    control = betareg::betareg.control(maxit = 500, reltol=1e-6))
    
    # Extract the optimiser iteration count from the final fitted model.
    mytail <- function(x) x[length(x)]
    final_n_iter <- tryCatch({
      if (as.character(cfg$optim.method) == "nlminb") {
        val <- final_model$optim$iterations
      } else {
        val <- final_model$optim$counts
        if (is.null(val)) val <- final_model$optim$count
        val <- na.omit(val)
      }
      out <- mytail(val)
      if (is.null(out) || length(out) == 0) NA_real_ else out
    }, error = function(e) NA_real_)
    
    # Record final convergence status and report whether the maximum iteration
    # limit was reached.
    final_converged <- isTRUE(final_model$converged)
    cat("Iterations used:", final_n_iter, "| Converged:", final_converged,
        if (!is.na(final_n_iter) && final_n_iter >= 500) " HIT MAXIT" else "", "\n")
    
    
    # Calculate the final model training time.
    final_elapsed <- as.numeric(difftime(Sys.time(), final_start, units = "secs"))
    cat("Training time:", round(final_elapsed, 2), "sec\n")
    
    # Generate MPI predictions for the independent test dataset.
    # Predictions
    test_pred <- as.numeric(predict(final_model, newdata = test_proc, type = "response"))
    
    # Store the model configuration, observed MPI values and predicted MPI values.
    pred_tbl <- tibble(recipe = rname, link = cfg$link, link_phi = cfg$link_phi,
                       optim.method = cfg$optim.method, grid_id = cfg$grid_id,
                       final_train_time_secs = final_elapsed,
                       mpi_raw = test_proc$mpi_raw, mpi_fit = test_proc$mpi, .pred = test_pred)
    
    write_csv(pred_tbl, pred_file, append = file.exists(pred_file))
    
    # Calculate regression and classification performance using valid test observations.
    valid_idx <- is.finite(pred_tbl$mpi_raw) & is.finite(pred_tbl$.pred)
    rmse_v <- rmse_vec(pred_tbl$mpi_raw[valid_idx], pred_tbl$.pred[valid_idx])
    mae_v <- mae_vec(pred_tbl$mpi_raw[valid_idx], pred_tbl$.pred[valid_idx])
    rsq_v <- rsq_vec(pred_tbl$mpi_raw[valid_idx], pred_tbl$.pred[valid_idx])
    class_tbl <- calc_class_metrics(pred_tbl$mpi_raw[valid_idx], pred_tbl$.pred[valid_idx])
    
    # Combine the test metrics with runtime and convergence information.
    metric_tbl <- tibble(recipe = rname, link = cfg$link, link_phi = cfg$link_phi,
                         optim.method = cfg$optim.method, grid_id = cfg$grid_id,
                         final_train_time_secs = final_elapsed,
                         n_test = sum(valid_idx), rmse = rmse_v, mae = mae_v, rsq = rsq_v,
                         n_iter = final_n_iter, converged = final_converged) %>%
      bind_cols(class_tbl)
    
    write_csv(metric_tbl, metric_file, append = file.exists(metric_file))
    all_test_metrics[[rname]] <- metric_tbl
    
    cat("RMSE:", round(rmse_v, 4), "| R²:", round(rsq_v, 4), "\n")
    
    # Extract coefficient results for the mean and precision components of the model.
    # Save model coefficients
    smry <- summary(final_model)
    coef_mean <- as.data.frame(smry$coefficients$mean) %>%
      rownames_to_column("parameter") %>% mutate(component = "mean", recipe = rname)
    coef_phi <- as.data.frame(smry$coefficients$precision) %>%
      rownames_to_column("parameter") %>% mutate(component = "precision", recipe = rname)
    
    # Combine coefficient results, rename columns and assign significance symbols
    # from the coefficient p-values.
    coef_table <- bind_rows(coef_mean, coef_phi) %>%
      rename(estimate = `Estimate`, std_error = `Std. Error`,
             z_value = `z value`, p_value = `Pr(>|z|)`) %>%
      mutate(signif = case_when(p_value < 0.001 ~ "***", p_value < 0.01 ~ "**",
                                p_value < 0.05 ~ "*", p_value < 0.10 ~ ".", TRUE ~ "")) %>%
      select(recipe, component, parameter, estimate, std_error, z_value, p_value, signif)
    
    # Save the coefficient table for the current preprocessing recipe.
    write_csv(coef_table, file.path(model_dir, "betareg",
                                    paste0("betareg_coefficients_", rname, "_", run_id, ".csv")))
  }
  
  cat("\n Predictions:", pred_file, "\n")
  cat(" Test metrics:", metric_file, "\n")
  
  # Combine and display independent test metrics across all recipes.
  test_summary <- bind_rows(all_test_metrics)
  cat("\n=== FINAL TEST METRICS ===\n")
  print(test_summary %>% select(recipe, rmse, mae, rsq, accuracy, f1))
  
  # Refit and save the single overall best beta regression model selected by CV.
  # Save overall best model
  best_recipe <- best_config$recipe
  best_rec <- preproc.betareg[[best_recipe]]
  prep_best <- prep(best_rec, training = ntl_train)
  train_best <- as.data.frame(bake(prep_best, new_data = ntl_train))
  train_best$mpi <- safe_beta_y(train_best$mpi)
  
  set.seed(seed_for_recipe(best_recipe))
  final_best_model <- betareg::betareg(mpi ~ ., data = train_best,
                                       link = as.character(best_config$link),
                                       link.phi = as.character(best_config$link_phi),
                                       method = as.character(best_config$optim.method),
                                       control = betareg::betareg.control(maxit = 500, reltol=1e-6))
  
  saveRDS(final_best_model, file.path(model_dir, "betareg",
                                      paste0("final_beta_regression_model_", run_id, ".rds")))
  cat("\n Final model saved\n")
  
} else {
  cat("SINGLE CONFIG TEST FAILED\n")
  plan(sequential)
}

# Save the R session and random-number settings used for the analysis.
# This provides a reproducibility record for the completed run.
# ===============================================================
# 10) SESSION INFO
# ===============================================================
session_file <- file.path(model_dir, "betareg", paste0("sessionInfo_", run_id, ".txt"))
sink(session_file)
cat("Beta Regression Analysis - Session Information\n")
cat("===============================================\n")
cat("Run ID:", run_id, "\n")
cat("Date:", as.character(Sys.time()), "\n")
cat("R Version:", R.version.string, "\n\n")
cat("RNG Kind:\n")
print(RNGkind())
cat("\nBase seed: 1238\n")
cat("Recipe seed function: seed_for_recipe(recipe_name)\n\n")
cat("===============================================\n\n")
sessionInfo()
sink()

# Confirm completion of the beta regression analysis.
cat("\n========================================================================\n")
cat(" ANALYSIS COMPLETE\n")
cat(" Session info saved:", session_file, "\n")
cat("========================================================================\n")
  
 


# ============================================================================
# BETA REGRESSION PLOTS AND DIAGNOSTIC ANALYSIS
# This section creates the diagnostic, residual, variable-importance, prediction
# and poverty-classification plots used to assess the final beta regression model.
# Training and independent test results are presented separately where required.
# ============================================================================

# ===============================================
# Final Model Residual Analysis (CORRECT APPROACH)
# ===============================================

cat("\n=== FINAL MODEL RESIDUAL ANALYSIS (Training Data) ===\n")

# Extract the residuals and fitted MPI values from the final beta regression model
# using the training dataset.
# Get residuals and fitted values from the final model on TRAINING data
final_model_residuals <- residuals(final_best_model)
final_model_fitted <- fitted(final_best_model)
train_mpi <- train_proc$mpi
test_mpi <- test_proc$mpi

# Create comprehensive residual analysis data
# final_model_residuals / sd(final_model_residuals)

# Create a structured training residual dataset containing fitted values,
# observed MPI, raw residuals and standardised residuals.
betareg_train_resid_data <- tibble(
  model = "betareg",
  Set = "Train",
  .fitted = as.numeric(final_model_fitted),
  .resid = as.numeric(final_model_residuals),
  mpi = as.numeric(train_mpi),
  .std_resid =   as.numeric(scale(final_model_residuals)) # Standardized residuals
)


# Apply the fitted preprocessing recipe to the independent test dataset and
# ensure that the MPI values remain within the interval required by beta regression.
test_proc <- as.data.frame(bake(prep_best, new_data = ntl_test))
test_proc$mpi <- safe_beta_y(test_proc$mpi)

# Generate predicted MPI values for the independent test observations.
test_pred <- as.numeric(predict(final_model, newdata = test_proc, type = "response"))

# Calculate test residuals as observed MPI minus predicted MPI.
betareg_resid_test <- as.numeric(test_proc$mpi) - test_pred

# Create the corresponding residual dataset for the independent test observations.
test_resid_data_betareg <- tibble(
  model = "betareg",
  Set = "Test",
  .fitted = test_pred,
  .resid  = as.numeric(betareg_resid_test),
  mpi     = as.numeric(test_proc$mpi),
  .std_resid = as.numeric(scale(betareg_resid_test)) # Standardized residuals
)

# Save the test residual dataset for use in later model comparison and bias analysis.
# Write the test tibble to file
file_name_betareg <- file.path(model_dir, "betareg", paste0("test_resid_data_betareg", ".csv"))
write_csv(test_resid_data_betareg, file_name_betareg)

# Save CV results
cat("\n CV results:", file_name_betareg, "\n")

#=========================Plots====================
# Plot training residuals against fitted MPI values.
# The zero reference line and LOESS smoother help identify systematic residual
# patterns and changes in residual spread across the fitted MPI range.
# 1. Residuals vs Fitted Values (Homoscedasticity Check)
p1 <- ggplot(betareg_train_resid_data, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Fitted Values",
    subtitle = "Final Model on Training Data - Check Homoscedasticity",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals (Actual - Predicted)"
  ) +
  theme_minimal()

# Create a scale-location plot using the standardised residuals.
# This provides an additional visual check of whether residual spread changes
# across the fitted values.
# 2. Scale-Location Plot (Spread of residuals)
p2 <- ggplot(betareg_train_resid_data, aes(x = .fitted, y = sqrt(abs(.std_resid)))) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  labs(
    title = "Scale-Location Plot",
    subtitle = "Check for Constant Variance (Homoscedasticity)",
    x = "Fitted Values",
    y = "√|Standardized Residuals|"
  ) +
  theme_minimal()

# Examine the distribution of the training residuals.
# The histogram and density curve are compared with a normal density curve.
# 3. Residuals Distribution

p3 <- ggplot(betareg_train_resid_data, aes(x = .resid)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 30,
    fill = "lightblue",
    color = "black",
    alpha = 0.7
  ) +
  geom_density(color = "darkred", linewidth = 1) +
  stat_function(
    fun = dnorm,
    args = list(
      mean = mean(betareg_train_resid_data$.resid, na.rm = TRUE),
      sd   = sd(betareg_train_resid_data$.resid, na.rm = TRUE)
    ),
    color = "blue",
    linetype = "dashed"
  ) +
  labs(
    title = "Distribution of Residuals",
    subtitle = "Blue dashed line shows normal distribution for comparison",
    x = "Residuals",
    y = "Density"
  ) +
  theme_minimal()

# Create a normal Q-Q plot to compare the residual quantiles with those expected
# under a normal distribution.
# 4. Q-Q Plot for Normality
p4 <- ggplot(betareg_train_resid_data, aes(sample = .resid)) +
  stat_qq(color = "darkblue", alpha = 0.6) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
    title = "Normal Q-Q Plot of Residuals",
    subtitle = "Check for Normality - Points should follow the red line",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  theme_minimal()

# Combine the four training diagnostic plots into a single 2-by-2 panel.
# Combine all diagnostic plots
#title = "Beta Regression Diagnostic Plots - Final Model on Training Data",

betareg_diagnostic_plots <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    
    # Add the selected preprocessing recipe and beta regression link information
    # to the combined diagnostic plot.
    subtitle = paste("Model:", best_config$recipe, 
                     "| Link:", best_config$link, 
                     "| Precision Link:", best_config$link_phi),
    caption = paste("Training observations:", nrow(betareg_train_resid_data))
  )

# Save the combined training diagnostic figure at publication-quality resolution.
# Save diagnostic plots
ggsave(paste0(plotsdir, "/", "betareg", "/", "final_model_diagnostic_plots_betareg.png"), betareg_diagnostic_plots, 
       width = 14, height = 10, dpi = 300)

cat("Final model diagnostic plots saved to: final_model_diagnostic_plots.png\n")

# ===============================================
# Detailed Residual Statistics - not used
# ===============================================

# Calculate descriptive statistics for the training residuals, including the
# residual mean, spread, RMSE, MAE and percentage of predictions within
# selected absolute-error ranges.
cat("\n=== FINAL MODEL RESIDUAL STATISTICS ===\n")
resid_stats <- betareg_train_resid_data %>%
  summarise(
    Observations = n(),
    Mean_Residual = round(mean(.resid, na.rm = TRUE), 6),
    SD_Residual = round(sd(.resid, na.rm = TRUE), 4),
    Min_Residual = round(min(.resid, na.rm = TRUE), 4),
    Max_Residual = round(max(.resid, na.rm = TRUE), 4),
    MSE = round(mean(.resid^2, na.rm = TRUE), 6),
    RMSE = round(sqrt(mean(.resid^2, na.rm = TRUE)), 4),
    MAE = round(mean(abs(.resid), na.rm = TRUE), 4),
    `Within_±0.05` = paste0(round(mean(abs(.resid) <= 0.05) * 100, 1), "%"),
    `Within_±0.10` = paste0(round(mean(abs(.resid) <= 0.10) * 100, 1), "%")
  )

print(resid_stats)

# ===============================================
# Homoscedasticity Test - not used
# ===============================================

cat("\n=== HOMOSCEDASTICITY ASSESSMENT ===\n")

# Apply the Breusch-Pagan test as an additional assessment of whether the
# residual variance changes systematically with the fitted values.
# Breusch-Pagan test for heteroscedasticity
# Install if needed: install.packages("lmtest")
library(lmtest)

# Fit the auxiliary linear model required for the Breusch-Pagan test.
# Create a linear model of residuals vs fitted for testing
lm_resid <- lm(.resid ~ .fitted, data = betareg_train_resid_data)
bp_test <- bptest(lm_resid)

cat("Breusch-Pagan Test for Heteroscedasticity:\n")
cat("BP =", round(bp_test$statistic, 4), 
    ", p-value =", format.pval(bp_test$p.value, digits = 4), "\n")

# Interpret the Breusch-Pagan p-value using the 5 percent significance level.
if (bp_test$p.value < 0.05) {
  cat("Significant evidence of heteroscedasticity (p < 0.05)\n")
  cat("   Residual variance appears non-constant\n")
} else {
  cat("No significant evidence of heteroscedasticity (p >= 0.05)\n")
  cat("   Residual variance appears constant (homoscedastic)\n")
}

# ===============================================
# Additional: Residuals vs Actual Values
# ===============================================

# Plot training residuals against the observed MPI values.
# The LOESS smoother is used to examine possible systematic prediction bias
# across the MPI distribution.
p5 <- ggplot(betareg_train_resid_data, aes(x = mpi, y = .resid)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Actual - Train Data",
    subtitle = "Check for systematic bias across MPI range",
    x = "Actual MPI",
    y = "Residuals"
  ) +
  theme_minimal()

# Save the training residuals-versus-actual MPI plot.
ggsave(paste0(plotsdir, "/", "betareg", "/", "train_residuals_vs_actual_betareg.png"), p5, width = 10, height = 6, dpi = 300)
cat("Residuals vs actual values plot saved to: residuals_vs_actual.png\n")

# ===============================================
# Model Assumptions Summary
# ===============================================

# Print a short reminder of the diagnostic features examined in the plots.
cat("\n=== MODEL ASSUMPTIONS SUMMARY ===\n")
cat("1. LINEARITY: Check residuals vs fitted plot for random scatter around zero\n")
cat("2. HOMOSCEDASTICITY: Check constant spread of residuals in scale-location plot\n")
cat("3. INDEPENDENCE: Residuals should show no autocorrelation patterns\n")
cat("4. DISTRIBUTION: Residuals should be approximately normally distributed\n")

# Calculate the correlation between fitted values and residuals as an additional
# descriptive check for systematic association.
# Check for patterns in residuals
resid_cor <- cor(betareg_train_resid_data$.fitted, betareg_train_resid_data$.resid)
cat("Correlation between fitted and residuals:", round(resid_cor, 6), "\n")
cat("(Should be close to 0 for well-specified models)\n")

#Extract the actual test mpi data
#actual_test <- test_processed_raw %>% select("mpi")
#test_pred <- as.numeric(predict(final_model, newdata = test_proc, type = "response"))



# Create the residuals-versus-fitted plot for the independent test dataset.
# This is used to examine generalisation and residual behaviour on unseen data.
# Residuals vs Fitted (Test)
#title = "Residuals vs Fitted (Test Data)",
#subtitle = "Check generalization and bias on unseen data",
testfr <- ggplot(test_resid_data_betareg, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Fitted - Test Data",
    subtitle = "Check for generalisation and systematic bias on unseen data",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals"
  ) +
  theme_minimal()

# Save the independent test residual plot.
ggsave(paste0(plotsdir, "/", "betareg", "/","test_residuals_vs_fitted_betareg.png"), testfr, width = 10, height = 6, dpi = 300)
cat("Residuals vs Actual plot saved as: residuals_vs_fitted_betareg.png\n")

# Combine the training residuals-versus-actual plot and the test
# residuals-versus-fitted plot into one figure for comparison.
diagp5testfr_plots <- (p5 | testfr) +
  plot_annotation(
    
    caption = paste("Residual vs actual (Train) and Residual vs Fitted (Test)")
  )

# Save the combined residual figure.
# Save combined plots
ggsave(
  filename = paste0(plotsdir, "/", "betareg", "/", "betareg_Residual_plots.png"),
  plot = diagp5testfr_plots,
  width = 14,
  height = 10,
  dpi = 300
)
cat("Diagnostic plots saved as: rfResidual_plots.png\n")



# ===============================================
# Variable Importance Analysis
# ===============================================

# Begin the beta regression variable-importance analysis.
cat("\n=== VARIABLE IMPORTANCE ANALYSIS ===\n")

# Use the magnitude of the fitted mean-model coefficients as the basis for
# constructing the beta regression variable-importance summary.
# Method 1: Coefficient Magnitude (Standardized)
# Since beta regression coefficients are on the link function scale,
# we can use absolute coefficient values as a measure of importance

# Extract the coefficient table from the mean component of the final model.
# Extract coefficients from the final model
model_summary <- summary(final_model)
coefficients_df <- as.data.frame(model_summary$coefficients$mean)

# Remove the intercept, calculate absolute coefficient magnitude, scale the
# resulting values to the interval 0 to 1 and retain coefficient significance.
# Clean up variable names and create importance data
var_importance <- coefficients_df %>%
  rownames_to_column("variable") %>%
  filter(variable != "(Intercept)") %>%  # Remove intercept
  rename(
    estimate   = Estimate,
    std_error  = `Std. Error`
  ) %>%
  mutate(
    abs_estimate = abs(estimate),
    importance   = abs_estimate / max(abs_estimate),
    significance = case_when(
      `Pr(>|z|)` < 0.001 ~ "***",
      `Pr(>|z|)` < 0.01  ~ "**",
      `Pr(>|z|)` < 0.05  ~ "*",
      TRUE               ~ ""
    )
  ) %>%
  arrange(desc(abs_estimate))


# Display the ten predictors with the largest coefficient magnitudes.
cat("\nTop 10 Most Important Variables by Coefficient Magnitude:\n")
print(var_importance %>% head(10) %>% select(variable, estimate, std_error, `Pr(>|z|)`, importance))

# Create a horizontal variable-importance plot for the fifteen highest-ranked
# predictors and display significance symbols above the bars.
# Create variable importance plot
# title = "Variable Importance – Beta Regression",

var_imp_plot <- ggplot(
  var_importance %>% head(15),
  aes(x = reorder(variable, importance), y = importance)
) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  geom_text(
    aes(label = significance, y = importance + 0.02),
    size = 4, hjust = 0
  ) +
  coord_flip() +
  labs(
    subtitle = "Based on absolute standardized coefficients",
    x = "Predictor variables",
    y = "Relative importance (0–1 scale)",
    caption = "Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none"
  )

# Save the coefficient-based variable-importance plot.
ggsave(
  file.path(plotsdir, "betareg", "variable_importance_coefficients_betareg.png"),
  var_imp_plot,
  width = 12,
  height = 8,
  dpi = 300
)

cat("Variable importance plot (coefficients) saved\n")

# Create a second version of the variable-importance plot without
# coefficient significance symbols.
##########Variable importance plot without stars#######
nopvalvar_imp_plot <- ggplot(
  var_importance %>% head(15),
  aes(x = reorder(variable, importance), y = importance)
) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  coord_flip() +
  labs(
    subtitle = "Based on absolute standardized coefficients",
    x = "Predictor variables",
    y = "Relative importance (0–1 scale)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "none"
  )

# Save the variable-importance plot without significance symbols.
ggsave(
  file.path(plotsdir, "betareg", "variable_importance_coefficients_withoup_betareg.png"),
  nopvalvar_imp_plot,
  width = 12,
  height = 8,
  dpi = 300
)
######################################################




# ===============================================
# Prediction vs Actual Scatterplots
# ===============================================
# Create comprehensive residual analysis data

# Define common axis limits so the predicted-versus-actual plots can be
# displayed on a comparable scale.
# Global limits across BOTH sets
#x_lim <- range(c(train_actual, test_actual), na.rm = TRUE)
#y_lim <- range(c(final_model_fitted, test_pred), na.rm = TRUE)


x_lim <- range(c(test_resid_data_betareg$mpi, test_resid_data_betareg$mpi), na.rm = TRUE)
y_lim <- range(c(test_resid_data_betareg$.fitted, test_resid_data_betareg$.fitted), na.rm = TRUE)


cat("\n=== CREATING PREDICTION VS ACTUAL SCATTERPLOTS ===\n")

# 1. Training Data Scatterplot
cat("\nCreating training data scatterplot...\n")

# train_predictions <- predict(final_model, newdata = train_processed, type = "response")
#train_actual <- train_processed_raw$mpi

#train_scatter_data <- tibble(
#  Actual = train_actual,
#  Predicted = final_model_fitted,
#  Set = "Training"
#)

# Calculate training RMSE, MAE, R-squared and correlation for annotation
# on the predicted-versus-actual training plot.
# Calculate training metrics for annotation
train_rmse <- rmse_vec(betareg_train_resid_data$mpi, betareg_train_resid_data$.fitted)
train_mae <- mae_vec(betareg_train_resid_data$mpi, betareg_train_resid_data$.fitted)
train_rsq <- rsq_vec(betareg_train_resid_data$mpi, betareg_train_resid_data$.fitted)
train_cor <- cor(betareg_train_resid_data$mpi, betareg_train_resid_data$.fitted)

# Plot predicted MPI against observed MPI for the training dataset.
# The 45-degree line represents perfect prediction, while the fitted linear
# trend provides a visual summary of agreement between predictions and observations.
# Create training scatterplot
train_scatter <- ggplot(betareg_train_resid_data, aes(x = mpi, y = .fitted)) +
  geom_point(alpha = 0.6, color = "steelblue", size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
  coord_equal(xlim = x_lim, ylim = y_lim, expand = FALSE) +
  labs(
    title = "Predicted vs Actual MPI",
    subtitle = "Training Data",
    x = "Actual MPI",
    y = "Predicted MPI"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12)
  ) +
  # Add the main training performance metrics directly to the plot.
  annotate("text",
           x = x_lim[1] + 0.02 * diff(x_lim),  # 2% from left edge of x axis
           y = y_lim[2] - 0.02 * diff(y_lim),  # 2% from top edge of y axis
           hjust = 0, vjust = 1,
           label = paste(
             sprintf("RMSE = %.4f", train_rmse),
             sprintf("MAE  = %.4f", train_mae),
             sprintf("R²   = %.4f", train_rsq),
             sprintf("Corr = %.4f", train_cor),
             sep = "\n"),
           size = 4, color = "darkgreen", fontface = "bold")

# Save the predicted-versus-actual training plot.
ggsave(paste0(plotsdir, "/", "betareg", "/", "train_prediction_vs_actual_betareg.png"), 
       train_scatter, width = 10, height = 8, dpi = 300)
cat("Training data scatterplot saved\n")

# 2. Test Data Scatterplot (if test data has actual MPI values)

# ------------------------------------------------
# Predicted vs Actual MPI - Test Data (MATCHED)
# ------------------------------------------------

#test_actual <- test_processed_raw$mpi    # raw (unclipped) MPI
#test_pred   <- test_pred                 # model predictions (response scale)

# Build plotting data
#test_scatter_data <- tibble(
#  Actual    = test_actual,
#  Predicted = test_pred,
#  Set = "Test"
#)

# Calculate the corresponding performance metrics for the independent test dataset.
# Calculate test metrics
test_rmse <- rmse_vec(test_resid_data_betareg$mpi, test_resid_data_betareg$.fitted)
test_mae  <- mae_vec(test_resid_data_betareg$mpi, test_resid_data_betareg$.fitted)
test_rsq  <- rsq_vec(test_resid_data_betareg$mpi, test_resid_data_betareg$.fitted)
test_cor  <- cor(test_resid_data_betareg$mpi, test_resid_data_betareg$.fitted)

# Create the predicted-versus-actual plot for the independent test dataset
# using the same general structure as the training plot.
# Create test scatterplot (same structure as training)
test_scatter <- ggplot(test_resid_data_betareg, aes(x = mpi, y = .fitted)) +
  geom_point(alpha = 0.6, color = "purple", size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "red",
              linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", color = "darkorange",
              se = TRUE, linewidth = 0.8) +
  coord_equal(xlim = x_lim, ylim = y_lim, expand = FALSE) +
  labs(
    title = "Predicted vs Actual MPI",
    subtitle = "Test Data",
    x = "Actual MPI",
    y = "Predicted MPI"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12)
  ) +
  # Add the independent test performance metrics directly to the plot.
  annotate(
    "text",
    x = min(test_resid_data_betareg$mpi),
    y = max(test_resid_data_betareg$.fitted),
    hjust = 0, vjust = 1,
    label = paste(
      sprintf("RMSE = %.4f", test_rmse),
      sprintf("MAE = %.4f", test_mae),
      sprintf("R² = %.4f", test_rsq),
      sprintf("Corr = %.4f", test_cor),
      sep = "\n"
    ),
    size = 4,
    color = "darkred",
    fontface = "bold"
  )

# Save the predicted-versus-actual test plot.
# Save
ggsave(
  paste0(plotsdir, "/betareg/test_prediction_vs_actual_betareg.png"),
  test_scatter,
  width = 10, height = 8, dpi = 300
)

cat("Test data scatterplot saved\n")



# Create the poverty-classification plot for the training dataset.
# 5a. Scatterplot with Poverty Classification - Training Dataset
cat("\nCreating scatterplot with poverty classification...\n")

# Classify each training observation according to the observed and predicted
# MPI values and identify correct and incorrect poverty classifications.
poverty_scatter_data <- betareg_train_resid_data %>%
  mutate(
    Actual_Poverty = ifelse(mpi > 0.3333, "Poor", "Non-Poor"),
    Predicted_Poverty = ifelse(.fitted > 0.3333, "Poor", "Non-Poor"),
    Classification = case_when(
      Actual_Poverty == "Poor" & Predicted_Poverty == "Poor" ~ "True Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Non-Poor" ~ "True Non-Poor",
      Actual_Poverty == "Poor" & Predicted_Poverty == "Non-Poor" ~ "False Non-Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Poor" ~ "False Poor"
    )
  )

# Plot observed against predicted MPI and colour observations according to
# their poverty-classification outcome.
# The horizontal and vertical dashed lines show the MPI poverty threshold.
poverty_scatter <- ggplot(poverty_scatter_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", linewidth = 0.5) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non-Poor" = "darkgreen", 
    "False Poor" = "orange",
    "False Non-Poor" = "purple"
  )) +
  coord_equal(xlim = x_lim, ylim = y_lim, expand = FALSE) +
  labs(title = "Poverty Classification (Train)", 
       subtitle = "Threshold MPI = 0.3333",
       x = "Actual MPI", 
       y = "Predicted MPI", 
       color = "Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12),
    legend.position = "top"
  ) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

# Summarise the number and percentage of observations in each training
# classification category.
# Calculate classification accuracy
classification_summary <- poverty_scatter_data %>%
  count(Classification) %>%
  mutate(Percentage = n / sum(n) * 100)

cat("\nPoverty Classification Summary:\n")
print(classification_summary)

# Save the training poverty-classification plot.
ggsave(paste0(plotsdir, "/", "betareg", "/", "Train_classification_prediction_vs_actual_poverty.png"), 
       poverty_scatter, width = 12, height = 10, dpi = 300)
cat("Poverty classification scatterplot saved\n")



# Repeat the poverty-classification analysis for the independent test dataset.
# 5b. Scatterplot with Poverty Classification - Test Dataset
cat("\nCreating scatterplot with poverty classification...\n")

# Classify the test observations into the four possible poverty-classification
# outcomes using observed and predicted MPI values.
poverty_scatter_test_data <- test_resid_data_betareg %>%
  mutate(
    Actual_Poverty = ifelse(mpi > 0.3333, "Poor", "Non-Poor"),
    Predicted_Poverty = ifelse(.fitted > 0.3333, "Poor", "Non-Poor"),
    Classification = case_when(
      Actual_Poverty == "Poor" & Predicted_Poverty == "Poor" ~ "True Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Non-Poor" ~ "True Non-Poor",
      Actual_Poverty == "Poor" & Predicted_Poverty == "Non-Poor" ~ "False Non-Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Poor" ~ "False Poor"
    )
  )

# Create the poverty-classification plot for the independent test dataset.
poverty_test_dataset_scatter <- ggplot(poverty_scatter_test_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", linewidth = 0.5) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non-Poor" = "darkgreen", 
    "False Poor" = "orange",
    "False Non-Poor" = "purple"
  )) +
  coord_equal(xlim = x_lim, ylim = y_lim, expand = FALSE) +
  labs(title = "Poverty Classification (Test)", 
       subtitle = "Threshold MPI = 0.3333",
       x = "Actual MPI", 
       y = "Predicted MPI", 
       color = "Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12),
    legend.position = "top"
  ) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

# Summarise the number and percentage of observations in each test
# classification category.
# Calculate classification accuracy
classification_summary_test <- poverty_scatter_test_data %>%
  count(Classification) %>%
  mutate(Percentage = n / sum(n) * 100)


cat("\nPoverty Classification Summary:\n")
print(classification_summary)

# Save the independent test poverty-classification plot.
ggsave(paste0(plotsdir, "/", "betareg", "/", "Test_classification_prediction_vs_actual_poverty.png"), 
       poverty_test_dataset_scatter, width = 12, height = 10, dpi = 300)
cat("Poverty classification scatterplot for Test Dataset saved\n")

# ---------------------------------------------------------------
# 15) MULTI-PANEL (2x2)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 2) Compose the 2x2 layout explicitly
# ---------------------------------------------------------------
# Create a subtitle containing the selected preprocessing recipe and link
# functions for the final beta regression model.
betareg_subtitle <- paste0(
  "Model: ", best_config$recipe,
  " | Link: ", best_config$link,
  " | Precision link: ", best_config$link_phi
)

# Combine the training and test predicted-versus-actual plots with the
# corresponding poverty-classification plots in a 2-by-2 layout.
multi_panel_betareg <- (
  train_scatter + theme(axis.text = element_text(size = 8)) |
    test_scatter  + theme(axis.text = element_text(size = 8))
) /
  (
    poverty_scatter + theme(axis.text = element_text(size = 8)) |
      poverty_test_dataset_scatter + theme(axis.text = element_text(size = 8))
  ) +
  # Keep the rows and columns of the multi-panel figure evenly sized.
  plot_layout(
    widths  = c(1, 1),   #  FORCE equal column widths
    heights = c(1, 1)    # optional, keeps rows balanced
  ) +
  plot_annotation(
    subtitle = betareg_subtitle,
    theme = theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12)
    )
  )

# Save the complete four-panel beta regression performance figure.
ggsave(
  file.path(plotsdir, "betareg", paste0("betareg_comprehensive_", run_id, ".png")),
  multi_panel_betareg, width = 16, height = 12, dpi = 300
)


# 6. Create a comprehensive multi-panel plot
cat("\nCreating comprehensive multi-panel plot...\n")


# Print the main training and test regression performance metrics to the console.
# Print summary statistics
cat("\n=== PREDICTION VS ACTUAL SUMMARY ===\n")
cat("Training Set Performance:\n")
cat(sprintf("  RMSE: %.4f\n", train_rmse))
cat(sprintf("  MAE:  %.4f\n", train_mae))
cat(sprintf("  R²:   %.4f\n", train_rsq))
cat(sprintf("  Corr: %.4f\n", train_cor))

if ("mpi" %in% names(test_proc)) {
  cat("Test Set Performance:\n")
  cat(sprintf("  RMSE: %.4f\n", test_rmse))
  cat(sprintf("  MAE:  %.4f\n", test_mae))
  cat(sprintf("  R²:   %.4f\n", test_rsq))
  cat(sprintf("  Corr: %.4f\n", test_cor))
}

# Calculate the overall training poverty-classification accuracy from the
# correctly classified poor and non-poor observations.
cat("\nPoverty Classification Accuracy (Training):\n")
true_classifications <- classification_summary %>%
  filter(Classification %in% c("True Poor", "True Non-Poor")) %>%
  summarise(accuracy = sum(Percentage))
cat(sprintf("  Overall Accuracy: %.1f%%\n", true_classifications$accuracy))

# Confirm completion of the prediction-versus-actual and classification
# plotting section and list the expected plot outputs.
cat("\n Prediction vs Actual analysis complete!\n")
cat(" Scatterplots saved:\n")
cat("   - prediction_vs_actual_training.png\n")
if ("mpi" %in% names(test_proc)) {
  cat("   - prediction_vs_actual_test.png\n")
  cat("   - prediction_vs_actual_combined.png\n")
}
cat("   - prediction_vs_actual_enhanced.png\n")
cat("   - prediction_vs_actual_poverty.png\n")
cat("   - comprehensive_prediction_analysis.png\n")




# ============================================================================
# COMBINED REGRESSION AND POVERTY-CLASSIFICATION METRICS
# This section defines the common performance metrics used for Random Forest,
# XGBoost and the Artificial Neural Network. Continuous MPI predictions are also
# converted into poverty classes using the MPI poverty threshold of 0.3333 
# so that regression and classification performance can be evaluated consistently.
# ============================================================================

# Define MPI poverty threshold used throughout the model evaluation.
# Regions with MPI greater than or equal to 0.3333 are classified as poor.
# We'll use "non-poor" and "poor" as labels (poor = mpi >= threshold)
thr <- 0.3333

# FIXED — underscore, poor is first level, event_level = "second" removed
# Convert continuous observed and predicted MPI values into poverty classes.
# The poor class is placed first so that it is treated as the positive class.
.make_classes <- function(truth, estimate, threshold = 0.3333) {
  truth_cl <- ifelse(truth >= threshold, "poor", "non_poor")   # underscore
  pred_cl  <- ifelse(estimate >= threshold, "poor", "non_poor")
  list(truth = factor(truth_cl, levels = c("poor", "non_poor")),   # poor = first = positive
       pred  = factor(pred_cl,  levels = c("poor", "non_poor")))
}

# Create descriptive classification labels for later plots and summaries.
.make_classes_label <- function(truth, estimate, threshold = 0.3333) {
  actual <- ifelse(truth >= threshold, "Poor", "Non_Poor")
  pred   <- ifelse(estimate >= threshold, "Poor", "Non_Poor")
  
  # Assign each observation to the appropriate correct or incorrect classification outcome.
  class <- dplyr::case_when(
    actual == "Poor"     & pred == "Poor"     ~ "True Poor",
    actual == "Non_Poor" & pred == "Non_Poor" ~ "True Non_Poor",
    actual == "Poor"     & pred == "Non_Poor" ~ "False Non_Poor",
    actual == "Non_Poor" & pred == "Poor"     ~ "False Poor"   # ← underscore
  )
  
  tibble(
    Classification = factor(
      class,
      levels = c("True Poor", "True Non_Poor", "False Poor", "False Non_Poor")  # ← underscores
    )
  )
}

# --------------------------
# 1) accuracy (numeric-registered)
# --------------------------
# Define a custom accuracy metric based on the MPI poverty threshold.
accuracy_033_vec <- function(truth, estimate, threshold = 0.3333, na_rm = TRUE, ...) {
  accuracy_033_impl <- function(truth, estimate, ...) {
    cls <- .make_classes(truth, estimate, threshold)
    yardstick::accuracy_vec(truth = cls$truth, estimate = cls$pred, na_rm = na_rm)
  }
  
  metric_vec_template(
    metric_impl = accuracy_033_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    ...
  )
}

# Register the custom accuracy function as a yardstick numeric metric.
accuracy_033 <- function(data, ...) UseMethod("accuracy_033")
accuracy_033 <- new_numeric_metric(accuracy_033, direction = "maximize")

accuracy_033.data.frame <- function(data, truth, estimate, threshold = 0.3333, na_rm = TRUE, ...) {
  metric_summarizer(
    metric_nm = "accuracy_033",
    metric_fn = accuracy_033_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    threshold = threshold,
    ...
  )
}

# --------------------------
# 2) sensitivity (registered as numeric)
# --------------------------
# Define sensitivity for the poor class.
# This measures the proportion of genuinely poor regions correctly identified as poor.
sensitivity_033_vec <- function(truth, estimate, threshold = 0.3333, na_rm = TRUE,
                                event_level = "first", ...) {
  sensitivity_033_impl <- function(truth, estimate, event_level = "first", ...) {
    cls <- .make_classes(truth, estimate, threshold)
    yardstick::sens_vec(truth = cls$truth, estimate = cls$pred,
                        event_level = event_level, na_rm = na_rm)
  }
  
  metric_vec_template(
    metric_impl = sensitivity_033_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    event_level = event_level,
    ...
  )
}

# Register sensitivity as a yardstick numeric metric to be maximised.
sensitivity_033 <- function(data, ...) UseMethod("sensitivity_033")
sensitivity_033 <- new_numeric_metric(sensitivity_033, direction = "maximize")

sensitivity_033.data.frame <- function(data, truth, estimate, threshold = 0.3333,
                                       na_rm = TRUE, event_level = "first", ...) {
  metric_summarizer(
    metric_nm = "sensitivity_033",
    metric_fn = sensitivity_033_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    threshold = threshold,
    event_level = event_level,
    ...
  )
}

# --------------------------
# 3) specificity (registered as numeric)
# --------------------------
# Define specificity for the non-poor class.
# This measures the proportion of genuinely non-poor regions correctly identified as non-poor.
specificity_033_vec <- function(truth, estimate, threshold = 0.3333, na_rm = TRUE,
                                event_level = "first", ...) {
  specificity_033_impl <- function(truth, estimate, event_level = "first", ...) {
    cls <- .make_classes(truth, estimate, threshold)
    yardstick::spec_vec(truth = cls$truth, estimate = cls$pred,
                        event_level = event_level, na_rm = na_rm)
  }
  
  metric_vec_template(
    metric_impl = specificity_033_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    event_level = event_level,
    ...
  )
}

# Register specificity as a yardstick numeric metric to be maximised.
specificity_033 <- function(data, ...) UseMethod("specificity_033")
specificity_033 <- new_numeric_metric(specificity_033, direction = "maximize")

specificity_033.data.frame <- function(data, truth, estimate, threshold = 0.3333,
                                       na_rm = TRUE, event_level = "first", ...) {
  metric_summarizer(
    metric_nm = "specificity_033",
    metric_fn = specificity_033_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    threshold = threshold,
    event_level = event_level,
    ...
  )
}

# --------------------------
# 4) precision (registered as numeric) - will not be directly used
# --------------------------
# Calculate precision directly from the confusion matrix.
# Return zero when no observations are predicted as poor to avoid division by zero.
precision_safe <- function(truth, estimate, threshold = 0.3333) {
  truth_cls <- factor(ifelse(truth >= threshold, "poor", "non_poor"),
                      levels = c("poor","non_poor"))
  pred_cls  <- factor(ifelse(estimate > threshold, "poor", "non_poor"),
                      levels = c("poor","non_poor"))
  cm <- table(truth_cls, pred_cls)
  tp <- cm["poor","poor"]
  fp <- cm["non_poor","poor"]
  if ((tp + fp) == 0) return(0)  # <- avoids NA + warning
  tp / (tp + fp)
}





# Define the yardstick-compatible precision metric using the same class mapping.
precision_033_vec <- function(truth, estimate, threshold = 0.3333, na_rm = TRUE,
                              event_level = "first", ...) {
  precision_033_impl <- function(truth, estimate, event_level = "first", ...) {
    cls <- .make_classes(truth, estimate, threshold)
    yardstick::precision_vec(truth = cls$truth, estimate = cls$pred,
                             event_level = event_level, na_rm = na_rm)
  }
  
  metric_vec_template(
    metric_impl = precision_033_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    event_level = event_level,
    ...
  )
}

# Register precision as a yardstick numeric metric to be maximised.
precision_033 <- function(data, ...) UseMethod("precision_033")
precision_033 <- new_numeric_metric(precision_033, direction = "maximize")

precision_033.data.frame <- function(data, truth, estimate, threshold = 0.3333,
                                     na_rm = TRUE, event_level = "first", ...) {
  metric_summarizer(
    metric_nm = "precision_033",
    metric_fn = precision_033_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    threshold = threshold,
    event_level = event_level,
    ...
  )
}

# --------------------------
# 5) F1 (registered as numeric)
# --------------------------
# Define the F1-score with poor treated as the positive class.
# This provides a combined measure of precision and sensitivity.
f1_033_vec <- function(truth, estimate, threshold = 0.3333, na_rm = TRUE,
                       event_level = "first", ...) {
  f1_033_impl <- function(truth, estimate, event_level = "first", ...) {
    cls <- .make_classes(truth, estimate, threshold)
    yardstick::f_meas_vec(truth = cls$truth, estimate = cls$pred,
                          event_level = event_level, na_rm = na_rm)
  }
  
  metric_vec_template(
    metric_impl = f1_033_impl,
    truth = truth,
    estimate = estimate,
    na_rm = na_rm,
    cls = "numeric",
    event_level = event_level,
    ...
  )
}

# Register the F1-score as a yardstick numeric metric to be maximised.
f1_033 <- function(data, ...) UseMethod("f1_033")
f1_033 <- new_numeric_metric(f1_033, direction = "maximize")

f1_033.data.frame <- function(data, truth, estimate, threshold = 0.3333,
                              na_rm = TRUE, event_level = "first", ...) {
  metric_summarizer(
    metric_nm = "f1_033",
    metric_fn = f1_033_vec,
    data = data,
    truth = !!enquo(truth),
    estimate = !!enquo(estimate),
    na_rm = na_rm,
    threshold = threshold,
    event_level = event_level,
    ...
  )
}


# FIXED calc_test_metrics — consistent with compute_class_metrics
# Calculate the complete regression and classification metrics for the independent test dataset.
calc_test_metrics <- function(truth, estimate, threshold = 0.3333) {
  # Calculate RMSE, MAE and R-squared from the continuous MPI predictions.
  rmse_v <- yardstick::rmse_vec(truth, estimate)
  mae_v  <- yardstick::mae_vec(truth, estimate)
  rsq_v  <- tryCatch(yardstick::rsq_vec(truth, estimate), error = function(e) NA_real_)
  
  # Convert observed and predicted MPI values into poverty classes using the same threshold.
  actual    <- factor(ifelse(truth    >= threshold, "poor", "non_poor"),
                      levels = c("poor", "non_poor"))   # poor = positive
  predicted <- factor(ifelse(estimate >= threshold, "poor", "non_poor"),
                      levels = c("poor", "non_poor"))
  
  # Construct the confusion matrix for the poverty-classification results.
  cm <- table(Actual = actual, Predicted = predicted)
  
  # Extract the confusion-matrix counts by class name, with poor treated as positive.
  # Named extraction — consistent with compute_class_metrics
  tp <- cm["poor",     "poor"]
  fn <- cm["poor",     "non_poor"]
  fp <- cm["non_poor", "poor"]
  tn <- cm["non_poor", "non_poor"]
  
  # Calculate classification metrics while protecting against division by zero.
  accuracy    <- (tp + tn) / max(1, sum(cm))
  sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else 0
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else 0
  precision   <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  f1          <- if ((precision + sensitivity) > 0) 
    2 * (precision * sensitivity) / (precision + sensitivity) else 0
  
  # Return regression metrics, classification metrics, confusion-matrix counts and threshold.
  tibble(
    rmse = rmse_v, mae = mae_v, rsq = rsq_v,
    accuracy = accuracy, sensitivity = sensitivity,
    specificity = specificity, f1 = f1,
    tp = as.numeric(tp), tn = as.numeric(tn),
    fp = as.numeric(fp), fn = as.numeric(fn),
    threshold = threshold
  )
}

# --------------------------
# 6) Build combined metric_set and test
# --------------------------


# Combine the regression and custom classification measures into one yardstick metric set.
# This allows the same metrics to be used consistently during model evaluation.
combined_metrics_num <- yardstick::metric_set(
  yardstick::rmse,
  yardstick::rsq,
  yardstick::mae,
  accuracy_033,
  sensitivity_033,
  specificity_033,
  f1_033
)

# Run a small sanity check before using the combined metrics in the main model analyses.
# Quick sanity test on small example
test_data <- tibble(
  mpi  = c(0.2, 0.5, 0.1, 0.8, 0.4),
  .pred = c(0.25, 0.6, 0.15, 0.7, 0.35)
)

combined_metrics_num(test_data, truth = mpi, estimate = .pred)


# Compare two candidate models and determine which model should be retained.
# Selection is based first on RMSE, followed by MAE and then R-squared.
# Function to identify which stage model is best. It first compares rmse, then MAE followed by R^2

# The tolerance prevents very small numerical differences from being treated as meaningful.
is_better_model <- function(model1, model2, tol = 1e-6){
  
  rmse1 <- as.numeric(model1$rmse[[1]])
  rmse2 <- as.numeric(model2$rmse[[1]])
  
  mae1 <- as.numeric(model1$mae[[1]])
  mae2 <- as.numeric(model2$mae[[1]])
  
  r21 <- as.numeric(model1$rsq[[1]])
  r22 <- as.numeric(model2$rsq[[1]])
  
  # Use RMSE as the primary model-selection criterion; lower RMSE is preferred.
  ## Primary criterion
  if(abs(rmse1 - rmse2) > tol){
    
    return(list(
      better = rmse1 < rmse2,
      criterion = "RMSE",
      model1_value = rmse1,
      model2_value = rmse2
    ))
  }
  
  # When RMSE is effectively tied, use MAE as the secondary criterion.
  ## Secondary criterion
  if(abs(mae1 - mae2) > tol){
    
    return(list(
      better = mae1 < mae2,
      criterion = "MAE",
      model1_value = mae1,
      model2_value = mae2
    ))
  }
  
  # When RMSE and MAE are effectively tied, use R-squared as the final criterion.
  # The model with the higher R-squared is preferred.
  ## Final criterion
  if(abs(r21 - r22) > tol){
    
    return(list(
      better = r21 > r22,
      criterion = "R-squared",
      model1_value = r21,
      model2_value = r22
    ))
  }
  
  # Treat the two candidate models as tied when all three measures are within the tolerance.
  ## Complete tie
  return(list(
    better = TRUE,
    criterion = "Tie",
    model1_value = NA,
    model2_value = NA
  ))
}


# ============================================================================
# RANDOM FOREST: INITIAL SPACE-FILLING HYPERPARAMETER SEARCH
# This section performs the first-stage Random Forest hyperparameter search.
# Three pre-processing recipes are evaluated using a space-filling grid across
# mtry, number of trees and minimum node size. Model performance is assessed
# using the common regression and poverty-classification metrics.
# UPDATED - 17th Feb
# ============================================================================


# ======================================================
# Reproducibility — Set ONCE, at the TOP, before anything
# ======================================================
RNGkind("L'Ecuyer-CMRG")
set.seed(1238)   # global seed — matches your data split seed for consistency

# Confirm seed is set
cat("RNG kind:", RNGkind()[1], "\n")
cat("Global seed set: 1238\n\n")


# ======================================================
# Preprocessing recipes
# ======================================================
# Store the three preprocessing recipes evaluated during the initial search.
preproc.randf <- list(
  all_pred_notfm                  = all_pred_notfm.recipe,
  all_pred_tfmwithlogcubenormalize = all_pred_tfmwithlogcubenormalize.recipe,
  all_pred_tfmwithlogcuberange    = all_pred_tfmwithlogcuberange.recipe
)


# ======================================================
# RF Model Specification
# ======================================================
# Define the Random Forest regression model.
# mtry, number of trees and minimum node size are left for tuning.
# Impurity-based variable importance is requested from the ranger engine.
rforest_spec <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity", seed = 124) %>%
  set_mode("regression")


# ======================================================
# Parameter Ranges (unchanged)
# ======================================================
# Define the initial search ranges for the three Random Forest hyperparameters.
randforest_custom_param <- parameters(list(
  trees = trees(range = c(250L, 2000L)),
  mtry  = mtry(range  = c(floor(sqrt(ncol(df))), ncol(df) - 1)),
  min_n = min_n(range = c(2, 12))
))


# ======================================================
# FIX 1 (CORE): Isolate grid_space_filling() with its own seed
# ======================================================
# ORIGINAL CODE did:
#   set.seed(124)
#   randforest_param_grid_sfill <- grid_space_filling(...)
#
# That set.seed() was separated from grid_space_filling() by other code,
# meaning the RNG was in a different state each run.
# Fix: set.seed() IMMEDIATELY before grid_space_filling(), nowhere else.

# Use a separate seed for generation of the space-filling hyperparameter grid.
GRID_SEED <- 124L   # document this seed explicitly for your dissertation

# Generate 95 space-filling hyperparameter combinations across the initial
# parameter ranges.
set.seed(GRID_SEED)
randforest_param_grid_sfill <- grid_space_filling(
  randforest_custom_param,
  size     = 95,
  original = TRUE    # Critical for integer params — keep this
)

cat("Grid rows generated:", nrow(randforest_param_grid_sfill), "\n")
cat("Grid seed used:", GRID_SEED, "(document this in dissertation)\n\n")

# Verify grid is deterministic by checking first 3 rows
cat("First 3 grid rows (should be identical across runs):\n")
print(head(randforest_param_grid_sfill, 3))


# ======================================================
# OUTPUT PATHS
# ======================================================
# Create a unique identifier for the current run and define the output
# directories and files used to retain CV results, predictions and timing.
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
out_dir <- file.path(model_dir, "randomforest")
images_dir <- file.path(out_dir, "images")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(images_dir, recursive = TRUE, showWarnings = FALSE)

metrics_file          <- file.path(out_dir, paste0("rf_cv_metrics_",      run_id, ".csv"))
preds_file            <- file.path(out_dir, paste0("rf_cv_predictions_",  run_id, ".csv"))
timing_file           <- file.path(out_dir, paste0("rf_cv_timing_",       run_id, ".csv"))
bestcvmetric_file     <- file.path(out_dir, paste0("rf_best_cv_metrics_", run_id, ".csv"))
singlebestmetric_file <- file.path(out_dir, paste0("rf_single_best_metrics_", run_id, ".csv"))

# Remove existing run-specific metric, prediction and timing files before
# starting the search.
for (f in c(metrics_file, preds_file, timing_file)) {
  if (file.exists(f)) file.remove(f)
}

# Use the generated space-filling grid as the first-stage Random Forest grid
# and confirm that all required tuning parameters are present.
rf_grid <- randforest_param_grid_sfill
stopifnot(all(c("mtry", "trees", "min_n") %in% names(rf_grid)))


# ======================================================
# FIX 2: Register parallel backend BEFORE setting seeds
# ======================================================

# --- OPTION A: No parallelism (safest for reproducibility) ---
# cores <- 1
# doParallel::registerDoParallel(cores = cores)

# --- OPTION B: Parallel with explicit seeds (faster) ---
#cores <- 2   # adjust to your machine
#doParallel::registerDoParallel(cores = cores)

# Configure a small parallel backend for the repeated cross-validation search.
cores <- max(1, min(2, parallel::detectCores() - 1))
doParallel::registerDoParallel(cores = cores)

cat("Parallel backend registered with", cores, "cores\n")
cat("NOTE: Fixed seeds are used to improve reproducibility\n\n")


# ======================================================
# FIX 3 + FIX 4: Deterministic seed function + control object
# ======================================================

# Seed function — same as original but documented clearly
# Generate a deterministic seed from the recipe name and grid position.
# This provides a traceable seed for each hyperparameter evaluation.
seed_for_run <- function(recipe_name, grid_id, base = 124L) {
  as.integer((base + sum(utf8ToInt(recipe_name)) + grid_id) %% .Machine$integer.max)
}

# FIX 4 (CORE): Generate explicit seeds for every resample
# tune_grid() accepts a `seeds` argument via control_grid() in some
# versions, but the most reliable method is to use allow_par = FALSE
# OR to pass seeds via the control object.
#
# The control object below sets allow_par = TRUE for speed, but
# if you still get non-reproducible results, change allow_par = FALSE.

# Configure tidymodels to retain fold predictions and fitted workflows during tuning.
ctrl <- control_grid(
  save_pred     = TRUE,
  save_workflow = TRUE,
  parallel_over = "resamples",   # "resamples" is more reproducible than "everything"
  allow_par     = TRUE,          # set FALSE to reduce variation from parallel execution
  verbose       = TRUE
)

# ADDITIONAL REPRODUCIBILITY NOTE:
# If allow_par = TRUE still gives different results, this is a known
# limitation of parallel tune_grid(). For a dissertation, document:
# "Hyperparameter tuning was conducted sequentially (allow_par = FALSE)
# with L'Ecuyer-CMRG RNG and seed [X] to ensure full reproducibility."


# ======================================================
# Timing helpers (unchanged)
# ======================================================
# Create a timing table and helper function for recording the duration of
# individual grid evaluations and complete recipe searches.
timing_log <- tibble()

log_time <- function(level, recipe, grid_id, t0, t1) {
  tibble(
    level         = level,
    recipe        = recipe,
    grid_id       = grid_id,
    start_time    = t0,
    end_time      = t1,
    duration_secs = as.numeric(difftime(t1, t0, units = "secs")),
    duration_mins = as.numeric(difftime(t1, t0, units = "mins"))
  )
}


# ======================================================
# MAIN LOOP: recipe x grid row
# ======================================================
# Store the cross-validation metrics generated for all preprocessing recipes.
all_metrics <- list()

# Evaluate each preprocessing recipe separately.
for (r in names(preproc.randf)) {
  
  recipe_start <- Sys.time()
  rec <- preproc.randf[[r]]
  
  
  # ── Load the recipe that was used during training ─────────────
  # You need to save all_pred_notfm.recipe after prep() from your training script.
  # In your training script, add this once:
  #   prep_rec <- prep(all_pred_notfm.recipe, training = ntl_train)
  #   saveRDS(prep_rec, file.path(model_dir, "xgboost_production_recipe.rds"))
  # prep_rec_tmp <- prep(rec, training = ntl_train)
  # saveRDS(prep_rec_tmp, file.path(images_dir, paste0("xgboost_", r, "_recipe.rds")))
  
  # Prepare the current recipe using the complete training dataset to determine
  # the number of numeric predictors remaining after preprocessing.
  # The prepared recipe is also saved for later use.
  # Prep recipe to count actual predictors post-processing
  prep_rec_tmp <- prep(rec, training = ntl_train)
  saveRDS(prep_rec_tmp, file.path(images_dir, paste0("randomforest_", r, "_recipe.rds")))
  
  train_baked  <- bake(prep_rec_tmp, new_data = ntl_train)
  
  # Count the numeric predictor variables after preprocessing, excluding MPI.
  outcome <- "mpi"
  pnames  <- setdiff(names(train_baked), outcome)
  pnames  <- pnames[sapply(train_baked[, pnames, drop = FALSE], is.numeric)]
  n_pred  <- length(pnames)
  
  cat(">>> Recipe:", r, "| predictors after recipe:", n_pred, "\n")
  
  # Restrict mtry to the number of predictors available after preprocessing.
  # Duplicate parameter combinations created by this restriction are removed.
  # Cap mtry to actual predictor count (unchanged)
  rf_grid_r <- rf_grid %>%
    mutate(mtry = pmin(mtry, n_pred)) %>%
    distinct(mtry, trees, min_n, .keep_all = TRUE) %>%
    arrange(mtry, trees, min_n)   # deterministic row ordering
  
  # Create the Random Forest workflow for the current preprocessing recipe.
  wf_base <- workflow() %>%
    add_recipe(rec) %>%
    add_model(rforest_spec)
  
  cat("\n=============================\n")
  cat("RECIPE:", r, "\n")
  cat("=============================\n")
  
  # Evaluate each first-stage hyperparameter combination for the current recipe.
  for (g in seq_len(nrow(rf_grid_r))) {
    
    cfg        <- rf_grid_r[g, , drop = FALSE]
    grid_start <- Sys.time()
    
    cat("  Grid", g, "| mtry=", cfg$mtry, " trees=", cfg$trees,
        " min_n=", cfg$min_n, "\n")
    
    # Set the run-specific seed immediately before the tuning call.
    # FIX 3 (CORE): Set seed IMMEDIATELY before tune_grid(), with no
    # intervening RNG-consuming operations.
    run_seed <- seed_for_run(r, g)
    set.seed(run_seed)
    
    # Evaluate the current Random Forest configuration using the predefined
    # repeated cross-validation folds and the common model-performance metrics.
    # tryCatch allows the complete search to continue if one configuration fails.
    res <- tryCatch(
      tune_grid(
        object    = wf_base,
        resamples = ntl_vfolds_cv,
        grid      = cfg,
        metrics   = combined_metrics_num,
        control   = ctrl
      ),
      error = function(e) e
    )
    
    grid_end   <- Sys.time()
    timing_log <- bind_rows(timing_log, log_time("grid", r, g, grid_start, grid_end))
    
    # If a grid configuration fails, retain a row of missing metrics together
    # with the parameter values and seed used for that run.
    if (inherits(res, "error")) {
      warning("Grid failed: recipe=", r, " grid=", g, " | ", conditionMessage(res))
      
      fail_row <- tibble(
        recipe = r, grid_id = g,
        mtry = cfg$mtry, trees = cfg$trees, min_n = cfg$min_n,
        rmse = NA_real_, rmse_se = NA_real_,
        mae  = NA_real_, mae_se  = NA_real_,
        rsq  = NA_real_, rsq_se  = NA_real_,
        accuracy_033     = NA_real_, accuracy_033_se     = NA_real_,
        sensitivity_033  = NA_real_, sensitivity_033_se  = NA_real_,
        specificity_033  = NA_real_, specificity_033_se  = NA_real_,
        f1_033           = NA_real_, f1_033_se           = NA_real_,
        n_resamples = NA_integer_,
        run_seed = run_seed      # FIX: save seed used for audit trail
      )
      
      if (!file.exists(metrics_file)) write_csv(fail_row, metrics_file)
      else write_csv(fail_row, metrics_file, append = TRUE)
      next
    }
    
    # Collect and reshape the cross-validation metrics for the current configuration.
    # Mean performance and standard errors across the resamples are retained.
    # Collect metrics
    m <- collect_metrics(res) %>%
      filter(.metric %in% c("rmse", "mae", "rsq",
                            "accuracy_033", "sensitivity_033",
                            "specificity_033", "f1_033")) %>%
      select(.metric, mean, std_err, n) %>%
      tidyr::pivot_wider(
        names_from  = .metric,
        values_from = c(mean, std_err),
        names_sep   = "_"
      ) %>%
      rename_with(~ str_replace(.x, "^mean_", ""),    starts_with("mean_")) %>%
      rename_with(~ paste0(.x, "_se"),                starts_with("std_err_")) %>%
      rename_with(~ str_replace(.x, "^std_err_", ""), starts_with("std_err_")) %>%
      mutate(
        recipe      = r,
        grid_id     = g,
        mtry        = cfg$mtry,
        trees       = cfg$trees,
        min_n       = cfg$min_n,
        n_resamples = n,
        run_seed    = run_seed    # FIX: save seed for audit trail
      ) %>%
      select(recipe, grid_id, mtry, trees, min_n, run_seed, everything(), -n)
    
    # Collect the out-of-fold predictions generated for the current configuration
    # and attach the corresponding recipe and hyperparameter values.
    # Collect fold predictions
    p <- collect_predictions(res) %>%
      arrange(id, .row) %>%
      mutate(
        recipe  = r,
        grid_id = g,
        mtry    = cfg$mtry,
        trees   = cfg$trees,
        min_n   = cfg$min_n
      )
    
    # Save metrics and fold predictions incrementally so completed results are
    # retained even if a later configuration fails.
    # Atomic incremental write
    if (!file.exists(metrics_file)) write_csv(m, metrics_file)
    else write_csv(m, metrics_file, append = TRUE)
    
    if (!file.exists(preds_file)) write_csv(p, preds_file)
    else write_csv(p, preds_file, append = TRUE)
  }
  
  # Record the total elapsed time for the completed preprocessing recipe.
  recipe_end <- Sys.time()
  timing_log <- bind_rows(timing_log,
                          log_time("recipe", r, NA_integer_, recipe_start, recipe_end))
}

# Save the timing log and stop the parallel backend after completion of the search.
write_csv(timing_log, timing_file)
doParallel::stopImplicitCluster()

cat("\n DONE.\n")
cat("Metrics:     ", metrics_file, "\n")
cat("Predictions: ", preds_file, "\n")
cat("Timing:      ", timing_file, "\n")


# ======================================================
# 8. ANALYZE RESULTS (unchanged logic, same output)
# ======================================================
# Read the completed first-stage CV results for model selection.
rf_metrics <- readr::read_csv(metrics_file, show_col_types = FALSE)

# Select the best configuration within each preprocessing recipe from the
# successful first-stage CV results.
best_per_recipe_rf <- rf_metrics %>%
  filter(!is.na(rmse)) %>%
  group_by(recipe) %>%
  arrange(rmse, mae, desc(rsq)) %>%
  slice(1) %>%
  ungroup()

print(best_per_recipe_rf)
write_csv(best_per_recipe_rf, bestcvmetric_file)

# Select the single best first-stage configuration across all preprocessing recipes.
rf_singlebestcv <- rf_metrics %>%
  filter(!is.na(rmse)) %>%
  arrange(rmse, mae, desc(rsq)) %>%
  slice(1) %>%
  ungroup()

print(rf_singlebestcv)
write_csv(rf_singlebestcv, singlebestmetric_file)


# --------------------------
# 1) Extract BEST RF config per recipe from workflow_set results
#     (works for tune_grid results)
# --------------------------


# -------------------------------
# Output files (new names so you don’t overwrite CV files)
# -------------------------------
# Define separate files for independent-test predictions and performance metrics.
final_test_metrics_file <- file.path(out_dir, paste0("rf_final_test_metrics_", run_id, ".csv"))
final_test_preds_file   <- file.path(out_dir, paste0("rf_final_test_predictions_", run_id, ".csv"))

if (file.exists(final_test_metrics_file)) file.remove(final_test_metrics_file)
if (file.exists(final_test_preds_file))   file.remove(final_test_preds_file)

# -------------------------------
# Loop over best model per recipe (from CSV)
# best_per_recipe must contain: recipe, mtry, trees, min_n
# -------------------------------
# Confirm that the best-per-recipe table contains the required Random Forest
# hyperparameters before fitting the final first-stage models.
stopifnot(all(c("recipe","mtry","trees","min_n") %in% names(best_per_recipe_rf)))

# Refit the best configuration from each preprocessing recipe using the full
# training dataset and evaluate it on the independent test dataset.
# Retain each fitted model and metric table so the selected initial model
# can be recovered after the loop.
firstround_fits <- list()
firstround_train_metrics_list <- list()
firstround_test_metrics_list <- list()

for (k in seq_len(nrow(best_per_recipe_rf))) {
  
  rec_name <- best_per_recipe_rf$recipe[k]
  mtry_k   <- best_per_recipe_rf$mtry[k]
  trees_k  <- best_per_recipe_rf$trees[k]
  min_n_k  <- best_per_recipe_rf$min_n[k]
  
  cat("\n=============================\n")
  cat("FINAL FIT:", rec_name, "\n")
  cat("mtry=", mtry_k, " trees=", trees_k, " min_n=", min_n_k, "\n")
  cat("=============================\n")
  
  # Retrieve the preprocessing recipe associated with the selected configuration.
  # get recipe
  rec <- preproc.randf[[rec_name]]
  if (is.null(rec)) stop("Recipe not found in preproc.randf: ", rec_name)
  
  # Finalise the Random Forest specification using the selected first-stage
  # mtry, number of trees and minimum node size.
  # finalize model spec with chosen params
  final_rf_spec <- rforest_spec %>%
    finalize_model(
      tibble(mtry = mtry_k, trees = trees_k, min_n = min_n_k)
    )
  
  # Build the final workflow by combining the selected recipe and finalised model.
  # build workflow
  wf_final <- workflow() %>%
    add_recipe(rec) %>%
    add_model(final_rf_spec)
  
  # Fit the selected configuration using the complete training dataset.
  # fit on full training data
  set.seed(124)
  final_fit <- fit(wf_final, data = ntl_train)
  firstround_fits[[rec_name]] <- final_fit
  
  # store the predict model
  # ---------------------------------------------------
  # Define model save path (FILE, not directory)
  # ---------------------------------------------------
  images_dir <- file.path(out_dir, "images")
  dir.create(images_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Define a run-specific file for saving the fitted workflow.
  model_file <- file.path(
    images_dir,
    paste0("final_", rec_name, "_", run_id, ".rds")
  )
  
  # ---------------------------------------------------
  # Save bundled model if it does not already exist
  # ---------------------------------------------------
  # Bundle and save the fitted workflow when a model file does not already exist.
  if (!file.exists(model_file)) {
    
    bundled_model <- bundle(final_fit)
    
    saveRDS(bundled_model, model_file)
    
    message("Model saved to: ", model_file)
    
  } else {
    message("Model already exists, skipping save: ", model_file)
  }
  
  
  # Generate predictions for the independent test dataset and retain the
  # observed MPI together with the selected hyperparameter values.
  # predict test
  set.seed(124)
  firstround_test_preds <- predict(final_fit, new_data = ntl_test) %>%
    bind_cols(ntl_test %>% dplyr::select(mpi)) %>%   # keep truth
    mutate(
      recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k
    )
  
  # Generate corresponding predictions for the complete training dataset.
  # Predict train
  set.seed(124)
  firstround_train_preds <- predict(final_fit, new_data = ntl_train) %>%
    bind_cols(ntl_train %>% dplyr::select(mpi)) %>%   # keep truth
    mutate(
      recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k
    )
  
  
  # Calculate regression performance manually for the training and test predictions
  # as an additional verification of the fitted model.
  #  Calculate metrics using the final fit
  # From predictions objects (these are correct)
  rf_firstround_train_metrics <- tibble(
    dataset = "train",
    rmse = rmse_vec(firstround_train_preds$mpi, firstround_train_preds$.pred),
    mae  = mae_vec(firstround_train_preds$mpi, firstround_train_preds$.pred),
    rsq  = rsq_vec(firstround_train_preds$mpi, firstround_train_preds$.pred),
    cor  = cor(firstround_train_preds$mpi, firstround_train_preds$.pred)
  )
  
  rf_firstround_test_metrics <- tibble(
    dataset = "test",
    rmse = rmse_vec(firstround_test_preds$mpi, firstround_test_preds$.pred),
    mae  = mae_vec(firstround_test_preds$mpi, firstround_test_preds$.pred),
    rsq  = rsq_vec(firstround_test_preds$mpi, firstround_test_preds$.pred),
    cor  = cor(firstround_test_preds$mpi, firstround_test_preds$.pred)
  )
  
  firstround_train_metrics_list[[rec_name]] <- rf_firstround_train_metrics
  firstround_test_metrics_list[[rec_name]] <- rf_firstround_test_metrics
  
  combined_metrics <- bind_rows(rf_firstround_train_metrics, rf_firstround_test_metrics)
  
  cat("\n=== MANUAL METRICS VERIFICATION ===\n")
  print(combined_metrics)
  
  # Save the independent-test predictions for each selected preprocessing recipe.
  # save predictions incrementally
  if (!file.exists(final_test_preds_file)) write_csv(firstround_test_preds, final_test_preds_file)
  else write_csv(firstround_test_preds, final_test_preds_file, append = TRUE)
  
  # Calculate and save the complete test metrics, including the poverty-
  # classification measures based on the MPI threshold of 0.3333.
  # compute + save test metrics
  met <- calc_test_metrics(truth = firstround_test_preds$mpi, estimate = firstround_test_preds$.pred, threshold = 0.3333) %>%
    mutate(recipe = rec_name, mtry = mtry_k, trees = trees_k, min_n = min_n_k)
  
  if (!file.exists(final_test_metrics_file)) write_csv(met, final_test_metrics_file)
  else write_csv(met, final_test_metrics_file, append = TRUE)
}

cat("\nFinal test metrics saved to:\n", final_test_metrics_file, "\n")
cat("Final test predictions saved to:\n", final_test_preds_file, "\n")


# ======================================================
# REPRODUCIBILITY VERIFICATION BLOCK
# (Run this after the search to confirm reproducibility)
# ======================================================
# Print the main seed and parallel-processing settings used during the
# first-stage hyperparameter search.
cat("\n=============================\n")
cat("REPRODUCIBILITY VERIFICATION\n")
cat("=============================\n")
cat("Grid seed (GRID_SEED):", GRID_SEED, "\n")
cat("Global seed:           1238\n")
cat("Seed function:         seed_for_run(recipe, grid_id, base=124)\n")
cat("RNG kind:              L'Ecuyer-CMRG\n")
cat("allow_par:             TRUE (set FALSE to reduce variation from parallel execution)\n")
cat("parallel_over:         resamples\n\n")
cat("To verify: re-run from top and compare rf_singlebestcv$recipe\n")
cat("Expected best recipe should be identical across runs.\n")
cat("If it still varies, set allow_par = FALSE in ctrl.\n")
cat("=============================\n\n")

# Save a reproducibility log containing the search settings, generated grid,
# selected configuration and R session information.
# Save reproducibility log
repro_log_file <- file.path(out_dir, paste0("rf_reproducibility_log_", run_id, ".txt"))
sink(repro_log_file)
cat("Random Forest Space-Filling Search — Reproducibility Log\n")
cat("==========================================================\n")
cat("Run ID:              ", run_id, "\n")
cat("Date:                ", as.character(Sys.time()), "\n")
cat("R version:           ", R.version.string, "\n")
cat("GRID_SEED:           ", GRID_SEED, "\n")
cat("Global seed:          1238\n")
cat("RNG kind:             L'Ecuyer-CMRG\n")
cat("allow_par:            TRUE\n")
cat("parallel_over:        resamples\n")
cat("Grid rows:           ", nrow(randforest_param_grid_sfill), "\n\n")
cat("First 5 grid rows:\n")
print(head(randforest_param_grid_sfill, 5))
cat("\nBest single config:\n")
print(rf_singlebestcv)
cat("\nSessionInfo:\n")
sessionInfo()
sink()

cat("Reproducibility log saved to:", repro_log_file, "\n")


# ============================================================================
# RANDOM FOREST: REFINED HYPERPARAMETER SEARCH
# This section refines the Random Forest hyperparameters around the best
# configuration identified during the initial space-filling search. A narrow
# search is performed around the selected mtry, number of trees and minimum
# node size before the refined models are evaluated on the training and
# independent test datasets.
# Refined Random Forest updated: (17th Feb)
# ============================================================================

# Purpose:
# 1) Take "singlebest" from your space-filling run
# 2) Build a local grid around the single best initial configuration
# 3) tune_grid() on that local grid
# 4) Save CV metrics (means + SE) + predictions
# 5) Identify the SINGLE BEST refined model (by RMSE, then MAE, then RSQ)
# 6) Fit on FULL train
# 7) Predict TRAIN + TEST, compute 6 metrics on TEST, save outputs
#

# ========================================================================
# REPRODUCIBILITY DOCUMENTATION
# ========================================================================
# Fixed and deterministic seeds are used to improve reproducibility:
#
# 1. RNGkind: Set ONCE at the top of the ENTIRE script (not here).
#    Do NOT repeat RNGkind() here — it lives at the global script top.
#
# 2. Recipe-specific seeds: Generated deterministically from recipe names
#    - CV refinement fold fitting: fold_seed = seed_for_refine(recipe) + i*100
#    - Final model fit:           rf_seed_final = seed_for_refine(recipe) + 2000
#    NOTE: set.seed() is placed IMMEDIATELY before fit(), after all
#    prep()/bake() calls, so RNG-consuming preprocessing does not
#    shift the seed away from the model fitting step.
#
# 3. All seeds are saved in output files for verification.
#
# ========================================================================

# -----------------------------
# Preconditions (fail fast)
# -----------------------------
# Confirm that the objects produced during the initial Random Forest search
# and the required training, test and cross-validation objects are available.
stopifnot(exists("rf_singlebestcv"))
stopifnot(exists("preproc.randf"))
stopifnot(exists("ntl_train"))
stopifnot(exists("ntl_test"))
stopifnot(exists("ntl_vfolds_cv"))
stopifnot(exists("out_dir"))
stopifnot(exists("run_id"))

# Confirm that the initial best-model table contains the hyperparameters
# required to construct the refined search grid.
req_cols <- c("recipe","mtry","trees","min_n")
miss <- setdiff(req_cols, names(rf_singlebestcv))
if (length(miss) > 0) stop("rf_singlebest missing: ", paste(miss, collapse = ", "))

# Define the regression and poverty-classification metrics expected during
# the refined cross-validation analysis.
needed_metrics <- c("rmse","mae","rsq","accuracy_033","sensitivity_033","specificity_033","f1_033")

# FIX A: Do NOT repeat RNGkind() here — it is set once at the top of the
# entire script. Repeating it here is misleading and could mask cases where
# it was accidentally changed by another algorithm's section.
#
# FIX B: Do NOT call set.seed() here at the section top. A single
# set.seed(124) before the loop is consumed by prep()/bake()/grid helpers
# before it can protect any model fitting. Per-operation seeds below handle
# this correctly instead.

# -----------------------------
# Seed function (unchanged)
# -----------------------------
# Generate a deterministic seed for each preprocessing recipe during refinement.
seed_for_refine <- function(recipe_name, base = 124L) {
  as.integer((base + sum(utf8ToInt(paste0("REFINE_RF_", recipe_name)))) %% .Machine$integer.max)
}

# -----------------------------
# Output files (unchanged)
# -----------------------------
# Define separate output files for refined CV metrics, fold predictions,
# selected configurations and final train/test results.
refine_metrics_file     <- file.path(out_dir, paste0("rf_refine_cv_metrics_",      run_id, ".csv"))
refine_preds_file       <- file.path(out_dir, paste0("rf_refine_cv_predictions_",  run_id, ".csv"))
refined_best_file       <- file.path(out_dir, paste0("rf_refine_bestrows_",        run_id, ".csv"))
final_train_preds_file  <- file.path(out_dir, paste0("rf_refine_FINAL_train_predictions_", run_id, ".csv"))
final_test_preds_file   <- file.path(out_dir, paste0("rf_refine_FINAL_test_predictions_",  run_id, ".csv"))
final_test_metrics_file <- file.path(out_dir, paste0("rf_refine_FINAL_test_metrics_",      run_id, ".csv"))

# Remove previous run-specific refined outputs so the current search starts fresh.
# Start fresh (comment out for resume)
for (f in c(refine_metrics_file, refine_preds_file, refined_best_file,
            final_train_preds_file, final_test_preds_file, final_test_metrics_file)) {
  if (file.exists(f)) file.remove(f)
}

# FIX D: Guard images_dir — defined in space-filling section but needed here
# if this section is ever run standalone or after a session restart.
# Ensure that the directory used to save fitted Random Forest models exists.
if (!exists("images_dir")) {
  images_dir <- file.path(out_dir, "images")
}
dir.create(images_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# Helpers (ALL UNCHANGED)
# -----------------------------
# Helper functions used to keep integer and numeric hyperparameters within
# their permitted ranges.
clamp_int <- function(x, lo, hi) as.integer(pmax(lo, pmin(hi, as.integer(round(x)))))
clamp_num <- function(x, lo, hi) pmax(lo, pmin(hi, as.numeric(x)))

# Return a usable standard deviation when calculating standardised distances
# between candidate hyperparameter combinations.
safe_sd <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) 1 else s
}

# Calculate poverty-classification performance from continuous MPI predictions.
# Poor is treated as the positive class using the study threshold of 0.3333.
compute_class_metrics <- function(truth, estimate, threshold = 0.3333) {
  actual <- factor(ifelse(truth >= threshold, "poor", "non_poor"), levels = c("poor","non_poor"))
  pred   <- factor(ifelse(estimate >= threshold, "poor", "non_poor"), levels = c("poor","non_poor"))
  cm <- table(Actual = actual, Predicted = pred)
  
  tp <- if ("poor"     %in% rownames(cm) && "poor"     %in% colnames(cm)) cm["poor",    "poor"]     else 0
  tn <- if ("non_poor" %in% rownames(cm) && "non_poor" %in% colnames(cm)) cm["non_poor","non_poor"] else 0
  fp <- if ("non_poor" %in% rownames(cm) && "poor"     %in% colnames(cm)) cm["non_poor","poor"]     else 0
  fn <- if ("poor"     %in% rownames(cm) && "non_poor" %in% colnames(cm)) cm["poor",    "non_poor"] else 0
  
  acc  <- (tp + tn) / max(1, sum(cm))
  sen  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  spe  <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  f1   <- if (is.finite(prec) && is.finite(sen) && (prec + sen) > 0) 2 * (prec * sen) / (prec + sen) else NA_real_
  
  list(cm = cm, accuracy_033 = acc, sensitivity_033 = sen, specificity_033 = spe, f1_033 = f1)
}

# Construct a local Random Forest grid around the best initial configuration.
# mtry and minimum node size are varied by two units in either direction,
# while the number of trees is varied by 200 in steps of 50.
make_local_rf_grid <- function(best_row, n_pred,
                               mtry_span  = 2,
                               trees_span = 200,
                               min_n_span = 2) {
  mtry0  <- as.integer(round(best_row$mtry))
  trees0 <- as.integer(round(best_row$trees))
  min_n0 <- as.integer(round(best_row$min_n))
  
  mtry_vals  <- clamp_int(seq(mtry0  - mtry_span,  mtry0  + mtry_span,  by = 1),  1L,    n_pred)
  trees_vals <- clamp_int(seq(trees0 - trees_span, trees0 + trees_span, by = 50), 25L,  10000L)
  min_n_vals <- clamp_int(seq(min_n0 - min_n_span, min_n0 + min_n_span, by = 1),  1L,    100L)
  
  tidyr::expand_grid(mtry = mtry_vals, trees = trees_vals, min_n = min_n_vals) %>%
    dplyr::distinct()
}

# Restrict large local grids to a maximum of 70 configurations.
# Candidate configurations closest to the initial best model are retained.
restrict_local_grid <- function(local_grid, best_row, target_n = 70) {
  if (nrow(local_grid) <= target_n) return(local_grid)
  
  cols <- setdiff(intersect(names(local_grid), names(best_row)), "recipe")
  
  g <- local_grid
  for (cc in cols) {
    if (is.numeric(g[[cc]])) {
      g[[paste0(cc, "_z")]] <- (g[[cc]] - as.numeric(best_row[[cc]])) / safe_sd(g[[cc]])
    } else {
      g[[paste0(cc, "_z")]] <- 0
    }
  }
  
  zcols  <- grep("_z$", names(g), value = TRUE)
  g$dist <- sqrt(rowSums((as.matrix(g[, zcols, drop = FALSE]))^2))
  
  out2 <- g %>% arrange(dist) %>% slice_head(n = target_n) %>% select(names(local_grid))
  
  best_grid_cols <- intersect(names(local_grid), names(best_row))
  best_candidate <- local_grid
  for (cc in best_grid_cols) best_candidate[[cc]] <- best_row[[cc]]
  
  bind_rows(best_candidate[1, names(local_grid), drop = FALSE], out2) %>%
    distinct() %>%
    slice_head(n = target_n)
}

# Calculate the standard error of a performance measure across CV folds.
se_of <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

# -----------------------------
# MAIN: refined loop (structure UNCHANGED)
# -----------------------------
# Store the best refined configuration obtained around the selected initial model.
all_best <- list()

# Refine the single best configuration carried forward from the initial search.
for (k in seq_len(nrow(rf_singlebestcv))) {
  
  rec_name <- as.character(rf_singlebestcv$recipe[[k]])
  best_row <- rf_singlebestcv[k, , drop = FALSE]
  
  cat("\n=============================\n")
  cat("REFINING AROUND BEST:", rec_name, "\n")
  cat("=============================\n")
  
  rec <- preproc.randf[[rec_name]]
  if (is.null(rec)) stop("Recipe not found: ", rec_name)
  
  # Prepare the current recipe and count the numeric predictors remaining after
  # preprocessing. This ensures that mtry does not exceed the available predictors.
  # Count numeric predictors post-bake (caps mtry) — unchanged
  prep_rec_tmp <- prep(rec, training = ntl_train)
  train_baked  <- bake(prep_rec_tmp, new_data = ntl_train)
  pnames <- setdiff(names(train_baked), "mpi")
  pnames <- pnames[sapply(train_baked[, pnames, drop = FALSE], is.numeric)]
  n_pred <- length(pnames)
  if (n_pred < 1) stop("No numeric predictors after baking for recipe: ", rec_name)
  
  # Build the complete local grid around the initial best configuration and
  # restrict it to a maximum of 70 candidate combinations.
  # Build local grid — unchanged
  local_grid_full <- make_local_rf_grid(best_row, n_pred = n_pred) %>%
    mutate(mtry = clamp_int(mtry, 1L, n_pred))
  
  local_grid <- restrict_local_grid(local_grid_full, best_row = best_row, target_n = 70)
  
  # Apply a deterministic ordering to the refined hyperparameter combinations.
  # Deterministic ordering — unchanged
  ord_cols   <- intersect(c("mtry","trees","min_n"), names(local_grid))
  local_grid <- local_grid %>% arrange(across(all_of(ord_cols)))
  
  cat("Local grid size:", nrow(local_grid),
      " (full was", nrow(local_grid_full), "; n_pred=", n_pred, ")\n")
  stopifnot(nrow(local_grid) <= 70)
  
  # FIX B: Removed set.seed(seed_for_refine(rec_name)) from here.
  # This seed was consumed by pmap_dfr overhead and prep()/bake() inside
  # the fold loop before reaching fit(). The per-fold seed below replaces it.
  
  # Initialise progress tracking for the local grid search.
  cat("Starting grid search...\n")
  grid_counter <- 0
  total_grid   <- nrow(local_grid)
  
  # Evaluate each candidate combination of mtry, trees and minimum node size.
  grid_results <- purrr::pmap_dfr(
    local_grid %>% dplyr::select(mtry, trees, min_n),
    function(mtry, trees, min_n) {
      
      grid_counter <<- grid_counter + 1
      
      mtry  <- as.integer(mtry)
      trees <- as.integer(trees)
      min_n <- as.integer(min_n)
      
      cat("\n[Config ", grid_counter, "/", total_grid, "] Testing: mtry=", mtry,
          ", trees=", trees, ", min_n=", min_n, "\n", sep = "")
      
      # Evaluate the current hyperparameter configuration across all predefined
      # cross-validation folds.
      fold_metrics <- purrr::map_dfr(seq_along(ntl_vfolds_cv$splits), function(i) {
        
        cat("     Fold ", i, "/", length(ntl_vfolds_cv$splits), "... ", sep = "")
        
        # Extract the analysis and assessment data for the current CV fold.
        sp <- ntl_vfolds_cv$splits[[i]]
        tr <- rsample::analysis(sp)
        va <- rsample::assessment(sp)
        
        # Prepare the recipe within the current training fold and apply the same
        # preprocessing to its corresponding validation fold.
        # prep/bake FIRST — these consume RNG state
        rec_p <- prep(rec, training = tr, retain = TRUE)
        tr_b  <- bake(rec_p, new_data = tr)
        va_b  <- bake(rec_p, new_data = va)
        
        y_va <- va_b$mpi
        
        # FIX C (CORE): set.seed() placed AFTER prep/bake, IMMEDIATELY before fit()
        # Original code set the seed before prep/bake, meaning preprocessing
        # consumed the RNG state and the model fit started from a shifted position.
        # Create a deterministic fold seed for the current preprocessing recipe and fold.
        fold_seed <- seed_for_refine(rec_name) + i * 100L
        
        # Define the Random Forest model using the current refined hyperparameters.
        # The same fold seed is passed to ranger to control its internal randomness.
        rf_spec <- parsnip::rand_forest(
          mtry  = mtry,
          trees = trees,
          min_n = min_n
        ) %>%
          parsnip::set_engine("ranger", importance = "impurity", seed = fold_seed) %>%   # seeds C++ layer
          parsnip::set_mode("regression")
        
        # Build the fold-specific workflow using the baked predictor data.
        rf_wflow <- workflows::workflow() %>%
          workflows::add_model(rf_spec) %>%
          workflows::add_formula(mpi ~ .)
        
        # Set the fold seed immediately before fitting the Random Forest model.
        set.seed(fold_seed)   # FIX C: immediately before fit(), nowhere else
        rf_fit <- fit(rf_wflow, data = tr_b)
        
        # Generate MPI predictions for the validation observations.
        pred <- predict(rf_fit, new_data = va_b)$.pred
        
        # Calculate regression and poverty-classification performance for the current fold.
        rmse <- yardstick::rmse_vec(y_va, pred)
        mae  <- yardstick::mae_vec(y_va, pred)
        rsq  <- suppressWarnings(yardstick::rsq_vec(y_va, pred))
        cls  <- compute_class_metrics(y_va, pred, threshold = 0.3333)
        
        cat("RMSE=", sprintf("%.4f", rmse), "\n", sep = "")
        
        # Recover the original training-row positions for the validation observations
        # so that out-of-fold predictions can be traced back to the source data.
        # stable original row id from ntl_train
        orig_row <- match(rownames(va), rownames(ntl_train))
        
        if (anyNA(orig_row)) {
          orig_row <- seq_along(y_va)
        }
        
        # Retain the resample and repeat identifiers associated with the current fold.
        resample_id <- if ("id" %in% names(ntl_vfolds_cv)) {
          as.character(ntl_vfolds_cv$id[[i]])
        } else {
          paste0("Fold", i)
        }
        
        repeat_id <- if ("id2" %in% names(ntl_vfolds_cv)) {
          as.character(ntl_vfolds_cv$id2[[i]])
        } else {
          NA_character_
        }
        
        # Save fold predictions — unchanged
        # Save the out-of-fold predictions together with the recipe, hyperparameters,
        # fold identifiers and seed used for the model fit.
        fold_pred_tbl <- tibble(
          run_id = run_id,
          recipe = rec_name,
          grid_mtry = mtry,
          grid_trees = trees,
          grid_min_n = min_n,
          #grid_tree_depth = tree_depth,
          #grid_learn_rate = learn_rate,
          #grid_sample_size = sample_size,
          fold = i,
          id = resample_id,
          id2 = repeat_id,
          fold_seed = fold_seed,
          .row = orig_row,
          row_id = seq_along(y_va),   # keep your current feature
          mpi_true = y_va,
          mpi_pred = pred
        )
        
        write_csv(fold_pred_tbl, refine_preds_file, append = file.exists(refine_preds_file))
        
        # Return the performance measures calculated for the current CV fold.
        tibble(
          fold            = i,
          rmse            = rmse,
          mae             = mae,
          rsq             = rsq,
          accuracy_033    = cls$accuracy_033,
          sensitivity_033 = cls$sensitivity_033,
          specificity_033 = cls$specificity_033,
          f1_033          = cls$f1_033
        )
      })
      
      # Calculate the mean RMSE and R-squared values for progress reporting.
      mean_rmse <- mean(fold_metrics$rmse, na.rm = TRUE)
      mean_rsq  <- mean(fold_metrics$rsq,  na.rm = TRUE)
      
      cat("     >> CV Complete: Mean RMSE=", sprintf("%.4f", mean_rmse),
          ", Mean R²=", sprintf("%.4f", mean_rsq), "\n\n", sep = "")
      
      # Summarise across folds — unchanged
      # Summarise regression and classification performance across the CV folds.
      # Both the mean and standard error are retained for each performance measure.
      tibble(
        recipe              = rec_name,
        mtry                = mtry,
        trees               = trees,
        min_n               = min_n,
        rmse                = mean(fold_metrics$rmse,            na.rm = TRUE),
        rmse_se             = se_of(fold_metrics$rmse),
        mae                 = mean(fold_metrics$mae,             na.rm = TRUE),
        mae_se              = se_of(fold_metrics$mae),
        rsq                 = mean(fold_metrics$rsq,             na.rm = TRUE),
        rsq_se              = se_of(fold_metrics$rsq),
        accuracy_033        = mean(fold_metrics$accuracy_033,    na.rm = TRUE),
        accuracy_033_se     = se_of(fold_metrics$accuracy_033),
        sensitivity_033     = mean(fold_metrics$sensitivity_033, na.rm = TRUE),
        sensitivity_033_se  = se_of(fold_metrics$sensitivity_033),
        specificity_033     = mean(fold_metrics$specificity_033, na.rm = TRUE),
        specificity_033_se  = se_of(fold_metrics$specificity_033),
        f1_033              = mean(fold_metrics$f1_033,          na.rm = TRUE),
        f1_033_se           = se_of(fold_metrics$f1_033)
      )
    }
  )
  
  # Save the completed refined CV results for the current preprocessing recipe.
  # Write all grid CV results — unchanged
  write_csv(grid_results, refine_metrics_file, append = file.exists(refine_metrics_file))
  
  # Select the best refined configuration for the current recipe from the
  # successfully evaluated candidate models.
  # Pick best for this recipe — unchanged
  best_k <- grid_results %>%
    filter(is.finite(rmse)) %>%
    arrange(rmse, mae, desc(rsq)) %>%
    slice(1)
  
  all_best[[rec_name]] <- best_k
  cat(">> Best refined for ", rec_name, ":\n", sep = "")
  print(best_k)
}

# -----------------------------
# Save final best rows (unchanged)
# -----------------------------
# Combine and save the best configuration obtained from the refined local search.
rf_refinedsinglebestcv <- bind_rows(all_best)
write_csv(rf_refinedsinglebestcv, refined_best_file)

cat("\n=============================\n")
cat("REFINEMENT COMPLETE\n")
cat("Best models saved to:", refined_best_file, "\n")
cat("=============================\n")

# =============================
# FINAL STAGE: Train on full training set and predict on test set
# (structure UNCHANGED — only images_dir guard added via FIX D above)
# =============================
cat("\n=============================\n")
cat("FINAL STAGE: Training best models on full training set\n")
cat("=============================\n")

# Store the final train/test evaluation results for the refined models.
final_test_results <- list()

# Refit the best refined configuration using the complete training dataset.
for (k in seq_len(nrow(rf_refinedsinglebestcv))) {
  
  rec_name    <- as.character(rf_refinedsinglebestcv$recipe[[k]])
  best_config <- rf_refinedsinglebestcv[k, , drop = FALSE]
  
  cat("\n>> Training final model for:", rec_name, "\n")
  
  # NOTE: set.seed(seed_for_refine(rec_name) + 1000L) was in the original here.
  # It is NOT needed: rf_seed_final below (+ 2000) is set immediately before
  # fit(), which is the correct location. The +1000 seed had nothing between
  # it and the next set.seed(+2000), so it was redundant. Removed for clarity.
  
  # Extract the selected refined hyperparameters for the current recipe.
  mtry_best  <- as.integer(best_config$mtry)
  trees_best <- as.integer(best_config$trees)
  min_n_best <- as.integer(best_config$min_n)
  
  cat("   Best config: mtry=", mtry_best, ", trees=", trees_best,
      ", min_n=", min_n_best, "\n", sep = "")
  
  # Retrieve the preprocessing recipe associated with the refined configuration.
  rec <- preproc.randf[[rec_name]]
  
  # Prep and bake FIRST — these consume RNG state
  # Prepare the selected recipe using the complete training dataset and apply
  # the fitted preprocessing steps to both training and independent test data.
  rec_prep_final    <- prep(rec, training = ntl_train, retain = TRUE)
  train_baked_final <- bake(rec_prep_final, new_data = ntl_train)
  test_baked_final  <- bake(rec_prep_final, new_data = ntl_test)
  
  y_train_final <- train_baked_final$mpi
  y_test_final  <- test_baked_final$mpi
  
  # Create a deterministic seed for fitting the final refined Random Forest model.
  rf_seed_final <- seed_for_refine(rec_name) + 2000L
  
  # Define the final Random Forest specification using the selected refined parameters.
  rf_spec_final <- parsnip::rand_forest(
    mtry  = mtry_best,
    trees = trees_best,
    min_n = min_n_best
  ) %>%
    parsnip::set_engine("ranger",  importance = "impurity", seed = rf_seed_final) %>%   # seeds C++ layer
    parsnip::set_mode("regression")
  
  # Build the final workflow from the fitted Random Forest specification.
  rf_wflow_final <- workflows::workflow() %>%
    workflows::add_model(rf_spec_final) %>%
    workflows::add_formula(mpi ~ .)
  
  cat("   Training final model (seed=", rf_seed_final, ")...\n", sep = "")
  # Set the final model seed immediately before fitting on the complete training data.
  set.seed(rf_seed_final)   # immediately before fit() — unchanged position, correct
  final_rf_fit <- fit(rf_wflow_final, data = train_baked_final)
  
  # Save bundled model — unchanged
  # Define a run-specific file for saving the fitted refined model.
  model_file <- file.path(images_dir,
                          paste0("final_refined_", rec_name, "_", run_id, ".rds"))
  
  # Bundle and save the fitted refined Random Forest model if it has not
  # already been saved for the current run.
  if (!file.exists(model_file)) {
    bundled_refined_model <- bundle(final_rf_fit)
    saveRDS(bundled_refined_model, model_file)
    message("Model saved to: ", model_file)
  } else {
    message("Model already exists, skipping save: ", model_file)
  }
  
  # Generate continuous MPI predictions for the complete training and test datasets.
  # Predict — unchanged
  pred_train_final <- predict(final_rf_fit, new_data = train_baked_final)$.pred
  pred_test_final  <- predict(final_rf_fit, new_data = test_baked_final)$.pred
  
  # Calculate regression and poverty-classification metrics for both datasets.
  # Metrics — unchanged
  rmse_train <- yardstick::rmse_vec(y_train_final, pred_train_final)
  mae_train  <- yardstick::mae_vec(y_train_final, pred_train_final)
  rsq_train  <- suppressWarnings(yardstick::rsq_vec(y_train_final, pred_train_final))
  cls_train  <- compute_class_metrics(y_train_final, pred_train_final, threshold = 0.3333)
  
  rmse_test  <- yardstick::rmse_vec(y_test_final, pred_test_final)
  mae_test   <- yardstick::mae_vec(y_test_final, pred_test_final)
  rsq_test   <- suppressWarnings(yardstick::rsq_vec(y_test_final, pred_test_final))
  cls_test   <- compute_class_metrics(y_test_final, pred_test_final, threshold = 0.3333)
  
  cat("   Train RMSE:", round(rmse_train, 4), " | Test RMSE:", round(rmse_test, 4), "\n")
  cat("   Train R²:",   round(rsq_train,  4), " | Test R²:",  round(rsq_test,  4), "\n")
  
  # Save training predictions — unchanged
  # Save the training predictions together with the selected hyperparameters
  # and final Random Forest seed.
  train_preds_tbl <- tibble(
    run_id   = run_id, recipe = rec_name,
    mtry     = mtry_best, trees = trees_best, min_n = min_n_best,
    rf_seed  = rf_seed_final,
    row_id   = seq_along(y_train_final),
    mpi_true = y_train_final, mpi_pred = pred_train_final
  )
  write_csv(train_preds_tbl, final_train_preds_file,
            append = file.exists(final_train_preds_file))
  
  # Save test predictions — unchanged
  # Save the corresponding independent-test predictions.
  test_preds_tbl <- tibble(
    run_id   = run_id, recipe = rec_name,
    mtry     = mtry_best, trees = trees_best, min_n = min_n_best,
    rf_seed  = rf_seed_final,
    row_id   = seq_along(y_test_final),
    mpi_true = y_test_final, mpi_pred = pred_test_final
  )
  write_csv(test_preds_tbl, final_test_preds_file,
            append = file.exists(final_test_preds_file))
  
  # Save test metrics — unchanged
  # Store the complete training and independent-test performance measures
  # for the current refined model.
  test_metrics_tbl <- tibble(
    run_id = run_id, recipe = rec_name,
    mtry   = mtry_best, trees = trees_best, min_n = min_n_best,
    rf_seed = rf_seed_final,
    train_rmse            = rmse_train, train_mae  = mae_train,  train_rsq  = rsq_train,
    train_accuracy_033    = cls_train$accuracy_033,
    train_sensitivity_033 = cls_train$sensitivity_033,
    train_specificity_033 = cls_train$specificity_033,
    train_f1_033          = cls_train$f1_033,
    test_rmse             = rmse_test,  test_mae   = mae_test,   test_rsq   = rsq_test,
    test_accuracy_033     = cls_test$accuracy_033,
    test_sensitivity_033  = cls_test$sensitivity_033,
    test_specificity_033  = cls_test$specificity_033,
    test_f1_033           = cls_test$f1_033
  )
  write_csv(test_metrics_tbl, final_test_metrics_file,
            append = file.exists(final_test_metrics_file))
  
  final_test_results[[rec_name]] <- test_metrics_tbl
}

cat("\n=============================\n")
cat("FINAL EVALUATION COMPLETE\n")
cat("Training predictions saved to:", final_train_preds_file, "\n")
cat("Test predictions saved to:",     final_test_preds_file,  "\n")
cat("Test metrics saved to:",         final_test_metrics_file, "\n")
cat("=============================\n")

cat("\n=============================\n")
cat("FINAL TEST SET PERFORMANCE SUMMARY\n")
cat("=============================\n")
# Combine the final results across preprocessing recipes and display the
# main independent-test performance measures.
final_test_df <- bind_rows(final_test_results)
print(final_test_df %>% select(recipe, test_rmse, test_mae, test_rsq,
                               test_accuracy_033, test_f1_033))
cat("\n")

# ========================================================================
# REPRODUCIBILITY: Save session information (unchanged)
# ========================================================================
# Save the R session and random-number settings used for the refined search.
# This provides a reproducibility record for the completed analysis.
session_info_file <- file.path(out_dir, paste0("rf_refine_sessionInfo_", run_id, ".txt"))
cat("\n=============================\n")
cat("Saving session info for reproducibility...\n")
cat("Session info saved to:", session_info_file, "\n")
cat("=============================\n")

sink(session_info_file)
cat("Random Forest Refinement Script - Session Information\n")
cat("======================================================\n")
cat("Run ID:", run_id, "\n")
cat("Date:", as.character(Sys.time()), "\n")
cat("R Version:", R.version.string, "\n\n")
cat("Random Number Generator:\n")
print(RNGkind())
cat("\nBase seed: 124\n")
cat("Seed function: seed_for_refine(recipe_name)\n")
cat("fold_seed:     seed_for_refine(recipe) + i * 100\n")
cat("rf_seed_final: seed_for_refine(recipe) + 2000\n\n")
cat("======================================================\n\n")
sessionInfo()
sink()

cat("\n=============================\n")
cat("ALL PROCESSING COMPLETE\n")
cat("=============================\n")



#################Preparing to plot##################
# if(rf_singlebestcv$rmse <= rf_refinedsinglebestcv$rmse){

# Compare the best initial-stage and refined-stage Random Forest configurations.
# The common model-selection function uses RMSE first, followed by MAE and
# R-squared when the preceding measures are effectively tied.
## Identifying the best model from the initial and refined stages.
comparison <- is_better_model(
  rf_singlebestcv,
  rf_refinedsinglebestcv
)

# Retain the initial space-filling model when it performs better according
# to the predefined model-selection criteria.
if (comparison$better) {
  
  rec_name <- as.character(rf_singlebestcv$recipe[[1]])
  rec <- preproc.randf[[rec_name]]
  
  tag <- "space-filling_grid"
  
  mtry_k  <- rf_singlebestcv$mtry[[1]]
  trees_k <- rf_singlebestcv$trees[[1]]
  min_n_k <- rf_singlebestcv$min_n[[1]]
  
  rfmodelfit <- firstround_fits[[rec_name]]
  
  selected_train_metrics <- firstround_train_metrics_list[[rec_name]]
  selected_test_metrics  <- firstround_test_metrics_list[[rec_name]]
  
  # Train Dataset metrics
  rf_train_rmse <- selected_train_metrics %>% filter(dataset = "train") %>% dplyr::select(rmse)
  rf_train_mae <- selected_train_metrics %>% filter(dataset = "train") %>% dplyr::select(mae)
  rf_train_rsq <- selected_train_metrics %>% filter(dataset = "train") %>% dplyr::select(rsq)
  # rf_train_cor <- selected_train_metrics %>% filter(dataset = "train") %>% dplyr::select(cor)
  
  # Test Dataset metrics
  rf_test_rmse <- selected_test_metrics %>% filter(dataset = "test") %>% dplyr::select(rmse)
  rf_test_mae <- selected_test_metrics %>% filter(dataset = "test") %>% dplyr::select(mae)
  rf_test_rsq <- selected_test_metrics %>% filter(dataset = "test") %>% dplyr::select(rsq)
  # rf_test_cor <- selected_test_metrics %>% filter(dataset = "test") %>% dplyr::select(cor)
  
  message(
    "Using initial search model",
    "\nRecipe: ", rec_name,
    "\nSelection criterion: ", comparison$criterion,
    "\nInitial value: ", comparison$model1_value,
    "\nRefined value: ", comparison$model2_value
  )
  
} else {
  
  # Otherwise, retain the model selected from the refined hyperparameter search.
  rec_name <- as.character(rf_refinedsinglebestcv$recipe[[1]])
  rec <- preproc.randf[[rec_name]]
  
  tag <- "refined_grid"
  
  mtry_k  <- rf_refinedsinglebestcv$mtry[[1]]
  trees_k <- rf_refinedsinglebestcv$trees[[1]]
  min_n_k <- rf_refinedsinglebestcv$min_n[[1]]
  
  rfmodelfit <- final_rf_fit
  
  # Train Dataset metrics
  rf_train_rmse <- test_metrics_tbl %>% dplyr::select(train_rmse)
  rf_train_mae <- test_metrics_tbl %>% dplyr::select(train_mae)
  rf_train_rsq <- test_metrics_tbl %>% dplyr::select(train_rsq)
  # rf_train_cor <- test_metrics_tbl %>% dplyr::select(train_rmse)
  
  # Test Dataset metrics
  rf_test_rmse <- test_metrics_tbl %>% dplyr::select(test_rmse)
  rf_test_mae <- test_metrics_tbl %>% dplyr::select(test_mae)
  rf_test_rsq <- test_metrics_tbl %>% dplyr::select(test_rsq)
  # rf_test_cor <- test_metrics_tbl  %>% dplyr::select(test_cor)
  
  message(
    "Using refined search model",
    "\nRecipe: ", rec_name,
    "\nSelection criterion: ", comparison$criterion,
    "\nInitial value: ", comparison$model1_value,
    "\nRefined value: ", comparison$model2_value
  )
}

# Save the ultimately selected Random Forest model for the later analysis stage.
# Write the ultimate best model to file
# Save bundled model — unchanged
ulmodel_file <- file.path(images_dir,
                          paste0("ultimate_best_", tag, "_", rec_name, "_", run_id, ".rds"))

bundled_ulbest_model <- bundle(rfmodelfit)
saveRDS(bundled_ulbest_model, ulmodel_file)
message("Model saved to: ", ulmodel_file)


# Generate the final analysis predictions using the correct data representation.
# The initial-stage workflow contains the recipe internally, while the refined
# workflow was fitted directly to baked data.
if (tag == "space-filling_grid") {
  
  set.seed(124)
  preds.train <- predict(rfmodelfit, new_data = ntl_train) %>%
    bind_cols(ntl_train %>% dplyr::select(mpi)) %>%
    mutate(
      recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k
    )
  
  set.seed(124)
  preds.test <- predict(rfmodelfit, new_data = ntl_test) %>%
    bind_cols(ntl_test %>% dplyr::select(mpi)) %>%
    mutate(
      recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k
    )
  
} else {
  
  rec_prep_final    <- prep(rec, training = ntl_train, retain = TRUE)
  train_baked_final <- bake(rec_prep_final, new_data = ntl_train)
  test_baked_final  <- bake(rec_prep_final, new_data = ntl_test)
  
  set.seed(124)
  preds.train <- predict(rfmodelfit, new_data = train_baked_final) %>%
    bind_cols(ntl_train %>% dplyr::select(mpi)) %>%
    mutate(
      recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k
    )
  
  set.seed(124)
  preds.test <- predict(rfmodelfit, new_data = test_baked_final) %>%
    bind_cols(ntl_test %>% dplyr::select(mpi)) %>%
    mutate(
      recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k
    )
}


# Calculate training residuals and create the residual dataset required for
# the later Random Forest diagnostic and bias analyses.
#Preparing variable for model analysis - training dataset
rf_resid_train <- preds.train$mpi - preds.train$.pred

#final_resid_data_rf
#.fitted = round(as.numeric(preds.train$.pred),3)
train_resid_data_rf <- tibble(
  .fitted = preds.train$.pred,
  .resid  = preds.train$mpi - preds.train$.pred,
  mpi     = preds.train$mpi,
  .std_resid = as.numeric(scale(rf_resid_train)),  # standardized residuals
  Set = "Train",
  model = "Random Forest"
)


# Retain the observed test MPI values for later model analysis.
#Extract the actual test mpi data
actual_test <- ntl_test %>% select("mpi")

# Calculate test residuals and create the corresponding residual dataset for
# the independent test observations.
#Preparing variable for model analysis - test dataset
rf_resid_test <- preds.test$mpi - preds.test$.pred

test_resid_data_rf <- tibble(
  .fitted = preds.test$.pred,
  .resid  = preds.test$mpi - preds.test$.pred,
  mpi     = preds.test$mpi,
  .std_resid = as.numeric(scale(rf_resid_test)),  # standardized residuals
  Set = "Test",
  model = "Random Forest"
)


# Save the independent-test residual dataset for later comparative analysis.
# Write the test tibble to file
file_name_rf <- file.path(model_dir, "randomforest", paste0("test_resid_data_rf", ".csv"))
write_csv(test_resid_data_rf, file_name_rf)

# Save CV results
cat("\nCV results:", file_name_rf, "\n")



#############complete new with correction######

# ===============================================
# Final Model Residual Analysis (CORRECT APPROACH)
# ===============================================

# Examine the residual behaviour of the ultimately selected Random Forest model
# using the training dataset.
cat("\n=== FINAL MODEL RESIDUAL ANALYSIS (Training Data) ===\n")

# 1. Residuals vs Fitted Values (Homoscedasticity Check)
# Plot residuals against fitted MPI values. The zero reference line and LOESS
# smoother help identify systematic residual patterns and changes in residual spread.
p1 <- ggplot(train_resid_data_rf, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Fitted Values",
    subtitle = "Final Model on Training Data - Check Homoscedasticity",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals (Actual - Predicted)"
  ) +
  theme_minimal()

# 2. Scale-Location Plot (Spread of residuals)
# Create a scale-location plot to assess whether the spread of the standardised
# residuals remains approximately constant across the fitted MPI range.
p2 <- ggplot(train_resid_data_rf, aes(x = .fitted, y = sqrt(abs(.std_resid)))) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  labs(
    title = "Scale-Location Plot",
    subtitle = "Check for Constant Variance (Homoscedasticity)",
    x = "Fitted Values",
    y = "√|Standardized Residuals|"
  ) +
  theme_minimal()

# 3. Residuals Distribution
# Examine the distribution of the training residuals and compare the observed
# density with a normal distribution having the same mean and standard deviation.
p3 <- ggplot(train_resid_data_rf, aes(x = .resid)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30,
                 fill = "lightblue", color = "black", alpha = 0.7) +
  geom_density(color = "darkred", linewidth = 1) +
  stat_function(
    fun = dnorm,
    args = list(
      mean = mean(train_resid_data_rf$.resid, na.rm = TRUE),
      sd   = sd(train_resid_data_rf$.resid, na.rm = TRUE)
    ),
    color = "blue", linetype = "dashed"
  ) +
  labs(
    title = "Distribution of Residuals",
    subtitle = "Blue dashed line shows normal distribution for comparison",
    x = "Residuals",
    y = "Density"
  ) +
  theme_minimal()

# 4. Q-Q Plot for Normality
# Create a Q-Q plot as an additional visual assessment of the residual distribution.
p4 <- ggplot(train_resid_data_rf, aes(sample = .resid)) +
  stat_qq(color = "darkblue", alpha = 0.6) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
    title = "Normal Q-Q Plot of Residuals",
    subtitle = "Check for Normality - Points should follow the red line",
    x = "Theoretical Quantiles",
    y = "Sample Quantiles"
  ) +
  theme_minimal()

# Combine all diagnostic plots ✅ fix title + caption dataset
#     title = "Random Forest Diagnostic Plots - Final Model on Training Data",

# Create a subtitle describing the Random Forest stage, preprocessing recipe and
# selected hyperparameter values used for the diagnostic plots.
rf_subtitle <- paste0(
  "Stage: ", tag, 
  " | Recipe: ", rec_name,
  " | mtry=", mtry_k,
  " | trees=", trees_k,
  " | min_n=", min_n_k
)

# Combine the four training diagnostic plots into a single panel for presentation.
diagnostic_plots <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    subtitle = rf_subtitle,
    caption = paste("Training observations:", nrow(train_resid_data_rf))
  )

# Save diagnostic plots
ggsave(paste0(plotsdir, "/", "randomforest", "/", "final_model_diagnostic_plots_rf.png"), diagnostic_plots, 
       width = 14, height = 10, dpi = 300)

cat("Final model diagnostic plots saved to: final_model_diagnostic_plots.png\n")

# ===============================================
# Detailed Residual Statistics
# ===============================================

cat("\n=== FINAL MODEL RESIDUAL STATISTICS ===\n")
# Summarise the main characteristics of the training residuals, including the
# prediction error and the proportion of residuals within selected MPI ranges.
resid_stats_rf <- train_resid_data_rf %>%
  summarise(
    Observations = n(),
    Mean_Residual = round(mean(.resid, na.rm = TRUE), 6),
    SD_Residual = round(sd(.resid, na.rm = TRUE), 4),
    Min_Residual = round(min(.resid, na.rm = TRUE), 4),
    Max_Residual = round(max(.resid, na.rm = TRUE), 4),
    MSE = round(mean(.resid^2, na.rm = TRUE), 6),
    RMSE = round(sqrt(mean(.resid^2, na.rm = TRUE)), 4),
    MAE = round(mean(abs(.resid), na.rm = TRUE), 4),
    `Within_±0.05` = paste0(round(mean(abs(.resid) <= 0.05) * 100, 1), "%"),
    `Within_±0.10` = paste0(round(mean(abs(.resid) <= 0.10) * 100, 1), "%")
  )

print(resid_stats_rf)

# ===============================================
# Homoscedasticity Test
# ===============================================

cat("\n=== HOMOSCEDASTICITY ASSESSMENT ===\n")

# Breusch-Pagan test for heteroscedasticity
# Install if needed: install.packages("lmtest")
library(lmtest)

# Create a linear model of residuals vs fitted for testing
# Apply the Breusch-Pagan test as an additional assessment of whether residual
# variance changes systematically with the fitted MPI values.
lm_resid_rf <- lm(.resid ~ .fitted, data = train_resid_data_rf)
bp_test_rf <- bptest(lm_resid_rf)

cat("Breusch-Pagan Test for Heteroscedasticity:\n")
cat("BP =", round(bp_test_rf$statistic, 4), 
    ", p-value =", format.pval(bp_test_rf$p.value, digits = 4), "\n")

if (bp_test_rf$p.value < 0.05) {
  cat("Significant evidence of heteroscedasticity (p < 0.05)\n")
  cat("   Residual variance appears non-constant\n")
} else {
  cat("No significant evidence of heteroscedasticity (p >= 0.05)\n")
  cat("   Residual variance appears constant (homoscedastic)\n")
}

# ===============================================
# Additional: Residuals vs Actual Values
# ===============================================
#title = "Residuals vs Actual MPI Values - Training Dataset",
#subtitle = "Check for systematic bias across MPI range",

# Plot training residuals against the observed MPI values to examine whether
# prediction error changes systematically across the observed MPI range.
p5 <- ggplot(train_resid_data_rf, aes(x = mpi, y = .resid)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals versus Actual MPI (Train Data)",
    subtitle = "Check for systematic bias",
    x = "Actual MPI",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "randomforest", "/", "train_residuals_vs_actual_rf.png"), p5, width = 10, height = 6, dpi = 300)
cat("Residuals vs actual values plot saved to: residuals_vs_actual.png\n")

# ===============================================
# Residual Diagnostic Summary
# ===============================================

cat("\n=== RESIDUAL DIAGNOSTIC SUMMARY ===\n")
cat("1. RESIDUAL PATTERN: Check residuals vs fitted for systematic structure around zero\n")
cat("2. RESIDUAL SPREAD: Check whether prediction error changes across fitted MPI values\n")
cat("3. SYSTEMATIC ERROR: Examine whether residual patterns suggest remaining prediction bias\n")
cat("4. RESIDUAL DISTRIBUTION: Review the shape and tails of the prediction errors\n")

# Check for patterns in residuals
# Calculate the correlation between fitted values and residuals as a simple
# numerical check for a remaining linear relationship.
train_resid_cor_rf <- cor(train_resid_data_rf$.fitted, train_resid_data_rf$.resid)
cat("Correlation between fitted and residuals:", round(train_resid_cor_rf, 6), "\n")
cat("(A value close to 0 indicates little remaining linear association)\n")

# Residuals vs Fitted (Test)
#title = "Residuals vs Fitted (Test Data)",
#subtitle = "Check generalization and bias on unseen data",
# Plot test residuals against fitted MPI values to assess prediction behaviour
# and possible systematic bias on observations not used for model fitting.
p6 <- ggplot(test_resid_data_rf, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals versus Fitted MPI (Test Data)",
    subtitle = "Check generalization and bias on unseen data",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "randomforest", "/","test_residuals_vs_fitted_rf.png"), p6, width = 10, height = 6, dpi = 300)
cat("Residuals vs Actual plot saved as: residuals_vs_fitted.png\n")

# Combine diagnostics (ensure patchwork is loaded)
#title = "Keras Model Residual Plots",
#subtitle = "Residual-based model checks",

# Combine the training residual-versus-actual plot and the test residual-versus-
# fitted plot to provide a compact comparison of model behaviour.
diagp5p6_plots <- (p5 | p6) +
  plot_annotation(
    
    caption = paste("Residual vs actual (Train) and Residual vs Fitted (Test)")
  )

# Save combined plots
ggsave(
  filename = paste0(plotsdir, "/", "randomforest", "/", "rfResidual_plots.png"),
  plot = diagp5p6_plots,
  width = 14,
  height = 10,
  dpi = 300
)
cat("Diagnostic plots saved as: rfResidual_plots.png\n")



# ===============================================
# Variable Importance Analysis
# ===============================================

# Assess the relative contribution of the predictor variables using the
# variable-importance information available from the selected Random Forest model.
cat("\n=== VARIABLE IMPORTANCE ANALYSIS (Random Forest) ===\n")

# ----------------------------
# Helper: predict wrapper for workflow fits
# ----------------------------
# Define a prediction wrapper used by the permutation-importance and partial-
# dependence functions.
rf_pred_wrapper <- function(object, newdata) {
  set.seed(124)
  as.numeric(predict(object, new_data = newdata)$.pred)
}

# ===============================================================
# Method A: Built-in ranger importance (impurity/permutation depending on engine)
# ===============================================================
cat("\n[Method A] Built-in ranger importance...\n")

# Extract the ranger engine from the ultimately selected Random Forest model so that
# the built-in importance scores can be obtained directly from the fitted model.
rf_engine <- workflows::extract_fit_engine(rfmodelfit)

# vip::vi() reads importance from ranger model when available
# Extract the built-in importance scores, rank the predictors and scale the
# importance values relative to the most influential predictor.
rf_vi_builtin <- vip::vi(rf_engine) %>%
  as_tibble() %>%
  rename(variable = Variable, importance = Importance) %>%
  arrange(desc(importance)) %>%
  mutate(rel_importance = importance / max(importance))

print(rf_vi_builtin %>% slice_head(n = 15))

write_csv(
  rf_vi_builtin,
  file.path(paste0(plotsdir, "/", "randomforest", "/", paste0("rf_variable_importance_builtin_", run_id, ".csv")))
)

# Plot the fifteen highest-ranked predictors from the built-in importance measure.
rfp_vi_builtin <- rf_vi_builtin %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variable, rel_importance), y = rel_importance)) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  coord_flip() +
  labs(
    title = "Variable Importance - Random Forest (Built-in)",
    subtitle = "ranger importance (scaled 0–1)",
    x = "Predictor Variables",
    y = "Relative Importance (0–1)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  file.path(plotsdir, "randomforest", paste0("rf_variable_importance_builtin_", run_id, ".png")),
  rfp_vi_builtin, width = 12, height = 8, dpi = 300
)

cat("Saved built-in importance CSV + plot\n")

# ===============================================================
# Method B: Permutation importance (robust, model-agnostic)
# ===============================================================
cat("\n[Method B] Permutation importance (RMSE-based)...\n")

# Use the same data representation that corresponds to the ultimately selected model.
# The initial workflow contains its preprocessing recipe, whereas the refined workflow
# was fitted directly to the baked training data.
importance_train_rf <- if (tag == "space-filling_grid") {
  ntl_train
} else {
  train_baked_final
}

# Set the seed before permutation importance so that the repeated predictor
# permutations can be reproduced as closely as possible.
set.seed(124)
# Calculate permutation importance using the increase in RMSE after each predictor
# is randomly permuted. Larger increases indicate greater predictive importance.
rf_vi_perm_tbl <- vip::vi(
  object = rfmodelfit,
  method = "permute",
  train  = importance_train_rf,
  target = "mpi",
  metric = yardstick::rmse_vec,
  smaller_is_better = TRUE,   # REQUIRED for RMSE
  pred_wrapper = rf_pred_wrapper,
  nsim = 10,                  # increase to 30–50 for final dissertation runs
  keep = TRUE
) %>%
  as_tibble() %>%
  rename(
    variable = Variable,
    importance = Importance
  ) %>%
  arrange(desc(importance)) %>%
  mutate(rel_importance = importance / max(importance))


print(rf_vi_perm_tbl %>% slice_head(n = 10))

write_csv(
  rf_vi_perm_tbl,
  file.path(plotsdir, "randomforest", paste0("rf_variable_importance_permutation_", run_id, ".csv"))
)

# Plot the fifteen highest-ranked predictors based on permutation importance.
p_vi_perm <- rf_vi_perm_tbl %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variable, rel_importance), y = rel_importance)) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  coord_flip() +
  labs(
    title = "Variable Importance - Random Forest (Permutation)",
    subtitle = "Permutation importance using RMSE increase (scaled 0–1)",
    x = "Predictor Variables",
    y = "Relative Importance (0–1)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  file.path(plotsdir, "randomforest", paste0("rf_variable_importance_permutation_", run_id, ".png")),
  p_vi_perm, width = 12, height = 8, dpi = 300
)

cat("Saved permutation importance CSV + plot\n")

# ===============================================================
# Optional: Partial Dependence for top variables (Permutation-based)
# ===============================================================
cat("\n[Optional] Partial dependence plots for top variables...\n")
# Select the six most important predictors from the permutation analysis for the
# optional partial-dependence assessment.
top_vars_rf <- rf_vi_perm_tbl %>% slice_head(n = 6) %>% pull(variable)

# Generate partial-dependence plots to show how the predicted MPI changes across
# the observed range of each selected predictor while averaging over other predictors.
pd_plots <- map(top_vars_rf, function(v) {
  pd <- pdp::partial(
    object = rfmodelfit,
    pred.var = v,
    train = importance_train_rf,
    grid.resolution = 20,
    pred.fun = function(object, newdata) rf_pred_wrapper(object, newdata)
  ) %>%
    as_tibble()
  
  # pd has columns: pred.var name + yhat
  ggplot(pd, aes(x = .data[[v]], y = yhat)) +
    geom_line(linewidth = 1) +
    geom_point(size = 1) +
    labs(
      title = paste("Partial Dependence:", v),
      x = v,
      y = "Predicted MPI"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10))
})

if (length(pd_plots) > 0) {
  pd_combined <- wrap_plots(pd_plots, ncol = 2) +
    plot_annotation(
      title = "Partial Dependence Plots - Random Forest (Top Variables)",
      subtitle = "Permutation-importance top predictors"
    )
  
  ggsave(
    file.path(plotsdir, "randomforest", paste0("rf_partial_dependence_plots_", run_id, ".png")),
    pd_combined, width = 14, height = 10, dpi = 300
  )
  cat("Saved partial dependence plots\n")
}

# ===============================================================
# Save bundle for reproducibility
# ===============================================================
# Store the variable-importance results, selected predictors and final Random Forest
# configuration together for later analysis and reproducibility.
importance_results_rf <- list(
  builtin_importance = rf_vi_builtin,
  permutation_importance = rf_vi_perm_tbl,
  top_vars = top_vars_rf,
  recipe = rec_name,
  mtry = mtry_k, trees = trees_k, min_n = min_n_k
)

saveRDS(
  importance_results_rf,
  file.path(plotsdir, "randomforest", paste0("rf_partial_variable_importance_results_", run_id, ".rds"))
)

cat("\nRF variable importance complete.\n")
cat("Saved to folder: ", file.path(plotsdir, "randomforest"), "\n")




# ===============================================
# Prediction vs Actual Scatterplots
# ===============================================

# Compare predicted and observed MPI values for the selected Random Forest model
# using both the training and independent test datasets.
cat("\n=== CREATING PREDICTION VS ACTUAL SCATTERPLOTS ===\n")

# 1. Training Data Scatterplot
cat("\nCreating training data scatterplot...\n")

# Calculate training metrics for annotation
# Calculate the main regression performance measures used to annotate the training
# predicted-versus-actual plot.
train_rmse <- rmse_vec(train_resid_data_rf$mpi, train_resid_data_rf$.fitted)
train_mae <- mae_vec(train_resid_data_rf$mpi, train_resid_data_rf$.fitted)
train_rsq <- yardstick::rsq_vec(train_resid_data_rf$mpi, train_resid_data_rf$.fitted)
train_cor <- cor(train_resid_data_rf$mpi, train_resid_data_rf$.fitted)

# Create training scatterplot
# Plot predicted MPI against observed MPI for the training dataset. The 45-degree
# line represents perfect agreement, while the fitted linear trend summarises the
# relationship between observed and predicted values.
train_scatter_plot_rf <- ggplot(train_resid_data_rf, aes(x = mpi, y = .fitted)) +
  geom_point(alpha = 0.6, color = "steelblue", size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
  coord_equal() +
  labs(
    title = "Predicted vs Actual MPI - Training Data",
    subtitle = "Random Forest Regression Model Performance",
    x = "Actual MPI",
    y = "Predicted MPI"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12)
  ) +
  # Add performance metrics as annotation
  annotate("text", 
           x = min(train_resid_data_rf$mpi), 
           y = max(train_resid_data_rf$.fitted),
           hjust = 0, vjust = 1,
           label = paste(
             sprintf("RMSE = %.4f", train_rmse),
             sprintf("MAE = %.4f", train_mae),
             sprintf("R² = %.4f", train_rsq),
             sprintf("Corr = %.4f", train_cor),
             sep = "\n"),
           size = 4, color = "darkgreen", fontface = "bold")

ggsave(paste0(plotsdir, "/", "randomforest", "/", "train_prediction_vs_actual_training_rf.png"), 
       train_scatter_plot_rf, width = 10, height = 8, dpi = 300)
cat("Training data scatterplot saved\n")

# 2. Test Data Scatterplot (if test data has actual MPI values)
if ("mpi" %in% names(test_resid_data_rf)) {
  cat("\nCreating test data scatterplot...\n")
  
  # Calculate test metrics for annotation
  # Calculate the corresponding regression performance measures for the independent
  # test dataset.
  test_rmse <- rmse_vec(test_resid_data_rf$mpi, test_resid_data_rf$.fitted)
  test_mae <- mae_vec(test_resid_data_rf$mpi, test_resid_data_rf$.fitted)
  test_rsq <- yardstick::rsq_vec(test_resid_data_rf$mpi, test_resid_data_rf$.fitted)
  test_cor <- cor(test_resid_data_rf$mpi, test_resid_data_rf$.fitted)
  
  # Create test scatterplot
  # Create the predicted-versus-actual plot for the independent test dataset using
  # the same reference line and performance annotations as the training plot.
  test_scatter_plot_rf <- ggplot(test_resid_data_rf, aes(x = mpi, y = .fitted)) +
    geom_point(alpha = 0.6, color = "purple", size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
    geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
    coord_equal() +
    labs(
      title = "Predicted vs Actual MPI - Test Data",
      subtitle = "Random Forest Model Generalization",
      x = "Actual MPI",
      y = "Predicted MPI"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 12)
    ) +
    # Add performance metrics as annotation
    annotate("text", 
             x = min(test_resid_data_rf$mpi), 
             y = max(test_resid_data_rf$.fitted),
             hjust = 0, vjust = 1,
             label = paste(
               sprintf("RMSE = %.4f", test_rmse),
               sprintf("MAE = %.4f", test_mae),
               sprintf("R² = %.4f", test_rsq),
               sprintf("Corr = %.4f", test_cor),
               sep = "\n"),
             size = 4, color = "darkred", fontface = "bold")
  
  ggsave(paste0(plotsdir, "/", "randomforest", "/", "test_prediction_vs_actual_rf.png"), 
         test_scatter_plot_rf, width = 10, height = 8, dpi = 300)
  cat("✅ Test data scatterplot saved\n")
  
  # 3. Combined Training vs Test Comparison
  cat("\nCreating combined training vs test comparison plot...\n")
  
  # Combine the training and test observations so their prediction behaviour can be
  # compared within a single plot.
  combined_scatter_data_rf <- bind_rows(train_resid_data_rf, test_resid_data_rf)
  
  # Plot the training and test predictions together to compare their agreement with
  # the observed MPI values.
  combined_scatter <- ggplot(combined_scatter_data_rf, aes(x = mpi, y = .fitted, color = Set)) +
    geom_point(alpha = 0.5, size = 1.2) +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", linewidth = 0.8) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
    scale_color_manual(values = c("Train" = "steelblue", "Test" = "purple")) +
    coord_equal() +
    labs(
      title = "Predicted vs Actual MPI - Training vs Test Comparison",
      subtitle = "Random Forest Model Performance",
      x = "Actual MPI",
      y = "Predicted MPI",
      color = "Dataset"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 12),
      legend.position = "top"
    ) +
    # Add metrics for both sets
    annotate("text", 
             x = min(combined_scatter_data_rf$mpi), 
             y = max(combined_scatter_data_rf$.fitted),
             hjust = 0, vjust = 1,
             label = paste(
               "TRAINING:",
               sprintf("RMSE = %.4f", train_rmse),
               sprintf("R² = %.4f", train_rsq),
               "",
               "TEST:",
               sprintf("RMSE = %.4f", test_rmse),
               sprintf("R² = %.4f", test_rsq),
               sep = "\n"),
             size = 3.5, color = "darkgreen", fontface = "bold")
  
  ggsave(paste0(plotsdir, "/", "randomforest", "/", "prediction_vs_actual_combined_rf.png"), 
         combined_scatter, width = 12, height = 8, dpi = 300)
  cat("Combined scatterplot saved\n")
}

# 4. Enhanced Scatterplot with Density Margins
cat("\nCreating enhanced scatterplot with density margins...\n")

# Create a density-based version of the training predicted-versus-actual plot to
# show areas where observations are concentrated.
enhanced_scatter_rf <- ggplot(train_resid_data_rf, aes(x = mpi, y = .fitted)) +
  # Add hexbin for better visualization of dense areas
  geom_hex(bins = 30, alpha = 0.7) +
  scale_fill_viridis_c(option = "plasma", name = "Count") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", color = "white", se = TRUE, linewidth = 0.8) +
  coord_equal() +
  labs(
    title = "Enhanced Prediction vs Actual MPI - Training Data",
    subtitle = "Color intensity shows point density",
    x = "Actual MPI",
    y = "Predicted MPI"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12)
  ) +
  annotate("text", 
           x = min(train_resid_data_rf$mpi), 
           y = max(train_resid_data_rf$.fitted),
           hjust = 0, vjust = 1,
           label = paste(
             sprintf("RMSE = %.4f", train_rmse),
             sprintf("MAE = %.4f", train_mae),
             sprintf("R² = %.4f", train_rsq),
             sprintf("Corr = %.4f", train_cor),
             sprintf("N = %d", nrow(train_resid_data_rf)),
             sep = "\n"),
           size = 4, color = "darkgreen", fontface = "bold")

ggsave(paste0(plotsdir, "/", "randomforest", "/", "train_prediction_vs_actual_enhanced.png"), 
       enhanced_scatter_rf, width = 10, height = 8, dpi = 300)
cat("Enhanced scatterplot saved\n")

# 5a. Scatterplot with Poverty Classification - Training Dataset
cat("\nCreating scatterplot with poverty classification...\n")

# Classify the training observations using the MPI poverty threshold and identify
# correct and incorrect poverty classifications.
poverty_scatter_traindata_rf <- train_resid_data_rf %>%
  mutate(
    Actual_Poverty = ifelse(mpi >= 0.3333, "Poor", "Non_Poor"),
    Predicted_Poverty = ifelse(.fitted >= 0.3333, "Poor", "Non_Poor"),
    Classification = case_when(
      Actual_Poverty == "Poor" & Predicted_Poverty == "Poor" ~ "True Poor",
      Actual_Poverty == "Non_Poor" & Predicted_Poverty == "Non_Poor" ~ "True Non_Poor",
      Actual_Poverty == "Poor" & Predicted_Poverty == "Non_Poor" ~ "False Non_Poor",
      Actual_Poverty == "Non_Poor" & Predicted_Poverty == "Poor" ~ "False Poor"
    )
  )

# Plot the training predictions by poverty-classification outcome. The horizontal
# and vertical threshold lines separate poor and non-poor regions.
poverty_scatter_traindata_plot_rf <- ggplot(poverty_scatter_traindata_rf, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", linewidth = 0.5) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non_Poor" = "darkgreen", 
    "False Poor" = "orange",
    "False Non_Poor" = "purple"
  )) +
  coord_equal() +
  labs(
    title = "Prediction vs Actual (Training)",
    subtitle = "Cutoff at MPI = 0.3333 | Red lines show poverty threshold",
    x = "Actual MPI",
    y = "Predicted MPI",
    color = "Classification"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12),
    legend.position = "top"
  ) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

# Calculate classification accuracy
# Summarise the number and percentage of training observations within each
# classification outcome.
classification_summary_traindata_rf <- poverty_scatter_traindata_rf %>%
  count(Classification) %>%
  mutate(Percentage = n / sum(n) * 100)

cat("\nPoverty Classification Summary:\n")
print(classification_summary_traindata_rf)

ggsave(paste0(plotsdir, "/", "randomforest", "/", "Train_classification_prediction_vs_actual_poverty_rf.png"), 
       poverty_scatter_traindata_plot_rf, width = 12, height = 10, dpi = 300)
cat("Poverty classification scatterplot saved\n")



# 5b. Scatterplot with Poverty Classification - Test Dataset
cat("\nCreating scatterplot with Test poverty classification...\n")

# Apply the same poverty-classification procedure to the independent test dataset.
poverty_scatter_test_data <- test_resid_data_rf %>%
  mutate(
    Actual_Poverty = ifelse(mpi >= 0.3333, "Poor", "Non_Poor"),
    Predicted_Poverty = ifelse(.fitted >= 0.3333, "Poor", "Non_Poor"),
    Classification = case_when(
      Actual_Poverty == "Poor" & Predicted_Poverty == "Poor" ~ "True Poor",
      Actual_Poverty == "Non_Poor" & Predicted_Poverty == "Non_Poor" ~ "True Non_Poor",
      Actual_Poverty == "Poor" & Predicted_Poverty == "Non_Poor" ~ "False Non_Poor",
      Actual_Poverty == "Non_Poor" & Predicted_Poverty == "Poor" ~ "False Poor"
    )
  )

# Plot the test predictions by poverty-classification outcome using the same MPI
# threshold applied to the training dataset.
poverty_scatter_testdata_plot_rf <- ggplot(poverty_scatter_test_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "solid", linewidth = 0.5) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non_Poor" = "darkgreen", 
    "False Poor" = "orange",
    "False Non_Poor" = "purple"
  )) +
  coord_equal() +
  labs(
    title = "Prediction vs Actual (Test)",
    subtitle = "Poverty classification at MPI = 0.3333",
    x = "Actual MPI",
    y = "Predicted MPI",
    color = "Classification"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 12),
    legend.position = "top"
  ) +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

# Calculate classification accuracy
# Summarise the number and percentage of test observations within each
# classification outcome.
classification_summary_test <- poverty_scatter_test_data %>%
  count(Classification) %>%
  mutate(Percentage = n / sum(n) * 100)

cat("\nPoverty Classification Summary:\n")
print(classification_summary_test)

ggsave(paste0(plotsdir, "/", "randomforest", "/", "Test_classification_prediction_vs_actual_poverty.png"), 
       poverty_scatter_testdata_plot_rf, width = 12, height = 10, dpi = 300)
cat("Poverty classification scatterplot for Test Dataset saved\n")


# ----------------------------
# 6) Multi-panel plot (RF)
# ----------------------------

#has_test <- exists("preds.test") && all(c("mpi", ".pred") %in% names(preds.test))



#title = "Comprehensive Random Forest Performance Analysis"
#if (has_test) {
# Combine the predicted-versus-actual and poverty-classification plots for the
# training and test datasets into a four-panel Random Forest performance summary.
multi_class_panel_plot_rf <- (train_scatter_plot_rf + theme(axis.text = element_text(size = 8)) +
                                test_scatter_plot_rf + theme(axis.text = element_text(size = 8))) /
  (poverty_scatter_traindata_plot_rf + theme(axis.text = element_text(size = 8)) +
     poverty_scatter_testdata_plot_rf + theme(axis.text = element_text(size = 8))) +
  plot_annotation(
    subtitle = rf_subtitle,
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )

ggsave(
  file.path(plotsdir, "randomforest", "comprehensive_prediction_analysis_rf.png"),
  multi_class_panel_plot_rf, width = 16, height = 12, dpi = 300
)

cat(" Comprehensive RF multi-panel plot saved:\n")
cat(file.path(plotsdir, "randomforest", "comprehensive_prediction_analysis_rf.png"), "\n\n")

# ----------------------------
# 7) Print summary
# ----------------------------
# Print the main training and test regression metrics used in the plotted
# Random Forest performance assessment.
cat("\n=== PREDICTION VS ACTUAL SUMMARY (RF) ===\n")
cat("Training Set Performance:\n")
cat(sprintf("  RMSE: %.4f\n", train_rmse))
cat(sprintf("  MAE:  %.4f\n", train_mae))
cat(sprintf("  R²:   %.4f\n", train_rsq))
cat(sprintf("  Corr: %.4f\n", train_cor))

#if (has_test) {
cat("Test Set Performance:\n")
cat(sprintf("  RMSE: %.4f\n", test_rmse))
cat(sprintf("  MAE:  %.4f\n", test_mae))
cat(sprintf("  R²:   %.4f\n", test_rsq))
cat(sprintf("  Corr: %.4f\n", test_cor))
#}

cat("\nCreating comprehensive multi-panel plot...\n")




######################## XGBOOST ######################

#######################18th Feb########################


# ===============================================================
# XGBOOST: SPACE-FILLING + REFINED SEARCH
# - Complete reproducibility with deterministic seeding
# - Test predictions and metrics for BOTH stages
# - All models saved as bundled RDS files
# - Ultimate best model identification across both stages
# ===============================================================

# ========================================================================
# REPRODUCIBILITY DOCUMENTATION (APPLIES TO ENTIRE XGBOOST SECTION)
# ========================================================================
# This script ensures full reproducibility through deterministic seeding:
#
# 1. Global RNG: Set to L'Ecuyer-CMRG with base seed 1238 (ONCE at script top)
#    - Do NOT call RNGkind() or set.seed(1238) here; it's already set globally
#
# 2. Space-filling stage:
#    - Grid generation: set.seed(seed_for_recipe(recipe)) before grid_space_filling()
#    - CV tuning: set.seed(seed_for_recipe(recipe) + 1L) before tune_grid()
#    - Final fit: set.seed(seed_for_sf_final(recipe)) before xgb.train()
#
# 3. Refined stage:
#    - Fold fitting: set.seed(fold_seed) IMMEDIATELY before xgb.train() in each fold
#    - Final fit: set.seed(xgb_seed_final) IMMEDIATELY before xgb.train()
#
# 4. XGBoost engine: seed parameter set per-model (fold_seed or xgb_seed_final)
#    - nthread = 1 for full reproducibility
#
# 5. All seeds are saved in output files for audit trail
# ========================================================================

# Confirm that all datasets, metric functions and model objects required for
# the XGBoost analysis are available before the search begins.
stopifnot(exists("model_dir"))
stopifnot(exists("ntl_train"))
stopifnot(exists("ntl_test"))
stopifnot(exists("ntl_vfolds_cv"))
stopifnot(exists("combined_metrics_num"))
stopifnot(exists("calc_test_metrics"))  # your helper for test metrics

# recipes must exist
stopifnot(exists("all_pred_notfm.recipe"))
stopifnot(exists("all_pred_tfmwithlogcubenormalize.recipe"))
stopifnot(exists("all_pred_tfmwithlogcuberange.recipe"))

# ---------------------------------------------------------------
# OUTPUT PATHS
# ---------------------------------------------------------------
# Create a unique identifier for the current run and define the folders and
# files used to retain cross-validation, prediction and model outputs.
run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")

out_dir <- file.path(model_dir, "xgboost", paste0("run_", run_id))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

images_dir <- file.path(out_dir, "images")
dir.create(images_dir, recursive = TRUE, showWarnings = FALSE)

# CV outputs (space-filling)
metrics_file          <- file.path(out_dir, paste0("xgb_cv_metrics_", run_id, ".csv"))
preds_file            <- file.path(out_dir, paste0("xgb_cv_predictions_", run_id, ".csv"))
timing_file           <- file.path(out_dir, paste0("xgb_cv_timing_", run_id, ".csv"))
bestcvmetric_file     <- file.path(out_dir, paste0("xgb_best_cv_metrics_", run_id, ".csv"))
singlebestmetric_file <- file.path(out_dir, paste0("xgb_single_best_metrics_", run_id, ".csv"))

# Final test outputs (space-filling stage)
final_sf_test_metrics_file <- file.path(out_dir, paste0("xgb_sf_FINAL_test_metrics_", run_id, ".csv"))
final_sf_test_preds_file   <- file.path(out_dir, paste0("xgb_sf_FINAL_test_predictions_", run_id, ".csv"))
final_sf_train_preds_file  <- file.path(out_dir, paste0("xgb_sf_FINAL_train_predictions_", run_id, ".csv"))

# Residual objects (space-filling)
resid_rds_file <- file.path(out_dir, paste0("xgb_residual_objects_", run_id, ".rds"))

# clean start
for (f in c(metrics_file, preds_file, timing_file,
            final_sf_test_metrics_file, final_sf_test_preds_file, final_sf_train_preds_file,
            bestcvmetric_file, singlebestmetric_file, resid_rds_file)) {
  if (file.exists(f)) file.remove(f)
}

# ---------------------------------------------------------------
# REPRODUCIBILITY NOTE:
# RNGkind("L'Ecuyer-CMRG") and set.seed(1238) should be set ONCE
# at the top of your entire script (before Beta, RF, XGB, NN sections).
# Do NOT repeat them here.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# PREPROCESS LIST
# ---------------------------------------------------------------
# Store the three preprocessing recipes evaluated during the XGBoost analysis.
preproc.xgb <- list(
  all_pred_notfm                    = all_pred_notfm.recipe,
  all_pred_tfmwithlogcubenormalize  = all_pred_tfmwithlogcubenormalize.recipe,
  all_pred_tfmwithlogcuberange      = all_pred_tfmwithlogcuberange.recipe
)

# ---------------------------------------------------------------
# XGBOOST SPEC (stable)
# ---------------------------------------------------------------
# Define the XGBoost regression model. The main tree and learning parameters
# are left for tuning, while loss reduction is fixed at zero.
xgboost_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  mtry           = tune(),
  min_n          = tune(),
  sample_size    = tune(),
  loss_reduction = 0
) %>%
  set_engine("xgboost", seed = 124, nthread = 1, verbosity = 0) %>%  # seed is placeholder; overridden per-fit
  set_mode("regression")

# ---------------------------------------------------------------
# PARAMETER RANGES (REQUIRED)
# mtry is placeholder; updated per recipe after bake()
# learn_rate is log10 scale because dials::learn_rate uses log10
# ---------------------------------------------------------------
# Define the initial search ranges for the XGBoost hyperparameters.
# The mtry range is later restricted to the number of predictors available
# after each preprocessing recipe has been applied.
xgboost_custom_param <- parameters(
  list(
    trees          = trees(range = c(250L, 2000L)),
    mtry           = mtry(range = c(1L, 200L)),          # placeholder; updated per recipe
    min_n          = min_n(range = c(5L, 20L)),
    sample_size    = sample_prop(range = c(0.5, 1)),
    tree_depth     = tree_depth(range = c(2L, 50L)),
    learn_rate     = learn_rate(range = c(-3, -0.5))    # log10 scale
  )
)

# ---------------------------------------------------------------
# CONTROL + PARALLEL
# ---------------------------------------------------------------
# Configure the tuning process to retain predictions and workflows, and use
# a small parallel backend during the space-filling search.
ctrl <- control_grid(
  save_pred     = TRUE,
  save_workflow = TRUE,
  parallel_over = "resamples",
  allow_par     = TRUE,
  verbose       = TRUE
)

cores <- max(1, min(2, parallel::detectCores() - 1))
doParallel::registerDoParallel(cores = cores)

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------
# Generate deterministic seeds from the recipe names so each main fitting
# stage can be repeated as closely as possible.
seed_for_recipe <- function(recipe_name, base = 124L) {
  as.integer((base + sum(utf8ToInt(paste0("XGB_", recipe_name)))) %% .Machine$integer.max)
}

seed_for_sf_final <- function(recipe_name, base = 124L) {
  as.integer((base + sum(utf8ToInt(paste0("SF_FINAL_XGB_", recipe_name)))) %% .Machine$integer.max)
}

add_grid_id <- function(grid_tbl) {
  grid_tbl %>% mutate(grid_id = row_number()) %>% relocate(grid_id)
}

log_time <- function(level, recipe, grid_id, t0, t1) {
  tibble(
    run_id = run_id,
    level = level,           # "recipe" or "grid_est"
    recipe = recipe,
    grid_id = grid_id,
    start_time = as.character(t0),
    end_time   = as.character(t1),
    duration_secs = as.numeric(difftime(t1, t0, units = "secs")),
    duration_mins = as.numeric(difftime(t1, t0, units = "mins"))
  )
}

append_grid_time_estimates <- function(recipe, recipe_start, recipe_end, n_grid) {
  total_secs <- as.numeric(difftime(recipe_end, recipe_start, units = "secs"))
  per_grid   <- total_secs / max(1, n_grid)
  tibble(
    run_id = run_id,
    level = "grid_est",
    recipe = recipe,
    grid_id = seq_len(n_grid),
    start_time = as.character(recipe_start),
    end_time   = as.character(recipe_end),
    duration_secs = per_grid,
    duration_mins = per_grid / 60
  )
}

timing_log <- tibble()

# ===============================================================
# SPACE-FILLING STAGE: CV TUNING LOOP
# ===============================================================
cat("\n=============================\n")
cat("STAGE 1: SPACE-FILLING SEARCH\n")
cat("=============================\n")

# Evaluate each preprocessing recipe separately during the initial
# space-filling hyperparameter search.
for (r in names(preproc.xgb)) {
  
  recipe_start <- Sys.time()
  cat("\n=============================\n")
  cat("RECIPE:", r, "\n")
  cat("=============================\n")
  
  rec <- preproc.xgb[[r]]
  
  # compute numeric predictor count AFTER recipe preprocessing (for valid mtry)
  
  
  # ── Load the recipe that was used during training ─────────────
  # You need to save all_pred_notfm.recipe after prep() from your training script.
  # In your training script, add this once:
  #   prep_rec <- prep(all_pred_notfm.recipe, training = ntl_train)
  #   saveRDS(prep_rec, file.path(model_dir, "xgboost_production_recipe.rds"))
  # Prepare the current recipe using the complete training dataset, save the
  # fitted recipe and determine the number of usable numeric predictors.
  prep_rec_tmp <- prep(rec, training = ntl_train)
  saveRDS(prep_rec_tmp, file.path(images_dir, paste0("xgboost_", r, "_recipe.rds")))
  
  
  train_baked  <- bake(prep_rec_tmp, new_data = ntl_train)
  
  outcome <- "mpi"
  pnames  <- setdiff(names(train_baked), outcome)
  pnames  <- pnames[sapply(train_baked[, pnames, drop = FALSE], is.numeric)]
  n_pred  <- length(pnames)
  
  if (n_pred < 2) stop("Not enough numeric predictors after baking for recipe: ", r)
  cat(">>> predictors after recipe:", n_pred, "\n")
  
  # recipe-specific param ranges (valid mtry range)
  # Update the mtry range for the current recipe and generate 95 space-filling
  # hyperparameter combinations within the permitted parameter ranges.
  xgb_param_r <- xgboost_custom_param %>%
    update(mtry = mtry(range = c(floor(sqrt(n_pred)), n_pred)))
  
  # FIX: Seed IMMEDIATELY before grid generation
  set.seed(seed_for_recipe(r))
  xgb_grid_r <- grid_space_filling(xgb_param_r, size = 95, original = TRUE) %>%
    mutate(mtry = pmin(mtry, n_pred)) %>%
    distinct() %>%
    arrange(mtry, trees, min_n, tree_depth, learn_rate, sample_size) %>%
    add_grid_id()
  
  cat(">>> grid size:", nrow(xgb_grid_r), "\n")
  
  # workflow
  # Combine the current preprocessing recipe with the tunable XGBoost model.
  wf_base <- workflow() %>%
    add_recipe(rec) %>%
    add_model(xgboost_spec)
  
  # FIX: Seed IMMEDIATELY before tune_grid
  set.seed(seed_for_recipe(r) + 1L)
  # Evaluate the complete space-filling grid using the predefined repeated
  # cross-validation folds and the common regression and classification metrics.
  res <- tryCatch(
    tune_grid(
      object    = wf_base,
      resamples = ntl_vfolds_cv,
      grid      = xgb_grid_r %>% select(-grid_id),
      metrics   = combined_metrics_num,
      control   = ctrl
    ),
    error = function(e) e
  )
  
  recipe_end <- Sys.time()
  
  # timing: recipe + per-grid estimates
  timing_log <- bind_rows(timing_log, log_time("recipe", r, NA_integer_, recipe_start, recipe_end))
  timing_log <- bind_rows(timing_log, append_grid_time_estimates(r, recipe_start, recipe_end, nrow(xgb_grid_r)))
  
  # total recipe failure -> write NA rows
  if (inherits(res, "error")) {
    warning("Recipe failed completely: recipe=", r, " | ", conditionMessage(res))
    
    fail_block <- xgb_grid_r %>%
      transmute(
        run_id = run_id,
        recipe = r, grid_id = grid_id,
        mtry, trees, min_n, tree_depth, learn_rate, sample_size, 
        rmse = NA_real_, rmse_se = NA_real_,
        mae  = NA_real_, mae_se  = NA_real_,
        rsq  = NA_real_, rsq_se  = NA_real_,
        accuracy_033 = NA_real_, accuracy_033_se = NA_real_,
        sensitivity_033 = NA_real_, sensitivity_033_se = NA_real_,
        specificity_033 = NA_real_, specificity_033_se = NA_real_,
        f1_033 = NA_real_, f1_033_se = NA_real_,
        n_resamples = NA_integer_
      )
    
    write_csv(fail_block, metrics_file, append = file.exists(metrics_file))
    next
  }
  
  # Collect the cross-validation metrics for all successful configurations
  # and reshape them into one row per hyperparameter combination.
  # Metrics: mean + SE per grid row
  m_long <- collect_metrics(res) %>%
    filter(.metric %in% c("rmse","mae","rsq","accuracy_033","sensitivity_033","specificity_033","f1_033"))
  
  join_cols <- intersect(names(xgb_grid_r), names(m_long))
  join_cols <- setdiff(join_cols, "grid_id")
  
  if (length(join_cols) == 0) {
    stop("Cannot join metrics back to grid for recipe: ", r,
         ". No shared parameter columns between grid and collect_metrics().")
  }
  
  m_joined <- m_long %>%
    left_join(xgb_grid_r, by = join_cols) %>%
    mutate(run_id = run_id, recipe = r)
  
  m_wide <- m_joined %>%
    select(run_id, recipe, grid_id,
           mtry, trees, min_n, tree_depth, learn_rate, sample_size,
           .metric, mean, std_err, n) %>%
    pivot_wider(
      names_from  = .metric,
      values_from = c(mean, std_err),
      names_glue  = "{.metric}_{.value}"
    ) %>%
    rename_with(~ str_replace(.x, "_mean$", ""), ends_with("_mean")) %>%
    rename_with(~ str_replace(.x, "_std_err$", "_se"), ends_with("_std_err")) %>%
    mutate(n_resamples = n) %>%
    select(-n) %>%
    arrange(grid_id)
  
  # Retain the out-of-fold predictions and attach the corresponding recipe
  # and hyperparameter information for later analysis.
  # Predictions: keep + attach grid_id when possible
  p <- collect_predictions(res) %>%
    arrange(id, .row) %>%
    mutate(run_id = run_id, recipe = r)
  
  join_cols_p <- intersect(names(xgb_grid_r), names(p))
  join_cols_p <- setdiff(join_cols_p, "grid_id")
  
  if (length(join_cols_p) > 0) {
    p <- p %>% left_join(xgb_grid_r, by = join_cols_p)
  } else {
    p <- p %>% mutate(grid_id = NA_integer_)
  }
  
  p <- p %>% relocate(run_id, recipe, grid_id)
  
  # write outputs
  write_csv(m_wide, metrics_file, append = file.exists(metrics_file))
  write_csv(p,      preds_file,   append = file.exists(preds_file))
  
  cat("wrote metrics + preds for recipe:", r, "\n")
}

# save timing --> start here
write_csv(timing_log, timing_file)
doParallel::stopImplicitCluster()

cat("\n SPACE-FILLING CV DONE.\n")
cat("Metrics:     ", metrics_file, "\n")
cat("Predictions: ", preds_file, "\n")
cat("Timing:      ", timing_file, "\n")

# ===============================================================
# ANALYZE RESULTS: best per recipe + single best
# ===============================================================
# Read the completed space-filling results and identify the best configuration
# within each preprocessing recipe.
xgb_metrics <- readr::read_csv(metrics_file, show_col_types = FALSE)

best_per_recipe <- xgb_metrics %>%
  filter(!is.na(rmse), n_resamples >= 1) %>%
  group_by(recipe) %>%
  arrange(rmse, mae, desc(rsq)) %>%
  slice(1) %>%
  ungroup()

write_csv(best_per_recipe, bestcvmetric_file)
cat("\n=== BEST PER RECIPE (SPACE-FILLING) ===\n")
print(best_per_recipe)

# Select the single best initial-stage XGBoost configuration across all
# preprocessing recipes using RMSE first, followed by MAE and R-squared.
xgb_singlebest <- xgb_metrics %>%
  filter(!is.na(rmse), n_resamples >= 1) %>%
  arrange(rmse, mae, desc(rsq)) %>%
  slice(1) %>%
  ungroup()

write_csv(xgb_singlebest, singlebestmetric_file)
cat("\n=== SINGLE BEST (SPACE-FILLING) ===\n")
print(xgb_singlebest)
View(xgb_singlebest)

# ===============================================================
# SPACE-FILLING STAGE: FINAL FIT + TEST PREDICTIONS
# Fits best model per recipe on full train, predicts train + test
# ===============================================================
cat("\n=============================\n")
cat("STAGE 1: FINAL FITS (SPACE-FILLING BEST)\n")
cat("=============================\n")

stopifnot(all(c("recipe", "mtry", "trees", "min_n", "tree_depth", "learn_rate", "sample_size") %in% names(best_per_recipe)))

# Refit the best initial-stage configuration from each preprocessing recipe
# using the complete training dataset and evaluate it on the independent test data.
for (k in seq_len(nrow(best_per_recipe))) {
  
  rec_name <- as.character(best_per_recipe$recipe[[k]])
  cfg_best <- best_per_recipe[k, , drop = FALSE]
  
  cat("\n>> Training final model for:", rec_name, "\n")
  
  rec <- preproc.xgb[[rec_name]]
  if (is.null(rec)) stop("Recipe not found in preproc.xgb: ", rec_name)
  
  # Extract hyperparameters
  mtry_k        <- as.integer(cfg_best$mtry)
  trees_k       <- as.integer(cfg_best$trees)
  min_n_k       <- as.integer(cfg_best$min_n)
  tree_depth_k  <- as.integer(cfg_best$tree_depth)
  learn_rate_k  <- as.numeric(cfg_best$learn_rate)
  sample_size_k <- as.numeric(cfg_best$sample_size)
  
  cat("   Config: mtry=", mtry_k, ", trees=", trees_k, ", min_n=", min_n_k,
      ", depth=", tree_depth_k, ", lr=", learn_rate_k, ", sample=", sample_size_k, "\n", sep = "")
  
  # Prepare the selected recipe using the complete training dataset and apply
  # the same fitted preprocessing steps to the independent test dataset.
  # Prep and bake FIRST
  rec_prep <- prep(rec, training = ntl_train, retain = TRUE)
  train_baked <- bake(rec_prep, new_data = ntl_train)
  test_baked  <- bake(rec_prep, new_data = ntl_test)
  
  # Count predictors
  pnames <- setdiff(names(train_baked), "mpi")
  pnames <- pnames[sapply(train_baked[, pnames, drop = FALSE], is.numeric)]
  n_pred <- length(pnames)
  
  # Separate the outcome from the predictor variables and convert the predictor
  # data into the numeric matrix format required by XGBoost.
  # Prepare data
  y_train <- train_baked$mpi
  y_test  <- test_baked$mpi
  
  X_train <- train_baked %>% select(-mpi) %>% as.data.frame()
  X_test  <- test_baked  %>% select(-mpi) %>% as.data.frame()
  
  X_train[] <- lapply(X_train, as.numeric)
  X_test[]  <- lapply(X_test,  as.numeric)
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test)
  
  # Recipe-specific seed
  xgb_seed <- seed_for_sf_final(rec_name)
  
  # Define the final XGBoost parameters using the selected initial-stage
  # hyperparameters and a recipe-specific fitting seed.
  params <- list(
    objective = "reg:squarederror",
    eta = learn_rate_k,
    max_depth = tree_depth_k,
    min_child_weight = min_n_k,
    subsample = sample_size_k,
    colsample_bytree = mtry_k / max(1, n_pred),
    gamma = 0,
    nthread = 1,
    verbosity = 0,
    seed = xgb_seed
  )
  
  # FIX: set.seed() IMMEDIATELY before xgb.train()
  cat("   Training (seed=", xgb_seed, ")...\n", sep = "")
  set.seed(xgb_seed)
  # Fit the selected XGBoost model on the complete training dataset.
  booster <- xgboost::xgb.train(
    params = params,
    data   = dtrain,
    nrounds = trees_k,
    verbose = 0
  )
  
  # Save bundled model
  model_file <- file.path(images_dir, paste0("xgb_sf_final_", rec_name, "_", run_id, ".rds"))
  
  if (!file.exists(model_file)) {
    saveRDS(bundle(booster), model_file)
    message("Model saved: ", model_file)
  } else {
    message("Model already exists, skipping: ", model_file)
  }
  
  # Generate predictions for the complete training dataset and the independent
  # test dataset using the fitted XGBoost booster.
  # Predict train and test
  pred_train <- predict(booster, dtrain)
  pred_test  <- predict(booster, dtest)
  
  # Calculate the main regression measures for the training and independent
  # test predictions.
  # Metrics
  rmse_train <- yardstick::rmse_vec(y_train, pred_train)
  mae_train  <- yardstick::mae_vec(y_train,  pred_train)
  rsq_train  <- suppressWarnings(yardstick::rsq_vec(y_train, pred_train))
  
  rmse_test  <- yardstick::rmse_vec(y_test, pred_test)
  mae_test   <- yardstick::mae_vec(y_test,  pred_test)
  rsq_test   <- suppressWarnings(yardstick::rsq_vec(y_test, pred_test))
  
  cat("   Train RMSE:", round(rmse_train, 4), " | Test RMSE:", round(rmse_test, 4), "\n")
  cat("   Train R²:",   round(rsq_train,  4), " | Test R²:",  round(rsq_test,  4), "\n")
  
  # Save train predictions
  write_csv(
    tibble(
      run_id = run_id, recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k,
      tree_depth = tree_depth_k, learn_rate = learn_rate_k, sample_size = sample_size_k,
      xgb_seed = xgb_seed,
      row_id = seq_along(y_train),
      mpi_true = y_train, mpi_pred = pred_train
    ),
    final_sf_train_preds_file,
    append = file.exists(final_sf_train_preds_file)
  )
  
  # Save test predictions
  write_csv(
    tibble(
      run_id = run_id, recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k,
      tree_depth = tree_depth_k, learn_rate = learn_rate_k, sample_size = sample_size_k,
      xgb_seed = xgb_seed,
      row_id = seq_along(y_test),
      mpi_true = y_test, mpi_pred = pred_test
    ),
    final_sf_test_preds_file,
    append = file.exists(final_sf_test_preds_file)
  )
  
  # Calculate the complete independent-test performance measures, including
  # the poverty-classification metrics based on the MPI threshold.
  # Save test metrics (using calc_test_metrics for consistency)
  met <- calc_test_metrics(truth = y_test, estimate = pred_test, threshold = 0.3333) %>%
    mutate(
      run_id = run_id, recipe = rec_name,
      mtry = mtry_k, trees = trees_k, min_n = min_n_k,
      tree_depth = tree_depth_k, learn_rate = learn_rate_k, sample_size = sample_size_k,
      xgb_seed = xgb_seed
    )
  
  write_csv(met, final_sf_test_metrics_file, append = file.exists(final_sf_test_metrics_file))
}

cat("\nSPACE-FILLING FINAL FITS COMPLETE\n")
cat("Train predictions: ", final_sf_train_preds_file, "\n")
cat("Test predictions:  ", final_sf_test_preds_file,  "\n")
cat("Test metrics:      ", final_sf_test_metrics_file, "\n")

# ===============================================================
# RESIDUAL OBJECTS FOR PLOTS (single best from space-filling only)
# ===============================================================
# Refit the single best space-filling model so the corresponding training
# and test residual objects can be created for later diagnostic plots.
rec_name_sb <- xgb_singlebest$recipe[[1]]
rec_sb <- preproc.xgb[[rec_name_sb]]

mtry_sb        <- as.integer(xgb_singlebest$mtry)
trees_sb       <- as.integer(xgb_singlebest$trees)
min_n_sb       <- as.integer(xgb_singlebest$min_n)
tree_depth_sb  <- as.integer(xgb_singlebest$tree_depth)
learn_rate_sb  <- as.numeric(xgb_singlebest$learn_rate)
sample_size_sb <- as.numeric(xgb_singlebest$sample_size)

rec_prep_sb <- prep(rec_sb, training = ntl_train, retain = TRUE)
train_baked_sb <- bake(rec_prep_sb, new_data = ntl_train)
test_baked_sb  <- bake(rec_prep_sb, new_data = ntl_test)

pnames_sb <- setdiff(names(train_baked_sb), "mpi")
pnames_sb <- pnames_sb[sapply(train_baked_sb[, pnames_sb, drop = FALSE], is.numeric)]
n_pred_sb <- length(pnames_sb)

y_train_sb <- train_baked_sb$mpi
y_test_sb  <- test_baked_sb$mpi

X_train_sb <- train_baked_sb %>% select(-mpi) %>% as.data.frame()
X_test_sb  <- test_baked_sb  %>% select(-mpi) %>% as.data.frame()

X_train_sb[] <- lapply(X_train_sb, as.numeric)
X_test_sb[]  <- lapply(X_test_sb,  as.numeric)

dtrain_sb <- xgb.DMatrix(data = as.matrix(X_train_sb), label = y_train_sb)
dtest_sb  <- xgb.DMatrix(data = as.matrix(X_test_sb),  label = y_test_sb)

xgb_seed_sb <- seed_for_sf_final(rec_name_sb)

params_sb <- list(
  objective = "reg:squarederror",
  eta = learn_rate_sb,
  max_depth = tree_depth_sb,
  min_child_weight = min_n_sb,
  subsample = sample_size_sb,
  colsample_bytree = mtry_sb / max(1, n_pred_sb),
  gamma = 0,
  nthread = 1,
  verbosity = 0,
  seed = xgb_seed_sb
)

set.seed(xgb_seed_sb)
fit_sb <- xgboost::xgb.train(
  params = params_sb,
  data   = dtrain_sb,
  nrounds = trees_sb,
  verbose = 0
)

# Save single best model
model_file_sb <- file.path(images_dir, paste0("final_cv_xgb_", rec_name_sb, "_", run_id, ".rds"))

if (!file.exists(model_file_sb)) {
  saveRDS(bundle(fit_sb), model_file_sb)
  message("Single best model saved: ", model_file_sb)
} else {
  message("Single best model already exists: ", model_file_sb)
}

preds_train_sb <- predict(fit_sb, dtrain_sb)
preds_test_sb  <- predict(fit_sb, dtest_sb)

# Calculate residuals as observed MPI minus predicted MPI for both datasets.
resid_train <- y_train_sb - preds_train_sb
resid_test  <- y_test_sb  - preds_test_sb

train_resid_data_xgb_sf <- tibble(
  model = "XGBoost",
  .fitted = as.numeric(preds_train_sb),
  .resid  = as.numeric(resid_train),
  mpi     = as.numeric(y_train_sb),
  .std_resid = as.numeric(scale(resid_train)),
  Set = "Train"
)

test_resid_data_xgb_sf <- tibble(
  model = "XGBoost",
  .fitted = as.numeric(preds_test_sb),
  .resid  = as.numeric(resid_test),
  mpi     = as.numeric(y_test_sb),
  .std_resid = as.numeric(scale(resid_test)),
  Set = "Test"
)


saveRDS(
  list(
    train_resid_data_xgb_sf = train_resid_data_xgb_sf,
    test_resid_data_xgb_sf  = test_resid_data_xgb_sf,
    singlebest = xgb_singlebest,
    best_per_recipe = best_per_recipe
  ),
  resid_rds_file
)

cat("\nResidual objects saved to:\n", resid_rds_file, "\n")


####################################################
############ REFINED XGBoost (DROP-IN) ##############
####################################################
# Purpose:
# 1) Take xgb_singlebest from space-filling run (best row(s))
# 2) Build a local grid around each best row (per recipe)
# 3) Manual fold loop with xgb.train() (deterministic seeding)
# 4) Save CV metrics (means + SE) + predictions
# 5) Identify SINGLE BEST refined model (by RMSE, then RMSE_SE, then RSQ)
# 6) Fit on FULL train
# 7) Predict TRAIN + TEST, compute test metrics, save outputs
# 8) Save ALL models as bundled RDS files
# 9) Identify ULTIMATE BEST across both stages

cat("\n=============================\n")
cat("STAGE 2: REFINED SEARCH\n")
cat("=============================\n")

# -----------------------------
# Preconditions (fail fast)
# -----------------------------
# Confirm that the initial-stage results and all required datasets are
# available before starting the refined XGBoost search.
stopifnot(exists("xgb_singlebest"))
stopifnot(exists("preproc.xgb"))
stopifnot(exists("ntl_train"))
stopifnot(exists("ntl_test"))
stopifnot(exists("ntl_vfolds_cv"))
stopifnot(exists("out_dir"))
stopifnot(exists("run_id"))

req_cols <- c("recipe","mtry","trees","min_n","tree_depth","learn_rate","sample_size")
miss <- setdiff(req_cols, names(xgb_singlebest))
if (length(miss) > 0) stop("xgb_singlebest missing: ", paste(miss, collapse = ", "))

needed_metrics <- c("rmse","mae","rsq","accuracy_033","sensitivity_033","specificity_033","f1_033")

# FIX: Do NOT call set.seed(124) here — it would be consumed by prep/bake/grid helpers
# before reaching any model fitting. Per-fold seeds handle this correctly.

# -----------------------------
# Seed function
# -----------------------------
# Generate a deterministic seed for the refined XGBoost analysis.
seed_for_refine <- function(recipe_name, base = 124L) {
  as.integer((base + sum(utf8ToInt(paste0("REFINE_XGB_", recipe_name)))) %% .Machine$integer.max)
}

# -----------------------------
# Output files
# -----------------------------
refine_metrics_file <- file.path(out_dir, paste0("xgb_refine_cv_metrics_", run_id, ".csv"))
refine_preds_file   <- file.path(out_dir, paste0("xgb_refine_cv_predictions_", run_id, ".csv"))
refined_best_file   <- file.path(out_dir, paste0("xgb_refine_bestrows_", run_id, ".csv"))

final_train_preds_file <- file.path(out_dir, paste0("xgb_refine_FINAL_train_predictions_", run_id, ".csv"))
final_test_preds_file  <- file.path(out_dir, paste0("xgb_refine_FINAL_test_predictions_", run_id, ".csv"))
final_test_metrics_file <- file.path(out_dir, paste0("xgb_refine_FINAL_test_metrics_", run_id, ".csv"))

# Start fresh
for (f in c(refine_metrics_file, refine_preds_file, refined_best_file,
            final_train_preds_file, final_test_preds_file, final_test_metrics_file)) {
  if (file.exists(f)) file.remove(f)
}

# -----------------------------
# Helpers
# -----------------------------
clamp_int <- function(x, lo, hi) as.integer(pmax(lo, pmin(hi, as.integer(round(x)))))
clamp_num <- function(x, lo, hi) pmax(lo, pmin(hi, as.numeric(x)))

safe_sd <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) 1 else s
}

# Convert continuous MPI values into poor and non-poor classes and calculate
# the associated poverty-classification performance measures.
compute_class_metrics <- function(truth, estimate, threshold = 0.3333) {
  actual <- factor(ifelse(truth >= threshold, "poor", "non_poor"), levels = c("poor","non_poor"))
  pred   <- factor(ifelse(estimate >= threshold, "poor", "non_poor"), levels = c("poor","non_poor"))
  cm <- table(Actual = actual, Predicted = pred)
  
  tp <- if ("poor"     %in% rownames(cm) && "poor"     %in% colnames(cm)) cm["poor",    "poor"]     else 0
  tn <- if ("non_poor" %in% rownames(cm) && "non_poor" %in% colnames(cm)) cm["non_poor","non_poor"] else 0
  fp <- if ("non_poor" %in% rownames(cm) && "poor"     %in% colnames(cm)) cm["non_poor","poor"]     else 0
  fn <- if ("poor"     %in% rownames(cm) && "non_poor" %in% colnames(cm)) cm["poor",    "non_poor"] else 0
  
  acc  <- (tp + tn) / max(1, sum(cm))
  sen  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  spe  <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  f1   <- if (is.finite(prec) && is.finite(sen) && (prec + sen) > 0) 2 * (prec * sen) / (prec + sen) else NA_real_
  
  list(cm = cm, accuracy_033 = acc, sensitivity_033 = sen, specificity_033 = spe, f1_033 = f1)
}

#trees_vals <- clamp_int(seq(trees0 - trees_span, trees0 + trees_span, by = 1), 1L, 1000000L)
# Construct a local hyperparameter grid around the best initial XGBoost
# configuration while keeping each parameter within its permitted range.
make_local_xgb_grid <- function(best_row, n_pred,
                                mtry_span = 2,
                                trees_span = 200,
                                min_n_span = 2,
                                depth_span = 3,
                                lr_span = c(-0.01, -0.005, 0, 0.005, 0.01),
                                sample_span = c(-0.1, -0.05, 0, 0.05, 0.1)) {
  
  mtry0   <- as.integer(round(best_row$mtry))
  trees0  <- as.integer(round(best_row$trees))
  min_n0  <- as.integer(round(best_row$min_n))
  depth0  <- as.integer(round(best_row$tree_depth))
  lr0     <- as.numeric(best_row$learn_rate)
  samp0   <- as.numeric(best_row$sample_size)
  
  mtry_vals  <- clamp_int(seq(mtry0 - mtry_span, mtry0 + mtry_span, by = 1), 1L, n_pred)
  trees_vals <- clamp_int(seq(trees0 - trees_span, trees0 + trees_span, by = 50), 25L, 2500L)
  min_n_vals <- clamp_int(seq(min_n0 - min_n_span, min_n0 + min_n_span, by = 1), 1L, 1000000L)
  depth_vals <- clamp_int(seq(depth0 - depth_span, depth0 + depth_span, by = 1), 1L, 1000L)
  
  
  # refine learning rate on log10 scale, then convert back to raw scale
  lr_log0 <- log10(lr0)
  lr_vals <- 10^(lr_log0 + lr_span) #lr_log_span
  lr_vals <- unique(clamp_num(lr_vals, 1e-4, 0.3))
  
  # lr_vals <- lr0 + lr_span
  samp_vals <- unique(clamp_num(samp0 + sample_span, 0.1, 1.0))
  
  tidyr::expand_grid(
    mtry        = mtry_vals,
    trees       = trees_vals,
    min_n       = min_n_vals,
    tree_depth  = depth_vals,
    learn_rate  = lr_vals,
    sample_size = samp_vals
  ) %>% dplyr::distinct()
}

# Restrict a large local grid to the configurations closest to the initial
# best model so the refined search remains computationally manageable.
restrict_local_grid <- function(local_grid, best_row, target_n = 60) {
  if (nrow(local_grid) <= target_n) return(local_grid)
  
  cols <- setdiff(intersect(names(local_grid), names(best_row)), "recipe")
  
  g <- local_grid
  for (cc in cols) {
    if (is.numeric(g[[cc]])) {
      g[[paste0(cc, "_z")]] <- (g[[cc]] - as.numeric(best_row[[cc]])) / safe_sd(g[[cc]])
    } else {
      g[[paste0(cc, "_z")]] <- 0
    }
  }
  
  zcols  <- grep("_z$", names(g), value = TRUE)
  g$dist <- sqrt(rowSums((as.matrix(g[, zcols, drop = FALSE]))^2))
  
  out2 <- g %>% arrange(dist) %>% slice_head(n = target_n) %>% select(names(local_grid))
  
  best_grid_cols <- intersect(names(local_grid), names(best_row))
  best_candidate <- local_grid
  for (cc in best_grid_cols) best_candidate[[cc]] <- best_row[[cc]]
  
  bind_rows(best_candidate[1, names(local_grid), drop = FALSE], out2) %>%
    distinct() %>%
    slice_head(n = target_n)
}

se_of <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

# -----------------------------
# MAIN: refined loop
# -----------------------------
all_best <- list()

# Refine the single best configuration carried forward from the initial
# space-filling search.
for (k in seq_len(nrow(xgb_singlebest))) {
  
  rec_name <- as.character(xgb_singlebest$recipe[[k]])
  best_row <- xgb_singlebest[k, , drop = FALSE]
  
  cat("\n=============================\n")
  cat("REFINING AROUND BEST:", rec_name, "\n")
  cat("=============================\n")
  
  rec <- preproc.xgb[[rec_name]]
  if (is.null(rec)) stop("Recipe not found: ", rec_name)
  
  # Prepare the selected recipe and determine the number of numeric predictors
  # remaining after preprocessing so mtry can be restricted correctly.
  # count numeric predictors post-bake (caps mtry)
  prep_rec_tmp <- prep(rec, training = ntl_train)
  train_baked  <- bake(prep_rec_tmp, new_data = ntl_train)
  pnames <- setdiff(names(train_baked), "mpi")
  pnames <- pnames[sapply(train_baked[, pnames, drop = FALSE], is.numeric)]
  n_pred <- length(pnames)
  if (n_pred < 1) stop("No numeric predictors after baking for recipe: ", rec_name)
  
  # Build the local refined grid around the selected initial configuration and
  # retain at most 70 nearby parameter combinations.
  # build local grid <= 60
  local_grid_full <- make_local_xgb_grid(best_row, n_pred = n_pred) %>%
    mutate(mtry = clamp_int(mtry, 1L, n_pred))
  
  local_grid <- restrict_local_grid(local_grid_full, best_row = best_row, target_n = 70)
  
  # deterministic ordering
  ord_cols <- intersect(c("mtry","trees","min_n","tree_depth","learn_rate","sample_size"), names(local_grid))
  local_grid <- local_grid %>% arrange(across(all_of(ord_cols)))
  
  cat("Local grid size:", nrow(local_grid),
      " (full was", nrow(local_grid_full), "; n_pred=", n_pred, ")\n")
  stopifnot(nrow(local_grid) <= 70)
  
  # Grid search with proper seeding
  cat("Starting grid search...\n")
  grid_counter <- 0
  total_grid <- nrow(local_grid)
  
  grid_results_list <- list()
  
  # Evaluate each refined hyperparameter combination across the predefined
  # repeated cross-validation folds.
  for (grid_idx in seq_len(nrow(local_grid))) {
    
    grid_counter <- grid_counter + 1
    
    # Extract current grid configuration
    mtry        <- as.integer(local_grid$mtry[grid_idx])
    trees       <- as.integer(local_grid$trees[grid_idx])
    min_n       <- as.integer(local_grid$min_n[grid_idx])
    tree_depth  <- as.integer(local_grid$tree_depth[grid_idx])
    learn_rate  <- as.numeric(local_grid$learn_rate[grid_idx])
    sample_size <- as.numeric(local_grid$sample_size[grid_idx])
    
    cat("\n[Config ", grid_counter, "/", total_grid, "] Testing: mtry=", mtry, 
        ", trees=", trees, ", min_n=", min_n, 
        ", depth=", tree_depth, ", lr=", sprintf("%.4f", learn_rate), 
        ", sample=", sprintf("%.3f", sample_size), "\n", sep="")
    
    fold_results_list <- list()
    
    for (i in seq_along(ntl_vfolds_cv$splits)) {
      
      cat("     Fold ", i, "/", length(ntl_vfolds_cv$splits), "... ", sep="")
      
      # Extract the analysis and assessment data for the current cross-validation fold.
      sp <- ntl_vfolds_cv$splits[[i]]
      tr <- rsample::analysis(sp)
      va <- rsample::assessment(sp)
      
      # Fit the preprocessing recipe within the current analysis fold and apply
      # the same fitted transformations to its corresponding assessment fold.
      # prep/bake FIRST — these consume RNG state
      rec_p <- prep(rec, training = tr, retain = TRUE)
      tr_b <- bake(rec_p, new_data = tr)
      va_b <- bake(rec_p, new_data = va)
      
      y_tr <- tr_b$mpi
      y_va <- va_b$mpi
      
      X_tr <- tr_b %>% select(-mpi) %>% as.data.frame()
      X_va <- va_b %>% select(-mpi) %>% as.data.frame()
      
      X_tr[] <- lapply(X_tr, as.numeric)
      X_va[] <- lapply(X_va, as.numeric)
      
      # Convert the fold predictor data into XGBoost DMatrix objects for model fitting
      # and validation prediction.
      dtr <- xgb.DMatrix(data = as.matrix(X_tr), label = y_tr)
      dva <- xgb.DMatrix(data = as.matrix(X_va), label = y_va)
      
      # FIX (CORE): Fold-specific seed
      # fold_seed <- seed_for_refine(rec_name) + i * 100L
      fold_seed <- 124L
      
      # Define the XGBoost parameters for the current refined configuration and fold.
      params <- list(
        objective = "reg:squarederror",
        eta = learn_rate,
        max_depth = tree_depth,
        min_child_weight = min_n,
        subsample = sample_size,
        colsample_bytree = mtry / max(1, n_pred),
        gamma = 0,
        nthread = 1,
        verbosity = 0,
        seed = fold_seed  # FIX: fold-specific seed, not static 124
      )
      
      # FIX (CORE): set.seed() IMMEDIATELY before xgb.train()
      set.seed(fold_seed)
      # Fit the current refined XGBoost model using only the analysis portion
      # of the cross-validation fold.
      booster <- xgboost::xgb.train(
        params = params,
        data   = dtr,
        nrounds = trees,
        verbose = 0
      )
      
      # Generate validation predictions and calculate regression and poverty-
      # classification performance for the current fold.
      pred <- predict(booster, dva)
      
      rmse <- yardstick::rmse_vec(y_va, pred)
      mae  <- yardstick::mae_vec(y_va, pred)
      rsq  <- suppressWarnings(yardstick::rsq_vec(y_va, pred))
      cls  <- compute_class_metrics(y_va, pred, threshold = 0.3333)
      
      cat("RMSE=", sprintf("%.4f", rmse), "\n", sep="")
      
      
      # stable original row id from ntl_train
      orig_row <- match(rownames(va), rownames(ntl_train))
      
      if (anyNA(orig_row)) {
        stop("Could not recover stable original row indices for refined-stage assessment data.")
      }
      
      resample_id <- if ("id" %in% names(ntl_vfolds_cv)) as.character(ntl_vfolds_cv$id[[i]]) else paste0("Fold", i)
      repeat_id   <- if ("id2" %in% names(ntl_vfolds_cv)) as.character(ntl_vfolds_cv$id2[[i]]) else NA_character_
      
      # Save the out-of-fold predictions together with the selected hyperparameters,
      # fold identifiers and observed MPI values.
      fold_pred_tbl <- tibble(
        run_id = run_id,
        recipe = rec_name,
        grid_mtry = mtry,
        grid_trees = trees,
        grid_min_n = min_n,
        grid_tree_depth = tree_depth,
        grid_learn_rate = learn_rate,
        grid_sample_size = sample_size,
        fold = i,
        id = resample_id,
        id2 = repeat_id,
        fold_seed = fold_seed,
        .row = orig_row,
        row_id = seq_along(y_va),   # keep your current feature
        mpi_true = y_va,
        mpi_pred = pred
      )
      
      # save fold preds
      #fold_pred_tbl <- tibble(
      #  run_id = run_id,
      #  recipe = rec_name,
      #  grid_mtry = mtry,
      #  grid_trees = trees,
      #  grid_min_n = min_n,
      #  grid_tree_depth = tree_depth,
      #  grid_learn_rate = learn_rate,
      #  grid_sample_size = sample_size,
      #  fold = i,
      #  fold_seed = fold_seed,
      #  row_id = seq_along(y_va),
      #  mpi_true = y_va,
      #  mpi_pred = pred
      #)
      write_csv(fold_pred_tbl, refine_preds_file, append = file.exists(refine_preds_file))
      
      # Store fold metrics
      fold_results_list[[length(fold_results_list) + 1]] <- tibble(
        fold = as.integer(i),
        rmse = as.numeric(rmse)[1],
        mae = as.numeric(mae)[1],
        rsq = as.numeric(rsq)[1],
        accuracy_033 = as.numeric(cls$accuracy_033)[1],
        sensitivity_033 = as.numeric(cls$sensitivity_033)[1],
        specificity_033 = as.numeric(cls$specificity_033)[1],
        f1_033 = as.numeric(cls$f1_033)[1]
      )
    }
    
    # Combine the fold-level results and summarise the mean performance and
    # standard error for the current refined hyperparameter configuration.
    fold_metrics <- dplyr::bind_rows(fold_results_list)
    
    mean_rmse <- mean(fold_metrics$rmse, na.rm = TRUE)
    mean_rsq  <- mean(fold_metrics$rsq,  na.rm = TRUE)
    
    cat("     >> CV Complete: Mean RMSE=", sprintf("%.4f", mean_rmse),
        ", Mean R²=", sprintf("%.4f", mean_rsq), "\n\n", sep="")
    
    # Store grid results
    grid_results_list[[grid_idx]] <- tibble(
      recipe = rec_name,
      mtry = mtry,
      trees = trees,
      min_n = min_n,
      tree_depth = tree_depth,
      learn_rate = learn_rate,
      sample_size = sample_size,
      
      rmse = mean(fold_metrics$rmse, na.rm = TRUE),
      rmse_se = se_of(fold_metrics$rmse),
      
      mae  = mean(fold_metrics$mae, na.rm = TRUE),
      mae_se = se_of(fold_metrics$mae),
      
      rsq  = mean(fold_metrics$rsq, na.rm = TRUE),
      rsq_se = se_of(fold_metrics$rsq),
      
      accuracy_033 = mean(fold_metrics$accuracy_033, na.rm = TRUE),
      accuracy_033_se = se_of(fold_metrics$accuracy_033),
      
      sensitivity_033 = mean(fold_metrics$sensitivity_033, na.rm = TRUE),
      sensitivity_033_se = se_of(fold_metrics$sensitivity_033),
      
      specificity_033 = mean(fold_metrics$specificity_033, na.rm = TRUE),
      specificity_033_se = se_of(fold_metrics$specificity_033),
      
      f1_033 = mean(fold_metrics$f1_033, na.rm = TRUE),
      f1_033_se = se_of(fold_metrics$f1_033)
    )
  }
  
  # Combine all grid results
  grid_results <- bind_rows(grid_results_list)
  
  # write all grid CV results
  write_csv(grid_results, refine_metrics_file, append = file.exists(refine_metrics_file))
  
  # pick best for this recipe
  # Select the best refined configuration using the recorded cross-validation
  # performance for the current preprocessing recipe.
  best_k <- grid_results %>%
    filter(is.finite(rmse)) %>%
    arrange(rmse, rmse_se, desc(rsq)) %>%
    slice(1)
  
  all_best[[rec_name]] <- best_k
  cat(">> Best refined for ", rec_name, ":\n", sep = "")
  print(best_k)
}

# -----------------------------
# Save final best rows
# -----------------------------
xgb_refinedsinglebestcv <- bind_rows(all_best)
write_csv(xgb_refinedsinglebestcv, refined_best_file)

cat("\n=============================\n")
cat("REFINEMENT COMPLETE\n")
cat("Best models saved to:", refined_best_file, "\n")
cat("=============================\n")

# =============================
# FINAL STAGE: Train on full training set and predict on test set
# =============================
cat("\n=============================\n")
cat("STAGE 2: FINAL FITS (REFINED BEST)\n")
cat("=============================\n")

final_test_results <- list()

# Refit the best refined XGBoost configuration using the complete training
# dataset and evaluate it on the independent test dataset.
for (k in seq_len(nrow(xgb_refinedsinglebestcv))) {
  
  rec_name    <- as.character(xgb_refinedsinglebestcv$recipe[[k]])
  best_config <- xgb_refinedsinglebestcv[k, , drop = FALSE]
  
  cat("\n>> Training final model for:", rec_name, "\n")
  
  # FIX: Removed set.seed(... + 1000L) — it was consumed by prep/bake
  
  mtry_best        <- as.integer(best_config$mtry)
  trees_best       <- as.integer(best_config$trees)
  min_n_best       <- as.integer(best_config$min_n)
  tree_depth_best  <- as.integer(best_config$tree_depth)
  learn_rate_best  <- as.numeric(best_config$learn_rate)
  sample_size_best <- as.numeric(best_config$sample_size)
  
  cat("   Best config: mtry=", mtry_best, ", trees=", trees_best, 
      ", min_n=", min_n_best, ", depth=", tree_depth_best,
      ", lr=", learn_rate_best, ", sample=", sample_size_best, "\n", sep="")
  
  rec <- preproc.xgb[[rec_name]]
  
  # Prepare the selected preprocessing recipe on the complete training data
  # and apply the fitted recipe to both training and test observations.
  # Prep and bake FIRST
  rec_prep_final <- prep(rec, training = ntl_train, retain = TRUE)
  train_baked_final <- bake(rec_prep_final, new_data = ntl_train)
  test_baked_final  <- bake(rec_prep_final, new_data = ntl_test)
  
  pnames_final <- setdiff(names(train_baked_final), "mpi")
  pnames_final <- pnames_final[sapply(train_baked_final[, pnames_final, drop = FALSE], is.numeric)]
  n_pred_final <- length(pnames_final)
  
  y_train_final <- train_baked_final$mpi
  y_test_final  <- test_baked_final$mpi
  
  # Separate the outcome from the predictors and convert the final datasets
  # into the numeric format required by XGBoost.
  X_train_final <- train_baked_final %>% select(-mpi) %>% as.data.frame()
  X_test_final  <- test_baked_final  %>% select(-mpi) %>% as.data.frame()
  
  X_train_final[] <- lapply(X_train_final, as.numeric)
  X_test_final[]  <- lapply(X_test_final,  as.numeric)
  
  dtrain_final <- xgb.DMatrix(data = as.matrix(X_train_final), label = y_train_final)
  dtest_final  <- xgb.DMatrix(data = as.matrix(X_test_final),  label = y_test_final)
  
  # Recipe-specific seed for final fit
  xgb_seed_final <- seed_for_refine(rec_name) + 2000L
  
  # Define the final refined XGBoost parameters using the selected
  # hyperparameter values and the final fitting seed.
  params_final <- list(
    objective = "reg:squarederror",
    eta = learn_rate_best,
    max_depth = tree_depth_best,
    min_child_weight = min_n_best,
    subsample = sample_size_best,
    colsample_bytree = mtry_best / max(1, n_pred_final),
    gamma = 0,
    nthread = 1,
    verbosity = 0,
    seed = xgb_seed_final
  )
  
  # FIX (CORE): set.seed() IMMEDIATELY before xgb.train()
  cat("   Training final model (seed=", xgb_seed_final, ")...\n", sep="")
  set.seed(xgb_seed_final)
  # Fit the final refined booster using the complete training dataset.
  final_booster <- xgboost::xgb.train(
    params = params_final,
    data   = dtrain_final,
    nrounds = trees_best,
    verbose = 0
  )
  
  # Save bundled model
  model_file <- file.path(images_dir, paste0("final_refined_xgb_", rec_name, "_", run_id, ".rds"))
  
  if (!file.exists(model_file)) {
    saveRDS(bundle(final_booster), model_file)
    message("Model saved: ", model_file)
  } else {
    message("Model already exists, skipping: ", model_file)
  }
  
  # Generate training and independent-test predictions from the refined model.
  # Predict on training and test
  pred_train_final <- predict(final_booster, dtrain_final)
  pred_test_final  <- predict(final_booster, dtest_final)
  
  # Calculate regression and poverty-classification performance for the
  # complete training and independent test datasets.
  # Calculate metrics
  rmse_train <- yardstick::rmse_vec(y_train_final, pred_train_final)
  mae_train  <- yardstick::mae_vec(y_train_final, pred_train_final)
  rsq_train  <- suppressWarnings(yardstick::rsq_vec(y_train_final, pred_train_final))
  cls_train  <- compute_class_metrics(y_train_final, pred_train_final, threshold = 0.3333)
  
  rmse_test  <- yardstick::rmse_vec(y_test_final, pred_test_final)
  mae_test   <- yardstick::mae_vec(y_test_final, pred_test_final)
  rsq_test   <- suppressWarnings(yardstick::rsq_vec(y_test_final, pred_test_final))
  cls_test   <- compute_class_metrics(y_test_final, pred_test_final, threshold = 0.3333)
  
  cat("   Train RMSE:", round(rmse_train, 4), " | Test RMSE:", round(rmse_test, 4), "\n")
  cat("   Train R²:",   round(rsq_train,  4), " | Test R²:",  round(rsq_test,  4), "\n")
  
  # Save training predictions
  write_csv(
    tibble(
      run_id = run_id, recipe = rec_name,
      mtry = mtry_best, trees = trees_best, min_n = min_n_best,
      tree_depth = tree_depth_best, learn_rate = learn_rate_best, sample_size = sample_size_best,
      xgb_seed = xgb_seed_final,
      row_id = seq_along(y_train_final),
      mpi_true = y_train_final, mpi_pred = pred_train_final
    ),
    final_train_preds_file,
    append = file.exists(final_train_preds_file)
  )
  
  # Save test predictions
  write_csv(
    tibble(
      run_id = run_id, recipe = rec_name,
      mtry = mtry_best, trees = trees_best, min_n = min_n_best,
      tree_depth = tree_depth_best, learn_rate = learn_rate_best, sample_size = sample_size_best,
      xgb_seed = xgb_seed_final,
      row_id = seq_along(y_test_final),
      mpi_true = y_test_final, mpi_pred = pred_test_final
    ),
    final_test_preds_file,
    append = file.exists(final_test_preds_file)
  )
  
  # Save test metrics
  test_metrics_tbl <- tibble(
    run_id = run_id, recipe = rec_name,
    mtry = mtry_best, trees = trees_best, min_n = min_n_best,
    tree_depth = tree_depth_best, learn_rate = learn_rate_best, sample_size = sample_size_best,
    xgb_seed = xgb_seed_final,
    
    train_rmse = rmse_train, train_mae = mae_train, train_rsq = rsq_train,
    train_accuracy_033 = cls_train$accuracy_033,
    train_sensitivity_033 = cls_train$sensitivity_033,
    train_specificity_033 = cls_train$specificity_033,
    train_f1_033 = cls_train$f1_033,
    
    test_rmse = rmse_test, test_mae = mae_test, test_rsq = rsq_test,
    test_accuracy_033 = cls_test$accuracy_033,
    test_sensitivity_033 = cls_test$sensitivity_033,
    test_specificity_033 = cls_test$specificity_033,
    test_f1_033 = cls_test$f1_033
  )
  write_csv(test_metrics_tbl, final_test_metrics_file, append = file.exists(final_test_metrics_file))
  
  final_test_results[[rec_name]] <- test_metrics_tbl
}

cat("\n=============================\n")
cat("REFINED FINAL FITS COMPLETE\n")
cat("Training predictions: ", final_train_preds_file, "\n")
cat("Test predictions:     ", final_test_preds_file,  "\n")
cat("Test metrics:         ", final_test_metrics_file, "\n")
cat("=============================\n")

cat("\n=============================\n")
cat("FINAL TEST SET PERFORMANCE SUMMARY (REFINED)\n")
cat("=============================\n")
final_test_df <- bind_rows(final_test_results)
print(final_test_df %>% select(recipe, test_rmse, test_mae, test_rsq, test_accuracy_033, test_f1_033))
cat("\n")


# ===============================================================
# ULTIMATE BEST: Compare space-filling vs refined CV metrics
# ===============================================================
# Compare the best space-filling and refined configurations using only
# cross-validation performance so the test dataset remains independent.
cat("\n=============================\n")
cat("ULTIMATE BEST MODEL SELECTION\n")
cat("=============================\n")
cat("Using CV metrics to select ultimate best (prevents data leakage)\n\n")

# Get best from space-filling stage (already computed)
sf_cv_best <- xgb_singlebest %>%
  mutate(stage = "space_filling") %>%
  select(stage, recipe, mtry, trees, min_n, tree_depth, learn_rate, sample_size,
         rmse, rmse_se, mae, mae_se, rsq, rsq_se,
         accuracy_033, sensitivity_033, specificity_033, f1_033)

# Get best from refined stage (already computed)
ref_cv_best <- xgb_refinedsinglebestcv %>%
  mutate(stage = "refined") %>%
  select(stage, recipe, mtry, trees, min_n, tree_depth, learn_rate, sample_size,
         rmse, rmse_se, mae, mae_se, rsq, rsq_se,
         accuracy_033, sensitivity_033, specificity_033, f1_033)

# Combine both stages with matching columns only
all_cv_best <- bind_rows(sf_cv_best, ref_cv_best)

# Compare the best initial-stage and refined-stage XGBoost configurations.
# The common model-selection function uses RMSE first, followed by MAE and
# R-squared when the preceding measures are effectively tied.
comparison_xgb <- is_better_model(
  xgb_singlebest,
  xgb_refinedsinglebestcv
)

# Retain the better XGBoost configuration according to the predefined
# model-selection criteria.
if (comparison_xgb$better) {
  
  ultimate_best <- xgb_singlebest %>%
    mutate(stage = "space_filling")
  
} else {
  
  ultimate_best <- xgb_refinedsinglebestcv %>%
    mutate(stage = "refined")
}

View(ultimate_best)

# Get the stage and recipe to be included in the ultimate rds file
stg <- ultimate_best$stage
recp <- ultimate_best$recipe

cat("\n=== ULTIMATE BEST MODEL (BASED ON CV METRICS) ===\n")
print(ultimate_best)
print(stg)
print(recp)


# Save ultimate best
ultimate_best_file <- file.path(out_dir, paste0("xgb_ULTIMATE_best_", stg,"_", recp, "_", run_id, ".csv"))
write_csv(ultimate_best, ultimate_best_file)
cat("\nUltimate best saved to:", ultimate_best_file, "\n")

# Now retrieve test metrics for the ultimate best model
cat("\n=== TEST SET PERFORMANCE OF ULTIMATE BEST ===\n")

# Retrieve the independent-test results that correspond exactly to the
# ultimately selected XGBoost configuration.
if (ultimate_best$stage == "space_filling") {
  # Read space-filling test metrics
  sf_test_all <- readr::read_csv(final_sf_test_metrics_file, show_col_types = FALSE)
  
  # Find matching row (by recipe and hyperparameters)
  ultimate_test_metrics <- sf_test_all %>%
    filter(
      recipe == ultimate_best$recipe,
      mtry == ultimate_best$mtry,
      trees == ultimate_best$trees,
      min_n == ultimate_best$min_n,
      tree_depth == ultimate_best$tree_depth,
      abs(learn_rate - ultimate_best$learn_rate) < 1e-6,
      abs(sample_size - ultimate_best$sample_size) < 1e-6
    ) %>%
    slice(1)
  
} else {
  # Read refined test metrics
  ref_test_all <- readr::read_csv(final_test_metrics_file, show_col_types = FALSE)
  
  # Find matching row (by recipe and hyperparameters)
  ultimate_test_metrics <- ref_test_all %>%
    filter(
      recipe == ultimate_best$recipe,
      mtry == ultimate_best$mtry,
      trees == ultimate_best$trees,
      min_n == ultimate_best$min_n,
      tree_depth == ultimate_best$tree_depth,
      abs(learn_rate - ultimate_best$learn_rate) < 1e-6,
      abs(sample_size - ultimate_best$sample_size) < 1e-6
    ) %>%
    slice(1)
}

if (nrow(ultimate_test_metrics) > 0) {
  cat("Test RMSE: ", round(ultimate_test_metrics$rmse, 3), "\n")
  cat("Test MAE:  ", round(ultimate_test_metrics$mae, 3), "\n")
  cat("Test R²:   ", round(ultimate_test_metrics$rsq, 3), "\n")
  cat("Test Acc:  ", round(ultimate_test_metrics$accuracy, 3), "\n")
  cat("Test F1:   ", round(ultimate_test_metrics$f1, 3), "\n")
} else {
  warning("Could not find test metrics for ultimate best model")
}

# Identify which model file to use and copy to ultimate best
# ultimate_best_model_file <- file.path(images_dir, paste0("xgb_ULTIMATE_best_model_", run_id, ".rds")

# Load the ultimate best model
ultimate_best_model_file <- file.path(images_dir, paste0("xgb_ULTIMATE_best_model_", ultimate_best$stage, "_", ultimate_best$recipe, "_", run_id, ".rds"))


if (ultimate_best$stage == "space_filling") {
  cat("\n Ultimate best is from SPACE-FILLING stage\n")
  cat("   Recipe:", ultimate_best$recipe, "\n")
  cat("   CV RMSE:", round(ultimate_best$rmse, 3), "±", round(ultimate_best$rmse_se, 3), "\n")
  
  # source_model_file <- file.path(images_dir, paste0("xgb_sf_final_", ultimate_best$recipe, "_", run_id, ".rds"))
  source_model_file <- file.path(images_dir,
                                 paste0("xgb_sf_final_", ultimate_best$recipe, "_", run_id, ".rds"))
  cat("   Source model: xgb_sf_final_", ultimate_best$recipe, "_", run_id, ".rds\n", sep = "")
  
  
  # Copy to ultimate best file
  if (file.exists(source_model_file)) {
    file.copy(source_model_file, ultimate_best_model_file, overwrite = TRUE)
    cat("   Ultimate best model saved to:  ", ultimate_best_model_file)
  } else {
    warning("Source model file not found: ", source_model_file)
  }
  
} else {
  cat("\n Ultimate best is from REFINED stage\n")
  cat("   Recipe:", ultimate_best$recipe, "\n")
  cat("   CV RMSE:", round(ultimate_best$rmse, 3), "±", round(ultimate_best$rmse_se, 3), "\n")
  
  source_model_file <- file.path(images_dir, paste0("final_refined_xgb_", ultimate_best$recipe, "_", run_id, ".rds"))
  cat("   Source model: final_refined_xgb_", ultimate_best$recipe, "_", run_id, ".rds\n", sep = "")
  
  # Copy to ultimate best file
  if (file.exists(source_model_file)) {
    file.copy(source_model_file, ultimate_best_model_file, overwrite = TRUE)
    cat("   Ultimate best model saved to: xgb_ULTIMATE_best_model_", run_id, ".rds\n", sep = "")
  } else {
    warning("Source model file not found: ", source_model_file)
  }
}

cat("\n Ultimate best model available at:\n   ", ultimate_best_model_file, "\n")

# ========================================================================
# REPRODUCIBILITY: Save session information
# ========================================================================
session_info_file <- file.path(out_dir, paste0("xgb_sessionInfo_", run_id, ".txt"))
cat("\n=============================\n")
cat("Saving session info for reproducibility...\n")
cat("Session info saved to:", session_info_file, "\n")
cat("=============================\n")

sink(session_info_file)
cat("XGBoost Complete Script - Session Information\n")
cat("==============================================\n")
cat("Run ID:", run_id, "\n")
cat("Date:", as.character(Sys.time()), "\n")
cat("R Version:", R.version.string, "\n\n")
cat("Random Number Generator:\n")
print(RNGkind())
cat("\nGlobal seed: 1238 (set at script top)\n")
cat("Space-filling seeds:\n")
cat("  Grid:      seed_for_recipe(recipe)\n")
cat("  CV tuning: seed_for_recipe(recipe) + 1L\n")
cat("  Final fit: seed_for_sf_final(recipe)\n")
cat("Refined seeds:\n")
cat("  Fold:      seed_for_refine(recipe) + i * 100L\n")
cat("  Final fit: seed_for_refine(recipe) + 2000L\n\n")
cat("==============================================\n\n")
sessionInfo()
sink()

cat("\n=============================\n")
cat("ALL XGBOOST PROCESSING COMPLETE\n")
cat("=============================\n")

#######################################################

# ========================================================================
# PREDICT WITH ULTIMATE BEST XGBOOST MODEL
# ========================================================================
# Load the ultimate best model
ultimate_best_model_file <- file.path(images_dir, paste0("xgb_ULTIMATE_best_model_", ultimate_best$stage, "_", ultimate_best$recipe, "_", run_id, ".rds"))


cat("Loading ultimate best XGBoost model...\n")
modelfitxgb_bundled <- readRDS(ultimate_best_model_file)
modelfitxgb <- bundle::unbundle(modelfitxgb_bundled)

cat("Model loaded successfully\n")
cat("Model type:", class(modelfitxgb), "\n\n")

# Get the recipe for this model (from ultimate_best)
rec_name_ultimate <- ultimate_best$recipe
rec_ultimate <- preproc.xgb[[rec_name_ultimate]]

if (is.null(rec_ultimate)) {
  stop("Recipe not found for ultimate best model: ", rec_name_ultimate)
}

cat("Using recipe:", rec_name_ultimate, "\n")

# ========================================================================
# STEP 1: PREP AND BAKE (REQUIRED for raw XGBoost models)
# ========================================================================
cat("\nPreprocessing data...\n")

# Prep the recipe on training data
rec_prep_ultimate <- prep(rec_ultimate, training = ntl_train, retain = TRUE)

# Bake both train and test
train_baked_ultimate <- bake(rec_prep_ultimate, new_data = ntl_train)
test_baked_ultimate  <- bake(rec_prep_ultimate, new_data = ntl_test)

# Extract target and features
y_train_ultimate <- train_baked_ultimate$mpi
y_test_ultimate  <- test_baked_ultimate$mpi

X_train_ultimate <- train_baked_ultimate %>% select(-mpi) %>% as.data.frame()
X_test_ultimate  <- test_baked_ultimate  %>% select(-mpi) %>% as.data.frame()

# Ensure numeric (required by XGBoost)
X_train_ultimate[] <- lapply(X_train_ultimate, as.numeric)
X_test_ultimate[]  <- lapply(X_test_ultimate,  as.numeric)

# Create DMatrix objects (required by raw XGBoost booster)
dtrain_ultimate <- xgb.DMatrix(data = as.matrix(X_train_ultimate), label = y_train_ultimate)
dtest_ultimate  <- xgb.DMatrix(data = as.matrix(X_test_ultimate),  label = y_test_ultimate)

cat("Data preprocessed\n")
cat("   Train samples:", nrow(X_train_ultimate), "\n")
cat("   Test samples: ", nrow(X_test_ultimate), "\n")
cat("   Features:     ", ncol(X_train_ultimate), "\n\n")

# ========================================================================
# STEP 2: PREDICT
# ========================================================================
cat("Generating predictions...\n")

# Predict on train
pred_train_ultimate <- predict(modelfitxgb, dtrain_ultimate)

# Predict on test
pred_test_ultimate <- predict(modelfitxgb, dtest_ultimate)

# Combine with actual MPI values (from ORIGINAL data, not baked)
preds.train <- tibble(
  .pred = pred_train_ultimate,
  mpi   = ntl_train$mpi,  # Use original MPI, not baked
  .resid = ntl_train$mpi - pred_train_ultimate
)

preds.test <- tibble(
  .pred = pred_test_ultimate,
  mpi   = ntl_test$mpi,   # Use original MPI, not baked
  .resid = ntl_test$mpi - pred_test_ultimate
)

cat("Predictions complete\n\n")

# ========================================================================
# STEP 3: COMPUTE METRICS
# ========================================================================
cat("Computing metrics...\n")

# Training metrics
rmse_train <- yardstick::rmse_vec(preds.train$mpi, preds.train$.pred)
mae_train  <- yardstick::mae_vec(preds.train$mpi,  preds.train$.pred)
rsq_train  <- suppressWarnings(yardstick::rsq_vec(preds.train$mpi, preds.train$.pred))

# Test metrics
rmse_test <- yardstick::rmse_vec(preds.test$mpi, preds.test$.pred)
mae_test  <- yardstick::mae_vec(preds.test$mpi,  preds.test$.pred)
rsq_test  <- suppressWarnings(yardstick::rsq_vec(preds.test$mpi, preds.test$.pred))

cat("\n=== ULTIMATE BEST MODEL PERFORMANCE ===\n")
cat("Recipe:", rec_name_ultimate, "\n")
cat("Stage: ", ultimate_best$stage, "\n\n")

cat("TRAINING SET:\n")
cat("  RMSE: ", round(rmse_train, 3), "\n")
cat("  MAE:  ", round(mae_train,  3), "\n")
cat("  R²:   ", round(rsq_train,  3), "\n\n")

cat("TEST SET:\n")
cat("  RMSE: ", round(rmse_test, 3), "\n")
cat("  MAE:  ", round(mae_test,  3), "\n")
cat("  R²:   ", round(rsq_test,  3), "\n\n")

# ========================================================================
# STEP 4: CREATE RESIDUAL DATA FOR PLOTS (optional)
# ========================================================================
train_resid_data_xgb <- tibble(
  model = "XGBoost",
  .fitted    = as.numeric(preds.train$.pred),
  .resid     = as.numeric(preds.train$.resid),
  mpi        = as.numeric(preds.train$mpi),
  .std_resid = as.numeric(scale(preds.train$.resid)),
  Set        = "Train"
)

test_resid_data_xgb <- tibble(
  model = "XGBoost",
  .fitted    = as.numeric(preds.test$.pred),
  .resid     = as.numeric(preds.test$.resid),
  mpi        = as.numeric(preds.test$mpi),
  .std_resid = as.numeric(scale(preds.test$.resid)),
  Set        = "Test"
)

# Write the test tibble to file
file_name_xgb <- file.path(model_dir, "xgboost", paste0("run_",run_id), paste0("test_resid_data_xgboost", ".csv"))
write_csv(test_resid_data_xgb, file_name_xgb)

cat(" Residual data objects created:\n")
cat("   - train_resid_data_xgb_ultimate\n")
cat("   - test_resid_data_xgb_ultimate\n")
cat("   - preds.train\n")
cat("   - preds.test\n\n")



# ===============================================================
# 11) VARIABLE IMPORTANCE (built-in + permutation)
# ===============================================================
cat("\n=== VARIABLE IMPORTANCE (XGBOOST) ===\n")

# Define the prediction wrapper used by vip when obtaining predictions from
# the ultimately selected XGBoost model.
# wrapper for vip
xgb_pred_wrapper <- function(object, newdata) {
  as.numeric(predict(object, new_data = newdata)$.pred)
}

# Extract the built-in XGBoost variable importance from the ultimately
# selected model and scale the values relative to the most important predictor.
# built-in importance from xgboost engine
# xgb_engine <- workflows::extract_fit_engine(modelfitxgb)
xgb_engine <- modelfitxgb

xgb_vi_builtin <- vip::vi(xgb_engine) %>%
  as_tibble() %>%
  rename(variable = Variable, importance = Importance) %>%
  arrange(desc(importance)) %>%
  mutate(rel_importance = importance / max(importance))

# Save the complete built-in variable-importance results for later reporting.
write_csv(xgb_vi_builtin, file.path(out_dir, paste0("xgb_ULTIMATE_best_model_vi_builtin_", run_id, ".csv")))

# Plot the 15 predictors with the highest built-in XGBoost importance.
p_vi_builtin <- xgb_vi_builtin %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variable, rel_importance), y = rel_importance)) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  coord_flip() +
  labs(
    title = "Variable Importance - XGBoost (Built-in)",
    subtitle = "xgboost importance (scaled 0–1)",
    x = "Predictor Variables",
    y = "Relative Importance (0–1)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_vi_builtin_", run_id, ".png")),
       p_vi_builtin, width = 12, height = 8, dpi = 300)



# ---------------------------------------------------------------
# 12) DIAGNOSTIC PLOTS (Train)
# ---------------------------------------------------------------
# Create a common subtitle containing the search stage, preprocessing recipe
# and main hyperparameters of the ultimately selected XGBoost model.
xgb_subtitle <- paste0(
  "Stage: ", ultimate_best$stage,
  " | Recipe: ", ultimate_best$recipe,
  " | mtry=", ultimate_best$mtry,
  " | trees=", ultimate_best$trees,
  " | min_n=", ultimate_best$min_n
)


# Plot training residuals against fitted MPI values. The zero reference line
# and LOESS smoother help identify systematic residual patterns across predictions.
p1 <- ggplot(train_resid_data_xgb, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 0.9) +
  geom_smooth(method = "loess", color = "darkorange", se = TRUE) +
  labs(title = "Residuals vs Fitted (Train)", x = "Fitted MPI", y = "Residuals") +
  theme_minimal()

# Examine whether the spread of the standardised residuals changes across
# the fitted MPI range.
p2 <- ggplot(train_resid_data_xgb, aes(x = .fitted, y = sqrt(abs(.std_resid)))) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  labs(title = "Scale-Location (Train)", x = "Fitted MPI", y = "√|Std Residuals|") +
  theme_minimal()

# Examine the distribution of the training residuals and compare it visually
# with a normal distribution having the same mean and standard deviation.
p3 <- ggplot(train_resid_data_xgb, aes(x = .resid)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30,
                 fill = "steelblue", color = "black", alpha = 0.55) +
  geom_density(color = "darkred", linewidth = 1) +
  stat_function(fun = dnorm,
                args = list(mean = mean(train_resid_data_xgb$.resid, na.rm = TRUE),
                            sd   = sd(train_resid_data_xgb$.resid, na.rm = TRUE)),
                color = "blue", linetype = "dashed") +
  labs(title = "Residual Distribution (Train)", x = "Residuals", y = "Density") +
  theme_minimal()

# Create a Q-Q plot as an additional visual assessment of the training
# residual distribution.
p4 <- ggplot(train_resid_data_xgb, aes(sample = .resid)) +
  stat_qq(color = "steelblue", alpha = 0.6) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(title = "Q-Q Plot (Train)", x = "Theoretical", y = "Sample") +
  theme_minimal()

# Combine the four training residual plots into a single diagnostic figure.
diagnostic_plots_xgb <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "XGBoost Diagnostic Plots (Train)",
    subtitle = xgb_subtitle,
    caption = paste("N =", nrow(train_resid_data_xgb))
  )

ggsave(
  file.path(plotsdir, "xgboost", paste0("final_model_diagnostics_plots_xgb_", run_id, ".png")),
  diagnostic_plots_xgb, width = 14, height = 10, dpi = 300
)

# ---------------------------------------------------------------
# 13) PREDICTED vs ACTUAL (Train/Test)
# ---------------------------------------------------------------
# Calculate the regression performance measures used to annotate the
# training predicted-versus-actual plot.
# Calculate training metrics for annotation
train_rmse <- rmse_vec(train_resid_data_xgb$mpi, train_resid_data_xgb$.fitted)
train_mae <- mae_vec(train_resid_data_xgb$mpi, train_resid_data_xgb$.fitted)
train_rsq <- rsq_vec(train_resid_data_xgb$mpi, train_resid_data_xgb$.fitted)
train_cor <- cor(train_resid_data_xgb$mpi, train_resid_data_xgb$.fitted)

# Calculate the corresponding performance measures for the independent
# test dataset.
# Calculate test metrics for annotation
test_rmse <- rmse_vec(test_resid_data_xgb$mpi, test_resid_data_xgb$.fitted)
test_mae <- mae_vec(test_resid_data_xgb$mpi, test_resid_data_xgb$.fitted)
test_rsq <- rsq_vec(test_resid_data_xgb$mpi, test_resid_data_xgb$.fitted)
test_cor <- cor(test_resid_data_xgb$mpi, test_resid_data_xgb$.fitted)


# Plot predicted MPI against observed MPI for the training dataset. The
# 45-degree reference line represents perfect agreement between both values.
train_scatter <- ggplot(train_resid_data_xgb, aes(x = mpi, y = .fitted)) +
  geom_point(alpha = 0.55, color = "steelblue", size = 1.2) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 0.9) +
  geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
  coord_equal() +
  labs(title = "Predicted vs Actual (Train)", x = "Actual MPI", y = "Predicted MPI") +
  theme_minimal() +
  annotate("text",
           x = min(train_resid_data_xgb$mpi, na.rm = TRUE),
           y = max(train_resid_data_xgb$.fitted, na.rm = TRUE),
           hjust = 0, vjust = 1,
           label = paste0("RMSE=", sprintf("%.4f", train_rmse),
                          "\nMAE=", sprintf("%.4f", train_mae),
                          "\nR²=", sprintf("%.4f", train_rsq),
                          "\nCorr=", sprintf("%.4f", train_cor)),
           size = 3.6, color = "black")

# Create the corresponding predicted-versus-actual plot for the independent
# test dataset to examine model performance on unseen observations.
test_scatter <- ggplot(test_resid_data_xgb, aes(x = mpi, y = .fitted)) +
  geom_point(alpha = 0.55, color = "steelblue", size = 1.2) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 0.9) +
  geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
  coord_equal() +
  labs(title = "Predicted vs Actual (Test)", x = "Actual MPI", y = "Predicted MPI") +
  theme_minimal() +
  annotate("text",
           x = min(test_resid_data_xgb$mpi, na.rm = TRUE),
           y = max(test_resid_data_xgb$.fitted, na.rm = TRUE),
           hjust = 0, vjust = 1,
           label = paste0("RMSE=", sprintf("%.4f", test_rmse),
                          "\nMAE=", sprintf("%.4f", test_mae),
                          "\nR²=", sprintf("%.4f", test_rsq),
                          "\nCorr=", sprintf("%.4f", test_cor)),
           size = 3.6, color = "black")

ggsave(file.path(plotsdir, "xgboost", paste0("finalmodel_xgb_pred_vs_actual_train_", run_id, ".png")),
       train_scatter, width = 10, height = 8, dpi = 300)

ggsave(file.path(plotsdir, "xgboost", paste0("finalmodel_xgb_pred_vs_actual_test_", run_id, ".png")),
       test_scatter, width = 10, height = 8, dpi = 300)

# ---------------------------------------------------------------
# 14) POVERTY CLASSIFICATION PLOTS (Train/Test)
# ---------------------------------------------------------------


# Classify the training observations according to the observed and predicted
# MPI values using the predefined poverty threshold.
train_cls_data <- bind_cols(
  train_resid_data_xgb,
  .make_classes_label(train_resid_data_xgb$mpi, train_resid_data_xgb$.fitted, thr)
)

# Apply the same poverty classification to the independent test observations.
test_cls_data <- bind_cols(
  test_resid_data_xgb,
  .make_classes_label(test_resid_data_xgb$mpi, test_resid_data_xgb$.fitted, thr)
)

# Plot the training observations by poverty-classification outcome. The
# vertical and horizontal reference lines represent the MPI poverty threshold.
train_class_scatter <- ggplot(train_cls_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.2) +
  geom_vline(xintercept = thr, color = "red", linetype = "dashed", alpha = 0.8) +
  geom_hline(yintercept = thr, color = "red", linetype = "dashed", alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.6) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non_Poor" = "darkgreen",
    "False Poor" = "orange",
    "False Non_Poor" = "purple"
  )) +
  coord_equal() +
  labs(title = "Poverty Classification (Train)", subtitle = "Threshold MPI = 0.3333",
       x = "Actual MPI", y = "Predicted MPI", color = "Class") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

# Create the corresponding poverty-classification plot for the independent
# test dataset.
test_class_scatter <- ggplot(test_cls_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.2) +
  geom_vline(xintercept = thr, color = "red", linetype = "dashed", alpha = 0.8) +
  geom_hline(yintercept = thr, color = "red", linetype = "dashed", alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.6) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non_Poor" = "darkgreen",
    "False Poor" = "orange",
    "False Non_Poor" = "purple"
  )) +
  coord_equal() +
  labs(title = "Poverty Classification (Test)", subtitle = "Threshold MPI = 0.3333",
       x = "Actual MPI", y = "Predicted MPI", color = "Class") +
  theme_minimal() +
  theme(legend.position = "top") +
  guides(color = guide_legend(nrow = 2, byrow = TRUE))

ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_poverty_class_train_", run_id, ".png")),
       train_class_scatter, width = 12, height = 10, dpi = 300)

ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_poverty_class_test_", run_id, ".png")),
       test_class_scatter, width = 12, height = 10, dpi = 300)

# ---------------------------------------------------------------
# 15) MULTI-PANEL (2x2)
# ---------------------------------------------------------------
#title = "Comprehensive XGBoost Performance Analysis",
# Combine the predicted-versus-actual and poverty-classification plots for
# the training and test datasets into a single 2-by-2 figure.
multi_panel_xgb <- (train_scatter + theme(axis.text = element_text(size = 8)) +
                      test_scatter  + theme(axis.text = element_text(size = 8))) /
  (train_class_scatter + theme(axis.text = element_text(size = 8)) +
     test_class_scatter + theme(axis.text = element_text(size = 8))) +
  plot_annotation(
    
    subtitle = xgb_subtitle,
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )

ggsave(
  file.path(plotsdir, "xgboost", paste0("final_model_diagnostics_classification_plots_xgb_", run_id, ".png")),
  multi_panel_xgb, width = 16, height = 12, dpi = 300
)

# ===============================================
# Additional: Residuals vs Actual Values
# ===============================================
#title = "Residuals vs Actual MPI Values - Training Dataset"

# Plot training residuals against the observed MPI values to examine whether
# prediction errors change systematically across the MPI range.
trainresidmpi_xgb <- ggplot(train_resid_data_xgb, aes(x = mpi, y = .resid)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    subtitle = "Check for systematic bias across MPI range",
    x = "Actual MPI",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "xgboost", "/", "final_model_train_residuals_vs_actual_xgboost.png"), trainresidmpi_xgb, width = 10, height = 6, dpi = 300)
cat("Residuals vs actual values plot saved to: residuals_vs_actual.png\n")

# Plot training residuals against fitted MPI values as an additional check
# for systematic prediction patterns.
trainresidmpifit_xgb <- ggplot(train_resid_data_xgb, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    subtitle = "Check for systematic bias across MPI range",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "xgboost", "/", "final_model_train_residuals_vs_fitted_xgboost.png"), trainresidmpifit_xgb, width = 10, height = 6, dpi = 300)
cat("Residuals vs actual values plot saved to: refined_finalmodel_residuals_vs_actual.png\n")

# ===============================================
# Model Assumptions Summary
# ===============================================

# Residuals vs Fitted (Test)
#title = "Residuals vs Fitted (Test Data)",
#subtitle = "Check generalization and bias on unseen data",
# Examine residuals against fitted values for the independent test dataset
# to assess prediction patterns on observations not used to fit the model.
testxgb <- ggplot(test_resid_data_xgb, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "xgboost", "/","final_model_test_residuals_vs_fitted_xgb.png"), testxgb, width = 10, height = 6, dpi = 300)
cat("Residuals vs Actual plot saved as: final_model_residuals_vs_fitted.png\n")



# ------------------------------------------------------------
# Residuals vs Actual
# ------------------------------------------------------------
# Create an additional training residual-versus-actual plot for the residual
# diagnostic summary.
trainactresid <- ggplot(train_resid_data_xgb, aes(x = mpi, y = .resid)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Actual MPI Values (Train Data)",
    subtitle = "Check for systematic bias",
    x = "Actual MPI", y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "xgboost", "/","train_residuals_vs_actual_xgb.png"), trainactresid, width = 10, height = 6, dpi = 300)
cat("residuals vs Actual plot saved as: train_residuals_vs_actual_xgb.png\n")

# Calculate the correlation between fitted values and training residuals as
# an additional numerical summary of the residual pattern.
rresid_cor <- cor(train_resid_data_xgb$.fitted, train_resid_data_xgb$.resid)
cat("\nCorrelation between fitted and residuals:", round(rresid_cor, 6), "\n")
cat("(Should be ≈ 0 for well-specified models)\n")


# Residuals vs Fitted (Test)
# Create the corresponding residual-versus-fitted plot for the independent
# test dataset.
testfitresid <- ggplot(test_resid_data_xgb, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Fitted (Test Data)",
    subtitle = "Check generalization and bias on unseen data",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "xgboost", "/","test_residuals_vs_fitted_nn.png"), testfitresid, width = 10, height = 6, dpi = 300)
cat("Residuals vs Actual plot saved as: refined_test_residuals_vs_fitted_nn.png\n")

# Combine diagnostics (ensure patchwork is loaded)
# Combine the selected training and test residual plots into one figure for
# comparison of residual behaviour across both datasets.
diag_plots <- (trainactresid | testfitresid) +
  plot_annotation(
    subtitle = "Residual-based model checks",
    caption = paste("Residual vs actual (Train) and Residual vs Fitted (Test)")
  )

# Save combined plots
ggsave(
  filename = paste0(plotsdir, "/", "xgboost", "/", "Residual_combine_plots.png"),
  plot = diag_plots,
  width = 14,
  height = 10,
  dpi = 300
)
cat("Diagnostic plots saved as: refined__Residual_plots.png\n")

# ---------------------------------------------------------------
# 17) SUMMARY PRINT
# ---------------------------------------------------------------

# Print the main training and independent-test regression measures used in
# the XGBoost performance plots.
cat("Training:\n")
cat(sprintf("  RMSE: %.4f | MAE: %.4f | R²: %.4f | Corr: %.4f\n", train_rmse, train_mae, train_rsq, train_cor))

cat("Test:\n")
cat(sprintf("  RMSE: %.4f | MAE: %.4f | R²: %.4f | Corr: %.4f\n", test_rmse, test_mae, test_rsq, test_cor))


#####################Neural Network########################


############New code

# ======================================================
# 1. Python/TensorFlow Environment Setup (run once)
# ======================================================
# 1 Remove the old virtualenv
# rm -rf ~/.virtualenvs/r-keras3
# sudo apt install python3.12-venv

# 2 Recreate it using your working system Python
#python3.10 -m venv ~/.virtualenvs/r-keras3

# 3 Activate it
#source ~/.virtualenvs/r-keras3/bin/activate

# 4 Install packages manually
#pip install --upgrade pip setuptools wheel
#pip install tensorflow keras numpy pandas
# pip show tensorflow

# Verify reticulate will use it
#Sys.unsetenv("RETICULATE_PYTHON")
library(reticulate)
library(tensorflow)
library(keras3)

# ===============================================================
# GLOBAL REPRODUCIBILITY (TF/Keras)  
# ===============================================================
Sys.setenv(
  PYTHONHASHSEED = "124",
  TF_CPP_MIN_LOG_LEVEL = "2",
  CUDA_VISIBLE_DEVICES = "-1"
)

reticulate::use_virtualenv("r-keras3", required = TRUE)
reticulate::py_config()
py_run_string("import sys; print(sys.executable)")
py_run_string("import tensorflow as tf; print('TF version:', tf.__version__)")
py_run_string("import keras; print('Keras version:', keras.__version__)")

try(tensorflow::tf$config$experimental$enable_op_determinism(), silent = TRUE)
try(tensorflow::tf$random$set_seed(124L), silent = TRUE)

try(tensorflow::tf$config$threading$set_intra_op_parallelism_threads(1L), silent = TRUE)
try(tensorflow::tf$config$threading$set_inter_op_parallelism_threads(1L), silent = TRUE)

# Create a deterministic seed from the recipe, grid configuration and fold
# so that each neural network fit can be reproduced as closely as possible.
seed_for_nn <- function(recipe_name, grid_id, fold_id, base = 124L) {
  as.integer((base +
                sum(utf8ToInt(paste0("NN_", recipe_name))) +
                1000L * grid_id +
                fold_id) %% .Machine$integer.max)
}

# -------------------------
# Output files
# -------------------------

runs_root <- file.path(model_dir, "neuralnetbetareg")

run_dirs <- list.dirs(
  runs_root,
  recursive = FALSE,
  full.names = TRUE
)

# Keep only folders named like "run_YYYYMMDD_HHMMSS"
run_dirs_run <- run_dirs[grepl("^run_\\d{8}_\\d{6}$", basename(run_dirs))]

run_folder <- NA_character_

if (length(run_dirs_run) > 0) {
  # Sort by run_id embedded in folder name (lexicographic works for YYYYMMDD_HHMMSS)
  run_dirs_run <- run_dirs_run[order(basename(run_dirs_run))]
  run_folder <- tail(run_dirs_run, 1)
}

run_folder 


run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")  # always set


if (!is.na(run_folder) && dir.exists(run_folder)) {
  out_dir <- run_folder
} else {
  out_dir <- file.path(model_dir, "neuralnetbetareg", paste0("run_", run_id))
}


cache_dir   <- file.path(out_dir, "fold_cache") 
metrics_dir <- file.path(out_dir, "metrics_by_grid") 
preds_dir   <- file.path(out_dir, "fold_preds_by_grid") 
timing_dir  <- file.path(out_dir, "timing")
errs_dir    <- file.path(out_dir, "errors_by_grid") 
br_dir      <- file.path(out_dir, "Best_Results")
bsttestpre_dir      <- file.path(out_dir, "bsttestpre") 

dir.create(out_dir,     recursive = TRUE, showWarnings = FALSE) 
dir.create(cache_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(preds_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(errs_dir,    recursive = TRUE, showWarnings = FALSE)
dir.create(timing_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(br_dir,      recursive = TRUE, showWarnings = FALSE)
dir.create(bsttestpre_dir,      recursive = TRUE, showWarnings = FALSE)

#Create files
timing_path <- file.path(timing_dir, paste0("timing_", run_id, ".csv"))

# ===============================================================
# 1.  Data Split - training and test
# ===============================================================
#set.seed(1238)
#data_split <- initial_split(df, prop = 0.70)
#ntl_train <- training(data_split)
#ntl_test  <- testing(data_split)

# ===============================================================
#  2. Cross validation Process
# ===============================================================
set.seed(1238)
ntl_vfolds_cv <- vfold_cv(ntl_train, v = 10, repeats = 4)

# ===============================================================
# 3. RECIPE DEFINITION
# ===============================================================
# use previously defined recipe at the top of the page

# ===============================================================
# 4. TRANSFORMATION FUNCTIONS (Link Functions)
# ===============================================================

# Logit: log(p / (1-p))
# Define the candidate link transformations used to map MPI from the
# probability scale before fitting the neural network.
logit_transform <- function(p) {
  eps <- 1e-6
  p_clipped <- pmax(pmin(p, 1 - eps), eps)
  log(p_clipped / (1 - p_clipped))
}

inverse_logit <- function(x) {
  1 / (1 + exp(-x))
}

# Probit: Inverse CDF of standard normal
probit_transform <- function(p) {
  eps <- 1e-6
  p_clipped <- pmax(pmin(p, 1 - eps), eps)
  qnorm(p_clipped)
}

inverse_probit <- function(x) {
  pnorm(x)
}

# Complementary log-log: log(-log(1-p))
cloglog_transform <- function(p) {
  eps <- 1e-6
  p_clipped <- pmax(pmin(p, 1 - eps), eps)
  log(-log(1 - p_clipped))
}

inverse_cloglog <- function(x) {
  1 - exp(-exp(x))
}

# Cauchit: Inverse CDF of Cauchy distribution
cauchit_transform <- function(p) {
  eps <- 1e-6
  p_clipped <- pmax(pmin(p, 1 - eps), eps)
  tan(pi * (p_clipped - 0.5))
}

inverse_cauchit <- function(x) {
  0.5 + atan(x) / pi
}

# Log: log(p)
log_transform <- function(p) {
  eps <- 1e-6
  p_clipped <- pmax(p, eps)
  log(p_clipped)
}

inverse_log <- function(x) {
  exp(x)
}

# Loglog: log(-log(p))
loglog_transform <- function(p) {
  eps <- 1e-6
  p_clipped <- pmax(pmin(p, 1 - eps), eps)
  log(-log(p_clipped))
}

inverse_loglog <- function(x) {
  exp(-exp(x))
}

# ===============================================================
# 5. LINK FUNCTION DISPATCHER
# ===============================================================
#"log"     = log_transform(p),
#"loglog"  = loglog_transform(p),

# Select the required forward link transformation according to the
# link function assigned to the current hyperparameter configuration.
apply_link <- function(p, link = "logit") {
  switch(link,
         "logit"   = logit_transform(p),
         "probit"  = probit_transform(p),
         "cloglog" = cloglog_transform(p),
         "cauchit" = cauchit_transform(p),
         stop("Unknown link function: ", link)
  )
}


# Convert neural network predictions from the transformed scale back to
# the original MPI scale.
apply_inverse_link <- function(x, link = "logit") {
  switch(link,
         "logit"   = inverse_logit(x),
         "probit"  = inverse_probit(x),
         "cloglog" = inverse_cloglog(x),
         "cauchit" = inverse_cauchit(x),
         stop("Unknown link function: ", link)
  )
}

# ===============================================================
# 6. HELPER FUNCTION FOR SAFE RSQ CALCULATION
# ===============================================================
eps <- 1e-6
cliptarg <- function(p, eps = 1e-6) pmax(pmin(p, 1 - eps), eps)



# HELPER FUNCTION FOR SAFE RSQ CALCULATION
rsq_safe <- function(truth, estimate) {
  tryCatch({
    rsq_vec(truth, estimate)
  }, error = function(e) {
    NA_real_
  })
}


# CLEANUP FUNCTION
cleanup_env <- function() {
  if (reticulate::py_available()) {
    tryCatch({
      keras3::clear_session()
      gc(verbose = FALSE)
    }, error = function(e) invisible(NULL))
  }
}


# Cleanup (fast TF cleanup always; expensive python gc only sometimes)

hard_cleanup <- function(py_gc = FALSE) {
  try(keras3::clear_session(), silent = TRUE)
  try(gc(verbose = FALSE), silent = TRUE)
  if (py_gc && reticulate::py_available()) {
    try(reticulate::py_run_string("import gc; gc.collect()"), silent = TRUE)
  }
  invisible(NULL)
}

# -------------------------
# Log ANY failure reason (even when we "return NA" without an R error)
# -------------------------
log_fail <- function(err_path, recipe, grid_id, fold, stage, cfg, msg) {
  row <- tibble(
    time = as.character(Sys.time()),
    recipe = recipe,
    grid_id = grid_id,
    fold = fold,
    stage = stage,
    link = as.character(cfg$link),
    units1 = as.integer(cfg$units1),
    units2 = as.integer(cfg$units2),
    dropout = as.numeric(cfg$dropout),
    learn_rate = as.numeric(cfg$learn_rate),
    penalty = as.numeric(cfg$penalty),
    optimizer = as.character(cfg$optimizer),
    activation1 = as.character(cfg$activation1),
    activation2 = as.character(cfg$activation2),
    epoch = as.integer(cfg$epoch),
    bsize = as.integer(cfg$bsize),
    batch_norm = as.character(cfg$batch_norm),
    msg = as.character(msg)
  )
  write_csv(row, err_path, append = file.exists(err_path))
  invisible(NULL)
}

# -------------------------
# Repair matrices: replace NA/Inf using TRAIN fold column means
# This prevents "all folds NA" from a single missing value.
# -------------------------
# Repair non-finite predictor values using means calculated from the
# training fold only, and apply the same values to the validation fold.
repair_fold_mats <- function(Xtrain, Xval) {
  Xtrain <- as.matrix(Xtrain)
  Xval   <- as.matrix(Xval)
  storage.mode(Xtrain) <- "double"
  storage.mode(Xval)   <- "double"
  
  # mark non-finite as NA
  Xtrain[!is.finite(Xtrain)] <- NA_real_
  Xval[!is.finite(Xval)]     <- NA_real_
  
  # column means from train (ignore NA)
  cm <- colMeans(Xtrain, na.rm = TRUE)
  
  # if a whole column is NA -> mean becomes NaN; set to 0
  cm[!is.finite(cm)] <- 0
  
  # impute train
  idx <- which(is.na(Xtrain), arr.ind = TRUE)
  if (nrow(idx) > 0) Xtrain[idx] <- cm[idx[,2]]
  
  # impute val using train means
  idx2 <- which(is.na(Xval), arr.ind = TRUE)
  if (nrow(idx2) > 0) Xval[idx2] <- cm[idx2[,2]]
  
  list(Xtrain = Xtrain, Xval = Xval)
}

# ----------------------------
# Standard Error helper
# SE = sd(x) / sqrt(n_non_na)
# ----------------------------
se_of <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n <= 1) return(NA_real_)
  stats::sd(x) / sqrt(n)
}

# ===============================================================
# OPTIMIZED MODEL BUILDER (prevents TensorFlow retracing)
# ===============================================================
# Build the neural network architecture using the hyperparameters supplied
# by the current configuration, including regularisation and optimisation settings.
build_nn_transform_fixed <- function(input_shape,
                                     units1,
                                     units2 = NULL,
                                     dropout,
                                     penalty,
                                     learn_rate,
                                     activation1 = "relu",
                                     activation2 = "relu",
                                     optimizer = "adam",
                                     batch_norm = TRUE,
                                     add_early_stopping = TRUE) {
  
  # Ensure consistent types (prevents retracing)
  lr  <- as.numeric(learn_rate)
  pen <- as.numeric(penalty)
  units1 <- as.integer(units1)
  units2 <- if (!is.null(units2) && is.finite(units2)) as.integer(units2) else NULL
  dropout <- as.numeric(dropout)
  
  # Clear session before building (prevents graph buildup)
  keras3::clear_session()
  
  input <- keras3::layer_input(shape = as.integer(input_shape), dtype = "float32")
  
  x <- input |>
    keras3::layer_dense(
      units = units1,
      activation = activation1,
      kernel_regularizer = keras3::regularizer_l2(pen),
      kernel_initializer = keras3::initializer_he_normal(),
      dtype = "float32"  # Explicit dtype
    )
  
  if (isTRUE(batch_norm)) x <- x |> keras3::layer_batch_normalization()
  x <- x |> keras3::layer_dropout(rate = dropout)
  
  if (!is.null(units2) && units2 > 0) {
    x <- x |>
      keras3::layer_dense(
        units = units2,
        activation = activation2,
        kernel_regularizer = keras3::regularizer_l2(pen),
        kernel_initializer = keras3::initializer_he_normal(),
        dtype = "float32"
      )
    if (isTRUE(batch_norm)) x <- x |> keras3::layer_batch_normalization()
    x <- x |> keras3::layer_dropout(rate = dropout)
  }
  
  output <- x |> keras3::layer_dense(units = 1L, activation = "linear", name = "output", dtype = "float32")
  model <- keras3::keras_model(inputs = input, outputs = output)
  
  opt <- switch(
    as.character(optimizer),
    "adam"    = keras3::optimizer_adam(learning_rate = lr, clipnorm = 1.0),
    "rmsprop" = keras3::optimizer_rmsprop(learning_rate = lr, clipnorm = 1.0),
    "sgd"     = keras3::optimizer_sgd(learning_rate = lr, momentum = 0.9, clipnorm = 1.0),
    keras3::optimizer_adam(learning_rate = lr, clipnorm = 1.0)
  )
  
  # Use jit_compile to reduce retracing (if supported)
  compile_args <- list(
    loss = "mse",
    optimizer = opt,
    metrics = list("mae")
  )
  
  # Try to enable jit_compile (TF 2.x feature)
  tryCatch({
    compile_args$jit_compile <- TRUE
    do.call(keras3::compile, c(list(object = model), compile_args))
  }, error = function(e) {
    # Fallback without jit_compile if not supported
    do.call(keras3::compile, c(list(object = model), compile_args))
  })
  
  if (isTRUE(add_early_stopping)) {
    attr(model, "callbacks") <- list(
      keras3::callback_early_stopping(
        patience = 10,  # Reduced from 15
        restore_best_weights = TRUE,
        monitor = "val_loss",
        min_delta = 1e-4,
        verbose = 0
      ),
      keras3::callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.5,
        patience = 5,  # Reduced from 7
        min_lr = 1e-6,
        verbose = 0
      )
    )
  }
  
  model
}

# ===============================================================
# 8. PARAMETER GRID WITH LINK FUNCTION
# ===============================================================
#32,128 - first, 16,64 - second
units1_param   <- new_quant_param(range = c(32, 256), inclusive = c(TRUE, TRUE), trans = NULL)
units2_param   <- new_quant_param(range = c(16, 128),  inclusive = c(TRUE, TRUE), trans = NULL)
dropout_param  <- new_quant_param(range = c(0.2, 0.4), inclusive = c(TRUE, TRUE), trans = NULL)

learn_rate_param <- dials::learn_rate(range = c(-5, -1))
penalty_param  <- dials::penalty(range = c(-5, -3))

activation1_param <- new_qual_param(values = c("relu", "elu", "selu", "tanh"))
activation2_param <- new_qual_param(values = c("relu", "elu", "tanh"))
optimizer_param <- new_qual_param(values = c("adam", "rmsprop", "sgd"))

# NEW: Link function parameter
link_param <- new_qual_param(values = c("logit", "probit", "cloglog", "cauchit"))

epoch_param <- new_quant_param(values = c(150, 200), trans = NULL)

# bs_param <- new_quant_param(values = c(32, 64, 128), trans = NULL)
bs_param <- new_quant_param(values = c(32, 64), trans = NULL)

batch_norm_param <- new_qual_param(values = c("TRUE", "FALSE"))

#  
# Combine the neural network tuning parameters into a single parameter set
# used to construct the initial space-filling hyperparameter search.
param_set <- parameters(list(
  units1      = units1_param,
  units2      = units2_param,
  dropout     = dropout_param,
  penalty     = penalty_param,
  learn_rate  = learn_rate_param,
  activation1 = activation1_param,
  activation2 = activation2_param,
  optimizer   = optimizer_param,
  link        = link_param,
  bsize       = bs_param,
  epoch       = epoch_param,
  batch_norm  = batch_norm_param
))

set.seed(1234)
# Generate the initial space-filling grid and convert parameters that must
# be represented as integer or logical values before model training.
param_grid <- grid_space_filling(param_set, size = 95) |>  # Increased to test more links
  mutate(
    units1 = as.integer(round(units1)),
    units2 = as.integer(round(units2)),
    epoch  = as.integer(round(epoch)),
    bsize  = as.integer(round(bsize)),
    batch_norm  = as.logical(batch_norm),
  )


# ===============================================================
# 10. COMPLETE TRAINING LOOP WITH LINK FUNCTIONS
# ===============================================================

preproc.neuralnet <- list(
  all_pred_tfmwithlogcubenormalize = all_pred_tfmwithlogcubenormalize.recipe,
  all_pred_tfmwithlogcuberange = all_pred_tfmwithlogcuberange.recipe
)

#======================Final improvement====================

# ===============================================================
# A) CACHE FOLD DATA ONCE PER RECIPE
# ===============================================================
# Preprocess each cross-validation fold once per recipe and save the resulting
# training and validation matrices so they can be reused across grid configurations.
make_recipe_cache <- function(recipe_name, rec_obj, folds, cache_dir, outcome = "mpi") {
  
  recipe_cache_dir <- file.path(cache_dir, recipe_name)
  dir.create(recipe_cache_dir, recursive = TRUE, showWarnings = FALSE)
  
  for (i in seq_along(folds$splits)) {
    
    split <- folds$splits[[i]]
    train <- rsample::analysis(split)
    val   <- rsample::assessment(split)
    
    prep_rec <- prep(rec_obj, training = train)
    
    train_proc <- bake(prep_rec, new_data = train)
    val_proc   <- bake(prep_rec, new_data = val)
    
    predictors <- setdiff(names(train_proc), outcome)
    predictors <- predictors[sapply(train_proc[, predictors, drop = FALSE], is.numeric)]
    
    Xtrain <- data.matrix(train_proc[, predictors, drop = FALSE])
    Xval   <- data.matrix(val_proc[, predictors, drop = FALSE])
    
    ytrain <- as.numeric(train_proc[[outcome]])
    yval   <- as.numeric(val_proc[[outcome]])
    
    # Repair predictor matrices BEFORE caching (big win)
    repaired <- repair_fold_mats(Xtrain, Xval)
    Xtrain <- repaired$Xtrain
    Xval   <- repaired$Xval
    
    payload <- list(
      predictors = predictors,
      Xtrain = Xtrain,
      Xval   = Xval,
      ytrain = ytrain,
      yval   = yval
    )
    
    saveRDS(payload, file = file.path(recipe_cache_dir, paste0("fold_", i, ".rds")))
    
    rm(payload, Xtrain, Xval, ytrain, yval, train_proc, val_proc, prep_rec, repaired)
    gc(FALSE)
  }
  
  invisible(recipe_cache_dir)
}

for (r in names(preproc.neuralnet)) {
  recipe_cache_dir <- file.path(cache_dir, r)
  if (!dir.exists(recipe_cache_dir) || length(list.files(recipe_cache_dir, pattern = "^fold_\\d+\\.rds$")) == 0) {
    cat("Caching recipe:", r, "\n")
    make_recipe_cache(r, preproc.neuralnet[[r]], ntl_vfolds_cv, cache_dir)
    hard_cleanup(py_gc = TRUE)
  } else {
    cat("Cache already exists for recipe:", r, "\n")
  }
}
cat("\n Fold cache ready at:", cache_dir, "\n")

# ===============================================================
# B) TRAIN USING CACHED FOLDS (stream metrics; per-grid preds/errors)
# ===============================================================

write_timing <- function(level, recipe, grid_id, start_time, end_time, extra = list()) {
  row <- tibble::tibble(
    run_id = run_id,
    level = level,                       # "recipe" or "grid"
    recipe = recipe,
    grid_id = grid_id,
    start_time = as.character(start_time),
    end_time   = as.character(end_time),
    duration_secs = as.numeric(difftime(end_time, start_time, units = "secs")),
    duration_mins = as.numeric(difftime(end_time, start_time, units = "mins"))
  )
  if (length(extra) > 0) row <- dplyr::bind_cols(row, tibble::as_tibble(extra))
  readr::write_csv(row, timing_path, append = file.exists(timing_path))
  invisible(NULL)
}


# ===============================================================
# RESUME HELPERS (robust skip + atomic write)
# ===============================================================

grid_metrics_path <- function(metrics_dir, recipe, g) {
  file.path(metrics_dir, paste0("metrics__", recipe, "__grid_", g, ".csv"))
}

grid_is_done <- function(metrics_path) {
  if (!file.exists(metrics_path)) return(FALSE)
  info <- file.info(metrics_path)
  if (is.na(info$size) || info$size <= 0) return(FALSE)
  
  ok <- tryCatch({
    x <- readr::read_csv(metrics_path, show_col_types = FALSE, progress = FALSE)
    # must have at least 1 row and rmse column to count as done
    nrow(x) >= 1 && ("rmse" %in% names(x))
  }, error = function(e) FALSE)
  
  ok
}

atomic_write_csv <- function(df, path) {
  tmp <- paste0(path, ".tmp")
  readr::write_csv(df, tmp)
  # overwrite if an old file exists
  if (file.exists(path)) file.remove(path)
  ok <- file.rename(tmp, path)
  if (!ok) stop("Failed to rename tmp metrics file: ", tmp, " -> ", path)
  invisible(TRUE)
}

# Configuration: Adjust based on your system
n_workers <- min(parallel::detectCores() - 1, 4)  # Use up to 4 workers
grids_per_chunk <- 3  # Process 3 grids per worker before cleanup

cat("\n Starting optimized training with", n_workers, "workers\n")
cat("   Processing", grids_per_chunk, "grids per chunk\n\n")

for (r in names(preproc.neuralnet)) {
  
  recipe_start <- Sys.time()
  cat("\n TRAIN Recipe:", r, "\n")
  
  recipe_cache_dir <- file.path(cache_dir, r)
  if (!dir.exists(recipe_cache_dir)) stop("Cache not found for recipe: ", r)
  
  # Restart workers per recipe to avoid memory buildup
  future::plan(sequential)
  hard_cleanup(py_gc = TRUE)
  future::plan(multisession, workers = n_workers)
  
  grid_ids <- seq_len(nrow(param_grid))
  
  # Split grids into chunks for batch processing
  grid_chunks <- split(grid_ids, ceiling(seq_along(grid_ids) / grids_per_chunk))
  
  cat("   Total grids:", length(grid_ids), "| Chunks:", length(grid_chunks), "\n")
  
  # Evaluate the hyperparameter configurations in parallel chunks. Each grid
  # is assessed across the cached cross-validation folds.
  furrr::future_walk(
    grid_chunks,
    function(chunk_grids) {
      
      suppressPackageStartupMessages({
        library(dplyr); library(tibble); library(purrr)
        library(yardstick); library(readr)
        library(reticulate); library(keras3); library(tensorflow)
      })
      
      reticulate::use_virtualenv("r-keras3", required = TRUE)
      
      #  Enable optimizations
      Sys.setenv(CUDA_VISIBLE_DEVICES = "-1", TF_CPP_MIN_LOG_LEVEL = "2")
      try(tensorflow::tf$config$experimental$enable_op_determinism(), silent = TRUE)
      try(tensorflow::tf$config$threading$set_intra_op_parallelism_threads(1L), silent = TRUE)
      try(tensorflow::tf$config$threading$set_inter_op_parallelism_threads(1L), silent = TRUE)
      try(tensorflow::tf$config$optimizer$set_jit(TRUE), silent = TRUE)  # XLA compilation
      
      # Process each grid in this chunk
      for (g in chunk_grids) {
        
        grid_start <- Sys.time()
        cfg <- as.list(param_grid[g, ])
        
        metrics_path <- grid_metrics_path(metrics_dir, r, g)
        preds_path   <- file.path(preds_dir,   paste0("fold_preds__", r, "__grid_", g, ".csv"))
        err_path     <- file.path(errs_dir,    paste0("errors__", r, "__grid_", g, ".csv"))
        
        # =========================================================
        # RESUME LOGIC (robust):
        # - skip only if metrics file is valid
        # - if file exists but is empty/corrupt, re-run the grid
        # =========================================================
        if (grid_is_done(metrics_path)) {
          grid_end <- Sys.time()
          write_timing(
            level = "grid", recipe = r, grid_id = g,
            start_time = grid_start, end_time = grid_end,
            extra = list(status = "skipped_exists_valid")
          )
          next  # Skip to next grid in chunk
        } else {
          # if a previous crash left partial files, remove them to avoid confusion
          if (file.exists(metrics_path)) file.remove(metrics_path)
          if (file.exists(paste0(metrics_path, ".tmp"))) file.remove(paste0(metrics_path, ".tmp"))
        }
        
        # Fit the current neural network configuration across all cross-validation
        # folds and collect regression and poverty-classification performance measures.
        fold_metrics <- purrr::map_dfr(seq_along(ntl_vfolds_cv$splits), function(i) {
          
          tryCatch({
            
            dat <- readRDS(file.path(recipe_cache_dir, paste0("fold_", i, ".rds")))
            
            Xtrain <- dat$Xtrain
            Xval   <- dat$Xval
            ytrain <- dat$ytrain
            yval   <- dat$yval
            
            ytrain <- cliptarg(ytrain)
            yval   <- cliptarg(yval)
            
            # Link (no target standardization)
            ytrain_link <- apply_link(ytrain, link = cfg$link)
            yval_link   <- apply_link(yval,   link = cfg$link)
            
            s <- seed_for_nn(r, g, i)
            set.seed(s)  # R-side
            try(tensorflow::tf$random$set_seed(as.integer(s)), silent = TRUE)
            
            keras3::clear_session()
            model <- build_nn_transform_fixed(
              input_shape = ncol(Xtrain),
              units1 = cfg$units1,
              units2 = cfg$units2,
              dropout = cfg$dropout,
              penalty = cfg$penalty,
              learn_rate = cfg$learn_rate,
              activation1 = cfg$activation1,
              activation2 = cfg$activation2,
              optimizer   = cfg$optimizer,
              batch_norm  = as.logical(cfg$batch_norm),
              add_early_stopping = TRUE
            )
            
            callbacks <- attr(model, "callbacks")
            
            model |> fit(
              Xtrain, ytrain_link,
              epochs = as.integer(cfg$epoch),
              batch_size = as.integer(cfg$bsize),
              validation_data = list(Xval, yval_link),
              callbacks = callbacks,
              shuffle = FALSE,
              verbose = 0
            )
            
            #pred_link <- as.numeric(predict(model, Xval, verbose = 0))
            #pred_prob <- cliptarg(apply_inverse_link(pred_link, link = cfg$link))
            
            Xval_tf <- tensorflow::tf$convert_to_tensor(Xval, dtype = "float32")
            
            pred_link <- as.numeric(model(Xval_tf, training = FALSE))
            pred_prob <- cliptarg(apply_inverse_link(pred_link, link = cfg$link))
            
            if (any(!is.finite(pred_prob))) {
              log_fail(err_path, r, g, i, "predict", cfg, "pred_prob has Inf/NaN")
              rm(model, dat); hard_cleanup(py_gc = FALSE)
              return(tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_,
                            accuracy=NA_real_, sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_))
            }
            
            # Save fold preds (append)
            fold_pred_tbl <- tibble(
              run_id = run_id,
              recipe = r, grid_id = g, fold = i, link = cfg$link,
              row_id = seq_along(yval),
              mpi_true = yval,
              mpi_pred = pred_prob
            )
            write_csv(fold_pred_tbl, preds_path, append = file.exists(preds_path))
            
            # Metrics
            rmse <- rmse_vec(yval, pred_prob)
            mae  <- mae_vec(yval, pred_prob)
            rsq  <- rsq_safe(yval, pred_prob)
            
            # Replace the existing cm block with this in both loops
            thr    <- 0.3333
            actual <- factor(ifelse(yval     >= thr, "poor", "non_poor"), 
                             levels = c("poor", "non_poor"))
            pred   <- factor(ifelse(pred_prob >= thr, "poor", "non_poor"), 
                             levels = c("poor", "non_poor"))
            
            cm <- table(Actual = actual, Predicted = pred)
            
            # Named extraction — poor is positive class
            tp <- cm["poor",     "poor"]
            fn <- cm["poor",     "non_poor"]
            fp <- cm["non_poor", "poor"]
            tn <- cm["non_poor", "non_poor"]
            
            accuracy    <- (tp + tn) / max(1, sum(cm))
            sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
            specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
            precision   <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
            f1          <- if (is.finite(precision) && is.finite(sensitivity) && 
                               (precision + sensitivity) > 0)
              2 * (precision * sensitivity) / (precision + sensitivity) 
            else NA_real_
            
            rm(model, dat)
            # Less frequent Python GC (only every 20 folds instead of 10)
            hard_cleanup(py_gc = (i %% 20 == 0))
            
            tibble(rmse=rmse, mae=mae, rsq=rsq,
                   accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, f1=f1)
            
          }, error = function(e) {
            log_fail(err_path, r, g, i, "R_error", cfg, conditionMessage(e))
            hard_cleanup(py_gc = FALSE)
            tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_,
                   accuracy=NA_real_, sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_)
          })
        })
        
        successful_folds <- sum(!is.na(fold_metrics$rmse))
        total_folds <- nrow(fold_metrics)
        
        # ----------------------------
        # SEs across folds (based on non-missing, finite values)
        # ----------------------------
        se_rmse <- se_of(fold_metrics$rmse)
        se_mae  <- se_of(fold_metrics$mae)
        se_rsq  <- se_of(fold_metrics$rsq)
        
        se_accuracy    <- se_of(fold_metrics$accuracy)
        se_sensitivity <- se_of(fold_metrics$sensitivity)
        se_specificity <- se_of(fold_metrics$specificity)
        se_f1          <- se_of(fold_metrics$f1)
        
        grid_row <- tibble(
          run_id = run_id,
          recipe = r, grid_id = g,
          link = cfg$link,
          units1 = cfg$units1, units2 = cfg$units2,
          dropout = cfg$dropout,
          learn_rate = cfg$learn_rate,
          penalty = cfg$penalty,
          optimizer = cfg$optimizer,
          activation1 = cfg$activation1,
          activation2 = cfg$activation2,
          epoch = cfg$epoch,
          bsize = cfg$bsize,
          batch_norm = cfg$batch_norm,
          
          rmse = mean(fold_metrics$rmse, na.rm = TRUE),
          se_rmse = se_rmse,
          
          mae  = mean(fold_metrics$mae,  na.rm = TRUE),
          se_mae = se_mae,
          
          rsq  = mean(fold_metrics$rsq,  na.rm = TRUE),
          se_rsq = se_rsq,
          
          accuracy = mean(fold_metrics$accuracy, na.rm = TRUE),
          se_accuracy = se_accuracy,
          
          sensitivity = mean(fold_metrics$sensitivity, na.rm = TRUE),
          se_sensitivity = se_sensitivity,
          
          specificity = mean(fold_metrics$specificity, na.rm = TRUE),
          se_specificity = se_specificity,
          
          f1 = mean(fold_metrics$f1, na.rm = TRUE),
          se_f1 = se_f1,
          
          n_successful_folds = successful_folds,
          total_folds = total_folds,
          success_rate = if (total_folds > 0) successful_folds / total_folds else NA_real_
        )
        
        # =========================================================
        # ATOMIC METRICS WRITE (prevents "exists but corrupt")
        # =========================================================
        atomic_write_csv(grid_row, metrics_path)
        
        # ---- GRID TIMING WRITE ----
        grid_end <- Sys.time()
        write_timing(
          level = "grid",
          recipe = r,
          grid_id = g,
          start_time = grid_start,
          end_time = grid_end,
          extra = list(
            status = "done",
            link = as.character(cfg$link),
            units1 = as.integer(cfg$units1),
            units2 = as.integer(cfg$units2),
            dropout = as.numeric(cfg$dropout),
            learn_rate = as.numeric(cfg$learn_rate),
            penalty = as.numeric(cfg$penalty),
            optimizer = as.character(cfg$optimizer),
            activation1 = as.character(cfg$activation1),
            activation2 = as.character(cfg$activation2),
            epoch = as.integer(cfg$epoch),
            bsize = as.integer(cfg$bsize),
            batch_norm = as.logical(cfg$batch_norm),
            n_successful_folds = successful_folds,
            total_folds = total_folds
          )
        )
        
        # Light cleanup after each grid (no Python GC)
        hard_cleanup(py_gc = FALSE)
        
      } # End of grid loop within chunk
      
      # Heavy cleanup after processing entire chunk
      hard_cleanup(py_gc = TRUE)
      invisible(NULL)
      
    },
    .options = furrr_options(seed = TRUE)
  )
  
  # Cleanup after recipe in main session
  future::plan(sequential)
  hard_cleanup(py_gc = TRUE)
  gc(FALSE)
  
  # ---- RECIPE TIMING WRITE ----
  recipe_end <- Sys.time()
  write_timing(
    level = "recipe",
    recipe = r,
    grid_id = NA_integer_,
    start_time = recipe_start,
    end_time = recipe_end,
    extra = list(status = "done")
  )
  
  cat(" Recipe", r, "completed in", 
      round(as.numeric(difftime(recipe_end, recipe_start, units = "hours")), 2), 
      "hours\n")
}

cat("\n All recipes completed!\n")

# Merge metrics
all_metrics <- list.files(metrics_dir, full.names = TRUE, pattern = "^metrics__.*grid_.*\\.csv$")
final_metrics <- purrr::map_dfr(all_metrics, readr::read_csv, show_col_types = FALSE)
readr::write_csv(final_metrics, file.path(out_dir, paste0("metrics_ALL_", run_id, ".csv")))

cat("\n Merged metrics saved to:\n", file.path(out_dir, paste0("metrics_ALL_", run_id, ".csv")), "\n")
cat(" Timing saved to:\n", timing_path, "\n")

final_results_file <- file.path(out_dir, paste0("metrics_ALL_", run_id, ".csv"))


cat("Results saved to:", final_results_file, "\n")


# ===============================================================
# 8. ANALYZE RESULTS
# ===============================================================

# Read and summarize results
if (file.exists(final_results_file)) {
  final_results <- read_csv(final_results_file, show_col_types = FALSE)
  
  cat("\nTotal configurations tested:", nrow(final_results), "\n")
  
  # Best by link function
  best_by_link <- final_results %>%
    filter(success_rate == 1) %>%
    group_by(recipe) %>%
    slice_min(rmse, n = 1) %>%
    arrange(rmse) %>%
    select(link, recipe, rmse, mae, rsq, accuracy, f1)
  
  cat("\n=== BEST CONFIGURATION BY LINK FUNCTION ===\n")
  print(best_by_link, n = Inf)
  
  # Overall best
  # Select the single best configuration from the initial search using RMSE
  # first, followed by MAE and R-squared.
  nn_best_overall <- final_results %>%
    filter(success_rate == 1) %>%
    arrange(rmse, mae, desc(rsq)) %>%
    slice(1)
  
  # Second best
  secondbest_overall <- final_results %>%
    filter(success_rate == 1) %>%
    arrange(rmse) %>%
    dplyr::slice(2)
  
  cat("\n=== OVERALL BEST CONFIGURATION ===\n")
  print(nn_best_overall)
  
  cat("\n=== SECOND BEST CONFIGURATION ===\n")
  print(secondbest_overall)
}

View(nn_best_overall)

# Ensure connections are properly closed
on.exit({
  plan(sequential)
  closeAllConnections()
}, add = TRUE)

# ===============================================
# 10) Results summary and saving
# ===============================================
cat("\n=== RESULTS SUMMARY ===\n")
cat("Total configurations:", nrow(final_results), "\n")

successful_configs <- sum(final_results$n_successful_folds > 0, na.rm = TRUE)
cat("Successful configurations:", successful_configs, "\n")


# ===============================================================
# 12) ANALYZE RESULTS (best per recipe + single best)
# ===============================================================
nnbest_per_recipe <- final_results %>%
  filter(!is.na(rmse), n_successful_folds >= 40) %>%
  group_by(recipe) %>%
  arrange(rmse) %>%
  dplyr::slice(1) %>%
  ungroup()

write_csv(nnbest_per_recipe, paste0(br_dir, "/", "br_crossval", run_id, ".csv"))
View(nnbest_per_recipe)

# Identify the best configurations per each recipe
best_config_normalize <- final_results %>%
  filter(recipe == "all_pred_tfmwithlogcubenormalize", success_rate == 1) %>%  # Only consider fully successful models
  arrange(rmse, desc(rsq)) %>%              # Sort by RMSE (lower is better)
  dplyr::slice(1)                       # Take the top one


best_config_range <- final_results %>%
  filter(recipe == "all_pred_tfmwithlogcuberange", success_rate == 1) %>%  # Only consider fully successful models
  arrange(rmse, desc(rsq)) %>%              # Sort by RMSE (lower is better)
  dplyr::slice(1)                       # Take the top one


# ---- Running Model on the Test Dataset ----
metrics_file <- file.path(br_dir, paste0("final_nn_test_metrics_", run_id, ".csv"))
preds_file   <- file.path(br_dir, paste0("final_nn_test_predictions_", run_id, ".csv"))

# remove if exists
if (file.exists(metrics_file)) file.remove(metrics_file)
if (file.exists(preds_file)) file.remove(preds_file)

# ---- List of configs to run ----
# Retain the best configuration from each preprocessing recipe for final
# training and evaluation on the independent test dataset.
configs_to_run <- list(
  normalize_best = best_config_normalize,
  range_best     = best_config_range
)


# ---- Helper: compute confusion-matrix based metrics robustly ----
# Calculate poverty-classification measures using the predefined MPI threshold
# with the poor category treated as the positive class.
compute_class_metrics <- function(truth, estimate, threshold = 0.3333) {
  actual <- factor(ifelse(truth    >= threshold, "poor", "non_poor"), 
                   levels = c("poor", "non_poor"))  # poor = positive (level 1)
  pred   <- factor(ifelse(estimate >= threshold, "poor", "non_poor"), 
                   levels = c("poor", "non_poor"))
  
  # table: rows = Actual, cols = Predicted
  # With poor first: [1,1]=TP, [1,2]=FN, [2,1]=FP, [2,2]=TN
  cm <- table(Actual = actual, Predicted = pred)
  
  # Extract using named indexing (robust to level reordering)
  tp <- cm["poor",     "poor"]
  fn <- cm["poor",     "non_poor"]
  fp <- cm["non_poor", "poor"]
  tn <- cm["non_poor", "non_poor"]
  
  accuracy    <- (tp + tn) / max(1, sum(cm))
  sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else 0   # Recall for poor
  specificity <- if ((tn + fp) > 0) tn / (tn + fp) else 0   # Recall for non_poor
  precision   <- if ((tp + fp) > 0) tp / (tp + fp) else 0   # PPV for poor
  f1 <- if ((precision + sensitivity) > 0) 
    2 * (precision * sensitivity) / (precision + sensitivity) else 0
  
  list(cm = cm, tp = tp, tn = tn, fp = fp, fn = fn,
       accuracy = accuracy, sensitivity = sensitivity,
       specificity = specificity, precision = precision, f1 = f1)
}

# ---- Main loop ----
for (name in names(configs_to_run)) {
  
  cfg_row <- configs_to_run[[name]]
  
  cat("\n====================================================\n")
  cat("FINAL TRAINING FOR:", name, "\n")
  print(cfg_row)
  cat("====================================================\n")
  
  # FIX: Set seed for final model training
  recipe_name <- cfg_row$recipe[[1]]
  final_seed_stage1 <- seed_for_nn(recipe_name, 888L, 0L)  # Unique seed for stage 1 finals
  set.seed(final_seed_stage1)
  try(tensorflow::tf$random$set_seed(as.integer(final_seed_stage1)), silent = TRUE)
  
  keras3::clear_session()
  gc(verbose = FALSE)
  
  # Extract config values
  link_name   <- cfg_row$link[[1]]
  
  rec <- preproc.neuralnet[[recipe_name]]
  if (is.null(rec)) stop("Recipe not found in preproc.neuralnet: ", recipe_name)
  
  # Prep/bake
  prep_rec <- prep(rec, training = ntl_train)
  train_processed <- bake(prep_rec, new_data = ntl_train)
  test_processed  <- bake(prep_rec, new_data = ntl_test)
  
  outcome <- "mpi"
  predictors <- setdiff(names(train_processed), outcome)
  
  Xtrain <- as.matrix(train_processed[, predictors])
  ytrain <- train_processed[[outcome]]
  
  Xtest <- as.matrix(test_processed[, predictors])
  ytest <- if ("mpi" %in% names(test_processed)) test_processed[[outcome]] else NULL
  
  # Transform training target
  ytrain <- cliptarg(ytrain)
  ytrain_trans <- apply_link(ytrain, link = link_name)
  
  if (anyNA(Xtrain) || anyNA(ytrain_trans) || any(is.infinite(ytrain_trans))) {
    warning("Invalid transformed ytrain for model: ", name, " (skipping)")
    next
  }
  
  # Build final model
  final_nn <- build_nn_transform_fixed(
    input_shape = ncol(Xtrain),
    units1 = as.integer(cfg_row$units1[[1]]),
    units2 = as.integer(cfg_row$units2[[1]]),
    dropout = as.numeric(cfg_row$dropout[[1]]),
    penalty = as.numeric(cfg_row$penalty[[1]]),
    learn_rate = as.numeric(cfg_row$learn_rate[[1]]),
    activation1 = cfg_row$activation1[[1]],
    activation2 = cfg_row$activation2[[1]],
    optimizer = cfg_row$optimizer[[1]],
    batch_norm = as.logical(cfg_row$batch_norm[[1]]),
    add_early_stopping = TRUE
  )
  
  callbacks <- attr(final_nn, "callbacks")
  
  # Train (now has seed!)
  cat("Training with seed:", final_seed_stage1, "\n")
  t0 <- Sys.time()
  history <- final_nn |> fit(
    Xtrain, ytrain_trans,
    epochs = as.integer(cfg_row$epoch[[1]]),
    batch_size = as.integer(cfg_row$bsize[[1]]),
    validation_split = 0.2,
    callbacks = callbacks,
    shuffle = FALSE,
    verbose = 0
  )
  t1 <- Sys.time()
  train_secs <- as.numeric(difftime(t1, t0, units = "secs"))
  
  # ---- Predict test ----
  preds_test_trans <- as.numeric(predict(final_nn, Xtest, verbose = 0))
  #pred_link <- preds_test_trans * sd_y + mu_y
  preds_test <- apply_inverse_link(preds_test_trans, link = link_name)
  preds_test <- cliptarg(preds_test)
  #preds_test <- pmax(pmin(preds_test, 0.999999), 0.000001)
  
  # ---- Save per-observation predictions (append) ----
  pred_tbl <- tibble(
    run_id = run_id,
    model_tag = name,
    recipe = recipe_name,
    link = link_name,
    row_id = seq_along(preds_test),
    .pred = preds_test
  )
  
  if (!is.null(ytest)) pred_tbl$mpi <- ytest
  
  if (!file.exists(preds_file)) write_csv(pred_tbl, preds_file)
  else write_csv(pred_tbl, preds_file, append = TRUE)
  
  # ---- Compute single-row metrics ----
  if (!is.null(ytest)) {
    rmse <- rmse_vec(ytest, preds_test)
    mae  <- mae_vec(ytest, preds_test)
    rsq  <- rsq_safe(ytest, preds_test)
    
    cm_metrics <- compute_class_metrics(ytest, preds_test, threshold = 0.3333)
    
    metric_row <- tibble(
      run_id = run_id,
      model_tag = name,
      recipe = recipe_name,
      link = link_name,
      rmse = rmse,
      mae  = mae,
      rsq  = rsq,
      accuracy = cm_metrics$accuracy,
      sensitivity = cm_metrics$sensitivity,
      specificity = cm_metrics$specificity,
      f1 = cm_metrics$f1,
      tp = cm_metrics$tp, tn = cm_metrics$tn, fp = cm_metrics$fp, fn = cm_metrics$fn,
      threshold = 0.3333,
      n_test = length(preds_test),
      train_time_secs = train_secs
    )
    
    cat("\n--- TEST METRICS:", name, "---\n")
    print(metric_row)
    cat("\nConfusion matrix:\n")
    print(cm_metrics$cm)
    
    if (!file.exists(metrics_file)) write_csv(metric_row, metrics_file)
    else write_csv(metric_row, metrics_file, append = TRUE)
    
    
  } else {
    warning("Test set has no mpi column; saved predictions only for model: ", name)
  }
  
  # Optional: save model + recipe
  saveRDS(final_nn, file.path(model_dir, "neuralnetbetareg", paste0("final_nn_", name, "_", run_id, ".rds")))
  saveRDS(prep_rec, file.path(model_dir, "neuralnetbetareg", paste0("final_recipe_", name, "_", run_id, ".rds")))
}


cat("\nDONE.\nSaved metrics to:\n", metrics_file, "\nSaved predictions to:\n", preds_file, "\n")

####################################################################
# Begin the second-stage search by refining the hyperparameters around the
# single best configuration identified from the initial search.
################New Refined Subsection=====#########################
####################################################################


# ---------------------------------------------------------------
# 0) SAFETY CHECKS (fail fast, reproducible)
# ---------------------------------------------------------------
stopifnot(
  exists("seed_for_nn"),
  exists("compute_class_metrics"),
  exists("build_nn_transform_fixed"),
  exists("write_timing"),
  exists("grid_is_done"),
  exists("atomic_write_csv"),
  exists("se_of"),
  exists("apply_link"),
  exists("apply_inverse_link"),
  exists("cliptarg")
)

# helper to clamp numeric values into [lo, hi]
clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))

# ---------------------------------------------------------------
# 1) OUTPUT FOLDERS (refined uses its own unique tree)
# ---------------------------------------------------------------
# run_id      <- "20260109_202310"
out_dir     <- file.path(model_dir, "neuralnetbetareg", paste0("run_", run_id), "refined")

metrics_dir       <- file.path(out_dir, "metrics")
preds_dir         <- file.path(out_dir, "preds")
errs_dir          <- file.path(out_dir, "errors_by_grid")
timing_dir        <- file.path(out_dir, "timing_by_grid")
best_results_dir  <- file.path(out_dir, "Best_Results")
images_dir        <- file.path(out_dir, "images")     # used later when saving model/recipe
dir.create(out_dir,        recursive = TRUE, showWarnings = FALSE)
dir.create(metrics_dir,    recursive = TRUE, showWarnings = FALSE)
dir.create(preds_dir,      recursive = TRUE, showWarnings = FALSE)
dir.create(errs_dir,       recursive = TRUE, showWarnings = FALSE)
dir.create(timing_dir,     recursive = TRUE, showWarnings = FALSE)
dir.create(best_results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(images_dir,     recursive = TRUE, showWarnings = FALSE)

timing_path <- file.path(timing_dir, paste0("timing_", run_id, ".csv"))

# Progress log (cat() from multisession often doesn't print)
refined_log <- file.path(out_dir, "refined_progress.log")
log_line <- function(...) {
  txt <- paste0(format(Sys.time(), "%F %T"), " | ", paste0(..., collapse=""), "\n")
  cat(txt, file = refined_log, append = TRUE)
}

log_line("=== Refined run start ===")
log_line("out_dir = ", out_dir)
log_line("timing_path = ", timing_path)

# ---------------------------------------------------------------
# 2) TARGET RECIPE (best_overall must exist from stage-1 results)
# ---------------------------------------------------------------
stopifnot(exists("nn_best_overall"), nrow(nn_best_overall) == 1)

target_recipe <- as.character(nn_best_overall$recipe[[1]])
if (!target_recipe %in% names(preproc.neuralnet)) {
  stop("best_overall$recipe ('", target_recipe, "') not found in preproc.neuralnet.")
}

recipe_cache_dir <- file.path(cache_dir, target_recipe)
if (!dir.exists(recipe_cache_dir)) stop("Cache directory not found: ", recipe_cache_dir)

log_line("target_recipe = ", target_recipe)
log_line("recipe_cache_dir = ", recipe_cache_dir)

# ---------------------------------------------------------------
# 3) UNIQUE METRICS PATH FOR REFINED (prevents skipping due to stage-1 files)
# ---------------------------------------------------------------
grid_metrics_path_refined <- function(metrics_dir, recipe, g) {
  file.path(metrics_dir, paste0("metrics_refined__", recipe, "__grid_", g, ".csv"))
}

# ---------------------------------------------------------------
# 4) BUILD REFINED GRID (your +/- around best values; same logic)
# ---------------------------------------------------------------
cat("\n=== STEP 2: BUILDING REFINED GRID ===\n")

refine_units_step   <- 16L
refine_dropout_step <- 0.05
refine_lr_steps     <- c(-0.25, 0, 0.25)  # log10 steps
refine_pen_steps    <- c(-0.5, 0, 0.5)    # log10 steps

best_units1     <- as.integer(nn_best_overall$units1)
best_units2     <- as.integer(nn_best_overall$units2)
best_dropout    <- as.numeric(nn_best_overall$dropout)
best_lr_log     <- log10(as.numeric(nn_best_overall$learn_rate))
best_pen_log    <- log10(as.numeric(nn_best_overall$penalty))
best_link       <- as.character(nn_best_overall$link)
best_optimizer  <- as.character(nn_best_overall$optimizer)
best_activation1<- as.character(nn_best_overall$activation1)
best_activation2<- as.character(nn_best_overall$activation2)
best_epoch      <- as.integer(nn_best_overall$epoch)
best_batch_norm <- as.logical(nn_best_overall$batch_norm)
best_bsize      <- as.numeric(nn_best_overall$bsize)
units1_vals   <- unique(clamp(best_units1 + (-refine_units_step:refine_units_step), 50L, 300L))
units2_vals   <- unique(clamp(best_units2 + (-refine_units_step:refine_units_step), 50L, 300L))
dropout_vals  <- unique(clamp(best_dropout + c(-refine_dropout_step, 0, refine_dropout_step), 0.1, 0.5))
lr_log_vals   <- unique(clamp(best_lr_log + refine_lr_steps, -4, -1))
pen_log_vals  <- unique(clamp(best_pen_log + refine_pen_steps, -6, -2))
bsize_vals   <- sort(unique(pmax(1L, as.integer(best_bsize +c(-3, 0, 3)))))

set.seed(1234)
# Construct the refined grid around the best initial configuration while
# retaining the selected link, optimiser, activations and other fixed settings.
param_grid_refined <- tidyr::expand_grid(
  units1     = units1_vals,
  units2     = units2_vals,
  dropout    = dropout_vals,
  learn_rate = 10^lr_log_vals,
  penalty    = 10^pen_log_vals,
  bsize      = bsize_vals
) %>%
  dplyr::mutate(
    link        = best_link,
    optimizer   = best_optimizer,
    activation1 = best_activation1,
    activation2 = best_activation2,
    epoch       = best_epoch,
    batch_norm  = best_batch_norm,
    units1      = as.integer(units1),
    units2      = as.integer(units2),
    bsize       = as.integer(bsize),
    epoch       = as.integer(epoch),
    batch_norm  = as.logical(batch_norm)
  ) %>%
  dplyr::distinct()

set.seed(1234)
if (nrow(param_grid_refined) > 70) {
  param_grid_refined <- dplyr::slice_sample(param_grid_refined, n = 70)
} else if (nrow(param_grid_refined) < 70) {
  warning("Refined grid has only ", nrow(param_grid_refined), " configs (target: 70)")
}

if (nrow(param_grid_refined) == 0) stop("param_grid_refined has 0 rows — nothing to run.")

cat("Refined grid size:", nrow(param_grid_refined), "\n")
print(head(param_grid_refined, 3))
log_line("param_grid_refined n = ", nrow(param_grid_refined))

# ===============================================================
# 5) OPTIMIZED CV REFINEMENT LOOP (Multi-worker with chunking)
# ===============================================================
cat("\n=== STEP 3: CROSS-VALIDATION REFINEMENT (OPTIMIZED) ===\n")

recipe_start <- Sys.time()

# Configuration: Adjust based on your system
n_workers <- min(parallel::detectCores() - 1, 4)  # Use up to 4 workers
grids_per_chunk <- 3  # Process 3 grids per worker before cleanup

cat("Starting refined search with", n_workers, "workers\n")
cat("Processing", grids_per_chunk, "grids per chunk\n")
log_line("n_workers = ", n_workers, " | grids_per_chunk = ", grids_per_chunk)

# Restart workers for refined search
future::plan(sequential)
hard_cleanup(py_gc = TRUE)
future::plan(multisession, workers = n_workers)

grid_ids <- seq_len(nrow(param_grid_refined))

# Split grids into chunks for batch processing
grid_chunks <- split(grid_ids, ceiling(seq_along(grid_ids) / grids_per_chunk))

cat("Total grids:", length(grid_ids), "| Chunks:", length(grid_chunks), "\n")
log_line("Total grids = ", length(grid_ids), " | Chunks = ", length(grid_chunks))

furrr::future_walk(
  grid_chunks,
  function(chunk_grids) {
    
    suppressPackageStartupMessages({
      library(dplyr); library(tibble); library(readr); library(purrr)
      library(yardstick); library(reticulate); library(keras3); library(tensorflow)
    })
    
    reticulate::use_virtualenv("r-keras3", required = TRUE)
    
    # Enable optimizations
    Sys.setenv(CUDA_VISIBLE_DEVICES = "-1", TF_CPP_MIN_LOG_LEVEL = "2")
    try(tensorflow::tf$config$experimental$enable_op_determinism(), silent = TRUE)
    try(tensorflow::tf$config$threading$set_intra_op_parallelism_threads(1L), silent = TRUE)
    try(tensorflow::tf$config$threading$set_inter_op_parallelism_threads(1L), silent = TRUE)
    try(tensorflow::tf$config$optimizer$set_jit(TRUE), silent = TRUE)  # XLA compilation
    
    # Process each grid in this chunk
    for (g in chunk_grids) {
      
      grid_start <- Sys.time()
      
      # Deterministic seed for this grid
      grid_seed <- seed_for_nn(target_recipe, g, 0L)
      set.seed(grid_seed)
      try(tensorflow::tf$random$set_seed(as.integer(grid_seed)), silent = TRUE)
      
      cfg <- as.list(param_grid_refined[g, ])
      
      # IMPORTANT: refined-specific metrics path (prevents skip collisions)
      metrics_path <- grid_metrics_path_refined(metrics_dir, target_recipe, g)
      preds_path   <- file.path(preds_dir, paste0("fold_preds_refined__", target_recipe, "__grid_", g, ".csv"))
      err_path     <- file.path(errs_dir,  paste0("errors_refined__", target_recipe, "__grid_", g, ".csv"))
      
      # =========================================================
      # RESUME LOGIC: Skip if already done
      # =========================================================
      if (grid_is_done(metrics_path)) {
        write_timing("grid", target_recipe, g, grid_start, Sys.time(),
                     list(status = "skipped_exists_valid"))
        log_line("[Grid ", g, "/", nrow(param_grid_refined), "] SKIPPED (already done)")
        next  # Skip to next grid in chunk
      } else {
        # Clean partial files from previous crashes
        if (file.exists(metrics_path)) file.remove(metrics_path)
        if (file.exists(paste0(metrics_path, ".tmp"))) file.remove(paste0(metrics_path, ".tmp"))
      }
      
      # Log progress to file (not console)
      log_line("[Grid ", g, "/", nrow(param_grid_refined), "] START | units1=", cfg$units1,
               " units2=", cfg$units2, " dropout=", sprintf("%.2f", cfg$dropout),
               " lr=", format(cfg$learn_rate, scientific = TRUE),
               " pen=", format(cfg$penalty, scientific = TRUE))
      
      fold_metrics <- purrr::map_dfr(seq_along(ntl_vfolds_cv$splits), function(i) {
        
        tryCatch({
          
          fold_seed <- seed_for_nn(target_recipe, g, i)
          set.seed(fold_seed)
          try(tensorflow::tf$random$set_seed(as.integer(fold_seed)), silent = TRUE)
          
          dat <- readRDS(file.path(recipe_cache_dir, paste0("fold_", i, ".rds")))
          
          Xtrain <- dat$Xtrain
          Xval   <- dat$Xval
          ytrain <- dat$ytrain
          yval   <- dat$yval
          
          ytrain <- cliptarg(ytrain)
          yval   <- cliptarg(yval)
          
          ytrain_link <- apply_link(ytrain, link = cfg$link)
          yval_link   <- apply_link(yval,   link = cfg$link)
          
          keras3::clear_session()
          model <- build_nn_transform_fixed(
            input_shape = ncol(Xtrain),
            units1      = cfg$units1,
            units2      = cfg$units2,
            dropout     = cfg$dropout,
            penalty     = cfg$penalty,
            learn_rate  = cfg$learn_rate,
            activation1 = cfg$activation1,
            activation2 = cfg$activation2,
            optimizer   = cfg$optimizer,
            batch_norm  = cfg$batch_norm,
            add_early_stopping = TRUE
          )
          
          callbacks <- attr(model, "callbacks")
          
          model %>% fit(
            Xtrain, ytrain_link,
            epochs = as.integer(cfg$epoch),
            batch_size = as.integer(cfg$bsize),
            validation_data = list(Xval, yval_link),
            callbacks = callbacks,
            shuffle = FALSE,
            verbose = 0
          )
          
          pred_link <- as.numeric(predict(model, Xval, verbose = 0))
          pred_prob <- cliptarg(apply_inverse_link(pred_link, link = cfg$link))
          
          if (any(!is.finite(pred_prob))) {
            log_fail(err_path, target_recipe, g, i, "predict", cfg, "Non-finite predictions")
            rm(model, dat)
            hard_cleanup(py_gc = FALSE)
            return(tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_,
                          accuracy=NA_real_, sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_))
          }
          
          fold_pred_tbl <- tibble(
            run_id   = run_id,
            recipe   = target_recipe,
            grid_id  = g,
            fold     = i,
            fold_seed = fold_seed,
            link     = cfg$link,
            row_id   = seq_along(yval),
            mpi_true = yval,
            mpi_pred = pred_prob
          )
          readr::write_csv(fold_pred_tbl, preds_path, append = file.exists(preds_path))
          
          rmse <- yardstick::rmse_vec(yval, pred_prob)
          mae  <- yardstick::mae_vec(yval, pred_prob)
          rsq  <- rsq_safe(yval, pred_prob)
          
          cls <- compute_class_metrics(yval, pred_prob, threshold = 0.3333)
          
          rm(model, dat)
          # Less frequent Python GC (only every 20 folds instead of 10)
          hard_cleanup(py_gc = (i %% 20 == 0))
          
          tibble(rmse=rmse, mae=mae, rsq=rsq,
                 accuracy=cls$accuracy, sensitivity=cls$sensitivity,
                 specificity=cls$specificity, f1=cls$f1)
          
        }, error = function(e) {
          log_fail(err_path, target_recipe, g, i, "R_error", cfg, conditionMessage(e))
          hard_cleanup(py_gc = FALSE)
          tibble(rmse=NA_real_, mae=NA_real_, rsq=NA_real_,
                 accuracy=NA_real_, sensitivity=NA_real_, specificity=NA_real_, f1=NA_real_)
        })
      })
      
      n_ok <- sum(!is.na(fold_metrics$rmse))
      
      grid_row <- tibble(
        run_id   = run_id,
        recipe   = target_recipe,
        grid_id  = g,
        grid_seed = grid_seed,
        link     = cfg$link,
        units1   = cfg$units1,
        units2   = cfg$units2,
        dropout  = cfg$dropout,
        learn_rate = cfg$learn_rate,
        penalty  = cfg$penalty,
        optimizer = cfg$optimizer,
        activation1 = cfg$activation1,
        activation2 = cfg$activation2,
        epoch    = cfg$epoch,
        bsize    = cfg$bsize,
        batch_norm = cfg$batch_norm,
        
        rmse = mean(fold_metrics$rmse, na.rm = TRUE),
        se_rmse = se_of(fold_metrics$rmse),
        mae = mean(fold_metrics$mae, na.rm = TRUE),
        se_mae = se_of(fold_metrics$mae),
        rsq = mean(fold_metrics$rsq, na.rm = TRUE),
        se_rsq = se_of(fold_metrics$rsq),
        
        accuracy = mean(fold_metrics$accuracy, na.rm = TRUE),
        se_accuracy = se_of(fold_metrics$accuracy),
        sensitivity = mean(fold_metrics$sensitivity, na.rm = TRUE),
        se_sensitivity = se_of(fold_metrics$sensitivity),
        specificity = mean(fold_metrics$specificity, na.rm = TRUE),
        se_specificity = se_of(fold_metrics$specificity),
        f1 = mean(fold_metrics$f1, na.rm = TRUE),
        se_f1 = se_of(fold_metrics$f1),
        
        n_successful_folds = n_ok,
        total_folds = nrow(fold_metrics),
        success_rate = n_ok / nrow(fold_metrics)
      )
      
      # =========================================================
      # ATOMIC METRICS WRITE (prevents "exists but corrupt")
      # =========================================================
      atomic_write_csv(grid_row, metrics_path)
      
      grid_end <- Sys.time()
      write_timing("grid", target_recipe, g, grid_start, grid_end,
                   list(status = "done", n_successful_folds = n_ok))
      
      log_line("[Grid ", g, "/", nrow(param_grid_refined), "] DONE | ok_folds=", n_ok, 
               "/", nrow(fold_metrics), " | rmse=", sprintf("%.6f", grid_row$rmse),
               " | time=", sprintf("%.1f", as.numeric(difftime(grid_end, grid_start, units = "mins"))), "min")
      
      #  Light cleanup after each grid (no Python GC)
      hard_cleanup(py_gc = FALSE)
      
    } # End of grid loop within chunk
    
    #  Heavy cleanup after processing entire chunk
    hard_cleanup(py_gc = TRUE)
    invisible(NULL)
    
  },
  .options = furrr::furrr_options(seed = 1234)
)

# Cleanup after refinement in main session
future::plan(sequential)
hard_cleanup(py_gc = TRUE)

recipe_end <- Sys.time()
write_timing("recipe", target_recipe, NA_integer_, recipe_start, recipe_end,
             list(status = "done"))

elapsed_time <- as.numeric(difftime(recipe_end, recipe_start, units = "mins"))
cat("\n CV refinement complete in", sprintf("%.1f", elapsed_time), "minutes\n")
log_line("=== CV refinement complete | Duration: ", sprintf("%.1f", elapsed_time), " minutes ===")

# ===============================================================
# 6) MERGE AND ANALYZE RESULTS (uses refined metrics filenames)
# ===============================================================
cat("\n=== STEP 4: ANALYZING RESULTS ===\n")

refined_metrics_files <- list.files(metrics_dir, full.names = TRUE,
                                    pattern = "^metrics_refined__.*__grid_\\d+\\.csv$")

if (length(refined_metrics_files) == 0) {
  stop("No refined metrics files found in: ", metrics_dir)
}

rfinal_results <- purrr::map_dfr(refined_metrics_files, readr::read_csv, show_col_types = FALSE)

rfinal_results_file <- file.path(best_results_dir, paste0("metrics_refined_", run_id, ".csv"))
readr::write_csv(rfinal_results, rfinal_results_file)

cat("Merged metrics saved to:", rfinal_results_file, "\n")
cat("Total configurations:", nrow(rfinal_results), "\n")
cat("Successful configurations:", sum(rfinal_results$success_rate == 1), "\n\n")

# Best configuration
rbest_config <- rfinal_results %>%
  dplyr::filter(success_rate == 1) %>%
  dplyr::arrange(rmse, mae, desc(rsq)) %>%
  dplyr::slice(1)

cat("=== BEST REFINED CONFIGURATION ===\n")
print(rbest_config %>% dplyr::select(recipe, rmse, mae, rsq, accuracy, f1,
                                     units1, units2, dropout, learn_rate, penalty))

readr::write_csv(rbest_config, file.path(best_results_dir, paste0("best_refined_config_", run_id, ".csv")))

log_line("Best refined rmse = ", ifelse(nrow(rbest_config)==1, rbest_config$rmse, NA))
log_line("refined_metrics_files n = ", length(refined_metrics_files))
log_line("Output directory = ", out_dir)
log_line("Progress log = ", refined_log)

cat("\n REFINEMENT COMPLETE\n")
cat("Output directory:\n  ", out_dir, "\n")
cat("Progress log:\n  ", refined_log, "\n")
cat("Best config saved to:\n  ", file.path(best_results_dir, paste0("best_refined_config_", run_id, ".csv")), "\n")

future::plan(sequential)
hard_cleanup(py_gc = TRUE)

# ===============================================================
# 7) FINAL MODEL TRAINING ON FULL DATA
# ===============================================================
cat("\n=== STEP 5: TRAINING FINAL MODEL ON FULL DATA ===\n")

metrics_file <- file.path(best_results_dir, paste0("refined_final_nn_test_metric_", run_id, ".csv"))
preds_file <- file.path(best_results_dir, paste0("refined_final_nn_test_predictions_", run_id, ".csv"))

#if (file.exists(metrics_file)) file.remove(metrics_file)
#if (file.exists(preds_file)) file.remove(preds_file)

# Refit the best refined configuration using the complete training dataset
# before evaluating its performance on the independent test dataset.
rrecipe_name <- as.character(rbest_config$recipe[[1]])
rlink_name <- as.character(rbest_config$link[[1]])

# Set seed for final model
final_seed <- seed_for_nn(rrecipe_name, 999L, 0L)
set.seed(final_seed)
try(tensorflow::tf$random$set_seed(as.integer(final_seed)), silent = TRUE)

keras3::clear_session()
gc(verbose = FALSE)

cat("Recipe:", rrecipe_name, "\n")
cat("Link:", rlink_name, "\n")
cat("Final model seed:", final_seed, "\n")

# Get recipe
rrec <- preproc.neuralnet[[rrecipe_name]]
if (is.null(rrec)) stop("Recipe not found: ", rrecipe_name)

# Prep and bake
rprep_rec <- prep(rrec, training = ntl_train)
rtrain_processed <- bake(rprep_rec, new_data = ntl_train)
rtest_processed <- bake(rprep_rec, new_data = ntl_test)

outcome <- "mpi"
predictors <- setdiff(names(rtrain_processed), outcome)

rXtrain <- as.matrix(rtrain_processed[, predictors])
rytrain <- rtrain_processed[[outcome]]

rXtest <- as.matrix(rtest_processed[, predictors])
rytest <- rtest_processed[[outcome]]

# Keep raw test MPI
rytest_raw <- rytest

# Clip and transform training
rytrain <- cliptarg(rytrain)
rytrain_trans <- apply_link(rytrain, link = rlink_name)

if (anyNA(rXtrain) || anyNA(rytrain_trans)) {
  stop("Invalid data after transformation")
}

# Build final model
rfinal_nn <- build_nn_transform_fixed(
  input_shape = ncol(rXtrain),
  units1 = as.integer(rbest_config$units1),
  units2 = as.integer(rbest_config$units2),
  dropout = as.numeric(rbest_config$dropout),
  penalty = as.numeric(rbest_config$penalty),
  learn_rate = as.numeric(rbest_config$learn_rate),
  activation1 = as.character(rbest_config$activation1),
  activation2 = as.character(rbest_config$activation2),
  optimizer = as.character(rbest_config$optimizer),
  batch_norm = as.logical(rbest_config$batch_norm),
  add_early_stopping = TRUE
)

callbacks <- attr(rfinal_nn, "callbacks")

# Train
cat("Training final model...\n")
t0 <- Sys.time()
history <- rfinal_nn %>% fit(
  rXtrain, rytrain_trans,
  epochs = as.integer(rbest_config$epoch),
  batch_size = as.integer(rbest_config$bsize),
  validation_split = 0.2,
  callbacks = callbacks,
  verbose = 1
)
t1 <- Sys.time()
rtrain_secs <- as.numeric(difftime(t1, t0, units = "secs"))

cat("Training time:", round(rtrain_secs, 2), "seconds\n")

# Predict on test
rpreds_test_trans <- as.numeric(predict(rfinal_nn, rXtest, verbose = 0))
rpreds_test <- apply_inverse_link(rpreds_test_trans, link = rlink_name)
rpreds_test <- cliptarg(rpreds_test)

# Save predictions (compare to RAW test MPI)
rpred_tbl <- tibble(
  run_id = run_id,
  recipe = rrecipe_name,
  link = rlink_name,
  final_seed = final_seed,
  row_id = seq_along(rpreds_test),
  mpi_true = rytest_raw,  # Use raw, not clipped
  mpi_pred = rpreds_test
)
write_csv(rpred_tbl, preds_file)

# Calculate metrics (against RAW test MPI)
rrmse <- rmse_vec(rytest_raw, rpreds_test)
rmae <- mae_vec(rytest_raw, rpreds_test)
rrsq <- rsq_safe(rytest_raw, rpreds_test)

rcm_metrics <- compute_class_metrics(rytest_raw, rpreds_test, threshold = 0.3333)

rmetric_row <- tibble(
  run_id = run_id,
  recipe = rrecipe_name,
  link = rlink_name,
  final_seed = final_seed,
  cv_rmse = rbest_config$rmse,
  cv_rsq = rbest_config$rsq,
  test_rmse = rrmse,
  test_mae = rmae,
  test_rsq = rrsq,
  test_accuracy = rcm_metrics$accuracy,
  test_sensitivity = rcm_metrics$sensitivity,
  test_specificity = rcm_metrics$specificity,
  test_f1 = rcm_metrics$f1,
  tp = rcm_metrics$tp,
  tn = rcm_metrics$tn,
  fp = rcm_metrics$fp,
  fn = rcm_metrics$fn,
  n_test = length(rpreds_test),
  train_time_secs = rtrain_secs
)

write_csv(rmetric_row, metrics_file)

cat("\n=== FINAL TEST METRICS ===\n")
print(rmetric_row %>% select(test_rmse, test_mae, test_rsq, test_accuracy, test_f1))
cat("\nConfusion Matrix:\n")
print(rcm_metrics$cm)

# Save model
saveRDS(rfinal_nn, file.path(images_dir, paste0("refined_final_nn_model_", run_id, ".rds")))
saveRDS(rprep_rec, file.path(images_dir, paste0("refined_final_recipe_", run_id, ".rds")))

cat("\nFinal model saved\n")
cat("Predictions:", preds_file, "\n")
cat("Metrics:", metrics_file, "\n")

## Read model from file if requried: 
#final_nn <- readRDS(file.path(images_dir, paste0("refined_final_nn_model_", run_id, ".rds")))

# ===============================================================
# 8) SESSION INFO FOR REPRODUCIBILITY
# ===============================================================
session_file <- file.path(out_dir, paste0("sessionInfo_", run_id, ".txt"))
sink(session_file)
print(sessionInfo())
sink() # close the connection

###########End of Refined section#################################

# Compare the best initial-stage and refined-stage neural network configurations
# using cross-validation results before selecting the configuration for the final fit.
###########New section comparing the 2 phase#################
# If refined best object isn't created yet, derive it (same as your refined script)
if (!exists("rbest_config")) {
  stopifnot(exists("rfinal_results"))
  rbest_config <- rfinal_results %>%
    filter(success_rate == 1) %>%
    slice_min(rmse, n = 1)
}

# ---------------------------------------------------------------
# 1) NORMALISE BOTH "BEST" ROWS INTO A COMMON SHAPE
# (adds `source` label + keeps same parameter columns)
# ---------------------------------------------------------------
pick_cols <- c(
  "recipe","link","units1","units2","dropout","learn_rate","penalty",
  "optimizer","activation1","activation2","epoch","bsize","batch_norm",
  "rmse","mae","rsq","accuracy","sensitivity","specificity","f1",
  "success_rate","n_successful_folds","total_folds"
)

sf_best <- nn_best_overall %>%
  mutate(source = "space_filling") %>%
  select(any_of(c("source", pick_cols)))

refined_best <- rbest_config %>%
  mutate(source = "refined_grid") %>%
  select(any_of(c("source", pick_cols)))

# Ensure numeric types line up (important for reliable comparison)
coerce_numeric <- function(x) suppressWarnings(as.numeric(x))
coerce_int     <- function(x) suppressWarnings(as.integer(x))
coerce_logical <- function(x) {
  if (is.logical(x)) return(x)
  if (is.numeric(x)) return(as.logical(x))
  if (is.character(x)) return(toupper(x) %in% "TRUE")
  as.logical(x)
}

fix_types <- function(df) {
  df %>%
    mutate(
      units1 = coerce_int(units1),
      units2 = coerce_int(units2),
      epoch  = coerce_int(epoch),
      bsize  = coerce_int(bsize),
      dropout = coerce_numeric(dropout),
      learn_rate = coerce_numeric(learn_rate),
      penalty = coerce_numeric(penalty),
      rmse = coerce_numeric(rmse),
      mae = coerce_numeric(mae),
      rsq = coerce_numeric(rsq),
      accuracy = coerce_numeric(accuracy),
      sensitivity = coerce_numeric(sensitivity),
      specificity = coerce_numeric(specificity),
      f1          = coerce_numeric(f1),
      batch_norm = coerce_logical(batch_norm)
    )
}

sf_best <- fix_types(sf_best)
refined_best <- fix_types(refined_best)


# ---------------------------------------------------------------
# 2) COMPARE INITIAL AND REFINED CV RESULTS AND SELECT WINNER
# ---------------------------------------------------------------

# Retain both configurations in one table for reporting.
comparison_tbl <- bind_rows(
  sf_best,
  refined_best
)

# Compare the best initial-stage and refined-stage Neural Network configurations.
# The common model-selection function uses RMSE first, followed by MAE and
# R-squared when the preceding measures are effectively tied.
comparison_nn <- is_better_model(
  sf_best,
  refined_best
)

# Retain the better Neural Network configuration according to the predefined
# model-selection criteria.
if (comparison_nn$better) {
  
  chosen <- sf_best
  
} else {
  
  chosen <- refined_best
}

View(chosen)

cat("\n================= MODEL SELECTION (NN) =================\n")
print(
  comparison_tbl %>%
    select(
      source, recipe, link, rmse, mae, rsq, f1,
      units1, units2, dropout, learn_rate, penalty, bsize
    )
)

cat("\nSelected for plots:", chosen$source, "\n")
cat(
  "Recipe:", chosen$recipe,
  " | Link:", chosen$link,
  " | CV RMSE:", chosen$rmse,
  "\n"
)
cat("Selection criterion:", comparison_nn$criterion, "\n")
cat("========================================================\n\n")

# ---------------------------------------------------------------
# 3) TRAIN ONE FINAL MODEL (on FULL training set) USING CHOSEN CONFIG
# AND PREPARE preds.train / preds.test for your plotting code
# ---------------------------------------------------------------

# Use the selected configuration to fit one final neural network on the full
# training dataset and prepare predictions for subsequent analysis and plotting.
chosen_recipe <- as.character(chosen$recipe[[1]])
chosen_link <- as.character(chosen$link[[1]])

rec_obj <- preproc.neuralnet[[chosen_recipe]]
if (is.null(rec_obj)) stop("Recipe not found in preproc.neuralnet: ", chosen_recipe)

# Prep/bake
prep_rec        <- prep(rec_obj, training = ntl_train)
train_processed <- bake(prep_rec, new_data = ntl_train)
test_processed  <- bake(prep_rec, new_data = ntl_test)

outcome <- "mpi"
predictors <- setdiff(names(train_processed), outcome)

Xtrain <- as.matrix(train_processed[, predictors])
ytrain <- train_processed[[outcome]]

Xtest <- as.matrix(test_processed[, predictors])
ytest <- test_processed[[outcome]]

# Keep RAW ytest for evaluation/plots if you want
ytest_raw <- ytest

# Target transform (same as your pipeline)
ytrain_clip <- cliptarg(ytrain)
ytrain_trans <- apply_link(ytrain_clip, link = chosen_link)

# Reproducible seed for the final fit used in plots
final_seed <- seed_for_nn(chosen_recipe, 999L, 0L)
set.seed(final_seed)
try(tensorflow::tf$random$set_seed(as.integer(final_seed)), silent = TRUE)

keras3::clear_session()
gc(FALSE)

final_nn <- build_nn_transform_fixed(
  input_shape = ncol(Xtrain),
  units1 = as.integer(chosen$units1),
  units2 = as.integer(chosen$units2),
  dropout = as.numeric(chosen$dropout),
  penalty = as.numeric(chosen$penalty),
  learn_rate = as.numeric(chosen$learn_rate),
  activation1 = as.character(chosen$activation1),
  activation2 = as.character(chosen$activation2),
  optimizer = as.character(chosen$optimizer),
  batch_norm = as.logical(chosen$batch_norm),
  add_early_stopping = TRUE
)

callbacks <- attr(final_nn, "callbacks")

t0 <- Sys.time()
history <- final_nn %>% fit(
  Xtrain, ytrain_trans,
  epochs = as.integer(chosen$epoch),
  batch_size = as.integer(chosen$bsize),
  validation_split = 0.2,
  callbacks = callbacks,
  shuffle = FALSE,
  verbose = 0
)

t1 <- Sys.time()
train_secs <- as.numeric(difftime(t1, t0, units = "secs"))

# Predictions (train + test) in MPI space
# Generate predictions for both training and independent test datasets and
# return them to the original MPI scale.
pred_train_link <- as.numeric(predict(final_nn, Xtrain, verbose = 0))
pred_train <- cliptarg(apply_inverse_link(pred_train_link, link = chosen_link))

pred_test_link <- as.numeric(predict(final_nn, Xtest, verbose = 0))
pred_test <- cliptarg(apply_inverse_link(pred_test_link, link = chosen_link))

# Tibbles for plots (match your RF/XGB style)
preds.train <- tibble(
  source = chosen$source,
  recipe = chosen$recipe,
  link = chosen$link,
  final_seed = final_seed,
  mpi = ytrain, # raw training truth
  .fitted = pred_train, 
  .resid = ytrain - pred_train, 
  .std_resid = (ytrain - pred_train) / sd(ytrain - pred_train)
)

preds.test <- tibble(
  source = chosen$source,
  recipe = chosen$recipe,
  link = chosen$link,
  final_seed = final_seed,
  mpi = ytest_raw, # raw test truth
  .fitted = pred_test, 
  .resid = ytest_raw - pred_test, 
  .std_resid = (ytest_raw - pred_test) / sd(ytest_raw - pred_test)
)

# Create residual data for Bias Analysis
test_resid_data_nn <- tibble(
  model = "Neural Network",
  .fitted = as.numeric(preds.test$.fitted),
  .resid  = as.numeric(preds.test$.resid),
  mpi     = as.numeric(preds.test$mpi),
  .std_resid = as.numeric(scale(preds.test$.resid)),
  Set = "Test"
)

# Write the test tibble to file
file_name_nn <- "test_resid_data_nn.csv"
write_csv(test_resid_data_nn, file.path(bsttestpre_dir, file_name_nn))

# Optional: compute and print RMSE for the *plot model* on train/test
train_rmse_plot <- rmse_vec(preds.train$mpi, preds.train$.fitted)
test_rmse_plot  <- rmse_vec(preds.test$mpi,  preds.test$.fitted)

cat("Plot-model training time (secs):", round(train_secs, 2), "\n")
cat("Plot-model RMSE (train):", round(train_rmse_plot, 6), "\n")
cat("Plot-model RMSE (test): ", round(test_rmse_plot, 6), "\n")


# Optional recovery section for rebuilding the final model and predictions
# when the original in-memory model objects are no longer available.
#########Optional Step - Load from rds file - 21Feb '26#####################
# Run this when the model was run previous execute and the model run is not available

# ===============================================================
# RETRAIN FINAL MODEL AND GENERATE PREDICTIONS
# Complete script - run this entire block
# ===============================================================
library(keras3)
library(reticulate)
library(bundle)
library(dplyr)
library(tibble)
library(readr)
library(yardstick)
library(recipes)

# Verify Python is configured
cat("Python configured:\n")
reticulate::py_config()

cat("\n=== Retraining Final Model ===\n")

# Paths - adjust if needed
run_id <- "20260125_204855"
base_dir <- "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI/modeldir/neuralnetbetareg"
run_dir <- file.path(base_dir, paste0("run_", run_id))

best_config_path <- file.path(run_dir, "refined/Best_Results", paste0("best_refined_config_", run_id, ".csv"))
recipe_path <- file.path(run_dir, "refined/images", paste0("refined_final_recipe_", run_id, ".rds"))
output_dir <- file.path(run_dir, "refined/images")

# Load best config
best_config <- read_csv(best_config_path, show_col_types = FALSE)
cat("Best config:\n")
print(best_config %>% select(recipe, link, units1, units2, dropout, learn_rate, penalty))

# Load recipe
nn_recipe <- readRDS(recipe_path)
cat("\n Recipe loaded\n")

# Prep and bake
cat("Preprocessing data...\n")
train_processed <- bake(nn_recipe, new_data = ntl_train)
test_processed  <- bake(nn_recipe, new_data = ntl_test)

outcome <- "mpi"
predictors <- setdiff(names(train_processed), outcome)

X_train <- as.matrix(train_processed[, predictors])
y_train <- train_processed[[outcome]]

X_test <- as.matrix(test_processed[, predictors])
y_test <- test_processed[[outcome]]

storage.mode(X_train) <- "double"
storage.mode(X_test) <- "double"

# Transform target
y_train <- cliptarg(y_train)
link_function <- as.character(best_config$link)
y_train_trans <- apply_link(y_train, link = link_function)

cat(" Data preprocessed\n")
cat("   Training samples:", nrow(X_train), "\n")
cat("   Features:        ", ncol(X_train), "\n\n")

# Build model
cat("Building model with best hyperparameters...\n")
final_nn <- build_nn_transform_fixed(
  input_shape = ncol(X_train),
  units1      = as.integer(best_config$units1),
  units2      = as.integer(best_config$units2),
  dropout     = as.numeric(best_config$dropout),
  penalty     = as.numeric(best_config$penalty),
  learn_rate  = as.numeric(best_config$learn_rate),
  activation1 = as.character(best_config$activation1),
  activation2 = as.character(best_config$activation2),
  optimizer   = as.character(best_config$optimizer),
  batch_norm  = as.logical(best_config$batch_norm),
  add_early_stopping = TRUE
)

callbacks <- attr(final_nn, "callbacks")

# Train
cat("\nTraining model...\n")
cat("(This will take 5-10 minutes)\n\n")

t0 <- Sys.time()
history <- final_nn %>% fit(
  X_train, y_train_trans,
  epochs = as.integer(best_config$epoch),
  batch_size = as.integer(best_config$bsize),
  validation_split = 0.2,
  callbacks = callbacks,
  verbose = 1
)
t1 <- Sys.time()

cat("\n Training complete in", round(difftime(t1, t0, units = "mins"), 1), "minutes\n")

# Save model properly
model_keras_path <- file.path(output_dir, paste0("refined_final_nn_model_", run_id, ".keras"))
keras3::save_model(final_nn, model_keras_path)
cat(" Model saved (Keras format):", model_keras_path, "\n")

bundled_model <- bundle::bundle(final_nn)
model_bundle_path <- file.path(output_dir, paste0("refined_final_nn_model_bundled_", run_id, ".rds"))
saveRDS(bundled_model, model_bundle_path)
cat(" Model saved (bundled RDS):", model_bundle_path, "\n")

# Predict
cat("\n=== Generating Predictions ===\n")

pred_transformed <- as.numeric(predict(final_nn, X_test, verbose = 0))
pred_prob <- apply_inverse_link(pred_transformed, link = link_function)
pred_prob <- cliptarg(pred_prob, eps = 1e-6)

# Results
nn_results <- tibble(
  model = "Neural Network",
  mpi_actual = ntl_test$mpi,
  mpi_predicted = pred_prob,
  residual = ntl_test$mpi - pred_prob
)


# Metrics
nn_rmse <- yardstick::rmse_vec(nn_results$mpi_actual, nn_results$mpi_predicted)
nn_mae  <- yardstick::mae_vec(nn_results$mpi_actual, nn_results$mpi_predicted)
nn_rsq  <- suppressWarnings(yardstick::rsq_vec(nn_results$mpi_actual, nn_results$mpi_predicted))

cat("\n=== NEURAL NETWORK TEST METRICS ===\n")
cat("RMSE:", round(nn_rmse, 4), "\n")
cat("MAE: ", round(nn_mae,  4), "\n")
cat("R²:  ", round(nn_rsq,  4), "\n\n")

# Create residual data for plots
test_resid_data_nn <- tibble(
  model = "Neural Network",
  .fitted = as.numeric(nn_results$mpi_predicted),
  .resid  = as.numeric(nn_results$residual),
  mpi     = as.numeric(nn_results$mpi_actual),
  .std_resid = as.numeric(scale(nn_results$residual)),
  Set = "Test"
)

# Write the test tibble to file



write_csv(
  test_resid_data_nn,
  file.path(output_dir, paste0("test_resid_data_nn_", run_id, ".csv"))
)

cat("Residual data object created: test_resid_data_nn\n")
cat("\n ALL DONE!\n")
cat("   Model files saved:\n")
cat("   - ", model_keras_path, "\n")
cat("   - ", model_bundle_path, "\n")

#End of optional section

#######################################################################

# ------------------------------------------------------------
# Diagnostic Plots
# ------------------------------------------------------------

# Residuals vs Fitted
# 1. Residuals vs Fitted
# Plot training residuals against fitted MPI values. The zero reference line
# and LOESS smoother help identify systematic residual patterns across predictions.
p1 <- ggplot(preds.train, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Fitted Values",
    subtitle = "Check Homoscedasticity for Final Keras Model",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals (Actual - Predicted)"
  ) +
  theme_minimal()

# 2. Scale-Location
# Examine whether the spread of the standardised training residuals changes
# across the fitted MPI range.
p2 <- ggplot(preds.train, aes(x = .fitted, y = sqrt(abs(.std_resid)))) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  labs(
    title = "Scale-Location Plot",
    subtitle = "Check for Constant Variance",
    x = "Fitted Values",
    y = "√|Standardized Residuals|"
  ) +
  theme_minimal()

# 3. Residual Distribution (Fixed density call)
# Examine the distribution of the training residuals and compare it visually
# with a normal distribution having the same mean and standard deviation.
p3 <- ggplot(preds.train, aes(x = .resid)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30,
                 fill = "lightblue", color = "black", alpha = 0.7) +
  geom_density(color = "darkred", linewidth = 1) +
  stat_function(fun = dnorm,
                args = list(mean = mean(preds.train$.resid),
                            sd = sd(preds.train$.resid)),
                color = "blue", linetype = "dashed") +
  labs(
    title = "Residual Distribution",
    subtitle = "Blue dashed line = Normal distribution reference",
    x = "Residuals", y = "Density"
  ) +
  theme_minimal()

# Q-Q Plot
# Create a Q-Q plot as an additional visual assessment of the training
# residual distribution.
p4 <- ggplot(preds.train, aes(sample = .resid)) +
  stat_qq(color = "darkblue", alpha = 0.6) +
  stat_qq_line(color = "red", linewidth = 1) +
  labs(
    title = "Normal Q-Q Plot of Residuals",
    subtitle = "Residuals should align along red line",
    x = "Theoretical Quantiles", y = "Sample Quantiles"
  ) +
  theme_minimal()

# Combine diagnostics (ensure patchwork is loaded)
# Combine the four training residual plots into a single diagnostic figure.
diagnostic_plots <- (p1 | p2) / (p3 | p4) +
  plot_annotation(
    title = "Keras Model Diagnostic Plots - Final Training Data",
    subtitle = "Residual-based model checks",
    caption = paste("Training observations:", nrow(preds.train))
  )

# Save combined plots
ggsave(
  filename = paste0(plotsdir, "/", "neuralnetbetareg", "/", "final_model_nn_diagnostic_plots.png"),
  plot = diagnostic_plots,
  width = 14,
  height = 10,
  dpi = 300
)
cat("Diagnostic plots saved as: rfinal_model_diagnostic_plots.png\n")

# ------------------------------------------------------------
# Residual Statistics Summary
# ------------------------------------------------------------
# Summarise the magnitude and distribution of the training residuals using
# the main residual-based descriptive measures.
rresid_stats <- preds.train %>%
  summarise(
    Observations = n(),
    Mean_Residual = mean(.resid),
    SD_Residual = sd(.resid),
    Min_Residual = min(.resid),
    Max_Residual = max(.resid),
    MSE = mean(.resid^2),
    RMSE = sqrt(mean(.resid^2)),
    MAE = mean(abs(.resid)),
    Within_pm_0.05 = paste0(round(mean(abs(.resid) <= 0.05) * 100, 1), "%"),
    Within_pm_0.10 = paste0(round(mean(abs(.resid) <= 0.10) * 100, 1), "%")
  )

cat("\n=== Residual Statistics ===\n")
print(rresid_stats)


# ------------------------------------------------------------
# Homoscedasticity (Breusch–Pagan Test)
# ------------------------------------------------------------
# Apply the Breusch-Pagan test as an additional assessment of whether the
# residual variance changes systematically with the fitted values.
rlm_resid <- lm(.resid ~ .fitted, data = preds.train)
rbp_test <- bptest(rlm_resid)

cat("\n=== Homoscedasticity Test (Breusch–Pagan) ===\n")
cat("BP =", round(rbp_test$statistic, 4),
    ", p-value =", format.pval(rbp_test$p.value, digits = 4), "\n")

if (rbp_test$p.value < 0.05) {
  cat("Evidence of heteroscedasticity (non-constant variance)\n")
} else {
  cat("No evidence of heteroscedasticity (constant variance)\n")
}

# ------------------------------------------------------------
# Residuals vs Actual
# ------------------------------------------------------------
# Plot training residuals against observed MPI values to examine whether
# prediction errors change systematically across the MPI range.
p5 <- ggplot(preds.train, aes(x = mpi, y = .resid)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Actual MPI Values (Train Data)",
    subtitle = "Check for systematic bias",
    x = "Actual MPI", y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "neuralnetbetareg", "/","train_residuals_vs_actual_nn.png"), p5, width = 10, height = 6, dpi = 300)
cat("Residuals vs Actual plot saved as: rrain_residuals_vs_actual_nn.png\n")

# ------------------------------------------------------------
# Model Assumptions Summary
# ------------------------------------------------------------
cat("\n=== MODEL ASSUMPTIONS SUMMARY ===\n")
cat("1. Linearity → Check Residuals vs Fitted (random scatter)\n")
cat("2. Homoscedasticity → Constant variance (scale-location plot)\n")
cat("3. Normality → Residual distribution and Q-Q plot\n")
cat("4. Independence → Residuals show no autocorrelation\n")

# Calculate the correlation between fitted values and training residuals as
# an additional numerical summary of the residual pattern.
rresid_cor <- cor(preds.train$.fitted, preds.train$.resid)
cat("\nCorrelation between fitted and residuals:", round(rresid_cor, 6), "\n")
cat("(Should be ≈ 0 for well-specified models)\n")


# Residuals vs Fitted (Test)
# Examine residuals against fitted values for the independent test dataset
# to assess prediction patterns on observations not used to fit the model.
rtestrf <- ggplot(preds.test, aes(x = .fitted, y = .resid)) +
  geom_point(alpha = 0.6, color = "darkred") +
  geom_hline(yintercept = 0, color = "blue", linetype = "dashed") +
  geom_smooth(method = "loess", color = "orange", se = TRUE) +
  labs(
    title = "Residuals vs Fitted (Test Data)",
    subtitle = "Check generalization and bias on unseen data",
    x = "Fitted Values (Predicted MPI)",
    y = "Residuals"
  ) +
  theme_minimal()

ggsave(paste0(plotsdir, "/", "neuralnetbetareg", "/","test_residuals_vs_fitted_nn.png"), rtestrf, width = 10, height = 6, dpi = 300)
cat("Residuals vs Actual plot saved as: test_residuals_vs_fitted_nn.png\n")

# Combine diagnostics (ensure patchwork is loaded)
# Combine the training residual-versus-actual and test residual-versus-fitted
# plots into a single figure for comparison.
diag_plots <- (p5 | rtestrf) +
  plot_annotation(
    title = "Keras Model Residual Plots",
    subtitle = "Residual-based model checks",
    caption = paste("Residual vs actual (Train) and Residual vs Fitted (Test)")
  )

# Save combined plots
ggsave(
  filename = paste0(plotsdir, "/", "neuralnetbetareg", "/", "Residual_plots.png"),
  plot = diag_plots,
  width = 14,
  height = 10,
  dpi = 300
)
# cat("Diagnostic plots saved as: final_model_diagnostic_plots.png\n")


# ===============================================
# VARIABLE IMPORTANCE ANALYSIS FOR KERAS MODEL - NEW
# ===============================================

cat("\n=== VARIABLE IMPORTANCE ANALYSIS (Keras Model) ===\n")

# -----------------------------
# 0) Inputs you must have
# -----------------------------
# Confirm that the ultimately selected neural network model is available
# before calculating the model-specific variable-importance summary.
stopifnot(exists("final_nn"))
#stopifnot(exists("train_full_proc"))

# Output folder
#if (!exists("destdftofile")) destdftofile <- getwd()
#if (!dir.exists(destdftofile)) dir.create(destdftofile, recursive = TRUE)

# Which link was used for the final NN (needed to inverse-transform predictions)
# If you don't have best_config$link, set nn_link manually.
#nn_link <- if (exists("rbest_overall") && "link" %in% names(rbest_overall)) rbest_overall$link else "logit"

# =========================================================
# 1) Weight-based importance (input -> first hidden layer)
# =========================================================
cat("\n[1/3] Calculating input-weight importance...\n")

# Extract the fitted neural network weights and use the absolute input-layer
# weights to construct a relative predictor-importance summary.
weights_list <- keras3::get_weights(final_nn)
if (length(weights_list) < 1) stop("Could not extract weights from final_nn.")

w1 <- weights_list[[1]]  # matrix: (n_features x units1)

input_vars <- colnames(Xtrain)

# Sum the absolute weights connecting each predictor to the first hidden layer
# and scale the resulting values relative to the largest importance value.
var_importance_w <- tibble(
  variable = input_vars,
  abs_weight_sum = apply(abs(w1), 1, sum)
) %>%
  mutate(
    importance = abs_weight_sum / max(abs_weight_sum, na.rm = TRUE)
  ) %>%
  arrange(desc(importance))

# Save table
readr::write_csv(var_importance_w, file.path(model_dir, "/", "neuralnetbetareg", "/", "rnn_var_importance_weights.csv"))

# Plot (single colour blue, dissertation-friendly)
# title = "Variable Importance - Neural Network (Built-in)",

# Plot the 15 predictors with the highest input-weight importance for the
# ultimately selected Neural Network model.
p_imp <- var_importance_w %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variable, importance), y = importance)) +
  geom_col(
    fill = "steelblue",
    alpha = 0.85,
    width = 0.85
  ) +
  coord_flip() +
  labs(
    subtitle = "Sum of absolute input-layer weights (scaled 0–1)",
    x = "Predictor variables",
    y = "Relative importance (0–1)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    panel.grid.major.y = element_blank()
  )

ggsave(
  file.path(plotsdir, "neuralnetbetareg", "variable_importance_weights.png"),
  p_imp,
  width = 10,
  height = 9,   
  dpi = 300
)


# =====================================================
# Prediction vs Actual Scatterplots for Keras Model
# =====================================================

cat("\n=== CREATING PREDICTION VS ACTUAL SCATTERPLOTS ===\n")

# ---------------------------
# 1. Training Data Scatterplot
# ---------------------------

cat("\nCreating training data scatterplot...\n")

# Calculate training metrics
# Calculate the regression performance measures used to annotate the
# training predicted-versus-actual plot.
train_rmse <- rmse_vec(preds.train$mpi, preds.train$.fitted)
train_mae <- mae_vec(preds.train$mpi, preds.train$.fitted)
train_rsq <- rsq_vec(preds.train$mpi, preds.train$.fitted)
train_cor <- cor(preds.train$mpi, preds.train$.fitted)

# Create training scatterplot
# Plot predicted MPI against observed MPI for the training dataset. The
# 45-degree reference line represents perfect agreement between both values.
train_scatter <- ggplot(preds.train, aes(x = mpi, y = .fitted)) +
  geom_point(alpha = 0.6, color = "steelblue", size = 1.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
  coord_equal() +
  labs(
    title = "Predicted vs Actual MPI (Train)",
    x = "Actual MPI",
    y = "Predicted MPI"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "plain", size = 14),
    plot.subtitle = element_text(size = 12)
  ) +
  annotate(
    "text",
    x = min(preds.train$mpi),
    y = max(preds.train$.fitted),
    hjust = 0, vjust = 1,
    label = paste(
      sprintf("RMSE = %.4f", train_rmse),
      sprintf("MAE = %.4f", train_mae),
      sprintf("R² = %.4f", train_rsq),
      sprintf("Corr = %.4f", train_cor),
      sep = "\n"
    ),
    size = 4,
    color = "darkgreen",
    fontface = "bold"
  )

ggsave(paste0(plotsdir, "/", "neuralnetbetareg", "/", "train_prediction_vs_actual_training.png"),
       train_scatter, width = 10, height = 8, dpi = 300)
cat("Training data scatterplot saved\n")

# ---------------------------
# 2. Test Data Scatterplot
# ---------------------------

# Create the corresponding predicted-versus-actual plot for the independent
# test dataset and calculate its regression performance measures.
if ("mpi" %in% names(preds.test)) {
  cat("\nCreating test data scatterplot...\n")
  
  
  
  # Calculate test metrics
  test_rmse <- rmse_vec(preds.test$mpi, preds.test$.fitted)
  test_mae <- mae_vec(preds.test$mpi, preds.test$.fitted)
  test_rsq <- rsq_vec(preds.test$mpi, preds.test$.fitted)
  test_cor <- cor(preds.test$mpi, preds.test$.fitted)
  
  test_scatter <- ggplot(preds.test, aes(x = mpi, y = .fitted)) +
    geom_point(alpha = 0.6, color = "purple", size = 1.5) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
    geom_smooth(method = "lm", color = "darkorange", se = TRUE, linewidth = 0.8) +
    coord_equal() +
    labs(
      title = "Predicted vs Actual MPI (Test)",
      x = "Actual MPI",
      y = "Predicted MPI"
    ) +
    theme_minimal() +
    annotate("text",
             x = min(preds.test$mpi),
             y = max(preds.test$.fitted),
             hjust = 0, vjust = 1,
             label = paste(
               sprintf("RMSE = %.4f", test_rmse),
               sprintf("MAE = %.4f", test_mae),
               sprintf("R² = %.4f", test_rsq),
               sprintf("Corr = %.4f", test_cor),
               sep = "\n"),
             size = 3.5, color = "darkred", fontface = "bold")
  
  ggsave(paste0(plotsdir,"/", "neuralnetbetareg", "/", "test_prediction_vs_actual_nn.png"),
         test_scatter, width = 10, height = 8, dpi = 300)
  cat("Test data scatterplot saved\n")
  
}

# ---------------------------
# 5. Poverty Classification Analysis - Training DataSet
# ---------------------------

cat("\nCreating poverty classification scatterplot...\n")

# Classify the training observations according to the observed and predicted
# MPI values using the predefined poverty threshold.
poverty_scatter_train_data <- preds.train %>%
  mutate(
    Actual_Poverty = ifelse(mpi > 0.3333, "Poor", "Non-Poor"),
    Predicted_Poverty = ifelse(.fitted > 0.3333, "Poor", "Non-Poor"),
    Classification = case_when(
      Actual_Poverty == "Poor" & Predicted_Poverty == "Poor" ~ "True Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Non-Poor" ~ "True Non-Poor",
      Actual_Poverty == "Poor" & Predicted_Poverty == "Non-Poor" ~ "False Non-Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Poor" ~ "False Poor"
    )
  )

# Plot the training observations by poverty-classification outcome. The
# vertical and horizontal reference lines represent the MPI poverty threshold.
poverty_scatter_train_plot <- ggplot(poverty_scatter_train_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.5) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non-Poor" = "darkgreen",
    "False Poor" = "orange",
    "False Non-Poor" = "purple"
  )) +
  coord_equal() +
  labs(title = "Poverty Classification (Train)", subtitle = "Threshold MPI = 0.3333",
       x = "Actual MPI", y = "Predicted MPI", color = "Class") +
  theme_minimal()

classification_summary_train <- poverty_scatter_train_data %>%
  count(Classification) %>%
  mutate(Percentage = n / sum(n) * 100)

cat("\nPoverty Classification Summary:\n")
print(classification_summary_train)

ggsave(paste0(plotsdir, "/", "neuralnetbetareg", "/", "training_class_prediction_vs_actual_poverty.png"),
       poverty_scatter_train_plot, width = 12, height = 10, dpi = 300)
cat(" Poverty classification scatterplot saved\n")


# ---------------------------
# 6. Poverty Classification Analysis - Test DataSet
# ---------------------------

cat("\nCreating poverty classification scatterplot...\n")

# Apply the same poverty classification to the independent test observations.
poverty_scatter_test_data <- preds.test %>%
  mutate(
    Actual_Poverty = ifelse(mpi > 0.3333, "Poor", "Non-Poor"),
    Predicted_Poverty = ifelse(.fitted > 0.3333, "Poor", "Non-Poor"),
    Classification = case_when(
      Actual_Poverty == "Poor" & Predicted_Poverty == "Poor" ~ "True Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Non-Poor" ~ "True Non-Poor",
      Actual_Poverty == "Poor" & Predicted_Poverty == "Non-Poor" ~ "False Non-Poor",
      Actual_Poverty == "Non-Poor" & Predicted_Poverty == "Poor" ~ "False Poor"
    )
  )

# Create the corresponding poverty-classification plot for the independent
# test dataset.
poverty_scatter_test_plot <- ggplot(poverty_scatter_test_data, aes(x = mpi, y = .fitted, color = Classification)) +
  geom_point(alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_hline(yintercept = 0.3333, color = "red", linetype = "dashed", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.5) +
  scale_color_manual(values = c(
    "True Poor" = "darkred",
    "True Non-Poor" = "darkgreen",
    "False Poor" = "orange",
    "False Non-Poor" = "purple"
  )) +
  coord_equal() +
  labs(title = "Poverty Classification (Test)", subtitle = "Threshold MPI = 0.3333",
       x = "Actual MPI", y = "Predicted MPI", color = "Class") +
  theme_minimal()

classification_summary_test <- poverty_scatter_test_data %>%
  count(Classification) %>%
  mutate(Percentage = n / sum(n) * 100)

cat("\nPoverty Classification Summary:\n")
print(classification_summary_test)

ggsave(paste0(plotsdir, "/", "neuralnetbetareg", "/", "test_class_prediction_vs_actual_poverty.png"),
       poverty_scatter_test_plot, width = 12, height = 10, dpi = 300)
cat(" Poverty classification scatterplot saved\n")

# Create a common subtitle containing the search stage, preprocessing recipe,
# link function and hidden-layer sizes of the ultimately selected model.
nn_subtitle <- paste0(
  "Stage: ", chosen$source,
  " | Recipe: ", chosen_recipe,
  " | link=", chosen_link,
  " | hidden layer 1=", chosen$units1,
  " | hidden layer 2=", chosen$units2
)

# Combine the training and test predicted-versus-actual plots with the
# corresponding poverty-classification plots in a 2-by-2 layout.
multi_panel_plot <- (train_scatter + theme(axis.text = element_text(size = 8)) +
                       test_scatter + theme(axis.text = element_text(size = 8))) /
  (poverty_scatter_train_plot + theme(axis.text = element_text(size = 8)) +
     poverty_scatter_test_plot + theme(axis.text = element_text(size = 8))) +
  plot_annotation(
    
    subtitle = nn_subtitle,
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )

ggsave(
  file.path(plotsdir, "neuralnetbetareg", "comprehensive_prediction_analysis_nn.png"),
  multi_panel_plot, width = 16, height = 12, dpi = 300
)

cat(" Comprehensive Neural Networkmulti-panel plot saved:\n")
cat(file.path(plotsdir, "neuralnetbetareg", "comprehensive_prediction_analysis_nn.png"), "\n\n")










########start New
# ============================================================================
# BIAS ANALYSIS FROM SAVED MODELS
# ============================================================================
# This script loads pre-trained models and performs systematic bias analysis
# across poverty status groups (Non-Poor vs Poor at MPI = 0.3333 threshold)
# ============================================================================

# Load the saved test residual datasets for all four models so that the
# systematic bias analysis is based on the same independent test observations.
# Base R — reads into a data.frame
test_resid_data_betareg <- read.csv(file.path(model_dir, "betareg", "test_resid_data_betareg.csv"))
test_resid_data_rf <- read.csv(file.path(model_dir, "randomforest", "test_resid_data_rf.csv"))
test_resid_data_xgb_cpy <- read.csv(file.path(model_dir, "xgboost", paste0("run_", run_id), "test_resid_data_xgboost.csv"))
test_resid_data_nn <- read.csv(file.path(model_dir, "neuralnetbetareg", "run_20260512_064338", "bsttestpre", "test_resid_data_nn.csv"))
#run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")



# ============================================================================
# SECTION 4: COMBINE ALL RESULTS
# ============================================================================
cat("\n=== Combining All Model Results ===\n")

# Combine the model-specific test residual datasets into one structure for
# direct comparison across the four learning algorithms.
all_results <- bind_rows(
  test_resid_data_betareg,
  test_resid_data_rf,
  test_resid_data_xgb_cpy,
  test_resid_data_nn
)

View(all_results)

cat("Total results combined: N =", nrow(all_results), "\n")
cat("Models included:", unique(all_results$model), "\n")

# Save CSV
all_results_csv <- file.path(model_dir, "biasanalysis", paste0("all_model_results_", run_id, ".csv"))
write_csv(all_results, all_results_csv)

# Save RDS
all_results_rds <- file.path(model_dir, "biasanalysis", paste0("all_model_results_", run_id, ".rds"))
saveRDS(all_results, all_results_rds)

cat("Files saved:\n")
cat(" - CSV:", all_results_csv, "\n")
cat(" - RDS:", all_results_rds, "\n")

# ============================================================================
# SECTION 5: BIAS ANALYSIS - 2 GROUP SPLIT (Non-Poor vs Poor)
# ============================================================================
cat("\n=== Performing Bias Analysis (2-Group Split) ===\n")

# Create poverty status groups based on 0.333 threshold
# Assign each test observation to the Non-Poor or Poor group using the
# predefined MPI poverty threshold of 0.3333.
bias_analysis <- all_results %>%
  mutate(
    poverty_status = case_when(
      mpi < 0.3333 ~ "Non_Poor",
      mpi >= 0.3333 ~ "Poor"
    ),
    poverty_status = factor(poverty_status, levels = c("Non_Poor", "Poor"))
  )

# Calculate bias statistics for each model and poverty group
# Summarise prediction errors separately for each model and poverty group.
# Residuals are defined as actual MPI minus predicted MPI; therefore, a
# positive mean residual indicates underprediction and a negative value
# indicates overprediction.
bias_summary <- bias_analysis %>%
  group_by(model, poverty_status) %>%
  summarise(
    n_regions = n(),
    mean_actual_mpi = mean(mpi),
    mean_predicted_mpi = mean(.fitted),
    mean_residual = mean(.resid),
    sd_residual = sd(.resid),
    se_residual = sd_residual / sqrt(n_regions),
    median_residual = median(.resid),
    rmse = sqrt(mean(.resid^2)),
    mae = mean(abs(.resid)),
    .groups = "drop"
  )

# Print summary table
cat("\n=== BIAS SUMMARY TABLE ===\n")
print(bias_summary, n = Inf)

# Save bias summary
write_csv(bias_summary, file.path(model_dir, "biasanalysis", "bias_analysis_summary.csv"))
cat("\nBias summary saved to: bias_analysis_summary.csv\n")


# Calculate one bias-magnitude value per model
# Measure the difference in mean residuals between the Poor and Non-Poor
# groups. A smaller value indicates more balanced systematic bias across
# the two poverty groups.
bias_magnitude_by_model <- bias_summary %>%
  select(model, poverty_status, mean_residual) %>%
  pivot_wider(
    names_from = poverty_status,
    values_from = mean_residual
  ) %>%
  mutate(
    bias_magnitude = abs(Poor - Non_Poor)
  ) %>%
  select(model, bias_magnitude)

# Attach the model-level bias magnitude to both poverty-group rows
bias_summary_table <- bias_summary %>%
  left_join(
    bias_magnitude_by_model,
    by = "model"
  ) %>%
  arrange(
    factor(
      model,
      levels = c(
        "Neural Network",
        "Random Forest",
        "XGBoost",
        "Beta Regression"
      )
    ),
    poverty_status
  )

write_csv(
  bias_summary_table,
  file.path(
    model_dir,
    "biasanalysis",
    "bias_analysis_summary_with_magnitude.csv"
  )
)

# ============================================================================
# SECTION 6: STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
cat("\n=== Testing Statistical Significance of Bias ===\n")

# For each model, test if mean residual differs between groups
# Compare the residual distributions between Poor and Non-Poor regions for
# each model using an independent two-sample t-test.
bias_tests <- bias_analysis %>%
  group_by(model) %>%
  summarise(
    # Test if residuals differ between Non-Poor and Poor
    t_statistic = t.test(
      .resid[poverty_status == "Poor"],
      .resid[poverty_status == "Non_Poor"]
    )$statistic,
    p_value = t.test(
      .resid[poverty_status == "Poor"],
      .resid[poverty_status == "Non_Poor"]
    )$p.value,
    significant = p_value < 0.05,
    .groups = "drop"
  )

cat("\n=== STATISTICAL TESTS ===\n")
print(bias_tests)

# Save test results
write_csv(bias_tests, paste0(model_dir,"/", "biasanalysis", "/", "bias_statistical_tests.csv"))


# ============================================================================
# SECTION 7: CREATE PUBLICATION-READY TABLES
# ============================================================================

# Table 1: Overall bias comparison across models
# Prepare a concise model-by-poverty-group table containing the mean residual,
# its standard error and RMSE for reporting.
table1 <- bias_summary %>%
  select(model, poverty_status, n_regions, mean_residual, se_residual, rmse) %>%
  mutate(
    mean_residual = sprintf("%.3f", mean_residual),
    se_residual = sprintf("%.3f", se_residual),
    rmse = sprintf("%.4f", rmse)
  ) %>%
  arrange(model, poverty_status)

cat("\n=== TABLE 1: Bias by Model and Poverty Status ===\n")
print(table1, n = Inf)

write_csv(table1, file.path(model_dir, "biasanalysis", "table1_bias_by_model.csv"))

# Table 2: Bias magnitude comparison (wide format)
# Reshape the group mean residuals into a wide table and calculate one
# systematic bias-magnitude value for each model.
table2 <- bias_summary %>%
  select(model, poverty_status, mean_residual) %>%
  pivot_wider(
    names_from = poverty_status,
    values_from = mean_residual
  ) %>%
  mutate(
    bias_magnitude = abs(`Poor` - `Non_Poor`),
    `Non-Poor` = sprintf("%.4f", `Non_Poor`),
    `Poor` = sprintf("%.4f", `Poor`),
    bias_magnitude = sprintf("%.4f", bias_magnitude)
  ) %>%
  arrange(bias_magnitude)

cat("\n=== TABLE 2: Bias Magnitude Comparison ===\n")
print(table2)

write_csv(table2, file.path(model_dir, "biasanalysis", "table2_bias_magnitude.csv"))


# ============================================================================
# SECTION 8: VISUALIZATION
# ============================================================================
cat("\n=== Creating Visualizations ===\n")

# Plot 1: Residuals by poverty status for all models
# Visualise the distribution of test residuals within each poverty group.
# Values above zero represent underprediction, while values below zero
# represent overprediction.
p1 <- ggplot(bias_analysis, aes(x = poverty_status, y = .resid, fill = poverty_status)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.1, size = 0.5) +
  facet_wrap(~ model, ncol = 2) +
  scale_fill_manual(values = c("Non_Poor" = "lightblue", "Poor" = "coral")) +
  labs(
    title = "Model Bias Across Poverty Status",
    subtitle = "Positive residuals indicate underprediction (model predicts lower than actual)",
    x = "Poverty Status (MPI threshold = 0.333)",
    y = "Residual (Actual MPI - Predicted MPI)",
    fill = "Poverty Status"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(file.path(model_dir, "biasanalysis","plot1_residuals_by_poverty_status.png"), p1, width = 10, height = 8, dpi = 300)
cat("Saved: plot1_residuals_by_poverty_status.png\n")

# Plot 2: Mean residuals comparison across models
# Compare mean residuals across models and poverty groups. Standard-error
# bars show the uncertainty around each group mean residual.
p2 <- bias_summary %>%
  ggplot(aes(x = model, y = mean_residual, fill = poverty_status)) +
  geom_col(position = position_dodge(width = 0.8), alpha = 0.8) +
  geom_errorbar(
    aes(ymin = mean_residual - se_residual, ymax = mean_residual + se_residual),
    position = position_dodge(width = 0.8),
    width = 0.3
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  scale_fill_manual(values = c("Non_Poor" = "lightblue", "Poor" = "coral")) +
  labs(
    title = "Mean Prediction Bias Across Poverty Groups",
    subtitle = "Error bars show standard errors",
    x = "Model",
    y = "Mean Residual (Actual - Predicted)",
    fill = "Poverty Status"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

ggsave(file.path(model_dir,"biasanalysis","plot2_mean_residuals_comparison.png"), p2, width = 10, height = 6, dpi = 300)
cat("Saved: plot2_mean_residuals_comparison.png\n")

# Plot 3: Residuals vs Actual MPI (to show regression-to-mean pattern)
# Plot residuals against observed MPI to examine the regression-to-the-mean
# pattern and whether prediction errors change across the poverty threshold.
p3 <- bias_analysis %>%
  ggplot(aes(x = mpi, y = .resid, color = model)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 0.3333, linetype = "dashed", color = "blue", alpha = 0.5) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
  facet_wrap(~ model, ncol = 2) +
  labs(
    title = "Residuals vs Actual MPI: Regression-to-Mean Pattern",
    subtitle = "Vertical line at MPI = 0.3333 (poverty threshold)",
    x = "Actual MPI",
    y = "Residual (Actual - Predicted)",
    color = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(file.path(model_dir,"biasanalysis", "plot3_residuals_vs_actual_mpi.png"), p3, width = 10, height = 8, dpi = 300)
cat("Saved: plot3_residuals_vs_actual_mpi.png\n")

# ============================================================================
# SECTION 10: GENERATE LATEX TABLE CODE
# ============================================================================
cat("\n=== Generating LaTeX Table Code ===\n")

# Function to create LaTeX table
# Convert the systematic bias summary into LaTeX code for direct inclusion
# in the dissertation results table.
create_latex_bias_table <- function(data, caption, label) {
  
  required_cols <- c(
    "model",
    "poverty_status",
    "n_regions",
    "mean_residual",
    "se_residual",
    "rmse",
    "bias_magnitude"
  )
  
  missing_cols <- setdiff(required_cols, names(data))
  
  if (length(missing_cols) > 0) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
  
  cat("\\begin{table}[htbp]\n")
  cat("\\centering\n")
  cat("\\caption{", caption, "}\n", sep = "")
  cat("\\label{", label, "}\n", sep = "")
  cat("\\begin{tabular}{l l l l l l l}\n")
  cat("\\hline\n")
  
  cat(
    "\\textbf{Model} & ",
    "\\textbf{Poverty\\_Status} & ",
    "\\textbf{N} & ",
    "\\textbf{Mean\\_Residual} & ",
    "\\textbf{SE} & ",
    "\\textbf{RMSE} & ",
    "\\textbf{Bias\\_Magnitude} \\\\\n",
    sep = ""
  )
  
  cat("\\hline\n")
  
  model_order <- unique(as.character(data$model))
  
  for (model_name in model_order) {
    
    model_data <- data %>%
      filter(model == model_name) %>%
      arrange(poverty_status)
    
    if (nrow(model_data) != 2) {
      stop(
        "Expected exactly two poverty-group rows for model: ",
        model_name
      )
    }
    
    non_poor <- model_data %>%
      filter(poverty_status == "Non_Poor")
    
    poor <- model_data %>%
      filter(poverty_status == "Poor")
    
    if (nrow(non_poor) != 1 || nrow(poor) != 1) {
      stop(
        "Missing or duplicated poverty group for model: ",
        model_name
      )
    }
    
    display_model <- model_name
    display_non_poor <- "Non-Poor"
    display_poor <- "Poor"
    
    cat(
      sprintf(
        "\\multirow{2}{*}{%s} & %s & %d & %.4f & %.4f & %.4f & \\multirow{2}{*}{%.4f} \\\\\n",
        display_model,
        display_non_poor,
        non_poor$n_regions,
        non_poor$mean_residual,
        non_poor$se_residual,
        non_poor$rmse,
        non_poor$bias_magnitude
      )
    )
    
    cat(
      sprintf(
        "& %s & %d & %.4f & %.4f & %.4f & \\\\\n",
        display_poor,
        poor$n_regions,
        poor$mean_residual,
        poor$se_residual,
        poor$rmse
      )
    )
    
    cat("\\hline\n")
  }
  
  cat("\\end{tabular}\n")
  cat("\\end{table}\n")
}

# Generate LaTeX table
cat("\n--- LaTeX Table Code ---\n")

cat("\n--- LaTeX Table Code ---\n")

create_latex_bias_table(
  bias_summary_table,
  caption = paste(
    "Systematic Bias Analysis Across Model",
    "Architectures and Poverty Status"
  ),
  label = "tab:bias_analysis"
)


# ============================================================================
# SECTION 11: SUMMARY STATISTICS FOR WRITE-UP
# ============================================================================
cat("\n=== KEY STATISTICS FOR DISSERTATION WRITE-UP ===\n")

# Get best model (lowest bias magnitude)
# Identify the model with the smallest difference in mean residual between
# the Poor and Non-Poor groups.
best_model <- table2 %>%
  slice_min(order_by = as.numeric(bias_magnitude), n = 1) %>%
  pull(model)

cat("\nBest Model (lowest bias):", best_model, "\n")

# Get statistics for best model
best_model_stats <- bias_summary %>%
  filter(model == best_model)

cat("\n--- Statistics for", best_model, "---\n")
print(best_model_stats)







########################Prediction Interval and Coverage########################
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tibble)
})

# ============================================================
# 1. Helper functions
# ============================================================

# Calculate the conformal residual quantile using the finite-sample
# rank adjustment for the requested miscoverage level alpha.
calc_conformal_q <- function(residuals, alpha = 0.10) {
  residuals <- residuals[is.finite(residuals)]
  if (length(residuals) == 0) stop("No finite residuals available to compute conformal q.")
  
  k <- ceiling((length(residuals) + 1) * (1 - alpha))
  q <- sort(residuals)[min(k, length(residuals))]
  
  list(
    q = q,
    k = k,
    n_residuals = length(residuals)
  )
}

# Construct symmetric prediction intervals around each point prediction and
# restrict the interval bounds to the valid MPI range from 0 to 1.
apply_conformal_interval <- function(pred, q, lower_bound = 0, upper_bound = 1) {
  tibble(
    lo = pmax(lower_bound, pred - q),
    hi = pmin(upper_bound, pred + q)
  )
}

# Evaluate whether each observed MPI value is covered by its interval,
# calculate interval width, assign the observed poverty group, and identify
# intervals that cross the MPI poverty threshold.
coverage_summary <- function(df, truth_col = "mpi_true", lo_col = "lo", hi_col = "hi",
                             threshold = 0.3333) {
  df %>%
    mutate(
      covered = .data[[truth_col]] >= .data[[lo_col]] & .data[[truth_col]] <= .data[[hi_col]],
      width = .data[[hi_col]] - .data[[lo_col]],
      poverty_group = ifelse(.data[[truth_col]] >= threshold, "poor", "non_poor"),
      threshold_uncertain = .data[[lo_col]] <= threshold & .data[[hi_col]] >= threshold
    )
}

# ============================================================
# 2. File paths - takes in the full cv prediction file and filters for results 
# using the best xgboost model - in run it comes from space-filling all_pred_no
# ============================================================
# Specify the saved XGBoost cross-validation prediction file used to obtain
# the out-of-fold absolute residuals for conformal calibration.
cv_sf_file     <- "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI/modeldir/xgboost/run_20260803_102459/xgb_cv_predictions_20260803_102459.csv"
#cv_sf_file     <- "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI/modeldir/xgboost/run_20260325_081905/xgb_refine_cv_predictions_20260325_081905.csv"
# cv_refine_file <- "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI/modeldir/xgboost/run_20260510_133304/xgb_refine_cv_predictions_20260510_133304.csv"


# ============================================================
# 3. Compute q from SPACE-FILLING CV file
# ============================================================
#get_q_space_filling <- function(best_row, cv_file, alpha = 0.10) {

#  cv_df <- read_csv(cv_file, show_col_types = FALSE)

#  cv_best <- cv_df %>%
#    filter(
#      recipe == best_row$recipe[[1]],
#      mtry == best_row$mtry[[1]],
#      trees == best_row$trees[[1]],
#      min_n == best_row$min_n[[1]],
#      tree_depth == best_row$tree_depth[[1]],
#      abs(learn_rate - best_row$learn_rate[[1]]) < 1e-12,
#      abs(sample_size - best_row$sample_size[[1]]) < 1e-12
#    )

#  # Aggregate repeated CV predictions by original training row
#  cv_agg <- cv_best %>%
#    group_by(.row) %>%
#    summarise(
#      mpi_true = first(mpi),
#      mpi_pred = mean(.pred, na.rm = TRUE),
#      .groups = "drop"
#    )



#  q_info <- calc_conformal_q(abs(cv_agg$mpi_true - cv_agg$mpi_pred), alpha = alpha)

#  list(
#    q = q_info$q,
#    k = q_info$k,
#    n_residuals = q_info$n_residuals,
#    cv_used = cv_agg
#  )
#}

# For a space-filling XGBoost winner, retain only the cross-validation
# predictions corresponding to the selected hyperparameter configuration
# and use their absolute out-of-fold residuals to estimate the conformal q.
get_q_from_xgb_fold_predictions <- function(best_row,cv_file, alpha = 0.10) {
  
  cv_df <- read_csv(cv_file, show_col_types = FALSE)
  
  cv_best <- cv_df %>%
    filter(
      recipe == best_row$recipe[[1]],
      mtry == best_row$mtry[[1]],
      trees == best_row$trees[[1]],
      min_n == best_row$min_n[[1]],
      tree_depth == best_row$tree_depth[[1]],
      abs(learn_rate - best_row$learn_rate[[1]]) < 1e-12,
      abs(sample_size - best_row$sample_size[[1]]) < 1e-12
    )
  
  cv_residuals <- cv_best %>%
    mutate(abs_resid = abs(mpi - .pred)) %>%
    filter(is.finite(abs_resid))
  
  q_info <- calc_conformal_q(cv_residuals$abs_resid, alpha = alpha)
  
  list(
    q = q_info$q,
    k = q_info$k,
    n_residuals = q_info$n_residuals,
    cv_used = cv_residuals
  )
}

# ============================================================
# 4. Compute q from REFINED CV file
# ============================================================


#get_q_refined <- function(best_row, cv_file, alpha = 0.10) {

#  cv_df <- read_csv(cv_file, show_col_types = FALSE)

#  cv_best <- cv_df %>%
#    filter(
#      recipe == best_row$recipe[[1]],
#      grid_mtry == best_row$mtry[[1]],
#      grid_trees == best_row$trees[[1]],
#      grid_min_n == best_row$min_n[[1]],
#      grid_tree_depth == best_row$tree_depth[[1]],
#      abs(grid_learn_rate - best_row$learn_rate[[1]]) < 1e-12,
#      abs(grid_sample_size - best_row$sample_size[[1]]) < 1e-12
#    )

#  if (nrow(cv_best) == 0) {
#    stop("No matching rows found in refined CV file for the selected best model.")
#  }

#  # aggregate repeated out-of-sample predictions to one value per original observation
#  cv_agg <- cv_best %>%
#    group_by(.row) %>%
#    summarise(
#      mpi_true = first(mpi_true),
#      mpi_pred = mean(mpi_pred, na.rm = TRUE),
#      .groups = "drop"
#    )

#  q_info <- calc_conformal_q(abs(cv_agg$mpi_true - cv_agg$mpi_pred), alpha = alpha)

#  list(
#    q = q_info$q,
#    k = q_info$k,
#    n_residuals = q_info$n_residuals,
#    cv_used = cv_agg
#  )
#}

# For a refined-stage XGBoost winner, match the selected refined-grid
# configuration and calculate the conformal calibration residuals from
# its saved cross-validation predictions.
get_q_from_xgb_refined_predictions <- function(
    best_row,
    cv_file,
    alpha = 0.10
) {
  
  cv_df <- readr::read_csv(
    cv_file,
    show_col_types = FALSE
  )
  
  cv_best <- cv_df %>%
    dplyr::filter(
      recipe == best_row$recipe[[1]],
      grid_mtry == best_row$mtry[[1]],
      grid_trees == best_row$trees[[1]],
      grid_min_n == best_row$min_n[[1]],
      grid_tree_depth == best_row$tree_depth[[1]],
      abs(
        grid_learn_rate -
          best_row$learn_rate[[1]]
      ) < 1e-12,
      abs(
        grid_sample_size -
          best_row$sample_size[[1]]
      ) < 1e-12
    ) %>%
    dplyr::mutate(
      abs_resid = abs(mpi_true - mpi_pred)
    ) %>%
    dplyr::filter(is.finite(abs_resid))
  
  if (nrow(cv_best) == 0) {
    stop(
      paste(
        "No matching refined XGBoost",
        "cross-validation predictions were found."
      )
    )
  }
  
  q_info <- calc_conformal_q(
    cv_best$abs_resid,
    alpha = alpha
  )
  
  list(
    q = q_info$q,
    k = q_info$k,
    n_residuals = q_info$n_residuals,
    cv_used = cv_best
  )
}

# ============================================================
# 5. Apply q to final test predictions
# ============================================================
# Match the final XGBoost test predictions to the selected configuration,
# apply the conformal q, and calculate coverage and threshold-uncertainty
# indicators without using the test outcomes to estimate q.
apply_q_to_test <- function(test_pred_file, best_row, q, threshold = 0.3333) {
  
  test_df <- read_csv(test_pred_file, show_col_types = FALSE)
  
  test_best <- test_df %>%
    filter(
      recipe == best_row$recipe[[1]],
      mtry == best_row$mtry[[1]],
      trees == best_row$trees[[1]],
      min_n == best_row$min_n[[1]],
      tree_depth == best_row$tree_depth[[1]],
      abs(learn_rate - best_row$learn_rate[[1]]) < 1e-12,
      abs(sample_size - best_row$sample_size[[1]]) < 1e-12
    ) %>%
    arrange(row_id)
  
  if (nrow(test_best) == 0) {
    stop("No matching rows found in test prediction file for the selected best model.")
  }
  
  
  int_df <- apply_conformal_interval(test_best$mpi_pred, q)
  
  out <- bind_cols(test_best, int_df) %>%
    coverage_summary(truth_col = "mpi_true", lo_col = "lo", hi_col = "hi", threshold = threshold)
  
  out
}

# ============================================================
# 6. Example usage
# ============================================================
# Use alpha = 0.10 for nominal 90% prediction-interval coverage and retain
# the dissertation poverty threshold of MPI = 0.3333.
alpha <- 0.10
threshold <- 0.3333

# Suppose your ultimate best row is already stored in:
# ultimate_best

# If ultimate best came from space-filling:
# q_obj <- get_q_space_filling(ultimate_best, cv_sf_file, alpha = alpha)

# If ultimate best came from refined stage:
# q_obj <- get_q_refined(ultimate_best, cv_refine_file, alpha = alpha)

# Then apply to the final test predictions:
# test_intervals <- apply_q_to_test(final_test_preds_file, ultimate_best, q_obj$q, threshold = threshold)

# ============================================================
# 7. Summaries
# ============================================================
# Summarise conformal performance overall and separately for Poor and
# Non-Poor test observations, including coverage, interval width and
# the proportion of intervals that span the poverty threshold.
summarise_conformal_results <- function(test_intervals) {
  
  overall <- test_intervals %>%
    summarise(
      n = n(),
      coverage = mean(covered),
      avg_width = mean(width),
      median_width = median(width),
      threshold_uncertain_rate = mean(threshold_uncertain)
    )
  
  by_group <- test_intervals %>%
    group_by(poverty_group) %>%
    summarise(
      n = n(),
      coverage = mean(covered),
      avg_width = mean(width),
      threshold_uncertain_rate = mean(threshold_uncertain),
      .groups = "drop"
    )
  
  uncertainty_summary <- test_intervals %>%
    summarise(
      threshold_uncertain_n = sum(threshold_uncertain),
      threshold_uncertain_rate = mean(threshold_uncertain)
    )
  
  list(
    overall = overall,
    by_group = by_group,
    uncertainty = uncertainty_summary
  )
}


# ============================================================
# 8. Plots
# ============================================================


# Plot the independent test predictions with their conformal intervals.
# Points are also classified according to whether the predicted poverty
# status agrees with the observed poverty status.
plot_conformal_intervals <- function(test_intervals, threshold = 0.3333, filename = "plot_conformal_intervals.png" ) {
  
  out_dir <- file.path(plotsdir, "conformalpred")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # classify each point
  test_intervals <- test_intervals %>%
    mutate(
      true_class = ifelse(mpi_true >= threshold, "poor", "non_poor"),
      pred_class = ifelse(mpi_pred >= threshold, "poor", "non_poor"),
      class_result = ifelse(true_class == pred_class, "Correct", "Misclassified")
    )
  
  # helper data frames for legend
  abline_df <- data.frame(
    intercept = 0,
    slope = 1,
    line_type = "Perfect prediction line"
  )
  
  vline_df <- data.frame(
    xintercept = threshold,
    line_type = "Observed MPI threshold"
  )
  
  hline_df <- data.frame(
    yintercept = threshold,
    line_type = "Predicted MPI threshold"
  )
  
  p <- ggplot(test_intervals, aes(x = mpi_true, y = mpi_pred)) +
    
    geom_errorbar(
      aes(ymin = lo, ymax = hi),
      colour = "steelblue",
      alpha = 0.6,
      width = 0
    ) +
    
    geom_point(
      aes(colour = class_result),
      size = 2,
      alpha = 0.85
    ) +
    
    geom_abline(
      data = abline_df,
      aes(intercept = intercept, slope = slope, linetype = line_type),
      colour = "grey40",
      linewidth = 0.7,
      inherit.aes = FALSE
    ) +
    
    geom_vline(
      data = vline_df,
      aes(xintercept = xintercept, linetype = line_type),
      colour = "grey40",
      linewidth = 1.1,
      inherit.aes = FALSE
    ) +
    
    geom_hline(
      data = hline_df,
      aes(yintercept = yintercept, linetype = line_type),
      colour = "grey40",
      linewidth = 1.1,
      inherit.aes = FALSE
    ) +
    
    scale_linetype_manual(
      name = "Reference lines",
      values = c(
        "Perfect prediction line" = "dashed",
        "Observed MPI threshold" = "dotted",
        "Predicted MPI threshold" = "dotted"
      )
    ) +
    
    scale_colour_manual(
      name = "Classification",
      values = c(
        "Correct" = "black",
        "Misclassified" = "red"
      )
    ) +
    
    labs(
      title = "Conformal prediction intervals on the test dataset",
      x = "Observed MPI",
      y = "Predicted MPI"
    ) +
    
    theme_minimal() +
    theme(
      legend.position = "bottom"
    )
  
  ggsave(
    filename = file.path(out_dir, filename),
    plot = p,
    width = 16,
    height = 12,
    dpi = 300
  )
  
  p
}

# Compare empirical prediction-interval coverage between the Poor and
# Non-Poor groups against the nominal coverage target.
plot_coverage_by_group <- function(by_group_df, target_coverage = 0.90, filename = "plot_coverage_by_group.png") {
  
  out_dir <- file.path(plotsdir, "conformalpred")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  p <- ggplot(by_group_df, aes(x = poverty_group, y = coverage)) +
    geom_col() +
    geom_hline(yintercept = target_coverage, linetype = "dashed") +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = "Coverage by poverty group",
      x = NULL,
      y = "Coverage rate"
    ) +
    theme_minimal()
  
  ggsave(
    filename = file.path(out_dir, filename),
    plot = p,
    width = 16,
    height = 12,
    dpi = 300
  )
  
  return(p)
}


# Examine whether conformal interval width changes across the observed MPI
# distribution and around the poverty threshold.
plot_interval_width <- function(test_intervals, threshold = 0.3333, filename = "plot_interval_width.png") {
  
  out_dir <- file.path(plotsdir, "conformalpred")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  p <- ggplot(test_intervals, aes(x = mpi_true, y = width)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE) +
    geom_vline(xintercept = threshold, linetype = "dashed") +
    labs(
      title = "Prediction interval width across the MPI distribution",
      x = "Observed MPI",
      y = "Prediction interval width"
    ) +
    theme_minimal()
  
  ggsave(
    filename = file.path(out_dir, filename),
    plot = p,
    width = 16,
    height = 12,
    dpi = 300
  )
  
  return(p)
}

# Apply the conformal workflow to the ultimately selected XGBoost model.
# The selected search stage determines which cross-validation residual file
# is used to estimate q before evaluating the independent test predictions.
## Analysis and Plot - xgboost
run_id <- "20260510_133304"
stg <- "space_filling"
recp <- "all_pred_notfm"
out_dir <- file.path(model_dir, "xgboost", paste0("run_", run_id))
ultimate_best_file <- file.path(out_dir, paste0("xgb_ULTIMATE_best_", stg,"_", recp, "_", run_id, ".csv"))
# Load the saved XGBoost configuration selected after comparing the
# space-filling and refined search stages.
ultimate_best <- read_csv(ultimate_best_file, show_col_types = FALSE)
final_sf_test_preds_file   <- file.path(out_dir, paste0("xgb_sf_FINAL_test_predictions_", run_id, ".csv"))


# Estimate q from the cross-validation predictions belonging to the selected
# XGBoost stage and apply it to the corresponding final test predictions.
if (ultimate_best$stage[[1]] == "space_filling") {
  
  q_obj <- get_q_from_xgb_fold_predictions(ultimate_best, cv_sf_file, alpha = alpha)
  test_intervals <- apply_q_to_test(final_sf_test_preds_file, ultimate_best, q_obj$q, threshold = threshold)
  print(paste0("using space_filling best model"))
  
  
} else if (ultimate_best$stage[[1]] == "refined") {
  
  q_obj <- get_q_refined(ultimate_best, cv_refine_file, alpha = alpha)
  test_intervals <- apply_q_to_test(final_test_preds_file, ultimate_best, q_obj$q, threshold = threshold)
  print(paste0("using refined grid-search best model"))
  
} else {
  stop("ultimate_best$stage must be either 'space_filling' or 'refined'")
}

# Produce the overall and poverty-group conformal summaries for XGBoost.
res_sum <- summarise_conformal_results(test_intervals)

print(q_obj$q)
print(q_obj$k)
print(q_obj$n_residuals)
print(res_sum)


plot_conformal_intervals(test_intervals)
plot_coverage_by_group(res_sum$by_group, target_coverage = 0.90)
plot_interval_width(test_intervals)




# Apply the same conformal prediction framework to the ultimately selected
# Neural Network model so that uncertainty is evaluated consistently.
## Analysis and Plot - Neural Network
# ============================================================
# 2. File paths
# ============================================================

# Specify the saved Neural Network cross-validation prediction file used
# to obtain the absolute out-of-fold residuals for conformal calibration.
nn_cv_file <- "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI/modeldir/neuralnetbetareg/run_20260512_064338/refined/preds/fold_preds_refined__all_pred_tfmwithlogcuberange__grid_52.csv"
#final_nn_test_preds_file <- "/home/ubuntu/Documents/study/MscDataScience/Dissertation/Rstudio/MPI/modeldir/neuralnetbetareg/run_20260512_064338/refined/Best_Results/refined_final_nn_test_predictions_20260512_064338.csv"


run_id <- "20260512_064338"
stg <- "refined"
recp <- "all_pred_tfmwithlogcuberange"
out_dir <- file.path(model_dir, "neuralnetbetareg", paste0("run_", run_id))
#ultimate_best_file <- file.path(out_dir, paste0("xgb_ULTIMATE_best_", stg,"_", recp, "_", run_id, ".csv"))
#ultimate_best <- read_csv(ultimate_best_file, show_col_types = FALSE)
#final_sf_test_preds_file   <- file.path(out_dir, paste0("xgb_sf_FINAL_test_predictions_", run_id, ".csv"))

#cv_df <- read_csv(cv_refine_file, show_col_types = FALSE)
#Test_results <- cv_df %>%
#  group_by(row_id) %>%
#  summarise(
#    n_rows = n(),
#    n_mpi_true = n_distinct(mpi_true),
#    min_mpi = min(mpi_true),
#    max_mpi = max(mpi_true),
#    .groups = "drop"
#  ) %>%
#  arrange(desc(n_mpi_true))

#View(Test_results)
## Base R method (Recommended)
#write.csv(Test_results, file.path(out_dir, paste0("writeTestResults.csv")), row.names = FALSE)


# Calculate the Neural Network conformal q from the absolute out-of-fold
# cross-validation residuals for the selected configuration.
get_q_from_fold_predictions_neuralnet <- function(cv_file, alpha = 0.10) {
  
  cv_df <- read_csv(cv_file, show_col_types = FALSE)
  
  cv_residuals <- cv_df %>%
    mutate(abs_resid = abs(mpi_true - mpi_pred)) %>%
    filter(is.finite(abs_resid))
  
  q_info <- calc_conformal_q(cv_residuals$abs_resid, alpha = alpha)
  
  list(
    q = q_info$q,
    k = q_info$k,
    n_residuals = q_info$n_residuals,
    cv_used = cv_residuals
  )
}

# Apply the Neural Network conformal q to the independent test predictions
# and calculate coverage, interval width and threshold uncertainty.
apply_q_to_test_nn <- function(test_pred_file, q, threshold = 0.3333) {
  
  test_df <- read_csv(test_pred_file, show_col_types = FALSE)
  
  required_cols <- c("mpi_true", "mpi_pred")
  
  missing_cols <- setdiff(required_cols, names(test_df))
  
  if (length(missing_cols) > 0) {
    stop(
      paste(
        "The test prediction file is missing required columns:",
        paste(missing_cols, collapse = ", ")
      )
    )
  }
  
  test_best <- test_df %>%
    filter(
      is.finite(mpi_true),
      is.finite(mpi_pred)
    )
  
  if (nrow(test_best) == 0) {
    stop("No valid test predictions found.")
  }
  
  int_df <- apply_conformal_interval(test_best$mpi_pred, q)
  
  out <- bind_cols(test_best, int_df) %>%
    coverage_summary(
      truth_col = "mpi_true",
      lo_col = "lo",
      hi_col = "hi",
      threshold = threshold
    )
  
  out
}


# Estimate the Neural Network conformal q using cross-validation residuals
# only, then apply the resulting interval width to the independent test set.
q_obj_nn <- get_q_from_fold_predictions_neuralnet(nn_cv_file, alpha = 0.10)
test_intervals_nn <- apply_q_to_test_nn(
  test_pred_file = final_nn_test_preds_file,
  q = q_obj_nn$q,
  threshold = 0.3333
)
# Summarise Neural Network prediction-interval performance overall and by
# observed poverty group.
res_sum_nn <- summarise_conformal_results(test_intervals_nn)

# if (ultimate_best$stage[[1]] == "space_filling") {

#  q_obj_nn <- get_q_from_fold_predictions_neuralnet(nn_cv_file, alpha = 0.10)
#  test_intervals_nn <- apply_q_to_test_nn(
#    test_pred_file = final_nn_test_preds_file,
#    q = q_obj_nn$q,
#    threshold = 0.3333
#  )
#  print(paste0("using space_filling best model"))


#q_obj <- get_q_refined(ultimate_best, cv_refine_file, alpha = alpha)
#test_intervals <- apply_q_to_test(final_test_preds_file, ultimate_best, q_obj$q, threshold = threshold)
#print(paste0("using refined grid-search best model"))

#} else if (ultimate_best$stage[[1]] == "refined") {

#  q_obj <- get_q_refined(ultimate_best, cv_refine_file, alpha = alpha)
#  test_intervals <- apply_q_to_test(final_test_preds_file, ultimate_best, q_obj$q, threshold = threshold)
#  print(paste0("using refined grid-search best model"))

#} else {
#  stop("ultimate_best$stage must be either 'space_filling' or 'refined'")
#}


# Display the conformal calibration quantities and final Neural Network
# coverage summaries used in the dissertation analysis.
print(q_obj_nn$q)
print(q_obj_nn$k)
print(q_obj_nn$n_residuals)
print(q_obj_nn)
print(res_sum_nn)


# Generate the Neural Network conformal interval, coverage and interval-width
# figures using the same plotting functions applied to XGBoost.
plot_conformal_intervals(test_intervals_nn, filename = "plot_conformal_intervals_nn.png")
plot_coverage_by_group(res_sum_nn$by_group, target_coverage = 0.90, filename = "plot_coverage_by_group_nn.png")
plot_interval_width(test_intervals_nn, filename = "plot_interval_width_nn.png")





# ===============================================================
# SHAP EXTENSION (shapviz) — OUT-OF-FOLD SHAP (10x5 CV)
# ===============================================================
library(shapviz)
library(dplyr)

# Confirm that the cross-validation folds, XGBoost preprocessing recipes,
# selected final configuration and training dataset are available.
stopifnot(exists("ntl_vfolds_cv"))
stopifnot(exists("preproc.xgb"))
stopifnot(exists("ultimate_best"))
stopifnot(exists("ntl_train"))

# Retrieve the preprocessing recipe and hyperparameters belonging to the
# ultimately selected XGBoost configuration.
rec_name_ultimate <- ultimate_best$recipe
rec_ultimate <- preproc.xgb[[rec_name_ultimate]]

mtry_u        <- as.integer(ultimate_best$mtry)
trees_u       <- as.integer(ultimate_best$trees)
min_n_u       <- as.integer(ultimate_best$min_n)
tree_depth_u  <- as.integer(ultimate_best$tree_depth)
learn_rate_u  <- as.numeric(ultimate_best$learn_rate)
sample_size_u <- as.numeric(ultimate_best$sample_size)

# Create a list to collect the out-of-fold SHAP contributions from each
# cross-validation fold.
oof_shap_list <- list()

cat("\n OUT-OF-FOLD SHAP: refitting winning config across",
    length(ntl_vfolds_cv$splits), "folds\n")

# Refit the selected XGBoost configuration independently within each
# cross-validation fold and calculate SHAP values only for the held-out
# assessment observations.
for (i in seq_along(ntl_vfolds_cv$splits)) {
  
  sp <- ntl_vfolds_cv$splits[[i]]
  tr <- rsample::analysis(sp)
  va <- rsample::assessment(sp)
  
  # Prepare the selected recipe using the analysis portion of the fold only,
  # then apply the fitted preprocessing steps to both analysis and assessment data.
  rec_p <- prep(rec_ultimate, training = tr, retain = TRUE)
  tr_b  <- bake(rec_p, new_data = tr)
  va_b  <- bake(rec_p, new_data = va)
  
  # Separate the outcome from the processed predictor matrices used to fit
  # the fold-specific XGBoost model and calculate validation SHAP values.
  y_tr <- tr_b$mpi
  X_tr <- tr_b %>% select(-mpi) %>% as.data.frame(); X_tr[] <- lapply(X_tr, as.numeric)
  X_va <- va_b %>% select(-mpi) %>% as.data.frame(); X_va[] <- lapply(X_va, as.numeric)
  
  n_pred_fold <- ncol(X_tr)
  dtr <- xgb.DMatrix(data = as.matrix(X_tr), label = y_tr)
  
  # Use a fixed seed to reduce run-to-run variation in each XGBoost fold fit
  fold_seed <- 124L
  
  # Reconstruct the selected XGBoost hyperparameters for the current fold.
  params_fold <- list(
    objective = "reg:squarederror",
    eta = learn_rate_u, max_depth = tree_depth_u, min_child_weight = min_n_u,
    subsample = sample_size_u, colsample_bytree = mtry_u / max(1, n_pred_fold),
    gamma = 0, nthread = 1, verbosity = 0, seed = fold_seed
  )
  
  set.seed(fold_seed)
  booster_fold <- xgboost::xgb.train(params = params_fold, data = dtr, nrounds = trees_u, verbose = 0)
  
  # Calculate TreeSHAP contributions for the held-out observations. The bias
  # contribution is removed because the analysis focuses on predictor effects.
  X_va_matrix <- as.matrix(X_va)
  shap_fold <- predict(booster_fold, X_va_matrix, predcontrib = TRUE)
  #shap_fold <- shap_fold[, seq_len(ncol(shap_fold) - 1), drop = FALSE]  # drop bias, whatever it's named
  
  #bias_col <- which(colnames(shap_fold) == "BIAS")
  #if (length(bias_col) == 1) shap_fold <- shap_fold[, -bias_col, drop = FALSE]
  
  
  bias_col <- which(colnames(shap_fold) %in% c("BIAS", "(Intercept)"))
  if (length(bias_col) == 1) {
    shap_fold <- shap_fold[, -bias_col, drop = FALSE]
  } else {
    # fallback: drop last column positionally if name-based check fails
    shap_fold <- shap_fold[, seq_len(ncol(shap_fold) - 1), drop = FALSE]
  }
  
  
  # Retain the fold identifiers and original training-row position so the
  # out-of-fold SHAP contributions can be traced back to each observation.
  orig_row <- match(rownames(va), rownames(ntl_train))
  resample_id <- if ("id" %in% names(ntl_vfolds_cv)) as.character(ntl_vfolds_cv$id[[i]]) else paste0("Fold", i)
  repeat_id   <- if ("id2" %in% names(ntl_vfolds_cv)) as.character(ntl_vfolds_cv$id2[[i]]) else NA_character_
  
  contrib <- as_tibble(shap_fold) %>%
    mutate(fold = i, id = resample_id, id2 = repeat_id, orig_row = orig_row)
  
  oof_shap_list[[i]] <- contrib
  cat("  Fold", i, "/", length(ntl_vfolds_cv$splits), "done (n_val =", nrow(X_va), ")\n")
}

# Combine all out-of-fold SHAP contributions and identify the predictor columns.
oof_shap_all <- bind_rows(oof_shap_list)
feat_cols <- setdiff(names(oof_shap_all), c("fold", "id", "id2", "orig_row"))

# Calculate mean absolute SHAP values within each fold to obtain a fold-level
# measure of global predictor importance.
fold_level_importance <- oof_shap_all %>%
  group_by(fold) %>%
  summarise(across(all_of(feat_cols), ~ mean(abs(.x), na.rm = TRUE)), .groups = "drop")

# Average the fold-level importance values across all CV folds and calculate
# their standard errors to summarise the stability of predictor importance.
cv_shap_summary <- fold_level_importance %>%
  select(-fold) %>%
  summarise(across(everything(), list(mean = ~mean(.x), se = ~sd(.x) / sqrt(length(.x))))) %>%
  tidyr::pivot_longer(everything(), names_to = c("variable", ".value"),
                      names_pattern = "(.*)_(mean|se)") %>%
  arrange(desc(mean))

# Save the cross-validation SHAP importance summary for subsequent reporting.
write_csv(cv_shap_summary, file.path(images_dir, paste0("xgb_CV_shap_importance_", run_id, ".csv")))

# Plot the 15 predictors with the largest mean absolute out-of-fold SHAP
# values together with their standard errors across the CV folds.
p_cv_shap_bar <- cv_shap_summary %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(variable, mean), y = mean)) +
  geom_col(fill = "steelblue", alpha = 0.85) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.25) +
  coord_flip() +
  labs(title = "Global Feature Importance — Out-of-Fold SHAP (10x5 CV)",
       subtitle = paste0("Recipe: ", rec_name_ultimate,
                         " | Mean |SHAP| ± SE across ", length(ntl_vfolds_cv$splits), " folds"),
       x = "Predictor Variables", y = "Mean |SHAP value| (out-of-fold)") +
  theme_minimal()

ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_shap_cv_bar_", run_id, ".png")),
       p_cv_shap_bar, width = 12, height = 8, dpi = 300)

# Save the complete out-of-fold SHAP contributions and summary objects so
# they can be reused without repeating the full cross-validation procedure.
saveRDS(list(oof_shap_all = oof_shap_all, fold_level_importance = fold_level_importance,
             cv_shap_summary = cv_shap_summary),
        file.path(images_dir, paste0("xgb_CV_shap_objects_", run_id, ".rds")))

cat("\n OUT-OF-FOLD SHAP COMPLETE\n")

# ===============================================================
# SHAP EXTENSION (shapviz) — GLOBAL EXPLANATION, TEST SET
# ===============================================================
# library(shapviz)

# Confirm that the final fitted XGBoost model, processed test predictors and
# selected configuration are available before calculating test-set SHAP values.
stopifnot(exists("modelfitxgb"))
stopifnot(exists("X_test_ultimate"))
stopifnot(exists("ultimate_best"))

X_test_matrix <- as.matrix(X_test_ultimate)

# Calculate TreeSHAP contributions for the independent test observations
# directly from the final fitted XGBoost booster.
shap_contrib <- predict(modelfitxgb, X_test_matrix, predcontrib = TRUE)


# Remove the additional bias contribution returned by predcontrib so that
# only predictor-specific SHAP values are retained.
bias_col <- which(colnames(shap_contrib) %in% c("BIAS", "(Intercept)"))
if (length(bias_col) == 1) {
  shap_contrib <- shap_contrib[, -bias_col, drop = FALSE]
} else {
  # fallback: drop last column positionally if name-based check fails
  shap_contrib <- shap_contrib[, seq_len(ncol(shap_contrib) - 1), drop = FALSE]
}

#  bias_col <- which(colnames(shap_contrib) == "(Intercept)")
#  if (length(bias_col) == 1) shap_contrib <- shap_contrib[, -bias_col, drop = FALSE]

# Organise the test-set SHAP contributions and corresponding predictor values
# in a shapviz object for subsequent visualisation.
shv_test <- shapviz(shap_contrib, X = X_test_ultimate)

# Calculate global test-set predictor importance using the mean absolute SHAP
# value for each predictor and rank the predictors from largest to smallest.
shap_importance <- sort(colMeans(abs(shap_contrib)), decreasing = TRUE)

write_csv(
  tibble(variable = names(shap_importance), mean_abs_shap = as.numeric(shap_importance)),
  file.path(images_dir, paste0("xgb_ULTIMATE_shap_importance_", run_id, ".csv"))
)

# Create a SHAP beeswarm plot to show both the magnitude and direction of
# predictor contributions across the independent test observations.
p_shap_summary <- sv_importance(shv_test, kind = "beeswarm")
ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_shap_summary_", run_id, ".png")),
       p_shap_summary, width = 10, height = 8, dpi = 300)

# Plot the 15 predictors with the largest mean absolute SHAP values on the
# independent test dataset.
p_shap_bar <- sv_importance(shv_test, kind = "bar", max_display = 15) +
  ggtitle("Global Feature Importance — SHAP (Mean |SHAP value|)",
          subtitle = paste0("Stage: ", ultimate_best$stage, " | Recipe: ", ultimate_best$recipe))
ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_shap_bar_", run_id, ".png")),
       p_shap_bar, width = 12, height = 8, dpi = 300)

# Generate dependence plots for the six most influential predictors to examine
# how predictor values are associated with their SHAP contributions.
top_feats <- names(shap_importance)[seq_len(min(6, length(shap_importance)))]
for (feat in top_feats) {
  p_dep <- sv_dependence(shv_test, v = feat) + ggtitle(paste0("SHAP Dependence: ", feat))
  ggsave(file.path(plotsdir, "xgboost", paste0("final_model_xgb_shap_dependence_", feat, "_", run_id, ".png")),
         p_dep, width = 8, height = 6, dpi = 300)
}

# Save the raw test-set SHAP contributions, global importance values and
# shapviz object for subsequent analysis without recalculating SHAP values.
saveRDS(
  list(shap_contrib = shap_contrib, shap_importance = shap_importance, shv_test = shv_test),
  file.path(images_dir, paste0("xgb_ULTIMATE_shap_objects_", run_id, ".rds"))
)

cat("\n SHAP GLOBAL EXPLANATION (TEST SET) COMPLETE\n")


# Compare the global SHAP rankings obtained from cross-validation with those
# obtained from the independent test dataset.
test_shap_rds <- file.path(images_dir, paste0("xgb_ULTIMATE_shap_objects_", run_id, ".rds"))

if (file.exists(test_shap_rds)) {
  test_shap_objects <- readRDS(test_shap_rds)
  shap_importance <- test_shap_objects$shap_importance
  
  # Rank predictors according to their mean absolute SHAP values on the
  # independent test dataset.
  test_rank <- tibble(variable = names(shap_importance), mean_abs_shap_test = as.numeric(shap_importance)) %>%
    arrange(desc(mean_abs_shap_test)) %>% mutate(rank_test = row_number())
  
  # Rank predictors using the mean absolute out-of-fold SHAP values obtained
  # from cross-validation.
  cv_rank <- cv_shap_summary %>% select(variable, mean_abs_shap_cv = mean) %>%
    arrange(desc(mean_abs_shap_cv)) %>% mutate(rank_cv = row_number())
  
  # Join both rankings by predictor and save the comparison for reporting.
  rank_compare <- inner_join(cv_rank, test_rank, by = "variable") %>% arrange(rank_cv)
  write_csv(rank_compare, file.path(out_dir, paste0("xgb_shap_cv_vs_test_ranking_", run_id, ".csv")))
  print(rank_compare)
} else {
  cat(" Test-set SHAP RDS not found — run Part 1 first.\n")
}
