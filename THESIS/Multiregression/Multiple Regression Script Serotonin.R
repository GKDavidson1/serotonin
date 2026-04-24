# ============================================================
# Run multiple regression on every combination of chosen predictors
# and return actual data-frame tables
# ============================================================

run_all_regressions <- function(data, response, predictors) {
  
  if (!is.data.frame(data)) {
    stop("'data' must be a data frame.")
  }
  
  if (!response %in% names(data)) {
    stop("Response variable not found in data.")
  }
  
  if (missing(predictors) || length(predictors) == 0) {
    stop("Please provide a vector of predictor names.")
  }
  
  if (!all(predictors %in% names(data))) {
    missing_vars <- predictors[!predictors %in% names(data)]
    stop(paste("These predictors were not found in data:",
               paste(missing_vars, collapse = ", ")))
  }
  
  data <- data[, c(response, predictors), drop = FALSE]
  data <- na.omit(data)
  
  model_results_list <- list()
  predictor_results_list <- list()
  counter <- 1
  
  for (k in 1:length(predictors)) {
    predictor_combos <- combn(predictors, k, simplify = FALSE)
    
    for (combo in predictor_combos) {
      formula_txt <- paste(response, "~", paste(combo, collapse = " + "))
      model_formula <- as.formula(formula_txt)
      
      model <- lm(model_formula, data = data)
      summary_model <- summary(model)
      coef_table <- as.data.frame(summary_model$coefficients)
      
      names(coef_table) <- c("estimate", "std_error", "t_value", "p_value")
      coef_table$term <- rownames(coef_table)
      rownames(coef_table) <- NULL
      
      fstat <- summary_model$fstatistic
      model_p_value <- pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)
      
      model_results_list[[counter]] <- data.frame(
        model_id = counter,
        formula = formula_txt,
        n_predictors = length(combo),
        predictors = paste(combo, collapse = ", "),
        n_obs = nobs(model),
        r_squared = summary_model$r.squared,
        adj_r_squared = summary_model$adj.r.squared,
        aic = AIC(model),
        bic = BIC(model),
        model_p_value = model_p_value,
        stringsAsFactors = FALSE
      )
      
      coef_table$model_id <- counter
      coef_table$formula <- formula_txt
      coef_table$n_predictors <- length(combo)
      
      predictor_results_list[[counter]] <- coef_table[
        ,
        c("model_id", "formula", "n_predictors",
          "term", "estimate", "std_error", "t_value", "p_value")
      ]
      
      counter <- counter + 1
    }
  }
  
  model_results <- do.call(rbind, model_results_list)
  predictor_results <- do.call(rbind, predictor_results_list)
  
  model_results <- model_results[order(-model_results$adj_r_squared), ]
  rownames(model_results) <- NULL
  
  predictor_results <- predictor_results[
    order(predictor_results$model_id, predictor_results$term),
  ]
  rownames(predictor_results) <- NULL
  
  return(list(
    model_results = model_results,
    predictor_results = predictor_results
  ))
}

# ============================================================
# Example usage
# ============================================================

df <- Stats_rundown_THESIS

response_var <- "best_difference_vector_cosine"

my_predictors <- c(
  "Stim",
  "Auto",
  "BOLD"
)

results <- run_all_regressions(
  data = df,
  response = response_var,
  predictors = my_predictors
)

# These are now actual data frames:
model_tableC <- results$model_results
predictor_tableC <- results$predictor_results

# Open as tables in RStudio:
View(model_tableC)
View(predictor_tableC)