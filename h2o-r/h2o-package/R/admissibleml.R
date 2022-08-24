h2o.calculate_fairness_metrics <- function(model, frame, protected_cols, reference, favorable_class) {
  model_id <- if (is.character(model)) model else model@model_id
  if (is.null(h2o.keyof(frame)))
    head(frame, n = 1) # force evaluation of frame (in case it was manipulated before (e.g. subset))
  list_to_string <- function(entries) paste0("[\"", paste0(entries, collapse = "\", \"") ,"\"]")
  expr <- sprintf("(fairnessMetrics %s %s %s %s \"%s\")",
                  model_id ,
                  h2o.keyof(frame),
                  list_to_string(protected_cols),
                  list_to_string(reference),
                  favorable_class)
  lst <- h2o.rapids(expr)
  res <- list()
  for (i in seq_along(lst$map_keys$string)) {
    res[[lst$map_keys$string[[i]]]] <- as.data.frame(h2o.getFrame(lst$frames[[i]]$key$name))
  }
  res[["overview"]] <- res[["overview"]][res[["overview"]]$total > 0, ]
  res
}


#' Get disparate analysis
#'
#' @param
#'
#' @export
.get_corrected_variance <- function(fm) {
  cv <-
    var(fm[["accuracy"]]) - mean((fm[["accuracy"]] * (1 - fm[["accuracy"]])) /
                                   fm[["total"]])
  cv
}

h2o.get_disparate_analysis <-
  function(models,
           newdata,
           sensitive_features,
           reference,
           favorable_class) {
    models_info <- .process_models_or_automl(
        models,
        newdata,
        check_x_y_consistency = FALSE,
        require_multiple_models = TRUE
      )
    y <- models_info$y
    is_classification <- is.factor(newdata[[y]])

    return(cbind(
      h2o:::.create_leaderboard(models_info, newdata, top_n = Inf),
      t(sapply(models, function(model) {
        capture.output({
          dm <-
            h2o.calculate_fairness_metrics(
              model = model,
              frame = newdata,
              protected_cols = sensitive_features,
              reference = reference,
              favorable_class = favorable_class
            )$overview
        })
        return(
          c(
            var = var(dm[["accuracy"]]),
            cvar = .get_corrected_variance(dm),
            air_min = min(dm$AIR_selectedRatio, na.rm = TRUE),
            air_mean = mean(dm$AIR_selectedRatio, na.rm = TRUE),
            air_median = median(dm$AIR_selectedRatio, na.rm = TRUE),
            air_max = max(dm$AIR_selectedRatio, na.rm = TRUE),
            cair = weighted.mean(dm$AIR_selectedRatio, dm$relativeSize, na.rm = TRUE),
            nair_min = min(dm$nAIR_selectedRatio, na.rm = TRUE),
            nair_mean = mean(dm$nAIR_selectedRatio, na.rm = TRUE),
            nair_median = median(dm$nAIR_selectedRatio, na.rm = TRUE),
            nair_max = max(dm$nAIR_selectedRatio, na.rm = TRUE),
            cnair = weighted.mean(dm$nAIR_selectedRatio, dm$relativeSize, na.rm = TRUE),
            `p.value_min` = min(dm[["p.value"]], na.rm = TRUE),
            `p.value_median` = median(dm[["p.value"]], na.rm = TRUE),
            `p.value_mean` = mean(dm[["p.value"]], na.rm = TRUE),
            `p.value_max` = max(dm[["p.value"]], na.rm = TRUE)
          )
        )
      }))
    ))
  }


infogram_grid <-
  function(ig, model_fun,
           train,
           test,
           x,
           y,
           protected_columns,
           reference,
           favorable_class) {

    score <- as.data.frame(ig@admissible_score)
    score <- score[order(score$safety_index, decreasing = TRUE), ]
    xs <- lapply(1:nrow(score), function(n)
      score$column[1:n])
    models <-
      do.call(c, lapply(xs, function(cols)
        model_fun(
          x = cols,
          y = y,
          training_frame = train
        )))
    h2o.get_disparate_analysis(models, test, protected_columns, reference, favorable_class = favorable_class)
  }