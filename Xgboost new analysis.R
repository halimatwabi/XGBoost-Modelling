# =============================================================================
#  Predicting monthly malaria incidence in Malawi with XGBoost
# =============================================================================
#
#  What this script does, step by step
#    1. Settings ............ file names, training/test years, output folder
#    2. Data ................ cases + climate + population -> incidence per 1,000
#    3. Predictors .......... climate lags, season, previous incidence
#    4. Train / test split .. train 2015-2022, test Jan 2023 - Jun 2025
#    5. Tuning .............. choose XGBoost settings using the training years only
#    6. Final model ......... fit on all training years, predict the test years
#    7. Evaluation .......... compare with a seasonal baseline (MAE, R2, MAPE, ...)
#    8. Prediction plots .... observed (solid) vs predicted (dashed)
#    9. SHAP ................ importance, beeswarm, dependence, by district
#
#  The key idea: the model never sees the test years while it is being fitted
#  or tuned, so the test results show how well it predicts the future.
#
#  Packages: install.packages(c("dplyr", "tidyr", "ggplot2", "readxl",
#                               "xgboost", "ggbeeswarm"))
# =============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(readxl)
library(xgboost)
# ggbeeswarm is optional: it draws nicer beeswarms; without it, points are jittered


# =============================================================================
# 1. SETTINGS  (the only part you normally need to change)
# =============================================================================
# setwd("C:/Users/user/Dropbox/PC (2)/Documents/Climate and Health/Data to use")

CASES_CLIMATE_FILE <- "climate_malaria_data.csv"  # District, date, malaria.cases, precipe, maxtemp, mintemp, relhum
POP_FILE           <- "popn.xlsx"                 # population 2018-2025 (one column per year)
POP_1517_FILE      <- "pop1517.xlsx"              # population 2015-2017 (District, year, pop)

# 2015-2017 population:
#   FALSE = estimate it from each district's 2018-2019 growth rate
#   TRUE  = read POP_1517_FILE. NOTE: in that file the values labelled Mchinji, Dedza,
#           Ntcheu, Lilongwe City and Lilongwe look shifted by one row. Check them first.
USE_POP_1517 <- FALSE

TRAIN_END <- as.Date("2022-12-01")   # last training month; everything after is the test period
CV_YEARS  <- 2020:2022               # years used, one at a time, to tune the model
OUT       <- "xgboost_outputs"       # all tables and figures are saved here

dir.create(OUT, showWarnings = FALSE)
dir.create(file.path(OUT, "beeswarm_by_district"), showWarnings = FALSE)
set.seed(123)


# =============================================================================
# 2. DATA: cases, climate and population  ->  incidence per 1,000 per month
# =============================================================================

# One spelling per district. Cities go into their districts; Mzimba North,
# Mzimba South and Mzuzu City become "Mzimba".
fix_district <- function(x) recode(trimws(x),
                                   "Mzimba-North" = "Mzimba", "Mzimba-South" = "Mzimba",
                                   "Mzimba North" = "Mzimba", "Mzimba South" = "Mzimba", "Mzuzu City" = "Mzimba",
                                   "Lilongwe City" = "Lilongwe", "Blantyre City" = "Blantyre", "Zomba City" = "Zomba",
                                   "Nkhata Bay" = "Nkhata-Bay", "Nkata bay" = "Nkhata-Bay")

# --- cases and climate ---------------------------------------------------------
cases <- read.csv(CASES_CLIMATE_FILE) %>%
  rename(cases = malaria.cases, temp = maxtemp) %>%
  mutate(District = fix_district(District),
         dt = as.Date(paste("01", date), format = "%d %B %Y")) %>%     # "January 2015" -> date
  group_by(District, dt) %>%                                          # adds Mzimba North + South
  summarise(cases = sum(cases),
            across(c(precipe, temp, mintemp, relhum), mean), .groups = "drop")
stopifnot(!anyNA(cases$dt))

# --- population ---------------------------------------------------------------
pop <- read_excel(POP_FILE) %>%
  pivot_longer(-district, names_to = "year", values_to = "pop") %>%
  mutate(District = fix_district(district), year = as.integer(year)) %>%
  group_by(District, year) %>% summarise(pop = sum(pop), .groups = "drop")

if (USE_POP_1517) {
  pop_early <- read_excel(POP_1517_FILE) %>%
    mutate(District = fix_district(District), year = as.integer(year)) %>%
    group_by(District, year) %>% summarise(pop = sum(pop), .groups = "drop")
} else {
  pop_early <- pop %>%
    group_by(District) %>%
    summarise(growth = pop[year == 2019] / pop[year == 2018], pop2018 = pop[year == 2018]) %>%
    crossing(year = 2015:2017) %>%
    mutate(pop = pop2018 / growth^(2018 - year)) %>%
    select(District, year, pop)
}
pop <- bind_rows(pop_early, pop)

# --- join and compute incidence ------------------------------------------------
dat <- cases %>%
  complete(District, dt = seq(min(dt), max(dt), by = "month")) %>%   # fills any missing month with NA
  mutate(year = as.integer(format(dt, "%Y")), month = as.integer(format(dt, "%m"))) %>%
  left_join(pop, by = c("District", "year")) %>%
  mutate(inc = 1000 * cases / pop) %>%                              # cases per 1,000 per month
  arrange(District, dt)


# =============================================================================
# 3. PREDICTORS
# =============================================================================
#   Climate:            each variable this month and 1, 2, 3 months earlier,
#                       plus rainfall summed over the previous 3 months
#   Season:             sine and cosine of the month (a smooth yearly cycle)
#   Previous incidence: incidence 1 and 2 months earlier (log scale)
#@Yambanso, you can start from here if you have all the data

dat <- dat %>%
  group_by(District) %>%
  mutate(
    across(c(precipe, temp, mintemp, relhum),
           list(lag1 = ~lag(.x, 1), lag2 = ~lag(.x, 2), lag3 = ~lag(.x, 3)),
           .names = "{.col}_{.fn}"),
    rain_3m      = precipe_lag1 + precipe_lag2 + precipe_lag3,
    log_inc_lag1 = log(lag(inc, 1)),
    log_inc_lag2 = log(lag(inc, 2))
  ) %>%
  ungroup() %>%
  mutate(sin12 = sin(2 * pi * month / 12),
         cos12 = cos(2 * pi * month / 12)) %>%
  drop_na()   # removes the first 3 months of each district and Mzimba (no climate data)

season_vars  <- c("sin12", "cos12")
climate_vars <- c("precipe", "temp", "mintemp", "relhum",
                  paste0(rep(c("precipe", "temp", "mintemp", "relhum"), each = 3), "_lag", 1:3),
                  "rain_3m")
previous_vars <- c("prev_inc_1", "prev_inc_2")   # created in step 4 (relative to each district's usual level)

# Three versions of the model, to see what climate adds
feature_sets <- list(
  "Season + climate"                      = c(season_vars, climate_vars),
  "Season + previous incidence"           = c(season_vars, previous_vars),
  "Season + climate + previous incidence" = c(season_vars, climate_vars, previous_vars)
)
MAIN_SET <- "Season + climate + previous incidence"   # the model used for plots and SHAP

# Readable names for plots
nice_name <- c(
  sin12 = "Season (sine)", cos12 = "Season (cosine)",
  precipe = "Rainfall", precipe_lag1 = "Rainfall, lag 1", precipe_lag2 = "Rainfall, lag 2",
  precipe_lag3 = "Rainfall, lag 3", rain_3m = "Rainfall, sum of lags 1-3",
  temp = "Max temperature", temp_lag1 = "Max temperature, lag 1", temp_lag2 = "Max temperature, lag 2",
  temp_lag3 = "Max temperature, lag 3",
  mintemp = "Min temperature", mintemp_lag1 = "Min temperature, lag 1",
  mintemp_lag2 = "Min temperature, lag 2", mintemp_lag3 = "Min temperature, lag 3",
  relhum = "Relative humidity", relhum_lag1 = "Relative humidity, lag 1",
  relhum_lag2 = "Relative humidity, lag 2", relhum_lag3 = "Relative humidity, lag 3",
  prev_inc_1 = "Previous incidence, lag 1", prev_inc_2 = "Previous incidence, lag 2")

group_of <- function(v) case_when(
  grepl("^prev_inc", v)    ~ "Previous incidence",
  v %in% season_vars       ~ "Season",
  grepl("^precipe|^rain", v) ~ "Rainfall",
  grepl("^mintemp", v)     ~ "Min temperature",
  grepl("^temp", v)        ~ "Max temperature",
  grepl("^relhum", v)      ~ "Relative humidity")


# =============================================================================
# 4. TRAIN / TEST SPLIT  (+ helper that prepares a train/test pair)
# =============================================================================
#  Each district has its own usual level of malaria. We take that level from the
#  TRAINING data only and:
#    - give it to XGBoost as a starting point ("base_margin"), together with
#      population, so the trees only learn departures from the usual level;
#    - express previous incidence relative to it, so 30 per 1,000 means "high"
#      in a low-burden district and "low" in a high-burden one.

train_all <- filter(dat, dt <= TRAIN_END)
test_all  <- filter(dat, dt >  TRAIN_END)
cat("Training:", format(min(train_all$dt)), "to", format(max(train_all$dt)), "-", nrow(train_all), "district-months\n")
cat("Test:    ", format(min(test_all$dt)),  "to", format(max(test_all$dt)),  "-", nrow(test_all),  "district-months\n")

prepare <- function(train, test) {
  level <- train %>% group_by(District) %>% summarise(level = mean(log(inc)), .groups = "drop")
  add <- function(d) d %>%
    left_join(level, by = "District") %>%
    mutate(prev_inc_1 = log_inc_lag1 - level,
           prev_inc_2 = log_inc_lag2 - level,
           margin     = level + log(pop / 1000))    # log(expected cases) before the trees
  list(train = add(train), test = add(test))
}


# =============================================================================
# 5. XGBOOST: fit, predict, and tune
# =============================================================================
#  Outcome: monthly case counts with a Poisson objective. Predictions are turned
#  back into incidence per 1,000 by dividing by population.

fit_xgb <- function(train, vars, s) {
  dtrain <- xgb.DMatrix(as.matrix(train[, vars]), label = train$cases, base_margin = train$margin)
  set.seed(123)
  xgb.train(
    params = list(objective = "count:poisson",
                  max_depth = s$max_depth,               # how complex each tree can be
                  eta = s$eta,                           # learning rate
                  min_child_weight = s$min_child_weight, # minimum data in a leaf
                  subsample = 0.8, colsample_bytree = 0.8, nthread = 2),
    data = dtrain, nrounds = s$nrounds, verbose = 0)
}

predict_inc <- function(model, newdata, vars) {
  dnew <- xgb.DMatrix(as.matrix(newdata[, vars]), base_margin = newdata$margin)
  1000 * predict(model, dnew) / newdata$pop               # predicted incidence per 1,000
}

# Settings to try (24 combinations)
grid <- expand.grid(max_depth = c(2, 3, 5), eta = c(0.03, 0.1),
                    nrounds = c(100, 300), min_child_weight = c(1, 10))

# Rolling-origin cross-validation: for each year in CV_YEARS, train on the years
# before it and predict it. Score = error on the log scale (fair across districts).
tune_xgb <- function(vars) {
  grid$cv_error <- sapply(seq_len(nrow(grid)), function(i) {
    mean(sapply(CV_YEARS, function(y) {
      s <- prepare(filter(train_all, year < y), filter(train_all, year == y))
      p <- predict_inc(fit_xgb(s$train, vars, grid[i, ]), s$test, vars)
      sqrt(mean((log(p) - log(s$test$inc))^2))
    }))
  })
  grid
}


# =============================================================================
# 6. FINAL MODELS: tune, fit on all training years, predict the test years
# =============================================================================
split <- prepare(train_all, test_all)
models <- list(); best_settings <- list(); preds <- list()

for (fs in names(feature_sets)) {
  vars <- feature_sets[[fs]]
  cat("Tuning:", fs, "\n")
  tuning <- tune_xgb(vars)
  best <- tuning[which.min(tuning$cv_error), ]
  best_settings[[fs]] <- best
  models[[fs]] <- fit_xgb(split$train, vars, best)
  preds[[fs]] <- split$test %>% select(District, dt, year, month, pop, inc) %>%
    mutate(features = fs, pred = predict_inc(models[[fs]], split$test, vars))
}

# Seasonal baseline: each district's average incidence for that calendar month in the training years
baseline <- split$train %>% group_by(District, month) %>% summarise(pred = mean(inc), .groups = "drop")
preds[["Seasonal baseline"]] <- split$test %>% select(District, dt, year, month, pop, inc) %>%
  left_join(baseline, by = c("District", "month")) %>% mutate(features = "Seasonal baseline")

preds <- bind_rows(preds)
write.csv(preds, file.path(OUT, "test_predictions.csv"), row.names = FALSE)
write.csv(bind_rows(best_settings, .id = "features"), file.path(OUT, "tuned_settings.csv"), row.names = FALSE)
invisible(xgb.save(models[[MAIN_SET]], file.path(OUT, "xgb_main_model.json")))


# =============================================================================
# 7. EVALUATION ON THE TEST YEARS
# =============================================================================
#  MAE   mean absolute error (cases per 1,000 per month). Lower is better.
#  RMSE  like MAE but penalises large errors more.
#  Bias  average over (+) or under (-) prediction.
#  R2    predictive R2 (Nash-Sutcliffe efficiency) = 1 - SSE/SST.
#        1 = perfect, 0 = no better than the test-period average, < 0 = worse.
#  MAPE  mean absolute percentage error.
#  MASE  error relative to "same month last year" in the training period; < 1 is better.
#  r     correlation between observed and predicted (pattern only, ignores bias).
#  Outbreak detection: outbreak month = incidence above the district's 75th
#        percentile for that calendar month in the training years.

naive_scale <- split$train %>% group_by(District) %>% arrange(dt) %>%
  summarise(scale = mean(abs(inc - lag(inc, 12)), na.rm = TRUE), .groups = "drop")
outbreak_thr <- split$train %>% group_by(District, month) %>%
  summarise(threshold = quantile(inc, 0.75), .groups = "drop")

ev <- preds %>%
  left_join(naive_scale, by = "District") %>%
  left_join(outbreak_thr, by = c("District", "month")) %>%
  mutate(obs_outbreak = inc > threshold, pred_outbreak = pred > threshold)

perf_district <- ev %>% group_by(features, District) %>%
  summarise(MAE  = mean(abs(pred - inc)),
            RMSE = sqrt(mean((pred - inc)^2)),
            Bias = mean(pred - inc),
            R2   = 1 - sum((inc - pred)^2) / sum((inc - mean(inc))^2),
            MAPE = 100 * mean(abs(pred - inc) / inc),
            MASE = mean(abs(pred - inc)) / first(scale),
            r    = cor(pred, inc), .groups = "drop")

base_mae <- perf_district %>% filter(features == "Seasonal baseline") %>% select(District, MAE_baseline = MAE)

perf_overall <- perf_district %>% left_join(base_mae, by = "District") %>%
  group_by(features) %>%
  summarise(median_R2   = median(R2),
            median_MAPE = median(MAPE),
            median_MASE = median(MASE),
            median_r    = median(r),
            districts_beating_baseline = sum(MAE < MAE_baseline), .groups = "drop") %>%
  left_join(ev %>% group_by(features) %>%
              summarise(MAE  = mean(abs(pred - inc)),
                        RMSE = sqrt(mean((pred - inc)^2)),
                        bias_pct    = 100 * (sum(pred * pop) - sum(inc * pop)) / sum(inc * pop),
                        sensitivity = mean(pred_outbreak[obs_outbreak]),
                        specificity = mean(!pred_outbreak[!obs_outbreak]),
                        PPV         = mean(obs_outbreak[pred_outbreak]), .groups = "drop"),
            by = "features") %>%
  select(features, MAE, RMSE, median_R2, median_MAPE, median_MASE, median_r, bias_pct,
         districts_beating_baseline, sensitivity, specificity, PPV) %>%
  arrange(MAE)

print(as.data.frame(perf_overall), digits = 3)
write.csv(perf_overall,  file.path(OUT, "performance_overall.csv"),     row.names = FALSE)
write.csv(perf_district, file.path(OUT, "performance_by_district.csv"), row.names = FALSE)


# =============================================================================
# 8. PREDICTION PLOTS
# =============================================================================
main_pred <- filter(preds, features == MAIN_SET)

# 8a. Whole series: observed (solid), predicted in the test years (dashed)
p_series <- ggplot() +
  annotate("rect", xmin = TRAIN_END, xmax = max(dat$dt), ymin = -Inf, ymax = Inf, fill = "grey92") +
  geom_line(data = dat, aes(dt, inc, colour = "Observed", linetype = "Observed"), linewidth = 0.45) +
  geom_line(data = main_pred, aes(dt, pred, colour = "Predicted", linetype = "Predicted"), linewidth = 0.7) +
  scale_colour_manual(values = c(Observed = "black", Predicted = "#D55E00"), name = NULL) +
  scale_linetype_manual(values = c(Observed = "solid", Predicted = "22"), name = NULL) +
  facet_wrap(~District, scales = "free_y", ncol = 6) +
  labs(title = "Observed malaria incidence and XGBoost predictions",
       subtitle = "Shaded: test period (January 2023 onwards), not used to fit or tune the model",
       x = NULL, y = "Cases per 1,000 per month") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom", legend.key.width = unit(1.2, "cm"),
        axis.text.x = element_text(angle = 45, hjust = 1), panel.grid.minor = element_blank())
ggsave(file.path(OUT, "1_observed_vs_predicted_series.png"), p_series, width = 15, height = 10, dpi = 150)

# 8b. Predictive R2 by district for the three XGBoost versions and the baseline
p_r2 <- perf_district %>%
  mutate(District = reorder(District, ifelse(features == MAIN_SET, R2, NA), max, na.rm = TRUE)) %>%
  ggplot(aes(R2, District, colour = features, shape = features)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
  geom_point(size = 2.4) +
  coord_cartesian(xlim = c(-1, 1)) +
  scale_colour_manual(values = c("Seasonal baseline" = "grey40", "Season + climate" = "#D55E00",
                                 "Season + previous incidence" = "#0072B2",
                                 "Season + climate + previous incidence" = "#009E73"), name = NULL) +
  scale_shape_manual(values = c(4, 16, 17, 15), name = NULL) +
  labs(title = "Predictive R2 by district on the test years",
       subtitle = "1 = perfect; 0 = no better than the district's test-period average; below 0 = worse",
       x = "Predictive R2 (Nash-Sutcliffe efficiency)", y = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "bottom") +
  guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2))
ggsave(file.path(OUT, "2_R2_by_district.png"), p_r2, width = 9, height = 9, dpi = 150)


# =============================================================================
# 9. SHAP: WHICH PREDICTORS DRIVE THE PREDICTIONS?
# =============================================================================
#  For each test month, SHAP splits the prediction into the contribution of each
#  predictor, on the log scale of incidence. A SHAP value of +0.2 means that
#  predictor raised the prediction by about 22% (exp(0.2) = 1.22) in that month.
#  SHAP values are computed on the TEST months, i.e. on data the model never saw.

vars <- feature_sets[[MAIN_SET]]
dtest <- xgb.DMatrix(as.matrix(split$test[, vars]), base_margin = split$test$margin)
shap <- predict(models[[MAIN_SET]], dtest, predcontrib = TRUE)[, vars]

shap_long <- as.data.frame(shap) %>%
  mutate(District = split$test$District, dt = split$test$dt) %>%
  pivot_longer(all_of(vars), names_to = "feature", values_to = "shap") %>%
  left_join(split$test %>% select(District, dt, all_of(vars)) %>%
              pivot_longer(all_of(vars), names_to = "feature", values_to = "value"),
            by = c("District", "dt", "feature")) %>%
  group_by(feature) %>%
  mutate(value_rank = (rank(value) - 1) / (n() - 1)) %>%    # 0 = lowest value, 1 = highest
  ungroup() %>%
  mutate(label = nice_name[feature], group = group_of(feature))
write.csv(shap_long, file.path(OUT, "shap_values_test.csv"), row.names = FALSE)

importance <- shap_long %>% group_by(feature, label, group) %>%
  summarise(mean_abs_shap = mean(abs(shap)), .groups = "drop") %>% arrange(desc(mean_abs_shap))
write.csv(importance, file.path(OUT, "shap_importance.csv"), row.names = FALSE)

group_colours <- c("Previous incidence" = "#0072B2", "Season" = "#999999", "Rainfall" = "#009E73",
                   "Max temperature" = "#D55E00", "Min temperature" = "#E69F00",
                   "Relative humidity" = "#CC79A7")

# Beeswarm layer (falls back to jitter if ggbeeswarm is not installed)
swarm <- if (requireNamespace("ggbeeswarm", quietly = TRUE)) {
  ggbeeswarm::geom_quasirandom(orientation = "y", width = 0.4, size = 0.9, alpha = 0.75)
} else {
  geom_jitter(height = 0.25, width = 0, size = 0.9, alpha = 0.75)
}
beeswarm_colours <- scale_colour_gradient(low = "#1E88E5", high = "#FF0D57", breaks = c(0, 1),
                                          labels = c("Low", "High"), name = "Predictor\nvalue")

# 9a. Overall importance (bar chart)
p_imp <- ggplot(importance, aes(mean_abs_shap, reorder(label, mean_abs_shap), fill = group)) +
  geom_col() + scale_fill_manual(values = group_colours, name = NULL) +
  labs(title = "Predictor importance (XGBoost, test years)",
       subtitle = "Mean |SHAP|: average size of each predictor's effect on predicted log incidence",
       x = "Mean |SHAP|", y = NULL) +
  theme_minimal(base_size = 11)
ggsave(file.path(OUT, "3_shap_importance.png"), p_imp, width = 8, height = 7, dpi = 150)

# 9b. Beeswarm, all districts
#   One dot = one district-month. Position: SHAP value (right = raised the prediction).
#   Colour: the predictor's value that month (red = high, blue = low).
#   Example: red dots on the right for "Rainfall, sum of lags 1-3" mean that
#   high rainfall in the previous 3 months pushed predicted incidence up.
p_bee <- ggplot(shap_long, aes(shap, reorder(label, abs(shap), mean), colour = value_rank)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  swarm + beeswarm_colours +
  labs(title = "SHAP beeswarm (XGBoost, all districts, test years)",
       subtitle = "Each dot is one district-month. Right of zero = raised predicted incidence; colour = predictor value",
       x = "SHAP value (effect on log incidence)", y = NULL) +
  theme_minimal(base_size = 11)
ggsave(file.path(OUT, "4_shap_beeswarm.png"), p_bee, width = 10, height = 8, dpi = 150)

# 9c. Beeswarm for each district (30 dots per predictor), saved in a sub-folder.
#     Here the colours are relative to the district itself (red = a high value for that district).
for (d in sort(unique(shap_long$District))) {
  p <- shap_long %>% filter(District == d) %>%
    group_by(feature) %>% mutate(value_rank = (rank(value) - 1) / (n() - 1)) %>% ungroup() %>%  # colours relative to this district
    ggplot(aes(shap, reorder(label, abs(shap), mean), colour = value_rank)) +
    geom_vline(xintercept = 0, colour = "grey60") +
    swarm + beeswarm_colours +
    labs(title = paste("SHAP beeswarm -", d),
         subtitle = "XGBoost, test months (January 2023 - June 2025)",
         x = "SHAP value (effect on log incidence)", y = NULL) +
    theme_minimal(base_size = 11)
  ggsave(file.path(OUT, "beeswarm_by_district", paste0("beeswarm_", d, ".png")), p, width = 9, height = 7, dpi = 150)
}

# 9d. Dependence plots for the six most important climate predictors:
#     how the effect changes with the value of the predictor
top_climate <- importance %>% filter(feature %in% climate_vars) %>% slice_head(n = 6) %>% pull(feature)
p_dep <- shap_long %>% filter(feature %in% top_climate) %>%
  mutate(label = factor(label, levels = nice_name[top_climate])) %>%
  ggplot(aes(value, shap)) +
  geom_hline(yintercept = 0, colour = "grey60") +
  geom_point(alpha = 0.35, size = 0.9, colour = "#0072B2") +
  geom_smooth(method = "loess", se = FALSE, colour = "black", linewidth = 0.8, formula = y ~ x) +
  facet_wrap(~label, scales = "free_x") +
  labs(title = "How the climate effects change with the climate value (SHAP dependence)",
       subtitle = "Above zero = raises predicted incidence. Rainfall in mm, temperature in degrees C, humidity in %",
       x = "Predictor value", y = "SHAP value") +
  theme_minimal(base_size = 11)
ggsave(file.path(OUT, "5_shap_dependence_climate.png"), p_dep, width = 11, height = 7, dpi = 150)

# 9e. Importance by district: top 8 predictors in each district
imp_district <- shap_long %>% group_by(District, feature, label, group) %>%
  summarise(mean_abs_shap = mean(abs(shap)), .groups = "drop") %>%
  group_by(District) %>% slice_max(mean_abs_shap, n = 8, with_ties = FALSE) %>% ungroup() %>%
  mutate(key = reorder(paste(label, District, sep = "___"), mean_abs_shap))
write.csv(imp_district, file.path(OUT, "shap_importance_by_district.csv"), row.names = FALSE)

p_imp_d <- ggplot(imp_district, aes(mean_abs_shap, key, fill = group)) +
  geom_col() + facet_wrap(~District, scales = "free_y", ncol = 5) +
  scale_y_discrete(labels = function(x) sub("___.*", "", x)) +
  scale_fill_manual(values = group_colours, name = NULL) +
  labs(title = "Predictor importance by district: top 8 (XGBoost, test years)",
       subtitle = "One model for all districts, so the effect of each predictor is the same everywhere; bars differ because the predictors varied differently in each district",
       x = "Mean |SHAP|", y = NULL) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom", panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())
ggsave(file.path(OUT, "6_shap_importance_by_district.png"), p_imp_d, width = 16, height = 14, dpi = 150)

cat("\nDone. Tables and figures are in the folder:", OUT, "\n")
