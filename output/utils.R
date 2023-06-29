get_estimates <- function(f_name, burn_in) {
  res_MCMC <- read_csv(f_name, col_types = cols(.default = col_double())) %>%
    mutate(iteration = 1:nrow(.)) %>%
    filter(iteration > burn_in)
  
  estimates <- res_MCMC %>%
    gather(-iteration, -loglik, key = "variable", value = "value") %>%
    group_by(variable) %>%
    summarize(mean = mean(value),
              lower = quantile(value, probs = 0.025),
              upper = quantile(value, probs = 0.975))
  
  deviance <- -2.0 * res_MCMC$loglik
  aveD <- mean(deviance)
  pV <- var(deviance) * 0.5
  DIC <- aveD + pV
  
  list(estimates = estimates, DIC = DIC)
}


get_simulations <- function(f_name_sims, t0) {
  res_sims <- read_csv(f_name_sims,
                       col_types = cols(type = col_character(),
                                        .default = col_double())) %>%
    gather(-sim, -type, key = "week", value = "value") %>%
    mutate(week = as.numeric(week)) %>%
    group_by(type, week) %>%
    summarize(mean = mean(value),
              median = median(value),
              lower = quantile(value, probs = 0.025),
              upper = quantile(value, probs = 0.975))
  res_sims$date <- res_sims$week * 7 + t0
  return(res_sims)
}

