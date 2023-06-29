library(tidyverse)
theme_set(theme_bw())
source("utils.R")

# Load data
t0 <- as.Date("2018-01-01")
epi_data <- read_delim("../data20190823.txt", delim = " ",
                       col_names = c("i", "confirmed_cases")) %>%
  mutate(date = t0 + i * 7)

# Get parameter estimates
res <- get_estimates("output_lamb_T15_R11.csv", burn_in = 1000)
res$estimates

# Load and plot simulations
sims <- get_simulations("output_lamb_T15_R11_sims.csv", t0)

sims %>%
  filter(type == "incidence", date <= max(epi_data$date)) %>%
  ggplot() +
  geom_ribbon(aes(x = date, ymin = lower, ymax = upper), fill = "dodgerblue3",
              alpha = 0.3) +
  geom_line(aes(x = date, y = median), color = "dodgerblue4") +
  geom_point(data = epi_data, aes(x = date, y = confirmed_cases)) +
  xlab("") + ylab("Weekly N. of Cases")
