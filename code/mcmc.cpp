////////////////////////////////////////////////////////////////////////////////
// File: mcmc.cpp
////////////////////////////////////////////////////////////////////////////////

#include "mcmc.hpp"


////////////////////////////////////////////////////////////////////////////////
// MCMC class
////////////////////////////////////////////////////////////////////////////////

std::set<std::string> MCMC::available_proposals = {"lognormal", "normal"};
std::set<std::string> MCMC::available_priors = {"uniform", "normal", "gamma"};


MCMC::MCMC(MCMCModel& model): model(&model) {
  n_params = this->model->get_number_of_parameters();
  out_name = "mcmc_output";
  chain_length = 100;
  thinning_factor = 1;
  elapsed_time = 0.0;
  norm_dist = std::normal_distribution<double>(0.0, 1.0);
  unif_dist = std::uniform_real_distribution<double>(0.0,
    nextafter(1.0, DBL_MAX));

  params.resize(n_params, 0.0);
  param_names.resize(n_params, "");
  for (size_t param_i = 0; param_i < n_params; ++param_i) {
    param_names[param_i] = "p" + std::to_string(param_i);
  }
  step_scales.resize(n_params, 1.0);
  n_moves_accepted.resize(n_params, 0.0);
  n_moves_proposed.resize(n_params, 0.0);
  acceptance_rates.resize(n_params, 0.0);
  proposal_funcs.resize(n_params, &lognormal_proposal);
  prior_funcs.resize(n_params, &uniform_prior);
  prior_params.resize(n_params, std::vector<double>{0.0, 1.0});
}


void MCMC::set_chain_length(size_t chain_length) {
  this->chain_length = chain_length;
}


size_t MCMC::get_chain_length() const {
  return chain_length;
}


void MCMC::set_thinning_factor(size_t thinning_factor) {
  this->thinning_factor = thinning_factor;
}


size_t MCMC::get_thinning_factor() const {
  return thinning_factor;
}


void MCMC::set_output_file_name(std::string output_file_name) {
  // Avoid empty string
  if (output_file_name != "") out_name = output_file_name;
}


std::string MCMC::get_output_file_name() const {
  return out_name;
}


void MCMC::set_proposal_type(size_t param_i, std::string proposal_type) {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("set_proposal_type", param_i);
    return;
  }
  if (available_proposals.count(proposal_type) == 0) {
    std::cout << "WARNING [set_proposal_type]: Invalid proposal type = "
      << proposal_type << std::endl;
    std::cout << "Implemented proposals are: ";
    for (std::string proposal : available_proposals) {
      std::cout << "'" << proposal << "' ";
    }
    std::cout << std::endl << std::endl;
    return;
  }

  if (proposal_type == "lognormal") {
    proposal_funcs[param_i] = &lognormal_proposal;
  } else if (proposal_type == "normal") {
    proposal_funcs[param_i] = &normal_proposal;
  }
}


std::string MCMC::get_proposal_type(size_t param_i) const {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("get_proposal_type", param_i);
  }

  std::string proposal_type = "";
  if (proposal_funcs[param_i] == &lognormal_proposal) {
    proposal_type = "lognormal";
  } else if (proposal_funcs[param_i] == &normal_proposal) {
    proposal_type = "normal";
  }
  return proposal_type;
}


void MCMC::set_prior(size_t param_i, std::string prior,
    double param1, double param2) {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("set_prior", param_i);
    return;
  }
  if (available_priors.count(prior) == 0) {
    std::cout << "WARNING [set_prior]: Invalid prior = "
      << prior << std::endl;
    std::cout << "Implemented priors are: ";
    for (std::string prior : available_priors) {
      std::cout << "'" << prior << "' ";
    }
    std::cout << std::endl << std::endl;
    return;
  }

  double safe_param1 = param1;
  double safe_param2 = param2;
  if (prior == "uniform") {
    prior_funcs[param_i] = &uniform_prior;
  } else if (prior == "normal") {
    prior_funcs[param_i] = &normal_prior;
    if (safe_param2 < 0.0) {
      safe_param2 = 1.0;
      issue_invalid_param_warning("set_prior", "gaussian std", param2,
        safe_param2);
    }
  } else if (prior == "gamma") {
    if (safe_param1 <= 0.0) {
      safe_param1 = 1.0;
      issue_invalid_param_warning("set_prior", "gamma shape", param1,
        safe_param1);
    }
    if (safe_param2 <= 0.0) {
      safe_param2 = 2.0;
      issue_invalid_param_warning("set_prior", "gamma shape", param2,
        safe_param2);
    }
    prior_funcs[param_i] = &gamma_prior;
  }
  prior_params[param_i][0] = safe_param1;
  prior_params[param_i][1] = safe_param2;
}


std::string MCMC::get_prior(size_t param_i) const {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("get_prior", param_i);
  }

  std::string prior = "";
  if (prior_funcs[param_i] == &uniform_prior) {
    prior = "uniform";
  } else if (prior_funcs[param_i] == &normal_prior) {
    prior = "normal";
  } else if (prior_funcs[param_i] == &gamma_prior) {
    prior = "gamma";
  }
  return prior;
}


std::vector<double> MCMC::get_prior_params(size_t param_i) const {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("get_prior_params", param_i);
    return std::vector<double>();
  }

  return std::vector<double>(prior_params[param_i]);
}


void MCMC::set_parameter_name(size_t param_i, std::string name) {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("set_parameter_name", param_i);
    return;
  }
  param_names[param_i] = name;
}


std::string MCMC::get_parameter_name(size_t param_i) const {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("get_parameter_name", param_i);
    return "";
  }
  return param_names[param_i];
}


void MCMC::set_step_scale(size_t param_i, double step_scale) {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("set_step_scale", param_i);
    return;
  }
  double safe_step_scale = step_scale;
  if (step_scale < 0.0) {
    safe_step_scale = 1.0;
    issue_invalid_param_warning("set_step_scale", "step scale", step_scale,
      safe_step_scale);
  }

  step_scales[param_i] = safe_step_scale;
}


double MCMC::get_step_scale(size_t param_i) const {
  if (is_out_of_bounds(param_i)) {
    issue_out_of_bounds_warning("get_step_scale", param_i);
    return -1.0;
  }

  return step_scales[param_i];
}


void MCMC::run(size_t seed, bool verbose) {
  initialize_run(seed);

  clock_t t_beg = clock();
  for (size_t iter = 0; iter < chain_length; ++iter) {
    if (verbose) cout_progress(iter, t_beg);
    update_parameters();
    if (iter % thinning_factor == 0) write_to_file();
  }
  clock_t t_end = clock();
  elapsed_time = double(t_end - t_beg) / CLOCKS_PER_SEC;

  finalize_run();
}


void MCMC::refine_step_scales(size_t n_iter, size_t max_chains,
    double opt_acceptance_rate, double tolerance, size_t seed,
    bool verbose) {
  size_t old_chain_length = chain_length;
  size_t old_thinning_factor = thinning_factor;
  set_thinning_factor(1);
  set_chain_length(n_iter);

  size_t repeat = 0;
  size_t decay_start = max_chains / 4;
  double delta = 1.0;
  while (true) {
    ++repeat;
    if (repeat >= decay_start) delta = 1.0 / (repeat - decay_start + 1);
    std::cout << "Fine tuning " << repeat << std::endl;
    run(seed, verbose);
    bool is_OK = true;
    for (size_t param_i = 0; param_i < n_params; ++param_i) {
      double curr_AR = acceptance_rates[param_i];
      double diff = curr_AR - opt_acceptance_rate;
      // Increase step scale if acceptance rate is too high, decrease viceversa
      step_scales[param_i] *= 1.0 + delta * diff;
      is_OK = is_OK && (std::abs(diff) <= tolerance);
    }
    if (is_OK || repeat >= max_chains) break;
  }

  // Reset parameters
  set_thinning_factor(old_thinning_factor);
  set_chain_length(old_chain_length);
}


bool MCMC::is_out_of_bounds(size_t param_i) const {
  return (param_i >= n_params) ? true : false;
}


void MCMC::update_parameters() {
  for (size_t param_i = 0; param_i < n_params; ++param_i) {
    if (step_scales[param_i] > 0.0) {
      n_moves_accepted[param_i] += update_parameter(param_i);
    }
    n_moves_proposed[param_i] += 1.0;
    acceptance_rates[param_i] =
      n_moves_accepted[param_i] / n_moves_proposed[param_i];
  }
}


double MCMC::update_parameter(size_t param_i) {
  double old_value = params[param_i];
  double new_value, log_proposals;
  double random_walk_step = step_scales[param_i] * norm_dist(gen);
  proposal_funcs[param_i](old_value, random_walk_step, new_value,
    log_proposals);
  double log_priors;
  bool is_outside_support;
  prior_funcs[param_i](old_value, new_value,
    prior_params[param_i][0], prior_params[param_i][1],
    log_priors, is_outside_support);

  // Immediately reject if outside prior support
  if (is_outside_support) return 0.0;

  params[param_i] = new_value;
  double new_log_lik = model->compute_log_likelihood(params);
  double proba = exp(new_log_lik - log_lik + log_priors + log_proposals);

  if (unif_dist(gen) < proba) {
    // Accept move
    log_lik = new_log_lik;
    return 1.0;
  }
  // Reject move
  params[param_i] = old_value;
  return 0.0;
}


void MCMC::write_to_file() {
  output_file << log_lik << ",";
  for (size_t curr = 0; curr < n_params; ++curr) {
    output_file << params[curr] << ",";
  }
  for (size_t curr = 0; curr < (n_params - 1); ++curr) {
    output_file << acceptance_rates[curr] << ",";
  }
  output_file << acceptance_rates[n_params - 1] << std::endl;
}


void MCMC::draw_initial_params_from_priors() {
  for (size_t param_i = 0; param_i < n_params; ++param_i) {
    std::string prior_i = get_prior(param_i);
    std::vector<double> prior_params = get_prior_params(param_i);
    if (prior_i == "uniform") {
      std::uniform_real_distribution<double> udist(
        prior_params[0], nextafter(prior_params[1], DBL_MAX)
      );
      params[param_i] = udist(gen);
    } else if (prior_i == "normal") {
      std::normal_distribution<double> ndist(prior_params[0], prior_params[1]);
      params[param_i] = ndist(gen);
    } else if (prior_i == "gamma") {
      std::gamma_distribution<double> gdist(prior_params[0], prior_params[1]);
      params[param_i] = gdist(gen);
    }
  }
}


void MCMC::initialize_run(size_t seed) {
  gen.seed(seed);
  for (size_t param_i = 0; param_i < n_params; ++param_i) {
    n_moves_accepted[param_i] = 0.0;
    n_moves_proposed[param_i] = 0.0;
    acceptance_rates[param_i] = 0.0;
  }
  output_file.open(out_name + ".csv");
  // Add header
  output_file << "loglik,";
  for (size_t param_i = 0; param_i < n_params; ++param_i) {
    output_file << param_names[param_i] << ",";
  }
  for (size_t param_i = 0; param_i < (n_params - 1); ++param_i) {
    output_file << ("acceptance_" + param_names[param_i]) << ",";
  }
  output_file << ("acceptance_" + param_names[n_params - 1]) << std::endl;

  size_t width = 51;
  cout_pretty_line(width, '-');
  std::cout << "Starting MCMC" << std::endl;
  cout_pretty_line(width, '-');

  draw_initial_params_from_priors();

  std::cout << std::endl;
  log_lik = model->compute_log_likelihood(params);
  std::cout << "LL at start   = " << std::scientific << std::setprecision(4) <<
    log_lik << std::endl;
}


void MCMC::finalize_run() {
  output_file.close();

  // Print timing and parameters
  std::cout << std::endl;
  std::cout << "LL at the end = " << std::scientific << std::setprecision(4) <<
    log_lik << std::endl;
  std::cout << std::endl;
  std::cout << "MCMC took " << std::fixed << std::setprecision(2) <<
    elapsed_time << " (s)" << std::endl;
  std::cout << std::endl;
  cout_pretty_table();
}


void MCMC::cout_pretty_line(size_t width, char ch) const {
  for (size_t curr = 0; curr < width; ++curr) std::cout << "-";
  std::cout << std::endl;
}


void MCMC::cout_progress(size_t iter, clock_t t_beg) const {
  double progress = 100.0 * (iter + 1.0) / chain_length;
  if (std::abs(progress - floor(progress)) < 1e-6) {
    // for (size_t curr = 0; curr < 22; ++curr) {
    //   std::cout << "\b";
    // }
    // Write percentage of work done
    std::cout << " " << std::fixed << std::setfill(' ') << std::setw(3) <<
      std::setprecision(0) << progress << "%";
    // Write estimated time left
    clock_t t_curr = clock();
    double seconds_per_iter = double(t_curr - t_beg) / CLOCKS_PER_SEC;
    seconds_per_iter /= (iter + 1.0);
    double left = (chain_length - iter - 1) * seconds_per_iter; // Seconds left
    size_t secs = size_t(left) % 60;
    left = (left - secs) / 60.0; // Minutes left
    size_t mins = size_t(left) % 60;
    left = (left - mins) / 60.0; // Hours left
    size_t hours = size_t(left) % 24;
    left = (left - hours) / 24.0; // Days left
    size_t days = size_t(left);
    std::cout << " ETA " << std::setfill('0') << std::setw(3) << days << ":"
      << std::setw(2) << hours << ":" << std::setw(2) << mins << ":" <<
      std::setw(2) << secs;
    std::cout << std::endl;
  }
}


void MCMC::cout_pretty_table() const {
  size_t width = 51;

  // Header
  cout_pretty_line(width, '-');
  std::cout << "| Param |        Value |";
  std::cout << "      RWStep |   Acc Rate |" << std::endl;
  cout_pretty_line(width, '-');

  // Table body
  for (size_t param_i = 0; param_i < n_params; ++param_i) {
    std::cout << "|" << std::setfill(' ') << std::setw(6) << param_i << " | ";
    std::cout << std::scientific << std::setprecision(4) << std::setw(12) <<
      params[param_i] << " | " << step_scales[param_i] << " | " <<
      acceptance_rates[param_i] << "|" << std::endl;
  }
  cout_pretty_line(width, '-');
}


void MCMC::issue_out_of_bounds_warning(std::string method,
    size_t param_i) const {
  std::cout << "WARNING [" << method << "]: Invalid parameter index = "
    << param_i << std::endl;
  std::cout << "Parameter index should be 0 <= i <= " << n_params
    << std::endl << std::endl;
}


void MCMC::issue_invalid_param_warning(std::string method,
    std::string param_name, double bad_value, double revert_value) const {
  std::cout << "WARNING [" << method << "]: Invalid " << param_name << " = "
    << bad_value << std::endl;
  std::cout << "Reverting to " << param_name << " = " << revert_value
    << std::endl << std::endl;
}


////////////////////////////////////////////////////////////////////////////////
// Proposal functions
////////////////////////////////////////////////////////////////////////////////

void lognormal_proposal(double old_value, double random_walk_step,
    double &new_value, double &log_proposal) {
  new_value = old_value * exp(random_walk_step);
  log_proposal = random_walk_step; // log(new_value) - log(old_value);
}


void normal_proposal(double old_value, double random_walk_step,
    double &new_value, double &log_proposal) {
  new_value = old_value + random_walk_step;
  log_proposal = 0.0;
}


////////////////////////////////////////////////////////////////////////////////
// Priors
////////////////////////////////////////////////////////////////////////////////

void uniform_prior(double old_value, double new_value,
    double lower_bound, double upper_bound,
    double &log_prior, bool &outside_support) {
  log_prior = 0.0;
  outside_support = false;
  if (new_value < lower_bound || new_value > upper_bound) {
    log_prior = -1.0;
    outside_support = true;
  }
}


void normal_prior(double old_value, double new_value,
    double prior_mean, double prior_std,
    double &log_prior, bool &outside_support) {
  log_prior = old_value - new_value;
  log_prior *= ((old_value + new_value) * 0.5 - prior_mean);
  log_prior /= (prior_std * prior_std);
  outside_support = false;
}


void gamma_prior(double old_value, double new_value,
    double prior_shape, double prior_scale,
    double &log_prior, bool &outside_support) {
  if (new_value <= 0.0) {
    log_prior = -1.0;
    outside_support = true;
    return;
  }
  log_prior = (prior_shape - 1.0) * log(new_value / old_value)
    - (new_value - old_value) / prior_scale;
  outside_support = false;
}

