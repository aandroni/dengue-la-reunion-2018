////////////////////////////////////////////////////////////////////////////////
// File: mcmc.hpp
////////////////////////////////////////////////////////////////////////////////
// MCMC sampling utilities
////////////////////////////////////////////////////////////////////////////////

#ifndef mcmc_hpp_
#define mcmc_hpp_

#include <iostream>
#include <cassert>
#include <vector>
#include <set>
#include <string>
#include <random>
#include <cfloat> // DBL_MAX
#include <cmath> // nextafter
#include <iomanip>
#include <fstream>


////////////////////////////////////////////////////////////////////////////////
// MCMC model
////////////////////////////////////////////////////////////////////////////////
struct MCMCModel {
  virtual size_t get_number_of_parameters() = 0;
  virtual double compute_log_likelihood(const std::vector<double> &params) = 0;
};


////////////////////////////////////////////////////////////////////////////////
// MCMC class
////////////////////////////////////////////////////////////////////////////////
struct MCMC {
  // -------------------- Public interface --------------------
  MCMC(MCMCModel& model);

  void set_chain_length(size_t chain_length);
  size_t get_chain_length() const;

  void set_thinning_factor(size_t thinning_factor);
  size_t get_thinning_factor() const;

  // Set output file name: the extension (.csv) is automatically appended
  // If not called, the default will be: "mcmc_output.csv"
  void set_output_file_name(std::string output_file_name);
  std::string get_output_file_name() const;

  // Proposal choices are: "lognormal" (default) or "normal"
  void set_proposal_type(size_t param_i, std::string proposal_type);
  std::string get_proposal_type(size_t param_i) const;

  // Prior choices are: "uniform", "normal" or "gamma"
  // Default choice is Uniform(0, 1)
  // If prior = "uniform" => param1 = lower_bound, param2 = upper_bound
  // If prior = "normal" => param1 = mean, param2 = standard deviation
  // If prior = "gamma" => param1 = shape, param2 = scale
  // Parametrization of gamma function corresponds to:
  //   PDF(x) ~ x^(shape - 1) exp(-x / scale)
  void set_prior(size_t param_i, std::string prior,
    double param1, double param2);
  std::string get_prior(size_t param_i) const;
  std::vector<double> get_prior_params(size_t param_i) const;

  void set_parameter_name(size_t param_i, std::string name);
  std::string get_parameter_name(size_t param_i) const;

  // Random walk step scale
  void set_step_scale(size_t param_i, double step_scale);
  double get_step_scale(size_t param_i) const;

  // Run MCMC with specified seed
  void run(size_t seed = 0, bool verbose = true);

  // Run multiple (at most max_chains) chains (each of n_iter iterations)
  // adjusting the random walk step scales until the acceptance rates are
  // close to opt_acceptance_rate (within tolerance)
  void refine_step_scales(size_t n_iter = 100, size_t max_chains = 100,
    double opt_acceptance_rate = 0.24, double tolerance = 0.05, size_t seed = 0,
    bool verbose = false);

  // -------------------- Internals --------------------
  MCMCModel *model;
  double log_lik;
  size_t n_params;
  std::vector<double> params;
  std::vector<std::string> param_names;
  std::vector<std::vector<double>> prior_params;
  std::vector<double> step_scales;
  std::vector<double> n_moves_accepted;
  std::vector<double> n_moves_proposed;
  std::vector<double> acceptance_rates;
  std::vector<void (*)(double, double, double&, double&)> proposal_funcs;
  std::vector<void (*)(double, double, double, double, double&, bool&)>
    prior_funcs;
  std::mt19937_64 gen;
  std::normal_distribution<double> norm_dist;
  std::uniform_real_distribution<double> unif_dist;
  std::string out_name;
  size_t chain_length;
  size_t thinning_factor;
  std::ofstream output_file;
  double elapsed_time;

  static std::set<std::string> available_proposals;
  static std::set<std::string> available_priors;

  bool is_out_of_bounds(size_t param_i) const;
  void update_parameters();
  double update_parameter(size_t param_i);
  void write_to_file();
  void draw_initial_params_from_priors();
  void initialize_run(size_t seed);
  void finalize_run();
  void cout_pretty_line(size_t width, char ch) const;
  void cout_progress(size_t iter, clock_t t_beg) const;
  void cout_pretty_table() const;
  void issue_out_of_bounds_warning(std::string method, size_t param_i) const;
  void issue_invalid_param_warning(std::string method, std::string param_name,
    double bad_value, double revert_value) const;
};


////////////////////////////////////////////////////////////////////////////////
// Proposal functions
////////////////////////////////////////////////////////////////////////////////
// They all take four parameters:
//  1) old parameter value (in)
//  2) random walk step (in)
//  3) new parameter value (out)
//  4) log of proposals ratio (out)
////////////////////////////////////////////////////////////////////////////////
void lognormal_proposal(double old_value, double random_walk_step,
  double &new_value, double &log_proposal);

void normal_proposal(double old_value, double random_walk_step,
  double &new_value, double &log_proposal);


////////////////////////////////////////////////////////////////////////////////
// Priors
////////////////////////////////////////////////////////////////////////////////
void uniform_prior(double old_value, double new_value,
  double lower_bound, double upper_bound,
  double &log_prior, bool &outside_support);


void normal_prior(double old_value, double new_value,
  double prior_mean, double prior_std,
  double &log_prior, bool &outside_support);


void gamma_prior(double old_value, double new_value,
  double prior_shape, double prior_scale,
  double &log_prior, bool &outside_support);


#endif // mcmc_hpp_