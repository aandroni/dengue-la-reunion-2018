#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include "varia.hpp"
#include "model.hpp"
#include "mcmc.hpp"
using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::odeint::detail;


//------------------------------------------------------------------------------
// Basic model configurations
//------------------------------------------------------------------------------
struct ModelConfiguration: public SIRConfiguration {
  ModelConfiguration(string transmission_model, double infectious_period,
      double reporting, double prop_immune0, double dt) {
    population = 865826; // 2018 data

    this->transmission_model = transmission_model;

    estimate_gamma = true;
    if (infectious_period > 0.0) {
      estimate_gamma = false;
      fixed_gamma = 1.0 / infectious_period;
    }

    estimate_reporting = true;
    if (reporting > 0.0 && reporting <= 1.0) {
      estimate_reporting = false;
      fixed_reporting = reporting;
    }

    assert(prop_immune0 >= 0.0 && prop_immune0 <= 1.0);
    this->prop_immune0 = prop_immune0;

    assert(dt >= 0.0);
    this->dt = dt;
  }

  size_t get_n_params() {
    size_t n_params = 2;
    if (estimate_reporting) ++n_params;
    if (two_sir) ++n_params;
    if (estimate_gamma) ++n_params;
    return n_params;
  }

  double get_s0() {
    return 1.0 - get_i0_sim1() - prop_immune0;
  }

  double get_i0_sim1() {
    return params[1];
  }

  double get_i0_sim2() {
    return params[2];
  }

  double get_beta(double curr_time) {
    // Get climate variable
    double temp;
    if (scenario == "cold" && curr_time >= n_data) {
      // Cold temp only for forecasts
      temp = cold_temperature(curr_time);
    } else if (scenario == "warm" && curr_time >= n_data) {
      // Warm temp only for forecasts
      temp = warm_temperature(curr_time);
    } else {
      temp = average_temperature(curr_time);
    }
    double rain = average_rainfall(curr_time);

    // Compute scaling
    double scaling;
    if (transmission_model == "lamb") {
      scaling = lambrechts_scaling(temp, rain);
    } else if (transmission_model == "perk") {
      scaling = perkins_scaling(temp, rain);
    } else if (transmission_model == "mord") {
      scaling = mordecai_scaling(temp, rain);
    } else if (transmission_model == "null") {
      scaling = null_scaling(temp, rain);
    } else {
      scaling = 1.0;
    }
    return params[0] * scaling;
  }

  double get_reporting() {
    if (estimate_reporting) {
      return two_sir ? params[3] : params[2];
    }
    return fixed_reporting;
  }

  double get_gamma() {
    if (estimate_gamma) {
      return params[get_n_params() - 1];
    }
    return fixed_gamma;
  }
};


void run(string epi_file, string transmission_model,
    double infectious_period_in_weeks, double reporting,
    double prop_immune0, double dt, size_t up_to, size_t split,
    string prefix) {

  ModelConfiguration config(transmission_model, infectious_period_in_weeks,
    reporting, prop_immune0, dt);
  DengueModel model(epi_file, up_to, split, config);

  MCMC mcmc(model);
  mcmc.set_parameter_name(0, "beta0");
  mcmc.set_prior(0, "uniform", 0.0, 10.0);
  mcmc.set_parameter_name(1, "alpha_i0_1");
  mcmc.set_prior(1, "uniform", 0.0, 0.1);
  if (config.two_sir) {
    mcmc.set_parameter_name(2, "alpha_i0_2");
    mcmc.set_prior(2, "uniform", 0.0, 0.1);
  }
  if (config.estimate_reporting) {
    size_t ind = config.two_sir ? 3 : 2;
    mcmc.set_parameter_name(ind, "rho");
    mcmc.set_prior(ind, "uniform", 0.0, 1.0);
  }
  if (config.estimate_gamma) {
    size_t ind = config.get_n_params() - 1;
    mcmc.set_parameter_name(ind, "gamma");
    mcmc.set_prior(ind, "uniform", 7.0 / 21., 7.0 / 4.0);
  }
  mcmc.set_output_file_name(prefix);

  // Find good step scales for random walk
  mcmc.refine_step_scales(10000, 200, 0.234, 0.02);

  // Inference
  mcmc.set_chain_length(500000);
  mcmc.set_thinning_factor(100);
  mcmc.run();

  // Simulations (2 years)
  vector<double> sim_ts;
  for (size_t curr = 0; curr < 52 * 2; ++curr) {
    sim_ts.push_back(double(curr));
  }
  string mcmc_file = prefix + ".csv";
  model.simulate(sim_ts, mcmc_file, prefix + "_sims.csv", 1000, 100, 5);

  config.scenario = "cold";
  model.simulate(sim_ts, mcmc_file, prefix + "_sims_cold.csv", 1000, 100, 5);

  config.scenario = "warm";
  model.simulate(sim_ts, mcmc_file, prefix + "_sims_warm.csv", 1000, 100, 5);
}


int main(int argc, char* argv[]) {
  if (argc < 9) {
    cout << "ERROR - Parameters needed are:" << endl;
    cout << "   1. Epi file" << endl;
    cout << "   2. Transmission_model ('lamb', 'perk', 'mord', 'null', or 'none')" << endl;
    cout << "   3. Infectious_periods (days). Pass -1 to estimate it." << endl;
    cout << "   4. Reporting probability (as integer). Pass -1 to estimate it." << endl;
    cout << "   5. Prop immune at beginning of epidemic (Jan 2018)" << endl;
    cout << "   6. dt (time step)" << endl;
    cout << "   7. up_to (number of data points to use). Pass large number to use all data." << endl;
    cout << "   8. split (point where we reinitialize the initial conditions)" << endl;
    exit(1);
  }

  string epi_file = argv[1];
  string transmission_model = argv[2];
  int infectious_period_in_days = atoi(argv[3]);
  int reporting_percentage = atoi(argv[4]);
  double prop_immune0 = atof(argv[5]);
  double dt = atof(argv[6]);
  size_t up_to = atoi(argv[7]);
  size_t split = atoi(argv[8]);

  set<string> valid_models = {"lamb", "perk", "mord", "null", "none"};
  if (valid_models.count(transmission_model) == 0) {
    cout << "ERROR: invalid transmission model '" << transmission_model
      << "'!" << endl;
    exit(1);
  }

  string prefix = "output_" + transmission_model + "_" + to_string(up_to);
  double infectious_period_in_weeks = infectious_period_in_days / 7.0;
  double reporting = reporting_percentage * 0.01;
  run(epi_file, transmission_model, infectious_period_in_weeks, reporting,
    prop_immune0, dt, up_to, split, prefix);

  return 0;
}
