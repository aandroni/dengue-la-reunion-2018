#ifndef model_hpp_
#define model_hpp_

#include "varia.hpp"
#include "mcmc.hpp"
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_times.hpp>


typedef std::vector<size_t> StateI;
typedef std::vector<double> StateD;


//------------------------------------------------------------------------------
// Model configuration
//------------------------------------------------------------------------------
struct SIRConfiguration {
  size_t n_data;
  bool two_sir;
  bool estimate_reporting;
  bool estimate_gamma;
  std::string transmission_model;
  std::string scenario;

  // Fixed params
  double population;
  double fixed_gamma;
  double fixed_reporting;
  double prop_immune0;
  double dt;

  std::vector<double> params;
  virtual size_t get_n_params() = 0;

  // Inferred params
  virtual double get_s0() = 0;
  virtual double get_i0_sim1() = 0;
  virtual double get_i0_sim2() = 0;
  virtual double get_beta(double curr_time) = 0;
  virtual double get_reporting() = 0;
  virtual double get_gamma() = 0;
};


//------------------------------------------------------------------------------
// Deterministic SIR
//------------------------------------------------------------------------------
struct ReusableObserver {
  std::vector<StateD> &trajectory;
  size_t pos;

  ReusableObserver(std::vector<StateD>& trajectory);
  void operator()(const StateD& x, double t);
};


struct DeterministicSIR {
  SIRConfiguration *config;

  DeterministicSIR(SIRConfiguration& config);
  void operator()(const StateD& x, StateD& dxdt, const double t);
};


//------------------------------------------------------------------------------
// Stochastic SIR
//------------------------------------------------------------------------------
struct StochasticSIR {
  SIRConfiguration *config;
  std::vector<StateI> epi;

  StochasticSIR(SIRConfiguration& config);
  void do_step(StateI& state, double curr_time, double curr_dt,
    std::mt19937_64& gen);
  void simulate(const StateI& initial_conditions, std::vector<double>& ts,
    std::vector<StateI>& epi, std::mt19937_64& gen);
};


//------------------------------------------------------------------------------
// Model
//------------------------------------------------------------------------------
struct DengueModel: public MCMCModel {
  SIRConfiguration *config;
  DeterministicSIR sir;
  ReusableObserver obs1;
  ReusableObserver obs2;
  StochasticSIR simulator;

  // Data
  size_t n_data;
  std::vector<double> real_ts;
  std::vector<size_t> real_incidence;

  // Varia
  size_t split;
  bool two_sir;
  std::vector<double> ts1;
  std::vector<double> ts2;
  std::vector<StateD> sim1;
  std::vector<StateD> sim2;
  boost::numeric::odeint::runge_kutta4<StateD> stepper;

  DengueModel(std::string epi_data, size_t up_to, size_t split,
    SIRConfiguration& config);
  size_t get_number_of_parameters();

  // Likelihood (based on determinist SIR model)
  double compute_log_likelihood(const std::vector<double>& params);
  double ll_two_sir();
  double ll_one_sir();

  // Simulations (from stochastic SIR model)
  void simulate(std::vector<double>& ts, std::string mcmc_file,
    std::string out_file, size_t burn_in, size_t n_param_sets = 100,
    size_t n_reps = 1, size_t seed = 20190628);
  void simulate_epi(const std::vector<double>& params, std::vector<double>& ts,
    std::mt19937_64& gen, std::vector<StateI>& epi,
    std::vector<size_t>& incidence);
  void simulate_epi_two_sir(std::vector<double>& ts, std::mt19937_64& gen,
    std::vector<StateI>& epi, std::vector<size_t>& incidence);
  void simulate_epi_one_sir(std::vector<double>& ts, std::mt19937_64& gen,
    std::vector<StateI>& epi, std::vector<size_t>& incidence);
};


//------------------------------------------------------------------------------
// Observers
//------------------------------------------------------------------------------
std::vector<size_t> observe_incidence_pois(SIRConfiguration* config,
  const std::vector<size_t>& incidence, std::mt19937_64& gen);


#endif // model_hpp_
