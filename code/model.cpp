#include "model.hpp"
using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::odeint::detail;


//------------------------------------------------------------------------------
// Deterministic SIR
//------------------------------------------------------------------------------
ReusableObserver::ReusableObserver(vector<StateD>& trajectory):
  trajectory(trajectory), pos(0) { }


void ReusableObserver::operator()(const StateD& x, double t) {
  trajectory[pos++] = x;
  pos = pos % trajectory.size();
}


DeterministicSIR::DeterministicSIR(SIRConfiguration& config):
  config(&config) { }


void DeterministicSIR::operator()(const StateD& x, StateD& dxdt,
    const double t) {
  double s_to_i = config->get_beta(t) * x[0] * x[1] / config->population;
  dxdt[0] = -s_to_i;
  dxdt[1] = s_to_i - config->get_gamma() * x[1];
  dxdt[2] = s_to_i; // Total number of infected
}


//------------------------------------------------------------------------------
// Stochastic SIR
//------------------------------------------------------------------------------
StochasticSIR::StochasticSIR(SIRConfiguration& config): config(&config) { }


void StochasticSIR::do_step(StateI& state, double curr_time, double curr_dt,
    mt19937_64& gen) {
  size_t prev_s = state[0];
  size_t prev_i = state[1];

  double beta = config->get_beta(curr_time);
  double population = config->population;
  double proba_infection = 1.0 - exp(-beta * curr_dt * prev_i / population);
  double proba_removal = 1.0 - exp(-config->get_gamma() * curr_dt);
  binomial_distribution<size_t> infect(prev_s, proba_infection);
  binomial_distribution<size_t> remove(prev_i, proba_removal);
  size_t s_to_i = infect(gen);
  size_t i_to_r = remove(gen);

  state[0] = prev_s - s_to_i;
  state[1] = prev_i + s_to_i - i_to_r;
  state[2] += s_to_i; // Total number of infected
}


void StochasticSIR::simulate(const StateI& initial_conditions,
    vector<double>& ts, vector<StateI>& epi, mt19937_64& gen) {
  size_t n_steps = ts.size();
  epi.resize(n_steps, StateI(3, 0));

  StateI state(initial_conditions);
  vector<double>::iterator start_time = ts.begin();
  vector<double>::iterator end_time = ts.end();
  double current_time;
  double dt = config->dt;
  double current_dt = dt;
  size_t step = 0;
  while (true) {
    current_time = *start_time++;
    epi[step] = state;      
    if (start_time == end_time) break;
    while (less_with_sign(current_time, *start_time, current_dt)) {
      current_dt = min_abs(dt, *start_time - current_time);
      do_step(state, current_time, current_dt, gen);
      current_time += current_dt;
    }
    ++step;
  }
}


//------------------------------------------------------------------------------
// Model
//------------------------------------------------------------------------------
DengueModel::DengueModel(string epi_data, size_t up_to, size_t split,
    SIRConfiguration& config):
    config(&config), sir(config), obs1(sim1), obs2(sim2), simulator(config) {

  // Load data
  load_epi(epi_data, real_ts, real_incidence, up_to);
  n_data = real_incidence.size();
  config.n_data = n_data;

  if (n_data < split) {
    two_sir = false;
    ts1 = vector<double>(real_ts.begin(), real_ts.end());
    ts1.push_back(real_ts[n_data - 1] + 1.0);
    sim1.resize(n_data + 1, StateD(3, 0.0));
  } else {
    two_sir = true;
    ts1 = vector<double>(real_ts.begin(), real_ts.begin() + split + 1);
    sim1.resize(split + 1, StateD(3, 0.0));
    ts2 = vector<double>(real_ts.begin() + split, real_ts.end());
    ts2.push_back(real_ts[n_data - 1] + 1.0);
    sim2.resize(n_data - split + 1, StateD(3, 0.0));
  }
  config.two_sir = two_sir;
  config.scenario = "average";

  this->split = split;
}


size_t DengueModel::get_number_of_parameters() {
  return config->get_n_params();
}


double DengueModel::compute_log_likelihood(const vector<double>& params) {
  config->params = params;
  if (two_sir) return ll_two_sir();
  return ll_one_sir();
}


double DengueModel::ll_two_sir() {
  double population = config->population;

  // Sim 1
  StateD initial_state = { // Susceptible, infectious, and already infected
    population * config->get_s0(),
    population * config->get_i0_sim1(),
    population * (config->get_i0_sim1() + config->prop_immune0)
  };
  integrate_times(
    stepper, sir, initial_state, ts1.begin(), ts1.end(), config->dt, obs1
  );

  // Sim 2 (reinitialize number of infectious)
  double still_naive = sim1[sim1.size() - 2][0];
  double infectious0_sim2 = still_naive * config->get_i0_sim2();
  double already_infected = config->population - still_naive;
  initial_state = { // Susceptible, infectious, and already infected
    still_naive,
    infectious0_sim2,
    already_infected + infectious0_sim2
  };
  integrate_times(
    stepper, sir, initial_state, ts2.begin(), ts2.end(), config->dt, obs2
  );

  double pois_ave;
  double incidence;
  size_t ind;
  double loglik = 0.0;
  double reporting = config->get_reporting();
  // Epi data contribution
  for (size_t curr = 0; curr < n_data; ++curr) {
    if (curr < split) {
      incidence = sim1[curr][0] - sim1[curr + 1][0];
    } else {
      ind = curr - split;
      incidence = sim2[ind][0] - sim2[ind + 1][0];
    }
    pois_ave = incidence * reporting;

    if (pois_ave < 1e-10) {
      loglik -= 1e20;
    } else {
      loglik += log_pois(real_incidence[curr], pois_ave);
    }
  }

  return loglik;
}


double DengueModel::ll_one_sir() {
  double population = config->population;

  // Sim 1
  StateD initial_state = { // Susceptible, infectious, and already infected
    population * config->get_s0(),
    population * config->get_i0_sim1(),
    population * (config->get_i0_sim1() + config->prop_immune0)
  };
  integrate_times(
    stepper, sir, initial_state, ts1.begin(), ts1.end(), config->dt, obs1
  );

  double pois_ave;
  double incidence;
  double loglik = 0.0;
  double reporting = config->get_reporting();
  // Epi data contribution
  for (size_t curr = 0; curr < n_data; ++curr) {
    incidence = sim1[curr][0] - sim1[curr + 1][0];
    pois_ave = incidence * reporting;

    if (pois_ave < 1e-10) {
      loglik -= 1e20;
    } else {
      loglik += log_pois(real_incidence[curr], pois_ave);
    }
  }

  return loglik;
}


void DengueModel::simulate(vector<double>& ts, string mcmc_file,
    string out_file, size_t burn_in, size_t n_param_sets, size_t n_reps,
    size_t seed) {
  // Load MCMC chain
  vector<string> chain_no_burn_in;
  load_mcmc_chain(mcmc_file, burn_in, chain_no_burn_in);

  // Check that after removing burn_in there are still enough iterations
  assert(chain_no_burn_in.size() > n_param_sets);

  // Reshuffle lines
  mt19937_64 gen(seed);
  shuffle(chain_no_burn_in.begin(), chain_no_burn_in.end(), gen);

  size_t n_steps = ts.size();
  ofstream ofs(out_file);
  ofs << "sim,type,";
  for (size_t curr = 0; curr < n_steps - 1; ++curr) {
    ofs << int(ts[curr]) << ",";
  }
  ofs << int(ts[n_steps - 1]) << endl;
  vector<StateI> epi(n_steps, StateI(3, 0));
  vector<size_t> incidence(n_steps, 0);

  size_t n_params = config->get_n_params();
  vector<double> params(n_params, 0.0);
  for (size_t curr_p = 0; curr_p < n_param_sets; ++curr_p) {
    cout << "Param set " << curr_p << endl;
    // Set parameters
    double d_to_skip;
    char c_to_skip;
    istringstream parser(chain_no_burn_in[curr_p]);
    parser >> d_to_skip >> c_to_skip; // loglik
    for (size_t curr_i = 0; curr_i < n_params; ++curr_i) {
      parser >> params[curr_i] >> c_to_skip; // params and comma
    }

    // Simulate
    simulate_epi(params, ts, gen, epi, incidence);
    for (size_t rep = 0; rep < n_reps; ++rep) {
      vector<size_t> observed = observe_incidence_pois(config, incidence, gen);
      ofs << n_reps * curr_p + rep << ",incidence,";
      for (size_t curr = 0; curr < observed.size() - 1; ++curr) {
        ofs << observed[curr] << ",";
      }
      ofs << observed[observed.size() - 1] << endl;
    }

    size_t step;

    ofs << curr_p << ",susceptible,";
    for (step = 0; step < n_steps - 1; ++step) {
      ofs << epi[step][0] << ",";
    }
    step = n_steps - 1;
    ofs << epi[step][0] << endl;

    ofs << curr_p << ",infectious,";
    for (step = 0; step < n_steps - 1; ++step) {
      ofs << epi[step][1] << ",";
    }
    step = n_steps - 1;
    ofs << epi[step][1] << endl;
  }
  ofs.close();
}


void DengueModel::simulate_epi(const vector<double>& params, vector<double>& ts,
    mt19937_64& gen, vector<StateI>& epi, vector<size_t>& incidence) {
  config->params = params;
  if (two_sir) return simulate_epi_two_sir(ts, gen, epi, incidence);
  return simulate_epi_one_sir(ts, gen, epi, incidence);
}


void DengueModel::simulate_epi_two_sir(vector<double>& ts, mt19937_64& gen,
    vector<StateI>& epi, vector<size_t>& incidence) {
  double population = config->population;
  size_t n_steps = ts.size();

  // Sim 1
  StateI initial_state = { // Susceptible, infectious, and already infected
    size_t(population * config->get_s0()),
    size_t(population * config->get_i0_sim1()),
    size_t(population * (config->get_i0_sim1() + config->prop_immune0))
  };
  vector<double> sim_ts1(ts.begin(), ts.begin() + split + 1);
  vector<StateI> epi1(split + 1, StateI(3, 0));
  simulator.simulate(initial_state, sim_ts1, epi1, gen);

  // Sim 2 (reinitialize number of infectious)
  size_t still_naive = epi1[epi1.size() - 2][0];
  size_t infectious0_sim2 = still_naive * config->get_i0_sim2();
  size_t already_infected = config->population - still_naive;
  initial_state = { // Susceptible, infectious, and already infected
    still_naive,
    infectious0_sim2,
    already_infected + infectious0_sim2
  };
  vector<double> sim_ts2(ts.begin() + split, ts.end());
  sim_ts2.push_back(ts[n_steps - 1] + 1.0);
  vector<StateI> epi2(n_steps - split + 1, StateI(3, 0));
  simulator.simulate(initial_state, sim_ts2, epi2, gen);

  // Merge simulations
  size_t ind;
  for (size_t curr = 0; curr < n_steps; ++curr) {
    if (curr < split) {
      epi[curr] = epi1[curr];
      incidence[curr] = epi1[curr][0] - epi1[curr + 1][0];
    } else {
      ind = curr - split;
      epi[curr] = epi2[ind];
      incidence[curr] = epi2[ind][0] - epi2[ind + 1][0];
    }
  }
}


void DengueModel::simulate_epi_one_sir(vector<double>& ts, mt19937_64& gen,
    vector<StateI>& epi, vector<size_t>& incidence) {
  double population = config->population;
  size_t n_steps = ts.size();

  // Sim 1
  StateI initial_state = { // Susceptible, infectious, and already infected
    size_t(population * config->get_s0()),
    size_t(population * config->get_i0_sim1()),
    size_t(population * (config->get_i0_sim1() + config->prop_immune0))
  };
  vector<double> sim_ts1(ts.begin(), ts.end());
  sim_ts1.push_back(ts[n_steps - 1] + 1.0);
  vector<StateI> epi1(n_steps + 1, StateI(3, 0));
  simulator.simulate(initial_state, sim_ts1, epi1, gen);

  for (size_t curr = 0; curr < n_steps; ++curr) {
    epi[curr] = epi1[curr];
    incidence[curr] = epi1[curr][0] - epi1[curr + 1][0];
  }
}


//------------------------------------------------------------------------------
// Observers
//------------------------------------------------------------------------------
vector<size_t> observe_incidence_pois(SIRConfiguration* config,
    const vector<size_t>& incidence, mt19937_64& gen) {
  double reporting = config->get_reporting();

  vector<size_t> res(incidence.size(), 0);
  for (size_t curr = 0; curr < incidence.size(); ++curr) {
    poisson_distribution<size_t> pois(incidence[curr] * reporting);
    res[curr] = pois(gen);
  }

  return res;
}

