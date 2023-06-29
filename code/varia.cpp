#include "varia.hpp"
using namespace std;


//------------------------------------------------------------------------------
// Log-densities and random number generation
//------------------------------------------------------------------------------
double log_binom(size_t k, size_t n, double p) {
  // Log of binomial probability mass function
  double res = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  res += k * log(p);
  res += (n - k) * log(1.0 - p);
  return res;
}


double log_neg_binom(size_t k, double m, double r) {
  // Log of negative binomial pmf
  double res = lgamma(r + k) - lgamma(k + 1) - lgamma(r);
  res += r * log(r / (r + m));
  res += k * log(m / (r + m));
  return res;
}


double log_pois(size_t k, double lambda) {
  // Log of poisson pmf
  return k * log(lambda) - lambda - lgamma(k + 1);
}


double draw_neg_binom(double m, double r, mt19937_64& gen) {
  // This exploits the fact that negative binomial distribution can be viewed
  // as a Gamma-Poisson mixture, i.e. a Poisson distribution with a mean
  // distributed as a gamma distribution
  // m = mean
  // r = overdispersion parameter
  double proba = r / (r + m);
  gamma_distribution<> gamma(r, (1.0 - proba) / proba);
  poisson_distribution<> pois(gamma(gen));
  return pois(gen);
}


//------------------------------------------------------------------------------
// Loading data
//------------------------------------------------------------------------------
void load_epi(string fname, vector<double>& ts,
    vector<size_t>& real_incidence, size_t up_to) {
  ifstream ifs(fname);
  if (!ifs.is_open()) {
    cout << "ERROR: file " << fname << " does not exist!" << endl;
    exit(1);
  }

  // Read data
  size_t curr_time;
  size_t incidence;
  string line;
  while (getline(ifs, line)) {
    if (line == "") continue;
    istringstream parser(line);
    parser >> curr_time;
    parser >> incidence;
    if (curr_time <= up_to) {
      ts.push_back(curr_time);
      real_incidence.push_back(incidence);
    }
  }
  ifs.close();
}


void load_mcmc_chain(std::string fname, size_t burn_in,
    vector<string>& chain_no_burn_in) {
  ifstream ifs(fname);
  if (!ifs.is_open()) {
    cout << "ERROR: file " << fname << " does not exist!" << endl;
    exit(1);
  }

  size_t line_count = 0;
  string line;
  while (!ifs.eof()) {
    getline(ifs, line);
    if (line_count > burn_in && line != "") {
      chain_no_burn_in.push_back(line);
    }
    ++line_count;
  }
  ifs.close();
}


//------------------------------------------------------------------------------
// Temperature and rainfall fit
//------------------------------------------------------------------------------
double average_temperature(double iweek) {
  double phase = TWO_PI * (iweek + 1.0) / 52.0;
  return 22.56378 + 2.27516 * cos(phase) + 2.015281 * sin(phase);
}


double cold_temperature(double iweek) {
  // Sinusoidal fit for coldest year (2005)
  double phase = TWO_PI * (iweek + 1.0) / 52.0;
  return 22.19837 + 2.42396 * cos(phase) + 2.282158 * sin(phase);
}


double warm_temperature(double iweek) {
  // Sinusoidal fit for warmest year (2015)
  double phase = TWO_PI * (iweek + 1.0) / 52.0;
  return 22.9144 + 2.049641 * cos(phase) + 1.928841 * sin(phase);
}


double average_rainfall(double iweek) {
  double phase = TWO_PI * (iweek + 1.0) / 52.0;
  return 6.480926 + 2.687326 * cos(phase) + 3.727384 * sin(phase);
}


//------------------------------------------------------------------------------
// Transmission scaling factors
//------------------------------------------------------------------------------
double briere(double temperature, double c, double t_min, double t_max) {
  double res = 0.0;
  if (temperature > t_min && temperature < t_max) {
    res = c * temperature * (temperature - t_min) * sqrt(t_max - temperature);
  }
  return res;
}


double quadratic(double temperature, double c, double t_min, double t_max) {
  double res = 0.0;
  if (temperature > t_min && temperature < t_max) {
    res = c * (temperature - t_min) * (t_max - temperature);
  }
  return res;
}


double lambrechts_scaling(double temperature, double rainfall) {
  // Taken from: L. Lambrechts et al, PNAS 108 (2011)
  // Squared version
  double res;
  if (temperature < 12.286 || temperature > 32.461) {
    res = 0.0;
  } else {
    res = 0.001044 * temperature * (temperature - 12.286) *
      sqrt(32.461 - temperature);
  }
  return res * res;
}


double perkins_scaling(double temperature, double rainfall) {
  // Taken from: T. A. Perkins et al, PLOS Currents Outbreaks (2015)
  double log_res = -25.66 + 2.121 * temperature + 1.188e-2 * rainfall
    - 4.231e-2 * temperature * temperature - 2.882e-5 * rainfall * rainfall;
  return exp(log_res);
}


double mordecai_scaling(double temperature, double rainfall) {
  // Taken from: E. A. Mordecai et al, PLoS Negl Trop Dis 11 (2017)
  // Parameters are those for Aedes albopictus
  double a = briere(temperature, 1.93e-4, 10.25, 38.32);
  double efd = briere(temperature, 4.88e-2, 8.02, 35.65);
  double pea = quadratic(temperature, 3.61e-3, 9.04, 39.33);
  double mdr = briere(temperature, 6.38e-5, 8.60, 39.66);
  double lf = quadratic(temperature, 1.43, 13.41, 31.51);
  double b = briere(temperature, 7.35e-4, 15.84, 36.40);
  double c = briere(temperature, 4.39e-4, 3.62, 36.82);
  double pdr = briere(temperature, 1.09e-4, 10.39, 43.05);

  double res = a * a * b * c;
  if (lf > 0.0 && pdr > 0.0) res *= exp(-1.0 / (lf * pdr));
  res *= efd * pea * mdr * lf * lf * lf;

  return sqrt(res) / 480.815; // So that max scaling = 1 (for T = 26.14)
}


double null_scaling(double temperature, double rainfall) {
  return temperature / 22.56378;
}

