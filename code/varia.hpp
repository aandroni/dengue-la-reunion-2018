#ifndef varia_hpp_
#define varia_hpp_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <cassert>
#include <unordered_map>

//------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------
const double PI = 3.14159265358979323846;
const double TWO_PI = 2.0 * PI;
const double SQRT_2_PI = sqrt(2.0 * PI);

//------------------------------------------------------------------------------
// Log-densities and random number generation
//------------------------------------------------------------------------------
double log_binom(size_t k, size_t n, double p);
double log_neg_binom(size_t k, double m, double r);
double log_pois(size_t k, double lambda);
double draw_neg_binom(double m, double r, std::mt19937_64& gen);

//------------------------------------------------------------------------------
// Loading data
//------------------------------------------------------------------------------
void load_epi(std::string fname, std::vector<double>& ts,
  std::vector<size_t>& real_incidence, size_t up_to = -1);
void load_mcmc_chain(std::string fname, size_t burn_in,
  std::vector<std::string>& chain_no_burn_in);

//------------------------------------------------------------------------------
// Temperature and rainfall fit
//------------------------------------------------------------------------------
double average_temperature(double iweek);
double cold_temperature(double iweek);
double warm_temperature(double iweek);
double average_rainfall(double iweek);

//------------------------------------------------------------------------------
// Transmission scaling factors
//------------------------------------------------------------------------------
double briere(double temperature, double c, double t_min, double t_max);
double quadratic(double temperature, double c, double t_min, double t_max);
double lambrechts_scaling(double temperature, double rainfall);
double perkins_scaling(double temperature, double rainfall);
double mordecai_scaling(double temperature, double rainfall);
double null_scaling(double temperature, double rainfall);

#endif // varia_hpp_