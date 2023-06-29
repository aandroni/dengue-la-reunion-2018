# Dengue in La Reunion

Data and source code to reproduce the analyses from [this article]().

## Dependencies
1. A C++ compiler, like `gcc`
2. GNU `make`
3. The [boost](https://www.boost.org) C++ libraries
4. R (optional, for visualization)

## How to run the code
* `cd` into the `code` folder.
* Compile with: `make`
* Run with the desired options (described in `main.cpp`), for example:
```
./dengue.exe data20190823.txt lamb 15 11 0.15 0.1 10000 46
```
* The program will generate 4 output files:
    1. `output.csv`: MCMC output (log-likelihood, parameters, and acceptance rates)
    2. `output_sims.csv`: Simulations (average temperature scenario)
    3. `output_sims_cold.csv`: Simulations (cold temperature scenario)
    4. `output_sims_warm.csv`: Simulations (warm temperature scenario)

## Note
* The `data` folder contains the epidemiological and climate data used for the analyses
* The `output` folder contains an example of visualization of simulated trajectories
