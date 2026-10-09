Mathematical Model

Sets:
  O: set of all generation options (option) from file_0_view_0 (energy.csv), indexed by o
Parameters:
  tech_o: technology type of option o (coal, gas, renewables), from column tech in file_0_view_0
  gen_per_lot_o: generation per lot for option o, from column gen_per_lot in file_0_view_0
  cost_per_lot_o: cost per lot for option o, from column cost_per_lot in file_0_view_0
  D: total demand to be met (scalar, 200)

Decision Variables:
  x_o ∈ {0, 1, 2, ...} for all o ∈ O (number of lots to purchase of option o; integer and nonnegative)

Objective:
  Minimize total cost:
    min ∑_{o ∈ O} cost_per_lot_o · x_o

Constraints:
  1. Demand satisfaction:
    ∑_{o ∈ O} gen_per_lot_o · x_o ≥ D

  2. Integer and nonnegativity:
    x_o ∈ {0, 1, 2, ...} for all o ∈ O

Data Mapping:
  - O: All rows in file_0_view_0 (energy.csv) with columns option, tech, gen_per_lot, cost_per_lot
  - tech_o: file_0_view_0, column tech, for each o
  - gen_per_lot_o: file_0_view_0, column gen_per_lot, for each o
  - cost_per_lot_o: file_0_view_0, column cost_per_lot, for each o
  - D: 200 (from user description)
  - x_o: integer variable for each o ∈ O

All generation options (coal, gas, renewables) are available; each lot must be purchased in whole units. The model selects the number of lots of each option to minimize total cost while meeting or exceeding the total demand.