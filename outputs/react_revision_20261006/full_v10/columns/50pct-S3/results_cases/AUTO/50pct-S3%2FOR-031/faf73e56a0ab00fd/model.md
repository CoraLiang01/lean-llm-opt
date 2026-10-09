Sets:
  O: set of all contract options (option) from file_0_view_0 (energy.csv)
  For each o in O:
    tech[o]: technology type (coal, gas, renewables) from column tech
    gen_per_lot[o]: generation per lot (numeric, units consistent with demand) from column gen_per_lot
    cost_per_lot[o]: cost per lot (numeric, same units as objective) from column cost_per_lot

Parameters:
  D: total demand to be met (scalar, D = 200)

Decision variables:
  x[o]: integer number of lots to purchase of option o, for all o in O
    Domain: x[o] ∈ {0, 1, 2, ...}

Objective:
  Minimize total cost:
    min ∑_{o ∈ O} cost_per_lot[o] * x[o]

Constraints:
  1. Demand satisfaction:
    ∑_{o ∈ O} gen_per_lot[o] * x[o] ≥ D

  2. Integrality:
    x[o] ∈ {0, 1, 2, ...}  for all o ∈ O

Data Mapping:
  - O: All rows in file_0_view_0 (energy.csv), column option
  - tech[o]: file_0_view_0, column tech, indexed by option
  - gen_per_lot[o]: file_0_view_0, column gen_per_lot, indexed by option
  - cost_per_lot[o]: file_0_view_0, column cost_per_lot, indexed by option
  - D: 200 (from user description)

Summary:
  min ∑_{o ∈ O} cost_per_lot[o] * x[o}
  s.t. ∑_{o ∈ O} gen_per_lot[o] * x[o] ≥ 200
       x[o] ∈ {0, 1, 2, ...} ∀ o ∈ O

All data and indices are mapped directly from file_0_view_0 (energy.csv) as described above.