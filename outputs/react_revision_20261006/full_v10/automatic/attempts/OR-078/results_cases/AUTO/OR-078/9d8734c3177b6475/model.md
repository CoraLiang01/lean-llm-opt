Mathematical Model

Sets:
  O: set of all generation contract options (option) in file_0_view_0 (energy.csv), indexed by o

Parameters (from file_0_view_0, energy.csv):
  gen_per_lot[o]: generation per lot for option o (column "gen_per_lot")
  cost_per_lot[o]: cost per lot for option o (column "cost_per_lot")

Decision Variables:
  x[o]: integer number of lots to purchase for option o, x[o] ∈ {0, 1, 2, ...}

Objective:
  Minimize total cost:
    min ∑_{o ∈ O} cost_per_lot[o] * x[o]

Subject to:
  Demand satisfaction:
    ∑_{o ∈ O} gen_per_lot[o] * x[o] ≥ 200

  Integer lot constraints:
    x[o] ∈ {0, 1, 2, ...} for all o ∈ O

Data Mapping:
  O = {option: all rows in file_0_view_0 (energy.csv) with tech ∈ {coal, gas, renewables}}
  gen_per_lot[o] = value in column "gen_per_lot" for option o in file_0_view_0
  cost_per_lot[o] = value in column "cost_per_lot" for option o in file_0_view_0

All data is mapped directly from file_0_view_0 (energy.csv) as described above.