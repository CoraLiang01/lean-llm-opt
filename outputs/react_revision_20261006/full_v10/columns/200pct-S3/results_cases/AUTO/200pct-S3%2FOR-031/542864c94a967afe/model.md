Sets:
  O: set of all generation contract options (option) from file_0_view_0 (energy.csv)
Parameters:
  gen_per_lot[o]: generation per lot for option o (from column gen_per_lot, file_0_view_0)
  cost_per_lot[o]: cost per lot for option o (from column cost_per_lot, file_0_view_0)
  tech[o]: technology type for option o (from column tech, file_0_view_0)
  D: total demand to be met (D = 200)
Decision variables:
  x[o]: integer number of lots to purchase for option o, x[o] ≥ 0 and integer, ∀ o ∈ O

Objective:
  minimize  ∑_{o ∈ O} cost_per_lot[o] * x[o]

Subject to:
  ∑_{o ∈ O} gen_per_lot[o] * x[o] ≥ D

  x[o] ∈ {0, 1, 2, ...}   ∀ o ∈ O

Data Mapping:
  O = all option values in file_0_view_0 (energy.csv), column option
  gen_per_lot[o], cost_per_lot[o], tech[o] from file_0_view_0, columns gen_per_lot, cost_per_lot, tech, indexed by option
  D = 200 (from user description)
  x[o]: integer variable for each o ∈ O

All contract options in energy.csv are available for selection. Each lot must be purchased in whole units. The total generation from all selected lots must meet or exceed the demand of 200. The objective is to minimize total cost.