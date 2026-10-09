Sets:
  L: set of all lot options (option) from file_0_view_0 (energy.csv), |L| = 131
Parameters:
  gen_per_lot_l: generation per lot for lot l ∈ L, from column gen_per_lot in file_0_view_0
  cost_per_lot_l: cost per lot for lot l ∈ L, from column cost_per_lot in file_0_view_0
  demand = 200 (total generation required)
Decision variables:
  x_l ∈ {0,1,2,...} for all l ∈ L (number of lots of option l to purchase; integer, nonnegative)

Objective:
  minimize ∑_{l∈L} cost_per_lot_l · x_l

Subject to:
  ∑_{l∈L} gen_per_lot_l · x_l ≥ demand

  x_l ∈ {0,1,2,...} for all l ∈ L

Data Mapping:
  L = {option: all 131 rows in file_0_view_0 (energy.csv)}
  gen_per_lot_l = file_0_view_0.gen_per_lot for each l
  cost_per_lot_l = file_0_view_0.cost_per_lot for each l

All lots are available in integer multiples (including zero). Each lot l is uniquely identified by its option code. The demand constraint ensures total scheduled generation meets or exceeds 200 units. The objective is to minimize total cost.