Sets:
  O: set of all generation options (option) from file_0_view_0 (energy.csv)
Parameters:
  gen_per_lot[o]: generation per lot for option o (from column gen_per_lot, file_0_view_0)
  cost_per_lot[o]: cost per lot for option o (from column cost_per_lot, file_0_view_0)
  demand: total required generation = 200

Decision variables:
  x[o]: integer number of lots to purchase for option o, x[o] ∈ {0,1,2,...}

Objective:
  minimize  sum_{o ∈ O} cost_per_lot[o] * x[o]

Constraint:
  sum_{o ∈ O} gen_per_lot[o] * x[o] ≥ demand

Variable domains:
  x[o] ∈ {0,1,2,...}  for all o ∈ O

Data Mapping:
  O = all option values in column option, file_0_view_0 (energy.csv)
  gen_per_lot[o] = value in column gen_per_lot for option o, file_0_view_0
  cost_per_lot[o] = value in column cost_per_lot for option o, file_0_view_0

Complete model:
minimize   ∑_{o ∈ O} cost_per_lot[o] * x[o]
subject to ∑_{o ∈ O} gen_per_lot[o] * x[o] ≥ 200
            x[o] ∈ {0,1,2,...}   ∀ o ∈ O

where all parameters and sets are mapped as above from file_0_view_0 (energy.csv).