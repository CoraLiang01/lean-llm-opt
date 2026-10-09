Sets:
  O: set of all generation options (option) in file_0_view_0 (energy.csv), indexed by o
Parameters:
  tech_o: technology type of option o (coal, gas, renewables), from column tech in file_0_view_0
  gen_per_lot_o: generation per lot for option o, from column gen_per_lot in file_0_view_0
  cost_per_lot_o: cost per lot for option o, from column cost_per_lot in file_0_view_0
  D: total demand = 200

Decision variables:
  x_o ∈ ℤ₊ number of lots purchased of option o ∈ O

Objective:
  minimize ∑_{o∈O} cost_per_lot_o · x_o

Subject to:
  ∑_{o∈O} gen_per_lot_o · x_o ≥ D

  x_o ≥ 0 and integer for all o ∈ O

Data Mapping:
  O = all rows in file_0_view_0 (energy.csv) with columns:
    option (option identifier)
    tech (coal, gas, renewables)
    gen_per_lot (generation per lot)
    cost_per_lot (cost per lot)
  D = 200 (from user description)
  x_o: integer variable for each o ∈ O

All parameters are mapped directly from file_0_view_0 (energy.csv) as described above.