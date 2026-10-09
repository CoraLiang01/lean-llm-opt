Mathematical Model

Sets:
O: set of all generation options (option) from file_0_view_0 (energy.csv), indexed by o

Parameters:
gen_per_lot[o]: generation per lot for option o (from column gen_per_lot, file_0_view_0)
cost_per_lot[o]: cost per lot for option o (from column cost_per_lot, file_0_view_0)
demand: total required generation = 200

Decision Variables:
x[o]: integer number of lots to purchase for option o, x[o] ∈ {0, 1, 2, ...}

Objective:
minimize total cost:
min ∑_{o ∈ O} cost_per_lot[o] * x[o]

Subject to:
∑_{o ∈ O} gen_per_lot[o] * x[o] ≥ demand

x[o] ∈ {0, 1, 2, ...} for all o ∈ O

Data Mapping:
- O: all rows in file_0_view_0 (energy.csv) with columns option, tech, gen_per_lot, cost_per_lot
- gen_per_lot[o]: file_0_view_0, column gen_per_lot, row o
- cost_per_lot[o]: file_0_view_0, column cost_per_lot, row o
- demand: 200 (from user description)
- x[o]: integer variable for each o ∈ O

All generation options (coal, gas, renewables) listed in file_0_view_0 are available for selection. Each lot must be purchased in whole units. The total scheduled generation must meet or exceed 200. The objective is to minimize total procurement cost.