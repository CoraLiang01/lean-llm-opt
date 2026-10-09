Mathematical Model

Sets:
L: set of all generation lot options, indexed by l (from energy.csv, column option)
Parameters:
gen_per_lot_l: generation per lot for lot l (energy.csv, column gen_per_lot, units consistent with demand)
cost_per_lot_l: cost per lot for lot l (energy.csv, column cost_per_lot)
D: total demand to be met (D = 200, from question)

Decision Variables:
x_l ∈ {0,1,2,...} for all l ∈ L
 (Number of lots of option l to purchase; integer and nonnegative)

Objective:
minimize ∑_{l∈L} cost_per_lot_l · x_l

Subject to:
∑_{l∈L} gen_per_lot_l · x_l ≥ D

x_l ∈ {0,1,2,...} for all l ∈ L

Data Mapping:
Set L: All rows in energy.csv, table_id file_0_view_0, column option
gen_per_lot_l: file_0_view_0, column gen_per_lot, for each l ∈ L
cost_per_lot_l: file_0_view_0, column cost_per_lot, for each l ∈ L
D: 200 (from question)

All lots are available for selection; each lot can be purchased any integer number of times (including zero). The model minimizes total cost while meeting or exceeding the required demand.