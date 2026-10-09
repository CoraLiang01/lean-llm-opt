Mathematical Model

Index Sets:
- $O$: set of generation options (option), from energy.csv
- Each $o \in O$ has attributes: technology type (tech), generation per lot ($g_o$), cost per lot ($c_o$)

Parameters:
- $g_o$: generation per lot for option $o$ (energy.csv: gen_per_lot)
- $c_o$: cost per lot for option $o$ (energy.csv: cost_per_lot)
- $D$: total demand to meet (given as 200)

Decision Variables:
- $x_o \in \mathbb{Z}_{\geq 0}$: number of lots to purchase for option $o$

Objective:
\[
\min \sum_{o \in O} c_o \, x_o
\]

Subject to:
\[
\sum_{o \in O} g_o \, x_o \geq D
\]
\[
x_o \in \mathbb{Z}_{\geq 0} \quad \forall o \in O
\]

Data Mapping

- $O$: All rows in energy.csv, column option (table_id: file_0_view_0, column: option)
- $g_o$: energy.csv, column gen_per_lot (table_id: file_0_view_0, column: gen_per_lot)
- $c_o$: energy.csv, column cost_per_lot (table_id: file_0_view_0, column: cost_per_lot)
- $x_o$: integer variable for each $o \in O$
- $D$: 200 (from user description)

All generation options, costs, and lot sizes are mapped directly from the returned rows of energy.csv. The model minimizes total cost while ensuring total generation meets or exceeds the required demand, with integer lot decisions.