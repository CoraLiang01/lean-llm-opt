Mathematical Model

Index Sets:
- $O$: set of generation options (option) from file_0_view_0 (energy.csv), each with technology type (tech)
- $T$: set of technology types (tech) from file_0_view_0 (coal, gas, renewables)

Parameters:
- $c_o$: cost per lot for option $o$ (cost_per_lot, file_0_view_0)
- $g_o$: generation per lot for option $o$ (gen_per_lot, file_0_view_0)
- $D$: total demand to be met (given as 200 in the user description)

Decision Variables:
- $x_o \in \mathbb{Z}_{\geq 0}$: number of lots to purchase of option $o \in O$

Objective:
\[
\min \sum_{o \in O} c_o x_o
\]

Subject to:
\[
\sum_{o \in O} g_o x_o \geq D
\]
\[
x_o \in \mathbb{Z}_{\geq 0} \quad \forall o \in O
\]

Data Mapping

- $O$: All rows with column "option" in table_id file_0_view_0 (energy.csv)
- $T$: All unique values in column "tech" in table_id file_0_view_0 (energy.csv)
- $c_o$: column "cost_per_lot" in table_id file_0_view_0, indexed by "option"
- $g_o$: column "gen_per_lot" in table_id file_0_view_0, indexed by "option"
- $D$: 200 (from user description)
- $x_o$: integer variable for each "option" in file_0_view_0

All data is mapped directly from file_0_view_0 (energy.csv) as returned above.