#### Mathematical Model

Let:
- $O$ = set of all generation options (option), as listed in energy.csv, each with technology type (tech), generation per lot ($g_o$), and cost per lot ($c_o$).
- $x_o$ = number of lots to purchase of option $o \in O$ (decision variable, integer, $x_o \geq 0$).

Parameters:
- $g_o$ = gen_per_lot for option $o$ (from energy.csv)
- $c_o$ = cost_per_lot for option $o$ (from energy.csv)
- $D$ = total demand to meet (given as 200)

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

#### Data Mapping

- $O$: All rows in table_id file_0_view_0, column "option"
- $g_o$: file_0_view_0, column "gen_per_lot", keyed by "option"
- $c_o$: file_0_view_0, column "cost_per_lot", keyed by "option"
- $x_o$: Decision variable for each "option" in file_0_view_0
- $D$: 200 (from user query)

All generation options, costs, and lot sizes are taken directly from file_0_view_0 (energy.csv), preserving original row and option identifiers. The model minimizes total cost while ensuring total generation meets or exceeds the required demand, with integer lot decisions.