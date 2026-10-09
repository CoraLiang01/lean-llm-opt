## Mathematical Model

**Sets**
- $O$: Set of all generation options (from energy.csv, column "option")
- For each $o \in O$:
    - $tech_o$: Technology type of option $o$ ("coal", "gas", "renewables")
    - $g_o$: Generation per lot for option $o$ (from "gen_per_lot")
    - $c_o$: Cost per lot for option $o$ (from "cost_per_lot")

**Parameters**
- $D = 200$: Total demand to be met

**Decision Variables**
- $x_o \in \mathbb{Z}_+ \quad \forall o \in O$: Number of lots to purchase of option $o$ (integer, nonnegative)

**Objective**
\[
\min \sum_{o \in O} c_o\, x_o
\]

**Subject to**
\[
\sum_{o \in O} g_o\, x_o \geq D
\]
\[
x_o \in \mathbb{Z}_+, \quad \forall o \in O
\]

---

**Data Mapping**

- $O$: All rows in energy.csv, column "option" (table_id: file_0_view_0, column: option)
- $tech_o$: energy.csv, column "tech" (table_id: file_0_view_0, column: tech)
- $g_o$: energy.csv, column "gen_per_lot" (table_id: file_0_view_0, column: gen_per_lot)
- $c_o$: energy.csv, column "cost_per_lot" (table_id: file_0_view_0, column: cost_per_lot)
- $x_o$: Integer variable for each $o \in O$
- $D$: Demand, given as 200 in the question

**Notes**
- All lots must be purchased in integer quantities.
- All options in the file are available for selection.
- No upper bound on $x_o$ unless specified in the data (none present).
- Only the total generation constraint and cost minimization are required.