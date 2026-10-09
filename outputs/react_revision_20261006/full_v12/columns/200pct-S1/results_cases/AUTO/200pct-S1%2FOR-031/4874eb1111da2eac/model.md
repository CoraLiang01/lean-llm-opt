## Mathematical Model

**Sets**
- $O$: Set of all generation options (from energy.csv, column "option")
- For each $o \in O$:
    - $g_o$: generation technology of option $o$ (from "tech")
    - $G_o$: generation per lot for option $o$ (from "gen_per_lot")
    - $C_o$: cost per lot for option $o$ (from "cost_per_lot")

**Parameters**
- $D$: total demand to meet ($D = 200$)

**Decision Variables**
- $x_o \in \mathbb{Z}_+ \quad \forall o \in O$  
  (number of lots to purchase for option $o$; integer and nonnegative)

**Objective**
\[
\min \sum_{o \in O} C_o\, x_o
\]

**Subject to**
\[
\sum_{o \in O} G_o\, x_o \geq D
\]
\[
x_o \in \mathbb{Z}_+ \quad \forall o \in O
\]

**Data Mapping**
- $O$: All rows in energy.csv with "tech" in $\{\text{coal}, \text{gas}, \text{renewables}\}$ (table_id: file_0_view_0, column: "option")
- $g_o$: file_0_view_0, column: "tech"
- $G_o$: file_0_view_0, column: "gen_per_lot"
- $C_o$: file_0_view_0, column: "cost_per_lot"
- $D$: 200 (from user description)

**Notes**
- Each $x_o$ is integer and nonnegative.
- All options in the data are available for selection; no further restrictions are imposed.
- The model minimizes total cost while meeting or exceeding the required demand.