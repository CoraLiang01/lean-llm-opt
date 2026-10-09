## Mathematical Model

**Sets**
- $O$: set of all generation options (option) from table_id = file_0_view_0

**Parameters (from file_0_view_0)**
- $c_o$: cost per lot for option $o$ ($\text{cost\_per\_lot}$)
- $g_o$: generation per lot for option $o$ ($\text{gen\_per\_lot}$)
- $D$: total demand $= 200$

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: number of lots to purchase for option $o \in O$

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

### Data Mapping

- $O$: All rows in file_0_view_0 (energy.csv) with columns: option, tech, gen_per_lot, cost_per_lot
- $c_o$: file_0_view_0, column "cost_per_lot", for each $o$
- $g_o$: file_0_view_0, column "gen_per_lot", for each $o$
- $x_o$: integer variable for each $o$ in $O$
- $D$: 200 (from user description)

**Constraint and variable domains are unconditional. All options in the file are available for selection.**