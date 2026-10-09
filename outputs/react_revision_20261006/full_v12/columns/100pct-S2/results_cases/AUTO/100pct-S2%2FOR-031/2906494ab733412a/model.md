## Mathematical Model

**Sets**
- $O$: set of all generation options (option) in energy.csv, $|O|=131$

**Parameters** (from energy.csv, table_id: file_0_view_0)
- $c_o$: cost per lot for option $o \in O$ (cost_per_lot)
- $g_o$: generation per lot for option $o \in O$ (gen_per_lot)
- $D$: total demand to meet, $D = 200$

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: number of lots to purchase for option $o \in O$

**Objective**
\[
\min \sum_{o \in O} c_o\, x_o
\]

**Constraint**
\[
\sum_{o \in O} g_o\, x_o \geq D
\]

\[
x_o \in \mathbb{Z}_+ \quad \forall o \in O
\]

---

**Data Mapping**
- $O$: All rows in energy.csv with columns [option, tech, gen_per_lot, cost_per_lot], table_id: file_0_view_0
- $c_o$: column cost_per_lot, table_id: file_0_view_0
- $g_o$: column gen_per_lot, table_id: file_0_view_0
- $x_o$: integer variable for each $o \in O$
- $D$: scalar, 200 (from user description)

**Summary**
- Minimize total cost of purchased lots.
- Meet or exceed total demand of 200 units.
- Each option can be selected in any nonnegative integer number of lots.