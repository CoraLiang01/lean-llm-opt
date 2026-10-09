## Mathematical Model

**Sets**
- $O$: set of all generation options (option) in file_0_view_0 (energy.csv), $|O|=131$
- For each $o \in O$, let $\text{tech}_o \in \{\text{coal}, \text{gas}, \text{renewables}\}$

**Parameters** (from file_0_view_0, energy.csv)
- $\text{gen\_per\_lot}_o$: generation per lot for option $o$ (column "gen_per_lot")
- $\text{cost\_per\_lot}_o$: cost per lot for option $o$ (column "cost_per_lot")
- $D$: total demand to meet, $D=200$

**Decision Variables**
- $x_o \in \mathbb{Z}_+$: number of lots to purchase for option $o \in O$

**Objective**
\[
\min \sum_{o \in O} \text{cost\_per\_lot}_o \cdot x_o
\]

**Constraints**
\[
\sum_{o \in O} \text{gen\_per\_lot}_o \cdot x_o \geq D
\]
\[
x_o \in \mathbb{Z}_+, \quad \forall o \in O
\]

**Data Mapping**
- $O$: All rows in file_0_view_0 (energy.csv) with columns: option, tech, gen_per_lot, cost_per_lot
- $\text{gen\_per\_lot}_o$: file_0_view_0, column "gen_per_lot", row $o$
- $\text{cost\_per\_lot}_o$: file_0_view_0, column "cost_per_lot", row $o$
- $x_o$: integer variable for each $o$ in $O$
- $D$: demand, given as 200 in the question

**Summary**
- Choose integer numbers of lots $x_o$ for each available contract $o$ (coal, gas, renewables), to minimize total cost, such that total generation meets or exceeds 200. All data is mapped directly from file_0_view_0 (energy.csv).