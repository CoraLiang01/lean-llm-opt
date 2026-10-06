**Abstract Mathematical Model**

**Index Sets**
- $O$: Set of generation options (option), from all rows in energy.csv.
- Each $o \in O$ has attributes: technology type (coal, gas, renewables), generation per lot, and cost per lot.

**Parameters**
- $g_o$: Generation per lot for option $o$ (from column gen_per_lot in energy.csv).
- $c_o$: Cost per lot for option $o$ (from column cost_per_lot in energy.csv).
- $D$: Total demand to be met (given as 200).

**Decision Variables**
- $x_o \in \mathbb{Z}_{\geq 0}$: Number of lots to purchase for option $o \in O$.

**Objective**
\[
\min \sum_{o \in O} c_o \, x_o
\]

**Constraints**
1. **Demand Satisfaction**
   \[
   \sum_{o \in O} g_o \, x_o \geq D
   \]
2. **Integrality**
   \[
   x_o \in \mathbb{Z}_{\geq 0} \quad \forall o \in O
   \]

---

**Data Mapping**

- $O$: All records in table_id = file_0_view_0, column option.
- $g_o$: file_0_view_0, column gen_per_lot, keyed by option.
- $c_o$: file_0_view_0, column cost_per_lot, keyed by option.
- $x_o$: Decision variable for each $o \in O$.
- $D$: 200 (from query).

**Notes**
- All 131 rows in energy.csv with tech in {coal, gas, renewables} are included.
- Each $x_o$ is a nonnegative integer, representing the number of lots of option $o$ to purchase.
- No additional constraints are imposed beyond those specified.