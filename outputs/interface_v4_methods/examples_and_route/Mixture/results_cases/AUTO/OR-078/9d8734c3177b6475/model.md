#### Abstract Mathematical Model

**Index Sets:**
- $O$: set of generation options (option), from energy.csv, e.g., coal_001, gas_001, renewables_001, etc.

**Parameters:**
- $c_o$: cost per lot for option $o$ (cost_per_lot, table_id: file_0_view_0)
- $g_o$: generation per lot for option $o$ (gen_per_lot, table_id: file_0_view_0)
- $D$: total demand to meet (given as 200 in the user description)

**Decision Variables:**
- $x_o \in \mathbb{Z}_{\geq 0}$: number of lots to purchase for option $o \in O$

**Objective:**
\[
\min \sum_{o \in O} c_o \, x_o
\]

**Constraints:**
\[
\sum_{o \in O} g_o \, x_o \geq D
\]
\[
x_o \in \mathbb{Z}_{\geq 0} \quad \forall o \in O
\]

---

#### Data Mapping

- $O$: All records in energy.csv with columns:
    - option (business identifier, e.g., coal_001, gas_001, renewables_001)
    - tech (coal, gas, renewables)
    - gen_per_lot (parameter $g_o$)
    - cost_per_lot (parameter $c_o$)
    - table_id: file_0_view_0
- $D$: 200 (from user description)

---

**Every generation option $o$ from energy.csv appears in the objective and the demand constraint, with its own cost and generation per lot. The variables $x_o$ are nonnegative integers representing the number of lots of each option to purchase.**