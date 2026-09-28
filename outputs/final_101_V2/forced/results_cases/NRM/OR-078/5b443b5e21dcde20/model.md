### Abstract Mathematical Model

#### Index Sets
- $O$: Set of all generation options (coal, gas, renewables), indexed by $o$.

#### Parameters
- $c_o$: Cost per lot for option $o$ (from column `cost_per_lot` in `energy.csv`).
- $g_o$: Generation per lot for option $o$ (from column `gen_per_lot` in `energy.csv`).
- $D$: Total electricity demand to be met (given as 200).

#### Decision Variables
- $x_o \in \mathbb{Z}_+$: Number of lots to purchase for option $o$.

#### Objective
$$
\min \sum_{o \in O} c_o \cdot x_o
$$

#### Constraints

1. **Demand Satisfaction**
   $$
   \sum_{o \in O} g_o \cdot x_o \geq D
   $$

2. **Integer Lot Purchases**
   $$
   x_o \in \mathbb{Z}_+, \quad \forall o \in O
   $$

---

#### Data Mapping

- Table: `energy.csv` (table_id: file_0_view_0)
    - Generation option index: `option`
    - Technology type: `tech`
    - Generation per lot: `gen_per_lot`
    - Cost per lot: `cost_per_lot`
- Demand parameter $D$ is given in the user query (not in the table). 

All records for which `tech` is in $\{\text{coal}, \text{gas}, \text{renewables}\}$ are included in $O$.