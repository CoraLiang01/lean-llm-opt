#### Index Sets
- $O$: Set of all generation options (coal, gas, renewables), indexed by $o$.

#### Parameters
- $c_o$: Cost per lot for option $o$ (from column "cost_per_lot" in table_id: file_0_view_0).
- $g_o$: Generation per lot for option $o$ (from column "gen_per_lot" in table_id: file_0_view_0).
- $D$: Total demand to be met (given as 200).

#### Decision Variables
- $x_o \in \mathbb{Z}_+, \quad \forall o \in O$: Number of lots to purchase of option $o$ (must be a non-negative integer).

#### Objective
$$
\min \sum_{o \in O} c_o \cdot x_o
$$

#### Constraints

1. **Demand Satisfaction**
   $$
   \sum_{o \in O} g_o \cdot x_o \geq D
   $$

2. **Integrality**
   $$
   x_o \in \mathbb{Z}_+, \quad \forall o \in O
   $$

---

#### Data Mapping

- All data is from table_id: file_0_view_0 (energy.csv).
    - Generation options: "option"
    - Technology type: "tech"
    - Generation per lot: "gen_per_lot"
    - Cost per lot: "cost_per_lot"
- No additional filters were applied; all records for coal, gas, and renewables are included as returned by CSVQA.
- Demand parameter $D$ is given in the user query (not from the table).