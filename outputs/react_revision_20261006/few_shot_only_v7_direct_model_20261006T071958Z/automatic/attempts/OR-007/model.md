##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from warehouse $i$ to store $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. **Demand satisfaction (for each store $j$):**
   $$
   \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
   $$
2. **Supply capacity (for each warehouse $i$):**
   $$
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (warehouses): all region values in table_id="file_1_view_0", column "region"
- $J$ (stores): all customer values in table_id="file_0_view_0", column "customer"
- $d_j$: demand for store $j$ from table_id="file_0_view_0", column "demand", keyed by "customer"
- $s_i$: supply capacity for warehouse $i$ from table_id="file_1_view_0", column "supply_capacity", keyed by "region"
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ from table_id="file_2_view_0", row where "Unnamed: 0" = $i$, column $j$ (where $j$ matches "D1", "D2", etc.)

All indices, parameters, and coefficients are to be taken exactly as specified in the current Observation.