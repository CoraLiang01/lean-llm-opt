#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$
- $d_j$: demand of store $j$
- $s_i$: supply capacity of warehouse $i$

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]

2. **Supply capacity (each warehouse):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (warehouses): all $\text{supplier_id}$ in `supply_capacity.csv` ([table_id: file_1_view_0], column: supplier_id)
- $J$ (stores): all $\text{customer_id}$ in `customer_demand.csv` ([table_id: file_0_view_0], column: customer_id)
- $d_j$: demand of store $j$ from `customer_demand.csv` ([table_id: file_0_view_0], column: demand_units, key: customer_id)
- $s_i$: supply capacity of warehouse $i$ from `supply_capacity.csv` ([table_id: file_1_view_0], column: supply_capacity_units, key: supplier_id)
- $c_{ij}$: transportation cost from warehouse $i$ to store $j$ from `transportation_costs.csv` ([table_id: file_2_view_0], row: supplier_id, columns: transportation_cost_to_D1, ..., transportation_cost_to_D5, mapped to $j$ by column suffix)

All index sets, parameters, and coefficients are defined exactly by the current CSV data and their identifiers. No data is omitted or aggregated.