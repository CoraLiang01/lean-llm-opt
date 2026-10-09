#### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous)
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$
- $d_j$: demand at store $j$
- $s_i$: supply capacity at warehouse $i$

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

- $I$ (warehouses): all $\texttt{supplier_id}$ in table_id: file_1_view_0 (supply_capacity.csv)
- $J$ (stores): all $\texttt{customer_id}$ in table_id: file_0_view_0 (customer_demand.csv)
- $d_j$: $\texttt{demand_units}$ for $j$ in table_id: file_0_view_0 (customer_demand.csv)
- $s_i$: $\texttt{supply_capacity_units}$ for $i$ in table_id: file_1_view_0 (supply_capacity.csv)
- $c_{ij}$: $\texttt{transportation_cost_to_{j}}$ for $i$ in $\texttt{supplier_id}$, $j$ in $\texttt{customer_id}$, table_id: file_2_view_0 (transportation_costs.csv)
- $x_{ij}$: decision variable for each $(i,j)$ pair as above

All index sets, parameters, and constraints are defined directly from the current CSV data, preserving all identifiers and bounds.