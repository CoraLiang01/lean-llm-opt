##### Mathematical Model

Let $S$ be the set of warehouses (indexed by $i$), and $D$ the set of stores (indexed by $j$):

- $S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $D = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in S$ to store $j \in D$ (continuous variable)
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$
- $d_j$: demand of store $j$
- $s_i$: supply capacity of warehouse $i$

**Objective:**
\[
\min \sum_{i \in S} \sum_{j \in D} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in S} x_{ij} \geq d_j \qquad \forall j \in D
   \]

2. **Supply capacity (each warehouse's shipments cannot exceed its capacity):**
   \[
   \sum_{j \in D} x_{ij} \leq s_i \qquad \forall i \in S
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in S,\, j \in D
   \]

---

##### Data Mapping

- $S$ (warehouses): All unique `supplier_id` in `file_1_view_0` (supply_capacity.csv), source order: S1, S2, S3, S4, S5.
- $D$ (stores): All unique `customer_id` in `file_0_view_0` (customer_demand.csv), source order: D1, D2, D3, D4, D5.
- $d_j$: `demand_units` from `file_0_view_0`, mapped by `customer_id`.
- $s_i$: `supply_capacity_units` from `file_1_view_0`, mapped by `supplier_id`.
- $c_{ij}$: For each $i$ (row, `supplier_id` in `file_2_view_0`), and $j$ (column, `transportation_cost_to_{customer_id}`), the value in `file_2_view_0` (transportation_costs.csv).

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving source order and identifiers. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.