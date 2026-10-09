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

1. **Demand satisfaction (each store's demand is met):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]

2. **Supply capacity (each warehouse's shipments do not exceed its capacity):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (warehouses): region column in `file_1_view_0` (supply_capacity.csv)
- $J$ (stores): customer column in `file_0_view_0` (customer_demand.csv)
- $d_j$: demand column in `file_0_view_0`, indexed by customer
- $s_i$: supply_capacity column in `file_1_view_0`, indexed by region
- $c_{ij}$: entry in `file_2_view_0` (transportation_costs.csv), row indexed by Unnamed: 0 (warehouse/region), column indexed by store/customer

All indices, parameters, and coefficients are mapped directly from the current source data as described above.