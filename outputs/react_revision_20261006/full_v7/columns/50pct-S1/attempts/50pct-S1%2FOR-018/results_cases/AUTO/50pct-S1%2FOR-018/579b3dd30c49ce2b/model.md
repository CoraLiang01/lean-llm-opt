##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups.

Let $x_{ij} \geq 0$ be the quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]

2. **Supply capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (distribution centers): all supplier_id in supply_capacity.csv and transportation_costs.csv
- $J$ (customer groups): all customer_id in customer_demand.csv and transportation_costs.csv
- $d_j$: demand for customer $j$ from customer_demand.csv, column demand, table_id file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv, column supply_capacity, table_id file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost from supplier $i$ to customer $j$ from transportation_costs.csv, table_id file_2_view_0, row supplier_id, column transportation_cost_to_Ck (mapping Ck to customer_id)

**Index sets:**
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

**Parameter mapping:**
- $d_j$: file_0_view_0, customer_id = $j$, demand
- $s_i$: file_1_view_0, supplier_id = $i$, supply_capacity
- $c_{ij}$: file_2_view_0, row supplier_id = $i$, column transportation_cost_to_Ck where $k$ matches $j$

**Variables:**
- $x_{ij} \geq 0$ continuous, for all $i \in I$, $j \in J$