##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the data.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i \in I$ to customer group $j \in J$.

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

**Subject to:**

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
   where $d_j$ is the demand of customer $j$.

2. **Supply capacity:**  
   For each distribution center $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
   where $s_i$ is the supply capacity of supplier $i$.

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (distribution centers): all supplier_id in supply_capacity.csv and transportation_costs.csv, in source order.
- $J$ (customer groups): all customer_id in customer_demand.csv and transportation_costs.csv, in source order.
- $d_j$: demand for customer $j$ from customer_demand.csv, column demand, table_id file_0_view_0.
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv, column supply_capacity, table_id file_1_view_0.
- $c_{ij}$: transportation_costs.csv, table_id file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$ (column_id_mapping: e.g., transportation_cost_to_C1 $\rightarrow$ C1).

**Index sets:**
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

**Variable domain:**  
$x_{ij} \geq 0$ continuous, for all $i \in I$, $j \in J$.