##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as listed in the data. Let $x_{ij} \geq 0$ denote the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

**Subject to:**

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
   where $d_j$ is the demand of customer $j$.

2. **Supply capacity:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
   where $s_i$ is the supply capacity of supplier $i$.

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (Suppliers): All supplier_id values from supply_capacity.csv and transportation_costs.csv (S1, S2, ..., S12).
- $J$ (Customers): All customer_id values from customer_demand.csv and transportation_costs.csv (C1, C2, ..., C12).
- $d_j$: demand for customer $j$ from customer_demand.csv, column demand, indexed by customer_id.
- $s_i$: supply_capacity for supplier $i$ from supply_capacity.csv, column supply_capacity, indexed by supplier_id.
- $c_{ij}$: transportation_costs.csv, value in column transportation_cost_to_Ck for supplier_id $i$ and customer $j = $Ck.

**Index sets and parameter mapping are as follows:**

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$
- $d_j$ from customer_demand.csv: table_id file_0_view_0, columns customer_id, demand
- $s_i$ from supply_capacity.csv: table_id file_1_view_0, columns supplier_id, supply_capacity
- $c_{ij}$ from transportation_costs.csv: table_id file_2_view_0, row supplier_id, columns transportation_cost_to_C1 ... transportation_cost_to_C12

**Decision variables:**
- $x_{ij} \geq 0$ continuous, for all $i \in I$, $j \in J$.

**All indices, parameters, and coefficients are to be taken exactly as listed in the source files and mapped as above.**