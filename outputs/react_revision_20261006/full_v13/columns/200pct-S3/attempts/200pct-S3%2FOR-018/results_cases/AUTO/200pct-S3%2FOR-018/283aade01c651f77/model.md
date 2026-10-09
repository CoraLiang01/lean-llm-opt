##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

**Decision Variables:**

For each $i \in I$, $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand satisfaction:**  
   For all $j \in J$,
   \[
   \sum_{i \in I} x_{ij} \geq d_j
   \]
2. **Supply capacity:**  
   For all $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq s_i
   \]
3. **Non-negativity:**  
   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \geq 0
   \]

**Index Sets:**
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

##### Data Mapping

- $d_j$: "customer_demand.csv", table_id: file_0_view_0, columns: customer_id, demand; $j$ = customer_id, $d_j$ = demand
- $s_i$: "supply_capacity.csv", table_id: file_1_view_0, columns: supplier_id, supply_capacity; $i$ = supplier_id, $s_i$ = supply_capacity
- $c_{ij}$: "transportation_costs.csv", table_id: file_2_view_0, row: supplier_id ($i$), column: transportation_cost_to_$j$ (where $j$ = customer_id)

- $x_{ij}$: decision variable for each $i$ in supplier_id, $j$ in customer_id

All index sets, parameters, and constraints are mapped directly to the current data, preserving all identifiers and coefficients. No data is omitted or aggregated.