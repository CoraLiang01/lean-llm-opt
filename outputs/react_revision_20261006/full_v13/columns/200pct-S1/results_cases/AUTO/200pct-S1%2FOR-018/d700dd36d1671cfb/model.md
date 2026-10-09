##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

**Decision Variables:**

For each $i\in I$, $j\in J$:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from file_2_view_0, columns "transportation_cost_to_Ck" for each $j$).

**Objective:**
\[
\min \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
\]

**Constraints:**
1. **Demand satisfaction:**  
   For all $j\in J$,
   \[
   \sum_{i\in I} x_{ij} \geq d_j
   \]
2. **Supply capacity:**  
   For all $i\in I$,
   \[
   \sum_{j\in J} x_{ij} \leq s_i
   \]
3. **Non-negativity:**  
   For all $i\in I$, $j\in J$,
   \[
   x_{ij} \geq 0
   \]

**Index Sets:**
- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (from file_1_view_0, column "supplier_id")
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (from file_0_view_0, column "customer_id")

##### Data Mapping

- $d_j$: file_0_view_0, column "demand", row with "customer_id" = $j$
- $s_i$: file_1_view_0, column "supply_capacity", row with "supplier_id" = $i$
- $c_{ij}$: file_2_view_0, row with "supplier_id" = $i$, column "transportation_cost_to_$j$"
- $x_{ij}$: decision variable for each $(i,j)\in I\times J$

All index sets, parameters, and coefficients are mapped directly from the current source data as described above. No data is omitted or aggregated. All constraints and variable domains are as specified in the user query.