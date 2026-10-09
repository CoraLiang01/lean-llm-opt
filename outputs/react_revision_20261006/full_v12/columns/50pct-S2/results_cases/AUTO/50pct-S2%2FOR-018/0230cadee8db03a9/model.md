#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

**Decision Variables:**

For each $i\in I$, $j\in J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, columns "transportation_cost_to_Ck")

**Objective:**
\[
\min \sum_{i\in I} \sum_{j\in J} c_{ij} x_{ij}
\]

**Subject to:**

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

#### Data Mapping

- $I$ (suppliers): all "supplier_id" in file_1_view_0 and file_2_view_0, in source order.
- $J$ (customers): all "customer_id" in file_0_view_0 and columns "transportation_cost_to_Ck" in file_2_view_0, in source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" $i$, column "transportation_cost_to_Ck" for customer $j$.

All indices, parameters, and coefficients are to be used exactly as in the current Observation, preserving source order and identifiers.