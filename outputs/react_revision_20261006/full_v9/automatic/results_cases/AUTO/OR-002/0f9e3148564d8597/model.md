#### Mathematical Model

Let $I$ be the set of stores (indexed by $i$), and $J$ the set of customer groups (indexed by $j$), as defined by the current data.

**Decision Variables:**

For each $i\in I$, $j\in J$:
- $x_{ij} \geq 0$: quantity shipped from store $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer group $j$.
- $s_i$: supply capacity of store $i$.
- $c_{ij}$: transportation cost per unit from store $i$ to customer group $j$.

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

- $I$: All store IDs from column "Unnamed: 0" in table_id="file_1_view_0" (supply_capacity.csv), i.e., $I = \{\text{S1}, \text{S2}, ..., \text{S11}\}$.
- $J$: All customer group IDs from column "customer" in table_id="file_0_view_0" (customer_demand.csv), i.e., $J = \{\text{C1}, \text{C2}, ..., \text{C12}\}$.
- $d_j$: Demand for customer group $j$ from column "demand" in table_id="file_0_view_0", keyed by "customer".
- $s_i$: Supply capacity for store $i$ from column "supply_capacity" in table_id="file_1_view_0", keyed by "Unnamed: 0".
- $c_{ij}$: Transportation cost per unit from store $i$ to customer group $j$ from table_id="file_2_view_0" (transportation_costs.csv), with rows indexed by "Unnamed: 0" (store IDs) and columns by customer group IDs.

All index sets, parameters, and coefficients are to be taken exactly as listed in the current data, preserving source order and identifiers.