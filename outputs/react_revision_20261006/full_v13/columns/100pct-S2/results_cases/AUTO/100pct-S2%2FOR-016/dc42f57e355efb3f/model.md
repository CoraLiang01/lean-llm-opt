#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

**Decision Variables:**

For each $i \in I$, $j \in J$:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand_units", indexed by "customer_id").
- $s_i$: supply capacity of distribution center $i$ (from file_1_view_0, column "supply_capacity_units", indexed by "supplier_id").
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from file_2_view_0, column "transportation_cost_to_{j}", row "supplier_id").

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

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

#### Data Mapping

- $I$ (distribution centers): all "supplier_id" in file_1_view_0 (supply_capacity.csv) and file_2_view_0 (transportation_costs.csv) rows.
- $J$ (customer groups): all "customer_id" in file_0_view_0 (customer_demand.csv) and file_2_view_0 (transportation_costs.csv) columns (suffix after "transportation_cost_to_").
- $d_j$: file_0_view_0, column "demand_units", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity_units", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" $i$, column "transportation_cost_to_{j}" for customer $j$.

All index sets, parameters, and coefficients are defined exactly as in the current source data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.