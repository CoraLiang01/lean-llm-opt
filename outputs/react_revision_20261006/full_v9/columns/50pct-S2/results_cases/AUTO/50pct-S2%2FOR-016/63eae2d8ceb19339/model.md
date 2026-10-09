#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

**Decision Variables:**

For each $i \in I$, $j \in J$:
- $x_{ij} \geq 0$: quantity of goods shipped from distribution center $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand (units) for customer group $j$ (from "customer_demand.csv").
- $s_i$: supply capacity (units) of distribution center $i$ (from "supply_capacity.csv").
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv").

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

**Index Sets:**
- $I = \{$ all supplier_id in "supply_capacity.csv" $\}$
- $J = \{$ all customer_id in "customer_demand.csv" $\}$

---

#### Data Mapping

- $I$: supplier_id from "supply_capacity.csv" (table_id: file_1_view_0, column: supplier_id)
- $J$: customer_id from "customer_demand.csv" (table_id: file_0_view_0, column: customer_id)
- $d_j$: demand_units from "customer_demand.csv" (table_id: file_0_view_0, columns: customer_id, demand_units)
- $s_i$: supply_capacity_units from "supply_capacity.csv" (table_id: file_1_view_0, columns: supplier_id, supply_capacity_units)
- $c_{ij}$: transportation_cost_to_Ck from "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck, where $k$ matches customer_id in $J$)

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving all identifiers and source order. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.