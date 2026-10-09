#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the current data.

**Decision Variables:**

For each $i\in I$, $j\in J$:
- $x_{ij} \geq 0$: quantity of goods shipped from distribution center $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer group $j$ (from "customer_demand.csv").
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv").
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv").

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

**Index sets:**
- $I = \{$S1, S2, ..., S18$\}$ (all supplier_id in "supply_capacity.csv")
- $J = \{$C1, C2, ..., C18$\}$ (all customer_id in "customer_demand.csv")

---

#### Data Mapping

- $I$ (distribution centers): All `supplier_id` in table_id: file_1_view_0, column: supplier_id
- $J$ (customer groups): All `customer_id` in table_id: file_0_view_0, column: customer_id
- $d_j$: Demand for customer $j$ from table_id: file_0_view_0, column: demand_units, keyed by customer_id
- $s_i$: Supply capacity for supplier $i$ from table_id: file_1_view_0, column: supply_capacity_units, keyed by supplier_id
- $c_{ij}$: Transportation cost from supplier $i$ to customer $j$ from table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_{customer_id}

All index sets, parameters, and matrix axes are defined by the current data and preserved in source order. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.