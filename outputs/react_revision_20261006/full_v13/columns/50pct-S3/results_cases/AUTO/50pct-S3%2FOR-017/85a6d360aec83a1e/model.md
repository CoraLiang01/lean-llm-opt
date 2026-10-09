#### Mathematical Model

Let $I$ be the set of suppliers (from the supplier_id column of supply_capacity.csv and transportation_costs.csv), and $J$ be the set of customer groups (from the customer_id column of customer_demand.csv and the transportation_cost_to_C* columns of transportation_costs.csv).

**Decision Variables:**

For each $i\in I$, $j\in J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer group $j$ (continuous).

**Parameters:**
- $d_j$: demand of customer group $j$ (from customer_demand.csv).
- $s_i$: supply capacity of supplier $i$ (from supply_capacity.csv).
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from transportation_costs.csv).

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

- $I$ (suppliers): All supplier_id values in supply_capacity.csv (file_1_view_0, column supplier_id) and transportation_costs.csv (file_2_view_0, column supplier_id).
- $J$ (customer groups): All customer_id values in customer_demand.csv (file_0_view_0, column customer_id) and all columns transportation_cost_to_C* in transportation_costs.csv (file_2_view_0).
- $d_j$: For each $j\in J$, demand from customer_demand.csv (file_0_view_0, columns customer_id, demand).
- $s_i$: For each $i\in I$, supply_capacity from supply_capacity.csv (file_1_view_0, columns supplier_id, supply_capacity).
- $c_{ij}$: For each $i\in I$, $j\in J$, transportation_costs.csv (file_2_view_0, row supplier_id $i$, column transportation_cost_to_$j$).

- Decision variables $x_{ij}$ are defined for all $(i,j)$ where $i$ is a supplier_id in supply_capacity.csv and $j$ is a customer_id in customer_demand.csv.

- The matrix mapping between suppliers and customers is validated by the relationships field, which aligns supplier_id (rows) and customer_id (columns) between the files.

- All constraints and the objective use these mapped parameters directly from the source data, with no aggregation or omission.

**Index sets, parameters, and variable domains are defined exactly as in the current data.**