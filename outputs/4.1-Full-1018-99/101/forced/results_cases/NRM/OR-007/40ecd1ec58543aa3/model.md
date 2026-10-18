#### Index Sets
- $S$: set of warehouses (from supply_capacity.csv, table_id: file_1_view_0, column: region)
- $D$: set of stores (from customer_demand.csv, table_id: file_0_view_0, column: customer)

#### Parameters
- $c_{sd}$: unit transportation cost from warehouse $s \in S$ to store $d \in D$ (from transportation_costs.csv, table_id: file_2_view_0, columns: Unnamed: 0 for $s$, D1–D5 for $d$)
- $a_s$: daily supply capacity of warehouse $s \in S$ (from supply_capacity.csv, table_id: file_1_view_0, column: supply_capacity)
- $b_d$: daily demand at store $d \in D$ (from customer_demand.csv, table_id: file_0_view_0, column: demand)

#### Decision Variables
- $x_{sd} \geq 0$: quantity shipped from warehouse $s \in S$ to store $d \in D$

#### Objective
$$
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \, x_{sd}
$$

#### Constraints

1. **Demand Satisfaction (for each store):**
   $$
   \sum_{s \in S} x_{sd} = b_d, \quad \forall d \in D
   $$

2. **Warehouse Supply Capacity (for each warehouse):**
   $$
   \sum_{d \in D} x_{sd} \leq a_s, \quad \forall s \in S
   $$

3. **Nonnegativity:**
   $$
   x_{sd} \geq 0, \quad \forall s \in S,\, d \in D
   $$

---

#### Data Mapping

- **customer_demand.csv** (table_id: file_0_view_0):  
  - Store index set $D$ from column "customer"
  - Parameter $b_d$ from column "demand"

- **supply_capacity.csv** (table_id: file_1_view_0):  
  - Warehouse index set $S$ from column "region"
  - Parameter $a_s$ from column "supply_capacity"

- **transportation_costs.csv** (table_id: file_2_view_0):  
  - Parameter $c_{sd}$ from columns:  
    - Warehouse $s$ from "Unnamed: 0"  
    - Store $d$ from columns "D1", "D2", "D3", "D4", "D5" (wide matrix format)