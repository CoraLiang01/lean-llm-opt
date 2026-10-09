#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of warehouses (indexed by $s$), from `supply_capacity.csv` column `region` and `transportation_costs.csv` row `Unnamed: 0`
- $D$: set of stores (indexed by $d$), from `customer_demand.csv` column `customer` and `transportation_costs.csv` columns

**Parameters:**
- $c_{sd}$: unit transportation cost from warehouse $s \in S$ to store $d \in D$  
  (from `transportation_costs.csv`, table_id: file_2_view_0, columns: `Unnamed: 0`, $D$)
- $a_s$: daily supply capacity of warehouse $s \in S$  
  (from `supply_capacity.csv`, table_id: file_1_view_0, columns: `region`, `supply_capacity`)
- $b_d$: daily demand at store $d \in D$  
  (from `customer_demand.csv`, table_id: file_0_view_0, columns: `customer`, `demand`)

**Decision Variables:**
- $x_{sd} \geq 0$: quantity shipped from warehouse $s$ to store $d$ (continuous, non-negative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{d \in D} c_{sd} \cdot x_{sd}
\]

**Constraints:**
1. **Demand Satisfaction (for each store):**
   \[
   \sum_{s \in S} x_{sd} = b_d, \quad \forall d \in D
   \]
2. **Warehouse Supply Capacity (for each warehouse):**
   \[
   \sum_{d \in D} x_{sd} \leq a_s, \quad \forall s \in S
   \]
3. **Non-negativity:**
   \[
   x_{sd} \geq 0, \quad \forall s \in S, d \in D
   \]

---

**Data Mapping:**

- $S$ (warehouses): `supply_capacity.csv` (table_id: file_1_view_0, column: `region`), `transportation_costs.csv` (table_id: file_2_view_0, row: `Unnamed: 0`)
- $D$ (stores): `customer_demand.csv` (table_id: file_0_view_0, column: `customer`), `transportation_costs.csv` (table_id: file_2_view_0, columns: $D$)
- $c_{sd}$: `transportation_costs.csv` (table_id: file_2_view_0, columns: $D$, rows: `Unnamed: 0`)
- $a_s$: `supply_capacity.csv` (table_id: file_1_view_0, column: `supply_capacity`)
- $b_d$: `customer_demand.csv` (table_id: file_0_view_0, column: `demand`)