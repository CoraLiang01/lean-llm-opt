##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Sets

- $I$: set of suppliers, from `supply_capacity.csv` and `transportation_costs.csv`  
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- $J$: set of customers, from `customer_demand.csv` and `transportation_costs.csv`  
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

##### Parameters

- $d_j$: demand of customer $j$  
  From `customer_demand.csv` (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply capacity of supplier $i$  
  From `supply_capacity.csv` (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$  
  From `transportation_costs.csv` (table_id: file_2_view_0, columns: supplier_id, transportation_cost_to_C1, ..., transportation_cost_to_C10)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   (Every customer receives at least their demand.)

2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   (No supplier ships more than their capacity.)

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (table_id: file_1_view_0) and `transportation_costs.csv` (table_id: file_2_view_0)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (table_id: file_0_view_0) and columns with suffix in `transportation_costs.csv` (table_id: file_2_view_0)
- $d_j$: `demand` column in `customer_demand.csv` (table_id: file_0_view_0), indexed by `customer_id`
- $s_i$: `supply_capacity` column in `supply_capacity.csv` (table_id: file_1_view_0), indexed by `supplier_id`
- $c_{ij}$: `transportation_cost_to_Ck` columns in `transportation_costs.csv` (table_id: file_2_view_0), indexed by `supplier_id` and customer $j$ (column suffix)

All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers. No data is omitted or invented.