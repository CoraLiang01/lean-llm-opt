##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Sets

- $I$: set of suppliers (distribution centers), from `supply_capacity.csv`  
  $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$: set of customers (customer groups), from `customer_demand.csv`  
  $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer $j$, from `customer_demand.csv`  
  $d_j = \text{demand}$, column "demand", table_id `file_0_view_0`, key "Customers"
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv`  
  $s_i = \text{supply_capacity}$, column "supply_capacity", table_id `file_1_view_0`, key "Supplier"
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv`  
  $c_{ij} = $ value at row with "Unnamed: 0" = $i$, column $j$, table_id `file_2_view_0`

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
   (No supplier ships more than its capacity.)

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all values in column "Supplier" of table_id `file_1_view_0`
- $J$ (customers): all values in column "Customers" of table_id `file_0_view_0`
- $d_j$: column "demand" in table_id `file_0_view_0`, indexed by "Customers"
- $s_i$: column "supply_capacity" in table_id `file_1_view_0`, indexed by "Supplier"
- $c_{ij}$: value at row with "Unnamed: 0" = $i$ and column $j$ in table_id `file_2_view_0` (matrix: rows = suppliers, columns = customers)

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data as described above.**