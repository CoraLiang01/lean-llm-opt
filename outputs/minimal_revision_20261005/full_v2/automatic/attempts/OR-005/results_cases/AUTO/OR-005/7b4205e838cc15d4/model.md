##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i$ to customer $j$, for all suppliers $i$ and customers $j$.

##### Sets

- $I$: set of suppliers, from column "Supplier" in `supply_capacity.csv` (table_id: file_1_view_0)
- $J$: set of customers, from column "Customers" in `customer_demand.csv` (table_id: file_0_view_0)

##### Parameters

- $d_j$: demand of customer $j$, from column "demand" in `customer_demand.csv` (table_id: file_0_view_0)
- $s_i$: supply capacity of supplier $i$, from column "supply_capacity" in `supply_capacity.csv` (table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from entry in row $i$ (matching "Unnamed: 0" to supplier) and column $j$ (matching customer name) in `transportation_costs.csv` (table_id: file_2_view_0)

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
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all values in column "Supplier" of `supply_capacity.csv` (table_id: file_1_view_0)
- $J$ (customers): all values in column "Customers" of `customer_demand.csv` (table_id: file_0_view_0)
- $d_j$: for each $j \in J$, value in column "demand" of `customer_demand.csv` (table_id: file_0_view_0), row where "Customers" = $j$
- $s_i$: for each $i \in I$, value in column "supply_capacity" of `supply_capacity.csv` (table_id: file_1_view_0), row where "Supplier" = $i$
- $c_{ij}$: for each $i \in I$, $j \in J$, value in `transportation_costs.csv` (table_id: file_2_view_0), row where "Unnamed: 0" = $i$, column $j$

---

All indices, coefficients, and constraints are bound directly to the retrieved data as specified above.