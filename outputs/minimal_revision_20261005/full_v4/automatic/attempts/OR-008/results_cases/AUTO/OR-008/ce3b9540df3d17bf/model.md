##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity of fresh produce shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$ (warehouses)
- $J = \{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$ (stores)

##### Parameters

- $d_j$: demand of store $j$ (from “customer_demand.csv”)
- $s_i$: supply capacity of warehouse $i$ (from “supply_capacity.csv”)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from “transportation_costs.csv”)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (warehouses):  
  From `supply_capacity.csv` (`file_1_view_0`), column `Suppliers`  
  $\rightarrow$ $\{\text{Supplier1}, \text{Supplier2}, \text{Supplier3}, \text{Supplier4}, \text{Supplier5}\}$

- $J$ (stores):  
  From `customer_demand.csv` (`file_0_view_0`), column `Customers`  
  $\rightarrow$ $\{\text{Customer1}, \text{Customer2}, \text{Customer3}, \text{Customer4}, \text{Customer5}, \text{Customer6}\}$

- $d_j$:  
  From `customer_demand.csv` (`file_0_view_0`), column `demand`, indexed by `Customers`

- $s_i$:  
  From `supply_capacity.csv` (`file_1_view_0`), column `supply_capacity`, indexed by `Suppliers`

- $c_{ij}$:  
  From `transportation_costs.csv` (`file_2_view_0`), row index `Unnamed: 0` (warehouses), column headers (stores)

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.**