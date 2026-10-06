##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i$ to customer $j$, for all $i \in I$, $j \in J$.

##### Sets and Indices

- $I$: set of suppliers (distribution centers), indexed by $i$.
- $J$: set of customers (customer groups), indexed by $j$.

From the data:
- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

##### Parameters

- $d_j$: demand of customer $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv")

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- **customer_demand.csv** (`file_0_view_0`):  
  - $J$ = values in column `"Customers"`
  - $d_j$ = value in column `"demand"` for customer $j$

- **supply_capacity.csv** (`file_1_view_0`):  
  - $I$ = values in column `"Supplier"`
  - $s_i$ = value in column `"supply_capacity"` for supplier $i$

- **transportation_costs.csv** (`file_2_view_0`):  
  - $c_{ij}$ = value in row with `"Unnamed: 0" = i$ and column $j$ (where $i \in I$, $j \in J$)

- **Row and column mappings for cost matrix:**  
  - row\_id\_mapping:  
    - `"supply1"` $\rightarrow$ `"supplier1"`
    - `"supply2"` $\rightarrow$ `"supplier2"`
    - `"supply3"` $\rightarrow$ `"supplier3"`
    - `"supply4"` $\rightarrow$ `"supplier4"`
    - `"supply5"` $\rightarrow$ `"supplier5"`
    - `"supply6"` $\rightarrow$ `"supplier6"`
    - `"supply7"` $\rightarrow$ `"supplier7"`
    - `"supply8"` $\rightarrow$ `"supplier8"`
  - column\_id\_mapping:  
    - `"demand1"` $\rightarrow$ `"demand1"`
    - `"demand2"` $\rightarrow$ `"demand2"`
    - `"demand3"` $\rightarrow$ `"demand3"`
    - `"demand4"` $\rightarrow$ `"demand4"`
    - `"demand5"` $\rightarrow$ `"demand5"`
    - `"demand6"` $\rightarrow$ `"demand6"`
    - `"demand7"` $\rightarrow$ `"demand7"`
    - `"demand8"` $\rightarrow$ `"demand8"`

---

##### Complete Model

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

with all parameters and indices mapped as above from the retrieved data.