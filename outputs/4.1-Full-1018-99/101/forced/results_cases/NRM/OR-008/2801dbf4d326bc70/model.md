#### Sets
- $S$: set of suppliers (warehouses), indexed by $s$  
  (Data: column "Suppliers" in table_id file_1_view_0)
- $C$: set of customers (stores), indexed by $c$  
  (Data: column "Customers" in table_id file_0_view_0)

#### Parameters
- $d_c$: demand of customer $c$  
  (Data: column "demand" in table_id file_0_view_0)
- $u_s$: supply capacity of supplier $s$  
  (Data: column "supply_capacity" in table_id file_1_view_0)
- $t_{s,c}$: transportation cost per unit from supplier $s$ to customer $c$  
  (Data: columns "Customer1", ..., "Customer6" in table_id file_2_view_0, with row identifier "Unnamed: 0" as supplier)

#### Decision Variables
- $x_{s,c} \geq 0$: amount of produce shipped from supplier $s$ to customer $c$

#### Objective
Minimize total transportation cost:
$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction (for each customer):**
$$
\sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
$$

2. **Supply Capacity (for each supplier):**
$$
\sum_{c \in C} x_{s,c} \leq u_s \quad \forall s \in S
$$

3. **Nonnegativity:**
$$
x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
$$

---

#### Data Mapping

- Table file_0_view_0 ("customer_demand.csv"):  
  "Customers" $\rightarrow$ $C$ (set of customers),  
  "demand" $\rightarrow$ $d_c$ (customer demand)
- Table file_1_view_0 ("supply_capacity.csv"):  
  "Suppliers" $\rightarrow$ $S$ (set of suppliers),  
  "supply_capacity" $\rightarrow$ $u_s$ (supplier capacity)
- Table file_2_view_0 ("transportation_costs.csv"):  
  "Unnamed: 0" $\rightarrow$ $S$ (supplier row id),  
  "Customer1", ..., "Customer6" $\rightarrow$ $t_{s,c}$ (cost from $s$ to $c$)