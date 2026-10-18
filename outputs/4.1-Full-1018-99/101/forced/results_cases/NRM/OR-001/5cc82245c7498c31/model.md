#### Index Sets
- $S$: set of distribution centers (sources), from column "Unnamed: 0" in supply_capacity.csv and transportation_costs.csv
- $C$: set of customer groups, from column "customer" in customer_demand.csv and columns "C1", ..., "C18" in transportation_costs.csv

#### Parameters
- $d_c$: daily demand for customer group $c \in C$ (customer_demand.csv, column "demand")
- $u_s$: daily supply capacity of distribution center $s \in S$ (supply_capacity.csv, column "supply_capacity")
- $t_{s,c}$: transportation cost per unit from distribution center $s$ to customer group $c$ (transportation_costs.csv, columns "C1", ..., "C18")

#### Decision Variables
- $x_{s,c} \geq 0$: quantity of goods transported from distribution center $s$ to customer group $c$ (continuous, non-negative)

#### Objective
$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction (for each customer group):**
   $$
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   $$

2. **Supply Capacity (for each distribution center):**
   $$
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   $$

3. **Non-negativity:**
   $$
   x_{s,c} \geq 0, \quad \forall s \in S, \forall c \in C
   $$

---

#### Data Mapping

- **customer_demand.csv**:  
  - Table ID: file_0_view_0  
  - Columns: "customer" $\rightarrow$ $C$, "demand" $\rightarrow$ $d_c$

- **supply_capacity.csv**:  
  - Table ID: file_1_view_0  
  - Columns: "Unnamed: 0" $\rightarrow$ $S$, "supply_capacity" $\rightarrow$ $u_s$

- **transportation_costs.csv**:  
  - Table ID: file_2_view_0  
  - Rows: "Unnamed: 0" $\rightarrow$ $S$  
  - Columns: "C1", ..., "C18" $\rightarrow$ $C$  
  - Values: $t_{s,c}$

All data is used as returned by CSVQA, with no filtering.