#### Index Sets
- $S$: set of suppliers (from fixed_cost.csv, column "Unnamed: 0", table_id file_1_view_0)
- $C$: set of stores (from demand.csv, column "customer", table_id file_0_view_0)

#### Parameters
- $f_s$: fixed cost of opening supplier $s \in S$ (from fixed_cost.csv, column "fixed_costs", table_id file_1_view_0)
- $t_{sc}$: transportation cost per unit from supplier $s \in S$ to store $c \in C$ (from transportation_costs.csv, columns "Unnamed: 0" for supplier and "C1", ..., "C6" for stores, table_id file_2_view_0)
- $d_c$: demand at store $c \in C$ (from demand.csv, column "demand", table_id file_0_view_0)

#### Decision Variables
- $y_s \in \{0,1\}$: 1 if supplier $s$ is operational, 0 otherwise
- $x_{sc} \geq 0$: quantity supplied from supplier $s$ to store $c$

#### Objective
$$
\min \sum_{s \in S} f_s y_s + \sum_{s \in S} \sum_{c \in C} t_{sc} x_{sc}
$$

#### Constraints
1. **Demand Satisfaction:**  
   $$
   \sum_{s \in S} x_{sc} = d_c, \quad \forall c \in C
   $$
2. **Supplier Activation:**  
   $$
   x_{sc} \leq M_{sc} y_s, \quad \forall s \in S,\, c \in C
   $$
   where $M_{sc}$ is a sufficiently large constant (e.g., $M_{sc} \geq d_c$).

3. **Variable Domains:**  
   $$
   y_s \in \{0,1\}, \quad \forall s \in S
   $$
   $$
   x_{sc} \geq 0, \quad \forall s \in S,\, c \in C
   $$

---

#### Data Mapping

- **Suppliers ($S$):** file_1_view_0, column "Unnamed: 0" (fixed_cost.csv)
- **Stores ($C$):** file_0_view_0, column "customer" (demand.csv)
- **Fixed Costs ($f_s$):** file_1_view_0, column "fixed_costs" (fixed_cost.csv)
- **Transportation Costs ($t_{sc}$):** file_2_view_0, columns "Unnamed: 0" (supplier), "C1"–"C6" (store) (transportation_costs.csv)
- **Demand ($d_c$):** file_0_view_0, column "demand" (demand.csv)