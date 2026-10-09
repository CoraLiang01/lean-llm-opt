#### Index Sets

- $S$: set of suppliers (from file_1_view_0, column "Unnamed: 0" in fixed_cost.csv)
- $C$: set of stores (from file_0_view_0, column "customer" in demand.csv)

#### Parameters

- $f_s$: fixed cost to open supplier $s \in S$ (from file_1_view_0, column "fixed_costs" in fixed_cost.csv)
- $t_{s,c}$: transportation cost per unit from supplier $s \in S$ to store $c \in C$ (from file_2_view_0, columns "Unnamed: 0" for supplier and $c$ for store in transportation_costs.csv)
- $d_c$: demand at store $c \in C$ (from file_0_view_0, column "demand" in demand.csv)

#### Decision Variables

- $y_s \in \{0,1\}$: 1 if supplier $s$ is operational (open), 0 otherwise
- $x_{s,c} \geq 0$: quantity supplied from supplier $s$ to store $c$

#### Objective

Minimize total cost (fixed + transportation):
$$
\min \quad \sum_{s \in S} f_s y_s + \sum_{s \in S} \sum_{c \in C} t_{s,c} x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction:**  
   $$
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   $$

2. **Supplier Activation:**  
   $$
   x_{s,c} \leq M_{s,c} y_s, \quad \forall s \in S, \forall c \in C
   $$
   where $M_{s,c}$ is a sufficiently large constant (e.g., $M_{s,c} \geq d_c$) to allow supply only if supplier $s$ is open.

3. **Variable Domains:**  
   $$
   y_s \in \{0,1\}, \quad \forall s \in S
   $$
   $$
   x_{s,c} \geq 0, \quad \forall s \in S, \forall c \in C
   $$

---

#### Data Mapping

- **Suppliers ($S$) and Fixed Costs ($f_s$):**  
  - Source: file_1_view_0 (fixed_cost.csv)  
  - Columns: "Unnamed: 0" (supplier ID), "fixed_costs"

- **Stores ($C$) and Demands ($d_c$):**  
  - Source: file_0_view_0 (demand.csv)  
  - Columns: "customer" (store ID), "demand"

- **Transportation Costs ($t_{s,c}$):**  
  - Source: file_2_view_0 (transportation_costs.csv)  
  - Row index: "Unnamed: 0" (supplier ID)  
  - Columns: "C1", "C2", ..., "C6" (store IDs)

- **All data used as returned by CSVQA; no additional filtering or aggregation applied.**