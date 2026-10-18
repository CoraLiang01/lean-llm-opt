#### Index Sets
- $S$: set of distribution centers (from supply_capacity.csv, column "Unnamed: 0", table_id file_1_view_0)
- $C$: set of customer groups (from customer_demand.csv, column "customer", table_id file_0_view_0)

#### Parameters
- $d_c$: demand of customer group $c \in C$ (from customer_demand.csv, column "demand", table_id file_0_view_0)
- $u_s$: supply capacity of distribution center $s \in S$ (from supply_capacity.csv, column "supply_capacity", table_id file_1_view_0)
- $t_{s,c}$: transportation cost per unit from distribution center $s$ to customer group $c$ (from transportation_costs.csv, columns "Unnamed: 0" for $s$ and $C1$–$C12$ for $c$, table_id file_2_view_0)

#### Decision Variables
- $x_{s,c} \geq 0$: quantity of goods shipped from distribution center $s \in S$ to customer group $c \in C$

#### Objective
$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction (for all $c \in C$):**
   $$
   \sum_{s \in S} x_{s,c} = d_c
   $$

2. **Supply Capacity (for all $s \in S$):**
   $$
   \sum_{c \in C} x_{s,c} \leq u_s
   $$

3. **Non-negativity:**
   $$
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   $$

---

#### Data Mapping

- $C$ (customer groups): file_0_view_0, column "customer"
- $d_c$: file_0_view_0, column "demand"
- $S$ (distribution centers): file_1_view_0, column "Unnamed: 0"
- $u_s$: file_1_view_0, column "supply_capacity"
- $t_{s,c}$: file_2_view_0, row "Unnamed: 0" (for $s$), columns "C1"–"C12" (for $c$)

All data is used as returned by CSVQA, with no additional filtering or transformation.