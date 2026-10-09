#### Index Sets
- $S$: set of distribution centers (from supply_capacity.csv, column "Unnamed: 0")
- $C$: set of customer groups (from customer_demand.csv, column "customer")

#### Parameters
- $d_c$: demand of customer group $c \in C$ (from customer_demand.csv, column "demand")
- $u_s$: supply capacity of distribution center $s \in S$ (from supply_capacity.csv, column "supply_capacity")
- $t_{s,c}$: transportation cost per unit from distribution center $s \in S$ to customer group $c \in C$ (from transportation_costs.csv, columns "Unnamed: 0" for $s$ and $C1,\ldots,C12$ for $c$)

#### Decision Variables
- $x_{s,c} \geq 0$: quantity of goods shipped from distribution center $s \in S$ to customer group $c \in C$

#### Objective
$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction** (all customer demands must be met):
   $$
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   $$

2. **Supply Capacity** (do not exceed supply at each distribution center):
   $$
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   $$

3. **Non-negativity**:
   $$
   x_{s,c} \geq 0, \quad \forall s \in S, \forall c \in C
   $$

---

#### Data Mapping

- $C$ (customer groups): file_id = "file_0_view_0", column = "customer"
- $d_c$: file_id = "file_0_view_0", column = "demand"
- $S$ (distribution centers): file_id = "file_1_view_0", column = "Unnamed: 0"
- $u_s$: file_id = "file_1_view_0", column = "supply_capacity"
- $t_{s,c}$: file_id = "file_2_view_0", row = "Unnamed: 0" (distribution center), columns = "C1" to "C12" (customer group)

All records from each file are included, as returned by CSVQA. No additional filtering or aggregation is applied.