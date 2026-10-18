#### Index Sets
- $S$: set of distribution centers (suppliers), from `supply_capacity.csv` [file_1_view_0, column "Supplier"]
- $C$: set of customer groups, from `customer_demand.csv` [file_0_view_0, column "Customers"]

#### Parameters
- $d_c$: daily demand of customer group $c \in C$, from `customer_demand.csv` [file_0_view_0, column "demand"]
- $u_s$: daily supply capacity of distribution center $s \in S$, from `supply_capacity.csv` [file_1_view_0, column "supply_capacity"]
- $t_{s,c}$: transportation cost per unit from distribution center $s \in S$ to customer group $c \in C$, from `transportation_costs.csv` [file_2_view_0, row "Unnamed: 0" (distribution center), columns "demand1", ..., "demand8" (customer groups)]

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

2. **Supply Capacity** (no center exceeds its capacity):
   $$
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   $$

3. **Non-negativity**:
   $$
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   $$

---

#### Data Mapping

- $C$ (customer groups): all records from `customer_demand.csv` [file_0_view_0, column "Customers"]
- $d_c$: from `customer_demand.csv` [file_0_view_0, column "demand"]
- $S$ (distribution centers): all records from `supply_capacity.csv` [file_1_view_0, column "Supplier"]
- $u_s$: from `supply_capacity.csv` [file_1_view_0, column "supply_capacity"]
- $t_{s,c}$: from `transportation_costs.csv` [file_2_view_0, row "Unnamed: 0" (distribution center), columns "demand1", ..., "demand8" (customer groups)]

All records and columns from each file are included as returned by CSVQA. No additional filtering or mapping is applied.