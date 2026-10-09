#### Index Sets
- $S$: set of distribution centers (from column "Unnamed: 0" in supply_capacity.csv and transportation_costs.csv)
- $C$: set of customer groups (from column "customer" in customer_demand.csv and columns in transportation_costs.csv)

#### Parameters
- $d_c$: daily demand of customer group $c \in C$ (from column "demand" in customer_demand.csv)
- $u_s$: daily supply capacity of distribution center $s \in S$ (from column "supply_capacity" in supply_capacity.csv)
- $t_{s,c}$: transportation cost per unit from distribution center $s$ to customer group $c$ (from table_id file_2_view_0, columns $C$, rows $S$ in transportation_costs.csv)

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

- $S$ (distribution centers): file_1_view_0, column "Unnamed: 0" (supply_capacity.csv); file_2_view_0, row "Unnamed: 0" (transportation_costs.csv)
- $C$ (customer groups): file_0_view_0, column "customer" (customer_demand.csv); file_2_view_0, columns "C1"..."C18" (transportation_costs.csv)
- $d_c$: file_0_view_0, column "demand" (customer_demand.csv)
- $u_s$: file_1_view_0, column "supply_capacity" (supply_capacity.csv)
- $t_{s,c}$: file_2_view_0, entry at row $s$, column $c$ (transportation_costs.csv)

All data is used as returned by CSVQA with no additional filtering.