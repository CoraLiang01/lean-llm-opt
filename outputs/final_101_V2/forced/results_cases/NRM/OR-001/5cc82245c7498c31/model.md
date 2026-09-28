#### Abstract Mathematical Model

Let:
- $S$ = set of distribution centers (indexed by $s$)
- $C$ = set of customer groups (indexed by $c$)

Parameters:
- $d_c$ = daily demand of customer group $c$
- $u_s$ = daily supply capacity of distribution center $s$
- $a_{s,c}$ = transportation cost per unit from distribution center $s$ to customer group $c$

Decision Variables:
- $x_{s,c} \geq 0$ = quantity of goods shipped from distribution center $s$ to customer group $c$

Objective:
$$
\min \sum_{s \in S} \sum_{c \in C} a_{s,c} \cdot x_{s,c}
$$

Subject to:

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

- $S$ (distribution centers): Identifiers from column "Unnamed: 0" in table_id file_1_view_0 ("supply_capacity.csv") and file_2_view_0 ("transportation_costs.csv")
- $C$ (customer groups): Identifiers from column "customer" in table_id file_0_view_0 ("customer_demand.csv") and columns "C1", ..., "C18" in table_id file_2_view_0 ("transportation_costs.csv")
- $d_c$: Parameter from column "demand" in table_id file_0_view_0 ("customer_demand.csv")
- $u_s$: Parameter from column "supply_capacity" in table_id file_1_view_0 ("supply_capacity.csv")
- $a_{s,c}$: Parameter from columns "C1", ..., "C18" in table_id file_2_view_0 ("transportation_costs.csv"), with row index "Unnamed: 0" matching $s$ and column matching $c$