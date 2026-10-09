#### Index Sets

- $W$: set of potential warehouse locations (from Warehouse ID in file_1_view_0)
- $C$: set of customers (from Customer ID in file_2_view_0)

#### Parameters

- $f_w$: fixed annual opening cost for warehouse $w \in W$ (Fixed_Cost, file_1_view_0)
- $cap_w$: maximum service capacity of warehouse $w \in W$ (Capacity, file_1_view_0)
- $d_c$: demand of customer $c \in C$ (Demand, file_2_view_0)
- $t_{w,c}$: variable transportation cost per unit from warehouse $w$ to customer $c$ (cost.csv, file_0_view_0, entry for Warehouse ID $w$, column $c$)

#### Decision Variables

- $y_w \in \{0,1\}$: 1 if warehouse $w$ is opened, 0 otherwise
- $x_{w,c} \geq 0$: amount of customer $c$'s demand served from warehouse $w$

#### Objective

$$
\min \quad \sum_{w \in W} f_w y_w + \sum_{w \in W} \sum_{c \in C} t_{w,c} x_{w,c}
$$

#### Constraints

1. **Demand Satisfaction:**  
   $$
   \sum_{w \in W} x_{w,c} = d_c, \quad \forall c \in C
   $$

2. **Warehouse Capacity:**  
   $$
   \sum_{c \in C} x_{w,c} \leq cap_w \cdot y_w, \quad \forall w \in W
   $$

3. **Variable Domains:**  
   $$
   y_w \in \{0,1\}, \quad \forall w \in W
   $$
   $$
   x_{w,c} \geq 0, \quad \forall w \in W, \forall c \in C
   $$

---

#### Data Mapping

- $W$, $f_w$, $cap_w$: file_1_view_0, columns Warehouse ID, Fixed_Cost, Capacity
- $C$, $d_c$: file_2_view_0, columns Customer ID, Demand
- $t_{w,c}$: file_0_view_0, row Warehouse ID $w$, column $c$ (where $c$ matches Customer ID in file_2_view_0)

All data used as returned by CSVQA; no additional filtering or transformation applied.