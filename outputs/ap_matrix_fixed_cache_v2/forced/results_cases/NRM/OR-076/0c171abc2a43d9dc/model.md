#### Index Sets

- $W$: set of potential warehouse locations (from file_1_view_0, column "Warehouse ID")
- $C$: set of customers (from file_2_view_0, column "Customer ID")

#### Parameters

- $F_w$: fixed annual opening cost for warehouse $w \in W$ (file_1_view_0, column "Fixed_Cost")
- $K_w$: maximum service capacity of warehouse $w \in W$ (file_1_view_0, column "Capacity")
- $d_c$: demand of customer $c \in C$ (file_2_view_0, column "Demand")
- $t_{w,c}$: transportation cost per unit from warehouse $w$ to customer $c$ (file_0_view_0, entry at row "Warehouse ID" $w$, column $c$)

#### Decision Variables

- $y_w \in \{0,1\}$: 1 if warehouse $w$ is opened, 0 otherwise
- $x_{w,c} \geq 0$: amount of customer $c$'s demand served from warehouse $w$

#### Objective

$$
\min \sum_{w \in W} F_w y_w + \sum_{w \in W} \sum_{c \in C} t_{w,c} x_{w,c}
$$

#### Constraints

1. **Demand Satisfaction:**  
   $$
   \sum_{w \in W} x_{w,c} = d_c, \quad \forall c \in C
   $$

2. **Warehouse Capacity:**  
   $$
   \sum_{c \in C} x_{w,c} \leq K_w y_w, \quad \forall w \in W
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

- $W$, $F_w$, $K_w$: file_1_view_0, columns "Warehouse ID", "Fixed_Cost", "Capacity"
- $C$, $d_c$: file_2_view_0, columns "Customer ID", "Demand"
- $t_{w,c}$: file_0_view_0, row "Warehouse ID", columns "C1"–"C20" (customer IDs)

All records from the above tables are included as returned by CSVQA.