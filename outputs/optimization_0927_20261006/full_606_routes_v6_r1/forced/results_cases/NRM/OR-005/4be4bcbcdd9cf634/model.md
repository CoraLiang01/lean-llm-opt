#### Index Sets

- $S$: set of distribution centers (suppliers), from supply_capacity.csv [file_1_view_0, column: Supplier] and transportation_costs.csv [file_2_view_0, column: Unnamed: 0]
- $C$: set of customer groups, from customer_demand.csv [file_0_view_0, column: Customers] and transportation_costs.csv [file_2_view_0, columns: demand1, ..., demand8]

#### Parameters

- $d_c$: daily demand of customer group $c \in C$  
  [from customer_demand.csv, file_0_view_0, column: demand]
- $u_s$: daily supply capacity of distribution center $s \in S$  
  [from supply_capacity.csv, file_1_view_0, column: supply_capacity]
- $t_{s,c}$: transportation cost per unit from distribution center $s$ to customer group $c$  
  [from transportation_costs.csv, file_2_view_0, row: Unnamed: 0 = $s$, column: $c$]

#### Decision Variables

- $x_{s,c} \geq 0$: quantity of goods shipped from distribution center $s \in S$ to customer group $c \in C$

#### Objective

$$
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Demand Satisfaction:**  
  For all $c \in C$,
$$
\sum_{s \in S} x_{s,c} = d_c
$$

2. **Supply Capacity:**  
  For all $s \in S$,
$$
\sum_{c \in C} x_{s,c} \leq u_s
$$

3. **Non-negativity:**  
  For all $s \in S$, $c \in C$,
$$
x_{s,c} \geq 0
$$

---

#### Data Mapping

- **customer_demand.csv** [table_id: file_0_view_0]:  
  - Customer group set $C$ from column "Customers"  
  - Parameter $d_c$ from column "demand"
- **supply_capacity.csv** [table_id: file_1_view_0]:  
  - Distribution center set $S$ from column "Supplier"  
  - Parameter $u_s$ from column "supply_capacity"
- **transportation_costs.csv** [table_id: file_2_view_0]:  
  - Distribution center set $S$ from column "Unnamed: 0"  
  - Customer group set $C$ from columns "demand1", ..., "demand8"  
  - Parameter $t_{s,c}$ from intersection of row "Unnamed: 0" = $s$ and column $c$

All data is used as returned by the query, with no additional filtering or transformation.