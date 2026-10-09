#### Index Sets

- $S$: set of suppliers (from supply_capacity.csv, column "Unnamed: 0")
- $C$: set of customer groups (from customer_demand.csv, column "customer")

#### Parameters

- $cap_s$: supply capacity of supplier $s \in S$ (from supply_capacity.csv, column "supply_capacity")
- $dem_c$: demand of customer group $c \in C$ (from customer_demand.csv, column "demand")
- $cost_{s,c}$: transportation cost per unit from supplier $s$ to customer group $c$ (from transportation_costs.csv, columns "Unnamed: 0" for supplier, columns $C$ for customers)

#### Decision Variables

- $x_{s,c} \geq 0$: quantity of goods transported from supplier $s$ to customer group $c$

#### Objective

$$
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
$$

#### Constraints

1. **Supply Capacity Constraints (for each supplier):**
   $$
   \sum_{c \in C} x_{s,c} \leq cap_s \quad \forall s \in S
   $$

2. **Demand Satisfaction Constraints (for each customer group):**
   $$
   \sum_{s \in S} x_{s,c} = dem_c \quad \forall c \in C
   $$

3. **Non-negativity:**
   $$
   x_{s,c} \geq 0 \quad \forall s \in S, \forall c \in C
   $$

---

#### Data Mapping

- Table: supply_capacity.csv, table_id: file_1_view_0, columns: "Unnamed: 0" (supplier index), "supply_capacity" (parameter $cap_s$)
- Table: customer_demand.csv, table_id: file_0_view_0, columns: "customer" (customer group index), "demand" (parameter $dem_c$)
- Table: transportation_costs.csv, table_id: file_2_view_0, columns: "Unnamed: 0" (supplier index), columns $C$ (customer group indices), entries (parameter $cost_{s,c}$ for each $s \in S$, $c \in C$)