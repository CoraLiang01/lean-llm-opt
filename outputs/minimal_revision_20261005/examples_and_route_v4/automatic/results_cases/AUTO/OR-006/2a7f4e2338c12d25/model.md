**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of warehouses (from `file_1_view_0`, column `Unnamed: 0`)
- $C$: set of customers/stores (from `file_0_view_0`, column `customer`)

**Parameters:**
- $d_c$: demand of customer $c \in C$ (from `file_0_view_0`, column `demand`)
- $u_s$: supply capacity of warehouse $s \in S$ (from `file_1_view_0`, column `supply_capacity`)
- $t_{s,c}$: transportation cost per unit from warehouse $s$ to customer $c$ (from `file_2_view_0`, row `Unnamed: 0` = $s$, column $c$)

**Decision Variables:**
- $x_{s,c} \geq 0$: quantity shipped from warehouse $s$ to customer $c$ (continuous, nonnegative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction (for each customer):**
   \[
   \sum_{s \in S} x_{s,c} = d_c \qquad \forall c \in C
   \]

2. **Warehouse Supply Capacity (for each warehouse):**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s \qquad \forall s \in S
   \]

3. **Nonnegativity:**
   \[
   x_{s,c} \geq 0 \qquad \forall s \in S,\, c \in C
   \]

---

**Data Mapping**

- $S$: All values of `Unnamed: 0` in `file_1_view_0` (supply_capacity.csv)
- $C$: All values of `customer` in `file_0_view_0` (customer_demand.csv)
- $d_c$: `file_0_view_0`, column `demand`, key `customer`
- $u_s$: `file_1_view_0`, column `supply_capacity`, key `Unnamed: 0`
- $t_{s,c}$: `file_2_view_0`, row `Unnamed: 0` = $s$, column $c$ (transportation_costs.csv)
- $x_{s,c}$: Decision variable for each $(s, c) \in S \times C$