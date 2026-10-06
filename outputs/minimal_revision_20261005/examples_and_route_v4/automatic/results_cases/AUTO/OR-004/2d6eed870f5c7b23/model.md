**Abstract Mathematical Model**

**Index Sets:**
- $S$: set of distribution centers (from `file_1_view_0`, column `Unnamed: 0`)
- $C$: set of customer groups (from `file_0_view_0`, column `customer`)

**Parameters:**
- $d_c$: demand of customer group $c \in C$ (from `file_0_view_0`, column `demand`)
- $u_s$: supply capacity of distribution center $s \in S$ (from `file_1_view_0`, column `supply_capacity`)
- $t_{sc}$: transportation cost per unit from distribution center $s$ to customer group $c$ (from `file_2_view_0`, row `Unnamed: 0` = $s$, column $c$)

**Decision Variables:**
- $x_{sc} \geq 0$: number of units shipped from distribution center $s$ to customer group $c$ (continuous, nonnegative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{sc} \, x_{sc}
\]

**Constraints:**

1. **Demand Satisfaction (for each customer group):**
   \[
   \sum_{s \in S} x_{sc} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity (for each distribution center):**
   \[
   \sum_{c \in C} x_{sc} \leq u_s, \quad \forall s \in S
   \]

3. **Nonnegativity:**
   \[
   x_{sc} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

**Data Mapping**

- $S$: All values of `Unnamed: 0` in `file_1_view_0` (supply_capacity.csv)
- $C$: All values of `customer` in `file_0_view_0` (customer_demand.csv)
- $d_c$: `demand` in `file_0_view_0`, keyed by `customer`
- $u_s$: `supply_capacity` in `file_1_view_0`, keyed by `Unnamed: 0`
- $t_{sc}$: value in `file_2_view_0`, row `Unnamed: 0` = $s$, column $c$ (transportation_costs.csv)
- $x_{sc}$: decision variable for each $(s, c) \in S \times C$