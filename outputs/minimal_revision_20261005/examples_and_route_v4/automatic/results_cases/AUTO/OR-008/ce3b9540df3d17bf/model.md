**Abstract Mathematical Model**

**Index Sets:**
- $S$: Set of suppliers (warehouses), from column `Suppliers` in `file_1_view_0`
- $C$: Set of customers (stores), from column `Customers` in `file_0_view_0$

**Parameters:**
- $d_c$: Demand of customer $c \in C$, from column `demand` in `file_0_view_0`
- $u_s$: Supply capacity of supplier $s \in S$, from column `supply_capacity` in `file_1_view_0`
- $t_{sc}$: Transportation cost per unit from supplier $s$ to customer $c$, from column $c$ in `file_2_view_0` where row `Unnamed: 0` = $s$

**Decision Variables:**
- $x_{sc} \geq 0$: Amount of product shipped from supplier $s$ to customer $c$ (continuous, nonnegative)

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{sc} \, x_{sc}
\]

**Constraints:**

1. **Demand Satisfaction (for each customer):**
   \[
   \sum_{s \in S} x_{sc} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity (for each supplier):**
   \[
   \sum_{c \in C} x_{sc} \leq u_s, \quad \forall s \in S
   \]

3. **Nonnegativity:**
   \[
   x_{sc} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

**Data Mapping**

- $S$: All values in `Suppliers` column of `file_1_view_0`
- $C$: All values in `Customers` column of `file_0_view_0`
- $d_c$: `file_0_view_0`, column `demand`, key `Customers`
- $u_s$: `file_1_view_0`, column `supply_capacity`, key `Suppliers`
- $t_{sc}$: `file_2_view_0`, value in column $c$ (customer) and row where `Unnamed: 0` = $s$ (supplier)

---

**Summary:**  
Shipments $x_{sc}$ from each supplier to each customer are chosen to minimize total transportation cost, while meeting all customer demands and not exceeding any warehouse's supply capacity. All data and index sets are mapped directly from the provided CSV files and columns.