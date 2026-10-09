#### Abstract Mathematical Model

**Index Sets:**
- $S$: Set of suppliers (warehouses), from column "Suppliers" in table_id file_1_view_0 and row identifiers in table_id file_2_view_0.
- $C$: Set of customers (stores), from column "Customers" in table_id file_0_view_0 and column headers in table_id file_2_view_0.

**Parameters:**
- $d_c$: Demand of customer $c \in C$, from column "demand" in table_id file_0_view_0.
- $u_s$: Supply capacity of supplier $s \in S$, from column "supply_capacity" in table_id file_1_view_0.
- $t_{s,c}$: Transportation cost per unit from supplier $s$ to customer $c$, from table_id file_2_view_0, with rows indexed by "Unnamed: 0" (supplier) and columns by customer.

**Decision Variables:**
- $x_{s,c} \geq 0$: Amount of fresh produce shipped from supplier $s$ to customer $c$.

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction (for each customer):**
   \[
   \sum_{s \in S} x_{s,c} = d_c, \quad \forall c \in C
   \]

2. **Supply Capacity (for each supplier):**
   \[
   \sum_{c \in C} x_{s,c} \leq u_s, \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0, \quad \forall s \in S,\, c \in C
   \]

---

**Data Mapping:**

- $S$ (suppliers): table_id file_1_view_0, column "Suppliers"; table_id file_2_view_0, row "Unnamed: 0"
- $C$ (customers): table_id file_0_view_0, column "Customers"; table_id file_2_view_0, columns "Customer1", ..., "Customer6"
- $d_c$: table_id file_0_view_0, column "demand"
- $u_s$: table_id file_1_view_0, column "supply_capacity"
- $t_{s,c}$: table_id file_2_view_0, row "Unnamed: 0" (supplier), columns "Customer1", ..., "Customer6" (customer)

No additional filtering or selection was applied; all records from each file are included as per the query.