#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of suppliers (from supply_capacity.csv, column "Unnamed: 0")
- $C$: set of customer groups (from customer_demand.csv, column "customer")

**Parameters:**
- $a_s$: supply capacity of supplier $s \in S$ (from supply_capacity.csv, column "supply_capacity")
- $d_c$: demand of customer group $c \in C$ (from customer_demand.csv, column "demand")
- $c_{s,c}$: transportation cost per unit from supplier $s$ to customer group $c$ (from transportation_costs.csv, column "cost_per_unit" for each $(s,c)$ pair)

**Decision Variables:**
- $x_{s,c} \geq 0$: quantity of goods transported from supplier $s$ to customer group $c$

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} c_{s,c} \cdot x_{s,c}
\]

**Constraints:**
1. **Supply Capacity (for each supplier):**
   \[
   \sum_{c \in C} x_{s,c} \leq a_s \quad \forall s \in S
   \]
2. **Demand Satisfaction (for each customer group):**
   \[
   \sum_{s \in S} x_{s,c} = d_c \quad \forall c \in C
   \]
3. **Non-negativity:**
   \[
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   \]

---

**Data Mapping:**

- Table: supply_capacity.csv, table_id: file_1_view_0
  - Supplier index set $S$: column "Unnamed: 0"
  - Parameter $a_s$: column "supply_capacity"
- Table: customer_demand.csv, table_id: file_0_view_0
  - Customer group index set $C$: column "customer"
  - Parameter $d_c$: column "demand"
- Table: transportation_costs.csv, table_id: file_2_view_0
  - Parameter $c_{s,c}$: entry at row $s$ (column "Unnamed: 0"), column $c$ (column header matches "customer" in customer_demand.csv)

No literal values or record counts are included; all identifiers and relationships are preserved as in the source.