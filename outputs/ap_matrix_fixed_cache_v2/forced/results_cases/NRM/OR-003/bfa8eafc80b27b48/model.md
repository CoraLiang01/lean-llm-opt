#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of suppliers (from supply_capacity.csv, column "Unnamed: 0")
- $C$: set of customer groups (from customer_demand.csv, column "customer")

**Parameters:**
- $cap_s$: supply capacity of supplier $s \in S$ (from supply_capacity.csv, column "supply_capacity")
- $dem_c$: demand of customer group $c \in C$ (from customer_demand.csv, column "demand")
- $cost_{s,c}$: transportation cost per unit from supplier $s$ to customer group $c$ (from transportation_costs.csv, columns "Unnamed: 0" for supplier, "C1"..."C10" for customer group)

**Decision Variables:**
- $x_{s,c} \geq 0$: quantity of goods transported from supplier $s$ to customer group $c$

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Supply Capacity Constraints (for each supplier):**
   \[
   \sum_{c \in C} x_{s,c} \leq cap_s \quad \forall s \in S
   \]

2. **Demand Satisfaction Constraints (for each customer group):**
   \[
   \sum_{s \in S} x_{s,c} = dem_c \quad \forall c \in C
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- **Supplier set $S$ and capacities $cap_s$**:  
  Table: supply_capacity.csv (table_id: file_1_view_0),  
  Columns: "Unnamed: 0" (supplier), "supply_capacity"

- **Customer group set $C$ and demands $dem_c$**:  
  Table: customer_demand.csv (table_id: file_0_view_0),  
  Columns: "customer" (customer_group), "demand"

- **Transportation costs $cost_{s,c}$**:  
  Table: transportation_costs.csv (table_id: file_2_view_0),  
  Row index: "Unnamed: 0" (supplier),  
  Columns: "C1", ..., "C10" (customer group)

All records from each table are included as returned by CSVQA, with no additional filtering.