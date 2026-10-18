#### Sets
- $S$: set of warehouses (indexed by $s$), from supply_capacity.csv column "Unnamed: 0" and transportation_costs.csv row "Unnamed: 0"
- $C$: set of stores/customers (indexed by $c$), from customer_demand.csv column "customer" and transportation_costs.csv columns "C1", ..., "C10"

#### Parameters
- $demand_c$: daily demand for store $c \in C$ (from customer_demand.csv, column "demand")
- $capacity_s$: daily supply capacity of warehouse $s \in S$ (from supply_capacity.csv, column "supply_capacity")
- $cost_{s,c}$: unit transportation cost from warehouse $s$ to store $c$ (from transportation_costs.csv, entry at row $s$, column $c$)

#### Decision Variables
- $x_{s,c} \geq 0$: quantity of product shipped from warehouse $s$ to store $c$

#### Objective
\[
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\]

#### Constraints

1. **Demand Satisfaction (for each store):**
   \[
   \sum_{s \in S} x_{s,c} = demand_c \quad \forall c \in C
   \]

2. **Warehouse Capacity (for each warehouse):**
   \[
   \sum_{c \in C} x_{s,c} \leq capacity_s \quad \forall s \in S
   \]

3. **Non-negativity:**
   \[
   x_{s,c} \geq 0 \quad \forall s \in S,\, c \in C
   \]

---

#### Data Mapping

- **customer_demand.csv**:  
  - Table ID: file_0_view_0  
  - Columns: "customer" (store index $c$), "demand" (parameter $demand_c$)

- **supply_capacity.csv**:  
  - Table ID: file_1_view_0  
  - Columns: "Unnamed: 0" (warehouse index $s$), "supply_capacity" (parameter $capacity_s$)

- **transportation_costs.csv**:  
  - Table ID: file_2_view_0  
  - Row: "Unnamed: 0" (warehouse index $s$)  
  - Columns: "C1", ..., "C10" (store index $c$), values are $cost_{s,c}$

All records from each table are included as returned by CSVQA.