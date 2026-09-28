#### Abstract Mathematical Model

**Index Sets:**
- $S$: set of warehouses (indexed by $s$)
- $C$: set of retail stores (indexed by $c$)

**Parameters:**
- $d_c$: daily demand at store $c$  
  (Data: table_id = file_0_view_0, column = demand)
- $u_s$: daily supply capacity at warehouse $s$  
  (Data: table_id = file_1_view_0, column = supply_capacity)
- $t_{s,c}$: unit transportation cost from warehouse $s$ to store $c$  
  (Data: table_id = file_2_view_0, columns = [Unnamed: 0] for $s$, [C1, ..., Cn] for $c$)

**Decision Variables:**
- $x_{s,c} \geq 0$: quantity of product shipped from warehouse $s$ to store $c$

**Objective:**
\[
\min \sum_{s \in S} \sum_{c \in C} t_{s,c} \cdot x_{s,c}
\]

**Constraints:**

1. **Demand Satisfaction (for each store):**
\[
\sum_{s \in S} x_{s,c} = d_c \qquad \forall c \in C
\]

2. **Warehouse Supply Capacity (for each warehouse):**
\[
\sum_{c \in C} x_{s,c} \leq u_s \qquad \forall s \in S
\]

3. **Non-negativity:**
\[
x_{s,c} \geq 0 \qquad \forall s \in S,\, c \in C
\]

---

#### Data Mapping

- **Store set $C$ and demand $d_c$:**  
  Source: customer_demand.csv  
  Table ID: file_0_view_0  
  Column: customer (store index), demand (parameter)

- **Warehouse set $S$ and supply capacity $u_s$:**  
  Source: supply_capacity.csv  
  Table ID: file_1_view_0  
  Column: Unnamed: 0 (warehouse index), supply_capacity (parameter)

- **Transportation cost $t_{s,c}$:**  
  Source: transportation_costs.csv  
  Table ID: file_2_view_0  
  Row: Unnamed: 0 (warehouse index $s$)  
  Columns: C1, ..., Cn (store index $c$), values are $t_{s,c}$

---

This model determines the optimal shipment plan from warehouses to stores to minimize total transportation cost, while meeting all store demands and not exceeding any warehouse's supply capacity.