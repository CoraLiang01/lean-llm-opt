**Mathematical Optimization Model**

**Sets:**
- \( S \): Set of suppliers (distribution centers), indexed by \( s \).
  - \( S = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\} \)
- \( D \): Set of customer groups, indexed by \( d \).
  - \( D = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\} \)

**Parameters:**
- \( \text{demand}_d \): Daily demand for customer group \( d \).
  - Data Mapping:  
    - Table: customer_demand.csv (table_id: file_0_view_0)
    - Column: demand
    - Index: Customers
- \( \text{supply\_capacity}_s \): Daily supply capacity of supplier \( s \).
  - Data Mapping:  
    - Table: supply_capacity.csv (table_id: file_1_view_0)
    - Column: supply_capacity
    - Index: Supplier
- \( c_{s,d} \): Transportation cost per unit from supplier \( s \) to customer group \( d \).
  - Data Mapping:  
    - Table: transportation_costs.csv (table_id: file_2_view_0)
    - Row: Unnamed: 0 (mapped to Supplier via row_id_mapping)
    - Columns: demand1, ..., demand8

**Decision Variables:**
- \( x_{s,d} \geq 0 \): Quantity of goods shipped from supplier \( s \) to customer group \( d \).

**Objective:**
\[
\min \sum_{s \in S} \sum_{d \in D} c_{s,d} \cdot x_{s,d}
\]

**Constraints:**

1. **Demand Satisfaction:**  
   For each customer group \( d \in D \):
   \[
   \sum_{s \in S} x_{s,d} = \text{demand}_d
   \]

2. **Supply Capacity:**  
   For each supplier \( s \in S \):
   \[
   \sum_{d \in D} x_{s,d} \leq \text{supply\_capacity}_s
   \]

3. **Non-negativity:**  
   \[
   x_{s,d} \geq 0 \quad \forall s \in S,\, d \in D
   \]

---

**Data Mapping**

- \( S \) (Suppliers): All values in column "Supplier" of table_id: file_1_view_0
- \( D \) (Customer Groups): All values in column "Customers" of table_id: file_0_view_0
- \( \text{demand}_d \): For each \( d \in D \), value in column "demand" of table_id: file_0_view_0, row where "Customers" = \( d \)
- \( \text{supply\_capacity}_s \): For each \( s \in S \), value in column "supply_capacity" of table_id: file_1_view_0, row where "Supplier" = \( s \)
- \( c_{s,d} \): For each \( s \in S \), \( d \in D \), value in table_id: file_2_view_0, row where "Unnamed: 0" maps to \( s \) via row_id_mapping, column \( d \)

---

**Variable Domains:**
- \( x_{s,d} \in [0, \infty) \) for all \( s \in S, d \in D \)

---

**Summary:**  
Shipments \( x_{s,d} \) from each supplier to each customer group are chosen to minimize total transportation cost, ensuring all customer demands are met and no supplier exceeds its capacity. All data is mapped directly from the provided tables and columns.