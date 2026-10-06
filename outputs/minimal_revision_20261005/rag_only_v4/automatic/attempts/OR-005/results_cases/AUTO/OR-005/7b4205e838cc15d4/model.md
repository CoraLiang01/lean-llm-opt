**Mathematical Optimization Model**

**Sets:**
- Let \( S = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\} \)  // Distribution centers (suppliers)
- Let \( D = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\} \)  // Customer groups

**Parameters:**
- \( \text{demand}_j \): Daily demand for customer group \( j \in D \)  
  [file_0_view_0, column: demand, key: Customers]
- \( \text{supply\_capacity}_i \): Daily supply capacity of distribution center \( i \in S \)  
  [file_1_view_0, column: supply_capacity, key: Supplier]
- \( c_{ij} \): Transportation cost per unit from supplier \( i \) to customer \( j \)  
  [file_2_view_0, columns: demand1–demand8, row: Unnamed: 0 (see Data Mapping below)]

**Decision Variables:**
- \( x_{ij} \geq 0 \): Quantity shipped from supplier \( i \) to customer \( j \)

**Objective:**
\[
\min \sum_{i \in S} \sum_{j \in D} c_{ij} \, x_{ij}
\]

**Constraints:**
1. **Demand satisfaction:**  
  For all \( j \in D \):
\[
\sum_{i \in S} x_{ij} = \text{demand}_j
\]

2. **Supply capacity:**  
  For all \( i \in S \):
\[
\sum_{j \in D} x_{ij} \leq \text{supply\_capacity}_i
\]

3. **Non-negativity:**  
  For all \( i \in S,\, j \in D \):
\[
x_{ij} \geq 0
\]

---

**Data Mapping**

- **Customer groups \( D \) and their demands (\( \text{demand}_j \)):**  
  From table_id: file_0_view_0  
   - Index: Customers  
   - Parameter: demand

- **Suppliers \( S \) and their capacities (\( \text{supply\_capacity}_i \)):**  
  From table_id: file_1_view_0  
   - Index: Supplier  
   - Parameter: supply_capacity

- **Transportation costs (\( c_{ij} \)):**  
  From table_id: file_2_view_0  
   - Row index: Unnamed: 0 (values: supply1–supply8, mapped to supplier1–supplier8)  
   - Column index: demand1–demand8  
   - For each \( i \in S \), \( j \in D \):  
    \( c_{ij} = \) value in row where Unnamed: 0 = supply\(k\) (mapped to supplier\(k\)), column = demand\(l\)

---

**Variable Domains:**  
\( x_{ij} \geq 0 \) for all \( i \in S,\, j \in D \)

**Objective Sense:**  
Minimize total transportation cost.