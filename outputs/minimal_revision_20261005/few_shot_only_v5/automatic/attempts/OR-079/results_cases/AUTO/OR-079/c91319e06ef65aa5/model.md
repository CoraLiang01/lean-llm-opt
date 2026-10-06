##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from factory $i \in I$ to distribution center $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if factory $i$ is constructed, 0 otherwise (binary).

##### Parameters

- $I = \{$A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15$\}$ (factory sites)
- $J = \{$B1, B2, B3, B4, B5, B6, B7, B8$\}$ (distribution centers)
- $f_i$: Fixed cost of constructing factory $i$ (see Data Mapping)
- $c_{ij}$: Shipping cost per unit from factory $i$ to distribution center $j$ (see Data Mapping)
- $d_j$: Demand at distribution center $j$ (see Data Mapping)
- $u_i$: Capacity of factory $i$ (see Data Mapping)

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction at each distribution center:**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Factory capacity (only if constructed):**
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i, \quad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

---

#### Data Mapping (from CSV columns)

- **Factories ($I$):** A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15
- **Distribution Centers ($J$):** B1, B2, B3, B4, B5, B6, B7, B8

- **Fixed Costs and Capacities ($f_i$, $u_i$):**  
  Source: facility_costs.csv  
  - Columns: Facility, FixedCost, Capacity

- **Shipping Costs ($c_{ij}$):**  
  Source: shipping_costs.csv  
  - Rows: Origin (A1–A15), Columns: B1–B8

- **Demand ($d_j$):**  
  Source: demand_requirements.csv  
  - Columns: Destination (B1–B8), Demand

---

**All parameters are to be taken directly from the corresponding CSV columns as described above, preserving all identifiers and values.**