##### Decision Variables

- $x_{ij} \geq 0$: Number of vehicles shipped from supplier $i \in I$ to dealership $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is operational (open), 0 otherwise (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Dealership Demand Satisfaction:**  
   For each dealership $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   where $d_j$ is the demand of dealership $j$.

2. **Supplier Activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M_i y_i
   \]
   where $M_i$ is a sufficiently large upper bound (e.g., $M_i = \sum_{j \in J} d_j$).

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Data Mapping

- **Suppliers ($I$):** S1, S2, S3, S4, S5, S6, S7, S8  
  (from `fixed_cost.csv` and `transportation_costs.csv`)

- **Dealerships ($J$):** C1, C2, C3, C4, C5, C6, C7, C8, C9  
  (from `demand.csv` and `transportation_costs.csv`)

- **Demands ($d_j$):**  
  - C1: 4,742,532,000  
  - C2: 1,600,594,000  
  - C3: 5,086,889,000  
  - C4: 1,027,326,000  
  - C5: 11,926,044,000  
  - C6: 9,058,407,000  
  - C7: 5,344,367,000  
  - C8: 677,201,000  
  - C9: 3,236,493,000  
  (from `demand.csv`)

- **Fixed Costs ($f_i$):**  
  - S1: 100.64  
  - S2: 98.72  
  - S3: 100.18  
  - S4: 96.58  
  - S5: 95.75  
  - S6: 99.06  
  - S7: 101.78  
  - S8: 93.86  
  (from `fixed_cost.csv`)

- **Transportation Costs ($c_{ij}$):**  
  $c_{ij}$ is the cost per vehicle from supplier $i$ to dealership $j$, as given in `transportation_costs.csv`.  
  (Rows: S1–S8; Columns: C1–C9)

- **Big-M ($M_i$):**  
  $M_i = \sum_{j \in J} d_j = 41,699,853,000$ for all $i$ (sum of all dealership demands).

##### Source-Column Data Mapping

- `demand.csv`:  
  - customer $\rightarrow$ $j$ (dealership index)  
  - demand $\rightarrow$ $d_j$

- `fixed_cost.csv`:  
  - Unnamed: 0 $\rightarrow$ $i$ (supplier index)  
  - fixed_costs $\rightarrow$ $f_i$

- `transportation_costs.csv`:  
  - Unnamed: 0 $\rightarrow$ $i$ (supplier index)  
  - C1–C9 $\rightarrow$ $c_{ij}$ for each $j$

All indices, parameters, and mappings are preserved as in the source data.