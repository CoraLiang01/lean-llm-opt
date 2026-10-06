##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to centre $i \in I$, 0 otherwise.

##### Parameters

- $I = \{\text{SC1}, \text{SC2}, \ldots, \text{SC10}\}$: set of candidate service centres.
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C15}\}$: set of customers.
- $f_i$: fixed opening cost for centre $i \in I$ (from service_centers_fixed_costs.csv).
- $c_{ij}$: cost to serve customer $j \in J$ from centre $i \in I$ (from expanded_customer_service_costs.csv).

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Each customer assigned to exactly one centre:**
   \[
   \sum_{i \in I} x_{ij} = 1 \quad \forall j \in J
   \]

2. **Customers assigned only to opened centres:**
   \[
   x_{ij} \leq y_i \quad \forall i \in I,\, j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\} \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Service Centres ($I$) and Fixed Opening Costs ($f_i$):**  
  From service_centers_fixed_costs.csv, columns:  
  - "Service Center": SC1, SC2, ..., SC10  
  - "Fixed Opening Cost": $f_{\text{SC1}}$, $f_{\text{SC2}}$, ..., $f_{\text{SC10}}$

- **Customers ($J$):**  
  From expanded_customer_service_costs.csv, column "Customer": C1, C2, ..., C15

- **Service Costs ($c_{ij}$):**  
  From expanded_customer_service_costs.csv, columns SC1–SC10 for each row "Customer" C1–C15:  
  - $c_{ij}$ is the entry in row $j$ (customer $j$) and column $i$ (centre $i$).

---

**All parameters, sets, and indices are defined exactly as in the CSV files.**