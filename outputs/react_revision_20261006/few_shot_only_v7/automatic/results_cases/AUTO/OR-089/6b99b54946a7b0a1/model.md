##### Decision Variables

- $y_i \in \{0,1\}$: 1 if service centre $i \in I$ is opened, 0 otherwise.
- $x_{ij} \in \{0,1\}$: 1 if customer $j \in J$ is assigned to service centre $i \in I$, 0 otherwise.

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
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]

2. **Customers assigned only to opened centres:**
   \[
   x_{ij} \leq y_i, \quad \forall i \in I, \forall j \in J
   \]

3. **Each centre serves at most 4 customers:**
   \[
   \sum_{j \in J} x_{ij} \leq 4 y_i, \quad \forall i \in I
   \]

4. **Variable domains:**
   \[
   x_{ij} \in \{0,1\}, \quad y_i \in \{0,1\}
   \]

---

##### Data Mapping

- $I$ (service centres): SC1, SC2, SC3, SC4, SC5, SC6, SC7, SC8, SC9, SC10  
  (from service_centers_fixed_costs.csv, column "Service Center" and expanded_customer_service_costs.csv, columns SC1–SC10)
- $J$ (customers): C1, C2, ..., C15  
  (from expanded_customer_service_costs.csv, column "Customer")
- $f_i$: "Fixed Opening Cost" for each SC$i$ from service_centers_fixed_costs.csv
- $c_{ij}$: entry in expanded_customer_service_costs.csv, row "Customer" = C$j$, column SC$i$
- All constraints and variables are as described above, with indices and coefficients directly mapped to the CSV columns and rows.