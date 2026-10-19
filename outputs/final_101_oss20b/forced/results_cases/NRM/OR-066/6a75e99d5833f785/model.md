##### Abstract Model  

**Index Sets**  
\(I\) – set of suppliers (indexed by `supplier_id` from *fixed_cost.csv*)  
\(J\) – set of supermarkets (indexed by `supermarket_id` from *demand.csv* and *transportation_costs.csv*)  

**Parameters**  
\(f_i\) – fixed activation cost of supplier \(i\) (from *fixed_cost.csv.fixed_costs*)  
\(c_{ij}\) – per‑unit transportation cost from supplier \(i\) to supermarket \(j\) (from *transportation_costs.csv* columns `C1`, `C2`, …)  
\(d_j\) – total demand of supermarket \(j\) (from *demand.csv.demand*)  

**Decision Variables**  
\(y_i \in \{0,1\}\) – 1 if supplier \(i\) is activated, 0 otherwise  
\(x_{ij} \ge 0\) – quantity of goods supplied from supplier \(i\) to supermarket \(j\)  

**Objective Function**  
\[
\min \quad \sum_{i \in I} f_i \, y_i \;+\; \sum_{i \in I}\sum_{j \in J} c_{ij} \, x_{ij}
\]

**Constraints**  

1. **Demand Satisfaction**  
\[
\sum_{i \in I} x_{ij} \;=\; d_j, \qquad \forall j \in J
\]

2. **Supplier Activation Coupling**  
\[
x_{ij} \;\le\; d_j \, y_i, \qquad \forall i \in I,\; \forall j \in J
\]

3. **Non‑negativity**  
\[
x_{ij} \;\ge\; 0, \qquad \forall i \in I,\; \forall j \in J
\]

4. **Binary Activation**  
\[
y_i \;\in\; \{0,1\}, \qquad \forall i \in I
\]

##### Data Mapping  

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\)   | fixed_cost.csv | fixed_costs |
| \(c_{ij}\)| transportation_costs.csv | `Cj` (where `j` is the supermarket identifier) |
| \(d_j\)   | demand.csv | demand |

Indices \(i\) and \(j\) are taken from the `Unnamed: 0` column of *fixed_cost.csv* and the `customer` column of *demand.csv*, respectively. No record values or counts are included.