**Index Sets**

- \(I\) : set of suppliers, indexed by the supplier identifiers that appear in the file *fixed_cost.csv* (table_id = `file_1_view_0`).  
- \(J\) : set of supermarkets (customers), indexed by the customer identifiers that appear in the file *demand.csv* (table_id = `file_0_view_0`).  

**Parameters**

- \(f_i\) : fixed opening cost of supplier \(i\).  
  - Source: table_id = `file_1_view_0`, column = `fixed_costs`.  
- \(t_{ij}\) : transportation cost per unit from supplier \(i\) to supermarket \(j\).  
  - Source: table_id = `file_2_view_0`, row identifier column = `Unnamed: 0` (supplier id), column identifier `Cj` (customer id).  
- \(d_j\) : total demand of supermarket \(j\).  
  - Source: table_id = `file_0_view_0`, column = `demand`.  

**Decision Variables**

- \(y_i \in \{0,1\}\) (1 if supplier \(i\) is opened, 0 otherwise).  
- \(x_{ij} \in \mathbb{Z}_{\ge 0}\) quantity of goods supplied from supplier \(i\) to supermarket \(j\).  

**Objective Function**

\[
\min \;\;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; d_j , \qquad \forall j\in J
   \]

2. **Supplier activation linkage**  
   \[
   x_{ij} \;\le\; d_j\,y_i , \qquad \forall i\in I,\; \forall j\in J
   \]

3. **Non‑negativity and integrality**  
   \[
   x_{ij} \;\ge\; 0 , \qquad \forall i\in I,\; \forall j\in J
   \]
   \[
   y_i \;\in\; \{0,1\} , \qquad \forall i\in I
   \]

**Data Mapping**

| Parameter | Source Table ID | Source Column |
|-----------|-----------------|---------------|
| \(f_i\)   | `file_1_view_0` | `fixed_costs` |
| \(t_{ij}\)| `file_2_view_0` | `Cj` (customer column) |
| \(d_j\)   | `file_0_view_0` | `demand` |

*No record values or counts are reproduced; only the mapping of parameters to their source tables and columns is provided.*