##### Abstract Model

**Index sets**  
\(I\) – set of suppliers (supplier identifiers)  
\(J\) – set of stores (store identifiers)

**Parameters**  
\(f_i\) – fixed cost of opening supplier \(i\)  \(i \in I\)  
\(t_{ij}\) – transportation cost per unit from supplier \(i\) to store \(j\)  \(i \in I,\; j \in J\)  
\(d_j\) – total demand of store \(j\)  \(j \in J\)  
\(M\) – a sufficiently large constant (e.g., \(M=\sum_{j\in J} d_j\))

**Decision variables**  
\(y_i \in \{0,1\}\)  \(i \in I\) (1 if supplier \(i\) is opened, 0 otherwise)  
\(x_{ij} \ge 0\)  \(i \in I,\; j \in J\) (quantity shipped from supplier \(i\) to store \(j\))

**Objective Function**  
\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
\[
\sum_{i\in I} x_{ij} \;\ge\; d_j \quad \forall j\in J
\]

2. **Linking supplier activation to shipments**  
\[
x_{ij} \;\le\; M\,y_i \quad \forall i\in I,\; j\in J
\]

3. **Non‑negativity**  
\[
x_{ij} \;\ge\; 0 \quad \forall i\in I,\; j\in J
\]

4. **Binary supplier variables**  
\[
y_i \in \{0,1\} \quad \forall i\in I
\]

---

##### Data Mapping

| Source | Table ID | Column Name | Parameter |
|--------|----------|-------------|-----------|
| fixed_cost.csv | fixed_cost | Unnamed: 0 | supplier_id (maps to \(i\)) |
| fixed_cost.csv | fixed_cost | fixed_costs | \(f_i\) |
| transportation_costs.csv | transportation_costs | Unnamed: 0 | supplier_id (maps to \(i\)) |
| transportation_costs.csv | transportation_costs | CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT | store_id (maps to \(j\)) |
| transportation_costs.csv | transportation_costs | values in store columns | \(t_{ij}\) |
| demand.csv | demand | Customer | store_id (maps to \(j\)) |
| demand.csv | demand | demand | \(d_j\) |

*All mappings use the exact table identifiers and column names as provided in the CSV files.*