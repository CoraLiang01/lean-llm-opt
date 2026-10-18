**Abstract Model**

| Symbol | Meaning |
|--------|---------|
| \(I\) | Set of suppliers \(i=1,\dots,|I|\) |
| \(J\) | Set of supermarkets \(j=1,\dots,|J|\) |
| \(f_i\) | Fixed opening cost of supplier \(i\) (from *fixed_cost.csv*) |
| \(t_{ij}\) | Transportation cost per unit from supplier \(i\) to supermarket \(j\) (from *transportation_costs.csv*) |
| \(d_j\) | Demand of supermarket \(j\) (from *demand.csv*) |
| \(y_i\) | Binary variable, 1 if supplier \(i\) is opened, 0 otherwise |
| \(x_{ij}\) | Quantity of goods shipped from supplier \(i\) to supermarket \(j\) |

---

### Objective Function
\[
\min \quad \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij}
\]

### Constraints

1. **Demand Satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;=\; d_j , \qquad \forall j\in J
   \]

2. **Supplier Activation (Big‑M)**  
   \[
   x_{ij} \;\le\; d_j \, y_i , \qquad \forall i\in I,\; \forall j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij} \;\ge\; 0 , \qquad \forall i\in I,\; \forall j\in J
   \]

4. **Binary Supplier Variables**  
   \[
   y_i \;\in\; \{0,1\} , \qquad \forall i\in I
   \]

---

### Data Mapping

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(t_{ij}\) | `file_2_view_0` | `C1` … `C12` (columns for each supermarket) |
| \(d_j\) | `file_0_view_0` | `demand` |

*No record values or counts are reproduced; only the table identifiers and column names are listed.*