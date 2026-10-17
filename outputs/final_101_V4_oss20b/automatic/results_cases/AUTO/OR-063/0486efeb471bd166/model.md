**Mathematical Model**

Let  

* \(I=\{S1,S2,\dots ,S7\}\) – warehouses  
* \(J=\{C1,C2,\dots ,C7\}\) – musicians/bands  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | fixed cost of activating warehouse \(i\) | `file_1_view_0` column `fixed_costs` (index `Unnamed: 0`) |
| \(c_{ij}\) | unit transportation cost from warehouse \(i\) to customer \(j\) | `file_2_view_0` columns `C1`–`C7` (index `Unnamed: 0`) |
| \(d_j\) | demand of customer \(j\) | `file_0_view_0` column `demand` (index `customer`) |
| \(M\) | upper bound on total shipment from any warehouse | \(M=\sum_{j\in J} d_j\) |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if warehouse \(i\) is activated, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from warehouse \(i\) to customer \(j\)

Objective  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}=d_j ,\qquad \forall j\in J
   \]

2. **Activation (big‑M)**  
   \[
   \sum_{j\in J} x_{ij}\;\le\; M\,y_i ,\qquad \forall i\in I
   \]

3. **Variable domains**  
   \[
   x_{ij}\ge 0,\; y_i\in\{0,1\},\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Key column | Value column |
|-----------|----------|------------|--------------|
| \(f_i\) | `file_1_view_0` | `Unnamed: 0` | `fixed_costs` |
| \(c_{ij}\) | `file_2_view_0` | `Unnamed: 0` | `C1`–`C7` (column matching \(j\)) |
| \(d_j\) | `file_0_view_0` | `customer` | `demand` |

All indices \(i\) and \(j\) are taken from the distinct values appearing in the corresponding key columns. No additional capacity limits are imposed beyond the big‑M bound \(M\).