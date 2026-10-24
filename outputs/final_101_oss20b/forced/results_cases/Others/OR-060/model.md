**Mathematical Model**

Let  

* \(I=\{1,\dots ,12\}\) – set of suppliers (S1–S12)  
* \(J=\{1,\dots ,12\}\) – set of customers (C1–C12)  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` column `fixed_costs` |
| \(d_j\) | Demand of customer \(j\) | `file_0_view_0` column `demand` |
| \(c_{ij}\) | Transportation cost per unit from supplier \(i\) to customer \(j\) | `file_2_view_0` columns `C1`–`C12` |
| \(M\) | Big‑M constant (e.g., \(M=\sum_{j\in J} d_j\)) | constant |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity supplied from supplier \(i\) to customer \(j\) (continuous)

**Objective**

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction** – each customer’s demand must be met exactly  

\[
\forall j\in J:\quad \sum_{i\in I} x_{ij}=d_j
\]

2. **Supplier activation** – if a supplier is closed, it supplies nothing  

\[
\forall i\in I,\;\forall j\in J:\quad x_{ij}\le M\,y_i
\]

3. **Non‑negativity**  

\[
\forall i\in I,\;\forall j\in J:\quad x_{ij}\ge 0
\]

4. **Binary supplier variables**  

\[
\forall i\in I:\quad y_i\in\{0,1\}
\]

---

### Data Mapping Rules

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `demand` |
| \(c_{ij}\) | `file_2_view_0` | `C1`–`C12` (column matching customer \(j\)) |
| \(M\) | constant | – |

*Index sets*  
- \(I\) is the set of row identifiers in `file_1_view_0` (values `S1`–`S12`).  
- \(J\) is the set of column identifiers in `file_2_view_0` excluding the row‑id column (`Unnamed: 0`), i.e., `C1`–`C12`.

All constraints and the objective use only these mapped parameters and the defined decision variables. No additional data or assumptions are introduced.