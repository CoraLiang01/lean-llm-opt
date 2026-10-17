**Mathematical Model**

Let  

* \(S\) – set of suppliers (indices \(i\))  
* \(C\) – set of customers (indices \(j\))

| Symbol | Meaning | Domain |
|--------|---------|--------|
| \(y_i\) | 1 if supplier \(i\) is opened, 0 otherwise | \(\{0,1\}\) |
| \(x_{ij}\) | quantity of goods shipped from supplier \(i\) to customer \(j\) | \(\mathbb{R}_{+}\) |

Parameters (all obtained from the supplied CSV files)

| Symbol | Source table | Column name | Description |
|--------|--------------|-------------|-------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Fixed opening cost of supplier \(i\) |
| \(d_j\) | `file_0_view_0` | `demand` | Demand of customer \(j\) |
| \(t_{ij}\) | `file_2_view_0` | `Cj` (column for customer \(j\)) | Transportation cost per unit from supplier \(i\) to customer \(j\) |

---

### Objective
\[
\min \;\; \sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij}
\]

### Constraints

1. **Demand satisfaction**  
   \[
   \sum_{i\in S} x_{ij} \;=\; d_j , \qquad \forall j\in C
   \]

2. **Supplier capacity linkage** (large‑\(M\) formulation)  
   \[
   x_{ij} \;\le\; d_j\,y_i , \qquad \forall i\in S,\; \forall j\in C
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 , \qquad \forall i\in S,\; \forall j\in C
   \]

4. **Binary opening decision**  
   \[
   y_i \in \{0,1\}, \qquad \forall i\in S
   \]

---

**Data Mapping**

| Index | Table ID | Column(s) |
|-------|----------|-----------|
| \(i\) | `file_1_view_0` | `Unnamed: 0` (supplier identifier) |
| \(j\) | `file_0_view_0` | `customer` (customer identifier) |
| \(f_i\) | `file_1_view_0` | `fixed_costs` |
| \(d_j\) | `file_0_view_0` | `demand` |
| \(t_{ij}\) | `file_2_view_0` | column named after customer \(j\) (e.g., `C1`, `C2`, …) |

All parameters are taken directly from the corresponding CSV files; no numerical values are listed in the model.