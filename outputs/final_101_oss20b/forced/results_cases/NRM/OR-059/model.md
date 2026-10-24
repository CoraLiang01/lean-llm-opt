**Mathematical Model (symbolic)**  

Let  

* \(S=\{S_1,\dots ,S_8\}\) – set of suppliers  
* \(C=\{C_1,\dots ,C_9\}\) – set of dealerships  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0` – column **fixed_costs** |
| \(d_j\) | Demand of dealership \(j\) | `file_0_view_0` – column **demand** |
| \(t_{ij}\) | Transportation cost per vehicle from supplier \(i\) to dealership \(j\) | `file_2_view_0` – columns **C1**–**C4** (mapped to \(j\)) |
| \(M\) | Big‑\(M\) bound (e.g. \(\sum_{j\in C} d_j\)) | Derived from data |

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij} \ge 0\) – quantity of vehicles supplied from \(i\) to \(j\)

---

**Objective**  

\[
\min \;\; \sum_{i\in S} f_i\,y_i \;+\; \sum_{i\in S}\sum_{j\in C} t_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in S} x_{ij} \;\ge\; d_j , \qquad \forall j\in C
   \]

2. **Supplier capacity linkage**  
   \[
   x_{ij} \;\le\; M\,y_i , \qquad \forall i\in S,\; \forall j\in C
   \]

3. **Variable domains**  
   \[
   y_i \in \{0,1\}, \quad x_{ij}\ge 0 , \qquad \forall i\in S,\; \forall j\in C
   \]

---

**Data Mapping**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | Supplier identifier in `Unnamed: 0` |
| \(d_j\) | `file_0_view_0` | `demand` | Dealership identifier in `customer` |
| \(t_{ij}\) | `file_2_view_0` | `C1`–`C4` | Supplier identifier in `Unnamed: 0`; each column maps to a dealership \(j\) |
| \(M\) | – | – | Computed as \(\sum_{j} d_j\) from `file_0_view_0` |

All indices, parameters, and variables are defined exactly once, and no numerical values are embedded in the symbolic model.