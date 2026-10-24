**Mathematical Model**

Let  

- \(I=\{A1,\dots ,A15\}\) (Factory sites)  
- \(J=\{B1,\dots ,B8\}\) (Distribution centers)

**Parameters**  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(f_i\) | Fixed construction cost of facility \(i\) | `file_0_view_0` – column `FixedCost` |
| \(cap_i\) | Production capacity of facility \(i\) | `file_0_view_0` – column `Capacity` |
| \(c_{ij}\) | Variable shipping cost per unit from facility \(i\) to destination \(j\) | `file_1_view_0` – matrix entry \((i,j)\) |
| \(d_j\) | Demand at destination \(j\) | `file_2_view_0` – column `Demand` |

**Decision Variables**  

- \(y_i \in \{0,1\}\) 1 if facility \(i\) is built, 0 otherwise  
- \(x_{ij}\ge 0\) quantity shipped from facility \(i\) to destination \(j\)

**Objective**  

\[
\min \;\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

**Constraints**

1. **Capacity (facility‑dependent)**  
   \[
   \sum_{j\in J} x_{ij} \;\le\; cap_i\,y_i \quad \forall i\in I
   \]

2. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \quad \forall j\in J
   \]

3. **Variable bounds**  
   \[
   x_{ij}\ge 0 \quad \forall i\in I,\; j\in J
   \]
   \[
   y_i\in\{0,1\}\quad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column(s) | Notes |
|-----------|----------|-----------|-------|
| \(f_i\) | `file_0_view_0` | `FixedCost` | Indexed by `Facility` |
| \(cap_i\) | `file_0_view_0` | `Capacity` | Indexed by `Facility` |
| \(c_{ij}\) | `file_1_view_0` | Matrix columns `B1`–`B8` | Indexed by `Origin` (facility) and destination columns |
| \(d_j\) | `file_2_view_0` | `Demand` | Indexed by `Destination` |

All indices and parameters are taken directly from the supplied CSV files; no additional data are introduced.