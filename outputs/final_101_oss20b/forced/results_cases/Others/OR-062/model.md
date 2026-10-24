**Mathematical Model**

Let  

* \(I=\{\text{MOUNT AYR},\text{WAUKEE},\text{WAVERLY},\text{PELLA},\text{DES MOINES}\}\) – suppliers  
* \(J=\{\text{CLARINDA},\text{FORT MADISON},\text{SIOUX CITY},\text{TOLEDO},\text{BANCROFT}\}\) – stores  

Parameters  

* \(f_i\) – fixed cost of opening supplier \(i\)  
  * source: **file_1_view_0**, column **fixed_costs**  
* \(c_{ij}\) – transportation cost per unit from supplier \(i\) to store \(j\)  
  * source: **file_2_view_0**, columns **CLARINDA**, **FORT MADISON**, **SIOUX CITY**, **TOLEDO**, **BANCROFT**  
* \(d_j\) – demand of store \(j\) (units)  
  * source: **file_0_view_0**, column **demand**  
  * mapping: row order in **file_0_view_0** corresponds to column order in **file_2_view_0**  
* \(M\) – a sufficiently large constant (e.g., \(M=\sum_{j\in J} d_j\))

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to store \(j\) (continuous)

Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Supplier activation**  
   \[
   x_{ij} \;\le\; M\,y_i \qquad \forall i\in I,\; j\in J
   \]

3. **Non‑negativity**  
   \[
   x_{ij}\;\ge\;0 \qquad \forall i\in I,\; j\in J
   \]

4. **Binary activation**  
   \[
   y_i \in \{0,1\} \qquad \forall i\in I
   \]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping Notes |
|-----------|--------------|----------------|---------------|
| \(f_i\) | **file_1_view_0** | `fixed_costs` | Row index corresponds to supplier name in `Unnamed: 0` |
| \(c_{ij}\) | **file_2_view_0** | `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` | Row index corresponds to supplier name in `Unnamed: 0`; column names are store identifiers |
| \(d_j\) | **file_0_view_0** | `demand` | Row order matches column order of transportation costs; map `Customer_1` → `CLARINDA`, `Customer_2` → `FORT MADISON`, etc. |
| \(M\) | Derived | – | \(M = \sum_{j\in J} d_j\) (or any large upper bound) |

All indices, parameters, and variables are defined exactly as above; no additional entities or constraints are introduced beyond those required by the problem statement.