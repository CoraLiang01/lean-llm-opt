**Symbolic Model**

Let  

* \(I=\{S1,S2,\dots ,S8\}\) ( suppliers )  
* \(J=\{C1,C2,\dots ,C9\}\) ( dealerships )

Parameters  

* \(f_i\) : fixed cost of opening supplier \(i\)  \(i\in I\)  
* \(c_{ij}\) : transportation cost per vehicle from supplier \(i\) to dealership \(j\) \(i\in I,\;j\in J\)  
* \(d_j\) : demand (vehicles) of dealership \(j\) \(j\in J\)

Decision variables  

* \(y_i \in \{0,1\}\) : 1 if supplier \(i\) is opened, 0 otherwise \(i\in I\)  
* \(x_{ij}\ge 0\) : number of vehicles supplied by \(i\) to \(j\) \(i\in I,\;j\in J\)

---

### Objective  

\[
\min \;\; \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij}
\]

### Constraints  

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J
   \]

2. **Supplier activation** (big‑\(M\) linking)  
   \[
   x_{ij} \;\le\; M_j\,y_i \qquad \forall i\in I,\;j\in J
   \]
   where \(M_j\) is a sufficiently large constant (e.g., \(M_j=d_j\)).

3. **Variable domains**  
   \[
   y_i \in \{0,1\}\quad \forall i\in I,\qquad
   x_{ij}\ge 0\quad \forall i\in I,\;j\in J
   \]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Index Mapping |
|-----------|--------------|---------------|---------------|
| \(f_i\)   | `file_1_view_0` | `fixed_costs` | row id `Unnamed: 0` → supplier \(i\) |
| \(c_{ij}\)| `file_2_view_0` | `C1`–`C9` (columns) | row id `Unnamed: 0` → supplier \(i\); column id `Cj` → dealership \(j\) |
| \(d_j\)   | `file_0_view_0` | `demand` | row id `customer` → dealership \(j\) |

All indices \(i\) and \(j\) are taken from the complete set of suppliers and dealerships present in the respective source tables. No numerical values are enumerated in the model; they are bound to the data via the mapping above.