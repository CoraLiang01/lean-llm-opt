**Mathematical Model**

\[
\begin{aligned}
\min\;& Z = \sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t.}\;& \sum_{i\in I} x_{ij} \;\ge\; d_j \qquad &&\forall j\in J \\[4pt]
& x_{ij} \;\le\; d_j\,y_i \qquad &&\forall i\in I,\; j\in J \\[4pt]
& y_i \in \{0,1\} \qquad &&\forall i\in I \\[2pt]
& x_{ij} \;\ge\; 0 \qquad &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

### Index Sets
- \(I = \{S1,S2,S3,S4,S5,S6,S7,S8\}\) (suppliers)  
- \(J = \{C1,C2,C3,C4,C5,C6,C7,C8,C9\}\) (dealerships)

---

### Parameters
| Symbol | Meaning | Source table & column |
|--------|---------|-----------------------|
| \(d_j\) | Demand of dealership \(j\) | `file_0_view_0`, column **demand** |
| \(f_i\) | Fixed opening cost of supplier \(i\) | `file_1_view_0`, column **fixed_costs** |
| \(c_{ij}\) | Transportation cost per vehicle from supplier \(i\) to dealership \(j\) | `file_2_view_0`, columns **C1**–**C9** |

---

### Variables
| Symbol | Domain | Interpretation |
|--------|--------|----------------|
| \(y_i\) | \(\{0,1\}\) | 1 if supplier \(i\) is opened, 0 otherwise |
| \(x_{ij}\) | \(\mathbb{R}_+\) | Quantity of vehicles supplied by \(i\) to dealership \(j\) |

---

### Data‑Mapping Rules
1. **Demand**  
   - Table ID: `file_0_view_0`  
   - Column: `demand`  
   - Row identifier: `customer` (values `C1`–`C9`)  
   - Map each row to \(d_j\).

2. **Fixed Costs**  
   - Table ID: `file_1_view_0`  
   - Column: `fixed_costs`  
   - Row identifier: `Unnamed: 0` (values `S1`–`S8`)  
   - Map each row to \(f_i\).

3. **Transportation Costs**  
   - Table ID: `file_2_view_0`  
   - Row identifier: `Unnamed: 0` (values `S1`–`S8`)  
   - Columns `C1`–`C9` map to \(c_{ij}\) for each supplier‑dealership pair.

All parameters are taken directly from the corresponding source tables; no additional numerical processing or enumeration of values is performed in the model.