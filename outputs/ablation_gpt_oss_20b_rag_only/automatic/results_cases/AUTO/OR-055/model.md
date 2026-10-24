**Mathematical Model**

Let  

- \(I\) = set of display areas (indexed by `DisplayID` from *capacity.csv*).  
- \(J\) = set of boat types (indexed by `ProductName` from *products.csv*).  

Parameters  

- \(C_i\) – capacity of display area \(i\) (units of space).  
- \(v_j\) – monetary value of one unit of boat type \(j\).  
- \(w_j\) – size (weight) of one unit of boat type \(j\).  

Decision variables  

- \(x_{ij}\) – number of units of boat type \(j\) placed in display area \(i\).  
  \[
  x_{ij}\in \mathbb{Z}_{\ge 0}\quad \forall i\in I,\; j\in J
  \]

Objective  

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

Capacity constraints (one per display area)

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; C_i \qquad \forall i\in I
\]

No other constraints are required.

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(C_i\)   | file_0_view_0 | Capacity |
| \(v_j\)   | file_1_view_0 | Value |
| \(w_j\)   | file_1_view_0 | Weight |

All tables are used in the order supplied; no sorting or re‑indexing is performed. The model respects the integer domain of the decision variables and the capacity limits exactly as provided in the source data.