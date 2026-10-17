**Mathematical model**

Let  

* \(I=\{1,\dots ,14\}\) – set of display areas (indices are the values of **DisplayID**).  
* \(J=\{\text{Speedboat},\dots ,\text{Skiff}\}\) – set of boat types (indices are the values of **ProductName**).  

Parameters (all taken from the supplied tables)

* \(c_i\) – capacity of display area \(i\)  (\(c_i\) from **Capacity** in *file_0_view_0*).  
* \(v_j\) – monetary value of one unit of boat type \(j\) (\(v_j\) from **Value** in *file_1_view_0*).  
* \(w_j\) – size (weight) of one unit of boat type \(j\) (\(w_j\) from **Weight** in *file_1_view_0*).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

*\(x_{ij}\)* = number of units of boat type \(j\) placed in display area \(i\).

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

**Constraints**

1. **Capacity constraints** – the total size of boats in each display area cannot exceed its capacity  

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I
\]

2. **Non‑negativity and integrality** – already encoded in the variable definition.  

---

### Data‑mapping table

| Parameter | Source table | Column name |
|-----------|--------------|-------------|
| \(c_i\)   | file_0_view_0 | Capacity |
| \(v_j\)   | file_1_view_0 | Value |
| \(w_j\)   | file_1_view_0 | Weight |

All indices \(i\) and \(j\) are taken from the complete set of rows returned in the observation. No additional data or constraints are introduced.