**Mathematical Model**

Let  

* \(I\) be the set of display areas indexed by the identifier **DisplayID**.  
* \(J\) be the set of boat types indexed by the identifier **ProductName**.  

Parameters (all taken directly from the supplied tables)

| Symbol | Source table | Column name | Description |
|--------|--------------|-------------|-------------|
| \(C_i\) | `file_0_view_0` | `Capacity` | Capacity of display area \(i\) |
| \(v_j\) | `file_1_view_0` | `Value` | Monetary value of boat type \(j\) |
| \(w_j\) | `file_1_view_0` | `Weight` | Size (weight) of boat type \(j\) |

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

### Objective
\[
\max \;\; \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}
\]

### Constraints
1. **Capacity of each display area**

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; C_i \qquad \forall i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(C_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are implicitly defined by the distinct values of **DisplayID** and **ProductName** in the respective tables. No additional data or constraints are introduced beyond those specified.