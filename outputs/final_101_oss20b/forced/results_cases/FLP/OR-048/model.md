**Mathematical Model**

Let  

* \(I\) – set of storage areas, indexed by the values in column **StorageID** of *file_0_view_0*.  
* \(J\) – set of air‑conditioner types, indexed by the values in column **ProductName** of *file_1_view_0*.  

Parameters  

* \(C_i\) – capacity of storage area \(i\) (units of space).  
  \(C_i =\) value of column **Capacity** in *file_0_view_0* for row \(i\).  
* \(V_j\) – monetary value of one unit of product \(j\).  
  \(V_j =\) value of column **Value** in *file_1_view_0* for row \(j\).  
* \(W_j\) – size (weight) of one unit of product \(j\).  
  \(W_j =\) value of column **Weight** in *file_1_view_0* for row \(j\).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\quad\text{for all }i\in I,\;j\in J
\]

*\(x_{ij}\)* = number of units of product \(j\) placed in storage area \(i\).

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J} V_j\,x_{ij}
\]

---

**Constraints**

1. **Capacity limits** – for each storage area \(i\):

\[
\sum_{j\in J} W_j\,x_{ij}\;\le\; C_i
\]

2. **Non‑negativity and integrality** – already encoded in the variable definition.

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C_i\)   | file_0_view_0 | Capacity |
| \(V_j\)   | file_1_view_0 | Value |
| \(W_j\)   | file_1_view_0 | Weight |

| Index Set | Source Table | Source Column |
|-----------|--------------|---------------|
| \(I\) | file_0_view_0 | StorageID |
| \(J\) | file_1_view_0 | ProductName |

All values are taken directly from the corresponding CSV files; no numerical enumeration is shown in the model.