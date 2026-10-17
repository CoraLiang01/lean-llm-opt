**Mathematical Model**

Let  

* \(I\) – set of cabinets (indices from *file_0_view_0*).  
* \(J\) – set of coffee products (indices from *file_1_view_0*).  

Parameters  

* \(\text{Capacity}[i]\) – weight capacity of cabinet \(i\) (column **Capacity** in *file_0_view_0*).  
* \(\text{Value}[j]\) – monetary value of one unit of product \(j\) (column **Value** in *file_1_view_0*).  
* \(\text{Weight}[j]\) – weight of one unit of product \(j\) (column **Weight** in *file_1_view_0*).  

Decision variables  

* \(x_{ij}\) – number of units of product \(j\) placed in cabinet \(i\).  
  \[
  x_{ij}\in \mathbb{Z}_{\ge 0}\quad \forall i\in I,\; j\in J
  \]

**Objective**

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\; \text{Value}[j]\; x_{ij}
\]

**Constraints**

1. **Cabinet capacity (weight) limits**

\[
\sum_{j\in J}\; \text{Weight}[j]\; x_{ij}\;\le\; \text{Capacity}[i]
\quad \forall i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\quad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | *file_0_view_0* | **CabinetID** |
| \(J\) | *file_1_view_0* | **ProductName** |
| \(\text{Capacity}[i]\) | *file_0_view_0* | **Capacity** |
| \(\text{Value}[j]\) | *file_1_view_0* | **Value** |
| \(\text{Weight}[j]\) | *file_1_view_0* | **Weight** |

All parameters are taken directly from the corresponding columns of the supplied CSV views; no additional data are introduced.