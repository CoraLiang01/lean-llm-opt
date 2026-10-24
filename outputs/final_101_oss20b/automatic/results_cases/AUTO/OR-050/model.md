**Mathematical Model**

Let  

* \(I=\{\,\text{ShelfID}\,\}\) – index set of shelves (from *file_0_view_0*).  
* \(J=\{\,\text{ProductName}\,\}\) – index set of products (from *file_1_view_0*).  
* \(j^{\ast}\) – the first product in the order of *file_1_view_0* (i.e. “Smartphone”).  

Parameters  

* \(\text{Capacity}_i\) – capacity of shelf \(i\) (column **Capacity** in *file_0_view_0*).  
* \(\text{Value}_j\) – monetary value of product \(j\) (column **Value** in *file_1_view_0*).  
* \(\text{Weight}_j\) – weight of product \(j\) (column **Weight** in *file_1_view_0*).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\sum_{j\in J}\text{Value}_j\,x_{ij}
\]

---

**Constraints**

1. **Shelf‑capacity (weight) constraints**

\[
\sum_{j\in J}\text{Weight}_j\,x_{ij}\;\le\;\text{Capacity}_i
\qquad \forall i\in I
\]

2. **Minimum quantity of the first product**

\[
\sum_{i\in I}x_{i\,j^{\ast}}\;\ge\;5
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | *file_0_view_0* | ShelfID |
| \(\text{Capacity}_i\) | *file_0_view_0* | Capacity |
| \(J\) | *file_1_view_0* | ProductName |
| \(\text{Value}_j\) | *file_1_view_0* | Value |
| \(\text{Weight}_j\) | *file_1_view_0* | Weight |
| \(j^{\ast}\) | *file_1_view_0* | first row (ProductName = “Smartphone”) |

All parameters are taken directly from the corresponding CSV files; no numerical values are hard‑coded in the model.