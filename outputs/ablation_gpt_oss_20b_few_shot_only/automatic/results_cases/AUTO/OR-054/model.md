**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) – set of shelves (ShelfID).  
* \(J=\{1,\dots ,20\}\) – set of products (ProductName).  

Parameters  

* \(c_i\) – capacity of shelf \(i\) (units of space).  
* \(w_j\) – weight (space requirement) of product \(j\).  
* \(v_j\) – value of product \(j\).  

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Objective**

\[
\max \;\; \sum_{i\in I}\;\sum_{j\in J} v_j\,x_{ij}
\]

---

**Constraints**

1. **Shelf capacity**

\[
\sum_{j\in J} w_j\,x_{ij}\;\le\; c_i \qquad \forall i\in I
\]

2. **Non‑negativity & integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source File | Column Name |
|-----------|-------------|-------------|
| \(c_i\)   | `Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv` | `Capacity` |
| \(w_j\)   | `Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv` | `Weight` |
| \(v_j\)   | `Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv` | `Value` |

The index sets \(I\) and \(J\) are derived from the `ShelfID` and `ProductName` columns of the respective CSV files. No additional constraints or parameters are introduced beyond those explicitly required by the query.