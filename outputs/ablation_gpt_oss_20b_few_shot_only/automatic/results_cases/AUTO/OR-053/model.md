**Mathematical Model**

Let  

* \(I\) – set of shelves, indexed by the values in column **ShelfID** of *capacity.csv*.  
* \(J\) – set of products, indexed by the values in column **ProductName** of *products.csv*.  

Parameters  

\[
\begin{aligned}
\text{Cap}_i &\quad\text{(capacity of shelf }i\text{)}\\
\text{Val}_j &\quad\text{(value of product }j\text{)}\\
\text{Wt}_j  &\quad\text{(weight of product }j\text{)}
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

### Objective  

\[
\max \;\sum_{i\in I}\;\sum_{j\in J}\; \text{Val}_j \; x_{ij}
\]

### Constraints  

1. **Shelf‑capacity (weight) constraints**  

\[
\sum_{j\in J}\; \text{Wt}_j \; x_{ij}\;\le\; \text{Cap}_i
\qquad \forall i\in I
\]

2. **Non‑negativity and integrality**  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

## Data Mapping

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(\text{Cap}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv` | `Capacity` |
| \(\text{Val}_j\) | `Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv` | `Value` |
| \(\text{Wt}_j\)  | `Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv` | `Weight` |

*Index sets*  
- \(I\) is derived from the `ShelfID` column of *capacity.csv*.  
- \(J\) is derived from the `ProductName` column of *products.csv*.

All numerical values are taken directly from the corresponding CSV files; no additional data are introduced.