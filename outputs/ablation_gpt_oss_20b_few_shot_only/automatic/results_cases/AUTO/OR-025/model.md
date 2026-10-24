**Mathematical Model**

Let  

\[
T=\{\,i \mid \text{Product Name}_i \text{ starts with } \texttt{TABLET\_}\}
\]

For each \(i\in T\) define

* \(rev_i\) – revenue per unit (column **Revenue**)
* \(inv_i\) – initial inventory (column **Initial Inventory**)
* \(dem_i\) – demand (column **Demand**)

Decision variable  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad (i\in T)
\]

---

**Objective**

\[
\max \;\sum_{i\in T} rev_i \, x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\le inv_i &&\forall i\in T \\
x_i &\le dem_i &&\forall i\in T
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column | Row Selection |
|-----------|-------------|--------|---------------|
| \(rev_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv` | `Revenue` | `Product Name = i` |
| \(inv_i\) | same file | `Initial Inventory` | `Product Name = i` |
| \(dem_i\) | same file | `Demand` | `Product Name = i` |
| Index set \(T\) | same file | `Product Name` | values beginning with `TABLET_` |

All other columns in the file are ignored.