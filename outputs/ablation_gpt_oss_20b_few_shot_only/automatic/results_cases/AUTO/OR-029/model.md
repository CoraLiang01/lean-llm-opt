**Mathematical Model**

Let  

* \(I=\{i\mid\text{Product Name}_i\text{ contains the string “FAUX”}\}\) – index set of all FAUX products.  

Parameters (obtained from the CSV file)  

* \(\text{Revenue}_i\) – revenue per unit of product \(i\) (column **Revenue**).  
* \(\text{Demand}_i\) – deterministic total demand for product \(i\) (column **Demand**).  
* \(\text{Inv}_i\) – initial inventory of product \(i\) (column **Initial Inventory**).  

Decision variables  

* \(x_i\in\mathbb{Z}_{+}\) – number of units of product \(i\) to fulfill.  

Objective  

\[
\max \;\sum_{i\in I}\text{Revenue}_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\le \text{Demand}_i &&\forall i\in I \quad\text{(demand limit)}\\[4pt]
x_i &\le \text{Inv}_i     &&\forall i\in I \quad\text{(inventory limit)}\\[4pt]
x_i &\ge 0                &&\forall i\in I \quad\text{(non‑negativity)}
\end{aligned}
\]

All variables are required to be integers (positive or zero).

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| \(\text{Revenue}_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv` | `Revenue` |
| \(\text{Demand}_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv` | `Demand` |
| \(\text{Inv}_i\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv` | `Initial Inventory` |
| Index set \(I\) | Same file | `Product Name` (filter for entries containing “FAUX”) |

No additional data or constraints are introduced beyond those specified in the query.