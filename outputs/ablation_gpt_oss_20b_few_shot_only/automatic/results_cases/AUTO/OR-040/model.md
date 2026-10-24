**Mathematical Model**

Let  

* \(I\) be the set of area names (rows of *products.csv*).  
* \(x_i\) be the daily development scale in area \(i\).

Parameters  

* \(\text{Value}_i\) – benefit coefficient for area \(i\) (column *Value* in *products.csv*).  
* \(\text{Cap}\) – overall development capacity (column *Capacity* in *capacity.csv*).

\[
\begin{aligned}
\max\;& \sum_{i\in I} \text{Value}_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} x_i \;\le\; \text{Cap} \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column |
|-----------|-------------|--------|
| \(\text{Value}_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv` | `Value` |
| \(\text{Cap}\) | `Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv` | `Capacity` |

*Index set \(I\) is derived from the `ProductName` column of *products.csv*.*