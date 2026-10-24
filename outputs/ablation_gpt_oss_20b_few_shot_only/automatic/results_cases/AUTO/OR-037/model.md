**Mathematical Model**

Let  

* \(I\) be the set of product types indexed by the values in the column **ProductName** of *products.csv*.  
* \(x_i\) – number of vehicles of type \(i\) ordered per day.  

Parameters  

* \(v_i\) – profit (value) of product \(i\) (column **Value** of *products.csv*).  
* \(w_i\) – weight/space requirement of product \(i\) (column **Weight** of *products.csv*).  
* \(C\) – total inventory capacity (single value in column **Capacity** of *capacity.csv*).  

The optimization problem is  

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0}\quad \forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column |
|-----------|-------------|--------|
| \(v_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv` | `Value` |
| \(w_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv` | `Weight` |
| \(C\)   | `Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv` | `Capacity` |
| Index set \(I\) | `Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv` | `ProductName` |

All values are taken directly from the corresponding CSV files; no additional data are introduced.