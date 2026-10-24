**Mathematical Model**

Let  

- \(I\) be the set of drug types (indexed by the values in the column *ProductName*).  
- \(v_i\) be the benefit coefficient of drug \(i\) (column *Value*).  
- \(w_i\) be the weight of one unit of drug \(i\) (column *Weight*).  
- \(C\) be the overall inventory capacity (column *Capacity*).  
- \(x_i\) be the daily order quantity of drug \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column Name |
|-----------|-------------|-------------|
| \(I\) | `Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv` | `ProductName` |
| \(v_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv` | `Value` |
| \(w_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv` | `Weight` |
| \(C\) | `Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv` | `Capacity` |

All parameters are taken directly from the specified CSV files; no values are enumerated or altered.