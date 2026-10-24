**Mathematical Model**

Let  

* \(I\) be the set of real‑estate areas (rows of *products.csv*).  
* For each \(i\in I\)  
  * \(v_i\) – benefit (Value) of developing one unit in area \(i\).  
  * \(w_i\) – resource consumption (Weight) of one unit in area \(i\).  
* \(C\) – overall development capacity (Capacity from *capacity.csv*).  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} v_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} w_i\,x_i \;\le\; C
\]

---

**Data Mapping**

| Parameter | Source file | Column name |
|-----------|-------------|-------------|
| \(v_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv` | `Value` |
| \(w_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv` | `Weight` |
| \(C\)   | `Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv` | `Capacity` |

The index set \(I\) is derived from all rows of *products.csv*. No additional constraints are imposed beyond the overall capacity.