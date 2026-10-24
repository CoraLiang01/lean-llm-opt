**Mathematical Model**

Let  

- \(I\) be the set of pizza products (the distinct values in the *Product Name* column).  
- For each \(i\in I\)  
  - \(p_i\) = revenue per unit (from column *Revenue*).  
  - \(d_i\) = demand (from column *Demand*).  
  - \(s_i\) = initial inventory (from column *Initial Inventory*).  

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} p_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I \quad &&\text{(demand limit)}\\
x_i &\le s_i &&\forall i\in I \quad &&\text{(inventory limit)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | File Path | Column Name |
|-----------|-----------|-------------|
| \(p_i\) (Revenue) | `Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv` | `Revenue` |
| \(d_i\) (Demand) | `Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv` | `Demand` |
| \(s_i\) (Initial Inventory) | `Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv` | `Initial Inventory` |
| Index set \(I\) | `Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv` | `Product Name` |

All parameters are taken directly from the specified columns; no additional data are introduced.