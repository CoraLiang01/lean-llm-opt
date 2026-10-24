**Mathematical Model**

Let  

- \(I\) be the set of bread types.  
- \(p_i\) be the expected profit per unit of bread type \(i\).  
- \(w_i\) be the storage weight per unit of bread type \(i\).  
- \(C\) be the total storage capacity.  
- \(x_i\) be the number of units of bread type \(i\) to order (integer, \(\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source File | Column |
|-----------|-------------|--------|
| \(C\) | `Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv` | `Capacity` |
| \(I\) | `Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv` | `ProductName` |
| \(p_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv` | `Value` |
| \(w_i\) | `Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv` | `Weight` |

All parameters are taken directly from the specified CSV columns; no additional data are introduced.