**Mathematical Model**

Let  

- \(I\) be the set of product types (rows of *products.csv*).  
- \(p_i\) be the profit per unit of product \(i\) (column **Value**).  
- \(w_i\) be the weight (inventory space) per unit of product \(i\) (column **Weight**).  
- \(C\) be the total inventory capacity (column **Capacity**).  
- \(x_i\) be the number of units of product \(i\) ordered per day.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} p_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \quad \forall i\in I \\[4pt]
& x_i \;\in\; \mathbb{Z} \quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(p_i\) | Profit | `file_1_view_0` | `Value` | `p_i = Value` |
| \(w_i\) | Weight | `file_1_view_0` | `Weight` | `w_i = Weight` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` | `I = {ProductName}` |

All indices, parameters, and variables are defined solely from the data returned in the observation. No numerical values are enumerated in the model.