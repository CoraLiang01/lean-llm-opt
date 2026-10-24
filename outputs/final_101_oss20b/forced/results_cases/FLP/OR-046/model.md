**Mathematical Model**

Let  

- \(I=\{i\mid i\text{ is a product name in } \text{products.csv}\}\)  
- \(x_i\ge 0\) : number of units of product \(i\) ordered each day  

Parameters (to be filled from the CSV files)

- \(v_i\) : value (benefit) per unit of product \(i\) (from column **Value** in *products.csv*)  
- \(w_i\) : weight (space occupied) per unit of product \(i\) (from column **Weight** in *products.csv*)  
- \(C\) : total available stock capacity (from column **Capacity** in *capacity.csv*)

The optimization problem is

\[
\begin{aligned}
\max_{x}\quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All values are taken directly from the corresponding CSV records; no numerical values are enumerated in the model.