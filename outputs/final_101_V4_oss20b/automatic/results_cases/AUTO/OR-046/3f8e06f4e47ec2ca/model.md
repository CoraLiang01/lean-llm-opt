**Mathematical Model**

Let  

* \(I\) be the set of products indexed by the values in the column **ProductName** of *file_1_view_0*.  
* \(x_i\) – number of units of product \(i\) ordered each day.  

Parameters  

* \(v_i\) – value (benefit) of one unit of product \(i\).  
* \(w_i\) – weight (space requirement) of one unit of product \(i\).  
* \(C\) – total available stock capacity.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| Index set \(I\) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding CSV records; no additional data are introduced.