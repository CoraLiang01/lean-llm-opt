**Mathematical Model**

Let  

- \(I\) be the set of all area names (rows of *products.csv*).  
- \(b_i\) be the benefit coefficient for area \(i\).  
- \(C\) be the overall development capacity (value in *capacity.csv*).  
- \(x_i\) be the daily scale of development in area \(i\) (integer, \(\ge 0\)).

\[
\begin{aligned}
\max\;& \sum_{i\in I} b_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{+}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(b_i\) | `file_1_view_0` | `ProductName`, `Value` | `b_{ProductName} = Value` |

No other data are required. The index set \(I\) is implicitly defined by the distinct `ProductName` values in `file_1_view_0`.