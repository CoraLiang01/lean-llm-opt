**Mathematical Model**

Let  

- \(I\) be the set of products.  
- For each \(i\in I\):  
  - \(w_i\) = weight of one unit of product \(i\) (kg).  
  - \(v_i\) = value (benefit) of one unit of product \(i\).  
- \(C\) = total stock‑capacity (kg).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) = number of units of product \(i\) to order each day.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t. } & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(C\) | Overall capacity | `file_0_view_0` | `Capacity` |
| \(w_i\) | Product weight | `file_1_view_0` | `Weight` |
| \(v_i\) | Product value | `file_1_view_0` | `Value` |
| \(I\) | Product list | `file_1_view_0` | `ProductName` |

*No numerical values are listed; each parameter is mapped directly to the corresponding column in the specified source table.*