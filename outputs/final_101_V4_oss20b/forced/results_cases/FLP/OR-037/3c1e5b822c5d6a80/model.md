**Mathematical Model**

Let  

- \(I\) be the set of product types (car models).  
- \(x_i\) – number of vehicles of type \(i\in I\) ordered per day.  

Parameters  

- \(P_i\) – profit per vehicle of type \(i\).  
- \(W_i\) – weight (inventory space) per vehicle of type \(i\).  
- \(C\) – total available inventory capacity.  

\[
\begin{aligned}
\max\;& \sum_{i\in I} P_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} W_i\,x_i \;\le\; C \\[4pt]
& x_i \;\ge\; 0,\; x_i \in \mathbb{Z}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(P_i\) | `file_1_view_0` | `Value` |
| \(W_i\) | `file_1_view_0` | `Weight` |

The index set \(I\) is the set of all `ProductName` values in `file_1_view_0`.