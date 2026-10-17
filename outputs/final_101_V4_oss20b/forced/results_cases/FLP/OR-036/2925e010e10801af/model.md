**Mathematical Model**

Let  

- \(I\) be the set of vehicle types.  
- \(v_i\) be the benefit coefficient (value) of vehicle type \(i\).  
- \(w_i\) be the weight (inventory consumption) of vehicle type \(i\).  
- \(C\) be the total inventory capacity.  
- \(x_i\) be the number of units of vehicle type \(i\) ordered daily.

\[
\begin{aligned}
\max\;& \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\;& \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |

The index set \(I\) is the set of distinct `ProductName` values in `file_1_view_0`. No other data are required.