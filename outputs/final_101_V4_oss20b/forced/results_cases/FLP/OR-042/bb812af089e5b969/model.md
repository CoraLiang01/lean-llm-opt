**Mathematical Model**

Let  

- \(I\) be the set of drug types (indexed by the values of *ProductName*).  
- \(v_i\) be the benefit coefficient of product \(i\) (from *Value*).  
- \(w_i\) be the weight of one unit of product \(i\) (from *Weight*).  
- \(C\) be the overall inventory capacity (from *Capacity*).  

Decision variables  

- \(x_i \in \mathbb{Z}_{\ge 0}\) – number of units of product \(i\) ordered daily.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0} \quad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(C\) | `file_0_view_0` | `Capacity` |
| \(v_i\) | `file_1_view_0` | `Value` |
| \(w_i\) | `file_1_view_0` | `Weight` |
| \(I\) | `file_1_view_0` | `ProductName` |

No other data are required. The model uses only the information supplied in the two CSV files.