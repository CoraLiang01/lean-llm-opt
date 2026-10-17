**Mathematical Model**

Let  

- \(P\) be the set of all drug products.  
- For each \(p\in P\)  
  - \(v_p\) = benefit per unit of product \(p\) (from *products.csv*).  
  - \(w_p\) = weight (stock space) per unit of product \(p\) (from *products.csv*).  
- \(C\) = total available stock capacity (from *capacity.csv*).  
- \(x_p\) = number of units of product \(p\) to order each day.

\[
\begin{aligned}
\max\;& \sum_{p\in P} v_p\,x_p \\[4pt]
\text{s.t. } & \sum_{p\in P} w_p\,x_p \;\le\; C \\[4pt]
& x_p \;\in\; \mathbb{Z}_{\ge 0}\quad \forall p\in P
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(C\) | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(v_p\) | `file_1_view_0` | `Value` | `v_p = Value` |
| \(w_p\) | `file_1_view_0` | `Weight` | `w_p = Weight` |
| \(P\) | `file_1_view_0` | `ProductName` | `P = {ProductName}` |

All values are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.