**Mathematical Model**

Let  

- \(I\) : set of product types (indexed by the values of *ProductName*).  
- \(v_i\) : benefit coefficient of product \(i\).  
- \(w_i\) : weight (inventory consumption) of product \(i\).  
- \(C\) : total inventory capacity.  
- \(x_i\) : number of units of product \(i\) ordered daily.

\[
\begin{aligned}
\max \quad & \sum_{i\in I} v_i\,x_i \\[4pt]
\text{s.t.}\quad & \sum_{i\in I} w_i\,x_i \;\le\; C \\[4pt]
& x_i \in \mathbb{Z}_{\ge 0} \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) | Mapping |
|-----------|--------|----------|-----------|---------|
| \(C\) | Capacity | `file_0_view_0` | `Capacity` | `C = Capacity` |
| \(I\) | Product names | `file_1_view_0` | `ProductName` | `I = {ProductName}` |
| \(v_i\) | Benefit | `file_1_view_0` | `Value` | `v_i = Value` |
| \(w_i\) | Weight | `file_1_view_0` | `Weight` | `w_i = Weight` |

All values are taken directly from the corresponding CSV records; no numerical enumeration is provided in the model.