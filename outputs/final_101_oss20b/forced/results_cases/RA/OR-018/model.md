**Mathematical Model**

Let  

* \(I\) be the set of Baby products (rows of the view `file_0_view_0`).  
* For each \(i\in I\) let  

\[
\begin{aligned}
\text{Revenue}_i &\;=\; \text{file\_0\_view\_0.Revenue} \\
\text{Demand}_i   &\;=\; \text{file\_0\_view\_0.Demand} \\
\text{Inv}_i      &\;=\; \text{file\_0\_view\_0.Initial Inventory}
\end{aligned}
\]

Decision variable  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Objective**

\[
\max \;\sum_{i\in I} \text{Revenue}_i \; x_i
\]

---

**Constraints**

1. **Demand limitation**  
   \[
   x_i \;\le\; \text{Demand}_i \qquad \forall i\in I
   \]

2. **Initial‑inventory limitation**  
   \[
   x_i \;\le\; \text{Inv}_i \qquad \forall i\in I
   \]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | *implicit* (rows where `Product Name` starts with “Baby”) |
| \(\text{Revenue}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{Demand}_i\) | `file_0_view_0` | `Demand` |
| \(\text{Inv}_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the returned rows of the view; no additional data are introduced. The model respects the integer, non‑negative nature of the decision variables and the two natural upper bounds (demand and initial inventory).