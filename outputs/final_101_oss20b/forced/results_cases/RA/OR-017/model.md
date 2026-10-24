**Mathematical Model**

Let  

- \(I\) be the set of all products whose SKU starts with “ZZ”.  
- For each \(i\in I\) let  

  \[
  \begin{aligned}
  &p_i &&\text{(revenue per unit)}\\
  &d_i &&\text{(demand)}\\
  &s_i &&\text{(initial inventory)}
  \end{aligned}
  \]

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} p_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\le d_i &&\forall i\in I \quad (\text{cannot sell more than demand})\\
x_i &\le s_i &&\forall i\in I \quad (\text{cannot sell more than inventory})
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) (index) | `file_0_view_0` | `SKU` |
| \(p_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

All values are taken directly from the rows returned by the query; no additional data are introduced.