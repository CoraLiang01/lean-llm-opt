**Mathematical Model**

Let  

- \(I\) be the set of book products (rows returned by the filter).  
- For each \(i\in I\)  

\[
\begin{aligned}
\text{rev}_i &\;=\; \text{Revenue of product }i,\\
\text{dem}_i &\;=\; \text{Demand of product }i,\\
\text{inv}_i &\;=\; \text{Initial Inventory of product }i.
\end{aligned}
\]

Decision variable  

\[
x_i \;=\; \text{number of units of product }i\text{ to be fulfilled},\qquad x_i\in\mathbb{Z}_{\ge 0}.
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\text{rev}_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\le \text{dem}_i &&\forall i\in I &&\text{(cannot exceed demand)}\\[4pt]
x_i &\le \text{inv}_i &&\forall i\in I &&\text{(cannot exceed available inventory)}\\[4pt]
x_i &\ge 0 &&\forall i\in I &&\text{(non‑negativity, integer)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column Name |
|----------------------|--------------|-------------|
| `rev_i`              | `file_0_view_0` | `Revenue` |
| `dem_i`              | `file_0_view_0` | `Demand` |
| `inv_i`              | `file_0_view_0` | `Initial Inventory` |
| `x_i`                | — (decision variable) | — |

All indices and parameters are taken directly from the rows returned by the query; no additional data are introduced.