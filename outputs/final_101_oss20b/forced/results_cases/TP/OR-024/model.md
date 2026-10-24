**Mathematical model**

Let  

* \(I\) – set of products whose *Product Name* starts with the prefix **S700_**.  
* For each \(i\in I\)  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue per unit of product }i,\\
d_i &\;=\;\text{Demand quantity of product }i,\\
s_i &\;=\;\text{Initial inventory of product }i.
\end{aligned}
\]

Decision variable  

\[
x_i \;\ge\;0 \qquad\text{(units of product }i\text{ to sell)}
\]

---

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; s_i &&\forall i\in I &&\text{(cannot sell more than inventory)}\\[4pt]
x_i &\;\ge\; 0 &&\forall i\in I
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Column name | Mapping |
|-----------|--------------|-------------|---------|
| \(I\) | `file_0_view_0` | `Product Name` | index set (rows where `Product Name` starts with `S700_`) |
| \(r_i\) | `file_0_view_0` | `Revenue` | revenue per unit |
| \(d_i\) | `file_0_view_0` | `Demand` | demand quantity |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` | initial inventory |

All numeric values are taken directly from the filtered rows of `SampleSalesData.csv` (rows 95–106 in the original file). No additional data or constraints are introduced beyond those specified.