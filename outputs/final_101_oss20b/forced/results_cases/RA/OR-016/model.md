**Mathematical Model**

Let  

- \(I\) be the set of products indexed by the *Product Name* column.  
- For each \(i\in I\) let  

  \[
  \begin{aligned}
  r_i &\;=\;\text{Revenue per unit of product }i,\\
  d_i &\;=\;\text{Demand (maximum units that can be sold) for product }i,\\
  s_i &\;=\;\text{Initial inventory available for product }i.
  \end{aligned}
  \]

- Decision variable  

  \[
  x_i \;=\;\text{number of units of product }i\text{ to sell},\qquad x_i\in\mathbb{Z}_{\ge 0}.
  \]

The optimization problem is

\[
\begin{aligned}
\max_{x}\;&\sum_{i\in I} r_i\,x_i \\[4pt]
\text{s.t. }&0 \le x_i \le d_i &&\forall i\in I,\\
            &0 \le x_i \le s_i &&\forall i\in I,\\
            &x_i \in \mathbb{Z}_{\ge 0} &&\forall i\in I.
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(r_i\) | `file_0_view_0` | `Revenue` |
| \(d_i\) | `file_0_view_0` | `Demand` |
| \(s_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or assumptions are introduced beyond what is specified in the query.