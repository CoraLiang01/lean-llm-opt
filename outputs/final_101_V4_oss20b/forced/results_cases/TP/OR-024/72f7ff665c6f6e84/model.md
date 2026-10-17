**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* starts with the prefix **S700_**.  
- For each \(i\in I\) define the parameters  

\[
\begin{aligned}
\text{rev}_i &\;=\;\text{Revenue of product }i,\\
\text{dem}_i &\;=\;\text{Demand quantity of product }i,\\
\text{inv}_i &\;=\;\text{Initial inventory of product }i.
\end{aligned}
\]

Decision variable  

\[
x_i \;\ge\; 0 \qquad\text{(units of product }i\text{ to be sold)}
\]

---

\[
\begin{aligned}
\max\;&\sum_{i\in I}\text{rev}_i\,x_i \\[4pt]
\text{s.t. }&x_i \;\le\; \text{dem}_i \qquad \forall i\in I \\[2pt]
&x_i \;\le\; \text{inv}_i \qquad \forall i\in I \\[2pt]
&x_i \;\ge\; 0 \qquad \forall i\in I
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Source Column | Notes |
|-----------|--------------|---------------|-------|
| \(I\) | `file_0_view_0` | `Product Name` | Include rows where `Product Name` starts with `S700_`. |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` | Numeric value per product. |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` | Numeric value per product. |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` | Numeric value per product. |

All parameters are taken directly from the specified columns of the table `file_0_view_0`; no aggregation or transformation beyond the prefix filter is performed. The model uses continuous decision variables and respects the inventory and demand limits for each product.