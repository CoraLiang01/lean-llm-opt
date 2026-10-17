**Symbolic Model**

Let  

* \(I\) – set of products (indexed by the *Product Name* values).  
* Parameters (all obtained from the table *file_0_view_0*):  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue of product }i,\\
d_i &\;=\;\text{Demand quantity of product }i,\\
inv_i &\;=\;\text{Initial inventory of product }i.
\end{aligned}
\]

*Decision variable*  

\[
x_i \;\ge\;0 \quad\text{(units of product }i\text{ to be sold)}.
\]

**Objective**

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; inv_i &&\forall i\in I &&\text{(cannot sell more than available inventory)}\\[4pt]
x_i &\;\ge\;0 &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Table ID | Column in model | Source column (CSV) |
|----------|-----------------|---------------------|
| file_0_view_0 | \(I\) (product identifiers) | *Product Name* |
| file_0_view_0 | \(r_i\) | *Revenue* |
| file_0_view_0 | \(d_i\) | *Demand* |
| file_0_view_0 | \(inv_i\) | *Initial Inventory* |

All parameter values are taken directly from the corresponding rows of the CSV file *WomenClothingEcommerceSalesData.csv* (view *file_0_view_0*). No additional numerical limits are introduced beyond those supplied in the data.