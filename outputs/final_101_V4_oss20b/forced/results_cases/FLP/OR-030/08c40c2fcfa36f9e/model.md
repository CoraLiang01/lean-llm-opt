**Mathematical Model**

Let  

- \(I\) be the set of rows in *BigMartSales.csv* whose *Product Name* starts with **FDK57**.  
- For each \(i\in I\) define the parameters  

\[
\begin{aligned}
r_i &\;=\;\text{Revenue of row }i,\\
d_i &\;=\;\text{Demand of row }i,\\
v_i &\;=\;\text{Initial Inventory of row }i .
\end{aligned}
\]

Decision variable  

\[
x_i \;\ge\; 0 \qquad (i\in I)
\]

represents the quantity of model \(i\) that will be sold.

---

### Objective

\[
\max \;\sum_{i\in I} r_i\,x_i
\]

---

### Constraints

\[
\begin{aligned}
x_i &\;\le\; d_i &&\forall i\in I &&\text{(cannot sell more than demand)}\\[4pt]
x_i &\;\le\; v_i &&\forall i\in I &&\text{(cannot sell more than inventory)}\\[4pt]
x_i &\;\ge\; 0   &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Columns | Mapping Rule |
|-----------|--------------|----------------|--------------|
| \(r_i\)   | file_0_view_0 | Revenue | \(r_i = \text{Revenue}_{i}\) |
| \(d_i\)   | file_0_view_0 | Demand | \(d_i = \text{Demand}_{i}\) |
| \(v_i\)   | file_0_view_0 | Initial Inventory | \(v_i = \text{Initial Inventory}_{i}\) |
| \(I\)     | file_0_view_0 | Product Name | \(I = \{\,i \mid \text{Product Name}_{i} \text{ starts with } \text{FDK57}\,\}\) |

All indices \(i\) refer to the rows returned by the filter in the observation (8 rows). No other data or constraints are required.