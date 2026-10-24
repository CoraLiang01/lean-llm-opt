**Mathematical Model**

Let  

- \(I\) be the set of all products whose *Product Name* contains the string **“27in”** (the rows returned by the filter).  
- For each \(i\in I\) define the parameters  

\[
\begin{aligned}
\text{rev}_i &\;:=\; \text{Revenue of product }i \\
\text{dem}_i &\;:=\; \text{Demand of product }i \\
\text{inv}_i &\;:=\; \text{Initial Inventory of product }i
\end{aligned}
\]

- Decision variable  

\[
x_i \;=\; \text{number of units of product }i\text{ to be sold}
\]

---

### Objective

\[
\max \;\sum_{i\in I} \text{rev}_i \; x_i
\]

### Constraints

\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; \text{dem}_i &&\forall i\in I \quad (\text{cannot sell more than demand})\\[4pt]
&0 \;\le\; x_i \;\le\; \text{inv}_i &&\forall i\in I \quad (\text{cannot sell more than available inventory})\\[4pt]
&x_i \;\in\; \mathbb{Z}_{\ge 0} &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

The index set \(I\) is derived from the rows of `file_0_view_0` that satisfy the filter condition `Product Name` contains `"27in"`. No other tables or columns are used.