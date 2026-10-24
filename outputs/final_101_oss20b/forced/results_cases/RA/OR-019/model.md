**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* contains “27in”.  
- For each \(i\in I\) denote  

\[
\begin{aligned}
\text{rev}_i &\;=\;\text{Revenue of product }i,\\
\text{dem}_i &\;=\;\text{Demand of product }i,\\
\text{init}_i &\;=\;\text{Initial Inventory of product }i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\;=\;\; \text{number of units of product }i\text{ to fulfill},\qquad
x_i\in\mathbb{Z}_{\ge 0}\;\;\forall i\in I .
\]

**Objective**

\[
\max \;\sum_{i\in I}\text{rev}_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le \text{dem}_i &&\forall i\in I,\\
x_i &\le \text{init}_i &&\forall i\in I,\\
x_i &\ge 0 &&\forall i\in I.
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column |
|--------|--------|----------|--------|
| \(I\) | Index set | `file_0_view_0` | `Product Name` |
| \(\text{rev}_i\) | Parameter | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | Parameter | `file_0_view_0` | `Demand` |
| \(\text{init}_i\) | Parameter | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

All parameters are taken directly from the filtered rows of `SalesDataAnalysis.csv` where *Product Name* contains “27in”. No additional data or assumptions are introduced.