**Mathematical Model**

Let  

\[
I=\{i\mid \text{Product Name contains “27in”}\}
\]

Parameters (all taken from *file_0_view_0*)  

\[
\begin{aligned}
\text{rev}_i &:= \text{Revenue of product }i \quad(\$)\\
\text{dem}_i &:= \text{Demand of product }i \quad(\text{units})\\
\text{inv}_i &:= \text{Initial Inventory of product }i \quad(\text{units})
\end{aligned}
\]

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad\forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I}\text{rev}_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\le \text{dem}_i &&\forall i\in I \quad(\text{cannot sell more than demand})\\
x_i &\le \text{inv}_i &&\forall i\in I \quad(\text{cannot sell more than available inventory})
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| `rev_i`   | `file_0_view_0` | `Revenue` |
| `dem_i`   | `file_0_view_0` | `Demand` |
| `inv_i`   | `file_0_view_0` | `Initial Inventory` |

| Variable | Domain |
|----------|--------|
| `x_i` | Integer, \(x_i \ge 0\) |

The index set \(I\) is defined by the filter applied to *SalesDataAnalysis.csv*:  
`Product Name` contains the substring `"27in"`.  
No other products are included in the model.