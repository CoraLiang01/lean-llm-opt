**Mathematical Model**

Let  

- \(I\) be the set of products with names beginning “S700_”.  
- For each \(i\in I\)  

\[
\begin{aligned}
\text{rev}_i &\;=\; \text{Revenue of product }i,\\
\text{dem}_i &\;=\; \text{Demand for product }i,\\
\text{inv}_i &\;=\; \text{Initial inventory of product }i.
\end{aligned}
\]

Decision variable  

\[
x_i \;=\; \text{number of units of product }i\text{ to fulfill}.
\]

\[
\begin{aligned}
\max\;&\sum_{i\in I}\text{rev}_i\,x_i \\[4pt]
\text{s.t. }&x_i \;\le\; \text{dem}_i &&\forall i\in I \\[2pt]
&x_i \;\le\; \text{inv}_i &&\forall i\in I \\[2pt]
&x_i \;\ge\; 0,\; x_i \in \mathbb{Z} &&\forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

All mappings use the exact identifiers and column names from the observation.