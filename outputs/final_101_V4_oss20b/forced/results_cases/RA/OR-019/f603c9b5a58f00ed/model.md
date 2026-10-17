**Mathematical Model**

Let  

- \(I\) be the set of products whose *Product Name* contains “27in”.  
- For each \(i\in I\) let  

  \[
  \begin{aligned}
  \text{rev}_i &\;=\;\text{Revenue of product }i,\\
  \text{dem}_i &\;=\;\text{Demand of product }i,\\
  \text{inv}_i &\;=\;\text{Initial Inventory of product }i.
  \end{aligned}
  \]

- Decision variable  

  \[
  x_i \;=\;\text{number of units of product }i\text{ to be fulfilled},\qquad x_i\in\mathbb{Z}_{\ge 0}.
  \]

**Objective**

\[
\max \;\sum_{i\in I}\text{rev}_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\le \text{dem}_i &&\forall i\in I &&\text{(cannot exceed demand)}\\
x_i &\le \text{inv}_i &&\forall i\in I &&\text{(cannot exceed available inventory)}\\
x_i &\ge 0 &&\forall i\in I &&\text{(non‑negativity)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Variable | Table ID          | Column Name          | Index Key      |
|----------------------|-------------------|----------------------|----------------|
| \(\text{rev}_i\)     | file_0_view_0     | Revenue              | Product Name   |
| \(\text{dem}_i\)     | file_0_view_0     | Demand               | Product Name   |
| \(\text{inv}_i\)     | file_0_view_0     | Initial Inventory    | Product Name   |
| \(x_i\)              | –                 | –                    | Product Name   |

All indices and parameters are derived directly from the filtered rows of **SalesDataAnalysis.csv** where *Product Name* contains “27in”.