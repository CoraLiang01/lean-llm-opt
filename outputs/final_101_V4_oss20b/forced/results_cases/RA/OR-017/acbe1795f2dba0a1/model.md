**Mathematical Model**

Let  

- \(I\) be the set of all SKUs that start with “ZZ”.  
- For each \(i\in I\) let  

\[
\begin{aligned}
\text{rev}_i &\;=\;\text{Revenue of SKU }i,\\
\text{dem}_i &\;=\;\text{Demand of SKU }i,\\
\text{inv}_i &\;=\;\text{Initial Inventory of SKU }i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\;=\;\;\text{number of units of SKU }i\text{ to sell},\qquad x_i\in\mathbb{Z}_{\ge 0}.
\]

---

**Objective**

\[
\max \;\sum_{i\in I}\text{rev}_i\,x_i
\]

---

**Constraints**

\[
\begin{aligned}
&0 \;\le\; x_i \;\le\; \text{dem}_i &&\forall i\in I \quad\text{(cannot sell more than demand)}\\[4pt]
&0 \;\le\; x_i \;\le\; \text{inv}_i &&\forall i\in I \quad\text{(cannot sell more than available inventory)}
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source | Table ID | Column Name |
|--------|--------|----------|-------------|
| \(I\) | Index set | `file_0_view_0` | `SKU` |
| \(\text{rev}_i\) | Parameter | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | Parameter | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | Parameter | `file_0_view_0` | `Initial Inventory` |
| \(x_i\) | Decision variable | – | – |

All numeric values are taken directly from the corresponding columns of the filtered view `file_0_view_0`.