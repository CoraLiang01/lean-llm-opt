**Mathematical Model**

Let  

- \(I\) be the set of products indexed by the values in the column **Product Name**.  
- For each \(i\in I\) let  

\[
\begin{aligned}
\text{rev}_i &\;=\;\text{Revenue of product }i,\\
\text{dem}_i &\;=\;\text{Demand of product }i,\\
\text{inv}_i &\;=\;\text{Initial Inventory of product }i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\;=\;\;\text{number of units of product }i\text{ to fulfill},\qquad
x_i\in\mathbb{Z}_{\ge 0}\;\;\forall i\in I .
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
x_i &\;\le\;\text{dem}_i &&\forall i\in I &&\text{(demand limit)}\\[4pt]
x_i &\;\le\;\text{inv}_i &&\forall i\in I &&\text{(inventory limit)}\\[4pt]
x_i &\;\ge\;0 &&\forall i\in I &&\text{(non‑negativity)}
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column Name |
|--------|--------------|-------------|
| \(I\) | `file_0_view_0` | `Product Name` |
| \(\text{rev}_i\) | `file_0_view_0` | `Revenue` |
| \(\text{dem}_i\) | `file_0_view_0` | `Demand` |
| \(\text{inv}_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. The variable domain is specified as non‑negative integers.