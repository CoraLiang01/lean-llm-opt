**Mathematical Model**

Let  

- \(I\) – set of all products in *MobileSalesDataset.csv* (index \(i\)).  

Parameters (all read from the table):  

\[
\begin{aligned}
R_i &\;=\; \text{Revenue of product } i,\\
D_i &\;=\; \text{Demand of product } i,\\
S_i &\;=\; \text{Initial Inventory of product } i.
\end{aligned}
\]

Decision variables  

\[
x_i \;\in\; \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} R_i\,x_i
\]

Constraints  

\[
\begin{aligned}
x_i &\le D_i &&\forall i\in I &&\text{(cannot exceed demand)}\\[4pt]
x_i &\le S_i &&\forall i\in I &&\text{(cannot exceed available inventory)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column Name |
|-----------|--------------|-------------|
| \(R_i\)   | `file_0_view_0` | `Revenue` |
| \(D_i\)   | `file_0_view_0` | `Demand` |
| \(S_i\)   | `file_0_view_0` | `Initial Inventory` |

All products in the table constitute the index set \(I\). No additional data or constraints are introduced beyond those specified in the query.