**Mathematical Model**

Let  

- \(I\) be the set of products indexed by the values in the column **Product Name**.  

Parameters (indexed by \(i\in I\))  

\[
\begin{aligned}
R_i &\;=\;\text{Revenue per unit of product }i,\\
D_i &\;=\;\text{Demand for product }i,\\
I_i &\;=\;\text{Initial inventory of product }i.
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

Subject to  

\[
\begin{aligned}
x_i &\le D_i &&\forall i\in I \quad\text{(cannot exceed demand)}\\
x_i &\le I_i &&\forall i\in I \quad\text{(cannot exceed inventory)}
\end{aligned}
\]

---

**Data Mapping**

| Parameter / Index | Source Table | Source Column |
|-------------------|--------------|---------------|
| \(I\) (index set) | `file_0_view_0` | `Product Name` |
| \(R_i\) | `file_0_view_0` | `Revenue` |
| \(D_i\) | `file_0_view_0` | `Demand` |
| \(I_i\) | `file_0_view_0` | `Initial Inventory` |

All parameters are taken directly from the corresponding columns of the table `file_0_view_0`. No additional data or transformations are introduced.