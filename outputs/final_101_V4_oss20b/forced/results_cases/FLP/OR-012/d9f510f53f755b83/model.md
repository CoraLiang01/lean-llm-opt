**Mathematical Model**

| Symbol | Meaning |
|--------|---------|
| $I$ | Set of products (index set) |
| $p_i$ | Revenue per unit of product $i$ |
| $d_i$ | Demand quantity of product $i$ |
| $I_i$ | Initial inventory of product $i$ |
| $x_i$ | Units of product $i$ to sell (decision variable) |

**Decision Variables**

\[
x_i \;\ge\; 0 , \qquad \forall i\in I
\]

**Objective**

\[
\max \;\sum_{i\in I} p_i\,x_i
\]

**Constraints**

\[
\begin{aligned}
x_i &\;\le\; d_i , &&\forall i\in I \quad (\text{cannot sell more than demand})\\[4pt]
x_i &\;\le\; I_i , &&\forall i\in I \quad (\text{cannot sell more than inventory})
\end{aligned}
\]

---

### Data Mapping Rules

| Parameter / Index | Source Table | Source Column |
|-------------------|--------------|---------------|
| $I$ | `file_0_view_0` | `"Product Name"` |
| $p_i$ | `file_0_view_0` | `"Revenue"` |
| $d_i$ | `file_0_view_0` | `"Demand"` |
| $I_i$ | `file_0_view_0` | `"Initial Inventory"` |

All parameters are real numbers extracted directly from the corresponding columns of the table `file_0_view_0`. No numerical values are enumerated in the model.