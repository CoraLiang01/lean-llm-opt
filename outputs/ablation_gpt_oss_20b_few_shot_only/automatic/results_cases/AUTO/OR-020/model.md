**Mathematical Model**

```math
\begin{aligned}
\text{Sets}
&\quad I \;\; \text{= set of all products (indexed by product name)} \\[4pt]

\text{Parameters}
&\quad r_i \;\; \text{= revenue per unit of product } i, \\
&\quad d_i \;\; \text{= deterministic demand for product } i, \\
&\quad I_i \;\; \text{= initial inventory of product } i. \\[4pt]

\text{Decision Variables}
&\quad x_i \;\; \in \mathbb{Z}, \; x_i \ge 1 \quad \forall i\in I \\[4pt]

\text{Objective}
&\quad \max \;\; \sum_{i\in I} r_i\,x_i \\[4pt]

\text{Constraints}
&\quad x_i \;\le\; d_i \qquad \forall i\in I \\[2pt]
&\quad x_i \;\le\; I_i \qquad \forall i\in I
\end{aligned}
```

---

**Data Mapping**

| Parameter | Source File | Source Column |
|-----------|-------------|---------------|
| `r_i` | `Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv` | `Revenue` |
| `d_i` | `Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv` | `Demand` |
| `I_i` | `Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv` | `Initial Inventory` |
| Index `i` | `Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv` | `Product Name` |

*No additional numerical values or derived parameters are included; all coefficients are taken directly from the specified columns.*