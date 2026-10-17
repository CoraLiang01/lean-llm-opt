**Mathematical Model**

```math
\begin{aligned}
\text{Sets}
&\quad I \;\; \text{= set of products whose id_number starts with “id999”} \\[4pt]

\text{Parameters}
&\quad \text{demand}[i] \;\; \text{(units demanded for product } i)\\
&\quad \text{inventory}[i] \;\; \text{(initial inventory for product } i)\\
&\quad \text{revenue}[i] \;\; \text{(revenue per unit for product } i)\\[4pt]

\text{Decision Variables}
&\quad x[i] \;\; \in \mathbb{Z}_{+} \quad \text{(units of product } i \text{ to fulfill)}\\[4pt]

\text{Objective}
&\quad \max \;\; \sum_{i\in I} \text{revenue}[i] \, x[i] \\[4pt]

\text{Constraints}
&\quad x[i] \;\le\; \text{demand}[i] \qquad \forall i\in I \\[2pt]
&\quad x[i] \;\le\; \text{inventory}[i] \qquad \forall i\in I
\end{aligned}
```

---

**Data Mapping Rules**

| Parameter / Variable | Source Table | Source Column | Notes |
|----------------------|--------------|---------------|-------|
| `I` | `file_0_view_0` | `id_number` | Each distinct `id_number` starting with “id999” becomes an index `i`. |
| `revenue[i]` | `file_0_view_0` | `Revenue` | Maps directly to the revenue per unit. |
| `demand[i]` | `file_0_view_0` | `Demand` | Maps directly to the known demand quantity. |
| `inventory[i]` | `file_0_view_0` | `Initial Inventory` | Maps directly to the available inventory. |
| `x[i]` | — | — | Decision variable, integer, non‑negative. |

All data are taken verbatim from the single returned row of `file_0_view_0` (row 899). No additional data or aggregation is introduced.