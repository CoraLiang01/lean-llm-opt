**Mathematical Model**

```math
\begin{aligned}
\text{Sets}
&\quad S \;=\;\{S_1,\dots ,S_{10}\} \\
&\quad D \;=\;\{D_1,\dots ,D_{20}\} \\[4pt]

\text{Parameters}
&\quad \text{supply}[s] \;\; (s\in S) \\
&\quad \text{demand}[d] \;\; (d\in D) \\
&\quad \text{cost}[s,d] \;\; (s\in S,\; d\in D) \\[4pt]

\text{Decision Variables}
&\quad x[s,d] \;\in\;\mathbb{Z}_{\ge 0} \quad\text{(units shipped)}\\
&\quad y[s,d] \;\in\;\mathbb{Z}_{\ge 0} \quad\text{(trucks dispatched)}\\[4pt]

\text{Objective}
&\quad \min \;\sum_{s\in S}\;\sum_{d\in D}\;\text{cost}[s,d]\;x[s,d] \\[6pt]

\text{Constraints}
&\quad \sum_{d\in D} x[s,d] \;\le\; \text{supply}[s] \qquad \forall s\in S \quad\text{(source capacity)}\\[4pt]
&\quad \sum_{s\in S} x[s,d] \;=\; \text{demand}[d] \qquad \forall d\in D \quad\text{(demand satisfaction)}\\[4pt]
&\quad x[s,d] \;\le\; 10\,y[s,d] \qquad \forall s\in S,\; d\in D \quad\text{(truck capacity)}\\[4pt]
&\quad x[s,d] \;\ge\; 0,\; y[s,d] \;\ge\; 0 \qquad \forall s\in S,\; d\in D \\
&\quad x[s,d],\,y[s,d] \;\in\;\mathbb{Z} \qquad \forall s\in S,\; d\in D
\end{aligned}
```

---

**Data Mapping**

| Parameter | Source Table | Row‑ID Column | Column(s) |
|-----------|--------------|---------------|-----------|
| `supply[s]` | `file_2_view_0` | `source_id` | `supply_units` |
| `demand[d]` | `file_1_view_0` | `destination_id` | `demand_units` |
| `cost[s,d]` | `file_0_view_0` | `source_id` | `D1`–`D20` (destination columns) |

All indices `s` and `d` are taken from the respective `source_id` and `destination_id` columns of the tables. No numerical values are included in the model; they are supplied by the data mapping above.