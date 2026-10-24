**Symbolic Model**

\[
\begin{aligned}
\text{Sets} \quad & T=\{1,\dots,48\} &&\text{(half‑hour time slots)}\\
& S=T &&\text{(possible start times of an 8‑hour shift)}\\[4pt]

\text{Parameters} \quad & R_t &&\text{required waitstaff at slot }t\in T\\
& C_{s,t} &&=
\begin{cases}
1 & \text{if }(t-s \bmod 48)\in\{0,\dots,15\}\\
0 & \text{otherwise}
\end{cases} &&\text{(coverage indicator)}\\[4pt]

\text{Decision Variables} \quad & x_s \in \mathbb{Z}_{+} &&\text{number of staff starting at slot }s\in S\\[4pt]

\text{Objective} \quad & \min \; Z = \sum_{s\in S} x_s \\[4pt]

\text{Constraints} \quad & \sum_{s\in S} C_{s,t}\,x_s \;\ge\; R_t
&&\forall\,t\in T \quad\text{(coverage of each time slot)}\\
& x_s \;\ge\; 0 \quad\forall\,s\in S \\
& x_s \in \mathbb{Z} \quad\forall\,s\in S
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(R_t\) | Requirement values | `file_0_view_0` | `Requirement` |
| \(C_{s,t}\) | Derived from time‑shift logic | – | – |

*The set \(T\) and \(S\) are defined by the 48 half‑hour slots present in the file `44.csv` (table ID `file_0_view_0`). No other data values are required for the model.*