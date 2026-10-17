**Mathematical Model**

Let  

* \(S=\{1,\dots ,48\}\) : index of possible shift start times (each half‑hour slot)  
* \(T=\{1,\dots ,48\}\) : index of half‑hour time slots in the 24‑hour horizon  

Parameters  

* \(R_t\) : required number of waitstaff in slot \(t\) (from *44.csv*)  
* \(A_{s,t}\) : 1 if a shift that starts at slot \(s\) covers slot \(t\), 0 otherwise  
  \[
  A_{s,t}= 
  \begin{cases}
  1 & \text{if } (t-s+48)\bmod 48 < 16\\[4pt]
  0 & \text{otherwise}
  \end{cases}
  \]

Decision variables  

* \(y_s \in \mathbb{Z}_{\ge 0}\) : number of waitstaff whose shift starts at slot \(s\)

Objective  

\[
\min \;\sum_{s\in S} y_s
\]

Constraints  

1. **Coverage of each time slot**  
   \[
   \sum_{s\in S} A_{s,t}\,y_s \;\ge\; R_t , \qquad \forall\, t\in T
   \]

2. **Non‑negativity and integrality**  
   \[
   y_s \in \mathbb{Z}_{\ge 0}, \qquad \forall\, s\in S
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(R_t\) | Requirement per half‑hour slot | `file_0_view_0` | `Requirement` |
| \(A_{s,t}\) | Shift coverage (derived) | – | – |

*The table `file_0_view_0` contains 48 rows, each with a `Time` label and a `Requirement` value. The mapping from row index to \(t\) is the row order (1 → first row, 48 → last row). The coverage matrix \(A_{s,t}\) is not stored in the CSV; it is defined by the 8‑hour (16‑slot) shift rule shown above.*