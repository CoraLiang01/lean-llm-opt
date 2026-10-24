**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{j\in J}\; \text{idle}_j \\[4pt]
\text{subject to}\quad
& \sum_{i\in I}\; t_{ij}\,x_i \;+\; \text{idle}_j \;=\; \text{cap}_j,
&& \forall j\in J, \\[4pt]
& x_i \;\in\; \mathbb{Z}_{\ge 0},
&& \forall i\in I, \\[4pt]
& \text{idle}_j \;\ge\; 0,
&& \forall j\in J .
\end{aligned}
\]

*Indices*  
- \(I=\{1,2,\dots ,101\}\) (HiFi‑1 … HiFi‑101)  
- \(J=\{1,2,3\}\) (Workstations 1–3)

*Parameters*  
- \(t_{ij}\) – processing time (minutes) per unit of model \(i\) at workstation \(j\).  
- \(\text{cap}_j\) – effective daily capacity of workstation \(j\) (minutes).

*Derived parameter*  

\[
\text{cap}_j \;=\; 1440 \times \Bigl(1-\frac{\text{Maintenance\_Percent}_j}{100}\Bigr),
\qquad j\in J .
\]

---

### Data Mapping

| Table ID | Column | Mapping |
|----------|--------|---------|
| `file_0_view_0` | `Workstation` | Index \(j\) of workstation. |
| `file_0_view_0` | `Maintenance_Percent` | Parameter \(\text{Maintenance\_Percent}_j\). |
| `file_0_view_0` | `HiFi{i}_Minutes` | Parameter \(t_{ij}\) for each \(i=1,\dots,101\). |

*Explanation*  
- For each row in `file_0_view_0`, the value in `Workstation` identifies \(j\).  
- The value in `Maintenance_Percent` gives the maintenance percentage for that workstation; use it to compute \(\text{cap}_j\) as shown above.  
- Each column named `HiFi{i}_Minutes` supplies the processing time \(t_{ij}\) for model \(i\) at workstation \(j\).  

No other data are required; the sets \(I\) and \(J\) are defined solely by the identifiers in the table.