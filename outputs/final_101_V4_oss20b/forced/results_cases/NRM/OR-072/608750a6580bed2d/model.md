**Symbolic Linear‑Programming Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{t\in T} x_t \\[4pt]
\text{subject to}\quad
& \sum_{t\in T} a_{s,t}\,x_t \;\ge\; r_s , && \forall\, s\in T \\[4pt]
& x_t \;\in\; \mathbb{Z}_{+}, && \forall\, t\in T
\end{aligned}
\]

* **Index set**  
  \(T=\{1,2,\dots ,24\}\)  (each element represents a 1‑hour shift).

* **Parameters**  
  \(r_s\) – number of drivers/crew required in shift \(s\).  
  \(a_{s,t}\in\{0,1\}\) – equals 1 if a worker who starts at shift \(t\) works during shift \(s\); otherwise 0.  
  For the 4‑hour continuous work rule  
  \[
  a_{s,t}=1 \iff (s-t)\bmod 24 \in \{0,1,2,3\}.
  \]

* **Decision variables**  
  \(x_t\) – number of workers who start working at the beginning of shift \(t\).

* **Objective**  
  Minimise the total number of workers assigned over the day.

* **Constraints**  
  1. **Coverage** – every shift’s requirement is met.  
  2. **Non‑negativity & integrality** – workers are counted in whole units.

---

### Data Mapping

| Parameter / Variable | Source | Table ID | Column(s) | Mapping |
|----------------------|--------|----------|-----------|---------|
| \(r_s\) | Required number of workers | `file_0_view_0` | `Number Required` | `r_{Shift}` |
| \(a_{s,t}\) | Coverage indicator | `file_0_view_0` | `Shift` (for \(t\)), `Shift` (for \(s\)) | Defined by the 4‑hour rule above |
| \(x_t\) | Decision variable | – | – | Decision variable, not sourced from data |

*The table `file_0_view_0` contains 24 rows, each with a `Shift` identifier (1–24) and the corresponding `Number Required` value.*