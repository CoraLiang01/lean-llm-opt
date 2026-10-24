**Symbolic Model**

| Symbol | Meaning |
|--------|---------|
| \(H\) | Set of hour indices \(\{1,2,\dots ,24\}\) |
| \(d_h\) | Number of drivers/crew required during hour \(h\) |
| \(x_h\) | Number of workers that start their 4‑hour shift at hour \(h\) |

**Parameters**

\[
\begin{aligned}
&d_h \quad &\forall h \in H
\end{aligned}
\]

**Decision Variables**

\[
x_h \in \mathbb{Z}_{\ge 0} \qquad \forall h \in H
\]

**Objective**

\[
\min \;\; \sum_{h \in H} x_h
\]

**Constraints**

For every hour \(h \in H\) the total number of workers whose 4‑hour shift covers hour \(h\) must be at least the required number:

\[
\boxed{\;\;x_h + x_{h-1} + x_{h-2} + x_{h-3} \;\ge\; d_h \;\;\;\forall h \in H\;}
\]

Indices are taken modulo 24, i.e. \(x_{0}=x_{24}\), \(x_{-1}=x_{23}\), \(x_{-2}=x_{22}\), \(x_{-3}=x_{21}\).

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(d_h\) | `file_0_view_0` | `Number Required` | For each record, the value in `Number Required` is assigned to \(d_{\text{Shift}}\). |
| \(H\) | `file_0_view_0` | `Shift` | The set \(H\) is the collection of all distinct `Shift` values (1–24). |

No other data are required. The model uses only the hourly requirement values from the CSV and the cyclic 4‑hour coverage implied by the problem statement.