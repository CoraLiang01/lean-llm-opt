**Mathematical Model**

Let  

- \(I\) be the set of display areas indexed by the values of **DisplayID** in *file_0_view_0*.  
- \(J\) be the set of boat types indexed by the values of **ProductName** in *file_1_view_0*.  

Parameters  

\[
\begin{aligned}
\text{Capacity}_i &\quad\text{(capacity of display area }i)\\
\text{Value}_j &\quad\text{(value of boat type }j)\\
\text{Weight}_j &\quad\text{(size/weight of boat type }j)
\end{aligned}
\]

Decision variables  

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

### Objective  

\[
\max \;\; \sum_{i\in I}\sum_{j\in J} \text{Value}_j \; x_{ij}
\]

### Constraints  

1. **Capacity of each display area**

\[
\sum_{j\in J} \text{Weight}_j \; x_{ij}\;\le\; \text{Capacity}_i
\qquad \forall i\in I
\]

2. **Non‑negativity and integrality**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\qquad \forall i\in I,\; j\in J
\]

---

**Data Mapping**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(\text{Capacity}_i\) | `file_0_view_0` | `Capacity` |
| \(\text{Value}_j\) | `file_1_view_0` | `Value` |
| \(\text{Weight}_j\) | `file_1_view_0` | `Weight` |

The index sets \(I\) and \(J\) are derived from the distinct values of **DisplayID** and **ProductName** in the respective tables. No other data are required.