**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & Z = \sum_{m\in M}\sum_{p\in P} c_{m,p}\,x_{m,p} \\[4pt]
\text{subject to}\quad
& \sum_{p\in P} x_{m,p}=1 && \forall m\in M \quad\text{(each manager gets one project)}\\
& \sum_{m\in M} x_{m,p}=1 && \forall p\in P \quad\text{(each project gets one manager)}\\[4pt]
& x_{m,p}\in\{0,1\} && \forall m\in M,\;p\in P
\end{aligned}
\]

**Data Mapping**

| Symbol | Source | Table ID | Column(s) | Description |
|--------|--------|----------|-----------|-------------|
| \(M\) | Manager index | `file_0_view_0` | `Unnamed: 0` | Unique manager identifiers |
| \(P\) | Project index | `file_0_view_0` | `P1`, `P2`, `P3` | Project identifiers |
| \(c_{m,p}\) | Cost parameter | `file_0_view_0` | Row where `Unnamed: 0 = m`, column `p` | Cost of assigning manager *m* to project *p* |

*All parameter values are taken directly from the corresponding cells of the CSV file; no numerical values are listed here.*