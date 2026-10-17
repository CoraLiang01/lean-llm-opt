**Mathematical Model**

Let  

* \(G=\{G1,G2,G3,G4,G5\}\) – set of component families  
* \(O=\{O1,O2,O3,O4\}\) – set of options available for every family  

Decision variables  

\[
x_{g,o}\in\{0,1\}\qquad \forall g\in G,\;o\in O
\]

Parameters (all taken directly from the source tables)

\[
\begin{aligned}
v_{g,o} &\;=\;\text{Value of option }o\text{ in family }g \\
w_{g,o} &\;=\;\text{Weight of option }o\text{ in family }g \\
b_{g,o} &\;=\;\text{BudgetUse of option }o\text{ in family }g \\
W_{\max} &=\;\text{Limit for Weight (48)} \\
B_{\max} &=\;\text{Limit for BudgetUse (60)}
\end{aligned}
\]

**Objective**

\[
\max \;\sum_{g\in G}\sum_{o\in O} v_{g,o}\,x_{g,o}
\]

**Constraints**

1. **Exactly one option per family**

\[
\sum_{o\in O} x_{g,o}=1 \qquad \forall g\in G
\]

2. **Weight capacity**

\[
\sum_{g\in G}\sum_{o\in O} w_{g,o}\,x_{g,o}\;\le\;W_{\max}
\]

3. **Budget‑use capacity**

\[
\sum_{g\in G}\sum_{o\in O} b_{g,o}\,x_{g,o}\;\le\;B_{\max}
\]

4. **Binary variables**

\[
x_{g,o}\in\{0,1\}\qquad \forall g\in G,\;o\in O
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Mapping Rule |
|-----------|--------------|--------|--------------|
| \(v_{g,o}\) | `file_0_view_0` | `Value` | For each record, set \(v_{g,o}\) to the value in the `Value` column. |
| \(w_{g,o}\) | `file_0_view_0` | `Weight` | For each record, set \(w_{g,o}\) to the value in the `Weight` column. |
| \(b_{g,o}\) | `file_0_view_0` | `BudgetUse` | For each record, set \(b_{g,o}\) to the value in the `BudgetUse` column. |
| \(W_{\max}\) | `file_1_view_0` | `Limit` where `Resource`=`Weight` | Set \(W_{\max}=48\). |
| \(B_{\max}\) | `file_1_view_0` | `Limit` where `Resource`=`BudgetUse` | Set \(B_{\max}=60\). |

All indices \(g\) and \(o\) are taken from the distinct values of the `Family` and `Option` columns in `file_0_view_0`. No additional rows or columns are introduced beyond those explicitly present in the source tables.