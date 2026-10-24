**Mathematical Model**

Let  

\[
\mathcal{M}=\{M_1,M_2,\dots ,M_{12}\}, \qquad 
\mathcal{T}=\{A,B,C,D,E,F,G,H,I,J,K,L\}
\]

\[
c_{m t}\;\; \text{(machining cost of assigning machine }m\text{ to task }t)
\]

Decision variables  

\[
x_{m t}\in\{0,1\}\quad \forall m\in\mathcal{M},\,t\in\mathcal{T}
\]

\[
\text{( }x_{m t}=1\text{ if machine }m\text{ is assigned to task }t\text{)}
\]

**Objective**

\[
\min \;\sum_{m\in\mathcal{M}}\sum_{t\in\mathcal{T}} c_{m t}\,x_{m t}
\]

**Constraints**

1. **One task per machine**

\[
\sum_{t\in\mathcal{T}} x_{m t}=1 \quad \forall m\in\mathcal{M}
\]

2. **One machine per task**

\[
\sum_{m\in\mathcal{M}} x_{m t}=1 \quad \forall t\in\mathcal{T}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{m t}\) | `Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv` | For each row label \(m\) (e.g., `M1`) and each column header \(t\) (e.g., `A`), assign the numeric value in that cell to \(c_{m t}\). |
| \(\mathcal{M}\) | Same CSV | Set of all row labels (`M1`–`M12`). |
| \(\mathcal{T}\) | Same CSV | Set of all column headers (`A`–`L`). |

No other data or constraints are required.