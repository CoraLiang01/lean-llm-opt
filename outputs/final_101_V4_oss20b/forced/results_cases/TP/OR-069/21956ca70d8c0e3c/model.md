**Mathematical Model**

Let  

* \(M=\{\text{Manager 1},\dots ,\text{Manager 11}\}\) (Managers)  
* \(P=\{\text{Project 1},\dots ,\text{Project 11}\}\) (Projects)

Decision variables  
\[
x_{mp}\in\{0,1\}\quad\forall m\in M,\;p\in P
\]
\(x_{mp}=1\) if manager \(m\) is assigned to project \(p\), otherwise \(0\).

Parameters  
\[
c_{mp}\ge 0 \quad\forall m\in M,\;p\in P
\]
cost of assigning manager \(m\) to project \(p\).

Objective  
\[
\min \sum_{m\in M}\sum_{p\in P} c_{mp}\,x_{mp}
\]

Constraints  

1. **One project per manager**  
\[
\sum_{p\in P} x_{mp}=1 \quad\forall m\in M
\]

2. **One manager per project**  
\[
\sum_{m\in M} x_{mp}=1 \quad\forall p\in P
\]

3. **Binary assignment**  
\[
x_{mp}\in\{0,1\}\quad\forall m\in M,\;p\in P
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Column Mapping |
|-----------|--------------|----------------|
| \(c_{mp}\) | `file_0_view_0` | `Manager` → \(m\); `Project k Cost` → \(c_{mp}\) where `k` is the project number (1–11) |

*The table `file_0_view_0` contains 11 rows (one per manager) and 12 columns: `Manager` plus `Project 1 Cost` through `Project 11 Cost`. Each `Project k Cost` column supplies the cost value \(c_{mp}\) for the corresponding manager–project pair.*