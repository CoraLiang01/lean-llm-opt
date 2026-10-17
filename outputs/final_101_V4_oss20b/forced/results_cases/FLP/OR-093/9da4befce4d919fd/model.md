**Mathematical Model**

Let  

- \(M=\{M_1,\dots ,M_{12}\}\) machines  
- \(T=\{A,B,\dots ,L\}\) tasks  

Parameters  

- \(c_{m t}\) machining cost of assigning machine \(m\) to task \(t\)  

Decision variables  

- \(x_{m t}\in\{0,1\}\) 1 if machine \(m\) is assigned to task \(t\), 0 otherwise  

\[
\begin{aligned}
\min\;&\sum_{m\in M}\sum_{t\in T} c_{m t}\,x_{m t} \\[4pt]
\text{s.t.}\;&\sum_{t\in T} x_{m t}=1 &&\forall m\in M \quad\text{(each machine to one task)}\\
&\sum_{m\in M} x_{m t}=1 &&\forall t\in T \quad\text{(each task to one machine)}\\
&x_{m t}\in\{0,1\} &&\forall m\in M,\;t\in T
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(c_{m t}\) | `file_0_view_0` | \(c_{m t} =\) value in row where `Machine = m` and column `t` (columns A–L) |
| Index set \(M\) | `file_0_view_0` | \(M =\) set of distinct `Machine` values (M1–M12) |
| Index set \(T\) | `file_0_view_0` | \(T =\) set of column headers A–L |

*Table ID:* `file_0_view_0`  
*Columns:* `Machine`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`  

No additional constraints or parameters are required.