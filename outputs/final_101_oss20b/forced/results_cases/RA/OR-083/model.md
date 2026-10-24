**Mathematical model**

Let  

* \(W=\{A,B,C,D,E,F,G,H,I,J\}\) (available workers)  
* \(T=\{1,2,\dots ,10\}\) (tasks)

Parameters  

* \(t_{ij}\) – time (hours) for worker \(i\in W\) to complete task \(j\in T\)  
  (obtained from *file_0_view_0*; see Data Mapping below)

Decision variables  

* \(x_{ij}\in\{0,1\}\) 1 if worker \(i\) is assigned to task \(j\)  
* \(y_i\in\{0,1\}\) 1 if worker \(i\) is selected (used only to enforce the
  “10 out of 12” requirement)

Objective  

\[
\min \;\sum_{i\in W}\sum_{j\in T} t_{ij}\,x_{ij}
\]

Constraints  

1. **Task assignment** – each task gets exactly one worker  

\[
\sum_{i\in W} x_{ij}=1 \qquad \forall j\in T
\]

2. **Worker capacity** – a worker can be assigned to at most one task and only
   if selected  

\[
\sum_{j\in T} x_{ij}\le y_i \qquad \forall i\in W
\]

3. **Number of workers used** – exactly ten workers are selected  

\[
\sum_{i\in W} y_i = 10
\]

4. **Linking** – assignment implies selection  

\[
x_{ij}\le y_i \qquad \forall i\in W,\; j\in T
\]

---

**Data Mapping**

| Parameter | Table ID | Row key | Column key |
|-----------|----------|---------|------------|
| \(t_{ij}\) | `file_0_view_0` | `Task Time Required` | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J` |

No external data is required for \(x_{ij}\) or \(y_i\).