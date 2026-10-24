**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{w\in W}\sum_{t\in T} c_{w,t}\,x_{w,t} \\[4pt]
\text{s.t.}\;&\sum_{w\in W} x_{w,t}=1 &&\forall\,t\in T \quad\text{(each task gets one worker)}\\[2pt]
&\sum_{t\in T} x_{w,t}\le 1 &&\forall\,w\in W \quad\text{(each worker does at most one task)}\\[2pt]
&\sum_{w\in W}\sum_{t\in T} x_{w,t}=10 &&\text{(exactly ten workers are used)}\\[4pt]
&x_{w,t}\in\{0,1\} &&\forall\,w\in W,\;t\in T
\end{aligned}
\]

**Index Sets**

| Symbol | Description | Elements |
|--------|-------------|----------|
| \(W\)  | Workers | \{A,B,C,D,E,F,G,H,I,J\} |
| \(T\)  | Tasks | \{1,2,3,4,5,6,7,8,9,10\} |

**Parameters**

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_{w,t}\) | Time required for worker \(w\) to complete task \(t\) | Table **file_0_view_0** (15.csv) – value in column \(w\) at row where *Task Time Required* = \(t\) |

**Data Mapping Rules**

- **Table ID**: `file_0_view_0`
- **Columns**: `Task Time Required`, `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`
- **Parameter mapping**:  
  \[
  c_{w,t} = \text{value in row with } \text{Task Time Required}=t \text{ and column } w
  \]
  for all \(w\in W\) and \(t\in T\).  
  (Rows 11–12 of the table are ignored because only tasks 1–10 are considered.)