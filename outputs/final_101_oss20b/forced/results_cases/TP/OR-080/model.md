**Mathematical Model**

Let  

* \(T=\{1,\dots ,10\}\) – set of trucks  
* \(P=\{1,2,3,4\}\) – set of periods  

Parameters (all given in *parameters.csv*)  

| Symbol | Meaning | Source column |
|--------|---------|---------------|
| \(Q_t\) | maximum capacity of truck \(t\) (kg) | `Q` |
| \(S_t\) | startup cost of truck \(t\) | `S` |
| \(C_t\) | unit transportation cost of truck \(t\) (per kg) | `C` |
| \(d_p\) | customer demand in period \(p\) (kg) | `d1`–`d4` (same for all trucks) |

Decision variables  

* \(a_{t,p}\in\{0,1\}\) – 1 if truck \(t\) is active in period \(p\)  
* \(s_{t,p}\in\{0,1\}\) – 1 if truck \(t\) is started in period \(p\)  
* \(w_{t,p}\ge 0\) – weight transported by truck \(t\) in period \(p\) (kg)

---

### Objective
\[
\min \;\sum_{t\in T}\sum_{p\in P}\Bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\Bigr)
\]

---

### Constraints  

1. **Startup definition**  
   \[
   \begin{aligned}
   &s_{t,1}=a_{t,1}\qquad &&\forall t\in T\\
   &s_{t,p}\ge a_{t,p}-a_{t,p-1}\qquad &&\forall t\in T,\;p=2,\dots ,4\\
   &s_{t,p}\le a_{t,p}\qquad &&\forall t\in T,\;p=1,\dots ,4\\
   &s_{t,p}\le 1-a_{t,p-1}\qquad &&\forall t\in T,\;p=2,\dots ,4\\
   &s_{t,4}=0\qquad &&\forall t\in T
   \end{aligned}
   \]

2. **Active‑period continuity**  
   \[
   \begin{aligned}
   &a_{t,1}\le a_{t,2}\le a_{t,3}\le a_{t,4}\qquad &&\forall t\in T\\
   &a_{t,4}\le a_{t,3}\qquad &&\forall t\in T
   \end{aligned}
   \]

3. **Shutdown restriction**  
   \[
   a_{t,p+1}\le a_{t,p}+(1-a_{t,p-1})\qquad &&\forall t\in T,\;p=2,3
   \]

4. **Capacity limits**  
   \[
   0\le w_{t,p}\le Q_t\,a_{t,p}\qquad &&\forall t\in T,\;p\in P
   \]

5. **Weight‑change limits**  
   \[
   \begin{aligned}
   &w_{t,p}-w_{t,p-1}\le 300\qquad &&\forall t\in T,\;p=2,\dots ,4\\
   &w_{t,p-1}-w_{t,p}\le 300\qquad &&\forall t\in T,\;p=2,\dots ,4
   \end{aligned}
   \]

6. **Demand satisfaction**  
   \[
   \sum_{t\in T} w_{t,p}\;\ge\; d_p\qquad &&\forall p\in P
   \]

7. **Spare‑capacity buffer**  
   \[
   \sum_{t\in T} w_{t,p}\;\le\;0.9\;\sum_{t\in T} Q_t\,a_{t,p}\qquad &&\forall p\in P
   \]

---

### Data Mapping Rules

| Parameter | Table ID | Column name |
|-----------|----------|-------------|
| \(Q_t\) | `file_0_view_0` | `Q` |
| \(S_t\) | `file_0_view_0` | `S` |
| \(C_t\) | `file_0_view_0` | `C` |
| \(d_1\) | `file_0_view_0` | `d1` |
| \(d_2\) | `file_0_view_0` | `d2` |
| \(d_3\) | `file_0_view_0` | `d3` |
| \(d_4\) | `file_0_view_0` | `d4` |

All indices \(t\) and \(p\) are taken from the sets \(T\) and \(P\) defined above.  
The model uses only the data supplied in *parameters.csv* and respects all constraints stated in the query.