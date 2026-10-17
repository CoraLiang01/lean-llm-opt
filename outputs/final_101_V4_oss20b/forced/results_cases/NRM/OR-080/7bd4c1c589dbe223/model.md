**Symbolic Model**

\[
\begin{aligned}
\text{Minimize}\quad
&\sum_{t\in T}\sum_{p\in P}\Bigl(S_t\,s_{t,p}+C_t\,w_{t,p}\Bigr) \\[4pt]
\text{subject to}\quad
&\text{(1) Activation dynamics}\quad
\begin{cases}
a_{t,1}=s_{t,1} & \forall t\\
a_{t,p}=a_{t,p-1}+s_{t,p}-dsh_{t,p-1} & \forall t,\;p=2,\dots,4
\end{cases}\\[4pt]
&\text{(2) Start‑up limits}\quad
\begin{cases}
s_{t,p}\le 1-a_{t,p-1} & \forall t,\;p=2,\dots,4\\
s_{t,4}=0 & \forall t
\end{cases}\\[4pt]
&\text{(3) Shutdown limits}\quad
\begin{cases}
dsh_{t,p}\le a_{t,p} & \forall t,\;p=1,\dots,3\\
a_{t,p+1}\le 1-dsh_{t,p} & \forall t,\;p=1,\dots,3\\
a_{t,p+2}\le 1-dsh_{t,p} & \forall t,\;p=1,\dots,2
\end{cases}\\[4pt]
&\text{(4) Minimum consecutive‑active period}\quad
a_{t,p}\le a_{t,p+1}\quad \forall t,\;p=1,2\\[4pt]
&\text{(5) Capacity limits}\quad
0\le w_{t,p}\le Q_t\,a_{t,p}\quad \forall t,p\\[4pt]
&\text{(6) Load‑change limits}\quad
\begin{cases}
w_{t,1}\le 300 & \forall t\\
|w_{t,p}-w_{t,p-1}|\le 300 & \forall t,\;p=2,\dots,4
\end{cases}\\[4pt]
&\text{(7) Demand satisfaction}\quad
\sum_{t\in T} w_{t,p}\;\ge\;d_p \quad \forall p\\[4pt]
&\text{(8) 10 % spare‑capacity buffer}\quad
\sum_{t\in T} w_{t,p}\;\le\;0.9\sum_{t\in T} Q_t\,a_{t,p}\quad \forall p\\[4pt]
&\text{(9) Variable domains}\quad
a_{t,p},\,s_{t,p},\,dsh_{t,p}\in\{0,1\},\;
w_{t,p}\ge 0
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(Q_t\) | Truck capacity | `file_0_view_0` | `Q` |
| \(S_t\) | Startup cost | `file_0_view_0` | `S` |
| \(C_t\) | Unit transport cost | `file_0_view_0` | `C` |
| \(d_p\) | Period demand | `file_0_view_0` | `d1`, `d2`, `d3`, `d4` |

All other constants (e.g., 300 kg change limit, 0.9 spare‑capacity factor) are fixed in the model and not sourced from the CSV.