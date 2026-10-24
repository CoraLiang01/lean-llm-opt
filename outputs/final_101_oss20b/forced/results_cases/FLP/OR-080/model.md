**Mathematical Model**

Let  

* \(I=\{1,\dots ,10\}\) – set of trucks (truck\_id).  
* \(T=\{1,2,3,4\}\) – set of time periods.

**Parameters**

\[
\begin{aligned}
Q_i &\quad\text{maximum capacity of truck }i,\\
S_i &\quad\text{startup cost of truck }i,\\
C_i &\quad\text{unit transportation cost of truck }i,\\
D_t &\quad\text{customer demand in period }t .
\end{aligned}
\]

**Decision Variables**

\[
\begin{aligned}
x_{i,t}&\ge 0 &&\text{weight transported by truck }i\text{ in period }t,\\
y_{i,t}&\in\{0,1\} &&\text{truck }i\text{ is active in period }t,\\
z_{i,t}&\in\{0,1\} &&\text{truck }i\text{ starts in period }t .
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\sum_{t\in T}\bigl(S_i\,z_{i,t}+C_i\,x_{i,t}\bigr)
\]

**Constraints**

1. **Initial state**  
   \[
   y_{i,0}=0 \qquad \forall i\in I
   \]

2. **Startup–activation link**  
   \[
   y_{i,1}=z_{i,1}\qquad\forall i\in I
   \]
   \[
   y_{i,t}\ge z_{i,t}\qquad\forall i\in I,\;t=2,\dots ,4
   \]
   \[
   y_{i,t}\le y_{i,t-1}+z_{i,t}\qquad\forall i\in I,\;t=2,\dots ,4
   \]

3. **Minimum two‑period activity**  
   \[
   z_{i,1}\le y_{i,2},\quad z_{i,2}\le y_{i,3},\quad z_{i,3}\le y_{i,4}\qquad\forall i\in I
   \]
   \[
   z_{i,4}=0\qquad\forall i\in I
   \]

4. **Shutdown rule**  
   \[
   y_{i,t-1}-y_{i,t}\le 1-y_{i,t+1}\qquad\forall i\in I,\;t=2,3
   \]

5. **Restart prohibition after shutdown**  
   \[
   y_{i,t-1}-y_{i,t}\le 1-z_{i,t+1}\qquad\forall i\in I,\;t=2,3
   \]

6. **Capacity limits**  
   \[
   x_{i,t}\le Q_i\,y_{i,t}\qquad\forall i\in I,\;t\in T
   \]

7. **Weight‑change limit**  
   \[
   x_{i,t}-x_{i,t-1}\le 300,\qquad
   x_{i,t-1}-x_{i,t}\le 300\qquad\forall i\in I,\;t=2,3,4
   \]

8. **Demand satisfaction**  
   \[
   \sum_{i\in I}x_{i,t}\;\ge\;D_t\qquad\forall t\in T
   \]

9. **Spare‑capacity buffer**  
   \[
   \sum_{i\in I}x_{i,t}\;\le\;0.9\sum_{i\in I}Q_i\,y_{i,t}\qquad\forall t\in T
   \]

10. **Variable domains**  
    \[
    x_{i,t}\ge 0,\; y_{i,t}\in\{0,1\},\; z_{i,t}\in\{0,1\}\qquad\forall i\in I,\;t\in T
    \]

---

### Data Mapping

| Parameter | Source Table | Column(s) | Mapping Rule |
|-----------|--------------|-----------|--------------|
| \(I\) | `file_0_view_0` | `truck_id` | Each distinct `truck_id` defines an index \(i\). |
| \(Q_i\) | `file_0_view_0` | `Q` | `Q_i` = value of column `Q` for truck \(i\). |
| \(S_i\) | `file_0_view_0` | `S` | `S_i` = value of column `S` for truck \(i\). |
| \(C_i\) | `file_0_view_0` | `C` | `C_i` = value of column `C` for truck \(i\). |
| \(D_t\) | `file_0_view_0` | `d1`–`d4` | `D_1` = `d1`, `D_2` = `d2`, `D_3` = `d3`, `D_4` = `d4`. |

All other data (e.g., the 10 % spare‑capacity buffer, the 300 kg change limit, and the period count) are constants defined in the model, not read from the CSV.