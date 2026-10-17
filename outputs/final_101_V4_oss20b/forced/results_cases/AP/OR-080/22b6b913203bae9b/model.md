**Mathematical Model**

Let  

* \(T=\{1,2,3,4\}\) – periods  
* \(I=\{1,\dots ,10\}\) – trucks  

Parameters (from *parameters.csv*)  

\[
\begin{aligned}
Q_i &\quad\text{maximum capacity of truck }i\\
S_i &\quad\text{startup cost of truck }i\\
C_i &\quad\text{unit transportation cost of truck }i\\
d_t &\quad\text{customer demand in period }t
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_{i,t} &\in\{0,1\}\quad\text{(truck }i\text{ active in period }t)\\
x_{i,t} &\in\{0,1\}\quad\text{(truck }i\text{ started in period }t)\\
z_{i,t} &\in\{0,1\}\quad\text{(truck }i\text{ shut down in period }t)\\
w_{i,t} &\ge 0\quad\text{(weight transported by truck }i\text{ in period }t)
\end{aligned}
\]

---

### Objective  

\[
\min \;\sum_{i\in I}\sum_{t\in T}\Bigl(S_i\,x_{i,t}+C_i\,w_{i,t}\Bigr)
\]

---

### Constraints  

1. **Activation dynamics**  
   \[
   y_{i,1}=x_{i,1}\qquad\forall i\in I
   \]
   \[
   y_{i,t}=y_{i,t-1}+x_{i,t}-z_{i,t}\qquad\forall i\in I,\;t=2,\dots ,4
   \]

2. **Startup limits**  
   \[
   x_{i,t}\le y_{i,t}\qquad\forall i,t
   \]
   \[
   x_{i,t}\le 1-y_{i,t-1}\qquad\forall i,\;t=2,\dots ,4
   \]
   \[
   x_{i,4}=0\qquad\forall i
   \]

3. **Shutdown rules**  
   \[
   z_{i,t}\le 1-y_{i,t}\qquad\forall i,\;t=1,\dots ,4
   \]
   \[
   z_{i,t}\le y_{i,t-1}\qquad\forall i,\;t=2,\dots ,4
   \]
   \[
   y_{i,t+1}\le 1-z_{i,t}\qquad\forall i,\;t=1,2
   \]
   \[
   y_{i,t+2}\le 1-z_{i,t}\qquad\forall i,\;t=1
   \]
   \[
   x_{i,t+2}\le 1-z_{i,t}\qquad\forall i,\;t=1
   \]

4. **Mandatory two‑period activity**  
   \[
   y_{i,t+1}\ge x_{i,t}\qquad\forall i,\;t=1,2,3
   \]

5. **Capacity limits**  
   \[
   0\le w_{i,t}\le Q_i\,y_{i,t}\qquad\forall i,t
   \]

6. **Weight‑change limits**  
   \[
   |w_{i,t}-w_{i,t-1}|\le 300\qquad\forall i,\;t=2,3,4
   \]
   \[
   w_{i,1}\le 300\qquad\forall i
   \]

7. **Demand satisfaction**  
   \[
   \sum_{i\in I} w_{i,t}\;\ge\; d_t\qquad\forall t\in T
   \]

8. **Spare‑capacity buffer**  
   \[
   \sum_{i\in I} w_{i,t}\;\le\;0.9\;\sum_{i\in I} Q_i\,y_{i,t}\qquad\forall t\in T
   \]

---

**Data Mapping**

| Parameter | Source | Column(s) | Notes |
|-----------|--------|-----------|-------|
| \(Q_i\) | *parameters.csv* | `Q` | Truck capacity |
| \(S_i\) | *parameters.csv* | `S` | Startup cost |
| \(C_i\) | *parameters.csv* | `C` | Unit transport cost |
| \(d_t\) | *parameters.csv* | `d1`–`d4` | Customer demand (identical across trucks) |
| \(I\) | *parameters.csv* | `truck_id` | Truck identifiers |
| \(T\) | – | – | Periods 1–4 |

All variables and constraints are defined over the index sets \(I\) and \(T\) as shown above. No numerical values are enumerated; the model references the data mapping for parameter values.