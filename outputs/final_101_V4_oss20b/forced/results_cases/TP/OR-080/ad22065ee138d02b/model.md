**Mathematical model**

Let  

* \(I=\{1,\dots ,10\}\) – set of trucks  
* \(T=\{1,2,3,4\}\) – set of time periods  

Parameters (all read from *parameters.csv*)  

\[
\begin{aligned}
Q_i &\;=\;\text{maximum capacity of truck }i,\\
S_i &\;=\;\text{startup cost of truck }i,\\
C_i &\;=\;\text{unit transportation cost of truck }i,\\
d_t &\;=\;\text{customer demand in period }t .
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
a_{i,t} &\in\{0,1\}\quad &&\text{truck }i\text{ active in period }t,\\
s_{i,t} &\in\{0,1\}\quad &&\text{truck }i\text{ started in period }t,\\
d_{i,t} &\in\{0,1\}\quad &&\text{truck }i\text{ shut down in period }t,\\
w_{i,t} &\ge 0 \quad &&\text{weight transported by truck }i\text{ in period }t .
\end{aligned}
\]

---

### Objective  

\[
\min \;\sum_{i\in I}\sum_{t\in T}\Bigl(S_i\,s_{i,t}+C_i\,w_{i,t}\Bigr)
\]

---

### Constraints  

1. **Initial state**  
   \[
   a_{i,1}=s_{i,1}\qquad\forall i\in I
   \]

2. **State evolution** (for \(t=2,\dots ,4\))  
   \[
   a_{i,t}=a_{i,t-1}+s_{i,t}-d_{i,t}\qquad\forall i\in I,\;t\in\{2,3,4\}
   \]

3. **No start in period 4**  
   \[
   s_{i,4}=0\qquad\forall i\in I
   \]

4. **Minimum‑up time 2** – if a truck starts in period \(t\) it must be active in \(t+1\)  
   \[
   a_{i,t+1}\;\ge\;s_{i,t}\qquad\forall i\in I,\;t\in\{1,2,3\}
   \]

5. **Shutdown rules** – if a truck shuts down in period \(t\) it must stay inactive in \(t+1\) and \(t+2\)  
   \[
   \begin{aligned}
   a_{i,t+1}&\le 1-d_{i,t}\qquad &\forall i\in I,\;t\in\{2,3\}\\
   a_{i,t+2}&\le 1-d_{i,t}\qquad &\forall i\in I,\;t=2
   \end{aligned}
   \]

6. **Restart after shutdown** – a truck cannot start in the two periods following a shutdown  
   \[
   \begin{aligned}
   s_{i,t+1}&\le 1-d_{i,t}\qquad &\forall i\in I,\;t\in\{2,3\}
   \end{aligned}
   \]

7. **Capacity limits**  
   \[
   0\le w_{i,t}\le Q_i\,a_{i,t}\qquad\forall i\in I,\;t\in T
   \]

8. **Weight‑change limits**  
   \[
   \begin{aligned}
   w_{i,1}&\le 300\,a_{i,1}\qquad &\forall i\in I\\
   w_{i,t}-w_{i,t-1}&\le 300\qquad &\forall i\in I,\;t\in\{2,3,4\}\\
   w_{i,t-1}-w_{i,t}&\le 300\qquad &\forall i\in I,\;t\in\{2,3,4\}
   \end{aligned}
   \]

9. **Demand satisfaction**  
   \[
   \sum_{i\in I} w_{i,t}\;\ge\; d_t\qquad\forall t\in T
   \]

10. **Spare‑capacity buffer (90 %)**  
    \[
    \sum_{i\in I} w_{i,t}\;\le\;0.9\;\sum_{i\in I} Q_i\,a_{i,t}\qquad\forall t\in T
    \]

11. **Binary and non‑negativity**  
    \[
    a_{i,t},\,s_{i,t},\,d_{i,t}\in\{0,1\},\qquad w_{i,t}\ge 0
    \]

---

## Data‑mapping rules

| Parameter / Variable | Source table | Source column | Notes |
|----------------------|--------------|---------------|-------|
| \(Q_i\) | `file_0_view_0` | `Q` | Truck maximum capacity |
| \(S_i\) | `file_0_view_0` | `S` | Truck startup cost |
| \(C_i\) | `file_0_view_0` | `C` | Truck unit transportation cost |
| \(d_t\) | `file_0_view_0` | `d1`–`d4` | Customer demand for periods 1–4 |
| \(a_{i,t}\) | – | – | Binary active flag |
| \(s_{i,t}\) | – | – | Binary start flag |
| \(d_{i,t}\) | – | – | Binary shutdown flag |
| \(w_{i,t}\) | – | – | Continuous transported weight |

All indices \(i\) and \(t\) are taken from the sets \(I=\{1,\dots,10\}\) and \(T=\{1,2,3,4\}\) defined by the rows and period columns of the source file. No additional data are required.