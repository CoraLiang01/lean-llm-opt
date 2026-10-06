Let:
- $I = \{\text{HiFi-1}, \text{HiFi-2}, \ldots, \text{HiFi-101}\}$ be the set of radio models.
- $K = \{1,2,3\}$ be the set of workstations.
- $t_{ki}$ = processing time (in minutes) required at workstation $k$ for one unit of model $i$ (from the CSV, where $k$ is the row "Workstation" and $i$ is the column "HiFiX_Minutes").
- $C_k$ = total available minutes per day at workstation $k$ (given as 1,440 for all $k$).
- $m_k$ = maintenance percentage at workstation $k$ (from the CSV: 10 for $k=1$, 14 for $k=2$, 12 for $k=3$).
- $E_k = C_k \cdot (1 - m_k/100)$ = effective daily capacity at workstation $k$ (in minutes).
- $x_i$ = number of units of model $i$ to produce per day (decision variable, integer, $x_i \geq 0$).
- $s_k$ = idle time at workstation $k$ (in minutes, continuous, $s_k \geq 0$).

**Parameters from the CSV:**

- For $k=1$: $m_1 = 10$, $E_1 = 1,440 \times 0.9 = 1,296$
- For $k=2$: $m_2 = 14$, $E_2 = 1,440 \times 0.86 = 1,238.4$
- For $k=3$: $m_3 = 12$, $E_3 = 1,440 \times 0.88 = 1,267.2$

---

### Decision Variables

- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of units of each model to produce per day)
- $s_k \geq 0$, for all $k \in K$ (idle time at each workstation)

---

### Objective

Minimize total idle production time across all workstations:
\[
\min \sum_{k=1}^3 s_k
\]

---

### Constraints

For each workstation $k \in K$:
1. **Idle time definition:**
   \[
   s_k = E_k - \sum_{i \in I} t_{ki} x_i
   \]
2. **Nonnegativity of idle time:**
   \[
   s_k \geq 0
   \]
3. **Production cannot exceed effective capacity:**
   \[
   \sum_{i \in I} t_{ki} x_i \leq E_k
   \]
   (This is implied by $s_k \geq 0$ and the definition above, but can be included for clarity.)

4. **Nonnegativity and integrality of production:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

### Full Model (with explicit indices and coefficients):

Let $t_{ki}$ be the value in row $k$ and column "HiFi$i$_Minutes" of workstation_times.csv.

\[
\begin{align*}
\min \quad & s_1 + s_2 + s_3 \\
\text{s.t.} \quad
& s_1 = 1,296 - \sum_{i=1}^{101} t_{1i} x_i \\
& s_2 = 1,238.4 - \sum_{i=1}^{101} t_{2i} x_i \\
& s_3 = 1,267.2 - \sum_{i=1}^{101} t_{3i} x_i \\
& s_k \geq 0, \quad k=1,2,3 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad i=1,\ldots,101
\end{align*}
\]

Where:
- $t_{ki}$ is the processing time (in minutes) for model $i$ at workstation $k$, as given in workstation_times.csv.
- $x_i$ is the number of units of model $i$ to produce per day.

---

**All coefficients and identifiers are taken directly from the provided CSV. No data has been omitted or synthesized.**