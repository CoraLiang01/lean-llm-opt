Let $x_i$ denote the monthly production quantity of product $i$ ($i \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$). All $x_i \geq 0$ and continuous.

Let $a_{di}$ be the processing time required by product $i$ on device $d$ (from **device_time.csv**), and $c_d$ be the monthly capacity of device $d$ (from **monthly_device_capacity.csv**). Let $p_i$ be the unit profit of product $i$ (from **unit_product_profits.csv**).

**Objective:**
\[
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
\]
where $p_i$ is as follows (partial list for illustration, full list from data):

\[
\begin{align*}
p_{\text{P1}} &= 28.55 \\
p_{\text{P2}} &= 12.78 \\
p_{\text{P3}} &= 45.21 \\
&\vdots \\
p_{\text{P111}} &= 9.99 \\
\end{align*}
\]

**Constraints:**

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:
\[
\sum_{i=\text{P1}}^{\text{P111}} a_{di} x_i \leq c_d
\]
where $a_{di}$ and $c_d$ are as follows (all values from the data):

- For device A:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{A},i} x_i \leq 3500
  \]
  with $a_{\text{A},i}$ as in the row for Device A in **device_time.csv** (e.g., $a_{\text{A},\text{P1}} = 8.1$, $a_{\text{A},\text{P2}} = 2.5$, ..., $a_{\text{A},\text{P111}} = 8.5$).

- For device B:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{B},i} x_i \leq 4200
  \]
  with $a_{\text{B},i}$ as in the row for Device B.

- For device C:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{C},i} x_i \leq 4500
  \]
  with $a_{\text{C},i}$ as in the row for Device C.

- For device D:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{D},i} x_i \leq 2800
  \]
  with $a_{\text{D},i}$ as in the row for Device D.

- For device E:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{E},i} x_i \leq 3300
  \]
  with $a_{\text{E},i}$ as in the row for Device E.

- For device F:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{F},i} x_i \leq 3800
  \]
  with $a_{\text{F},i}$ as in the row for Device F.

- For device G:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{G},i} x_i \leq 4100
  \]
  with $a_{\text{G},i}$ as in the row for Device G.

- For device H:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{H},i} x_i \leq 3900
  \]
  with $a_{\text{H},i}$ as in the row for Device H.

- For device I:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{I},i} x_i \leq 4800
  \]
  with $a_{\text{I},i}$ as in the row for Device I.

- For device J:
  \[
  \sum_{i=\text{P1}}^{\text{P111}} a_{\text{J},i} x_i \leq 3100
  \]
  with $a_{\text{J},i}$ as in the row for Device J.

**Variable domains:**
\[
x_i \geq 0 \quad \text{and continuous} \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\]

---

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{i=\text{P1}}^{\text{P111}} p_i x_i \\
\text{s.t.} \quad & \sum_{i=\text{P1}}^{\text{P111}} a_{\text{A},i} x_i \leq 3500 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{B},i} x_i \leq 4200 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{C},i} x_i \leq 4500 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{D},i} x_i \leq 2800 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{E},i} x_i \leq 3300 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{F},i} x_i \leq 3800 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{G},i} x_i \leq 4100 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{H},i} x_i \leq 3900 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{I},i} x_i \leq 4800 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{\text{J},i} x_i \leq 3100 \\
& x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
\]

where:
- $p_i$ is the Unit_Profit for product $i$ from **unit_product_profits.csv**,
- $a_{d,i}$ is the processing time for product $i$ on device $d$ from **device_time.csv**,
- $c_d$ is the Monthly_Capacity for device $d$ from **monthly_device_capacity.csv**.

All coefficients and identifiers are as in the original data.