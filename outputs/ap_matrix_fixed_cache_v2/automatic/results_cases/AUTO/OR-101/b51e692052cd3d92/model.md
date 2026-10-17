Let $x_i$ denote the production quantity (in continuous, nonnegative units) of product $i$ ($i \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$).

**Parameters:**

- $p_i$: Unit profit of product $i$ (from unit_product_profits.csv)
- $a_{ji}$: Processing time required by product $i$ on device $j$ (from device_time.csv, $j \in \{\text{A},\ldots,\text{J}\}$)
- $c_j$: Monthly capacity of device $j$ (from monthly_device_capacity.csv)

---

**Objective:**

$$
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
$$

where the unit profits $p_i$ are:

\[
\begin{aligned}
&p_{\text{P1}} = 28.55,\quad p_{\text{P2}} = 12.78,\quad p_{\text{P3}} = 45.21,\quad p_{\text{P4}} = 18.92,\quad p_{\text{P5}} = 33.47,\\
&p_{\text{P6}} = 8.64,\quad p_{\text{P7}} = 25.88,\quad p_{\text{P8}} = 40.15,\quad p_{\text{P9}} = 14.39,\quad p_{\text{P10}} = 37.62,\\
&\ldots\\
&p_{\text{P111}} = 9.99
\end{aligned}
\]

(Full list as in the retrieved data.)

---

**Constraints:**

For each device $j \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:

\[
\sum_{i=\text{P1}}^{\text{P111}} a_{ji} x_i \leq c_j
\]

where the device capacities $c_j$ are:

\[
\begin{aligned}
&c_{\text{A}} = 3500 \\
&c_{\text{B}} = 4200 \\
&c_{\text{C}} = 4500 \\
&c_{\text{D}} = 2800 \\
&c_{\text{E}} = 3300 \\
&c_{\text{F}} = 3800 \\
&c_{\text{G}} = 4100 \\
&c_{\text{H}} = 3900 \\
&c_{\text{I}} = 4800 \\
&c_{\text{J}} = 3100 \\
\end{aligned}
\]

and the processing times $a_{ji}$ are as given in device_time.csv, e.g.:

- For device A: $a_{\text{A},\text{P1}} = 8.1$, $a_{\text{A},\text{P2}} = 2.5$, ..., $a_{\text{A},\text{P111}} = 8.5$
- For device B: $a_{\text{B},\text{P1}} = 10.5$, $a_{\text{B},\text{P2}} = 5.2$, ..., $a_{\text{B},\text{P111}} = 8.7$
- $\ldots$
- For device J: $a_{\text{J},\text{P1}} = 4.6$, $a_{\text{J},\text{P2}} = 0.2$, ..., $a_{\text{J},\text{P111}} = 12.4$

---

**Variable domains:**

\[
x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\]

---

**Complete Model:**

\[
\begin{aligned}
&\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i \\
\text{s.t.} \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{A},i} x_i \leq 3500 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{B},i} x_i \leq 4200 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{C},i} x_i \leq 4500 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{D},i} x_i \leq 2800 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{E},i} x_i \leq 3300 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{F},i} x_i \leq 3800 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{G},i} x_i \leq 4100 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{H},i} x_i \leq 3900 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{I},i} x_i \leq 4800 \\
&\sum_{i=\text{P1}}^{\text{P111}} a_{\text{J},i} x_i \leq 3100 \\
&x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\end{aligned}
\]

where all coefficients $p_i$ and $a_{ji}$ are as given in the retrieved data above, and all device and product identifiers are preserved as in the source files.