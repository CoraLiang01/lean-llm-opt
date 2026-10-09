Let $x_p \geq 0$ be the continuous production quantity of product $p$ for each $p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$.

**Parameters:**

- Unit profits $u_p$ for each product $p$ (from unit_product_profits.csv)
- Processing times $a_{dp}$: time required by product $p$ on device $d$ (from device_time.csv)
- Device monthly capacities $c_d$ for each device $d$ (from monthly_device_capacity.csv)

**Objective:**
\[
\max \sum_{p} u_p \cdot x_p
\]
where $u_p$ is as follows (in source order):

- P1: 28.55
- P2: 12.78
- P3: 45.21
- P4: 18.92
- P5: 33.47
- P6: 8.64
- P7: 25.88
- P8: 40.15
- P9: 14.39
- P10: 37.62
- P11: 10.25
- P12: 29.81
- P13: 48.77
- P14: 16.53
- P15: 42.09
- P16: 9.11
- P17: 22.46
- P18: 36.73
- P19: 11.99
- P20: 31.28
- P21: 7.45
- P22: 27.6
- P23: 43.91
- P24: 15.82
- P25: 39.36
- P26: 6.23
- P27: 20.71
- P28: 34.98
- P29: 13.24
- P30: 32.53
- P31: 5.1
- P32: 24.18
- P33: 47.85
- P34: 19.3
- P35: 41.44
- P36: 11.02
- P37: 26.57
- P38: 38.2
- P39: 17.76
- P40: 46.12
- P41: 9.89
- P42: 21.34
- P43: 33.86
- P44: 14.88
- P45: 30.41
- P46: 6.5
- P47: 28.93
- P48: 49.2
- P49: 18.15
- P50: 44.78
- P51: 10.57
- P52: 23.69
- P53: 35.51
- P54: 16.03
- P55: 38.83
- P56: 5.88
- P57: 29.45
- P58: 42.33
- P59: 12.41
- P60: 37.06
- P61: 7.99
- P62: 21.9
- P63: 46.99
- P64: 19.87
- P65: 40.7
- P66: 8.34
- P67: 26.11
- P68: 39.54
- P69: 14.07
- P70: 35.79
- P71: 6.92
- P72: 23.03
- P73: 45.56
- P74: 17.2
- P75: 43.27
- P76: 9.52
- P77: 28.08
- P78: 41.1
- P79: 15.46
- P80: 34.3
- P81: 5.43
- P82: 20.25
- P83: 48.38
- P84: 18.68
- P85: 40.01
- P86: 11.45
- P87: 25.32
- P88: 37.58
- P89: 13.62
- P90: 31.84
- P91: 7.27
- P92: 24.75
- P93: 49.88
- P94: 16.85
- P95: 42.82
- P96: 10.13
- P97: 27.36
- P98: 36.19
- P99: 12.8
- P100: 30.09
- P101: 6.07
- P102: 22.67
- P103: 44.42
- P104: 19.54
- P105: 47.41
- P106: 8.78
- P107: 29.97
- P108: 38.96
- P109: 15.11
- P110: 33.05
- P111: 9.99

**Constraints:**

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:

\[
\sum_{p} a_{dp} \cdot x_p \leq c_d
\]

where $a_{dp}$ is the processing time required by product $p$ on device $d$ (from device_time.csv, see below), and $c_d$ is the monthly capacity for device $d$:

- A: 3500
- B: 4200
- C: 4500
- D: 2800
- E: 3300
- F: 3800
- G: 4100
- H: 3900
- I: 4800
- J: 3100

**Device processing times $a_{dp}$** (for each device $d$ and product $p$):

- For Device A: (P1: 8.1, P2: 2.5, ..., P111: 8.5)
- For Device B: (P1: 10.5, P2: 5.2, ..., P111: 8.7)
- For Device C: (P1: 2.1, P2: 13.4, ..., P111: 2.9)
- For Device D: (P1: 5.8, P2: 1.2, ..., P111: 8.3)
- For Device E: (P1: 9.3, P2: 4.1, ..., P111: 6.7)
- For Device F: (P1: 3.8, P2: 14.2, ..., P111: 8.1)
- For Device G: (P1: 7.2, P2: 2.8, ..., P111: 5.9)
- For Device H: (P1: 11.7, P2: 6.3, ..., P111: 1.4)
- For Device I: (P1: 1.1, P2: 11.3, ..., P111: 7.3)
- For Device J: (P1: 4.6, P2: 0.2, ..., P111: 12.4)

(Use the full table as retrieved above for all $a_{dp}$.)

**Variable domains:**
\[
x_p \geq 0 \quad \text{(continuous)}, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\]

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{p=\text{P1}}^{\text{P111}} u_p \cdot x_p \\
\text{s.t.} \quad & \sum_{p=\text{P1}}^{\text{P111}} a_{dp} \cdot x_p \leq c_d, \quad \forall d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\} \\
& x_p \geq 0, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
\]

Where all coefficients and identifiers are as listed above and in the retrieved data.