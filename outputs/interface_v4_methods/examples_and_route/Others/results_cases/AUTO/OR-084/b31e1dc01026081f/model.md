##### Parameters

- Let $T = \{1, 2, \ldots, 40\}$ be the set of tasks.
- Let $P = \{1, 2, 3\}$ be the set of CPUs.
- Let $f_1 = 1.33$ GHz, $f_2 = 2$ GHz, $f_3 = 2.66$ GHz be the frequencies of CPUs 1, 2, and 3, respectively.
- Let $I_t$ be the number of instructions (in billions) for task $t \in T$:

\[
\begin{align*}
I_1 &= 1.1 \\
I_2 &= 2.1 \\
I_3 &= 3 \\
I_4 &= 1 \\
I_5 &= 0.7 \\
I_6 &= 5 \\
I_7 &= 3 \\
I_8 &= 3.5 \\
I_9 &= 4.4 \\
I_{10} &= 3.8 \\
I_{11} &= 3.5 \\
I_{12} &= 2.8 \\
I_{13} &= 4.1 \\
I_{14} &= 2.9 \\
I_{15} &= 5.4 \\
I_{16} &= 5.8 \\
I_{17} &= 2.6 \\
I_{18} &= 4.9 \\
I_{19} &= 3.4 \\
I_{20} &= 3.6 \\
I_{21} &= 5.6 \\
I_{22} &= 0.9 \\
I_{23} &= 1 \\
I_{24} &= 0.6 \\
I_{25} &= 5.1 \\
I_{26} &= 4.8 \\
I_{27} &= 5.3 \\
I_{28} &= 5.9 \\
I_{29} &= 4.9 \\
I_{30} &= 3 \\
I_{31} &= 4.8 \\
I_{32} &= 1.2 \\
I_{33} &= 4 \\
I_{34} &= 1.3 \\
I_{35} &= 5.7 \\
I_{36} &= 3.4 \\
I_{37} &= 2.8 \\
I_{38} &= 2 \\
I_{39} &= 4.8 \\
I_{40} &= 3 \\
\end{align*}
\]

- Let $f_p$ be the frequency (in GHz) of CPU $p$:
  - $f_1 = 1.33$
  - $f_2 = 2$
  - $f_3 = 2.66$

##### Decision Variables

- $x_{tp} \in \{0,1\}$: $x_{tp} = 1$ if task $t$ is assigned to CPU $p$, $0$ otherwise.
- $C_{\max} \geq 0$: the makespan (completion time of the last task).

##### Objective Function

\[
\min C_{\max}
\]

##### Constraints

1. **Each task is assigned to exactly one CPU:**

\[
\sum_{p=1}^3 x_{tp} = 1 \quad \forall t \in T
\]

2. **Each CPU can process only one task at a time (sequentially), so the total processing time on each CPU cannot exceed $C_{\max}$:**

\[
\sum_{t=1}^{40} \frac{I_t}{f_p} x_{tp} \leq C_{\max} \quad \forall p \in \{1,2,3\}
\]

3. **Variable domains:**

\[
x_{tp} \in \{0,1\} \quad \forall t \in T, \; p \in \{1,2,3\}
\]
\[
C_{\max} \geq 0
\]

##### Retrieved Information

{
  "tasks": {
    "1": 1.1,
    "2": 2.1,
    "3": 3,
    "4": 1,
    "5": 0.7,
    "6": 5,
    "7": 3,
    "8": 3.5,
    "9": 4.4,
    "10": 3.8,
    "11": 3.5,
    "12": 2.8,
    "13": 4.1,
    "14": 2.9,
    "15": 5.4,
    "16": 5.8,
    "17": 2.6,
    "18": 4.9,
    "19": 3.4,
    "20": 3.6,
    "21": 5.6,
    "22": 0.9,
    "23": 1,
    "24": 0.6,
    "25": 5.1,
    "26": 4.8,
    "27": 5.3,
    "28": 5.9,
    "29": 4.9,
    "30": 3,
    "31": 4.8,
    "32": 1.2,
    "33": 4,
    "34": 1.3,
    "35": 5.7,
    "36": 3.4,
    "37": 2.8,
    "38": 2,
    "39": 4.8,
    "40": 3
  },
  "cpus": {
    "1": 1.33,
    "2": 2,
    "3": 2.66
  }
}