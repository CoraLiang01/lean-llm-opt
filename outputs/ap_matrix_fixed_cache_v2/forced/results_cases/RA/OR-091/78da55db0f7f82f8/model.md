Let $x_i \in \{0,1\}$ for each Operations Research course $i$, where $x_i = 1$ if course $i$ is selected, $0$ otherwise.

Let the set of Operations Research courses (in source order) be:
- C22: Operations Research: Linear Programming (credits: 5, interest points: 95)
- C23: Integer Programming (credits: 5, interest points: 92)
- C24: Stochastic Processes (credits: 4, interest points: 86)
- C25: Simulation Modeling (credits: 4, interest points: 82)
- C26: Network Flows (credits: 4, interest points: 85)
- C27: Queueing Theory (credits: 4, interest points: 80)
- C28: Revenue Management (credits: 4, interest points: 88)

The mathematical model is:

Maximize
$$
95x_{22} + 92x_{23} + 86x_{24} + 82x_{25} + 85x_{26} + 80x_{27} + 88x_{28}
$$

Subject to
$$
5x_{22} + 5x_{23} + 4x_{24} + 4x_{25} + 4x_{26} + 4x_{27} + 4x_{28} \leq 20
$$

$$
x_{22}, x_{23}, x_{24}, x_{25}, x_{26}, x_{27}, x_{28} \in \{0,1\}
$$

Where:
- $x_{22}$: 1 if "Operations Research: Linear Programming" (C22) is selected, 0 otherwise
- $x_{23}$: 1 if "Integer Programming" (C23) is selected, 0 otherwise
- $x_{24}$: 1 if "Stochastic Processes" (C24) is selected, 0 otherwise
- $x_{25}$: 1 if "Simulation Modeling" (C25) is selected, 0 otherwise
- $x_{26}$: 1 if "Network Flows" (C26) is selected, 0 otherwise
- $x_{27}$: 1 if "Queueing Theory" (C27) is selected, 0 otherwise
- $x_{28}$: 1 if "Revenue Management" (C28) is selected, 0 otherwise

All coefficients and course IDs are as retrieved and in source order.