##### Parameters

Let the day be divided into $N=48$ half-hour periods, indexed by $t=1,2,\ldots,48$, corresponding to the time intervals:

1. 2:00am - 2:30am
2. 2:30am - 3:00am
3. 3:00am - 3:30am
4. 3:30am - 4:00am
5. 4:00am - 4:30am
6. 4:30am - 5:00am
7. 5:00am - 5:30am
8. 5:30am - 6:00am
9. 6:00am - 6:30am
10. 6:30am - 7:00am
11. 7:00am - 7:30am
12. 7:30am - 8:00am
13. 8:00am - 8:30am
14. 8:30am - 9:00am
15. 9:00am - 9:30am
16. 9:30am - 10:00am
17. 10:00am - 10:30am
18. 10:30am - 11:00am
19. 11:00am - 11:30am
20. 11:30am - 12:00pm
21. 12:00pm - 12:30pm
22. 12:30pm - 1:00pm
23. 1:00pm - 1:30pm
24. 1:30pm - 2:00pm
25. 2:00pm - 2:30pm
26. 2:30pm - 3:00pm
27. 3:00pm - 3:30pm
28. 3:30pm - 4:00pm
29. 4:00pm - 4:30pm
30. 4:30pm - 5:00pm
31. 5:00pm - 5:30pm
32. 5:30pm - 6:00pm
33. 6:00pm - 6:30pm
34. 6:30pm - 7:00pm
35. 7:00pm - 7:30pm
36. 7:30pm - 8:00pm
37. 8:00pm - 8:30pm
38. 8:30pm - 9:00pm
39. 9:00pm - 9:30pm
40. 9:30pm - 10:00pm
41. 10:00pm - 10:30pm
42. 10:30pm - 11:00pm
43. 11:00pm - 11:30pm
44. 11:30pm - 12:00am
45. 12:00am - 12:30am
46. 12:30am - 1:00am
47. 1:00am - 1:30am
48. 1:30am - 2:00am

Let $r_t$ be the required minimum number of waitstaff for period $t$:

\[
\begin{align*}
r_1 &= 2 \\
r_2 &= 3 \\
r_3 &= 4 \\
r_4 &= 6 \\
r_5 &= 5 \\
r_6 &= 4 \\
r_7 &= 5 \\
r_8 &= 6 \\
r_9 &= 7 \\
r_{10} &= 8 \\
r_{11} &= 9 \\
r_{12} &= 9 \\
r_{13} &= 8 \\
r_{14} &= 8 \\
r_{15} &= 9 \\
r_{16} &= 9 \\
r_{17} &= 10 \\
r_{18} &= 12 \\
r_{19} &= 11 \\
r_{20} &= 11 \\
r_{21} &= 12 \\
r_{22} &= 11 \\
r_{23} &= 10 \\
r_{24} &= 9 \\
r_{25} &= 8 \\
r_{26} &= 7 \\
r_{27} &= 6 \\
r_{28} &= 5 \\
r_{29} &= 5 \\
r_{30} &= 6 \\
r_{31} &= 7 \\
r_{32} &= 8 \\
r_{33} &= 9 \\
r_{34} &= 10 \\
r_{35} &= 9 \\
r_{36} &= 8 \\
r_{37} &= 7 \\
r_{38} &= 6 \\
r_{39} &= 5 \\
r_{40} &= 4 \\
r_{41} &= 4 \\
r_{42} &= 3 \\
r_{43} &= 3 \\
r_{44} &= 3 \\
r_{45} &= 3 \\
r_{46} &= 4 \\
r_{47} &= 4 \\
r_{48} &= 4 \\
\end{align*}
\]

##### Decision Variables

Let $x_t$ be the number of waitstaff who start their 8-hour (16-period) shift at period $t$, for $t=1,\ldots,48$.

$x_t \geq 0$ and integer, for all $t$.

##### Objective Function

\[
\min \sum_{t=1}^{48} x_t
\]

##### Constraints

For each period $s=1,\ldots,48$, the total number of waitstaff on duty must be at least $r_s$:

\[
\sum_{k=0}^{15} x_{(s-k-1 \bmod 48) + 1} \geq r_s, \quad \forall s=1,\ldots,48
\]

That is, for each period $s$, sum the number of staff who started in the previous 16 periods (including $s$ itself, since a shift covers 16 periods), wrapping around the 48-period day.

##### Variable Domains

\[
x_t \in \mathbb{Z}_{\geq 0}, \quad \forall t=1,\ldots,48
\]

##### Complete Model

\[
\begin{align*}
\min \quad & \sum_{t=1}^{48} x_t \\
\text{s.t.} \quad & \sum_{k=0}^{15} x_{(s-k-1 \bmod 48) + 1} \geq r_s, \quad \forall s=1,\ldots,48 \\
& x_t \in \mathbb{Z}_{\geq 0}, \quad \forall t=1,\ldots,48
\end{align*}
\]

Where $r_s$ is as listed above for each period $s$.

##### Retrieved Information

{
  "periods": [
    {"Time": "2:00am - 2:30am", "Requirement": 2},
    {"Time": "2:30am - 3:00am", "Requirement": 3},
    {"Time": "3:00am - 3:30am", "Requirement": 4},
    {"Time": "3:30am - 4:00am", "Requirement": 6},
    {"Time": "4:00am - 4:30am", "Requirement": 5},
    {"Time": "4:30am - 5:00am", "Requirement": 4},
    {"Time": "5:00am - 5:30am", "Requirement": 5},
    {"Time": "5:30am - 6:00am", "Requirement": 6},
    {"Time": "6:00am - 6:30am", "Requirement": 7},
    {"Time": "6:30am - 7:00am", "Requirement": 8},
    {"Time": "7:00am - 7:30am", "Requirement": 9},
    {"Time": "7:30am - 8:00am", "Requirement": 9},
    {"Time": "8:00am - 8:30am", "Requirement": 8},
    {"Time": "8:30am - 9:00am", "Requirement": 8},
    {"Time": "9:00am - 9:30am", "Requirement": 9},
    {"Time": "9:30am - 10:00am", "Requirement": 9},
    {"Time": "10:00am - 10:30am", "Requirement": 10},
    {"Time": "10:30am - 11:00am", "Requirement": 12},
    {"Time": "11:00am - 11:30am", "Requirement": 11},
    {"Time": "11:30am - 12:00pm", "Requirement": 11},
    {"Time": "12:00pm - 12:30pm", "Requirement": 12},
    {"Time": "12:30pm - 1:00pm", "Requirement": 11},
    {"Time": "1:00pm - 1:30pm", "Requirement": 10},
    {"Time": "1:30pm - 2:00pm", "Requirement": 9},
    {"Time": "2:00pm - 2:30pm", "Requirement": 8},
    {"Time": "2:30pm - 3:00pm", "Requirement": 7},
    {"Time": "3:00pm - 3:30pm", "Requirement": 6},
    {"Time": "3:30pm - 4:00pm", "Requirement": 5},
    {"Time": "4:00pm - 4:30pm", "Requirement": 5},
    {"Time": "4:30pm - 5:00pm", "Requirement": 6},
    {"Time": "5:00pm - 5:30pm", "Requirement": 7},
    {"Time": "5:30pm - 6:00pm", "Requirement": 8},
    {"Time": "6:00pm - 6:30pm", "Requirement": 9},
    {"Time": "6:30pm - 7:00pm", "Requirement": 10},
    {"Time": "7:00pm - 7:30pm", "Requirement": 9},
    {"Time": "7:30pm - 8:00pm", "Requirement": 8},
    {"Time": "8:00pm - 8:30pm", "Requirement": 7},
    {"Time": "8:30pm - 9:00pm", "Requirement": 6},
    {"Time": "9:00pm - 9:30pm", "Requirement": 5},
    {"Time": "9:30pm - 10:00pm", "Requirement": 4},
    {"Time": "10:00pm - 10:30pm", "Requirement": 4},
    {"Time": "10:30pm - 11:00pm", "Requirement": 3},
    {"Time": "11:00pm - 11:30pm", "Requirement": 3},
    {"Time": "11:30pm - 12:00am", "Requirement": 3},
    {"Time": "12:00am - 12:30am", "Requirement": 3},
    {"Time": "12:30am - 1:00am", "Requirement": 4},
    {"Time": "1:00am - 1:30am", "Requirement": 4},
    {"Time": "1:30am - 2:00am", "Requirement": 4}
  ]
}