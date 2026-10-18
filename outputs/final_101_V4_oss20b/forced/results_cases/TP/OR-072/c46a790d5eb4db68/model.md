##### Decision Variables

$x_{t}\geq0$: number of drivers who start their 4‑hour shift at the beginning of time period $t$, for $t=1,\dots ,21$ (integer).

##### Objective Function

$\displaystyle \min \sum_{t=1}^{21}x_{t}$

##### Constraints

1. **Coverage of period 1**  
$\displaystyle x_{1}\;\geq\;20$

2. **Coverage of period 2**  
$\displaystyle x_{1}+x_{2}\;\geq\;18$

3. **Coverage of period 3**  
$\displaystyle x_{1}+x_{2}+x_{3}\;\geq\;15$

4. **Coverage of period 4**  
$\displaystyle x_{1}+x_{2}+x_{3}+x_{4}\;\geq\;15$

5. **Coverage of period 5**  
$\displaystyle x_{2}+x_{3}+x_{4}+x_{5}\;\geq\;20$

6. **Coverage of period 6**  
$\displaystyle x_{3}+x_{4}+x_{5}+x_{6}\;\geq\;30$

7. **Coverage of period 7**  
$\displaystyle x_{4}+x_{5}+x_{6}+x_{7}\;\geq\;60$

8. **Coverage of period 8**  
$\displaystyle x_{5}+x_{6}+x_{7}+x_{8}\;\geq\;70$

9. **Coverage of period 9**  
$\displaystyle x_{6}+x_{7}+x_{8}+x_{9}\;\geq\;50$

10. **Coverage of period 10**  
$\displaystyle x_{7}+x_{8}+x_{9}+x_{10}\;\geq\;55$

11. **Coverage of period 11**  
$\displaystyle x_{8}+x_{9}+x_{10}+x_{11}\;\geq\;65$

12. **Coverage of period 12**  
$\displaystyle x_{9}+x_{10}+x_{11}+x_{12}\;\geq\;75$

13. **Coverage of period 13**  
$\displaystyle x_{10}+x_{11}+x_{12}+x_{13}\;\geq\;80$

14. **Coverage of period 14**  
$\displaystyle x_{11}+x_{12}+x_{13}+x_{14}\;\geq\;70$

15. **Coverage of period 15**  
$\displaystyle x_{12}+x_{13}+x_{14}+x_{15}\;\geq\;60$

16. **Coverage of period 16**  
$\displaystyle x_{13}+x_{14}+x_{15}+x_{16}\;\geq\;55$

17. **Coverage of period 17**  
$\displaystyle x_{14}+x_{15}+x_{16}+x_{17}\;\geq\;60$

18. **Coverage of period 18**  
$\displaystyle x_{15}+x_{16}+x_{17}+x_{18}\;\geq\;75$

19. **Coverage of period 19**  
$\displaystyle x_{16}+x_{17}+x_{18}+x_{19}\;\geq\;85$

20. **Coverage of period 20**  
$\displaystyle x_{17}+x_{18}+x_{19}+x_{20}\;\geq\;70$

21. **Coverage of period 21**  
$\displaystyle x_{18}+x_{19}+x_{20}+x_{21}\;\geq\;50$

22. **Coverage of period 22**  
$\displaystyle x_{19}+x_{20}+x_{21}\;\geq\;40$

23. **Coverage of period 23**  
$\displaystyle x_{20}+x_{21}\;\geq\;35$

24. **Coverage of period 24**  
$\displaystyle x_{21}\;\geq\;25$

##### Non‑negativity and integrality

$x_{t}\in\mathbb{Z}_{+}$ for $t=1,\dots ,21$.

##### Retrieved Information

```json
{
  "periods": [
    {"t":1,"required":20},
    {"t":2,"required":18},
    {"t":3,"required":15},
    {"t":4,"required":15},
    {"t":5,"required":20},
    {"t":6,"required":30},
    {"t":7,"required":60},
    {"t":8,"required":70},
    {"t":9,"required":50},
    {"t":10,"required":55},
    {"t":11,"required":65},
    {"t":12,"required":75},
    {"t":13,"required":80},
    {"t":14,"required":70},
    {"t":15,"required":60},
    {"t":16,"required":55},
    {"t":17,"required":60},
    {"t":18,"required":75},
    {"t":19,"required":85},
    {"t":20,"required":70},
    {"t":21,"required":50},
    {"t":22,"required":40},
    {"t":23,"required":35},
    {"t":24,"required":25}
  ]
}
```