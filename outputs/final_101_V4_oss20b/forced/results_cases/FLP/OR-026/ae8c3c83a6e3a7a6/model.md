##### Decision Variables

$x_i \geq 0$: number of units of Fashion product $i$ that will be fulfilled, for $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i\,x_i$

##### Constraints

1. Demand satisfaction: $\displaystyle \sum_{i\in I} x_i \leq \sum_{i\in I} d_i$ (implicitly satisfied by individual bounds).  
2. Individual demand limits: $\displaystyle x_i \leq d_i,\quad \forall i\in I$.  
3. Inventory limits: $\displaystyle x_i \leq I_i,\quad \forall i\in I$.  
4. Non‑negativity: $\displaystyle x_i \geq 0,\quad \forall i\in I$.

$I=\{1,2,\dots,49\}$ (the 49 Fashion products listed below).

##### Retrieved Information

**Product Names**  
1. Fashion accessories_10.18  
2. Fashion accessories_12.09  
3. Fashion accessories_12.19  
4. Fashion accessories_12.54  
5. Fashion accessories_12.78  
6. Fashion accessories_14.48  
7. Fashion accessories_15.43  
8. Fashion accessories_15.5  
9. Fashion accessories_15.62  
10. Fashion accessories_16.28  
11. Fashion accessories_16.45  
12. Fashion accessories_17.48  
13. Fashion accessories_17.49  
14. Fashion accessories_17.87  
15. Fashion accessories_17.94  
16. Fashion accessories_18.08  
17. Fashion accessories_19.66  
18. Fashion accessories_19.7  
19. Fashion accessories_19.77  
20. Fashion accessories_20.01  
21. Fashion accessories_21.32  
22. Fashion accessories_21.48  
23. Fashion accessories_21.94  
24. Fashion accessories_22.32  
25. Fashion accessories_22.51  
26. Fashion accessories_23.82  
27. Fashion accessories_25.42  
28. Fashion accessories_25.56  
29. Fashion accessories_27.02  
30. Fashion accessories_27.18  
31. Fashion accessories_27.38  
32. Fashion accessories_29.42  
33. Fashion accessories_29.56  
34. Fashion accessories_30.14  
35. Fashion accessories_30.37  
36. Fashion accessories_30.61  
37. Fashion accessories_30.62  
38. Fashion accessories_31.73  
39. Fashion accessories_31.9  
40. Fashion accessories_32.62  
41. Fashion accessories_33.52  
42. Fashion accessories_33.63  
43. Fashion accessories_34.7  
44. Fashion accessories_35.19  
45. Fashion accessories_36.51  
46. Fashion accessories_36.85  
47. Fashion accessories_37.15  
48. Fashion accessories_37.55  
49. Fashion accessories_37.95  

**Demand vector**  
$d=[12,2,10,2,2,5,2,2,11,2,6,9,14,6,8,6,14,2,14,11,2,3,8,10,10,7,11,11,5,3,8,13,7,14,4,2,2,14,2,6,2,2,3,14,12,7,6,13,14]$

**Revenue vector**  
$r=[10.18,12.09,12.19,12.54,12.78,14.48,15.43,15.5,15.62,16.28,16.45,17.48,17.49,17.87,17.94,18.08,19.66,19.7,19.77,20.01,21.32,21.48,21.94,22.32,22.51,23.82,25.42,25.56,27.02,27.18,27.38,29.42,29.56,30.14,30.37,30.61,30.62,31.73,31.9,32.62,33.52,33.63,34.7,35.19,36.51,36.85,37.15,37.55,37.95]$

**Initial inventory vector**  
$I=[80,10,80,10,10,40,10,10,80,10,40,60,100,40,50,40,100,10,100,90,10,20,50,80,70,50,80,70,30,20,60,100,50,100,30,10,10,90,10,40,10,10,20,100,90,50,40,100,100]$