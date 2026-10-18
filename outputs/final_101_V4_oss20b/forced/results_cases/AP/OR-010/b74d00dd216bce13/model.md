##### Objective Function: 

$ \quad \quad \max \sum_{i=1}^{71} r_{i} x_{i} $

##### Constraints

###### 1. Demand Constraints: 

$ x_{i} \leq d_{i} \quad \forall i \in \{1,\dots,71\} $

###### 2. Inventory Constraints: 

$ x_{i} \leq I_{i} \quad \forall i \in \{1,\dots,71\} $

###### 3. Non‑negativity Constraints: 

$ x_{i} \geq 0 \quad \forall i \in \{1,\dots,71\} $

###### 4. Integer Constraints: 

$ x_{i} \in \mathbb{Z} \quad \forall i \in \{1,\dots,71\} $

###### 5. Decision Variable Definition: 

$ x_{i} $ denotes the quantity of product $i$ fulfilled.

###### 6. Parameter Definitions: 

$ r_{i} $ – revenue per unit of product $i$  
$ d_{i} $ – deterministic demand for product $i$  
$ I_{i} $ – initial inventory available for product $i$

###### 7. Index Set: 

$ i \in \{1,\dots,71\} $

###### 8. Variable Bounds: 

$ 0 \leq x_{i} \leq \min(d_{i}, I_{i}) \quad \forall i \in \{1,\dots,71\} $

###### 9. Objective Coefficients: 

$ r_{i} $ values are taken from the “Revenue” column.

###### 10. Demand Coefficients: 

$ d_{i} $ values are taken from the “Demand” column.

###### 11. Inventory Coefficients: 

$ I_{i} $ values are taken from the “Initial Inventory” column.

###### 12. Product List: 

$ i \in \{1,\dots,71\} $ corresponds to the following products in order:

1. Adams Group_service  
2. Anderson-Leach_against  
3. Anderson-Valdez_somebody  
4. Anderson-White_son  
5. Andrews LLC_matter  
6. Andrews PLC_enter  
7. Andrews-Martin_build  
8. Arias-Mendoza_life  
9. Bennett and Sons_down  
10. Bennett, Foster and Moreno_enter  
11. Fernandez, Long and Nelson_member  
12. Fernandez-Fischer_million  
13. Ferrell Inc_prove  
14. Fields, Christensen and Daniels_nor  
15. Figueroa LLC_involve  
16. Finley Group_itself  
17. Fisher Ltd_speak  
18. Fisher-Marshall_cover  
19. Fuller-Torres_behavior  
20. Fuller-Walters_radio  
21. Gallagher-Campbell_dark  
22. Gallagher-Kirby_talk  
23. Gallagher-Parker_recognize  
24. Gonzalez PLC_task  
25. Gonzalez, Coleman and Le_heavy  
26. Gonzalez, Lowe and Robinson_education  
27. Gonzalez-Horn_light  
28. Good, Davis and Smith_station  
29. Goodman, Hughes and White_realize  
30. Goodwin PLC_bad  
31. Gordon PLC_detail  
32. Graham LLC_stage  
33. Graham Ltd_marriage  
34. Graham-Swanson_message  
35. Grant, Cross and Bennett_religious  
36. Grant, Mcdonald and Watson_owner  
37. Graves, Turner and Crawford_wait  
38. Gray, Smith and Barnes_much  
39. Green Inc_direction  
40. Green-Rogers_could  
41. Greene-Baxter_them  
42. Greer and Sons_keep  
43. Griffin, Boyle and Dawson_anything  
44. Harper Inc_well  
45. Harper Ltd_off  
46. Harrington, Sosa and Mccarty_coach  
47. Harris Group_different  
48. Harris and Sons_audience  
49. Harris and Sons_fear  
50. Harris, Hamilton and Rose_contain  
51. Harris, Stevens and Hall_answer  
52. Harris-Bell_painting  
53. Harris-Melton_often  
54. Harris-Rogers_after  
55. Hart Group_cold  
56. Hutchinson, Roberts and Mcbride_among  
57. Ibarra, Jackson and Potter_upon  
58. Jackson Inc_anyone  
59. Jackson LLC_report  
60. Jackson, Collier and Barber_result  
61. Jackson, White and Brown_expect  
62. Jackson-Carroll_six  
63. Lee-Jones_soldier  
64. Lewis, Sanchez and Turner_point  
65. Lin LLC_performance  
66. Lindsey, Avila and Brown_candidate  
67. Lindsey-Strickland_modern  
68. Little-Perez_choice  
69. Lloyd, Stone and Mcguire_national  
70. Long, Hughes and Gallegos_receive  
71. Lopez PLC_analysis

###### 13. Parameter Arrays

{
  "revenue": [
    1003.56, 549.92, 169.7, 1285.81, 1269.71, 805.82, 1134.19, 1479.23, 139.14, 518.67,
    1088.57, 371.56, 956.78, 1283.7, 1184.54, 377.8, 701.97, 875.91, 951.85, 869.95,
    887.06, 1318.33, 448.61, 167.99, 1068.1, 872.16, 483.26, 1279.24, 894.34, 367.91,
    178.22, 1030.8, 479.44, 973.62, 1211.35, 467.93, 678.72, 478.36, 1196.95, 1118.63,
    972.1, 338.6, 178.52, 992.14, 498.77, 965.34, 321.11, 287.36, 462.26, 641.22,
    1136.87, 1385.88, 867.3, 1175.92, 153.61, 753.55, 245.81, 507.13, 513.64, 1316.82,
    1409.21, 1237.37, 1042.66, 156.02, 728.62, 971.93, 1128.07, 256.42, 1064.72, 348.43,
    1183.94
  ],
  "demand": [
    10, 6, 75, 121, 130, 112, 125, 54, 75, 102,
    48, 4, 125, 9, 132, 29, 43, 5, 40, 48,
    87, 79, 20, 33, 60, 25, 42, 52, 116, 95,
    66, 25, 78, 138, 32, 28, 84, 102, 112, 7,
    62, 109, 81, 131, 26, 109, 65, 121, 90, 31,
    24, 122, 101, 76, 100, 125, 45, 80, 64, 98,
    102, 40, 110, 59, 97, 90, 43, 96, 21, 55,
    30
  ],
  "inventory": [
    80, 40, 560, 820, 960, 850, 950, 360, 580, 800,
    340, 30, 990, 70, 890, 240, 290, 40, 300, 350,
    600, 610, 160, 260, 480, 200, 290, 350, 900, 760,
    440, 170, 600, 920, 240, 190, 600, 760, 850, 50,
    470, 790, 580, 880, 170, 790, 500, 890, 680, 250,
    170, 830, 680, 590, 700, 830, 360, 570, 510, 690,
    760, 270, 880, 470, 670, 620, 310, 670, 150, 440,
    210
  ],
  "products": [
    "Adams Group_service",
    "Anderson-Leach_against",
    "Anderson-Valdez_somebody",
    "Anderson-White_son",
    "Andrews LLC_matter",
    "Andrews PLC_enter",
    "Andrews-Martin_build",
    "Arias-Mendoza_life",
    "Bennett and Sons_down",
    "Bennett, Foster and Moreno_enter",
    "Fernandez, Long and Nelson_member",
    "Fernandez-Fischer_million",
    "Ferrell Inc_prove",
    "Fields, Christensen and Daniels_nor",
    "Figueroa LLC_involve",
    "Finley Group_itself",
    "Fisher Ltd_speak",
    "Fisher-Marshall_cover",
    "Fuller-Torres_behavior",
    "Fuller-Walters_radio",
    "Gallagher-Campbell_dark",
    "Gallagher-Kirby_talk",
    "Gallagher-Parker_recognize",
    "Gonzalez PLC_task",
    "Gonzalez, Coleman and Le_heavy",
    "Gonzalez, Lowe and Robinson_education",
    "Gonzalez-Horn_light",
    "Good, Davis and Smith_station",
    "Goodman, Hughes and White_realize",
    "Goodwin PLC_bad",
    "Gordon PLC_detail",
    "Graham LLC_stage",
    "Graham Ltd_marriage",
    "Graham-Swanson_message",
    "Grant, Cross and Bennett_religious",
    "Grant, Mcdonald and Watson_owner",
    "Graves, Turner and Crawford_wait",
    "Gray, Smith and Barnes_much",
    "Green Inc_direction",
    "Green-Rogers_could",
    "Greene-Baxter_them",
    "Greer and Sons_keep",
    "Griffin, Boyle and Dawson_anything",
    "Harper Inc_well",
    "Harper Ltd_off",
    "Harrington, Sosa and Mccarty_coach",
    "Harris Group_different",
    "Harris and Sons_audience",
    "Harris and Sons_fear",
    "Harris, Hamilton and Rose_contain",
    "Harris, Stevens and Hall_answer",
    "Harris-Bell_painting",
    "Harris-Melton_often",
    "Harris-Rogers_after",
    "Hart Group_cold",
    "Hutchinson, Roberts and Mcbride_among",
    "Ibarra, Jackson and Potter_upon",
    "Jackson Inc_anyone",
    "Jackson LLC_report",
    "Jackson, Collier and Barber_result",
    "Jackson, White and Brown_expect",
    "Jackson-Carroll_six",
    "Lee-Jones_soldier",
    "Lewis, Sanchez and Turner_point",
    "Lin LLC_performance",
    "Lindsey, Avila and Brown_candidate",
    "Lindsey-Strickland_modern",
    "Little-Perez_choice",
    "Lloyd, Stone and Mcguire_national",
    "Long, Hughes and Gallegos_receive",
    "Lopez PLC_analysis"
  ]
}