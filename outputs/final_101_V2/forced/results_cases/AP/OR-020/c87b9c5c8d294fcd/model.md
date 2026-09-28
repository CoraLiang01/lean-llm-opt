##### Decision Variables

Let $x_i$ denote the number of units of product $i$ to fulfill, for each product $i$ in the set of all products.

##### Objective Function

$\quad \max \sum_{i} r_i x_i$

where $r_i$ is the revenue per unit for product $i$.

##### Constraints

For each product $i$:

$\quad 0 \leq x_i \leq \min\{\text{Demand}_i, \text{Initial Inventory}_i\}$

$\quad x_i \in \mathbb{Z}_{\geq 0}$

##### Retrieved Information

{
  "products": [
    {"Product Name": "S10_1678", "Revenue": 95.7, "Demand": 1280, "Initial Inventory": 9440},
    {"Product Name": "S10_1949", "Revenue": 100.0, "Demand": 1316, "Initial Inventory": 9610},
    {"Product Name": "S10_2016", "Revenue": 99.91, "Demand": 1242, "Initial Inventory": 9280},
    {"Product Name": "S10_4698", "Revenue": 100.0, "Demand": 1258, "Initial Inventory": 9210},
    {"Product Name": "S10_4757", "Revenue": 100.0, "Demand": 1292, "Initial Inventory": 9520},
    {"Product Name": "S10_4962", "Revenue": 100.0, "Demand": 1262, "Initial Inventory": 9320},
    {"Product Name": "S12_1099", "Revenue": 100.0, "Demand": 1132, "Initial Inventory": 8380},
    {"Product Name": "S12_1108", "Revenue": 100.0, "Demand": 1334, "Initial Inventory": 9730},
    {"Product Name": "S12_1666", "Revenue": 100.0, "Demand": 1323, "Initial Inventory": 9720},
    {"Product Name": "S12_2823", "Revenue": 100.0, "Demand": 1340, "Initial Inventory": 9640},
    {"Product Name": "S12_3148", "Revenue": 100.0, "Demand": 1232, "Initial Inventory": 8980},
    {"Product Name": "S12_3380", "Revenue": 100.0, "Demand": 1157, "Initial Inventory": 8530},
    {"Product Name": "S12_3891", "Revenue": 100.0, "Demand": 1237, "Initial Inventory": 9210},
    {"Product Name": "S12_3990", "Revenue": 89.38, "Demand": 1091, "Initial Inventory": 8000},
    {"Product Name": "S12_4473", "Revenue": 100.0, "Demand": 1386, "Initial Inventory": 10240},
    {"Product Name": "S12_4675", "Revenue": 100.0, "Demand": 1331, "Initial Inventory": 9640},
    {"Product Name": "S18_1097", "Revenue": 100.0, "Demand": 1364, "Initial Inventory": 9990},
    {"Product Name": "S18_1129", "Revenue": 100.0, "Demand": 1287, "Initial Inventory": 9470},
    {"Product Name": "S18_1342", "Revenue": 100.0, "Demand": 1378, "Initial Inventory": 9970},
    {"Product Name": "S18_1367", "Revenue": 50.14, "Demand": 1244, "Initial Inventory": 8900},
    {"Product Name": "S18_1589", "Revenue": 100.0, "Demand": 1278, "Initial Inventory": 9140},
    {"Product Name": "S18_1662", "Revenue": 100.0, "Demand": 1277, "Initial Inventory": 9400},
    {"Product Name": "S18_1749", "Revenue": 100.0, "Demand": 1063, "Initial Inventory": 8020},
    {"Product Name": "S18_1889", "Revenue": 82.39, "Demand": 1267, "Initial Inventory": 9510},
    {"Product Name": "S18_1984", "Revenue": 100.0, "Demand": 1233, "Initial Inventory": 9170},
    {"Product Name": "S18_2238", "Revenue": 100.0, "Demand": 1293, "Initial Inventory": 9660},
    {"Product Name": "S18_2248", "Revenue": 67.8, "Demand": 994, "Initial Inventory": 7430},
    {"Product Name": "S18_2319", "Revenue": 100.0, "Demand": 1363, "Initial Inventory": 9930},
    {"Product Name": "S18_2325", "Revenue": 100.0, "Demand": 1134, "Initial Inventory": 8280},
    {"Product Name": "S18_2432", "Revenue": 54.09, "Demand": 1362, "Initial Inventory": 9980},
    {"Product Name": "S18_2581", "Revenue": 90.39, "Demand": 1028, "Initial Inventory": 7460},
    {"Product Name": "S18_2625", "Revenue": 70.87, "Demand": 1200, "Initial Inventory": 8720},
    {"Product Name": "S18_2795", "Revenue": 100.0, "Demand": 1056, "Initial Inventory": 7890},
    {"Product Name": "S18_2870", "Revenue": 100.0, "Demand": 1139, "Initial Inventory": 8550},
    {"Product Name": "S18_2949", "Revenue": 83.07, "Demand": 1335, "Initial Inventory": 9910},
    {"Product Name": "S18_2957", "Revenue": 57.46, "Demand": 1267, "Initial Inventory": 9320},
    {"Product Name": "S18_3029", "Revenue": 83.44, "Demand": 1172, "Initial Inventory": 8720},
    {"Product Name": "S18_3136", "Revenue": 100.0, "Demand": 1175, "Initial Inventory": 8730},
    {"Product Name": "S18_3140", "Revenue": 100.0, "Demand": 1153, "Initial Inventory": 8220},
    {"Product Name": "S18_3232", "Revenue": 100.0, "Demand": 2457, "Initial Inventory": 17740},
    {"Product Name": "S18_3259", "Revenue": 100.0, "Demand": 1157, "Initial Inventory": 8600},
    {"Product Name": "S18_3278", "Revenue": 68.35, "Demand": 1139, "Initial Inventory": 8320},
    {"Product Name": "S18_3320", "Revenue": 100.0, "Demand": 1240, "Initial Inventory": 9090},
    {"Product Name": "S18_3482", "Revenue": 100.0, "Demand": 1206, "Initial Inventory": 8600},
    {"Product Name": "S18_3685", "Revenue": 100.0, "Demand": 1312, "Initial Inventory": 9480},
    {"Product Name": "S18_3782", "Revenue": 67.77, "Demand": 1224, "Initial Inventory": 8960},
    {"Product Name": "S18_3856", "Revenue": 100.0, "Demand": 1361, "Initial Inventory": 9970},
    {"Product Name": "S18_4027", "Revenue": 100.0, "Demand": 1241, "Initial Inventory": 9220},
    {"Product Name": "S18_4409", "Revenue": 86.51, "Demand": 1002, "Initial Inventory": 7500},
    {"Product Name": "S18_4522", "Revenue": 82.5, "Demand": 1246, "Initial Inventory": 9100},
    {"Product Name": "S18_4600", "Revenue": 100.0, "Demand": 1394, "Initial Inventory": 10310},
    {"Product Name": "S18_4668", "Revenue": 47.29, "Demand": 1327, "Initial Inventory": 9510},
    {"Product Name": "S18_4721", "Revenue": 100.0, "Demand": 1211, "Initial Inventory": 8940},
    {"Product Name": "S18_4933", "Revenue": 61.29, "Demand": 957, "Initial Inventory": 7140},
    {"Product Name": "S24_1046", "Revenue": 85.25, "Demand": 988, "Initial Inventory": 7240},
    {"Product Name": "S24_1444", "Revenue": 55.49, "Demand": 1347, "Initial Inventory": 9760},
    {"Product Name": "S24_1578", "Revenue": 100.0, "Demand": 1286, "Initial Inventory": 9310},
    {"Product Name": "S24_1628", "Revenue": 59.37, "Demand": 1193, "Initial Inventory": 8830},
    {"Product Name": "S24_1785", "Revenue": 88.63, "Demand": 1057, "Initial Inventory": 7840},
    {"Product Name": "S24_1937", "Revenue": 31.2, "Demand": 1165, "Initial Inventory": 8440},
    {"Product Name": "S24_2000", "Revenue": 83.03, "Demand": 1263, "Initial Inventory": 9290},
    {"Product Name": "S24_2011", "Revenue": 100.0, "Demand": 1288, "Initial Inventory": 9600},
    {"Product Name": "S24_2022", "Revenue": 53.76, "Demand": 1177, "Initial Inventory": 8510},
    {"Product Name": "S24_2300", "Revenue": 100.0, "Demand": 1358, "Initial Inventory": 9960},
    {"Product Name": "S24_2360", "Revenue": 58.87, "Demand": 1177, "Initial Inventory": 8450},
    {"Product Name": "S24_2766", "Revenue": 78.15, "Demand": 1191, "Initial Inventory": 8900},
    {"Product Name": "S24_2840", "Revenue": 39.6, "Demand": 1329, "Initial Inventory": 9830},
    {"Product Name": "S24_2841", "Revenue": 74.68, "Demand": 1135, "Initial Inventory": 8470},
    {"Product Name": "S24_2887", "Revenue": 100.0, "Demand": 1122, "Initial Inventory": 8100},
    {"Product Name": "S24_2972", "Revenue": 32.1, "Demand": 1223, "Initial Inventory": 9120},
    {"Product Name": "S24_3151", "Revenue": 72.58, "Demand": 1317, "Initial Inventory": 9550},
    {"Product Name": "S24_3191", "Revenue": 73.62, "Demand": 1086, "Initial Inventory": 7790},
    {"Product Name": "S24_3371", "Revenue": 63.07, "Demand": 1263, "Initial Inventory": 9200},
    {"Product Name": "S24_3420", "Revenue": 52.6, "Demand": 1183, "Initial Inventory": 8590},
    {"Product Name": "S24_3432", "Revenue": 100.0, "Demand": 1119, "Initial Inventory": 8240},
    {"Product Name": "S24_3816", "Revenue": 79.67, "Demand": 1182, "Initial Inventory": 8700},
    {"Product Name": "S24_3856", "Revenue": 100.0, "Demand": 1450, "Initial Inventory": 10520},
    {"Product Name": "S24_3949", "Revenue": 64.83, "Demand": 1357, "Initial Inventory": 10080},
    {"Product Name": "S24_3969", "Revenue": 34.47, "Demand": 1013, "Initial Inventory": 7450},
    {"Product Name": "S24_4048", "Revenue": 100.0, "Demand": 1145, "Initial Inventory": 8440},
    {"Product Name": "S24_4258", "Revenue": 100.0, "Demand": 1247, "Initial Inventory": 9000},
    {"Product Name": "S24_4278", "Revenue": 63.76, "Demand": 1205, "Initial Inventory": 8770},
    {"Product Name": "S24_4620", "Revenue": 68.71, "Demand": 1149, "Initial Inventory": 8330},
    {"Product Name": "S32_1268", "Revenue": 100.0, "Demand": 1185, "Initial Inventory": 8730},
    {"Product Name": "S32_1374", "Revenue": 92.9, "Demand": 1194, "Initial Inventory": 8680},
    {"Product Name": "S32_2206", "Revenue": 43.45, "Demand": 1160, "Initial Inventory": 8360},
    {"Product Name": "S32_2509", "Revenue": 47.62, "Demand": 1292, "Initial Inventory": 9550},
    {"Product Name": "S32_3207", "Revenue": 65.87, "Demand": 1253, "Initial Inventory": 9070},
    {"Product Name": "S32_3522", "Revenue": 75.63, "Demand": 1268, "Initial Inventory": 9570},
    {"Product Name": "S32_4289", "Revenue": 72.92, "Demand": 1193, "Initial Inventory": 8620},
    {"Product Name": "S32_4485", "Revenue": 100.0, "Demand": 1093, "Initial Inventory": 8170},
    {"Product Name": "S50_1341", "Revenue": 40.15, "Demand": 1351, "Initial Inventory": 9990},
    {"Product Name": "S50_1392", "Revenue": 100.0, "Demand": 1330, "Initial Inventory": 9790},
    {"Product Name": "S50_1514", "Revenue": 53.31, "Demand": 1278, "Initial Inventory": 9450},
    {"Product Name": "S50_4713", "Revenue": 82.99, "Demand": 1237, "Initial Inventory": 9120},
    {"Product Name": "S700_1138", "Revenue": 70.67, "Demand": 1254, "Initial Inventory": 9020},
    {"Product Name": "S700_1691", "Revenue": 100.0, "Demand": 1158, "Initial Inventory": 8370},
    {"Product Name": "S700_1938", "Revenue": 70.15, "Demand": 1144, "Initial Inventory": 8390},
    {"Product Name": "S700_2047", "Revenue": 100.0, "Demand": 1189, "Initial Inventory": 8680},
    {"Product Name": "S700_2466", "Revenue": 100.0, "Demand": 1274, "Initial Inventory": 9400},
    {"Product Name": "S700_2610", "Revenue": 65.77, "Demand": 1329, "Initial Inventory": 9900},
    {"Product Name": "S700_2824", "Revenue": 100.0, "Demand": 1342, "Initial Inventory": 9760},
    {"Product Name": "S700_2834", "Revenue": 100.0, "Demand": 1170, "Initial Inventory": 8610},
    {"Product Name": "S700_3167", "Revenue": 74.4, "Demand": 1247, "Initial Inventory": 9380},
    {"Product Name": "S700_3505", "Revenue": 81.14, "Demand": 1236, "Initial Inventory": 9170},
    {"Product Name": "S700_3962", "Revenue": 100.0, "Demand": 1175, "Initial Inventory": 8520},
    {"Product Name": "S700_4002", "Revenue": 61.44, "Demand": 1394, "Initial Inventory": 10290},
    {"Product Name": "S72_1253", "Revenue": 52.64, "Demand": 1252, "Initial Inventory": 9200},
    {"Product Name": "S72_3212", "Revenue": 56.78, "Demand": 1291, "Initial Inventory": 9270}
  ]
}

##### Full Model

Let $I$ be the set of all products listed above.

For each $i \in I$:

- $r_i$ = revenue per unit for product $i$ (see above)
- $d_i$ = demand for product $i$ (see above)
- $s_i$ = initial inventory for product $i$ (see above)
- $x_i$ = number of units of product $i$ to fulfill

The model is:

$\max \sum_{i \in I} r_i x_i$

subject to

$0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I$

$x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$