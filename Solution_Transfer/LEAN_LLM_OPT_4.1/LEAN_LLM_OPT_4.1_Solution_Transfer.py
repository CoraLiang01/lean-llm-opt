from __future__ import annotations



import contextlib
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from IPython.display import display

print('Python', sys.version.split()[0], '| Gurobi', '.'.join(map(str, gp.gurobi.version())))


PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
OUTPUT_DIR = PROJECT_ROOT / 'results'
METHODS = ['LEAN-GPT-4.1']





REPLAY_METHODS = set()

EXPECTED_IDS = {f'OR-{n:03}' for n in range(1, 102)}
CHECK_KINDS = ['constraint', 'bound', 'integrality', 'binary_domain']
TOLERANCES = {
    'diagnostic_1e-12': Fraction('1e-12'),
    'absolute_1e-9': Fraction('1e-9'),
    'absolute_1e-6': Fraction('1e-6'),
    'historical_scaled': None,
}
assert METHODS and len(set(METHODS)) == len(METHODS)
assert set(REPLAY_METHODS) <= set(METHODS)
assert (PROJECT_ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101.csv').is_file(), (
    'Open the notebook from the root of the LEAN repository.')


COHORTS = json.loads(r"""
{
 "LEAN-GPT-4.1": {
  "relative_directory": "source",
  "source_commit": "af1c3738342d2704129843749784d4acdc6d6031",
  "results_sha256": "3ff4a55a25e2e06cbfd7663e3024621a98ec9ecda6f98c3b5a799ec09087af10",
  "incumbent_protocol": "published_saved_solution"
 }
}
""")


DATA_SHA256 = json.loads(r"""
{
 "Test_Dataset/Large-scale-or/Large-scale-or-101.csv": "4574d324f111202b96c2791b588f21b4a2a509e6e0635cb0d8bf9a1fdf887f49",
 "Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv": "5499656a0e8e937cb80210810116f39b670b04c1e0883d781151e75e1c10defc",
 "Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv": "f8f169fca61e02f19bff5da273652ba016368ea526f6d0abead02007c23ee950",
 "Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv": "d15d2c1db282122b170697a44c1a87961d3266f8d803948d470218c22c652275",
 "Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv": "fe1b7a4ce4ffee22c4a587ad73d63b0a0d998dcb269b70bcbf136303df61896c",
 "Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv": "ef72030251e7df43df8ced4f7c244856382b2b861f87107525ffec7a5c792382",
 "Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv": "4d2713827c424fb5373c8954abaed7f04708c5b7a4bdc4383fcc8631dd082689",
 "Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv": "8e3d8d2c378a0e0548d15dd005b36d73c42d047c7052d9caab4c6bc0bc798587",
 "Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv": "4ac61a580b82bc701375e863fa695c1c95a005010d926bf8cdc35d8f68c7088a",
 "Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv": "ae79d34e4daa13feb2d66559f492e9954589a5c52230a61cace9fc08ec8a17a3",
 "Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv": "389d029a2912ef374cff24d8240b60e77e4532316e2b015e1940911f36e941e5",
 "Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv": "c4049a14f8f8ee520589fc1e41219427652b9a3cfb3149f4b0646a5e9fc9a248",
 "Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv": "58655183760198106bb4c8dac8357b5f5bf29497b81788a5e8e85bf780912091",
 "Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv": "f5e4cb6ecf3218b45aec7ff1ac4b5a6705e096733b32a280b40c395c39b9ca23",
 "Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv": "59c3133d4e8e3e06b02c2c55c21e58315f41ccfe76c5c1e76bc4771a74688d0f",
 "Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv": "0f94a7bd4b4ca29fd4cee9a17bb88e26d071e275103d70dfc1b6927e944a34bd",
 "Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv": "641d516cde0d5cf1e21d3195a84221718bf0bb38ffcfb116b4a471fc0607a5de",
 "Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv": "76e9dff477f1743a0d66a1d0abf80e05f942d6a0a3039492d83eaab3b83f4844",
 "Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv": "7c4f1e6a2b1efc9bac54b0a1fb1a377e2e6eaf1be6c97ce266b41b0ce5625d16",
 "Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv": "dc47c704349c8cc5b57c3e969f966f6941a787f41a361a2a076d83d549f55209",
 "Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv": "a979ed7c81c89a279aa6c934cfb4237b08544954ed0f62196d4362f988740eed",
 "Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv": "47e377319f1d321307a4f0b972bde6f82fb3571298ed672c7ba1af0312f52f74",
 "Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv": "bc9f6c01a98786697a4add9b4bd46026aaa4733ebf95cafb42b3d8761aff4f50",
 "Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv": "341fa6b12f085076276f4a5c444d2abda8de8122f3693ed271ac19d443575efe",
 "Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv": "7912f35b974272abd5f88573ed2b43ad6badb90b2a098eb9ecdb59bdf612fd3c",
 "Test_Dataset/Large-scale-or/TP_testing/TP9/customer_demand.csv": "33a5052d984ae6bdf6f8212c2b8b2f364e552dc0c4b87f59bb31cfe81870f877",
 "Test_Dataset/Large-scale-or/TP_testing/TP9/supply_capacity.csv": "38529367dcdda1f85260f9ca463f213477e28d05c1a1afd1eb36b5ed1166cd7d",
 "Test_Dataset/Large-scale-or/TP_testing/TP9/transportation_costs.csv": "bc2cf7cca51d17ccad835aa34e008dbf6f40fde0d3e07884e6a8d0cb78cbda1f",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv": "091b6e833fce948797100177d28c06714b4aeb9836bcd0c66f78407f2931bc8a",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv": "080279712e7e543d375862556b8f8be47c8461ca18210b381a921ca02bc9169b",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv": "58b733e4d1fcb2c3364a5f3213c48b397251f48946327e608d9e31d922897f74",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM4/OnlineSalesinUSA.csv": "ab9f7da9229fdf20bed21c961486c34e573d1b59d113d117dab76e154286fd3e",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv": "943a4628e70a4271a46923293851cdc654dfc8f209d9437eb84d0ee1bd698492",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv": "a086c472bc608c023fce018c8ea32237cc892db2023f7c2aad925bf293db4ba6",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv": "ffcea5fa0c066a16a069a68b0b09da4b5b6e7017df160b6f7af6b4347786e959",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv": "5ae5bae71639c142c44284dca4cb4bcd956a4f83c22ae684c5ad21b636f8d902",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv": "b120b37ea174d07288096c411f3f418090d43914bb073dbdc57694e485d2dcda",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv": "ad0f584338b2e87704c16d806abfba492df7258eba5b7548cb2568a347c3e28f",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv": "c249e8c3eaaa01a661bdccec6c5ff28d6b0f2e591cea86189f35ee4570a9fc5f",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv": "ad535d5e07e6d95c9806e866454aa617993bfca2b563d07d5a4e63a45f1b17d5",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv": "8caf12f4dfbe49778ea70b0202abe941ef161482b35b6ad53b21c77be263727a",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv": "dacce5b81b38dcb84638a419fd1eefa67f0c0155961cc181507be47405208f0c",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv": "42a7b837b14eee0f30c0d6d77d1c40f38533f26d283e85585952d9aaec23955a",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv": "a73870273fe085d8d48af748013b85c78373dab56309637f59f8cad4e00ccd6b",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv": "697698bc34e0c9c5a1edacc8a51f013309b0af6763e15606160970c6a060629c",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv": "eba917a7d7addb6e9e8990dd56433b98e72e24da2b4ba94cb458cdef65719756",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv": "b67f871b94f2a8d4da1632e82dd1bfed1ec335ab7f2bdf300d874f4d67f74842",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv": "785fc304e8147a2db575f7eb457eabce6e2688c57b19fcd2f7be7201324f3405",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv": "2c7017f632b5d326776214dcd1e8483a467137baa917b60f5ee0099f7aa1a1e3",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv": "00a197b026d710e5c00433f10182b37968a11e642315379125d2c2300e2950fd",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv": "772722170921e15da09e40aba9c4c48d8bf13943b9b5fc20af39a49a794fcc66",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv": "f55f7281a09e5fe6ec794864b4007b8c781d1d11a732363f4534f37be76848ba",
 "Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv": "d11ca52230f2f22a11ad494b1d90ea64c403eb405d7e88e45fcd1649e623e967",
 "Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv": "a7c5f189f0e4ecbc20b65c1bfe3e373db8b09e86cd489b28f1c0ef2477cbc1bf",
 "Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv": "decf9883b14104043c8b3c38cdb035d513fea3d08973f7eebd9f8115ca3fa0b7",
 "Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv": "6e20b18f49ea9eac6ec56922cc490b85823ebd0eed6b945d75bd140ee0736eae",
 "Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv": "cb6d87d80fa376b6ad1297aedb03fb9d2cb985b335f49ff5265ee9cc853f98d0",
 "Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv": "d5f91a46877a5e87fa389563e18807c4445cf03d67ac8977a9812f9766cfda68",
 "Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv": "b962c38ebc2e7b60da0a298de2df9408b5b9dca9ac3f6510a1d400cd08030628",
 "Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv": "5b034b564fba4cf7a4483bba53a6755f66ef113bd253932eb151fa9c9fbe833e",
 "Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv": "0d302bcd3dcb4a964f2914dd9db16070c17cc00172655c96350ee8f655d33196",
 "Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv": "0b2c362c76a4ee61be32016c6c8a66b6543f36cb77dce06eef1de73eec20a71f",
 "Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv": "0d302bcd3dcb4a964f2914dd9db16070c17cc00172655c96350ee8f655d33196",
 "Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv": "81a62b5cce59145bb5193df5fd68bf32e876ecaea683355801c4a369daa6c153",
 "Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv": "17194b5272b83b15e56cdd79f6c388e7c232ad2c7830f933d0fddc0b6df38b56",
 "Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv": "5587cc518877ecfd5f06b0d03f9c68fc8723c2d0e26e9b8a9eceba766efb631d",
 "Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv": "fdb52c2bf73f5d5e3de0e2503c288c7bd9a810f0235e581d657dd06e35942da6",
 "Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv": "d6711c7cae88ee9168d9850aabbdce5cf26fde234cf76719fc86ffdccd1fa527",
 "Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv": "04a35e5f71a15d1e774c8f9b1ba10f2751f60e095af5d60674da66a55bd72f1f",
 "Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv": "fa12c4d31d2bd3b5a770c032d2e45992fa6039837362cba061b8c34f9f832f74",
 "Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv": "0b586c45cf1a32658fc9cf4e474d7147bedc53b1f3ac998da123870e9d60758f",
 "Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv": "e3858f52a1f37d5a8ed2478bb833612e82ec0388241b5398a614105a069a0b7e",
 "Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv": "40bcc32492cc448f30a658700aa97a9e053a0a6fb06cd2fc94cdc8ffa620b9b2",
 "Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv": "5b437ca0a6fc363975a7670b6884f35ea8eaecdf6353652d7bf892ff485a182f",
 "Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv": "5eef4f77759f35c4161542d77c02725336e1403d326c7380795c0e986e9ffeaf",
 "Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv": "aabe26fed61769837013f24a2a6f846ed53a8a842b75ba1711dd18cf612c66c6",
 "Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv": "6214469728f168bacec685f3951cd9e544c9622bfd642bab81cae703494ef5e9",
 "Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv": "8b422a8e656dac50824672fef50a40988405f077c255f0e43e245baa56bf0a47",
 "Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv": "642182426b08987f13b1a30332bf170e8e42042ee8accc64e48fddd5a91340ca",
 "Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv": "bb47bd52d7b613eda4a67c1bebb9a1f786c9b85038ad70a8bf06f2e2b688e0dd",
 "Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv": "a9f3eb0a19da26b09a6d339bb3334a4fb6a1099ea5de82b364ffb4b400965a38",
 "Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv": "032c92e34b4bff880048b0a681b39ecde1aac4394bbae2eb82badb6d300383a6",
 "Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv": "d52b806a7e3a75d27314b1b00fa1e5d1f0a1e4d2d1d81a72058696604a75f20d",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv": "032c92e34b4bff880048b0a681b39ecde1aac4394bbae2eb82badb6d300383a6",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv": "d52b806a7e3a75d27314b1b00fa1e5d1f0a1e4d2d1d81a72058696604a75f20d",
 "Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv": "10b89c2fd24d9b7513915fcdb9dfcc435378184cb75dba81d18ec39df936e4da",
 "Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv": "2957906447e56e06f2fe259ddd0598df5b3ea3945e3bab13afdb5aa382d211d0",
 "Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv": "cbef27788783ac65a658a6630826c04435b33985af2b5feda528ec981eed019b",
 "Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv": "a39b65cb5615a233b130186e5ad0a8ca49080fb1bedbee8e492d25103223edff",
 "Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv": "1ccea65da55f787178a86efef6366b7f5c6908727aa22da8a304b6217fd561d0",
 "Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv": "8cfbf989364a66aee34280fd0a7e86ef95a2362301218ef69be5a77640b3e008",
 "Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv": "a1f6a1f7fc0cea95e69f1a028300e8d75d0d6e0afe8d47953d8177d980eda3be",
 "Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv": "a560cc91ca1f73321af1e80f678dde661df1117ae67088fb283a7cfdc4e4631d",
 "Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv": "739c15a341822fd704496b387684695638fd7c035f8c65cb6800b739be479548",
 "Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv": "648584dfb51b73ee43d6fbebf40b1d123bc59caffab17a78b96a680e927e1bc4",
 "Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv": "e49a6ff41156e305723de5aca1edcab6adc4946d197c241ceca8b7c28673dc03",
 "Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv": "1ab5603c1dcf8f6944ae6ef8f8dfc498a0d2b4a3492c6c0862aeca0a319b2723",
 "Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv": "affae625129d2747eaa518f5651a909c9f1325e0085005109ace84254de762ed",
 "Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv": "8d994d7e2a347458fe784d023db8d9cf795b4732f125a13f195400be5dc253c2",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv": "40207a7fdf31ec6fdf6d8479c6c30067344e172f87b5e96a77266414a6e941e0",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv": "92df27b6c9871fc064e749e1d3ed8b658e11b0bf23bae8754fa746aa82312bc9",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv": "86408e602473a9642c8c7e0e4e8ec7d686f51adf698b3677ce770b9fdbd30a88",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv": "8a667ef0c2783b6c04394b871705b3a86541927c370c01a89ebe6a4b1d72b281",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv": "740abb6c0c85be936766e14a8a6b734c787d4fa0239fa3b4c15f577153e3e1a7",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv": "1282b81bb52b6434832f29748b4e0ed9bafbc4794d385c87dc9178ddca4757e7",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv": "a55cd981f4c5415b8d6f2076de09b17ccc80e9130fa1ce46ce42008dda320f6c",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv": "0778a8d6a66a93b63c4d5b7526b0a584d3be7c6444a674bb57b5b7cc986c6e3e",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv": "b1640937d5dc429ed28a237ec38f469b63fb6c6a8a100a441ebd9a71bf4c9481",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv": "4ed478a31585ed0f8098f469615ae748c6b24692a8e597ecda2c7b4a4f51c802",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv": "916b87211250108d2e0f32b643a99667e1d04e8a891f51ab5c18c5f51ce2d254",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv": "ad6c9353bb673f0b5931ee2b1df20c00b2080736dbe112a82af965e11f9569c0",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv": "8fbfdf11ea3723ad548dfcfbc455412a03055744c86c6e6214dbc371729534fb",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv": "b741371b4938c2478495ae9bfe536e0df0f5be3a093f6ce9593f042bac64b89a",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv": "1f9c3dee57b89cbd7d57956b7bb05ea5aec870368d673a865fa8b66c428bf60f",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv": "06dd80a13fcf1e7176f4c2761b82b698e1dfcf011b7dde5799ea236a463bd14c",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv": "de421fc63e9194e67dc06af7f8d51d83111e3837dcb043f694c5f926e73a278a",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv": "08993dd8d77f1990d4887bac7c8f61de5e03afe8719fa95ff15827de5278bac6",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv": "ec9b8fa194d8d55bdce213f19f055f81a12d4a5ae84da844820c273da37de9e7",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv": "e377f19d39860785a32e1f394328e1ed658836d7277d059c7f4363fdc8194ca0",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv": "572981048c3a604af854fb49cc71442ae1bf2237055f7ce4ec14250058d01566",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv": "16ad9778dff705c7de7c4780c7e953c87f62497e5bd4402c182ba190fbbe5941",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv": "33c3e53c24f98fa551a5c8394a820db48938edd44b442dbc12f2bf39443f3217",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv": "21613a8e9b9fbe531f86bf7ab764d4c1e4d7f1d9742a4c1aa584f672f3c88616",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv": "4bb1eb6d09329d99391279e86b50c439ff70b3dc16cb959065a6e7422c0e3e72",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv": "b046b351d9b64f3e2cca6800097af5b61308e79b75c06bd175bc2c2679ba2170",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv": "86ef1b1c5bc7da597220a32598e080d00c3a1edd70d09b48ca97861304862b8b",
 "Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv": "8eb5bbaa573801aeef079026a61ab48f069750af43dc11c36113ad5df6673ee5",
 "Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv": "4fdfa931f220fd28b949e50bba1be1cfc333020a76455f01059e108a3812eefd",
 "Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv": "279314ecbe2c09bf2f895033cc5d586d78f5c8a9c9ac19701c9c86286cfe9a22",
 "Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv": "be1de4df46ef32f13fe8813171c9ac6a788442e8d9a4a6a923b7dab7f1a1a0f3",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv": "ac6d39cb56e43da01b2a120af793ac98eb2cddd26a9d8af9b9fbea1d88c471b2",
 "Test_Dataset/Large-scale-or/Others_example/Others1/42.csv": "377b0bacab66e4d1fd49b1b0649d35c9d3f5b1b7aba584bab568da2b07abc711",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv": "72379f5520f929974f0ccbfd804a0d936e6124f76f9bdf83f50ed340baa513a2",
 "Test_Dataset/Large-scale-or/Others_example/Others2/44.csv": "c0f3e39ceb3b0890b514a8101e9f20f24fe22ed93c2fbb227122cb557631306c",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv": "6f922e537ec513c2889bbb9bbdbc89ac30c344a1d8bd28729f91599fd9435727",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv": "15bfef9f5b175352c174c8454f0670586356248ff8b8b062de7a116db2ca643c",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv": "e580021ce3cbb416ee1355595f5901c8a18b3309b27f89ff6b65f43a637aa7e7",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv": "31bea2c0460f809eeb71099e6e5fcb1ad1c914fe9e3a5d5454a30a24ec004bdc",
 "Test_Dataset/Large-scale-or/Others_example/Others3/value.csv": "1c75d4b0cefcfe10ce1dc91fc3e0086b7809ea29333bdd7b986541b52fe9b407",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv": "51ce25a4b37890137be448c95dbdfcac876c4207052c6a7cf906a56184abb219",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv": "438c922486a934afc89711b4168e8efa70157daa8bd3f1c12edc0621bebc8291",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv": "22e52580a770cceb5b5c11854c30f5b9fcfbafd5d76862a61cba986622149ea6",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv": "4d1304af2c062c28454beb69ffd9fa8dd62ba7fda9022555c2314747065d5359",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv": "b8edc2a683bf767727bd5c58ae48aa7147d06caa77827ca5b3c9972ec8d6442a",
 "Test_Dataset/Large-scale-or/Others_example/Others4/cost.csv": "4bb18894a791c334d702df07f1d21842f63499a97603f377ade3e8423c0f5c77",
 "Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv": "14f025b663b876e4e19476e2492cd6124c7542f80bd88f15e05511c7c0b77499",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv": "72db476adbe42cc7f36e7e9624efd6b2185d5fb5f9d2f766f194b043b3542015",
 "Test_Dataset/Large-scale-or/Others_example/Others6/18.csv": "f2605fd69187beaa9b5a9c9c7cf8837f6eb1c910404ffdf6eb2033487014042d",
 "Test_Dataset/Large-scale-or/Others_example/Others7/20.csv": "ca2beb1786679ff7a3dfa4f0825c5358f473c042c7e93c85085069ca808cd330",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv": "128665bb0c5ba0190b8c83be81ca8ca4932b099fb1c63e298a00bd331819b4a3",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv": "e4be0a2bb45608859026e6997d1807c9c9bee071dfa591d8bf199f92450bc7a7",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-1.csv": "23db8b6295bf887e6d7358a90610ab93fe879b37622acc3aa409cdc566a1f46d",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-2.csv": "ac8c7b3bd87d1c6fbfa41d18f33e622a5b9f66efd0ac5c627358efa5c172f006",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture9/36-3.csv": "d81967b90f184d62f0972cebda63a2d5101c180dd033b5e7be5997644345e770",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv": "cd85765800b1faf8bcc06995cb533c7e65ab13709c5abec0f34a47b356ef6a72",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv": "55ceabde8a8201593949dfb3adca8cd1c81c42c2b990567f034a6a3d52895039",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv": "a46e4934b67eb043f35c820777ae41907dfc9c023c98c603c088b19379b2300d",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv": "48f890d274766a82ace6b8e4462af7c5b1a14470e989323144334a982bc85a54",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv": "e1aa0e6b716a1fb60101d2fc7c785cf05acc40205535f34e948fefd9bbcc6e3d",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv": "c0e94eb4e5eb446fd37b71641f3eff69c681aec8211d62be0955edfedaaf58a5",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv": "000fb30f81d96db2a130b4ba308839597d79c656417f880c01b916e946c4a998",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv": "b306fbd1796980bbe0ea31b89cf78bdde95aafcc585ff8359a933bcb9388cd73",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv": "0a83587726cdb710084d35f77f355cd91d4ba2e52bed3e3f3f6a9b37c6df8c0a",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv": "fe5efc318907f3e9802adec2da8a2a12e2579dfe360b1ffe87b23d5466338322",
 "Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv": "aa3f2ed7ddb5720955695d1a1bda294dba2c9a5b83204189a2adaf8a16c57053",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv": "ea7d6e9f80214360197a627c1d082e76de5b5052b158f4ea6e5920cadcc3487c",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv": "6d018fd46384aa0db74c4f66b15261362c60837070464e14eb57f09313dfce53",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv": "e34650f8c91879d30ae0a31ef0294a927bcb3bf62e66545060cff0be5b26c0d2",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv": "d19ab57215a092571eb1998c55c925f04078d3445d8068b60172b71e66363681",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv": "abc5baee5025d1362e03297b15f5f6c3436f50ba218e9d7ac42f0045a9e01ba4",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv": "ea9243808e680555ab432441ca1152cb162e5753900d573e1c891448bd93816b",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv": "8d5b080b391f45b9cd7ad096a3a137bae0425752501aeebf5609257a60c44f76",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv": "3306cc2e10f21d60fd534e46147816edad4edeaafb7a111e9b0335d3707b702b",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture17/work_days.csv": "a933be3291673fbd19260290a7e2f3735ab84338107c82663d7007b548392519",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv": "0ea3f3850f5d13ae6f6f7fa66505cdd54003c3588eb7d6cf9c92bdf4999ce015",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv": "aa0a2bfbaecc84e1f3e65ee812dee4d5a765febc69b9f68baee32c1f5f880717",
 "Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv": "10baa2e9faf7b332664455ca7c4828b34138ce3ee05bd278979321fdbf25c826",
 "Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv": "91405471ec3bfea22ba3380d7e11dcfa02e67d76be7555e9b2ee9811ed8a9d95",
 "Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv": "b7f5f0100a8e0cc8b5e03b0daed18ae6746d6b7cd2ffe959019f38bd8c607ce2",
 "Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv": "717d0cb255300196a34c8eb776f809c3bce2885a667770558a24670becd17f5e",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv": "1b75e2e512d8ebf60abef46c67cea6edb892f3df74bc260cf889b6e2b09251b1",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv": "5242f2683630d1634b8ed7fbe760f974a4e765315d4129b5c44499e7b7d040e0",
 "Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv": "62ffc58156fdc5ff6f3176f58566cfc224878eaa1c3ced55c871ed53371dc4d1"
}
""")


REFERENCE_CERTIFICATES = json.loads(r"""
{
 "OR-001": {
  "problem_id": "OR-001",
  "status": 2,
  "solution_count": 1,
  "objective": 172752.82458839758,
  "objective_bound": 172752.82458839758,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 172752.82458839758,
  "historical_reference_objective_verified": true
 },
 "OR-002": {
  "problem_id": "OR-002",
  "status": 2,
  "solution_count": 1,
  "objective": 6464420.6996351,
  "objective_bound": 6464420.6996351,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 6464420.6996351,
  "historical_reference_objective_verified": true
 },
 "OR-003": {
  "problem_id": "OR-003",
  "status": 2,
  "solution_count": 1,
  "objective": 36778.51094448115,
  "objective_bound": 36778.51094448115,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 36778.51094448115,
  "historical_reference_objective_verified": true
 },
 "OR-004": {
  "problem_id": "OR-004",
  "status": 2,
  "solution_count": 1,
  "objective": 913520.1122932937,
  "objective_bound": 913520.1122932937,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 913520.1122932937,
  "historical_reference_objective_verified": true
 },
 "OR-005": {
  "problem_id": "OR-005",
  "status": 2,
  "solution_count": 1,
  "objective": 2710.9341837593297,
  "objective_bound": 2710.9341837593297,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 2710.9341837593297,
  "historical_reference_objective_verified": true
 },
 "OR-006": {
  "problem_id": "OR-006",
  "status": 2,
  "solution_count": 1,
  "objective": 6340.972286592267,
  "objective_bound": 6340.972286592267,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 6340.972286592267,
  "historical_reference_objective_verified": true
 },
 "OR-007": {
  "problem_id": "OR-007",
  "status": 2,
  "solution_count": 1,
  "objective": 10291.277006657501,
  "objective_bound": 10291.277006657501,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 10291.277006657501,
  "historical_reference_objective_verified": true
 },
 "OR-008": {
  "problem_id": "OR-008",
  "status": 2,
  "solution_count": 1,
  "objective": 660.0,
  "objective_bound": 660.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 660.0,
  "historical_reference_objective_verified": true
 },
 "OR-009": {
  "problem_id": "OR-009",
  "status": 2,
  "solution_count": 1,
  "objective": 238781.51192549666,
  "objective_bound": 238781.51192549666,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 238781.51192549666,
  "historical_reference_objective_verified": true
 },
 "OR-010": {
  "problem_id": "OR-010",
  "status": 2,
  "solution_count": 1,
  "objective": 4003445.2099999995,
  "objective_bound": 4003445.2099999995,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 4003445.2100000004,
  "historical_reference_objective_verified": true
 },
 "OR-011": {
  "problem_id": "OR-011",
  "status": 2,
  "solution_count": 1,
  "objective": 3552260.54,
  "objective_bound": 3552260.54,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 3552260.54,
  "historical_reference_objective_verified": true
 },
 "OR-012": {
  "problem_id": "OR-012",
  "status": 2,
  "solution_count": 1,
  "objective": 83748.37000000001,
  "objective_bound": 83748.37000000001,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 83748.37,
  "historical_reference_objective_verified": true
 },
 "OR-013": {
  "problem_id": "OR-013",
  "status": 2,
  "solution_count": 1,
  "objective": 532.3,
  "objective_bound": 532.3,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 532.3,
  "historical_reference_objective_verified": true
 },
 "OR-014": {
  "problem_id": "OR-014",
  "status": 2,
  "solution_count": 1,
  "objective": 1794269.0,
  "objective_bound": 1794269.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1794269.0,
  "historical_reference_objective_verified": true
 },
 "OR-015": {
  "problem_id": "OR-015",
  "status": 2,
  "solution_count": 1,
  "objective": 29660.0,
  "objective_bound": 29660.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 29660.0,
  "historical_reference_objective_verified": true
 },
 "OR-016": {
  "problem_id": "OR-016",
  "status": 2,
  "solution_count": 1,
  "objective": 1568955.0,
  "objective_bound": 1568955.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1568955.0,
  "historical_reference_objective_verified": true
 },
 "OR-017": {
  "problem_id": "OR-017",
  "status": 2,
  "solution_count": 1,
  "objective": 2015.08,
  "objective_bound": 2015.08,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 2015.08,
  "historical_reference_objective_verified": true
 },
 "OR-018": {
  "problem_id": "OR-018",
  "status": 2,
  "solution_count": 1,
  "objective": 782819438.64,
  "objective_bound": 782819438.64,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 782819438.64,
  "historical_reference_objective_verified": true
 },
 "OR-019": {
  "problem_id": "OR-019",
  "status": 2,
  "solution_count": 1,
  "objective": 7123134.6899999995,
  "objective_bound": 7123134.6899999995,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 7123134.6899999995,
  "historical_reference_objective_verified": true
 },
 "OR-020": {
  "problem_id": "OR-020",
  "status": 2,
  "solution_count": 1,
  "objective": 11258129.67,
  "objective_bound": 11258129.67,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 11258129.67,
  "historical_reference_objective_verified": true
 },
 "OR-021": {
  "problem_id": "OR-021",
  "status": 2,
  "solution_count": 1,
  "objective": 6020981.48,
  "objective_bound": 6020981.48,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 6020981.48,
  "historical_reference_objective_verified": true
 },
 "OR-022": {
  "problem_id": "OR-022",
  "status": 2,
  "solution_count": 1,
  "objective": 4049812.4247,
  "objective_bound": 4049812.4247,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 4049812.4247,
  "historical_reference_objective_verified": true
 },
 "OR-023": {
  "problem_id": "OR-023",
  "status": 2,
  "solution_count": 1,
  "objective": 63628.0,
  "objective_bound": 63628.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 63628.0,
  "historical_reference_objective_verified": true
 },
 "OR-024": {
  "problem_id": "OR-024",
  "status": 2,
  "solution_count": 1,
  "objective": 1264095.5,
  "objective_bound": 1264095.5,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1264095.5,
  "historical_reference_objective_verified": true
 },
 "OR-025": {
  "problem_id": "OR-025",
  "status": 2,
  "solution_count": 1,
  "objective": 5486959.829999999,
  "objective_bound": 5486959.829999999,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 5486959.83,
  "historical_reference_objective_verified": true
 },
 "OR-026": {
  "problem_id": "OR-026",
  "status": 2,
  "solution_count": 1,
  "objective": 8851.75,
  "objective_bound": 8851.75,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 8851.75,
  "historical_reference_objective_verified": true
 },
 "OR-027": {
  "problem_id": "OR-027",
  "status": 2,
  "solution_count": 1,
  "objective": 784297054.11,
  "objective_bound": 784297054.11,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 784297054.1099999,
  "historical_reference_objective_verified": true
 },
 "OR-028": {
  "problem_id": "OR-028",
  "status": 2,
  "solution_count": 1,
  "objective": 285816.0,
  "objective_bound": 285816.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 285816.0,
  "historical_reference_objective_verified": true
 },
 "OR-029": {
  "problem_id": "OR-029",
  "status": 2,
  "solution_count": 1,
  "objective": 5090631.58,
  "objective_bound": 5090631.58,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 5090631.580000001,
  "historical_reference_objective_verified": true
 },
 "OR-030": {
  "problem_id": "OR-030",
  "status": 2,
  "solution_count": 1,
  "objective": 34856.759999999995,
  "objective_bound": 34856.759999999995,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 34856.76,
  "historical_reference_objective_verified": true
 },
 "OR-031": {
  "problem_id": "OR-031",
  "status": 2,
  "solution_count": 1,
  "objective": 67016511.28,
  "objective_bound": 67016511.28,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 67016511.28000001,
  "historical_reference_objective_verified": true
 },
 "OR-032": {
  "problem_id": "OR-032",
  "status": 2,
  "solution_count": 1,
  "objective": 1240406.25,
  "objective_bound": 1240406.25,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1240406.25,
  "historical_reference_objective_verified": true
 },
 "OR-033": {
  "problem_id": "OR-033",
  "status": 2,
  "solution_count": 1,
  "objective": 195506188.0,
  "objective_bound": 195506188.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 195506188.0,
  "historical_reference_objective_verified": true
 },
 "OR-034": {
  "problem_id": "OR-034",
  "status": 2,
  "solution_count": 1,
  "objective": 914600.2500000002,
  "objective_bound": 914600.2500000002,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 914600.25,
  "historical_reference_objective_verified": true
 },
 "OR-035": {
  "problem_id": "OR-035",
  "status": 2,
  "solution_count": 2,
  "objective": 150660.0,
  "objective_bound": 150660.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 150660.0,
  "historical_reference_objective_verified": true
 },
 "OR-036": {
  "problem_id": "OR-036",
  "status": 2,
  "solution_count": 2,
  "objective": 509925.0,
  "objective_bound": 509925.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 509925.0,
  "historical_reference_objective_verified": true
 },
 "OR-037": {
  "problem_id": "OR-037",
  "status": 2,
  "solution_count": 4,
  "objective": 722376.0,
  "objective_bound": 722376.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 722376.0,
  "historical_reference_objective_verified": true
 },
 "OR-038": {
  "problem_id": "OR-038",
  "status": 2,
  "solution_count": 1,
  "objective": 1314500.0,
  "objective_bound": 1314500.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1314500.0,
  "historical_reference_objective_verified": true
 },
 "OR-039": {
  "problem_id": "OR-039",
  "status": 2,
  "solution_count": 3,
  "objective": 429000.0,
  "objective_bound": 429000.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 429000.0,
  "historical_reference_objective_verified": true
 },
 "OR-040": {
  "problem_id": "OR-040",
  "status": 2,
  "solution_count": 2,
  "objective": 74260.0,
  "objective_bound": 74260.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 74260.0,
  "historical_reference_objective_verified": true
 },
 "OR-041": {
  "problem_id": "OR-041",
  "status": 2,
  "solution_count": 2,
  "objective": 3912.0,
  "objective_bound": 3912.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 3912.0,
  "historical_reference_objective_verified": true
 },
 "OR-042": {
  "problem_id": "OR-042",
  "status": 2,
  "solution_count": 2,
  "objective": 81918.0,
  "objective_bound": 81918.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 81918.0,
  "historical_reference_objective_verified": true
 },
 "OR-043": {
  "problem_id": "OR-043",
  "status": 2,
  "solution_count": 2,
  "objective": 13570.0,
  "objective_bound": 13570.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 13570.0,
  "historical_reference_objective_verified": true
 },
 "OR-044": {
  "problem_id": "OR-044",
  "status": 2,
  "solution_count": 2,
  "objective": 8000.0,
  "objective_bound": 8000.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 8000.0,
  "historical_reference_objective_verified": true
 },
 "OR-045": {
  "problem_id": "OR-045",
  "status": 2,
  "solution_count": 2,
  "objective": 885.0,
  "objective_bound": 885.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 885.0,
  "historical_reference_objective_verified": true
 },
 "OR-046": {
  "problem_id": "OR-046",
  "status": 2,
  "solution_count": 1,
  "objective": 212.0,
  "objective_bound": 212.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 212.0,
  "historical_reference_objective_verified": true
 },
 "OR-047": {
  "problem_id": "OR-047",
  "status": 2,
  "solution_count": 5,
  "objective": 5433.0,
  "objective_bound": 5433.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 5433.0,
  "historical_reference_objective_verified": true
 },
 "OR-048": {
  "problem_id": "OR-048",
  "status": 2,
  "solution_count": 1,
  "objective": 635052.0,
  "objective_bound": 635052.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 635052.0,
  "historical_reference_objective_verified": true
 },
 "OR-049": {
  "problem_id": "OR-049",
  "status": 2,
  "solution_count": 4,
  "objective": 85500.0,
  "objective_bound": 85500.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 85500.0,
  "historical_reference_objective_verified": true
 },
 "OR-050": {
  "problem_id": "OR-050",
  "status": 2,
  "solution_count": 4,
  "objective": 80250.0,
  "objective_bound": 80250.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 80250.0,
  "historical_reference_objective_verified": true
 },
 "OR-051": {
  "problem_id": "OR-051",
  "status": 2,
  "solution_count": 3,
  "objective": 594900.0,
  "objective_bound": 594900.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 594900.0,
  "historical_reference_objective_verified": true
 },
 "OR-052": {
  "problem_id": "OR-052",
  "status": 2,
  "solution_count": 3,
  "objective": 30240.0,
  "objective_bound": 30240.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 30240.0,
  "historical_reference_objective_verified": true
 },
 "OR-053": {
  "problem_id": "OR-053",
  "status": 2,
  "solution_count": 3,
  "objective": 41040.0,
  "objective_bound": 41040.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 41040.0,
  "historical_reference_objective_verified": true
 },
 "OR-054": {
  "problem_id": "OR-054",
  "status": 2,
  "solution_count": 3,
  "objective": 98670.0,
  "objective_bound": 98670.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 98670.0,
  "historical_reference_objective_verified": true
 },
 "OR-055": {
  "problem_id": "OR-055",
  "status": 2,
  "solution_count": 4,
  "objective": 27088640.0,
  "objective_bound": 27088640.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 27088640.0,
  "historical_reference_objective_verified": true
 },
 "OR-056": {
  "problem_id": "OR-056",
  "status": 2,
  "solution_count": 5,
  "objective": 55642671.0,
  "objective_bound": 55642671.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 55642671.0,
  "historical_reference_objective_verified": true
 },
 "OR-057": {
  "problem_id": "OR-057",
  "status": 2,
  "solution_count": 2,
  "objective": 12502.0,
  "objective_bound": 12502.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 12502.0,
  "historical_reference_objective_verified": true
 },
 "OR-058": {
  "problem_id": "OR-058",
  "status": 2,
  "solution_count": 2,
  "objective": 35950.649999999994,
  "objective_bound": 35950.64999999999,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 35950.649999999994,
  "historical_reference_objective_verified": true
 },
 "OR-059": {
  "problem_id": "OR-059",
  "status": 2,
  "solution_count": 1,
  "objective": 1862205008776.5698,
  "objective_bound": 1862205007990.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 4.223862681668492e-10,
  "historical_reference_objective": 1862205008776.57,
  "historical_reference_objective_verified": true
 },
 "OR-060": {
  "problem_id": "OR-060",
  "status": 2,
  "solution_count": 4,
  "objective": 8643.2,
  "objective_bound": 8643.2,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 8643.2,
  "historical_reference_objective_verified": true
 },
 "OR-061": {
  "problem_id": "OR-061",
  "status": 2,
  "solution_count": 3,
  "objective": 10199.960000000001,
  "objective_bound": 10199.960000000001,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 10199.960000000001,
  "historical_reference_objective_verified": true
 },
 "OR-062": {
  "problem_id": "OR-062",
  "status": 2,
  "solution_count": 2,
  "objective": 172612.46,
  "objective_bound": 172612.46,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": null,
  "historical_reference_objective_verified": null,
  "purpose": "reference_only_no_generated_incumbent_available"
 },
 "OR-063": {
  "problem_id": "OR-063",
  "status": 2,
  "solution_count": 2,
  "objective": 778095.44,
  "objective_bound": 778095.44,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 778095.44,
  "historical_reference_objective_verified": true
 },
 "OR-064": {
  "problem_id": "OR-064",
  "status": 2,
  "solution_count": 5,
  "objective": 18461.11,
  "objective_bound": 18461.109999999997,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 18461.11,
  "historical_reference_objective_verified": true
 },
 "OR-065": {
  "problem_id": "OR-065",
  "status": 2,
  "solution_count": 2,
  "objective": 317318.5,
  "objective_bound": 317318.5,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 317318.50000000006,
  "historical_reference_objective_verified": true
 },
 "OR-066": {
  "problem_id": "OR-066",
  "status": 2,
  "solution_count": 1,
  "objective": 11396.51,
  "objective_bound": 11396.51,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 11396.51,
  "historical_reference_objective_verified": true
 },
 "OR-067": {
  "problem_id": "OR-067",
  "status": 2,
  "solution_count": 1,
  "objective": 9000.0,
  "objective_bound": 9000.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 9000.0,
  "historical_reference_objective_verified": true
 },
 "OR-068": {
  "problem_id": "OR-068",
  "status": 2,
  "solution_count": 1,
  "objective": 9089.0,
  "objective_bound": 9089.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 9089.0,
  "historical_reference_objective_verified": true
 },
 "OR-069": {
  "problem_id": "OR-069",
  "status": 2,
  "solution_count": 1,
  "objective": 4903.0,
  "objective_bound": 4903.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 4903.0,
  "historical_reference_objective_verified": true
 },
 "OR-070": {
  "problem_id": "OR-070",
  "status": 2,
  "solution_count": 1,
  "objective": 8809.0,
  "objective_bound": 8809.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 8809.0,
  "historical_reference_objective_verified": true
 },
 "OR-071": {
  "problem_id": "OR-071",
  "status": 2,
  "solution_count": 1,
  "objective": 24877.394636015328,
  "objective_bound": 24877.394636015328,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 24877.394636015324,
  "historical_reference_objective_verified": true
 },
 "OR-072": {
  "problem_id": "OR-072",
  "status": 2,
  "solution_count": 1,
  "objective": 320.0,
  "objective_bound": 320.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 320.0,
  "historical_reference_objective_verified": true
 },
 "OR-073": {
  "problem_id": "OR-073",
  "status": 2,
  "solution_count": 1,
  "objective": 1146.5665024630537,
  "objective_bound": 1146.5665024630537,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1146.5665024630543,
  "historical_reference_objective_verified": true
 },
 "OR-074": {
  "problem_id": "OR-074",
  "status": 2,
  "solution_count": 2,
  "objective": 25.0,
  "objective_bound": 25.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 25.0,
  "historical_reference_objective_verified": true
 },
 "OR-075": {
  "problem_id": "OR-075",
  "status": 2,
  "solution_count": 4,
  "objective": 1514.0,
  "objective_bound": 1514.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1514.0,
  "historical_reference_objective_verified": true
 },
 "OR-076": {
  "problem_id": "OR-076",
  "status": 2,
  "solution_count": 2,
  "objective": 103090.0,
  "objective_bound": 103090.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 103090.0,
  "historical_reference_objective_verified": true
 },
 "OR-077": {
  "problem_id": "OR-077",
  "status": 2,
  "solution_count": 2,
  "objective": 133.0,
  "objective_bound": 133.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 133.0,
  "historical_reference_objective_verified": true
 },
 "OR-078": {
  "problem_id": "OR-078",
  "status": 2,
  "solution_count": 2,
  "objective": 152.76,
  "objective_bound": 152.76,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 152.76,
  "historical_reference_objective_verified": true
 },
 "OR-079": {
  "problem_id": "OR-079",
  "status": 2,
  "solution_count": 5,
  "objective": 2965.0,
  "objective_bound": 2965.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 2965.0,
  "historical_reference_objective_verified": true
 },
 "OR-080": {
  "problem_id": "OR-080",
  "status": 2,
  "solution_count": 7,
  "objective": 14300.0,
  "objective_bound": 14300.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 14300.0,
  "historical_reference_objective_verified": true
 },
 "OR-081": {
  "problem_id": "OR-081",
  "status": 2,
  "solution_count": 1,
  "objective": 3.910164341169253,
  "objective_bound": 3.910164341169253,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 3.910164341169253,
  "historical_reference_objective_verified": true
 },
 "OR-082": {
  "problem_id": "OR-082",
  "status": 2,
  "solution_count": 1,
  "objective": 199.0,
  "objective_bound": 199.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 199.0,
  "historical_reference_objective_verified": true
 },
 "OR-083": {
  "problem_id": "OR-083",
  "status": 2,
  "solution_count": 2,
  "objective": 34.0,
  "objective_bound": 34.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 34.0,
  "historical_reference_objective_verified": true
 },
 "OR-084": {
  "problem_id": "OR-084",
  "status": 2,
  "solution_count": 4,
  "objective": 22.85,
  "objective_bound": 22.85,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 22.85,
  "historical_reference_objective_verified": true
 },
 "OR-085": {
  "problem_id": "OR-085",
  "status": 2,
  "solution_count": 10,
  "objective": 419.0,
  "objective_bound": 419.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 419.0,
  "historical_reference_objective_verified": true
 },
 "OR-086": {
  "problem_id": "OR-086",
  "status": 2,
  "solution_count": 1,
  "objective": 4500.0,
  "objective_bound": 4500.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 4500.0,
  "historical_reference_objective_verified": true
 },
 "OR-087": {
  "problem_id": "OR-087",
  "status": 2,
  "solution_count": 2,
  "objective": 6243055.959999999,
  "objective_bound": 6243055.959999999,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 6243055.960000001,
  "historical_reference_objective_verified": true
 },
 "OR-088": {
  "problem_id": "OR-088",
  "status": 2,
  "solution_count": 7,
  "objective": 150992.5,
  "objective_bound": 150992.5,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 150992.5,
  "historical_reference_objective_verified": true
 },
 "OR-089": {
  "problem_id": "OR-089",
  "status": 2,
  "solution_count": 4,
  "objective": 1555.4999999999993,
  "objective_bound": 1555.4999999999993,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1555.5,
  "historical_reference_objective_verified": true
 },
 "OR-090": {
  "problem_id": "OR-090",
  "status": 2,
  "solution_count": 9,
  "objective": 108659.80000000002,
  "objective_bound": 108667.06586613023,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 6.686802414703586e-05,
  "historical_reference_objective": 108659.8,
  "historical_reference_objective_verified": true
 },
 "OR-091": {
  "problem_id": "OR-091",
  "status": 2,
  "solution_count": 2,
  "objective": 421.0,
  "objective_bound": 421.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 421.0,
  "historical_reference_objective_verified": true
 },
 "OR-092": {
  "problem_id": "OR-092",
  "status": 2,
  "solution_count": 1,
  "objective": 3148.9299999999994,
  "objective_bound": 3148.9299999999994,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 3148.9300000000003,
  "historical_reference_objective_verified": true
 },
 "OR-093": {
  "problem_id": "OR-093",
  "status": 2,
  "solution_count": 2,
  "objective": 1577.6,
  "objective_bound": 1577.6,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1577.6,
  "historical_reference_objective_verified": true
 },
 "OR-094": {
  "problem_id": "OR-094",
  "status": 2,
  "solution_count": 6,
  "objective": 0.599999999999909,
  "objective_bound": 0.599999999999909,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 0.599999999999909,
  "historical_reference_objective_verified": true
 },
 "OR-095": {
  "problem_id": "OR-095",
  "status": 2,
  "solution_count": 1,
  "objective": 1344480.0,
  "objective_bound": 1344480.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1344480.0,
  "historical_reference_objective_verified": true
 },
 "OR-096": {
  "problem_id": "OR-096",
  "status": 2,
  "solution_count": 3,
  "objective": 5044.0,
  "objective_bound": 5044.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 5044.0,
  "historical_reference_objective_verified": true
 },
 "OR-097": {
  "problem_id": "OR-097",
  "status": 2,
  "solution_count": 8,
  "objective": 6.0,
  "objective_bound": 6.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 6.0,
  "historical_reference_objective_verified": true
 },
 "OR-098": {
  "problem_id": "OR-098",
  "status": 2,
  "solution_count": 1,
  "objective": 0.0,
  "objective_bound": 0.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 0.0,
  "historical_reference_objective_verified": true
 },
 "OR-099": {
  "problem_id": "OR-099",
  "status": 2,
  "solution_count": 5,
  "objective": 10419.0,
  "objective_bound": 10419.0,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 10419.0,
  "historical_reference_objective_verified": true
 },
 "OR-100": {
  "problem_id": "OR-100",
  "status": 2,
  "solution_count": 1,
  "objective": 1382200.7222841426,
  "objective_bound": 1382200.7222841426,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 1382200.7222841426,
  "historical_reference_objective_verified": true
 },
 "OR-101": {
  "problem_id": "OR-101",
  "status": 2,
  "solution_count": 1,
  "objective": 25820.15355597092,
  "objective_bound": 25820.15355597092,
  "configured_mip_gap": 0.0001,
  "achieved_mip_gap": 0.0,
  "historical_reference_objective": 25820.15355597092,
  "historical_reference_objective_verified": true
 }
}
""")


IDENTITY_ARTIFACTS = json.loads(r"""
{
 "LEAN-GPT-4.1": {
  "OR-001": {
   "source_code_sha256": "5dea27e38b524dd455e22096203942815c4e8fd91439928ab5afe5555a278a12",
   "original_solution_sha256": "b6d1e0d04a8fbbf0ccbf6d0f0246d7b37ae46aa0bcfc86c385fbae4a1694d871",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S1,C11]"
    ],
    [
     "C11",
     "x_vars[S1,C12]"
    ],
    [
     "C12",
     "x_vars[S1,C13]"
    ],
    [
     "C13",
     "x_vars[S1,C14]"
    ],
    [
     "C14",
     "x_vars[S1,C15]"
    ],
    [
     "C15",
     "x_vars[S1,C16]"
    ],
    [
     "C16",
     "x_vars[S1,C17]"
    ],
    [
     "C17",
     "x_vars[S1,C18]"
    ],
    [
     "C18",
     "x_vars[S2,C1]"
    ],
    [
     "C19",
     "x_vars[S2,C2]"
    ],
    [
     "C20",
     "x_vars[S2,C3]"
    ],
    [
     "C21",
     "x_vars[S2,C4]"
    ],
    [
     "C22",
     "x_vars[S2,C5]"
    ],
    [
     "C23",
     "x_vars[S2,C6]"
    ],
    [
     "C24",
     "x_vars[S2,C7]"
    ],
    [
     "C25",
     "x_vars[S2,C8]"
    ],
    [
     "C26",
     "x_vars[S2,C9]"
    ],
    [
     "C27",
     "x_vars[S2,C10]"
    ],
    [
     "C28",
     "x_vars[S2,C11]"
    ],
    [
     "C29",
     "x_vars[S2,C12]"
    ],
    [
     "C30",
     "x_vars[S2,C13]"
    ],
    [
     "C31",
     "x_vars[S2,C14]"
    ],
    [
     "C32",
     "x_vars[S2,C15]"
    ],
    [
     "C33",
     "x_vars[S2,C16]"
    ],
    [
     "C34",
     "x_vars[S2,C17]"
    ],
    [
     "C35",
     "x_vars[S2,C18]"
    ],
    [
     "C36",
     "x_vars[S3,C1]"
    ],
    [
     "C37",
     "x_vars[S3,C2]"
    ],
    [
     "C38",
     "x_vars[S3,C3]"
    ],
    [
     "C39",
     "x_vars[S3,C4]"
    ],
    [
     "C40",
     "x_vars[S3,C5]"
    ],
    [
     "C41",
     "x_vars[S3,C6]"
    ],
    [
     "C42",
     "x_vars[S3,C7]"
    ],
    [
     "C43",
     "x_vars[S3,C8]"
    ],
    [
     "C44",
     "x_vars[S3,C9]"
    ],
    [
     "C45",
     "x_vars[S3,C10]"
    ],
    [
     "C46",
     "x_vars[S3,C11]"
    ],
    [
     "C47",
     "x_vars[S3,C12]"
    ],
    [
     "C48",
     "x_vars[S3,C13]"
    ],
    [
     "C49",
     "x_vars[S3,C14]"
    ],
    [
     "C50",
     "x_vars[S3,C15]"
    ],
    [
     "C51",
     "x_vars[S3,C16]"
    ],
    [
     "C52",
     "x_vars[S3,C17]"
    ],
    [
     "C53",
     "x_vars[S3,C18]"
    ],
    [
     "C54",
     "x_vars[S4,C1]"
    ],
    [
     "C55",
     "x_vars[S4,C2]"
    ],
    [
     "C56",
     "x_vars[S4,C3]"
    ],
    [
     "C57",
     "x_vars[S4,C4]"
    ],
    [
     "C58",
     "x_vars[S4,C5]"
    ],
    [
     "C59",
     "x_vars[S4,C6]"
    ],
    [
     "C60",
     "x_vars[S4,C7]"
    ],
    [
     "C61",
     "x_vars[S4,C8]"
    ],
    [
     "C62",
     "x_vars[S4,C9]"
    ],
    [
     "C63",
     "x_vars[S4,C10]"
    ],
    [
     "C64",
     "x_vars[S4,C11]"
    ],
    [
     "C65",
     "x_vars[S4,C12]"
    ],
    [
     "C66",
     "x_vars[S4,C13]"
    ],
    [
     "C67",
     "x_vars[S4,C14]"
    ],
    [
     "C68",
     "x_vars[S4,C15]"
    ],
    [
     "C69",
     "x_vars[S4,C16]"
    ],
    [
     "C70",
     "x_vars[S4,C17]"
    ],
    [
     "C71",
     "x_vars[S4,C18]"
    ],
    [
     "C72",
     "x_vars[S5,C1]"
    ],
    [
     "C73",
     "x_vars[S5,C2]"
    ],
    [
     "C74",
     "x_vars[S5,C3]"
    ],
    [
     "C75",
     "x_vars[S5,C4]"
    ],
    [
     "C76",
     "x_vars[S5,C5]"
    ],
    [
     "C77",
     "x_vars[S5,C6]"
    ],
    [
     "C78",
     "x_vars[S5,C7]"
    ],
    [
     "C79",
     "x_vars[S5,C8]"
    ],
    [
     "C80",
     "x_vars[S5,C9]"
    ],
    [
     "C81",
     "x_vars[S5,C10]"
    ],
    [
     "C82",
     "x_vars[S5,C11]"
    ],
    [
     "C83",
     "x_vars[S5,C12]"
    ],
    [
     "C84",
     "x_vars[S5,C13]"
    ],
    [
     "C85",
     "x_vars[S5,C14]"
    ],
    [
     "C86",
     "x_vars[S5,C15]"
    ],
    [
     "C87",
     "x_vars[S5,C16]"
    ],
    [
     "C88",
     "x_vars[S5,C17]"
    ],
    [
     "C89",
     "x_vars[S5,C18]"
    ],
    [
     "C90",
     "x_vars[S6,C1]"
    ],
    [
     "C91",
     "x_vars[S6,C2]"
    ],
    [
     "C92",
     "x_vars[S6,C3]"
    ],
    [
     "C93",
     "x_vars[S6,C4]"
    ],
    [
     "C94",
     "x_vars[S6,C5]"
    ],
    [
     "C95",
     "x_vars[S6,C6]"
    ],
    [
     "C96",
     "x_vars[S6,C7]"
    ],
    [
     "C97",
     "x_vars[S6,C8]"
    ],
    [
     "C98",
     "x_vars[S6,C9]"
    ],
    [
     "C99",
     "x_vars[S6,C10]"
    ],
    [
     "C100",
     "x_vars[S6,C11]"
    ],
    [
     "C101",
     "x_vars[S6,C12]"
    ],
    [
     "C102",
     "x_vars[S6,C13]"
    ],
    [
     "C103",
     "x_vars[S6,C14]"
    ],
    [
     "C104",
     "x_vars[S6,C15]"
    ],
    [
     "C105",
     "x_vars[S6,C16]"
    ],
    [
     "C106",
     "x_vars[S6,C17]"
    ],
    [
     "C107",
     "x_vars[S6,C18]"
    ],
    [
     "C108",
     "x_vars[S7,C1]"
    ],
    [
     "C109",
     "x_vars[S7,C2]"
    ],
    [
     "C110",
     "x_vars[S7,C3]"
    ],
    [
     "C111",
     "x_vars[S7,C4]"
    ],
    [
     "C112",
     "x_vars[S7,C5]"
    ],
    [
     "C113",
     "x_vars[S7,C6]"
    ],
    [
     "C114",
     "x_vars[S7,C7]"
    ],
    [
     "C115",
     "x_vars[S7,C8]"
    ],
    [
     "C116",
     "x_vars[S7,C9]"
    ],
    [
     "C117",
     "x_vars[S7,C10]"
    ],
    [
     "C118",
     "x_vars[S7,C11]"
    ],
    [
     "C119",
     "x_vars[S7,C12]"
    ],
    [
     "C120",
     "x_vars[S7,C13]"
    ],
    [
     "C121",
     "x_vars[S7,C14]"
    ],
    [
     "C122",
     "x_vars[S7,C15]"
    ],
    [
     "C123",
     "x_vars[S7,C16]"
    ],
    [
     "C124",
     "x_vars[S7,C17]"
    ],
    [
     "C125",
     "x_vars[S7,C18]"
    ],
    [
     "C126",
     "x_vars[S8,C1]"
    ],
    [
     "C127",
     "x_vars[S8,C2]"
    ],
    [
     "C128",
     "x_vars[S8,C3]"
    ],
    [
     "C129",
     "x_vars[S8,C4]"
    ],
    [
     "C130",
     "x_vars[S8,C5]"
    ],
    [
     "C131",
     "x_vars[S8,C6]"
    ],
    [
     "C132",
     "x_vars[S8,C7]"
    ],
    [
     "C133",
     "x_vars[S8,C8]"
    ],
    [
     "C134",
     "x_vars[S8,C9]"
    ],
    [
     "C135",
     "x_vars[S8,C10]"
    ],
    [
     "C136",
     "x_vars[S8,C11]"
    ],
    [
     "C137",
     "x_vars[S8,C12]"
    ],
    [
     "C138",
     "x_vars[S8,C13]"
    ],
    [
     "C139",
     "x_vars[S8,C14]"
    ],
    [
     "C140",
     "x_vars[S8,C15]"
    ],
    [
     "C141",
     "x_vars[S8,C16]"
    ],
    [
     "C142",
     "x_vars[S8,C17]"
    ],
    [
     "C143",
     "x_vars[S8,C18]"
    ],
    [
     "C144",
     "x_vars[S9,C1]"
    ],
    [
     "C145",
     "x_vars[S9,C2]"
    ],
    [
     "C146",
     "x_vars[S9,C3]"
    ],
    [
     "C147",
     "x_vars[S9,C4]"
    ],
    [
     "C148",
     "x_vars[S9,C5]"
    ],
    [
     "C149",
     "x_vars[S9,C6]"
    ],
    [
     "C150",
     "x_vars[S9,C7]"
    ],
    [
     "C151",
     "x_vars[S9,C8]"
    ],
    [
     "C152",
     "x_vars[S9,C9]"
    ],
    [
     "C153",
     "x_vars[S9,C10]"
    ],
    [
     "C154",
     "x_vars[S9,C11]"
    ],
    [
     "C155",
     "x_vars[S9,C12]"
    ],
    [
     "C156",
     "x_vars[S9,C13]"
    ],
    [
     "C157",
     "x_vars[S9,C14]"
    ],
    [
     "C158",
     "x_vars[S9,C15]"
    ],
    [
     "C159",
     "x_vars[S9,C16]"
    ],
    [
     "C160",
     "x_vars[S9,C17]"
    ],
    [
     "C161",
     "x_vars[S9,C18]"
    ],
    [
     "C162",
     "x_vars[S10,C1]"
    ],
    [
     "C163",
     "x_vars[S10,C2]"
    ],
    [
     "C164",
     "x_vars[S10,C3]"
    ],
    [
     "C165",
     "x_vars[S10,C4]"
    ],
    [
     "C166",
     "x_vars[S10,C5]"
    ],
    [
     "C167",
     "x_vars[S10,C6]"
    ],
    [
     "C168",
     "x_vars[S10,C7]"
    ],
    [
     "C169",
     "x_vars[S10,C8]"
    ],
    [
     "C170",
     "x_vars[S10,C9]"
    ],
    [
     "C171",
     "x_vars[S10,C10]"
    ],
    [
     "C172",
     "x_vars[S10,C11]"
    ],
    [
     "C173",
     "x_vars[S10,C12]"
    ],
    [
     "C174",
     "x_vars[S10,C13]"
    ],
    [
     "C175",
     "x_vars[S10,C14]"
    ],
    [
     "C176",
     "x_vars[S10,C15]"
    ],
    [
     "C177",
     "x_vars[S10,C16]"
    ],
    [
     "C178",
     "x_vars[S10,C17]"
    ],
    [
     "C179",
     "x_vars[S10,C18]"
    ],
    [
     "C180",
     "x_vars[S11,C1]"
    ],
    [
     "C181",
     "x_vars[S11,C2]"
    ],
    [
     "C182",
     "x_vars[S11,C3]"
    ],
    [
     "C183",
     "x_vars[S11,C4]"
    ],
    [
     "C184",
     "x_vars[S11,C5]"
    ],
    [
     "C185",
     "x_vars[S11,C6]"
    ],
    [
     "C186",
     "x_vars[S11,C7]"
    ],
    [
     "C187",
     "x_vars[S11,C8]"
    ],
    [
     "C188",
     "x_vars[S11,C9]"
    ],
    [
     "C189",
     "x_vars[S11,C10]"
    ],
    [
     "C190",
     "x_vars[S11,C11]"
    ],
    [
     "C191",
     "x_vars[S11,C12]"
    ],
    [
     "C192",
     "x_vars[S11,C13]"
    ],
    [
     "C193",
     "x_vars[S11,C14]"
    ],
    [
     "C194",
     "x_vars[S11,C15]"
    ],
    [
     "C195",
     "x_vars[S11,C16]"
    ],
    [
     "C196",
     "x_vars[S11,C17]"
    ],
    [
     "C197",
     "x_vars[S11,C18]"
    ],
    [
     "C198",
     "x_vars[S12,C1]"
    ],
    [
     "C199",
     "x_vars[S12,C2]"
    ],
    [
     "C200",
     "x_vars[S12,C3]"
    ],
    [
     "C201",
     "x_vars[S12,C4]"
    ],
    [
     "C202",
     "x_vars[S12,C5]"
    ],
    [
     "C203",
     "x_vars[S12,C6]"
    ],
    [
     "C204",
     "x_vars[S12,C7]"
    ],
    [
     "C205",
     "x_vars[S12,C8]"
    ],
    [
     "C206",
     "x_vars[S12,C9]"
    ],
    [
     "C207",
     "x_vars[S12,C10]"
    ],
    [
     "C208",
     "x_vars[S12,C11]"
    ],
    [
     "C209",
     "x_vars[S12,C12]"
    ],
    [
     "C210",
     "x_vars[S12,C13]"
    ],
    [
     "C211",
     "x_vars[S12,C14]"
    ],
    [
     "C212",
     "x_vars[S12,C15]"
    ],
    [
     "C213",
     "x_vars[S12,C16]"
    ],
    [
     "C214",
     "x_vars[S12,C17]"
    ],
    [
     "C215",
     "x_vars[S12,C18]"
    ],
    [
     "C216",
     "x_vars[S13,C1]"
    ],
    [
     "C217",
     "x_vars[S13,C2]"
    ],
    [
     "C218",
     "x_vars[S13,C3]"
    ],
    [
     "C219",
     "x_vars[S13,C4]"
    ],
    [
     "C220",
     "x_vars[S13,C5]"
    ],
    [
     "C221",
     "x_vars[S13,C6]"
    ],
    [
     "C222",
     "x_vars[S13,C7]"
    ],
    [
     "C223",
     "x_vars[S13,C8]"
    ],
    [
     "C224",
     "x_vars[S13,C9]"
    ],
    [
     "C225",
     "x_vars[S13,C10]"
    ],
    [
     "C226",
     "x_vars[S13,C11]"
    ],
    [
     "C227",
     "x_vars[S13,C12]"
    ],
    [
     "C228",
     "x_vars[S13,C13]"
    ],
    [
     "C229",
     "x_vars[S13,C14]"
    ],
    [
     "C230",
     "x_vars[S13,C15]"
    ],
    [
     "C231",
     "x_vars[S13,C16]"
    ],
    [
     "C232",
     "x_vars[S13,C17]"
    ],
    [
     "C233",
     "x_vars[S13,C18]"
    ],
    [
     "C234",
     "x_vars[S14,C1]"
    ],
    [
     "C235",
     "x_vars[S14,C2]"
    ],
    [
     "C236",
     "x_vars[S14,C3]"
    ],
    [
     "C237",
     "x_vars[S14,C4]"
    ],
    [
     "C238",
     "x_vars[S14,C5]"
    ],
    [
     "C239",
     "x_vars[S14,C6]"
    ],
    [
     "C240",
     "x_vars[S14,C7]"
    ],
    [
     "C241",
     "x_vars[S14,C8]"
    ],
    [
     "C242",
     "x_vars[S14,C9]"
    ],
    [
     "C243",
     "x_vars[S14,C10]"
    ],
    [
     "C244",
     "x_vars[S14,C11]"
    ],
    [
     "C245",
     "x_vars[S14,C12]"
    ],
    [
     "C246",
     "x_vars[S14,C13]"
    ],
    [
     "C247",
     "x_vars[S14,C14]"
    ],
    [
     "C248",
     "x_vars[S14,C15]"
    ],
    [
     "C249",
     "x_vars[S14,C16]"
    ],
    [
     "C250",
     "x_vars[S14,C17]"
    ],
    [
     "C251",
     "x_vars[S14,C18]"
    ],
    [
     "C252",
     "x_vars[S15,C1]"
    ],
    [
     "C253",
     "x_vars[S15,C2]"
    ],
    [
     "C254",
     "x_vars[S15,C3]"
    ],
    [
     "C255",
     "x_vars[S15,C4]"
    ],
    [
     "C256",
     "x_vars[S15,C5]"
    ],
    [
     "C257",
     "x_vars[S15,C6]"
    ],
    [
     "C258",
     "x_vars[S15,C7]"
    ],
    [
     "C259",
     "x_vars[S15,C8]"
    ],
    [
     "C260",
     "x_vars[S15,C9]"
    ],
    [
     "C261",
     "x_vars[S15,C10]"
    ],
    [
     "C262",
     "x_vars[S15,C11]"
    ],
    [
     "C263",
     "x_vars[S15,C12]"
    ],
    [
     "C264",
     "x_vars[S15,C13]"
    ],
    [
     "C265",
     "x_vars[S15,C14]"
    ],
    [
     "C266",
     "x_vars[S15,C15]"
    ],
    [
     "C267",
     "x_vars[S15,C16]"
    ],
    [
     "C268",
     "x_vars[S15,C17]"
    ],
    [
     "C269",
     "x_vars[S15,C18]"
    ],
    [
     "C270",
     "x_vars[S16,C1]"
    ],
    [
     "C271",
     "x_vars[S16,C2]"
    ],
    [
     "C272",
     "x_vars[S16,C3]"
    ],
    [
     "C273",
     "x_vars[S16,C4]"
    ],
    [
     "C274",
     "x_vars[S16,C5]"
    ],
    [
     "C275",
     "x_vars[S16,C6]"
    ],
    [
     "C276",
     "x_vars[S16,C7]"
    ],
    [
     "C277",
     "x_vars[S16,C8]"
    ],
    [
     "C278",
     "x_vars[S16,C9]"
    ],
    [
     "C279",
     "x_vars[S16,C10]"
    ],
    [
     "C280",
     "x_vars[S16,C11]"
    ],
    [
     "C281",
     "x_vars[S16,C12]"
    ],
    [
     "C282",
     "x_vars[S16,C13]"
    ],
    [
     "C283",
     "x_vars[S16,C14]"
    ],
    [
     "C284",
     "x_vars[S16,C15]"
    ],
    [
     "C285",
     "x_vars[S16,C16]"
    ],
    [
     "C286",
     "x_vars[S16,C17]"
    ],
    [
     "C287",
     "x_vars[S16,C18]"
    ],
    [
     "C288",
     "x_vars[S17,C1]"
    ],
    [
     "C289",
     "x_vars[S17,C2]"
    ],
    [
     "C290",
     "x_vars[S17,C3]"
    ],
    [
     "C291",
     "x_vars[S17,C4]"
    ],
    [
     "C292",
     "x_vars[S17,C5]"
    ],
    [
     "C293",
     "x_vars[S17,C6]"
    ],
    [
     "C294",
     "x_vars[S17,C7]"
    ],
    [
     "C295",
     "x_vars[S17,C8]"
    ],
    [
     "C296",
     "x_vars[S17,C9]"
    ],
    [
     "C297",
     "x_vars[S17,C10]"
    ],
    [
     "C298",
     "x_vars[S17,C11]"
    ],
    [
     "C299",
     "x_vars[S17,C12]"
    ],
    [
     "C300",
     "x_vars[S17,C13]"
    ],
    [
     "C301",
     "x_vars[S17,C14]"
    ],
    [
     "C302",
     "x_vars[S17,C15]"
    ],
    [
     "C303",
     "x_vars[S17,C16]"
    ],
    [
     "C304",
     "x_vars[S17,C17]"
    ],
    [
     "C305",
     "x_vars[S17,C18]"
    ],
    [
     "C306",
     "x_vars[S18,C1]"
    ],
    [
     "C307",
     "x_vars[S18,C2]"
    ],
    [
     "C308",
     "x_vars[S18,C3]"
    ],
    [
     "C309",
     "x_vars[S18,C4]"
    ],
    [
     "C310",
     "x_vars[S18,C5]"
    ],
    [
     "C311",
     "x_vars[S18,C6]"
    ],
    [
     "C312",
     "x_vars[S18,C7]"
    ],
    [
     "C313",
     "x_vars[S18,C8]"
    ],
    [
     "C314",
     "x_vars[S18,C9]"
    ],
    [
     "C315",
     "x_vars[S18,C10]"
    ],
    [
     "C316",
     "x_vars[S18,C11]"
    ],
    [
     "C317",
     "x_vars[S18,C12]"
    ],
    [
     "C318",
     "x_vars[S18,C13]"
    ],
    [
     "C319",
     "x_vars[S18,C14]"
    ],
    [
     "C320",
     "x_vars[S18,C15]"
    ],
    [
     "C321",
     "x_vars[S18,C16]"
    ],
    [
     "C322",
     "x_vars[S18,C17]"
    ],
    [
     "C323",
     "x_vars[S18,C18]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_0_8": "x_vars[S1,C9]",
    "x_0_9": "x_vars[S1,C10]",
    "x_0_10": "x_vars[S1,C11]",
    "x_0_11": "x_vars[S1,C12]",
    "x_0_12": "x_vars[S1,C13]",
    "x_0_13": "x_vars[S1,C14]",
    "x_0_14": "x_vars[S1,C15]",
    "x_0_15": "x_vars[S1,C16]",
    "x_0_16": "x_vars[S1,C17]",
    "x_0_17": "x_vars[S1,C18]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_1_7": "x_vars[S2,C8]",
    "x_1_8": "x_vars[S2,C9]",
    "x_1_9": "x_vars[S2,C10]",
    "x_1_10": "x_vars[S2,C11]",
    "x_1_11": "x_vars[S2,C12]",
    "x_1_12": "x_vars[S2,C13]",
    "x_1_13": "x_vars[S2,C14]",
    "x_1_14": "x_vars[S2,C15]",
    "x_1_15": "x_vars[S2,C16]",
    "x_1_16": "x_vars[S2,C17]",
    "x_1_17": "x_vars[S2,C18]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_2_10": "x_vars[S3,C11]",
    "x_2_11": "x_vars[S3,C12]",
    "x_2_12": "x_vars[S3,C13]",
    "x_2_13": "x_vars[S3,C14]",
    "x_2_14": "x_vars[S3,C15]",
    "x_2_15": "x_vars[S3,C16]",
    "x_2_16": "x_vars[S3,C17]",
    "x_2_17": "x_vars[S3,C18]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_3_10": "x_vars[S4,C11]",
    "x_3_11": "x_vars[S4,C12]",
    "x_3_12": "x_vars[S4,C13]",
    "x_3_13": "x_vars[S4,C14]",
    "x_3_14": "x_vars[S4,C15]",
    "x_3_15": "x_vars[S4,C16]",
    "x_3_16": "x_vars[S4,C17]",
    "x_3_17": "x_vars[S4,C18]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_4_10": "x_vars[S5,C11]",
    "x_4_11": "x_vars[S5,C12]",
    "x_4_12": "x_vars[S5,C13]",
    "x_4_13": "x_vars[S5,C14]",
    "x_4_14": "x_vars[S5,C15]",
    "x_4_15": "x_vars[S5,C16]",
    "x_4_16": "x_vars[S5,C17]",
    "x_4_17": "x_vars[S5,C18]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_5_10": "x_vars[S6,C11]",
    "x_5_11": "x_vars[S6,C12]",
    "x_5_12": "x_vars[S6,C13]",
    "x_5_13": "x_vars[S6,C14]",
    "x_5_14": "x_vars[S6,C15]",
    "x_5_15": "x_vars[S6,C16]",
    "x_5_16": "x_vars[S6,C17]",
    "x_5_17": "x_vars[S6,C18]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "x_6_7": "x_vars[S7,C8]",
    "x_6_8": "x_vars[S7,C9]",
    "x_6_9": "x_vars[S7,C10]",
    "x_6_10": "x_vars[S7,C11]",
    "x_6_11": "x_vars[S7,C12]",
    "x_6_12": "x_vars[S7,C13]",
    "x_6_13": "x_vars[S7,C14]",
    "x_6_14": "x_vars[S7,C15]",
    "x_6_15": "x_vars[S7,C16]",
    "x_6_16": "x_vars[S7,C17]",
    "x_6_17": "x_vars[S7,C18]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_7_10": "x_vars[S8,C11]",
    "x_7_11": "x_vars[S8,C12]",
    "x_7_12": "x_vars[S8,C13]",
    "x_7_13": "x_vars[S8,C14]",
    "x_7_14": "x_vars[S8,C15]",
    "x_7_15": "x_vars[S8,C16]",
    "x_7_16": "x_vars[S8,C17]",
    "x_7_17": "x_vars[S8,C18]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_1": "x_vars[S9,C2]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_3": "x_vars[S9,C4]",
    "x_8_4": "x_vars[S9,C5]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_6": "x_vars[S9,C7]",
    "x_8_7": "x_vars[S9,C8]",
    "x_8_8": "x_vars[S9,C9]",
    "x_8_9": "x_vars[S9,C10]",
    "x_8_10": "x_vars[S9,C11]",
    "x_8_11": "x_vars[S9,C12]",
    "x_8_12": "x_vars[S9,C13]",
    "x_8_13": "x_vars[S9,C14]",
    "x_8_14": "x_vars[S9,C15]",
    "x_8_15": "x_vars[S9,C16]",
    "x_8_16": "x_vars[S9,C17]",
    "x_8_17": "x_vars[S9,C18]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_1": "x_vars[S10,C2]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_3": "x_vars[S10,C4]",
    "x_9_4": "x_vars[S10,C5]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_6": "x_vars[S10,C7]",
    "x_9_7": "x_vars[S10,C8]",
    "x_9_8": "x_vars[S10,C9]",
    "x_9_9": "x_vars[S10,C10]",
    "x_9_10": "x_vars[S10,C11]",
    "x_9_11": "x_vars[S10,C12]",
    "x_9_12": "x_vars[S10,C13]",
    "x_9_13": "x_vars[S10,C14]",
    "x_9_14": "x_vars[S10,C15]",
    "x_9_15": "x_vars[S10,C16]",
    "x_9_16": "x_vars[S10,C17]",
    "x_9_17": "x_vars[S10,C18]",
    "x_10_0": "x_vars[S11,C1]",
    "x_10_1": "x_vars[S11,C2]",
    "x_10_2": "x_vars[S11,C3]",
    "x_10_3": "x_vars[S11,C4]",
    "x_10_4": "x_vars[S11,C5]",
    "x_10_5": "x_vars[S11,C6]",
    "x_10_6": "x_vars[S11,C7]",
    "x_10_7": "x_vars[S11,C8]",
    "x_10_8": "x_vars[S11,C9]",
    "x_10_9": "x_vars[S11,C10]",
    "x_10_10": "x_vars[S11,C11]",
    "x_10_11": "x_vars[S11,C12]",
    "x_10_12": "x_vars[S11,C13]",
    "x_10_13": "x_vars[S11,C14]",
    "x_10_14": "x_vars[S11,C15]",
    "x_10_15": "x_vars[S11,C16]",
    "x_10_16": "x_vars[S11,C17]",
    "x_10_17": "x_vars[S11,C18]",
    "x_11_0": "x_vars[S12,C1]",
    "x_11_1": "x_vars[S12,C2]",
    "x_11_2": "x_vars[S12,C3]",
    "x_11_3": "x_vars[S12,C4]",
    "x_11_4": "x_vars[S12,C5]",
    "x_11_5": "x_vars[S12,C6]",
    "x_11_6": "x_vars[S12,C7]",
    "x_11_7": "x_vars[S12,C8]",
    "x_11_8": "x_vars[S12,C9]",
    "x_11_9": "x_vars[S12,C10]",
    "x_11_10": "x_vars[S12,C11]",
    "x_11_11": "x_vars[S12,C12]",
    "x_11_12": "x_vars[S12,C13]",
    "x_11_13": "x_vars[S12,C14]",
    "x_11_14": "x_vars[S12,C15]",
    "x_11_15": "x_vars[S12,C16]",
    "x_11_16": "x_vars[S12,C17]",
    "x_11_17": "x_vars[S12,C18]",
    "x_12_0": "x_vars[S13,C1]",
    "x_12_1": "x_vars[S13,C2]",
    "x_12_2": "x_vars[S13,C3]",
    "x_12_3": "x_vars[S13,C4]",
    "x_12_4": "x_vars[S13,C5]",
    "x_12_5": "x_vars[S13,C6]",
    "x_12_6": "x_vars[S13,C7]",
    "x_12_7": "x_vars[S13,C8]",
    "x_12_8": "x_vars[S13,C9]",
    "x_12_9": "x_vars[S13,C10]",
    "x_12_10": "x_vars[S13,C11]",
    "x_12_11": "x_vars[S13,C12]",
    "x_12_12": "x_vars[S13,C13]",
    "x_12_13": "x_vars[S13,C14]",
    "x_12_14": "x_vars[S13,C15]",
    "x_12_15": "x_vars[S13,C16]",
    "x_12_16": "x_vars[S13,C17]",
    "x_12_17": "x_vars[S13,C18]",
    "x_13_0": "x_vars[S14,C1]",
    "x_13_1": "x_vars[S14,C2]",
    "x_13_2": "x_vars[S14,C3]",
    "x_13_3": "x_vars[S14,C4]",
    "x_13_4": "x_vars[S14,C5]",
    "x_13_5": "x_vars[S14,C6]",
    "x_13_6": "x_vars[S14,C7]",
    "x_13_7": "x_vars[S14,C8]",
    "x_13_8": "x_vars[S14,C9]",
    "x_13_9": "x_vars[S14,C10]",
    "x_13_10": "x_vars[S14,C11]",
    "x_13_11": "x_vars[S14,C12]",
    "x_13_12": "x_vars[S14,C13]",
    "x_13_13": "x_vars[S14,C14]",
    "x_13_14": "x_vars[S14,C15]",
    "x_13_15": "x_vars[S14,C16]",
    "x_13_16": "x_vars[S14,C17]",
    "x_13_17": "x_vars[S14,C18]",
    "x_14_0": "x_vars[S15,C1]",
    "x_14_1": "x_vars[S15,C2]",
    "x_14_2": "x_vars[S15,C3]",
    "x_14_3": "x_vars[S15,C4]",
    "x_14_4": "x_vars[S15,C5]",
    "x_14_5": "x_vars[S15,C6]",
    "x_14_6": "x_vars[S15,C7]",
    "x_14_7": "x_vars[S15,C8]",
    "x_14_8": "x_vars[S15,C9]",
    "x_14_9": "x_vars[S15,C10]",
    "x_14_10": "x_vars[S15,C11]",
    "x_14_11": "x_vars[S15,C12]",
    "x_14_12": "x_vars[S15,C13]",
    "x_14_13": "x_vars[S15,C14]",
    "x_14_14": "x_vars[S15,C15]",
    "x_14_15": "x_vars[S15,C16]",
    "x_14_16": "x_vars[S15,C17]",
    "x_14_17": "x_vars[S15,C18]",
    "x_15_0": "x_vars[S16,C1]",
    "x_15_1": "x_vars[S16,C2]",
    "x_15_2": "x_vars[S16,C3]",
    "x_15_3": "x_vars[S16,C4]",
    "x_15_4": "x_vars[S16,C5]",
    "x_15_5": "x_vars[S16,C6]",
    "x_15_6": "x_vars[S16,C7]",
    "x_15_7": "x_vars[S16,C8]",
    "x_15_8": "x_vars[S16,C9]",
    "x_15_9": "x_vars[S16,C10]",
    "x_15_10": "x_vars[S16,C11]",
    "x_15_11": "x_vars[S16,C12]",
    "x_15_12": "x_vars[S16,C13]",
    "x_15_13": "x_vars[S16,C14]",
    "x_15_14": "x_vars[S16,C15]",
    "x_15_15": "x_vars[S16,C16]",
    "x_15_16": "x_vars[S16,C17]",
    "x_15_17": "x_vars[S16,C18]",
    "x_16_0": "x_vars[S17,C1]",
    "x_16_1": "x_vars[S17,C2]",
    "x_16_2": "x_vars[S17,C3]",
    "x_16_3": "x_vars[S17,C4]",
    "x_16_4": "x_vars[S17,C5]",
    "x_16_5": "x_vars[S17,C6]",
    "x_16_6": "x_vars[S17,C7]",
    "x_16_7": "x_vars[S17,C8]",
    "x_16_8": "x_vars[S17,C9]",
    "x_16_9": "x_vars[S17,C10]",
    "x_16_10": "x_vars[S17,C11]",
    "x_16_11": "x_vars[S17,C12]",
    "x_16_12": "x_vars[S17,C13]",
    "x_16_13": "x_vars[S17,C14]",
    "x_16_14": "x_vars[S17,C15]",
    "x_16_15": "x_vars[S17,C16]",
    "x_16_16": "x_vars[S17,C17]",
    "x_16_17": "x_vars[S17,C18]",
    "x_17_0": "x_vars[S18,C1]",
    "x_17_1": "x_vars[S18,C2]",
    "x_17_2": "x_vars[S18,C3]",
    "x_17_3": "x_vars[S18,C4]",
    "x_17_4": "x_vars[S18,C5]",
    "x_17_5": "x_vars[S18,C6]",
    "x_17_6": "x_vars[S18,C7]",
    "x_17_7": "x_vars[S18,C8]",
    "x_17_8": "x_vars[S18,C9]",
    "x_17_9": "x_vars[S18,C10]",
    "x_17_10": "x_vars[S18,C11]",
    "x_17_11": "x_vars[S18,C12]",
    "x_17_12": "x_vars[S18,C13]",
    "x_17_13": "x_vars[S18,C14]",
    "x_17_14": "x_vars[S18,C15]",
    "x_17_15": "x_vars[S18,C16]",
    "x_17_16": "x_vars[S18,C17]",
    "x_17_17": "x_vars[S18,C18]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-002": {
   "source_code_sha256": "1f90c79b2083e10e2390aba027f5be2c3ebc45b30e763c3a4fc929fb39c5c8f9",
   "original_solution_sha256": "b69dadf89947a0ef575782b3cf72fcd95b8609b6eb70a5015203d3669730a53c",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S1,C11]"
    ],
    [
     "C11",
     "x_vars[S1,C12]"
    ],
    [
     "C12",
     "x_vars[S2,C1]"
    ],
    [
     "C13",
     "x_vars[S2,C2]"
    ],
    [
     "C14",
     "x_vars[S2,C3]"
    ],
    [
     "C15",
     "x_vars[S2,C4]"
    ],
    [
     "C16",
     "x_vars[S2,C5]"
    ],
    [
     "C17",
     "x_vars[S2,C6]"
    ],
    [
     "C18",
     "x_vars[S2,C7]"
    ],
    [
     "C19",
     "x_vars[S2,C8]"
    ],
    [
     "C20",
     "x_vars[S2,C9]"
    ],
    [
     "C21",
     "x_vars[S2,C10]"
    ],
    [
     "C22",
     "x_vars[S2,C11]"
    ],
    [
     "C23",
     "x_vars[S2,C12]"
    ],
    [
     "C24",
     "x_vars[S3,C1]"
    ],
    [
     "C25",
     "x_vars[S3,C2]"
    ],
    [
     "C26",
     "x_vars[S3,C3]"
    ],
    [
     "C27",
     "x_vars[S3,C4]"
    ],
    [
     "C28",
     "x_vars[S3,C5]"
    ],
    [
     "C29",
     "x_vars[S3,C6]"
    ],
    [
     "C30",
     "x_vars[S3,C7]"
    ],
    [
     "C31",
     "x_vars[S3,C8]"
    ],
    [
     "C32",
     "x_vars[S3,C9]"
    ],
    [
     "C33",
     "x_vars[S3,C10]"
    ],
    [
     "C34",
     "x_vars[S3,C11]"
    ],
    [
     "C35",
     "x_vars[S3,C12]"
    ],
    [
     "C36",
     "x_vars[S4,C1]"
    ],
    [
     "C37",
     "x_vars[S4,C2]"
    ],
    [
     "C38",
     "x_vars[S4,C3]"
    ],
    [
     "C39",
     "x_vars[S4,C4]"
    ],
    [
     "C40",
     "x_vars[S4,C5]"
    ],
    [
     "C41",
     "x_vars[S4,C6]"
    ],
    [
     "C42",
     "x_vars[S4,C7]"
    ],
    [
     "C43",
     "x_vars[S4,C8]"
    ],
    [
     "C44",
     "x_vars[S4,C9]"
    ],
    [
     "C45",
     "x_vars[S4,C10]"
    ],
    [
     "C46",
     "x_vars[S4,C11]"
    ],
    [
     "C47",
     "x_vars[S4,C12]"
    ],
    [
     "C48",
     "x_vars[S5,C1]"
    ],
    [
     "C49",
     "x_vars[S5,C2]"
    ],
    [
     "C50",
     "x_vars[S5,C3]"
    ],
    [
     "C51",
     "x_vars[S5,C4]"
    ],
    [
     "C52",
     "x_vars[S5,C5]"
    ],
    [
     "C53",
     "x_vars[S5,C6]"
    ],
    [
     "C54",
     "x_vars[S5,C7]"
    ],
    [
     "C55",
     "x_vars[S5,C8]"
    ],
    [
     "C56",
     "x_vars[S5,C9]"
    ],
    [
     "C57",
     "x_vars[S5,C10]"
    ],
    [
     "C58",
     "x_vars[S5,C11]"
    ],
    [
     "C59",
     "x_vars[S5,C12]"
    ],
    [
     "C60",
     "x_vars[S6,C1]"
    ],
    [
     "C61",
     "x_vars[S6,C2]"
    ],
    [
     "C62",
     "x_vars[S6,C3]"
    ],
    [
     "C63",
     "x_vars[S6,C4]"
    ],
    [
     "C64",
     "x_vars[S6,C5]"
    ],
    [
     "C65",
     "x_vars[S6,C6]"
    ],
    [
     "C66",
     "x_vars[S6,C7]"
    ],
    [
     "C67",
     "x_vars[S6,C8]"
    ],
    [
     "C68",
     "x_vars[S6,C9]"
    ],
    [
     "C69",
     "x_vars[S6,C10]"
    ],
    [
     "C70",
     "x_vars[S6,C11]"
    ],
    [
     "C71",
     "x_vars[S6,C12]"
    ],
    [
     "C72",
     "x_vars[S7,C1]"
    ],
    [
     "C73",
     "x_vars[S7,C2]"
    ],
    [
     "C74",
     "x_vars[S7,C3]"
    ],
    [
     "C75",
     "x_vars[S7,C4]"
    ],
    [
     "C76",
     "x_vars[S7,C5]"
    ],
    [
     "C77",
     "x_vars[S7,C6]"
    ],
    [
     "C78",
     "x_vars[S7,C7]"
    ],
    [
     "C79",
     "x_vars[S7,C8]"
    ],
    [
     "C80",
     "x_vars[S7,C9]"
    ],
    [
     "C81",
     "x_vars[S7,C10]"
    ],
    [
     "C82",
     "x_vars[S7,C11]"
    ],
    [
     "C83",
     "x_vars[S7,C12]"
    ],
    [
     "C84",
     "x_vars[S8,C1]"
    ],
    [
     "C85",
     "x_vars[S8,C2]"
    ],
    [
     "C86",
     "x_vars[S8,C3]"
    ],
    [
     "C87",
     "x_vars[S8,C4]"
    ],
    [
     "C88",
     "x_vars[S8,C5]"
    ],
    [
     "C89",
     "x_vars[S8,C6]"
    ],
    [
     "C90",
     "x_vars[S8,C7]"
    ],
    [
     "C91",
     "x_vars[S8,C8]"
    ],
    [
     "C92",
     "x_vars[S8,C9]"
    ],
    [
     "C93",
     "x_vars[S8,C10]"
    ],
    [
     "C94",
     "x_vars[S8,C11]"
    ],
    [
     "C95",
     "x_vars[S8,C12]"
    ],
    [
     "C96",
     "x_vars[S9,C1]"
    ],
    [
     "C97",
     "x_vars[S9,C2]"
    ],
    [
     "C98",
     "x_vars[S9,C3]"
    ],
    [
     "C99",
     "x_vars[S9,C4]"
    ],
    [
     "C100",
     "x_vars[S9,C5]"
    ],
    [
     "C101",
     "x_vars[S9,C6]"
    ],
    [
     "C102",
     "x_vars[S9,C7]"
    ],
    [
     "C103",
     "x_vars[S9,C8]"
    ],
    [
     "C104",
     "x_vars[S9,C9]"
    ],
    [
     "C105",
     "x_vars[S9,C10]"
    ],
    [
     "C106",
     "x_vars[S9,C11]"
    ],
    [
     "C107",
     "x_vars[S9,C12]"
    ],
    [
     "C108",
     "x_vars[S10,C1]"
    ],
    [
     "C109",
     "x_vars[S10,C2]"
    ],
    [
     "C110",
     "x_vars[S10,C3]"
    ],
    [
     "C111",
     "x_vars[S10,C4]"
    ],
    [
     "C112",
     "x_vars[S10,C5]"
    ],
    [
     "C113",
     "x_vars[S10,C6]"
    ],
    [
     "C114",
     "x_vars[S10,C7]"
    ],
    [
     "C115",
     "x_vars[S10,C8]"
    ],
    [
     "C116",
     "x_vars[S10,C9]"
    ],
    [
     "C117",
     "x_vars[S10,C10]"
    ],
    [
     "C118",
     "x_vars[S10,C11]"
    ],
    [
     "C119",
     "x_vars[S10,C12]"
    ],
    [
     "C120",
     "x_vars[S11,C1]"
    ],
    [
     "C121",
     "x_vars[S11,C2]"
    ],
    [
     "C122",
     "x_vars[S11,C3]"
    ],
    [
     "C123",
     "x_vars[S11,C4]"
    ],
    [
     "C124",
     "x_vars[S11,C5]"
    ],
    [
     "C125",
     "x_vars[S11,C6]"
    ],
    [
     "C126",
     "x_vars[S11,C7]"
    ],
    [
     "C127",
     "x_vars[S11,C8]"
    ],
    [
     "C128",
     "x_vars[S11,C9]"
    ],
    [
     "C129",
     "x_vars[S11,C10]"
    ],
    [
     "C130",
     "x_vars[S11,C11]"
    ],
    [
     "C131",
     "x_vars[S11,C12]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_0_8": "x_vars[S1,C9]",
    "x_0_9": "x_vars[S1,C10]",
    "x_0_10": "x_vars[S1,C11]",
    "x_0_11": "x_vars[S1,C12]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_1_7": "x_vars[S2,C8]",
    "x_1_8": "x_vars[S2,C9]",
    "x_1_9": "x_vars[S2,C10]",
    "x_1_10": "x_vars[S2,C11]",
    "x_1_11": "x_vars[S2,C12]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_2_10": "x_vars[S3,C11]",
    "x_2_11": "x_vars[S3,C12]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_3_10": "x_vars[S4,C11]",
    "x_3_11": "x_vars[S4,C12]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_4_10": "x_vars[S5,C11]",
    "x_4_11": "x_vars[S5,C12]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_5_10": "x_vars[S6,C11]",
    "x_5_11": "x_vars[S6,C12]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "x_6_7": "x_vars[S7,C8]",
    "x_6_8": "x_vars[S7,C9]",
    "x_6_9": "x_vars[S7,C10]",
    "x_6_10": "x_vars[S7,C11]",
    "x_6_11": "x_vars[S7,C12]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_7_10": "x_vars[S8,C11]",
    "x_7_11": "x_vars[S8,C12]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_1": "x_vars[S9,C2]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_3": "x_vars[S9,C4]",
    "x_8_4": "x_vars[S9,C5]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_6": "x_vars[S9,C7]",
    "x_8_7": "x_vars[S9,C8]",
    "x_8_8": "x_vars[S9,C9]",
    "x_8_9": "x_vars[S9,C10]",
    "x_8_10": "x_vars[S9,C11]",
    "x_8_11": "x_vars[S9,C12]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_1": "x_vars[S10,C2]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_3": "x_vars[S10,C4]",
    "x_9_4": "x_vars[S10,C5]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_6": "x_vars[S10,C7]",
    "x_9_7": "x_vars[S10,C8]",
    "x_9_8": "x_vars[S10,C9]",
    "x_9_9": "x_vars[S10,C10]",
    "x_9_10": "x_vars[S10,C11]",
    "x_9_11": "x_vars[S10,C12]",
    "x_10_0": "x_vars[S11,C1]",
    "x_10_1": "x_vars[S11,C2]",
    "x_10_2": "x_vars[S11,C3]",
    "x_10_3": "x_vars[S11,C4]",
    "x_10_4": "x_vars[S11,C5]",
    "x_10_5": "x_vars[S11,C6]",
    "x_10_6": "x_vars[S11,C7]",
    "x_10_7": "x_vars[S11,C8]",
    "x_10_8": "x_vars[S11,C9]",
    "x_10_9": "x_vars[S11,C10]",
    "x_10_10": "x_vars[S11,C11]",
    "x_10_11": "x_vars[S11,C12]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-003": {
   "source_code_sha256": "c0d0c8ca0bac72135aec87f9a25b41912212aed968d1db7b8f7cd8b954e912fa",
   "original_solution_sha256": "8595ff8cc874b03b28a1a1da093c74144fe369f0871dd21a8de22e8489fb5031",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S2,C1]"
    ],
    [
     "C11",
     "x_vars[S2,C2]"
    ],
    [
     "C12",
     "x_vars[S2,C3]"
    ],
    [
     "C13",
     "x_vars[S2,C4]"
    ],
    [
     "C14",
     "x_vars[S2,C5]"
    ],
    [
     "C15",
     "x_vars[S2,C6]"
    ],
    [
     "C16",
     "x_vars[S2,C7]"
    ],
    [
     "C17",
     "x_vars[S2,C8]"
    ],
    [
     "C18",
     "x_vars[S2,C9]"
    ],
    [
     "C19",
     "x_vars[S2,C10]"
    ],
    [
     "C20",
     "x_vars[S3,C1]"
    ],
    [
     "C21",
     "x_vars[S3,C2]"
    ],
    [
     "C22",
     "x_vars[S3,C3]"
    ],
    [
     "C23",
     "x_vars[S3,C4]"
    ],
    [
     "C24",
     "x_vars[S3,C5]"
    ],
    [
     "C25",
     "x_vars[S3,C6]"
    ],
    [
     "C26",
     "x_vars[S3,C7]"
    ],
    [
     "C27",
     "x_vars[S3,C8]"
    ],
    [
     "C28",
     "x_vars[S3,C9]"
    ],
    [
     "C29",
     "x_vars[S3,C10]"
    ],
    [
     "C30",
     "x_vars[S4,C1]"
    ],
    [
     "C31",
     "x_vars[S4,C2]"
    ],
    [
     "C32",
     "x_vars[S4,C3]"
    ],
    [
     "C33",
     "x_vars[S4,C4]"
    ],
    [
     "C34",
     "x_vars[S4,C5]"
    ],
    [
     "C35",
     "x_vars[S4,C6]"
    ],
    [
     "C36",
     "x_vars[S4,C7]"
    ],
    [
     "C37",
     "x_vars[S4,C8]"
    ],
    [
     "C38",
     "x_vars[S4,C9]"
    ],
    [
     "C39",
     "x_vars[S4,C10]"
    ],
    [
     "C40",
     "x_vars[S5,C1]"
    ],
    [
     "C41",
     "x_vars[S5,C2]"
    ],
    [
     "C42",
     "x_vars[S5,C3]"
    ],
    [
     "C43",
     "x_vars[S5,C4]"
    ],
    [
     "C44",
     "x_vars[S5,C5]"
    ],
    [
     "C45",
     "x_vars[S5,C6]"
    ],
    [
     "C46",
     "x_vars[S5,C7]"
    ],
    [
     "C47",
     "x_vars[S5,C8]"
    ],
    [
     "C48",
     "x_vars[S5,C9]"
    ],
    [
     "C49",
     "x_vars[S5,C10]"
    ],
    [
     "C50",
     "x_vars[S6,C1]"
    ],
    [
     "C51",
     "x_vars[S6,C2]"
    ],
    [
     "C52",
     "x_vars[S6,C3]"
    ],
    [
     "C53",
     "x_vars[S6,C4]"
    ],
    [
     "C54",
     "x_vars[S6,C5]"
    ],
    [
     "C55",
     "x_vars[S6,C6]"
    ],
    [
     "C56",
     "x_vars[S6,C7]"
    ],
    [
     "C57",
     "x_vars[S6,C8]"
    ],
    [
     "C58",
     "x_vars[S6,C9]"
    ],
    [
     "C59",
     "x_vars[S6,C10]"
    ],
    [
     "C60",
     "x_vars[S7,C1]"
    ],
    [
     "C61",
     "x_vars[S7,C2]"
    ],
    [
     "C62",
     "x_vars[S7,C3]"
    ],
    [
     "C63",
     "x_vars[S7,C4]"
    ],
    [
     "C64",
     "x_vars[S7,C5]"
    ],
    [
     "C65",
     "x_vars[S7,C6]"
    ],
    [
     "C66",
     "x_vars[S7,C7]"
    ],
    [
     "C67",
     "x_vars[S7,C8]"
    ],
    [
     "C68",
     "x_vars[S7,C9]"
    ],
    [
     "C69",
     "x_vars[S7,C10]"
    ],
    [
     "C70",
     "x_vars[S8,C1]"
    ],
    [
     "C71",
     "x_vars[S8,C2]"
    ],
    [
     "C72",
     "x_vars[S8,C3]"
    ],
    [
     "C73",
     "x_vars[S8,C4]"
    ],
    [
     "C74",
     "x_vars[S8,C5]"
    ],
    [
     "C75",
     "x_vars[S8,C6]"
    ],
    [
     "C76",
     "x_vars[S8,C7]"
    ],
    [
     "C77",
     "x_vars[S8,C8]"
    ],
    [
     "C78",
     "x_vars[S8,C9]"
    ],
    [
     "C79",
     "x_vars[S8,C10]"
    ],
    [
     "C80",
     "x_vars[S9,C1]"
    ],
    [
     "C81",
     "x_vars[S9,C2]"
    ],
    [
     "C82",
     "x_vars[S9,C3]"
    ],
    [
     "C83",
     "x_vars[S9,C4]"
    ],
    [
     "C84",
     "x_vars[S9,C5]"
    ],
    [
     "C85",
     "x_vars[S9,C6]"
    ],
    [
     "C86",
     "x_vars[S9,C7]"
    ],
    [
     "C87",
     "x_vars[S9,C8]"
    ],
    [
     "C88",
     "x_vars[S9,C9]"
    ],
    [
     "C89",
     "x_vars[S9,C10]"
    ],
    [
     "C90",
     "x_vars[S10,C1]"
    ],
    [
     "C91",
     "x_vars[S10,C2]"
    ],
    [
     "C92",
     "x_vars[S10,C3]"
    ],
    [
     "C93",
     "x_vars[S10,C4]"
    ],
    [
     "C94",
     "x_vars[S10,C5]"
    ],
    [
     "C95",
     "x_vars[S10,C6]"
    ],
    [
     "C96",
     "x_vars[S10,C7]"
    ],
    [
     "C97",
     "x_vars[S10,C8]"
    ],
    [
     "C98",
     "x_vars[S10,C9]"
    ],
    [
     "C99",
     "x_vars[S10,C10]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_0_8": "x_vars[S1,C9]",
    "x_0_9": "x_vars[S1,C10]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_1_7": "x_vars[S2,C8]",
    "x_1_8": "x_vars[S2,C9]",
    "x_1_9": "x_vars[S2,C10]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "x_6_7": "x_vars[S7,C8]",
    "x_6_8": "x_vars[S7,C9]",
    "x_6_9": "x_vars[S7,C10]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_1": "x_vars[S9,C2]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_3": "x_vars[S9,C4]",
    "x_8_4": "x_vars[S9,C5]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_6": "x_vars[S9,C7]",
    "x_8_7": "x_vars[S9,C8]",
    "x_8_8": "x_vars[S9,C9]",
    "x_8_9": "x_vars[S9,C10]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_1": "x_vars[S10,C2]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_3": "x_vars[S10,C4]",
    "x_9_4": "x_vars[S10,C5]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_6": "x_vars[S10,C7]",
    "x_9_7": "x_vars[S10,C8]",
    "x_9_8": "x_vars[S10,C9]",
    "x_9_9": "x_vars[S10,C10]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-004": {
   "source_code_sha256": "0d2519be77f43c4bb9627f8066d19735973f7c6e19ea753da3c7f009949c4df0",
   "original_solution_sha256": "38f1e81fb6054e108af14a4b7625e64c33a20f61dc0ecf751ec3e531ef66e660",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S1,C11]"
    ],
    [
     "C11",
     "x_vars[S1,C12]"
    ],
    [
     "C12",
     "x_vars[S2,C1]"
    ],
    [
     "C13",
     "x_vars[S2,C2]"
    ],
    [
     "C14",
     "x_vars[S2,C3]"
    ],
    [
     "C15",
     "x_vars[S2,C4]"
    ],
    [
     "C16",
     "x_vars[S2,C5]"
    ],
    [
     "C17",
     "x_vars[S2,C6]"
    ],
    [
     "C18",
     "x_vars[S2,C7]"
    ],
    [
     "C19",
     "x_vars[S2,C8]"
    ],
    [
     "C20",
     "x_vars[S2,C9]"
    ],
    [
     "C21",
     "x_vars[S2,C10]"
    ],
    [
     "C22",
     "x_vars[S2,C11]"
    ],
    [
     "C23",
     "x_vars[S2,C12]"
    ],
    [
     "C24",
     "x_vars[S3,C1]"
    ],
    [
     "C25",
     "x_vars[S3,C2]"
    ],
    [
     "C26",
     "x_vars[S3,C3]"
    ],
    [
     "C27",
     "x_vars[S3,C4]"
    ],
    [
     "C28",
     "x_vars[S3,C5]"
    ],
    [
     "C29",
     "x_vars[S3,C6]"
    ],
    [
     "C30",
     "x_vars[S3,C7]"
    ],
    [
     "C31",
     "x_vars[S3,C8]"
    ],
    [
     "C32",
     "x_vars[S3,C9]"
    ],
    [
     "C33",
     "x_vars[S3,C10]"
    ],
    [
     "C34",
     "x_vars[S3,C11]"
    ],
    [
     "C35",
     "x_vars[S3,C12]"
    ],
    [
     "C36",
     "x_vars[S4,C1]"
    ],
    [
     "C37",
     "x_vars[S4,C2]"
    ],
    [
     "C38",
     "x_vars[S4,C3]"
    ],
    [
     "C39",
     "x_vars[S4,C4]"
    ],
    [
     "C40",
     "x_vars[S4,C5]"
    ],
    [
     "C41",
     "x_vars[S4,C6]"
    ],
    [
     "C42",
     "x_vars[S4,C7]"
    ],
    [
     "C43",
     "x_vars[S4,C8]"
    ],
    [
     "C44",
     "x_vars[S4,C9]"
    ],
    [
     "C45",
     "x_vars[S4,C10]"
    ],
    [
     "C46",
     "x_vars[S4,C11]"
    ],
    [
     "C47",
     "x_vars[S4,C12]"
    ],
    [
     "C48",
     "x_vars[S5,C1]"
    ],
    [
     "C49",
     "x_vars[S5,C2]"
    ],
    [
     "C50",
     "x_vars[S5,C3]"
    ],
    [
     "C51",
     "x_vars[S5,C4]"
    ],
    [
     "C52",
     "x_vars[S5,C5]"
    ],
    [
     "C53",
     "x_vars[S5,C6]"
    ],
    [
     "C54",
     "x_vars[S5,C7]"
    ],
    [
     "C55",
     "x_vars[S5,C8]"
    ],
    [
     "C56",
     "x_vars[S5,C9]"
    ],
    [
     "C57",
     "x_vars[S5,C10]"
    ],
    [
     "C58",
     "x_vars[S5,C11]"
    ],
    [
     "C59",
     "x_vars[S5,C12]"
    ],
    [
     "C60",
     "x_vars[S6,C1]"
    ],
    [
     "C61",
     "x_vars[S6,C2]"
    ],
    [
     "C62",
     "x_vars[S6,C3]"
    ],
    [
     "C63",
     "x_vars[S6,C4]"
    ],
    [
     "C64",
     "x_vars[S6,C5]"
    ],
    [
     "C65",
     "x_vars[S6,C6]"
    ],
    [
     "C66",
     "x_vars[S6,C7]"
    ],
    [
     "C67",
     "x_vars[S6,C8]"
    ],
    [
     "C68",
     "x_vars[S6,C9]"
    ],
    [
     "C69",
     "x_vars[S6,C10]"
    ],
    [
     "C70",
     "x_vars[S6,C11]"
    ],
    [
     "C71",
     "x_vars[S6,C12]"
    ],
    [
     "C72",
     "x_vars[S7,C1]"
    ],
    [
     "C73",
     "x_vars[S7,C2]"
    ],
    [
     "C74",
     "x_vars[S7,C3]"
    ],
    [
     "C75",
     "x_vars[S7,C4]"
    ],
    [
     "C76",
     "x_vars[S7,C5]"
    ],
    [
     "C77",
     "x_vars[S7,C6]"
    ],
    [
     "C78",
     "x_vars[S7,C7]"
    ],
    [
     "C79",
     "x_vars[S7,C8]"
    ],
    [
     "C80",
     "x_vars[S7,C9]"
    ],
    [
     "C81",
     "x_vars[S7,C10]"
    ],
    [
     "C82",
     "x_vars[S7,C11]"
    ],
    [
     "C83",
     "x_vars[S7,C12]"
    ],
    [
     "C84",
     "x_vars[S8,C1]"
    ],
    [
     "C85",
     "x_vars[S8,C2]"
    ],
    [
     "C86",
     "x_vars[S8,C3]"
    ],
    [
     "C87",
     "x_vars[S8,C4]"
    ],
    [
     "C88",
     "x_vars[S8,C5]"
    ],
    [
     "C89",
     "x_vars[S8,C6]"
    ],
    [
     "C90",
     "x_vars[S8,C7]"
    ],
    [
     "C91",
     "x_vars[S8,C8]"
    ],
    [
     "C92",
     "x_vars[S8,C9]"
    ],
    [
     "C93",
     "x_vars[S8,C10]"
    ],
    [
     "C94",
     "x_vars[S8,C11]"
    ],
    [
     "C95",
     "x_vars[S8,C12]"
    ],
    [
     "C96",
     "x_vars[S9,C1]"
    ],
    [
     "C97",
     "x_vars[S9,C2]"
    ],
    [
     "C98",
     "x_vars[S9,C3]"
    ],
    [
     "C99",
     "x_vars[S9,C4]"
    ],
    [
     "C100",
     "x_vars[S9,C5]"
    ],
    [
     "C101",
     "x_vars[S9,C6]"
    ],
    [
     "C102",
     "x_vars[S9,C7]"
    ],
    [
     "C103",
     "x_vars[S9,C8]"
    ],
    [
     "C104",
     "x_vars[S9,C9]"
    ],
    [
     "C105",
     "x_vars[S9,C10]"
    ],
    [
     "C106",
     "x_vars[S9,C11]"
    ],
    [
     "C107",
     "x_vars[S9,C12]"
    ],
    [
     "C108",
     "x_vars[S10,C1]"
    ],
    [
     "C109",
     "x_vars[S10,C2]"
    ],
    [
     "C110",
     "x_vars[S10,C3]"
    ],
    [
     "C111",
     "x_vars[S10,C4]"
    ],
    [
     "C112",
     "x_vars[S10,C5]"
    ],
    [
     "C113",
     "x_vars[S10,C6]"
    ],
    [
     "C114",
     "x_vars[S10,C7]"
    ],
    [
     "C115",
     "x_vars[S10,C8]"
    ],
    [
     "C116",
     "x_vars[S10,C9]"
    ],
    [
     "C117",
     "x_vars[S10,C10]"
    ],
    [
     "C118",
     "x_vars[S10,C11]"
    ],
    [
     "C119",
     "x_vars[S10,C12]"
    ],
    [
     "C120",
     "x_vars[S11,C1]"
    ],
    [
     "C121",
     "x_vars[S11,C2]"
    ],
    [
     "C122",
     "x_vars[S11,C3]"
    ],
    [
     "C123",
     "x_vars[S11,C4]"
    ],
    [
     "C124",
     "x_vars[S11,C5]"
    ],
    [
     "C125",
     "x_vars[S11,C6]"
    ],
    [
     "C126",
     "x_vars[S11,C7]"
    ],
    [
     "C127",
     "x_vars[S11,C8]"
    ],
    [
     "C128",
     "x_vars[S11,C9]"
    ],
    [
     "C129",
     "x_vars[S11,C10]"
    ],
    [
     "C130",
     "x_vars[S11,C11]"
    ],
    [
     "C131",
     "x_vars[S11,C12]"
    ],
    [
     "C132",
     "x_vars[S12,C1]"
    ],
    [
     "C133",
     "x_vars[S12,C2]"
    ],
    [
     "C134",
     "x_vars[S12,C3]"
    ],
    [
     "C135",
     "x_vars[S12,C4]"
    ],
    [
     "C136",
     "x_vars[S12,C5]"
    ],
    [
     "C137",
     "x_vars[S12,C6]"
    ],
    [
     "C138",
     "x_vars[S12,C7]"
    ],
    [
     "C139",
     "x_vars[S12,C8]"
    ],
    [
     "C140",
     "x_vars[S12,C9]"
    ],
    [
     "C141",
     "x_vars[S12,C10]"
    ],
    [
     "C142",
     "x_vars[S12,C11]"
    ],
    [
     "C143",
     "x_vars[S12,C12]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_0_8": "x_vars[S1,C9]",
    "x_0_9": "x_vars[S1,C10]",
    "x_0_10": "x_vars[S1,C11]",
    "x_0_11": "x_vars[S1,C12]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_1_7": "x_vars[S2,C8]",
    "x_1_8": "x_vars[S2,C9]",
    "x_1_9": "x_vars[S2,C10]",
    "x_1_10": "x_vars[S2,C11]",
    "x_1_11": "x_vars[S2,C12]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_2_10": "x_vars[S3,C11]",
    "x_2_11": "x_vars[S3,C12]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_3_10": "x_vars[S4,C11]",
    "x_3_11": "x_vars[S4,C12]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_4_10": "x_vars[S5,C11]",
    "x_4_11": "x_vars[S5,C12]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_5_10": "x_vars[S6,C11]",
    "x_5_11": "x_vars[S6,C12]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "x_6_7": "x_vars[S7,C8]",
    "x_6_8": "x_vars[S7,C9]",
    "x_6_9": "x_vars[S7,C10]",
    "x_6_10": "x_vars[S7,C11]",
    "x_6_11": "x_vars[S7,C12]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_7_10": "x_vars[S8,C11]",
    "x_7_11": "x_vars[S8,C12]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_1": "x_vars[S9,C2]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_3": "x_vars[S9,C4]",
    "x_8_4": "x_vars[S9,C5]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_6": "x_vars[S9,C7]",
    "x_8_7": "x_vars[S9,C8]",
    "x_8_8": "x_vars[S9,C9]",
    "x_8_9": "x_vars[S9,C10]",
    "x_8_10": "x_vars[S9,C11]",
    "x_8_11": "x_vars[S9,C12]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_1": "x_vars[S10,C2]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_3": "x_vars[S10,C4]",
    "x_9_4": "x_vars[S10,C5]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_6": "x_vars[S10,C7]",
    "x_9_7": "x_vars[S10,C8]",
    "x_9_8": "x_vars[S10,C9]",
    "x_9_9": "x_vars[S10,C10]",
    "x_9_10": "x_vars[S10,C11]",
    "x_9_11": "x_vars[S10,C12]",
    "x_10_0": "x_vars[S11,C1]",
    "x_10_1": "x_vars[S11,C2]",
    "x_10_2": "x_vars[S11,C3]",
    "x_10_3": "x_vars[S11,C4]",
    "x_10_4": "x_vars[S11,C5]",
    "x_10_5": "x_vars[S11,C6]",
    "x_10_6": "x_vars[S11,C7]",
    "x_10_7": "x_vars[S11,C8]",
    "x_10_8": "x_vars[S11,C9]",
    "x_10_9": "x_vars[S11,C10]",
    "x_10_10": "x_vars[S11,C11]",
    "x_10_11": "x_vars[S11,C12]",
    "x_11_0": "x_vars[S12,C1]",
    "x_11_1": "x_vars[S12,C2]",
    "x_11_2": "x_vars[S12,C3]",
    "x_11_3": "x_vars[S12,C4]",
    "x_11_4": "x_vars[S12,C5]",
    "x_11_5": "x_vars[S12,C6]",
    "x_11_6": "x_vars[S12,C7]",
    "x_11_7": "x_vars[S12,C8]",
    "x_11_8": "x_vars[S12,C9]",
    "x_11_9": "x_vars[S12,C10]",
    "x_11_10": "x_vars[S12,C11]",
    "x_11_11": "x_vars[S12,C12]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-005": {
   "source_code_sha256": "a5c5e45f8f8fcda550a0c85c269e41c5624620733cf52464f65585959dd75d32",
   "original_solution_sha256": "989e223f54ab77751bee58028a934fde31a33086f26e206f1ba847b7b280a029",
   "bindings": [
    [
     "C0",
     "x_vars[supplier1,demand1]"
    ],
    [
     "C1",
     "x_vars[supplier1,demand2]"
    ],
    [
     "C2",
     "x_vars[supplier1,demand3]"
    ],
    [
     "C3",
     "x_vars[supplier1,demand4]"
    ],
    [
     "C4",
     "x_vars[supplier1,demand5]"
    ],
    [
     "C5",
     "x_vars[supplier1,demand6]"
    ],
    [
     "C6",
     "x_vars[supplier1,demand7]"
    ],
    [
     "C7",
     "x_vars[supplier1,demand8]"
    ],
    [
     "C8",
     "x_vars[supplier2,demand1]"
    ],
    [
     "C9",
     "x_vars[supplier2,demand2]"
    ],
    [
     "C10",
     "x_vars[supplier2,demand3]"
    ],
    [
     "C11",
     "x_vars[supplier2,demand4]"
    ],
    [
     "C12",
     "x_vars[supplier2,demand5]"
    ],
    [
     "C13",
     "x_vars[supplier2,demand6]"
    ],
    [
     "C14",
     "x_vars[supplier2,demand7]"
    ],
    [
     "C15",
     "x_vars[supplier2,demand8]"
    ],
    [
     "C16",
     "x_vars[supplier3,demand1]"
    ],
    [
     "C17",
     "x_vars[supplier3,demand2]"
    ],
    [
     "C18",
     "x_vars[supplier3,demand3]"
    ],
    [
     "C19",
     "x_vars[supplier3,demand4]"
    ],
    [
     "C20",
     "x_vars[supplier3,demand5]"
    ],
    [
     "C21",
     "x_vars[supplier3,demand6]"
    ],
    [
     "C22",
     "x_vars[supplier3,demand7]"
    ],
    [
     "C23",
     "x_vars[supplier3,demand8]"
    ],
    [
     "C24",
     "x_vars[supplier4,demand1]"
    ],
    [
     "C25",
     "x_vars[supplier4,demand2]"
    ],
    [
     "C26",
     "x_vars[supplier4,demand3]"
    ],
    [
     "C27",
     "x_vars[supplier4,demand4]"
    ],
    [
     "C28",
     "x_vars[supplier4,demand5]"
    ],
    [
     "C29",
     "x_vars[supplier4,demand6]"
    ],
    [
     "C30",
     "x_vars[supplier4,demand7]"
    ],
    [
     "C31",
     "x_vars[supplier4,demand8]"
    ],
    [
     "C32",
     "x_vars[supplier5,demand1]"
    ],
    [
     "C33",
     "x_vars[supplier5,demand2]"
    ],
    [
     "C34",
     "x_vars[supplier5,demand3]"
    ],
    [
     "C35",
     "x_vars[supplier5,demand4]"
    ],
    [
     "C36",
     "x_vars[supplier5,demand5]"
    ],
    [
     "C37",
     "x_vars[supplier5,demand6]"
    ],
    [
     "C38",
     "x_vars[supplier5,demand7]"
    ],
    [
     "C39",
     "x_vars[supplier5,demand8]"
    ],
    [
     "C40",
     "x_vars[supplier6,demand1]"
    ],
    [
     "C41",
     "x_vars[supplier6,demand2]"
    ],
    [
     "C42",
     "x_vars[supplier6,demand3]"
    ],
    [
     "C43",
     "x_vars[supplier6,demand4]"
    ],
    [
     "C44",
     "x_vars[supplier6,demand5]"
    ],
    [
     "C45",
     "x_vars[supplier6,demand6]"
    ],
    [
     "C46",
     "x_vars[supplier6,demand7]"
    ],
    [
     "C47",
     "x_vars[supplier6,demand8]"
    ],
    [
     "C48",
     "x_vars[supplier7,demand1]"
    ],
    [
     "C49",
     "x_vars[supplier7,demand2]"
    ],
    [
     "C50",
     "x_vars[supplier7,demand3]"
    ],
    [
     "C51",
     "x_vars[supplier7,demand4]"
    ],
    [
     "C52",
     "x_vars[supplier7,demand5]"
    ],
    [
     "C53",
     "x_vars[supplier7,demand6]"
    ],
    [
     "C54",
     "x_vars[supplier7,demand7]"
    ],
    [
     "C55",
     "x_vars[supplier7,demand8]"
    ],
    [
     "C56",
     "x_vars[supplier8,demand1]"
    ],
    [
     "C57",
     "x_vars[supplier8,demand2]"
    ],
    [
     "C58",
     "x_vars[supplier8,demand3]"
    ],
    [
     "C59",
     "x_vars[supplier8,demand4]"
    ],
    [
     "C60",
     "x_vars[supplier8,demand5]"
    ],
    [
     "C61",
     "x_vars[supplier8,demand6]"
    ],
    [
     "C62",
     "x_vars[supplier8,demand7]"
    ],
    [
     "C63",
     "x_vars[supplier8,demand8]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[supplier1,demand1]",
    "x_0_1": "x_vars[supplier1,demand2]",
    "x_0_2": "x_vars[supplier1,demand3]",
    "x_0_3": "x_vars[supplier1,demand4]",
    "x_0_4": "x_vars[supplier1,demand5]",
    "x_0_5": "x_vars[supplier1,demand6]",
    "x_0_6": "x_vars[supplier1,demand7]",
    "x_0_7": "x_vars[supplier1,demand8]",
    "x_1_0": "x_vars[supplier2,demand1]",
    "x_1_1": "x_vars[supplier2,demand2]",
    "x_1_2": "x_vars[supplier2,demand3]",
    "x_1_3": "x_vars[supplier2,demand4]",
    "x_1_4": "x_vars[supplier2,demand5]",
    "x_1_5": "x_vars[supplier2,demand6]",
    "x_1_6": "x_vars[supplier2,demand7]",
    "x_1_7": "x_vars[supplier2,demand8]",
    "x_2_0": "x_vars[supplier3,demand1]",
    "x_2_1": "x_vars[supplier3,demand2]",
    "x_2_2": "x_vars[supplier3,demand3]",
    "x_2_3": "x_vars[supplier3,demand4]",
    "x_2_4": "x_vars[supplier3,demand5]",
    "x_2_5": "x_vars[supplier3,demand6]",
    "x_2_6": "x_vars[supplier3,demand7]",
    "x_2_7": "x_vars[supplier3,demand8]",
    "x_3_0": "x_vars[supplier4,demand1]",
    "x_3_1": "x_vars[supplier4,demand2]",
    "x_3_2": "x_vars[supplier4,demand3]",
    "x_3_3": "x_vars[supplier4,demand4]",
    "x_3_4": "x_vars[supplier4,demand5]",
    "x_3_5": "x_vars[supplier4,demand6]",
    "x_3_6": "x_vars[supplier4,demand7]",
    "x_3_7": "x_vars[supplier4,demand8]",
    "x_4_0": "x_vars[supplier5,demand1]",
    "x_4_1": "x_vars[supplier5,demand2]",
    "x_4_2": "x_vars[supplier5,demand3]",
    "x_4_3": "x_vars[supplier5,demand4]",
    "x_4_4": "x_vars[supplier5,demand5]",
    "x_4_5": "x_vars[supplier5,demand6]",
    "x_4_6": "x_vars[supplier5,demand7]",
    "x_4_7": "x_vars[supplier5,demand8]",
    "x_5_0": "x_vars[supplier6,demand1]",
    "x_5_1": "x_vars[supplier6,demand2]",
    "x_5_2": "x_vars[supplier6,demand3]",
    "x_5_3": "x_vars[supplier6,demand4]",
    "x_5_4": "x_vars[supplier6,demand5]",
    "x_5_5": "x_vars[supplier6,demand6]",
    "x_5_6": "x_vars[supplier6,demand7]",
    "x_5_7": "x_vars[supplier6,demand8]",
    "x_6_0": "x_vars[supplier7,demand1]",
    "x_6_1": "x_vars[supplier7,demand2]",
    "x_6_2": "x_vars[supplier7,demand3]",
    "x_6_3": "x_vars[supplier7,demand4]",
    "x_6_4": "x_vars[supplier7,demand5]",
    "x_6_5": "x_vars[supplier7,demand6]",
    "x_6_6": "x_vars[supplier7,demand7]",
    "x_6_7": "x_vars[supplier7,demand8]",
    "x_7_0": "x_vars[supplier8,demand1]",
    "x_7_1": "x_vars[supplier8,demand2]",
    "x_7_2": "x_vars[supplier8,demand3]",
    "x_7_3": "x_vars[supplier8,demand4]",
    "x_7_4": "x_vars[supplier8,demand5]",
    "x_7_5": "x_vars[supplier8,demand6]",
    "x_7_6": "x_vars[supplier8,demand7]",
    "x_7_7": "x_vars[supplier8,demand8]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-006": {
   "source_code_sha256": "e8e519c8f667d5448aa1129b9f401247263a0b5f8b98d0bcdde9570238c662f4",
   "original_solution_sha256": "408badf617759d1d597d06caee6f8c313f246f59585721b7cce5056f9a2cb7d4",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S2,C1]"
    ],
    [
     "C11",
     "x_vars[S2,C2]"
    ],
    [
     "C12",
     "x_vars[S2,C3]"
    ],
    [
     "C13",
     "x_vars[S2,C4]"
    ],
    [
     "C14",
     "x_vars[S2,C5]"
    ],
    [
     "C15",
     "x_vars[S2,C6]"
    ],
    [
     "C16",
     "x_vars[S2,C7]"
    ],
    [
     "C17",
     "x_vars[S2,C8]"
    ],
    [
     "C18",
     "x_vars[S2,C9]"
    ],
    [
     "C19",
     "x_vars[S2,C10]"
    ],
    [
     "C20",
     "x_vars[S3,C1]"
    ],
    [
     "C21",
     "x_vars[S3,C2]"
    ],
    [
     "C22",
     "x_vars[S3,C3]"
    ],
    [
     "C23",
     "x_vars[S3,C4]"
    ],
    [
     "C24",
     "x_vars[S3,C5]"
    ],
    [
     "C25",
     "x_vars[S3,C6]"
    ],
    [
     "C26",
     "x_vars[S3,C7]"
    ],
    [
     "C27",
     "x_vars[S3,C8]"
    ],
    [
     "C28",
     "x_vars[S3,C9]"
    ],
    [
     "C29",
     "x_vars[S3,C10]"
    ],
    [
     "C30",
     "x_vars[S4,C1]"
    ],
    [
     "C31",
     "x_vars[S4,C2]"
    ],
    [
     "C32",
     "x_vars[S4,C3]"
    ],
    [
     "C33",
     "x_vars[S4,C4]"
    ],
    [
     "C34",
     "x_vars[S4,C5]"
    ],
    [
     "C35",
     "x_vars[S4,C6]"
    ],
    [
     "C36",
     "x_vars[S4,C7]"
    ],
    [
     "C37",
     "x_vars[S4,C8]"
    ],
    [
     "C38",
     "x_vars[S4,C9]"
    ],
    [
     "C39",
     "x_vars[S4,C10]"
    ],
    [
     "C40",
     "x_vars[S5,C1]"
    ],
    [
     "C41",
     "x_vars[S5,C2]"
    ],
    [
     "C42",
     "x_vars[S5,C3]"
    ],
    [
     "C43",
     "x_vars[S5,C4]"
    ],
    [
     "C44",
     "x_vars[S5,C5]"
    ],
    [
     "C45",
     "x_vars[S5,C6]"
    ],
    [
     "C46",
     "x_vars[S5,C7]"
    ],
    [
     "C47",
     "x_vars[S5,C8]"
    ],
    [
     "C48",
     "x_vars[S5,C9]"
    ],
    [
     "C49",
     "x_vars[S5,C10]"
    ],
    [
     "C50",
     "x_vars[S6,C1]"
    ],
    [
     "C51",
     "x_vars[S6,C2]"
    ],
    [
     "C52",
     "x_vars[S6,C3]"
    ],
    [
     "C53",
     "x_vars[S6,C4]"
    ],
    [
     "C54",
     "x_vars[S6,C5]"
    ],
    [
     "C55",
     "x_vars[S6,C6]"
    ],
    [
     "C56",
     "x_vars[S6,C7]"
    ],
    [
     "C57",
     "x_vars[S6,C8]"
    ],
    [
     "C58",
     "x_vars[S6,C9]"
    ],
    [
     "C59",
     "x_vars[S6,C10]"
    ],
    [
     "C60",
     "x_vars[S7,C1]"
    ],
    [
     "C61",
     "x_vars[S7,C2]"
    ],
    [
     "C62",
     "x_vars[S7,C3]"
    ],
    [
     "C63",
     "x_vars[S7,C4]"
    ],
    [
     "C64",
     "x_vars[S7,C5]"
    ],
    [
     "C65",
     "x_vars[S7,C6]"
    ],
    [
     "C66",
     "x_vars[S7,C7]"
    ],
    [
     "C67",
     "x_vars[S7,C8]"
    ],
    [
     "C68",
     "x_vars[S7,C9]"
    ],
    [
     "C69",
     "x_vars[S7,C10]"
    ],
    [
     "C70",
     "x_vars[S8,C1]"
    ],
    [
     "C71",
     "x_vars[S8,C2]"
    ],
    [
     "C72",
     "x_vars[S8,C3]"
    ],
    [
     "C73",
     "x_vars[S8,C4]"
    ],
    [
     "C74",
     "x_vars[S8,C5]"
    ],
    [
     "C75",
     "x_vars[S8,C6]"
    ],
    [
     "C76",
     "x_vars[S8,C7]"
    ],
    [
     "C77",
     "x_vars[S8,C8]"
    ],
    [
     "C78",
     "x_vars[S8,C9]"
    ],
    [
     "C79",
     "x_vars[S8,C10]"
    ],
    [
     "C80",
     "x_vars[S9,C1]"
    ],
    [
     "C81",
     "x_vars[S9,C2]"
    ],
    [
     "C82",
     "x_vars[S9,C3]"
    ],
    [
     "C83",
     "x_vars[S9,C4]"
    ],
    [
     "C84",
     "x_vars[S9,C5]"
    ],
    [
     "C85",
     "x_vars[S9,C6]"
    ],
    [
     "C86",
     "x_vars[S9,C7]"
    ],
    [
     "C87",
     "x_vars[S9,C8]"
    ],
    [
     "C88",
     "x_vars[S9,C9]"
    ],
    [
     "C89",
     "x_vars[S9,C10]"
    ],
    [
     "C90",
     "x_vars[S10,C1]"
    ],
    [
     "C91",
     "x_vars[S10,C2]"
    ],
    [
     "C92",
     "x_vars[S10,C3]"
    ],
    [
     "C93",
     "x_vars[S10,C4]"
    ],
    [
     "C94",
     "x_vars[S10,C5]"
    ],
    [
     "C95",
     "x_vars[S10,C6]"
    ],
    [
     "C96",
     "x_vars[S10,C7]"
    ],
    [
     "C97",
     "x_vars[S10,C8]"
    ],
    [
     "C98",
     "x_vars[S10,C9]"
    ],
    [
     "C99",
     "x_vars[S10,C10]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_8": "x_vars[S1,C9]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_8": "x_vars[S2,C9]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_8": "x_vars[S7,C9]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_8": "x_vars[S9,C9]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_8": "x_vars[S10,C9]",
    "x_0_1": "x_vars[S1,C2]",
    "x_1_1": "x_vars[S2,C2]",
    "x_6_1": "x_vars[S7,C2]",
    "x_8_1": "x_vars[S9,C2]",
    "x_9_1": "x_vars[S10,C2]",
    "x_0_3": "x_vars[S1,C4]",
    "x_1_3": "x_vars[S2,C4]",
    "x_6_3": "x_vars[S7,C4]",
    "x_8_3": "x_vars[S9,C4]",
    "x_9_3": "x_vars[S10,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_1_4": "x_vars[S2,C5]",
    "x_6_4": "x_vars[S7,C5]",
    "x_8_4": "x_vars[S9,C5]",
    "x_9_4": "x_vars[S10,C5]",
    "x_0_6": "x_vars[S1,C7]",
    "x_1_6": "x_vars[S2,C7]",
    "x_6_6": "x_vars[S7,C7]",
    "x_8_6": "x_vars[S9,C7]",
    "x_9_6": "x_vars[S10,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_1_7": "x_vars[S2,C8]",
    "x_6_7": "x_vars[S7,C8]",
    "x_8_7": "x_vars[S9,C8]",
    "x_9_7": "x_vars[S10,C8]",
    "x_0_9": "x_vars[S1,C10]",
    "x_1_9": "x_vars[S2,C10]",
    "x_6_9": "x_vars[S7,C10]",
    "x_8_9": "x_vars[S9,C10]",
    "x_9_9": "x_vars[S10,C10]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-007": {
   "source_code_sha256": "8b1f81d0212ef48d67b44f34322c9605a989219e705491315e0ce9ea67b70a1f",
   "original_solution_sha256": "35f7e8418b62a6aa52e6a6a758ea99034d20855b05651a939bb6f779c8a0809e",
   "bindings": [
    [
     "C0",
     "x_vars[S1,D1]"
    ],
    [
     "C1",
     "x_vars[S1,D2]"
    ],
    [
     "C2",
     "x_vars[S1,D3]"
    ],
    [
     "C3",
     "x_vars[S1,D4]"
    ],
    [
     "C4",
     "x_vars[S1,D5]"
    ],
    [
     "C5",
     "x_vars[S2,D1]"
    ],
    [
     "C6",
     "x_vars[S2,D2]"
    ],
    [
     "C7",
     "x_vars[S2,D3]"
    ],
    [
     "C8",
     "x_vars[S2,D4]"
    ],
    [
     "C9",
     "x_vars[S2,D5]"
    ],
    [
     "C10",
     "x_vars[S3,D1]"
    ],
    [
     "C11",
     "x_vars[S3,D2]"
    ],
    [
     "C12",
     "x_vars[S3,D3]"
    ],
    [
     "C13",
     "x_vars[S3,D4]"
    ],
    [
     "C14",
     "x_vars[S3,D5]"
    ],
    [
     "C15",
     "x_vars[S4,D1]"
    ],
    [
     "C16",
     "x_vars[S4,D2]"
    ],
    [
     "C17",
     "x_vars[S4,D3]"
    ],
    [
     "C18",
     "x_vars[S4,D4]"
    ],
    [
     "C19",
     "x_vars[S4,D5]"
    ],
    [
     "C20",
     "x_vars[S5,D1]"
    ],
    [
     "C21",
     "x_vars[S5,D2]"
    ],
    [
     "C22",
     "x_vars[S5,D3]"
    ],
    [
     "C23",
     "x_vars[S5,D4]"
    ],
    [
     "C24",
     "x_vars[S5,D5]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,D1]",
    "x_0_1": "x_vars[S1,D2]",
    "x_0_2": "x_vars[S1,D3]",
    "x_0_3": "x_vars[S1,D4]",
    "x_0_4": "x_vars[S1,D5]",
    "x_1_0": "x_vars[S2,D1]",
    "x_1_1": "x_vars[S2,D2]",
    "x_1_2": "x_vars[S2,D3]",
    "x_1_3": "x_vars[S2,D4]",
    "x_1_4": "x_vars[S2,D5]",
    "x_2_0": "x_vars[S3,D1]",
    "x_2_1": "x_vars[S3,D2]",
    "x_2_2": "x_vars[S3,D3]",
    "x_2_3": "x_vars[S3,D4]",
    "x_2_4": "x_vars[S3,D5]",
    "x_3_0": "x_vars[S4,D1]",
    "x_3_1": "x_vars[S4,D2]",
    "x_3_2": "x_vars[S4,D3]",
    "x_3_3": "x_vars[S4,D4]",
    "x_3_4": "x_vars[S4,D5]",
    "x_4_0": "x_vars[S5,D1]",
    "x_4_1": "x_vars[S5,D2]",
    "x_4_2": "x_vars[S5,D3]",
    "x_4_3": "x_vars[S5,D4]",
    "x_4_4": "x_vars[S5,D5]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-008": {
   "source_code_sha256": "96804db2e68ef1003482abd7d0846f6f139facabaee7a08b99f1bbadafd9c7eb",
   "original_solution_sha256": "5d45a995db09f4196573b0de17bc12170a9cad541aca69ded95259fd83ef6909",
   "bindings": [
    [
     "C0",
     "x_vars[Supplier1,Customer1]"
    ],
    [
     "C1",
     "x_vars[Supplier1,Customer2]"
    ],
    [
     "C2",
     "x_vars[Supplier1,Customer3]"
    ],
    [
     "C3",
     "x_vars[Supplier1,Customer4]"
    ],
    [
     "C4",
     "x_vars[Supplier1,Customer5]"
    ],
    [
     "C5",
     "x_vars[Supplier1,Customer6]"
    ],
    [
     "C6",
     "x_vars[Supplier2,Customer1]"
    ],
    [
     "C7",
     "x_vars[Supplier2,Customer2]"
    ],
    [
     "C8",
     "x_vars[Supplier2,Customer3]"
    ],
    [
     "C9",
     "x_vars[Supplier2,Customer4]"
    ],
    [
     "C10",
     "x_vars[Supplier2,Customer5]"
    ],
    [
     "C11",
     "x_vars[Supplier2,Customer6]"
    ],
    [
     "C12",
     "x_vars[Supplier3,Customer1]"
    ],
    [
     "C13",
     "x_vars[Supplier3,Customer2]"
    ],
    [
     "C14",
     "x_vars[Supplier3,Customer3]"
    ],
    [
     "C15",
     "x_vars[Supplier3,Customer4]"
    ],
    [
     "C16",
     "x_vars[Supplier3,Customer5]"
    ],
    [
     "C17",
     "x_vars[Supplier3,Customer6]"
    ],
    [
     "C18",
     "x_vars[Supplier4,Customer1]"
    ],
    [
     "C19",
     "x_vars[Supplier4,Customer2]"
    ],
    [
     "C20",
     "x_vars[Supplier4,Customer3]"
    ],
    [
     "C21",
     "x_vars[Supplier4,Customer4]"
    ],
    [
     "C22",
     "x_vars[Supplier4,Customer5]"
    ],
    [
     "C23",
     "x_vars[Supplier4,Customer6]"
    ],
    [
     "C24",
     "x_vars[Supplier5,Customer1]"
    ],
    [
     "C25",
     "x_vars[Supplier5,Customer2]"
    ],
    [
     "C26",
     "x_vars[Supplier5,Customer3]"
    ],
    [
     "C27",
     "x_vars[Supplier5,Customer4]"
    ],
    [
     "C28",
     "x_vars[Supplier5,Customer5]"
    ],
    [
     "C29",
     "x_vars[Supplier5,Customer6]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Supplier1,Customer1]",
    "x_0_1": "x_vars[Supplier1,Customer2]",
    "x_0_2": "x_vars[Supplier1,Customer3]",
    "x_0_3": "x_vars[Supplier1,Customer4]",
    "x_0_4": "x_vars[Supplier1,Customer5]",
    "x_0_5": "x_vars[Supplier1,Customer6]",
    "x_1_0": "x_vars[Supplier2,Customer1]",
    "x_1_1": "x_vars[Supplier2,Customer2]",
    "x_1_2": "x_vars[Supplier2,Customer3]",
    "x_1_3": "x_vars[Supplier2,Customer4]",
    "x_1_4": "x_vars[Supplier2,Customer5]",
    "x_1_5": "x_vars[Supplier2,Customer6]",
    "x_2_0": "x_vars[Supplier3,Customer1]",
    "x_2_1": "x_vars[Supplier3,Customer2]",
    "x_2_2": "x_vars[Supplier3,Customer3]",
    "x_2_3": "x_vars[Supplier3,Customer4]",
    "x_2_4": "x_vars[Supplier3,Customer5]",
    "x_2_5": "x_vars[Supplier3,Customer6]",
    "x_3_0": "x_vars[Supplier4,Customer1]",
    "x_3_1": "x_vars[Supplier4,Customer2]",
    "x_3_2": "x_vars[Supplier4,Customer3]",
    "x_3_3": "x_vars[Supplier4,Customer4]",
    "x_3_4": "x_vars[Supplier4,Customer5]",
    "x_3_5": "x_vars[Supplier4,Customer6]",
    "x_4_0": "x_vars[Supplier5,Customer1]",
    "x_4_1": "x_vars[Supplier5,Customer2]",
    "x_4_2": "x_vars[Supplier5,Customer3]",
    "x_4_3": "x_vars[Supplier5,Customer4]",
    "x_4_4": "x_vars[Supplier5,Customer5]",
    "x_4_5": "x_vars[Supplier5,Customer6]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-009": {
   "source_code_sha256": "d4890b5f51209188242de9e4bc24dc639b45adae05ad1ea41d1d7b0e7ce7d3c4",
   "original_solution_sha256": "5109ec802a33962fb5da569d909759c713ff6a56cd702641611b3ef7840b820b",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S2,C1]"
    ],
    [
     "C5",
     "x_vars[S2,C2]"
    ],
    [
     "C6",
     "x_vars[S2,C3]"
    ],
    [
     "C7",
     "x_vars[S2,C4]"
    ],
    [
     "C8",
     "x_vars[S3,C1]"
    ],
    [
     "C9",
     "x_vars[S3,C2]"
    ],
    [
     "C10",
     "x_vars[S3,C3]"
    ],
    [
     "C11",
     "x_vars[S3,C4]"
    ],
    [
     "C12",
     "x_vars[S4,C1]"
    ],
    [
     "C13",
     "x_vars[S4,C2]"
    ],
    [
     "C14",
     "x_vars[S4,C3]"
    ],
    [
     "C15",
     "x_vars[S4,C4]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-010": {
   "source_code_sha256": "4ce6f0ec8adc60060f39c94e7e2cf27dd59db84da5392e8b01a7ad4a49e3e280",
   "original_solution_sha256": "56cffd5a63e43b70d8a7822bf4e1b5444ea8d5cd3cf8a7e3f257cb408a8780ce",
   "bindings": [
    [
     "C0",
     "x_vars[Adams Group_service]"
    ],
    [
     "C1",
     "x_vars[Anderson-Leach_against]"
    ],
    [
     "C2",
     "x_vars[Anderson-Valdez_somebody]"
    ],
    [
     "C3",
     "x_vars[Anderson-White_son]"
    ],
    [
     "C4",
     "x_vars[Andrews LLC_matter]"
    ],
    [
     "C5",
     "x_vars[Andrews PLC_enter]"
    ],
    [
     "C6",
     "x_vars[Andrews-Martin_build]"
    ],
    [
     "C7",
     "x_vars[Arias-Mendoza_life]"
    ],
    [
     "C8",
     "x_vars[Bennett and Sons_down]"
    ],
    [
     "C9",
     "x_vars[Bennett, Foster and Moreno_enter]"
    ],
    [
     "C10",
     "x_vars[Fernandez, Long and Nelson_member]"
    ],
    [
     "C11",
     "x_vars[Fernandez-Fischer_million]"
    ],
    [
     "C12",
     "x_vars[Ferrell Inc_prove]"
    ],
    [
     "C13",
     "x_vars[Fields, Christensen and Daniels_nor]"
    ],
    [
     "C14",
     "x_vars[Figueroa LLC_involve]"
    ],
    [
     "C15",
     "x_vars[Finley Group_itself]"
    ],
    [
     "C16",
     "x_vars[Fisher Ltd_speak]"
    ],
    [
     "C17",
     "x_vars[Fisher-Marshall_cover]"
    ],
    [
     "C18",
     "x_vars[Fuller-Torres_behavior]"
    ],
    [
     "C19",
     "x_vars[Fuller-Walters_radio]"
    ],
    [
     "C20",
     "x_vars[Gallagher-Campbell_dark]"
    ],
    [
     "C21",
     "x_vars[Gallagher-Kirby_talk]"
    ],
    [
     "C22",
     "x_vars[Gallagher-Parker_recognize]"
    ],
    [
     "C23",
     "x_vars[Gonzalez PLC_task]"
    ],
    [
     "C24",
     "x_vars[Gonzalez, Coleman and Le_heavy]"
    ],
    [
     "C25",
     "x_vars[Gonzalez, Lowe and Robinson_education]"
    ],
    [
     "C26",
     "x_vars[Gonzalez-Horn_light]"
    ],
    [
     "C27",
     "x_vars[Good, Davis and Smith_station]"
    ],
    [
     "C28",
     "x_vars[Goodman, Hughes and White_realize]"
    ],
    [
     "C29",
     "x_vars[Goodwin PLC_bad]"
    ],
    [
     "C30",
     "x_vars[Gordon PLC_detail]"
    ],
    [
     "C31",
     "x_vars[Graham LLC_stage]"
    ],
    [
     "C32",
     "x_vars[Graham Ltd_marriage]"
    ],
    [
     "C33",
     "x_vars[Graham-Swanson_message]"
    ],
    [
     "C34",
     "x_vars[Grant, Cross and Bennett_religious]"
    ],
    [
     "C35",
     "x_vars[Grant, Mcdonald and Watson_owner]"
    ],
    [
     "C36",
     "x_vars[Graves, Turner and Crawford_wait]"
    ],
    [
     "C37",
     "x_vars[Gray, Smith and Barnes_much]"
    ],
    [
     "C38",
     "x_vars[Green Inc_direction]"
    ],
    [
     "C39",
     "x_vars[Green-Rogers_could]"
    ],
    [
     "C40",
     "x_vars[Greene-Baxter_them]"
    ],
    [
     "C41",
     "x_vars[Greer and Sons_keep]"
    ],
    [
     "C42",
     "x_vars[Griffin, Boyle and Dawson_anything]"
    ],
    [
     "C43",
     "x_vars[Harper Inc_well]"
    ],
    [
     "C44",
     "x_vars[Harper Ltd_off]"
    ],
    [
     "C45",
     "x_vars[Harrington, Sosa and Mccarty_coach]"
    ],
    [
     "C46",
     "x_vars[Harris Group_different]"
    ],
    [
     "C47",
     "x_vars[Harris and Sons_audience]"
    ],
    [
     "C48",
     "x_vars[Harris and Sons_fear]"
    ],
    [
     "C49",
     "x_vars[Harris, Hamilton and Rose_contain]"
    ],
    [
     "C50",
     "x_vars[Harris, Stevens and Hall_answer]"
    ],
    [
     "C51",
     "x_vars[Harris-Bell_painting]"
    ],
    [
     "C52",
     "x_vars[Harris-Melton_often]"
    ],
    [
     "C53",
     "x_vars[Harris-Rogers_after]"
    ],
    [
     "C54",
     "x_vars[Hart Group_cold]"
    ],
    [
     "C55",
     "x_vars[Hutchinson, Roberts and Mcbride_among]"
    ],
    [
     "C56",
     "x_vars[Ibarra, Jackson and Potter_upon]"
    ],
    [
     "C57",
     "x_vars[Jackson Inc_anyone]"
    ],
    [
     "C58",
     "x_vars[Jackson LLC_report]"
    ],
    [
     "C59",
     "x_vars[Jackson, Collier and Barber_result]"
    ],
    [
     "C60",
     "x_vars[Jackson, White and Brown_expect]"
    ],
    [
     "C61",
     "x_vars[Jackson-Carroll_six]"
    ],
    [
     "C62",
     "x_vars[Lee-Jones_soldier]"
    ],
    [
     "C63",
     "x_vars[Lewis, Sanchez and Turner_point]"
    ],
    [
     "C64",
     "x_vars[Lin LLC_performance]"
    ],
    [
     "C65",
     "x_vars[Lindsey, Avila and Brown_candidate]"
    ],
    [
     "C66",
     "x_vars[Lindsey-Strickland_modern]"
    ],
    [
     "C67",
     "x_vars[Little-Perez_choice]"
    ],
    [
     "C68",
     "x_vars[Lloyd, Stone and Mcguire_national]"
    ],
    [
     "C69",
     "x_vars[Long, Hughes and Gallegos_receive]"
    ],
    [
     "C70",
     "x_vars[Lopez PLC_analysis]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Adams Group_service]",
    "x_2": "x_vars[Anderson-Leach_against]",
    "x_3": "x_vars[Anderson-Valdez_somebody]",
    "x_4": "x_vars[Anderson-White_son]",
    "x_5": "x_vars[Andrews LLC_matter]",
    "x_6": "x_vars[Andrews PLC_enter]",
    "x_7": "x_vars[Andrews-Martin_build]",
    "x_8": "x_vars[Arias-Mendoza_life]",
    "x_9": "x_vars[Bennett and Sons_down]",
    "x_10": "x_vars[Bennett, Foster and Moreno_enter]",
    "x_11": "x_vars[Fernandez, Long and Nelson_member]",
    "x_12": "x_vars[Fernandez-Fischer_million]",
    "x_13": "x_vars[Ferrell Inc_prove]",
    "x_14": "x_vars[Fields, Christensen and Daniels_nor]",
    "x_15": "x_vars[Figueroa LLC_involve]",
    "x_16": "x_vars[Finley Group_itself]",
    "x_17": "x_vars[Fisher Ltd_speak]",
    "x_18": "x_vars[Fisher-Marshall_cover]",
    "x_19": "x_vars[Fuller-Torres_behavior]",
    "x_20": "x_vars[Fuller-Walters_radio]",
    "x_21": "x_vars[Gallagher-Campbell_dark]",
    "x_22": "x_vars[Gallagher-Kirby_talk]",
    "x_23": "x_vars[Gallagher-Parker_recognize]",
    "x_24": "x_vars[Gonzalez PLC_task]",
    "x_25": "x_vars[Gonzalez, Coleman and Le_heavy]",
    "x_26": "x_vars[Gonzalez, Lowe and Robinson_education]",
    "x_27": "x_vars[Gonzalez-Horn_light]",
    "x_28": "x_vars[Good, Davis and Smith_station]",
    "x_29": "x_vars[Goodman, Hughes and White_realize]",
    "x_30": "x_vars[Goodwin PLC_bad]",
    "x_31": "x_vars[Gordon PLC_detail]",
    "x_32": "x_vars[Graham LLC_stage]",
    "x_33": "x_vars[Graham Ltd_marriage]",
    "x_34": "x_vars[Graham-Swanson_message]",
    "x_35": "x_vars[Grant, Cross and Bennett_religious]",
    "x_36": "x_vars[Grant, Mcdonald and Watson_owner]",
    "x_37": "x_vars[Graves, Turner and Crawford_wait]",
    "x_38": "x_vars[Gray, Smith and Barnes_much]",
    "x_39": "x_vars[Green Inc_direction]",
    "x_40": "x_vars[Green-Rogers_could]",
    "x_41": "x_vars[Greene-Baxter_them]",
    "x_42": "x_vars[Greer and Sons_keep]",
    "x_43": "x_vars[Griffin, Boyle and Dawson_anything]",
    "x_44": "x_vars[Harper Inc_well]",
    "x_45": "x_vars[Harper Ltd_off]",
    "x_46": "x_vars[Harrington, Sosa and Mccarty_coach]",
    "x_47": "x_vars[Harris Group_different]",
    "x_48": "x_vars[Harris and Sons_audience]",
    "x_49": "x_vars[Harris and Sons_fear]",
    "x_50": "x_vars[Harris, Hamilton and Rose_contain]",
    "x_51": "x_vars[Harris, Stevens and Hall_answer]",
    "x_52": "x_vars[Harris-Bell_painting]",
    "x_53": "x_vars[Harris-Melton_often]",
    "x_54": "x_vars[Harris-Rogers_after]",
    "x_55": "x_vars[Hart Group_cold]",
    "x_56": "x_vars[Hutchinson, Roberts and Mcbride_among]",
    "x_57": "x_vars[Ibarra, Jackson and Potter_upon]",
    "x_58": "x_vars[Jackson Inc_anyone]",
    "x_59": "x_vars[Jackson LLC_report]",
    "x_60": "x_vars[Jackson, Collier and Barber_result]",
    "x_61": "x_vars[Jackson, White and Brown_expect]",
    "x_62": "x_vars[Jackson-Carroll_six]",
    "x_63": "x_vars[Lee-Jones_soldier]",
    "x_64": "x_vars[Lewis, Sanchez and Turner_point]",
    "x_65": "x_vars[Lin LLC_performance]",
    "x_66": "x_vars[Lindsey, Avila and Brown_candidate]",
    "x_67": "x_vars[Lindsey-Strickland_modern]",
    "x_68": "x_vars[Little-Perez_choice]",
    "x_69": "x_vars[Lloyd, Stone and Mcguire_national]",
    "x_70": "x_vars[Long, Hughes and Gallegos_receive]",
    "x_71": "x_vars[Lopez PLC_analysis]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-011": {
   "source_code_sha256": "4719be4724cf6a92d9faa86442069fdccf15db0ce36be90900f470b23ef3dca0",
   "original_solution_sha256": "68203b0c38371b684628431f6920ddc720a25bd2b165f6fb4fdc65d32eebc533",
   "bindings": [
    [
     "C0",
     "x_vars[id999]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[id999]"
   },
   "mapping_basis": "frozen_automatic_name_axis_projection"
  },
  "OR-012": {
   "source_code_sha256": "4ae8594a9a62d3a4b3621e9ad8a1556ffd38176270e3b6d5db7ca082db8df8e9",
   "original_solution_sha256": "ffe95067bb7d348f9095f922b44dc1c6aa9280ab237c8d02ff9ab509a52b858e",
   "bindings": [
    [
     "C0",
     "x_vars[1984 by George Orwell]"
    ],
    [
     "C1",
     "x_vars[Adidas 3-Stripes Shorts]"
    ],
    [
     "C2",
     "x_vars[Adidas Essential Track Pants]"
    ],
    [
     "C3",
     "x_vars[Adidas FIFA World Cup Football]"
    ],
    [
     "C4",
     "x_vars[Adidas Originals Superstar Sneakers]"
    ],
    [
     "C5",
     "x_vars[Adidas Originals Trefoil Hoodie]"
    ],
    [
     "C6",
     "x_vars[Adidas Ultraboost Running Shoes]"
    ],
    [
     "C7",
     "x_vars[Adidas Ultraboost Shoes]"
    ],
    [
     "C8",
     "x_vars[Amazon Echo Dot (4th Gen)]"
    ],
    [
     "C9",
     "x_vars[Amazon Echo Show 10]"
    ],
    [
     "C10",
     "x_vars[Amazon Fire TV Stick 4K]"
    ],
    [
     "C11",
     "x_vars[Anastasia Beverly Hills Brow Wiz]"
    ],
    [
     "C12",
     "x_vars[Anker PowerCore Portable Charger]"
    ],
    [
     "C13",
     "x_vars[Anova Precision Cooker]"
    ],
    [
     "C14",
     "x_vars[Anova Precision Oven]"
    ],
    [
     "C15",
     "x_vars[Apple AirPods Max]"
    ],
    [
     "C16",
     "x_vars[Apple AirPods Pro]"
    ],
    [
     "C17",
     "x_vars[Apple MacBook Air]"
    ],
    [
     "C18",
     "x_vars[Apple MacBook Pro 16-inch]"
    ],
    [
     "C19",
     "x_vars[Apple TV 4K]"
    ],
    [
     "C20",
     "x_vars[Apple Watch Series 8]"
    ],
    [
     "C21",
     "x_vars[Apple iPad Air]"
    ],
    [
     "C22",
     "x_vars[Atomic Habits by James Clear]"
    ],
    [
     "C23",
     "x_vars[Babolat Pure Drive Tennis Racket]"
    ],
    [
     "C24",
     "x_vars[Becoming by Michelle Obama]"
    ],
    [
     "C25",
     "x_vars[Biore UV Aqua Rich Watery Essence Sunscreen]"
    ],
    [
     "C26",
     "x_vars[Blueair Classic 480i]"
    ],
    [
     "C27",
     "x_vars[Bose QuietComfort 35 Headphones]"
    ],
    [
     "C28",
     "x_vars[Bose QuietComfort 35 II Wireless Headphones]"
    ],
    [
     "C29",
     "x_vars[Bose SoundLink Color Bluetooth Speaker II]"
    ],
    [
     "C30",
     "x_vars[Bose SoundLink Revolve+ Speaker]"
    ],
    [
     "C31",
     "x_vars[Bose SoundSport Wireless Earbuds]"
    ],
    [
     "C32",
     "x_vars[Bowflex SelectTech 1090 Adjustable Dumbbells]"
    ],
    [
     "C33",
     "x_vars[Bowflex SelectTech 552 Dumbbells]"
    ],
    [
     "C34",
     "x_vars[Breville Nespresso Creatista Plus]"
    ],
    [
     "C35",
     "x_vars[Breville Smart Coffee Grinder Pro]"
    ],
    [
     "C36",
     "x_vars[Breville Smart Grill]"
    ],
    [
     "C37",
     "x_vars[Breville Smart Oven]"
    ],
    [
     "C38",
     "x_vars[Calvin Klein Boxer Briefs]"
    ],
    [
     "C39",
     "x_vars[Canon EOS R5 Camera]"
    ],
    [
     "C40",
     "x_vars[Canon EOS Rebel T7i DSLR Camera]"
    ],
    [
     "C41",
     "x_vars[Caudalie Vinoperfect Radiance Serum]"
    ],
    [
     "C42",
     "x_vars[CeraVe Hydrating Facial Cleanser]"
    ],
    [
     "C43",
     "x_vars[Champion Reverse Weave Hoodie]"
    ],
    [
     "C44",
     "x_vars[Chanel No. 5 Perfume]"
    ],
    [
     "C45",
     "x_vars[Charlotte Tilbury Magic Cream]"
    ],
    [
     "C46",
     "x_vars[Clinique Dramatically Different Moisturizing Lotion]"
    ],
    [
     "C47",
     "x_vars[Clinique Moisture Surge]"
    ],
    [
     "C48",
     "x_vars[Columbia Fleece Jacket]"
    ],
    [
     "C49",
     "x_vars[Crock-Pot 6-Quart Slow Cooker]"
    ],
    [
     "C50",
     "x_vars[Cuisinart Coffee Center]"
    ],
    [
     "C51",
     "x_vars[Cuisinart Custom 14-Cup Food Processor]"
    ],
    [
     "C52",
     "x_vars[Cuisinart Griddler Deluxe]"
    ],
    [
     "C53",
     "x_vars[De'Longhi Magnifica Espresso Machine]"
    ],
    [
     "C54",
     "x_vars[Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment]"
    ],
    [
     "C55",
     "x_vars[Drunk Elephant C-Firma Day Serum]"
    ],
    [
     "C56",
     "x_vars[Dune by Frank Herbert]"
    ],
    [
     "C57",
     "x_vars[Dyson Pure Cool Link]"
    ],
    [
     "C58",
     "x_vars[Dyson Supersonic Hair Dryer]"
    ],
    [
     "C59",
     "x_vars[Dyson V11 Vacuum]"
    ],
    [
     "C60",
     "x_vars[Dyson V8 Absolute]"
    ],
    [
     "C61",
     "x_vars[Educated by Tara Westover]"
    ],
    [
     "C62",
     "x_vars[Estee Lauder Advanced Night Repair]"
    ],
    [
     "C63",
     "x_vars[Eufy RoboVac 11S]"
    ],
    [
     "C64",
     "x_vars[Fenty Beauty Killawatt Highlighter]"
    ],
    [
     "C65",
     "x_vars[First Aid Beauty Ultra Repair Cream]"
    ],
    [
     "C66",
     "x_vars[Fitbit Charge 5]"
    ],
    [
     "C67",
     "x_vars[Fitbit Inspire 2]"
    ],
    [
     "C68",
     "x_vars[Fitbit Luxe]"
    ],
    [
     "C69",
     "x_vars[Fitbit Versa 3]"
    ],
    [
     "C70",
     "x_vars[Forever 21 Graphic Tee]"
    ],
    [
     "C71",
     "x_vars[Fresh Sugar Lip Treatment]"
    ],
    [
     "C72",
     "x_vars[Gap 1969 Original Fit Jeans]"
    ],
    [
     "C73",
     "x_vars[Gap Crewneck Sweatshirt]"
    ],
    [
     "C74",
     "x_vars[Gap Essential Crewneck T-Shirt]"
    ],
    [
     "C75",
     "x_vars[Gap High Rise Skinny Jeans]"
    ],
    [
     "C76",
     "x_vars[Garmin Edge 530]"
    ],
    [
     "C77",
     "x_vars[Garmin Fenix 6X Pro]"
    ],
    [
     "C78",
     "x_vars[Garmin Forerunner 245]"
    ],
    [
     "C79",
     "x_vars[Garmin Forerunner 945]"
    ],
    [
     "C80",
     "x_vars[GlamGlow Supermud Clearing Treatment]"
    ],
    [
     "C81",
     "x_vars[Glossier Boy Brow]"
    ],
    [
     "C82",
     "x_vars[Glossier Cloud Paint]"
    ],
    [
     "C83",
     "x_vars[GoPro HERO10 Black]"
    ],
    [
     "C84",
     "x_vars[GoPro HERO9 Black]"
    ],
    [
     "C85",
     "x_vars[Gone Girl by Gillian Flynn]"
    ],
    [
     "C86",
     "x_vars[Google Nest Hub Max]"
    ],
    [
     "C87",
     "x_vars[Google Nest Wifi Router]"
    ],
    [
     "C88",
     "x_vars[Google Pixel 6 Pro]"
    ],
    [
     "C89",
     "x_vars[Google Pixelbook Go]"
    ],
    [
     "C90",
     "x_vars[H&M Slim Fit Jeans]"
    ],
    [
     "C91",
     "x_vars[HP Spectre x360 Laptop]"
    ],
    [
     "C92",
     "x_vars[Hamilton Beach FlexBrew Coffee Maker]"
    ],
    [
     "C93",
     "x_vars[Hanes ComfortSoft T-Shirt]"
    ],
    [
     "C94",
     "x_vars[Harry Potter and the Sorcerer's Stone]"
    ],
    [
     "C95",
     "x_vars[Hydro Flask Standard Mouth Water Bottle]"
    ],
    [
     "C96",
     "x_vars[Hydro Flask Wide Mouth Water Bottle]"
    ],
    [
     "C97",
     "x_vars[Hyperice Hypervolt Massager]"
    ],
    [
     "C98",
     "x_vars[Instant Pot Duo]"
    ],
    [
     "C99",
     "x_vars[Instant Pot Duo Crisp]"
    ],
    [
     "C100",
     "x_vars[Instant Pot Duo Evo Plus]"
    ],
    [
     "C101",
     "x_vars[Instant Pot Duo Nova]"
    ],
    [
     "C102",
     "x_vars[Instant Pot Ultra]"
    ],
    [
     "C103",
     "x_vars[Keurig K-Elite Coffee Maker]"
    ],
    [
     "C104",
     "x_vars[Keurig K-Mini Coffee Maker]"
    ],
    [
     "C105",
     "x_vars[Kiehl's Midnight Recovery Concentrate]"
    ],
    [
     "C106",
     "x_vars[Kindle Paperwhite]"
    ],
    [
     "C107",
     "x_vars[KitchenAid Artisan Stand Mixer]"
    ],
    [
     "C108",
     "x_vars[KitchenAid Stand Mixer]"
    ],
    [
     "C109",
     "x_vars[L'Occitane Shea Butter Hand Cream]"
    ],
    [
     "C110",
     "x_vars[L'Oreal Revitalift Serum]"
    ],
    [
     "C111",
     "x_vars[LG OLED TV]"
    ],
    [
     "C112",
     "x_vars[La Mer Cr_¨me de la Mer Moisturizer]"
    ],
    [
     "C113",
     "x_vars[Lancome La Vie Est Belle]"
    ],
    [
     "C114",
     "x_vars[Laneige Water Sleeping Mask]"
    ],
    [
     "C115",
     "x_vars[Levi's 501 Jeans]"
    ],
    [
     "C116",
     "x_vars[Levi's 511 Slim Fit Jeans]"
    ],
    [
     "C117",
     "x_vars[Levi's Sherpa Trucker Jacket]"
    ],
    [
     "C118",
     "x_vars[Levi's Trucker Jacket]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[1984 by George Orwell]",
    "x_2": "x_vars[Adidas 3-Stripes Shorts]",
    "x_3": "x_vars[Adidas Essential Track Pants]",
    "x_4": "x_vars[Adidas FIFA World Cup Football]",
    "x_5": "x_vars[Adidas Originals Superstar Sneakers]",
    "x_6": "x_vars[Adidas Originals Trefoil Hoodie]",
    "x_7": "x_vars[Adidas Ultraboost Running Shoes]",
    "x_8": "x_vars[Adidas Ultraboost Shoes]",
    "x_9": "x_vars[Amazon Echo Dot (4th Gen)]",
    "x_10": "x_vars[Amazon Echo Show 10]",
    "x_11": "x_vars[Amazon Fire TV Stick 4K]",
    "x_12": "x_vars[Anastasia Beverly Hills Brow Wiz]",
    "x_13": "x_vars[Anker PowerCore Portable Charger]",
    "x_14": "x_vars[Anova Precision Cooker]",
    "x_15": "x_vars[Anova Precision Oven]",
    "x_16": "x_vars[Apple AirPods Max]",
    "x_17": "x_vars[Apple AirPods Pro]",
    "x_18": "x_vars[Apple MacBook Air]",
    "x_19": "x_vars[Apple MacBook Pro 16-inch]",
    "x_20": "x_vars[Apple TV 4K]",
    "x_21": "x_vars[Apple Watch Series 8]",
    "x_22": "x_vars[Apple iPad Air]",
    "x_23": "x_vars[Atomic Habits by James Clear]",
    "x_24": "x_vars[Babolat Pure Drive Tennis Racket]",
    "x_25": "x_vars[Becoming by Michelle Obama]",
    "x_26": "x_vars[Biore UV Aqua Rich Watery Essence Sunscreen]",
    "x_27": "x_vars[Blueair Classic 480i]",
    "x_28": "x_vars[Bose QuietComfort 35 Headphones]",
    "x_29": "x_vars[Bose QuietComfort 35 II Wireless Headphones]",
    "x_30": "x_vars[Bose SoundLink Color Bluetooth Speaker II]",
    "x_31": "x_vars[Bose SoundLink Revolve+ Speaker]",
    "x_32": "x_vars[Bose SoundSport Wireless Earbuds]",
    "x_33": "x_vars[Bowflex SelectTech 1090 Adjustable Dumbbells]",
    "x_34": "x_vars[Bowflex SelectTech 552 Dumbbells]",
    "x_35": "x_vars[Breville Nespresso Creatista Plus]",
    "x_36": "x_vars[Breville Smart Coffee Grinder Pro]",
    "x_37": "x_vars[Breville Smart Grill]",
    "x_38": "x_vars[Breville Smart Oven]",
    "x_39": "x_vars[Calvin Klein Boxer Briefs]",
    "x_40": "x_vars[Canon EOS R5 Camera]",
    "x_41": "x_vars[Canon EOS Rebel T7i DSLR Camera]",
    "x_42": "x_vars[Caudalie Vinoperfect Radiance Serum]",
    "x_43": "x_vars[CeraVe Hydrating Facial Cleanser]",
    "x_44": "x_vars[Champion Reverse Weave Hoodie]",
    "x_45": "x_vars[Chanel No. 5 Perfume]",
    "x_46": "x_vars[Charlotte Tilbury Magic Cream]",
    "x_47": "x_vars[Clinique Dramatically Different Moisturizing Lotion]",
    "x_48": "x_vars[Clinique Moisture Surge]",
    "x_49": "x_vars[Columbia Fleece Jacket]",
    "x_50": "x_vars[Crock-Pot 6-Quart Slow Cooker]",
    "x_51": "x_vars[Cuisinart Coffee Center]",
    "x_52": "x_vars[Cuisinart Custom 14-Cup Food Processor]",
    "x_53": "x_vars[Cuisinart Griddler Deluxe]",
    "x_54": "x_vars[De'Longhi Magnifica Espresso Machine]",
    "x_55": "x_vars[Dr. Jart+ Cicapair Tiger Grass Color Correcting Treatment]",
    "x_56": "x_vars[Drunk Elephant C-Firma Day Serum]",
    "x_57": "x_vars[Dune by Frank Herbert]",
    "x_58": "x_vars[Dyson Pure Cool Link]",
    "x_59": "x_vars[Dyson Supersonic Hair Dryer]",
    "x_60": "x_vars[Dyson V11 Vacuum]",
    "x_61": "x_vars[Dyson V8 Absolute]",
    "x_62": "x_vars[Educated by Tara Westover]",
    "x_63": "x_vars[Estee Lauder Advanced Night Repair]",
    "x_64": "x_vars[Eufy RoboVac 11S]",
    "x_65": "x_vars[Fenty Beauty Killawatt Highlighter]",
    "x_66": "x_vars[First Aid Beauty Ultra Repair Cream]",
    "x_67": "x_vars[Fitbit Charge 5]",
    "x_68": "x_vars[Fitbit Inspire 2]",
    "x_69": "x_vars[Fitbit Luxe]",
    "x_70": "x_vars[Fitbit Versa 3]",
    "x_71": "x_vars[Forever 21 Graphic Tee]",
    "x_72": "x_vars[Fresh Sugar Lip Treatment]",
    "x_73": "x_vars[Gap 1969 Original Fit Jeans]",
    "x_74": "x_vars[Gap Crewneck Sweatshirt]",
    "x_75": "x_vars[Gap Essential Crewneck T-Shirt]",
    "x_76": "x_vars[Gap High Rise Skinny Jeans]",
    "x_77": "x_vars[Garmin Edge 530]",
    "x_78": "x_vars[Garmin Fenix 6X Pro]",
    "x_79": "x_vars[Garmin Forerunner 245]",
    "x_80": "x_vars[Garmin Forerunner 945]",
    "x_81": "x_vars[GlamGlow Supermud Clearing Treatment]",
    "x_82": "x_vars[Glossier Boy Brow]",
    "x_83": "x_vars[Glossier Cloud Paint]",
    "x_84": "x_vars[GoPro HERO10 Black]",
    "x_85": "x_vars[GoPro HERO9 Black]",
    "x_86": "x_vars[Gone Girl by Gillian Flynn]",
    "x_87": "x_vars[Google Nest Hub Max]",
    "x_88": "x_vars[Google Nest Wifi Router]",
    "x_89": "x_vars[Google Pixel 6 Pro]",
    "x_90": "x_vars[Google Pixelbook Go]",
    "x_91": "x_vars[H&M Slim Fit Jeans]",
    "x_92": "x_vars[HP Spectre x360 Laptop]",
    "x_93": "x_vars[Hamilton Beach FlexBrew Coffee Maker]",
    "x_94": "x_vars[Hanes ComfortSoft T-Shirt]",
    "x_95": "x_vars[Harry Potter and the Sorcerer's Stone]",
    "x_96": "x_vars[Hydro Flask Standard Mouth Water Bottle]",
    "x_97": "x_vars[Hydro Flask Wide Mouth Water Bottle]",
    "x_98": "x_vars[Hyperice Hypervolt Massager]",
    "x_99": "x_vars[Instant Pot Duo]",
    "x_100": "x_vars[Instant Pot Duo Crisp]",
    "x_101": "x_vars[Instant Pot Duo Evo Plus]",
    "x_102": "x_vars[Instant Pot Duo Nova]",
    "x_103": "x_vars[Instant Pot Ultra]",
    "x_104": "x_vars[Keurig K-Elite Coffee Maker]",
    "x_105": "x_vars[Keurig K-Mini Coffee Maker]",
    "x_106": "x_vars[Kiehl's Midnight Recovery Concentrate]",
    "x_107": "x_vars[Kindle Paperwhite]",
    "x_108": "x_vars[KitchenAid Artisan Stand Mixer]",
    "x_109": "x_vars[KitchenAid Stand Mixer]",
    "x_110": "x_vars[L'Occitane Shea Butter Hand Cream]",
    "x_111": "x_vars[L'Oreal Revitalift Serum]",
    "x_112": "x_vars[LG OLED TV]",
    "x_113": "x_vars[La Mer Cr_¨me de la Mer Moisturizer]",
    "x_114": "x_vars[Lancome La Vie Est Belle]",
    "x_115": "x_vars[Laneige Water Sleeping Mask]",
    "x_116": "x_vars[Levi's 501 Jeans]",
    "x_117": "x_vars[Levi's 511 Slim Fit Jeans]",
    "x_118": "x_vars[Levi's Sherpa Trucker Jacket]",
    "x_119": "x_vars[Levi's Trucker Jacket]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-013": {
   "source_code_sha256": "0b156960c2e1be592e319e91ca83deb033065e2d6521e1d39d9c50f028b7847b",
   "original_solution_sha256": "78ac7981f5eb5af57c28b181727bffaef0f9fcb3bc848f14a6137c55d5293446",
   "bindings": [
    [
     "C0",
     "x_vars[4U_Service_22]"
    ],
    [
     "C1",
     "x_vars[4U_Service_36]"
    ],
    [
     "C2",
     "x_vars[4U_Service_7]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[4U_Service_22]",
    "x_2": "x_vars[4U_Service_36]",
    "x_3": "x_vars[4U_Service_7]"
   },
   "mapping_basis": "frozen_automatic_name_axis_projection"
  },
  "OR-014": {
   "source_code_sha256": "ae6dbbcf71c8284bdd7e3122b7338458faa0805b08b47149e1181101d3d2997c",
   "original_solution_sha256": "62176e138bbf4a4e705142dd1b3963137c511d7cb0c41dfb54ed2dce188be1cb",
   "bindings": [
    [
     "C0",
     "x_vars[bbq_ckn_l]"
    ],
    [
     "C1",
     "x_vars[bbq_ckn_m]"
    ],
    [
     "C2",
     "x_vars[bbq_ckn_s]"
    ],
    [
     "C3",
     "x_vars[big_meat_s]"
    ],
    [
     "C4",
     "x_vars[brie_carre_s]"
    ],
    [
     "C5",
     "x_vars[calabrese_l]"
    ],
    [
     "C6",
     "x_vars[calabrese_m]"
    ],
    [
     "C7",
     "x_vars[calabrese_s]"
    ],
    [
     "C8",
     "x_vars[cali_ckn_l]"
    ],
    [
     "C9",
     "x_vars[cali_ckn_m]"
    ],
    [
     "C10",
     "x_vars[cali_ckn_s]"
    ],
    [
     "C11",
     "x_vars[ckn_alfredo_l]"
    ],
    [
     "C12",
     "x_vars[ckn_alfredo_m]"
    ],
    [
     "C13",
     "x_vars[ckn_alfredo_s]"
    ],
    [
     "C14",
     "x_vars[ckn_pesto_l]"
    ],
    [
     "C15",
     "x_vars[ckn_pesto_m]"
    ],
    [
     "C16",
     "x_vars[ckn_pesto_s]"
    ],
    [
     "C17",
     "x_vars[classic_dlx_l]"
    ],
    [
     "C18",
     "x_vars[classic_dlx_m]"
    ],
    [
     "C19",
     "x_vars[classic_dlx_s]"
    ],
    [
     "C20",
     "x_vars[five_cheese_l]"
    ],
    [
     "C21",
     "x_vars[four_cheese_l]"
    ],
    [
     "C22",
     "x_vars[four_cheese_m]"
    ],
    [
     "C23",
     "x_vars[green_garden_l]"
    ],
    [
     "C24",
     "x_vars[green_garden_m]"
    ],
    [
     "C25",
     "x_vars[green_garden_s]"
    ],
    [
     "C26",
     "x_vars[hawaiian_l]"
    ],
    [
     "C27",
     "x_vars[hawaiian_m]"
    ],
    [
     "C28",
     "x_vars[hawaiian_s]"
    ],
    [
     "C29",
     "x_vars[ital_cpcllo_l]"
    ],
    [
     "C30",
     "x_vars[ital_cpcllo_m]"
    ],
    [
     "C31",
     "x_vars[ital_cpcllo_s]"
    ],
    [
     "C32",
     "x_vars[ital_supr_l]"
    ],
    [
     "C33",
     "x_vars[ital_supr_m]"
    ],
    [
     "C34",
     "x_vars[ital_supr_s]"
    ],
    [
     "C35",
     "x_vars[ital_veggie_l]"
    ],
    [
     "C36",
     "x_vars[ital_veggie_m]"
    ],
    [
     "C37",
     "x_vars[ital_veggie_s]"
    ],
    [
     "C38",
     "x_vars[mediterraneo_l]"
    ],
    [
     "C39",
     "x_vars[mediterraneo_m]"
    ],
    [
     "C40",
     "x_vars[mediterraneo_s]"
    ],
    [
     "C41",
     "x_vars[mexicana_l]"
    ],
    [
     "C42",
     "x_vars[mexicana_m]"
    ],
    [
     "C43",
     "x_vars[mexicana_s]"
    ],
    [
     "C44",
     "x_vars[napolitana_l]"
    ],
    [
     "C45",
     "x_vars[napolitana_m]"
    ],
    [
     "C46",
     "x_vars[napolitana_s]"
    ],
    [
     "C47",
     "x_vars[pep_msh_pep_l]"
    ],
    [
     "C48",
     "x_vars[pep_msh_pep_m]"
    ],
    [
     "C49",
     "x_vars[pep_msh_pep_s]"
    ],
    [
     "C50",
     "x_vars[pepperoni_l]"
    ],
    [
     "C51",
     "x_vars[pepperoni_m]"
    ],
    [
     "C52",
     "x_vars[pepperoni_s]"
    ],
    [
     "C53",
     "x_vars[peppr_salami_l]"
    ],
    [
     "C54",
     "x_vars[peppr_salami_m]"
    ],
    [
     "C55",
     "x_vars[peppr_salami_s]"
    ],
    [
     "C56",
     "x_vars[prsc_argla_l]"
    ],
    [
     "C57",
     "x_vars[prsc_argla_m]"
    ],
    [
     "C58",
     "x_vars[prsc_argla_s]"
    ],
    [
     "C59",
     "x_vars[sicilian_l]"
    ],
    [
     "C60",
     "x_vars[sicilian_m]"
    ],
    [
     "C61",
     "x_vars[sicilian_s]"
    ],
    [
     "C62",
     "x_vars[soppressata_l]"
    ],
    [
     "C63",
     "x_vars[soppressata_m]"
    ],
    [
     "C64",
     "x_vars[soppressata_s]"
    ],
    [
     "C65",
     "x_vars[southw_ckn_l]"
    ],
    [
     "C66",
     "x_vars[southw_ckn_m]"
    ],
    [
     "C67",
     "x_vars[southw_ckn_s]"
    ],
    [
     "C68",
     "x_vars[spicy_ital_l]"
    ],
    [
     "C69",
     "x_vars[spicy_ital_m]"
    ],
    [
     "C70",
     "x_vars[spicy_ital_s]"
    ],
    [
     "C71",
     "x_vars[spin_pesto_l]"
    ],
    [
     "C72",
     "x_vars[spin_pesto_m]"
    ],
    [
     "C73",
     "x_vars[spin_pesto_s]"
    ],
    [
     "C74",
     "x_vars[spinach_fet_l]"
    ],
    [
     "C75",
     "x_vars[spinach_fet_m]"
    ],
    [
     "C76",
     "x_vars[spinach_fet_s]"
    ],
    [
     "C77",
     "x_vars[spinach_supr_l]"
    ],
    [
     "C78",
     "x_vars[spinach_supr_m]"
    ],
    [
     "C79",
     "x_vars[spinach_supr_s]"
    ],
    [
     "C80",
     "x_vars[thai_ckn_l]"
    ],
    [
     "C81",
     "x_vars[thai_ckn_m]"
    ],
    [
     "C82",
     "x_vars[thai_ckn_s]"
    ],
    [
     "C83",
     "x_vars[the_greek_l]"
    ],
    [
     "C84",
     "x_vars[the_greek_m]"
    ],
    [
     "C85",
     "x_vars[the_greek_s]"
    ],
    [
     "C86",
     "x_vars[the_greek_xl]"
    ],
    [
     "C87",
     "x_vars[the_greek_xxl]"
    ],
    [
     "C88",
     "x_vars[veggie_veg_l]"
    ],
    [
     "C89",
     "x_vars[veggie_veg_m]"
    ],
    [
     "C90",
     "x_vars[veggie_veg_s]"
    ],
    [
     "C91",
     "x_vars[fried_duck_l]"
    ],
    [
     "C92",
     "x_vars[fried_duck_m]"
    ],
    [
     "C93",
     "x_vars[fried_duck_s]"
    ],
    [
     "C94",
     "x_vars[mapo_Tofu_l]"
    ],
    [
     "C95",
     "x_vars[mapo_Tofu_m]"
    ],
    [
     "C96",
     "x_vars[mapo_Tofu_s]"
    ],
    [
     "C97",
     "x_vars[sea_food_l]"
    ],
    [
     "C98",
     "x_vars[sea_food_m]"
    ],
    [
     "C99",
     "x_vars[sea_food_s]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[bbq_ckn_l]",
    "x_2": "x_vars[bbq_ckn_m]",
    "x_3": "x_vars[bbq_ckn_s]",
    "x_4": "x_vars[big_meat_s]",
    "x_5": "x_vars[brie_carre_s]",
    "x_6": "x_vars[calabrese_l]",
    "x_7": "x_vars[calabrese_m]",
    "x_8": "x_vars[calabrese_s]",
    "x_9": "x_vars[cali_ckn_l]",
    "x_10": "x_vars[cali_ckn_m]",
    "x_11": "x_vars[cali_ckn_s]",
    "x_12": "x_vars[ckn_alfredo_l]",
    "x_13": "x_vars[ckn_alfredo_m]",
    "x_14": "x_vars[ckn_alfredo_s]",
    "x_15": "x_vars[ckn_pesto_l]",
    "x_16": "x_vars[ckn_pesto_m]",
    "x_17": "x_vars[ckn_pesto_s]",
    "x_18": "x_vars[classic_dlx_l]",
    "x_19": "x_vars[classic_dlx_m]",
    "x_20": "x_vars[classic_dlx_s]",
    "x_21": "x_vars[five_cheese_l]",
    "x_22": "x_vars[four_cheese_l]",
    "x_23": "x_vars[four_cheese_m]",
    "x_24": "x_vars[green_garden_l]",
    "x_25": "x_vars[green_garden_m]",
    "x_26": "x_vars[green_garden_s]",
    "x_27": "x_vars[hawaiian_l]",
    "x_28": "x_vars[hawaiian_m]",
    "x_29": "x_vars[hawaiian_s]",
    "x_30": "x_vars[ital_cpcllo_l]",
    "x_31": "x_vars[ital_cpcllo_m]",
    "x_32": "x_vars[ital_cpcllo_s]",
    "x_33": "x_vars[ital_supr_l]",
    "x_34": "x_vars[ital_supr_m]",
    "x_35": "x_vars[ital_supr_s]",
    "x_36": "x_vars[ital_veggie_l]",
    "x_37": "x_vars[ital_veggie_m]",
    "x_38": "x_vars[ital_veggie_s]",
    "x_39": "x_vars[mediterraneo_l]",
    "x_40": "x_vars[mediterraneo_m]",
    "x_41": "x_vars[mediterraneo_s]",
    "x_42": "x_vars[mexicana_l]",
    "x_43": "x_vars[mexicana_m]",
    "x_44": "x_vars[mexicana_s]",
    "x_45": "x_vars[napolitana_l]",
    "x_46": "x_vars[napolitana_m]",
    "x_47": "x_vars[napolitana_s]",
    "x_48": "x_vars[pep_msh_pep_l]",
    "x_49": "x_vars[pep_msh_pep_m]",
    "x_50": "x_vars[pep_msh_pep_s]",
    "x_51": "x_vars[pepperoni_l]",
    "x_52": "x_vars[pepperoni_m]",
    "x_53": "x_vars[pepperoni_s]",
    "x_54": "x_vars[peppr_salami_l]",
    "x_55": "x_vars[peppr_salami_m]",
    "x_56": "x_vars[peppr_salami_s]",
    "x_57": "x_vars[prsc_argla_l]",
    "x_58": "x_vars[prsc_argla_m]",
    "x_59": "x_vars[prsc_argla_s]",
    "x_60": "x_vars[sicilian_l]",
    "x_61": "x_vars[sicilian_m]",
    "x_62": "x_vars[sicilian_s]",
    "x_63": "x_vars[soppressata_l]",
    "x_64": "x_vars[soppressata_m]",
    "x_65": "x_vars[soppressata_s]",
    "x_66": "x_vars[southw_ckn_l]",
    "x_67": "x_vars[southw_ckn_m]",
    "x_68": "x_vars[southw_ckn_s]",
    "x_69": "x_vars[spicy_ital_l]",
    "x_70": "x_vars[spicy_ital_m]",
    "x_71": "x_vars[spicy_ital_s]",
    "x_72": "x_vars[spin_pesto_l]",
    "x_73": "x_vars[spin_pesto_m]",
    "x_74": "x_vars[spin_pesto_s]",
    "x_75": "x_vars[spinach_fet_l]",
    "x_76": "x_vars[spinach_fet_m]",
    "x_77": "x_vars[spinach_fet_s]",
    "x_78": "x_vars[spinach_supr_l]",
    "x_79": "x_vars[spinach_supr_m]",
    "x_80": "x_vars[spinach_supr_s]",
    "x_81": "x_vars[thai_ckn_l]",
    "x_82": "x_vars[thai_ckn_m]",
    "x_83": "x_vars[thai_ckn_s]",
    "x_84": "x_vars[the_greek_l]",
    "x_85": "x_vars[the_greek_m]",
    "x_86": "x_vars[the_greek_s]",
    "x_87": "x_vars[the_greek_xl]",
    "x_88": "x_vars[the_greek_xxl]",
    "x_89": "x_vars[veggie_veg_l]",
    "x_90": "x_vars[veggie_veg_m]",
    "x_91": "x_vars[veggie_veg_s]",
    "x_92": "x_vars[fried_duck_l]",
    "x_93": "x_vars[fried_duck_m]",
    "x_94": "x_vars[fried_duck_s]",
    "x_95": "x_vars[mapo_Tofu_l]",
    "x_96": "x_vars[mapo_Tofu_m]",
    "x_97": "x_vars[mapo_Tofu_s]",
    "x_98": "x_vars[sea_food_l]",
    "x_99": "x_vars[sea_food_m]",
    "x_100": "x_vars[sea_food_s]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-015": {
   "source_code_sha256": "bef1f6d5476d6160dc02a61f3d2bef3e7bffb63478c52685d12b5d0009254a76",
   "original_solution_sha256": "58ef4a44824dc79dc007715fabf283521d701f225a3ee6bda32c27164bea70dc",
   "bindings": [
    [
     "C0",
     "x_vars[Aalopuri]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Aalopuri]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-016": {
   "source_code_sha256": "e21f0a94c951fefd03e1793555491a021fa892faac63e9ccf450306eccf6a163",
   "original_solution_sha256": "a9f0bd8386118d0927b04658047118395831172b45be9d00d6b7fdff5dfb1b08",
   "bindings": [
    [
     "C0",
     "x_vars[Beauty - 25]"
    ],
    [
     "C1",
     "x_vars[Beauty - 30]"
    ],
    [
     "C2",
     "x_vars[Beauty - 300]"
    ],
    [
     "C3",
     "x_vars[Beauty - 50]"
    ],
    [
     "C4",
     "x_vars[Beauty - 500]"
    ],
    [
     "C5",
     "x_vars[Clothing - 25]"
    ],
    [
     "C6",
     "x_vars[Clothing - 30]"
    ],
    [
     "C7",
     "x_vars[Clothing - 300]"
    ],
    [
     "C8",
     "x_vars[Clothing - 50]"
    ],
    [
     "C9",
     "x_vars[Clothing - 500]"
    ],
    [
     "C10",
     "x_vars[Electronics - 25]"
    ],
    [
     "C11",
     "x_vars[Electronics - 30]"
    ],
    [
     "C12",
     "x_vars[Electronics - 300]"
    ],
    [
     "C13",
     "x_vars[Electronics - 50]"
    ],
    [
     "C14",
     "x_vars[Electronics - 500]"
    ],
    [
     "C15",
     "x_vars[Home Goods - 25]"
    ],
    [
     "C16",
     "x_vars[Home Goods - 30]"
    ],
    [
     "C17",
     "x_vars[Home Goods - 50]"
    ],
    [
     "C18",
     "x_vars[Home Goods - 300]"
    ],
    [
     "C19",
     "x_vars[Home Goods - 500]"
    ],
    [
     "C20",
     "x_vars[Sports - 25]"
    ],
    [
     "C21",
     "x_vars[Sports - 30]"
    ],
    [
     "C22",
     "x_vars[Sports - 50]"
    ],
    [
     "C23",
     "x_vars[Sports - 300]"
    ],
    [
     "C24",
     "x_vars[Sports - 500]"
    ],
    [
     "C25",
     "x_vars[Furniture - 25]"
    ],
    [
     "C26",
     "x_vars[Furniture - 30]"
    ],
    [
     "C27",
     "x_vars[Furniture - 50]"
    ],
    [
     "C28",
     "x_vars[Furniture - 300]"
    ],
    [
     "C29",
     "x_vars[Furniture - 500]"
    ],
    [
     "C30",
     "x_vars[Toys - 25]"
    ],
    [
     "C31",
     "x_vars[Toys - 30]"
    ],
    [
     "C32",
     "x_vars[Toys - 50]"
    ],
    [
     "C33",
     "x_vars[Toys - 300]"
    ],
    [
     "C34",
     "x_vars[Toys - 500]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Beauty - 25]",
    "x_2": "x_vars[Beauty - 30]",
    "x_3": "x_vars[Beauty - 300]",
    "x_4": "x_vars[Beauty - 50]",
    "x_5": "x_vars[Beauty - 500]",
    "x_6": "x_vars[Clothing - 25]",
    "x_7": "x_vars[Clothing - 30]",
    "x_8": "x_vars[Clothing - 300]",
    "x_9": "x_vars[Clothing - 50]",
    "x_10": "x_vars[Clothing - 500]",
    "x_11": "x_vars[Electronics - 25]",
    "x_12": "x_vars[Electronics - 30]",
    "x_13": "x_vars[Electronics - 300]",
    "x_14": "x_vars[Electronics - 50]",
    "x_15": "x_vars[Electronics - 500]",
    "x_16": "x_vars[Home Goods - 25]",
    "x_17": "x_vars[Home Goods - 30]",
    "x_18": "x_vars[Home Goods - 50]",
    "x_19": "x_vars[Home Goods - 300]",
    "x_20": "x_vars[Home Goods - 500]",
    "x_21": "x_vars[Sports - 25]",
    "x_22": "x_vars[Sports - 30]",
    "x_23": "x_vars[Sports - 50]",
    "x_24": "x_vars[Sports - 300]",
    "x_25": "x_vars[Sports - 500]",
    "x_26": "x_vars[Furniture - 25]",
    "x_27": "x_vars[Furniture - 30]",
    "x_28": "x_vars[Furniture - 50]",
    "x_29": "x_vars[Furniture - 300]",
    "x_30": "x_vars[Furniture - 500]",
    "x_31": "x_vars[Toys - 25]",
    "x_32": "x_vars[Toys - 30]",
    "x_33": "x_vars[Toys - 50]",
    "x_34": "x_vars[Toys - 300]",
    "x_35": "x_vars[Toys - 500]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-017": {
   "source_code_sha256": "74b6557cfcd3be2518bf1e32ddac43e4884f2d3c6cd3a99aaa7cb699c9db3377",
   "original_solution_sha256": "1adb3ecb4c1e88130f315d3cea4a93a6a9c6e0aed6138a7e474088e6485d8aaa",
   "bindings": [
    [
     "C0",
     "x_vars[ZZ2AO]"
    ],
    [
     "C1",
     "x_vars[ZZDW7]"
    ],
    [
     "C2",
     "x_vars[ZZM1A]"
    ],
    [
     "C3",
     "x_vars[ZZNC5]"
    ],
    [
     "C4",
     "x_vars[ZZX6K]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[ZZ2AO]",
    "x_2": "x_vars[ZZDW7]",
    "x_3": "x_vars[ZZM1A]",
    "x_4": "x_vars[ZZNC5]",
    "x_5": "x_vars[ZZX6K]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-018": {
   "source_code_sha256": "748ca5f30fc14faeae6dad91d921a2a17b34fdf6198902d883880e8e055b912e",
   "original_solution_sha256": "055485cf17ec2c7dee00c799a4f10e17a7660386f1dec9c5add0a9ca664b7cbe",
   "bindings": [
    [
     "C0",
     "x_vars[Baby Food_255.28]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Baby Food_255.28]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-019": {
   "source_code_sha256": "3618fdd808910f25c0cc658b23fe82801661d8ef226af596b328663b9ec0f230",
   "original_solution_sha256": "071705e568cc90567aae65a7638d2c73c37f55cdc04aa3a33d254db5d4d1a1d8",
   "bindings": [
    [
     "C0",
     "x_vars[27in 4K Gaming Monitor]"
    ],
    [
     "C1",
     "x_vars[27in FHD Monitor]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[27in 4K Gaming Monitor]",
    "x_2": "x_vars[27in FHD Monitor]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-020": {
   "source_code_sha256": "151703e8aa42b078fdb1a7558da9483ef956b7b0168586c3b6530bae06916c71",
   "original_solution_sha256": "ec2a59bed57dae6bf5b79cfd83373bc59780941287823cb98bd8179ae4bab9df",
   "bindings": [
    [
     "C0",
     "x_vars[S10_1678]"
    ],
    [
     "C1",
     "x_vars[S10_1949]"
    ],
    [
     "C2",
     "x_vars[S10_2016]"
    ],
    [
     "C3",
     "x_vars[S10_4698]"
    ],
    [
     "C4",
     "x_vars[S10_4757]"
    ],
    [
     "C5",
     "x_vars[S10_4962]"
    ],
    [
     "C6",
     "x_vars[S12_1099]"
    ],
    [
     "C7",
     "x_vars[S12_1108]"
    ],
    [
     "C8",
     "x_vars[S12_1666]"
    ],
    [
     "C9",
     "x_vars[S12_2823]"
    ],
    [
     "C10",
     "x_vars[S12_3148]"
    ],
    [
     "C11",
     "x_vars[S12_3380]"
    ],
    [
     "C12",
     "x_vars[S12_3891]"
    ],
    [
     "C13",
     "x_vars[S12_3990]"
    ],
    [
     "C14",
     "x_vars[S12_4473]"
    ],
    [
     "C15",
     "x_vars[S12_4675]"
    ],
    [
     "C16",
     "x_vars[S18_1097]"
    ],
    [
     "C17",
     "x_vars[S18_1129]"
    ],
    [
     "C18",
     "x_vars[S18_1342]"
    ],
    [
     "C19",
     "x_vars[S18_1367]"
    ],
    [
     "C20",
     "x_vars[S18_1589]"
    ],
    [
     "C21",
     "x_vars[S18_1662]"
    ],
    [
     "C22",
     "x_vars[S18_1749]"
    ],
    [
     "C23",
     "x_vars[S18_1889]"
    ],
    [
     "C24",
     "x_vars[S18_1984]"
    ],
    [
     "C25",
     "x_vars[S18_2238]"
    ],
    [
     "C26",
     "x_vars[S18_2248]"
    ],
    [
     "C27",
     "x_vars[S18_2319]"
    ],
    [
     "C28",
     "x_vars[S18_2325]"
    ],
    [
     "C29",
     "x_vars[S18_2432]"
    ],
    [
     "C30",
     "x_vars[S18_2581]"
    ],
    [
     "C31",
     "x_vars[S18_2625]"
    ],
    [
     "C32",
     "x_vars[S18_2795]"
    ],
    [
     "C33",
     "x_vars[S18_2870]"
    ],
    [
     "C34",
     "x_vars[S18_2949]"
    ],
    [
     "C35",
     "x_vars[S18_2957]"
    ],
    [
     "C36",
     "x_vars[S18_3029]"
    ],
    [
     "C37",
     "x_vars[S18_3136]"
    ],
    [
     "C38",
     "x_vars[S18_3140]"
    ],
    [
     "C39",
     "x_vars[S18_3232]"
    ],
    [
     "C40",
     "x_vars[S18_3259]"
    ],
    [
     "C41",
     "x_vars[S18_3278]"
    ],
    [
     "C42",
     "x_vars[S18_3320]"
    ],
    [
     "C43",
     "x_vars[S18_3482]"
    ],
    [
     "C44",
     "x_vars[S18_3685]"
    ],
    [
     "C45",
     "x_vars[S18_3782]"
    ],
    [
     "C46",
     "x_vars[S18_3856]"
    ],
    [
     "C47",
     "x_vars[S18_4027]"
    ],
    [
     "C48",
     "x_vars[S18_4409]"
    ],
    [
     "C49",
     "x_vars[S18_4522]"
    ],
    [
     "C50",
     "x_vars[S18_4600]"
    ],
    [
     "C51",
     "x_vars[S18_4668]"
    ],
    [
     "C52",
     "x_vars[S18_4721]"
    ],
    [
     "C53",
     "x_vars[S18_4933]"
    ],
    [
     "C54",
     "x_vars[S24_1046]"
    ],
    [
     "C55",
     "x_vars[S24_1444]"
    ],
    [
     "C56",
     "x_vars[S24_1578]"
    ],
    [
     "C57",
     "x_vars[S24_1628]"
    ],
    [
     "C58",
     "x_vars[S24_1785]"
    ],
    [
     "C59",
     "x_vars[S24_1937]"
    ],
    [
     "C60",
     "x_vars[S24_2000]"
    ],
    [
     "C61",
     "x_vars[S24_2011]"
    ],
    [
     "C62",
     "x_vars[S24_2022]"
    ],
    [
     "C63",
     "x_vars[S24_2300]"
    ],
    [
     "C64",
     "x_vars[S24_2360]"
    ],
    [
     "C65",
     "x_vars[S24_2766]"
    ],
    [
     "C66",
     "x_vars[S24_2840]"
    ],
    [
     "C67",
     "x_vars[S24_2841]"
    ],
    [
     "C68",
     "x_vars[S24_2887]"
    ],
    [
     "C69",
     "x_vars[S24_2972]"
    ],
    [
     "C70",
     "x_vars[S24_3151]"
    ],
    [
     "C71",
     "x_vars[S24_3191]"
    ],
    [
     "C72",
     "x_vars[S24_3371]"
    ],
    [
     "C73",
     "x_vars[S24_3420]"
    ],
    [
     "C74",
     "x_vars[S24_3432]"
    ],
    [
     "C75",
     "x_vars[S24_3816]"
    ],
    [
     "C76",
     "x_vars[S24_3856]"
    ],
    [
     "C77",
     "x_vars[S24_3949]"
    ],
    [
     "C78",
     "x_vars[S24_3969]"
    ],
    [
     "C79",
     "x_vars[S24_4048]"
    ],
    [
     "C80",
     "x_vars[S24_4258]"
    ],
    [
     "C81",
     "x_vars[S24_4278]"
    ],
    [
     "C82",
     "x_vars[S24_4620]"
    ],
    [
     "C83",
     "x_vars[S32_1268]"
    ],
    [
     "C84",
     "x_vars[S32_1374]"
    ],
    [
     "C85",
     "x_vars[S32_2206]"
    ],
    [
     "C86",
     "x_vars[S32_2509]"
    ],
    [
     "C87",
     "x_vars[S32_3207]"
    ],
    [
     "C88",
     "x_vars[S32_3522]"
    ],
    [
     "C89",
     "x_vars[S32_4289]"
    ],
    [
     "C90",
     "x_vars[S32_4485]"
    ],
    [
     "C91",
     "x_vars[S50_1341]"
    ],
    [
     "C92",
     "x_vars[S50_1392]"
    ],
    [
     "C93",
     "x_vars[S50_1514]"
    ],
    [
     "C94",
     "x_vars[S50_4713]"
    ],
    [
     "C95",
     "x_vars[S700_1138]"
    ],
    [
     "C96",
     "x_vars[S700_1691]"
    ],
    [
     "C97",
     "x_vars[S700_1938]"
    ],
    [
     "C98",
     "x_vars[S700_2047]"
    ],
    [
     "C99",
     "x_vars[S700_2466]"
    ],
    [
     "C100",
     "x_vars[S700_2610]"
    ],
    [
     "C101",
     "x_vars[S700_2824]"
    ],
    [
     "C102",
     "x_vars[S700_2834]"
    ],
    [
     "C103",
     "x_vars[S700_3167]"
    ],
    [
     "C104",
     "x_vars[S700_3505]"
    ],
    [
     "C105",
     "x_vars[S700_3962]"
    ],
    [
     "C106",
     "x_vars[S700_4002]"
    ],
    [
     "C107",
     "x_vars[S72_1253]"
    ],
    [
     "C108",
     "x_vars[S72_3212]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[S10_1678]",
    "x_2": "x_vars[S10_1949]",
    "x_3": "x_vars[S10_2016]",
    "x_4": "x_vars[S10_4698]",
    "x_5": "x_vars[S10_4757]",
    "x_6": "x_vars[S10_4962]",
    "x_7": "x_vars[S12_1099]",
    "x_8": "x_vars[S12_1108]",
    "x_9": "x_vars[S12_1666]",
    "x_10": "x_vars[S12_2823]",
    "x_11": "x_vars[S12_3148]",
    "x_12": "x_vars[S12_3380]",
    "x_13": "x_vars[S12_3891]",
    "x_14": "x_vars[S12_3990]",
    "x_15": "x_vars[S12_4473]",
    "x_16": "x_vars[S12_4675]",
    "x_17": "x_vars[S18_1097]",
    "x_18": "x_vars[S18_1129]",
    "x_19": "x_vars[S18_1342]",
    "x_20": "x_vars[S18_1367]",
    "x_21": "x_vars[S18_1589]",
    "x_22": "x_vars[S18_1662]",
    "x_23": "x_vars[S18_1749]",
    "x_24": "x_vars[S18_1889]",
    "x_25": "x_vars[S18_1984]",
    "x_26": "x_vars[S18_2238]",
    "x_27": "x_vars[S18_2248]",
    "x_28": "x_vars[S18_2319]",
    "x_29": "x_vars[S18_2325]",
    "x_30": "x_vars[S18_2432]",
    "x_31": "x_vars[S18_2581]",
    "x_32": "x_vars[S18_2625]",
    "x_33": "x_vars[S18_2795]",
    "x_34": "x_vars[S18_2870]",
    "x_35": "x_vars[S18_2949]",
    "x_36": "x_vars[S18_2957]",
    "x_37": "x_vars[S18_3029]",
    "x_38": "x_vars[S18_3136]",
    "x_39": "x_vars[S18_3140]",
    "x_40": "x_vars[S18_3232]",
    "x_41": "x_vars[S18_3259]",
    "x_42": "x_vars[S18_3278]",
    "x_43": "x_vars[S18_3320]",
    "x_44": "x_vars[S18_3482]",
    "x_45": "x_vars[S18_3685]",
    "x_46": "x_vars[S18_3782]",
    "x_47": "x_vars[S18_3856]",
    "x_48": "x_vars[S18_4027]",
    "x_49": "x_vars[S18_4409]",
    "x_50": "x_vars[S18_4522]",
    "x_51": "x_vars[S18_4600]",
    "x_52": "x_vars[S18_4668]",
    "x_53": "x_vars[S18_4721]",
    "x_54": "x_vars[S18_4933]",
    "x_55": "x_vars[S24_1046]",
    "x_56": "x_vars[S24_1444]",
    "x_57": "x_vars[S24_1578]",
    "x_58": "x_vars[S24_1628]",
    "x_59": "x_vars[S24_1785]",
    "x_60": "x_vars[S24_1937]",
    "x_61": "x_vars[S24_2000]",
    "x_62": "x_vars[S24_2011]",
    "x_63": "x_vars[S24_2022]",
    "x_64": "x_vars[S24_2300]",
    "x_65": "x_vars[S24_2360]",
    "x_66": "x_vars[S24_2766]",
    "x_67": "x_vars[S24_2840]",
    "x_68": "x_vars[S24_2841]",
    "x_69": "x_vars[S24_2887]",
    "x_70": "x_vars[S24_2972]",
    "x_71": "x_vars[S24_3151]",
    "x_72": "x_vars[S24_3191]",
    "x_73": "x_vars[S24_3371]",
    "x_74": "x_vars[S24_3420]",
    "x_75": "x_vars[S24_3432]",
    "x_76": "x_vars[S24_3816]",
    "x_77": "x_vars[S24_3856]",
    "x_78": "x_vars[S24_3949]",
    "x_79": "x_vars[S24_3969]",
    "x_80": "x_vars[S24_4048]",
    "x_81": "x_vars[S24_4258]",
    "x_82": "x_vars[S24_4278]",
    "x_83": "x_vars[S24_4620]",
    "x_84": "x_vars[S32_1268]",
    "x_85": "x_vars[S32_1374]",
    "x_86": "x_vars[S32_2206]",
    "x_87": "x_vars[S32_2509]",
    "x_88": "x_vars[S32_3207]",
    "x_89": "x_vars[S32_3522]",
    "x_90": "x_vars[S32_4289]",
    "x_91": "x_vars[S32_4485]",
    "x_92": "x_vars[S50_1341]",
    "x_93": "x_vars[S50_1392]",
    "x_94": "x_vars[S50_1514]",
    "x_95": "x_vars[S50_4713]",
    "x_96": "x_vars[S700_1138]",
    "x_97": "x_vars[S700_1691]",
    "x_98": "x_vars[S700_1938]",
    "x_99": "x_vars[S700_2047]",
    "x_100": "x_vars[S700_2466]",
    "x_101": "x_vars[S700_2610]",
    "x_102": "x_vars[S700_2824]",
    "x_103": "x_vars[S700_2834]",
    "x_104": "x_vars[S700_3167]",
    "x_105": "x_vars[S700_3505]",
    "x_106": "x_vars[S700_3962]",
    "x_107": "x_vars[S700_4002]",
    "x_108": "x_vars[S72_1253]",
    "x_109": "x_vars[S72_3212]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-021": {
   "source_code_sha256": "7fadb7a67b19a65b959e751203e1149b9b7f6f6f33d450e4282df71c96abf926",
   "original_solution_sha256": "adbf0a6c3c119ba3f6846ed025969f53556bb0da6ebc832920cabda62c22537e",
   "bindings": [
    [
     "C0",
     "x_vars[ 3 Colors New Fashion Summer Ladies Casual Jumpsuit Long Suspender Overalls Bib Pants]"
    ],
    [
     "C1",
     "x_vars[ Men's Casual Workout Athletic Gym Jersey Shorts Elastic Waist Drawstring Summer Training Running Knee Length Shorts with Zipper Pocket]"
    ],
    [
     "C2",
     "x_vars[ New Fashion Autumn Summer Women's Long Sleeve V Neck Long Dress Floral Print Split Maxi Dress Holiday Party Beach Sundress Evening Dresses]"
    ],
    [
     "C3",
     "x_vars[ New The New Men's Stitching Design Jogging Sports Cropped Trousers]"
    ],
    [
     "C4",
     "x_vars[ Plus Size S-5XL Women Summer Tops Casual V-Neck Short Sleeve Shirts Ladies Cotton Loose T Shirt Candy Color Lady Pullovers Blouse]"
    ],
    [
     "C5",
     "x_vars[ Plus Size Women Halter Striped Wide Leg Pants Casual Jumpsuit Rompers Fashion Shorts ]"
    ],
    [
     "C6",
     "x_vars[ Spring/Summer Fashion Women \"honey\"Letter Print Sleeveless Shirt Sexy Embroidered Bodycon Vest Knitted Cotton Vest Casual Tank Top ]"
    ],
    [
     "C7",
     "x_vars[ Summer Students Style Bodycon Rompers Womens Slim Fit Jumpsuit Chest Zipper Contrast Color Short Sleeve V-neck Shorts Bodysuit]"
    ],
    [
     "C8",
     "x_vars[ Summer Women s Fashion Lace Up Tie Pants Plus Size Casual High Waist Short Pants(S-5XL)]"
    ],
    [
     "C9",
     "x_vars[ Women's Fashion Graphic \\Don't Flatter Yourself....\\Tee for Women Summer Casual Tee T Shirts for Girls]"
    ],
    [
     "C10",
     "x_vars[ XS - XXL Fashion Women's Summer Casual T - Shirt Love Gesture Printed Short Sleeve Blouse XS - XXL]"
    ],
    [
     "C11",
     "x_vars['Let That Shit Go' Women Graphic Tee Casual Cotton Short Sleeve T-shirts Bohemia Style Mandala Namaste Printed Top Blouse]"
    ],
    [
     "C12",
     "x_vars[(S-5XL) Women Fashion Summer Double-Layer Sports Shorts Quick-Drying Yoga Sports Leggings Fitness Shorts Plus Size]"
    ],
    [
     "C13",
     "x_vars[(S-7XL) Women Camisole Summer Printed Vest Floral Sleeveless Vest V-neck Shirt Plus Size]"
    ],
    [
     "C14",
     "x_vars[(US Size) Cotton Graphic Tees for Women / Girls: 4 Colors, Spring Summer Tee, Cute Funny Short Sleeve T-Shirt, Loose Blouse Tops]"
    ],
    [
     "C15",
     "x_vars[1 Pcs Swimming Chair/Bed Swimming Pool Seat Inflatable Lazy Bed Lounge Chair Air Mattress Floating Bed/Chair with Net Foldable for Swimming Pool/Beach Water Relaxation]"
    ],
    [
     "C16",
     "x_vars[10 Color Women Summer Shorts Lace Up Elastic Waistband Loose Panties Plus Size S-6XL]"
    ],
    [
     "C17",
     "x_vars[10/20/30ml Slimming Firming Anti-Cellulite Massage Cream Fat Burning Weight Losing Moisturizer Gel for Shaping Waist]"
    ],
    [
     "C18",
     "x_vars[10/30/50ml Hair Removal Spray Super Natural Painless Permanent Depilatory Cream]"
    ],
    [
     "C19",
     "x_vars[100%Cotton ZANZEA S-5XL NEW Vintage Women Strap Dungaree Jumpsuit Casual Loose Trousers Overalls Rompers Jumpsuit]"
    ],
    [
     "C20",
     "x_vars[100pcs Summer Disposable Sweat Pad Perspiration Absorbing Guard Underarm Armpit Sweat Pad Pure Antiperspirant Adhesive Underarm Pads]"
    ],
    [
     "C21",
     "x_vars[11 Colors V Neck Women Sleeveless Tops Halter Neck Plus Size Chiffon CropTops(S~5XL)]"
    ],
    [
     "C22",
     "x_vars[16 Color Fashion Women Casual Sleeveless Camisoles Loose T-shirts with Ziper Deep V-neck Blouses Ladies Chiffon Shirts Tank Tops ]"
    ],
    [
     "C23",
     "x_vars[170cm Inflatable Spray Water Cushion Summer Kids Play Water Mat Lawn Games Pad Sprinkler Play Toys Outdoor Tub Swiming Pool ]"
    ],
    [
     "C24",
     "x_vars[18 Styles Women Summer Sexy Printing Buttons Off Shoulder Sleeveless Dress Princess Dress Spaghetti Straps Dresses]"
    ],
    [
     "C25",
     "x_vars[1PCS Swimwear Monokini Swimsuit Backless Bodysuit Women Swimsuit]"
    ],
    [
     "C26",
     "x_vars[1Set Lace Bikini Diamond Swimsuit Crystal Women Swimwear Nude Bikinis Brazilian Rhinestone Beachwear Push Up Bikini]"
    ],
    [
     "C27",
     "x_vars[1pcs Men's Running Shorts 2 in 1 Sports Jogging Fitness Shorts Summer Training Quick Dry Short Gym Shorts Sport Pants]"
    ],
    [
     "C28",
     "x_vars[20 Pcs Fashion Comfortable Short Socks Candy Color Invisible Silicone Non-slip Socks Slippers Socks]"
    ],
    [
     "C29",
     "x_vars[20/10pcs Women Ankle Invisible No Show Nonslip Loafer Boat Liner Cotton Socks Comfortable Socks Women Shoes Accessories  (Application Size: 33-43)]"
    ],
    [
     "C30",
     "x_vars[2017 Women Ladies Summer Dress Sleeveless Casual Sexy Floral Print Beach Dress Fashion Spaghetti Strap Mini Short Dress]"
    ],
    [
     "C31",
     "x_vars[2018 6 Color Summer Womenâ__s New Fashion Sexy Cute Whiskey Print Lace Patchwork Spaghetti Strap T-Shirts Slim Bodycon Off The Shoulder Short Sleeve Blouse Casual Cotton Outdoor Tops Plus Size S-XXXXL]"
    ],
    [
     "C32",
     "x_vars[2018 Black Flora Flower Printed Short Dress Women]"
    ],
    [
     "C33",
     "x_vars[2018 Fashion Summer Dress Women Sexy Dresses V Neck Backless Lace Stitching Dress Beach Dresses White Dress]"
    ],
    [
     "C34",
     "x_vars[2018 Hot Selling Spring Summer Women Flared Causal Trousers Loose Pants Drawstring Elastic Waist Middle Waisted  Wide Leg Pants]"
    ],
    [
     "C35",
     "x_vars[2018 New Fashion Women Bikini Set Push-up Padded Bra Swimsuit -Swimwear]"
    ],
    [
     "C36",
     "x_vars[2018 New Fashion Women Casual Playsuit Ladies Jumpsuit Romper Summer Floral Playsuit Brand New (3 Colors)]"
    ],
    [
     "C37",
     "x_vars[2018 New Fashion Women's Tops Sexy Strappy Sleeveless Lace Crop Tops]"
    ],
    [
     "C38",
     "x_vars[2018 New Summer Women Sexy V-neck Bandage T-shirts Camouflage Printed Short Sleeve Plus size Tops Tee(S-4XL)]"
    ],
    [
     "C39",
     "x_vars[2018 New The new cross-strait national wind body bikini Swimsuit  Swimwear LBT]"
    ],
    [
     "C40",
     "x_vars[2018 New ZANZEA Beach Romper Women Jumpsuits Front Zipper Sleeveless Sexy Playsuits]"
    ],
    [
     "C41",
     "x_vars[2018 Plus Size Summer Women Fashion Sexy V-neck Lace Up Plaid Blouse Tops Irregular Short Sleeve Shirtï¼_S-5XLï¼_]"
    ],
    [
     "C42",
     "x_vars[2018 Summer Fashion Women Casual Camouflage Tank Top Sleeveless O-neck Vest]"
    ],
    [
     "C43",
     "x_vars[2018 Summer Fashion Women Tank Tops Sexy Women Sleeveless Crop Tops Casual Style Women Cotton Print Lace Stitching Irregular Blouse Topsï¼_S-5XL)]"
    ],
    [
     "C44",
     "x_vars[2018 Summer New Women Fashion Sexy Sleeveless Tank Top Floral Print Casual Women Cotton T-shirts Plus Size Women Vestï¼_S-5XLï¼_]"
    ],
    [
     "C45",
     "x_vars[2018 Summer New Women Long Skirts Solid Sexy Split Pencil Skirts]"
    ],
    [
     "C46",
     "x_vars[2018 Summer Women Print Top Fashion Women Casual Army Camo Camouflage Tank Top Sleeveless O-neck Slim T-Shirts Plus Size S-XXXXXL]"
    ],
    [
     "C47",
     "x_vars[2018 Summer Women Sexy Off Shoulder Shirt Casual Half Sleeve Top Ladies Fashion Printing Loose T-shirts Plus Size Cotton Blouse XS-5XL]"
    ],
    [
     "C48",
     "x_vars[2018 Summer Womenâ__s New Fashion Whiskey Print Lace Patchwork Spaghetti Strap T-Shirts Casual Sexy Cotton Tops]"
    ],
    [
     "C49",
     "x_vars[2018 T Shirts + Shorts Brand Clothing Tshirt Men Homme Letter Printed Basketball Running Sports Set T-shirt Suit Male]"
    ],
    [
     "C50",
     "x_vars[2018 Women Cool T-shirt Funny 3d Tshirt Print Two Cat Short Sleeve Summer Tops Tees Teen Graphic Tee Cute Shirt Funny Gifts]"
    ],
    [
     "C51",
     "x_vars[2018 Women Fashion Stretchy Camisole  Spaghetti Strap Long Tank Top Slip Summer Fashion Floral Mini Dress 7 Color SIZE XXL]"
    ],
    [
     "C52",
     "x_vars[2018 Women Fashion Summer Sleeveless Maxi Halter Dresses Solid Color Halter Maxi Dress with Halter Tie and Pockets]"
    ],
    [
     "C53",
     "x_vars[2018 Women Ladies Fashion Crocheted Lace Summer Dress Maxi Dress Long Dress]"
    ],
    [
     "C54",
     "x_vars[2018 Women Ladies Fashion Musical Note Asymmetric Tank Top]"
    ],
    [
     "C55",
     "x_vars[2018 Women Summer Casual Solid Color Loose Sleeveless Beach Tank Top A-line Pocket Dress Sexy Deep V-neck Short Club Party Mini Halter Midi Dresses Robe Ete Femme Knee Length Pleated Skirts Ladies Fashion Swing Cotton T-Shirt Dress Plus Size S-6XL]"
    ],
    [
     "C56",
     "x_vars[2018 Women fashion Summer Lace Patchwork Tank Tops Casual Sleeveless Tops Vest Blouse (S-5XL) Plus Size]"
    ],
    [
     "C57",
     "x_vars[2019 Fashion Family matching swimwear beachwear mommy and me swimsuit mother daughter father son clothes dresses high waist bikini look mum]"
    ],
    [
     "C58",
     "x_vars[2019 Fashion Women Summer V-Neck Sleeveless Collect Waist Boho Print Maxi Long Dress S-XXXXXL]"
    ],
    [
     "C59",
     "x_vars[2019 Men's Summer Cool Jogging Shorts]"
    ],
    [
     "C60",
     "x_vars[2019 NEW Men's Short Sleeve T-Shirt Fitness Round Neck Casual Men's Zippers Summer T-Shirt]"
    ],
    [
     "C61",
     "x_vars[2019 New Fashion Spring Summer Women's Long Pants Short Sleeves Off Shoulder Jumpsuit]"
    ],
    [
     "C62",
     "x_vars[2019 New Fashion Summer Women Casual Dress Round Neck Loose Big Swing Skirt Sleeveless Soild Color Beach dress]"
    ],
    [
     "C63",
     "x_vars[2019 New Fashion Women Blouse Tops Sleeveless Spaghetti Strap Criss Cross V-neck T Shirt Tops Summer Casual Blouse Shirts]"
    ],
    [
     "C64",
     "x_vars[2019 New Fashion Women Casual Shorts Suit Summer Tie-Dye Print Halter Bandage Sleeveless Backless Crop Top And Elastic Waist Shorts Pants Two Piece Set]"
    ],
    [
     "C65",
     "x_vars[2019 New Fashion Women Loose Round Neck Dress Short Sleeve Solid Color Casual Dress]"
    ],
    [
     "C66",
     "x_vars[2019 New Fashion Women Summer Funny Penguin print T-Shirts Casual Short Sleeve O-Neck Tops Loose Cotton Cute Cool Tee Plus Size S-5XL 4 Colors]"
    ],
    [
     "C67",
     "x_vars[2019 New Fashion Women's Slim Plus Size Maxi Dress Ink Printing Spaghetti Strap Dress]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[ 3 Colors New Fashion Summer Ladies Casual Jumpsuit Long Suspender Overalls Bib Pants]",
    "x_2": "x_vars[ Men's Casual Workout Athletic Gym Jersey Shorts Elastic Waist Drawstring Summer Training Running Knee Length Shorts with Zipper Pocket]",
    "x_3": "x_vars[ New Fashion Autumn Summer Women's Long Sleeve V Neck Long Dress Floral Print Split Maxi Dress Holiday Party Beach Sundress Evening Dresses]",
    "x_4": "x_vars[ New The New Men's Stitching Design Jogging Sports Cropped Trousers]",
    "x_5": "x_vars[ Plus Size S-5XL Women Summer Tops Casual V-Neck Short Sleeve Shirts Ladies Cotton Loose T Shirt Candy Color Lady Pullovers Blouse]",
    "x_6": "x_vars[ Plus Size Women Halter Striped Wide Leg Pants Casual Jumpsuit Rompers Fashion Shorts ]",
    "x_7": "x_vars[ Spring/Summer Fashion Women \"honey\"Letter Print Sleeveless Shirt Sexy Embroidered Bodycon Vest Knitted Cotton Vest Casual Tank Top ]",
    "x_8": "x_vars[ Summer Students Style Bodycon Rompers Womens Slim Fit Jumpsuit Chest Zipper Contrast Color Short Sleeve V-neck Shorts Bodysuit]",
    "x_9": "x_vars[ Summer Women s Fashion Lace Up Tie Pants Plus Size Casual High Waist Short Pants(S-5XL)]",
    "x_10": "x_vars[ Women's Fashion Graphic \\Don't Flatter Yourself....\\Tee for Women Summer Casual Tee T Shirts for Girls]",
    "x_11": "x_vars[ XS - XXL Fashion Women's Summer Casual T - Shirt Love Gesture Printed Short Sleeve Blouse XS - XXL]",
    "x_12": "x_vars['Let That Shit Go' Women Graphic Tee Casual Cotton Short Sleeve T-shirts Bohemia Style Mandala Namaste Printed Top Blouse]",
    "x_13": "x_vars[(S-5XL) Women Fashion Summer Double-Layer Sports Shorts Quick-Drying Yoga Sports Leggings Fitness Shorts Plus Size]",
    "x_14": "x_vars[(S-7XL) Women Camisole Summer Printed Vest Floral Sleeveless Vest V-neck Shirt Plus Size]",
    "x_15": "x_vars[(US Size) Cotton Graphic Tees for Women / Girls: 4 Colors, Spring Summer Tee, Cute Funny Short Sleeve T-Shirt, Loose Blouse Tops]",
    "x_16": "x_vars[1 Pcs Swimming Chair/Bed Swimming Pool Seat Inflatable Lazy Bed Lounge Chair Air Mattress Floating Bed/Chair with Net Foldable for Swimming Pool/Beach Water Relaxation]",
    "x_17": "x_vars[10 Color Women Summer Shorts Lace Up Elastic Waistband Loose Panties Plus Size S-6XL]",
    "x_18": "x_vars[10/20/30ml Slimming Firming Anti-Cellulite Massage Cream Fat Burning Weight Losing Moisturizer Gel for Shaping Waist]",
    "x_19": "x_vars[10/30/50ml Hair Removal Spray Super Natural Painless Permanent Depilatory Cream]",
    "x_20": "x_vars[100%Cotton ZANZEA S-5XL NEW Vintage Women Strap Dungaree Jumpsuit Casual Loose Trousers Overalls Rompers Jumpsuit]",
    "x_21": "x_vars[100pcs Summer Disposable Sweat Pad Perspiration Absorbing Guard Underarm Armpit Sweat Pad Pure Antiperspirant Adhesive Underarm Pads]",
    "x_22": "x_vars[11 Colors V Neck Women Sleeveless Tops Halter Neck Plus Size Chiffon CropTops(S~5XL)]",
    "x_23": "x_vars[16 Color Fashion Women Casual Sleeveless Camisoles Loose T-shirts with Ziper Deep V-neck Blouses Ladies Chiffon Shirts Tank Tops ]",
    "x_24": "x_vars[170cm Inflatable Spray Water Cushion Summer Kids Play Water Mat Lawn Games Pad Sprinkler Play Toys Outdoor Tub Swiming Pool ]",
    "x_25": "x_vars[18 Styles Women Summer Sexy Printing Buttons Off Shoulder Sleeveless Dress Princess Dress Spaghetti Straps Dresses]",
    "x_26": "x_vars[1PCS Swimwear Monokini Swimsuit Backless Bodysuit Women Swimsuit]",
    "x_27": "x_vars[1Set Lace Bikini Diamond Swimsuit Crystal Women Swimwear Nude Bikinis Brazilian Rhinestone Beachwear Push Up Bikini]",
    "x_28": "x_vars[1pcs Men's Running Shorts 2 in 1 Sports Jogging Fitness Shorts Summer Training Quick Dry Short Gym Shorts Sport Pants]",
    "x_29": "x_vars[20 Pcs Fashion Comfortable Short Socks Candy Color Invisible Silicone Non-slip Socks Slippers Socks]",
    "x_30": "x_vars[20/10pcs Women Ankle Invisible No Show Nonslip Loafer Boat Liner Cotton Socks Comfortable Socks Women Shoes Accessories  (Application Size: 33-43)]",
    "x_31": "x_vars[2017 Women Ladies Summer Dress Sleeveless Casual Sexy Floral Print Beach Dress Fashion Spaghetti Strap Mini Short Dress]",
    "x_32": "x_vars[2018 6 Color Summer Womenâ__s New Fashion Sexy Cute Whiskey Print Lace Patchwork Spaghetti Strap T-Shirts Slim Bodycon Off The Shoulder Short Sleeve Blouse Casual Cotton Outdoor Tops Plus Size S-XXXXL]",
    "x_33": "x_vars[2018 Black Flora Flower Printed Short Dress Women]",
    "x_34": "x_vars[2018 Fashion Summer Dress Women Sexy Dresses V Neck Backless Lace Stitching Dress Beach Dresses White Dress]",
    "x_35": "x_vars[2018 Hot Selling Spring Summer Women Flared Causal Trousers Loose Pants Drawstring Elastic Waist Middle Waisted  Wide Leg Pants]",
    "x_36": "x_vars[2018 New Fashion Women Bikini Set Push-up Padded Bra Swimsuit -Swimwear]",
    "x_37": "x_vars[2018 New Fashion Women Casual Playsuit Ladies Jumpsuit Romper Summer Floral Playsuit Brand New (3 Colors)]",
    "x_38": "x_vars[2018 New Fashion Women's Tops Sexy Strappy Sleeveless Lace Crop Tops]",
    "x_39": "x_vars[2018 New Summer Women Sexy V-neck Bandage T-shirts Camouflage Printed Short Sleeve Plus size Tops Tee(S-4XL)]",
    "x_40": "x_vars[2018 New The new cross-strait national wind body bikini Swimsuit  Swimwear LBT]",
    "x_41": "x_vars[2018 New ZANZEA Beach Romper Women Jumpsuits Front Zipper Sleeveless Sexy Playsuits]",
    "x_42": "x_vars[2018 Plus Size Summer Women Fashion Sexy V-neck Lace Up Plaid Blouse Tops Irregular Short Sleeve Shirtï¼_S-5XLï¼_]",
    "x_43": "x_vars[2018 Summer Fashion Women Casual Camouflage Tank Top Sleeveless O-neck Vest]",
    "x_44": "x_vars[2018 Summer Fashion Women Tank Tops Sexy Women Sleeveless Crop Tops Casual Style Women Cotton Print Lace Stitching Irregular Blouse Topsï¼_S-5XL)]",
    "x_45": "x_vars[2018 Summer New Women Fashion Sexy Sleeveless Tank Top Floral Print Casual Women Cotton T-shirts Plus Size Women Vestï¼_S-5XLï¼_]",
    "x_46": "x_vars[2018 Summer New Women Long Skirts Solid Sexy Split Pencil Skirts]",
    "x_47": "x_vars[2018 Summer Women Print Top Fashion Women Casual Army Camo Camouflage Tank Top Sleeveless O-neck Slim T-Shirts Plus Size S-XXXXXL]",
    "x_48": "x_vars[2018 Summer Women Sexy Off Shoulder Shirt Casual Half Sleeve Top Ladies Fashion Printing Loose T-shirts Plus Size Cotton Blouse XS-5XL]",
    "x_49": "x_vars[2018 Summer Womenâ__s New Fashion Whiskey Print Lace Patchwork Spaghetti Strap T-Shirts Casual Sexy Cotton Tops]",
    "x_50": "x_vars[2018 T Shirts + Shorts Brand Clothing Tshirt Men Homme Letter Printed Basketball Running Sports Set T-shirt Suit Male]",
    "x_51": "x_vars[2018 Women Cool T-shirt Funny 3d Tshirt Print Two Cat Short Sleeve Summer Tops Tees Teen Graphic Tee Cute Shirt Funny Gifts]",
    "x_52": "x_vars[2018 Women Fashion Stretchy Camisole  Spaghetti Strap Long Tank Top Slip Summer Fashion Floral Mini Dress 7 Color SIZE XXL]",
    "x_53": "x_vars[2018 Women Fashion Summer Sleeveless Maxi Halter Dresses Solid Color Halter Maxi Dress with Halter Tie and Pockets]",
    "x_54": "x_vars[2018 Women Ladies Fashion Crocheted Lace Summer Dress Maxi Dress Long Dress]",
    "x_55": "x_vars[2018 Women Ladies Fashion Musical Note Asymmetric Tank Top]",
    "x_56": "x_vars[2018 Women Summer Casual Solid Color Loose Sleeveless Beach Tank Top A-line Pocket Dress Sexy Deep V-neck Short Club Party Mini Halter Midi Dresses Robe Ete Femme Knee Length Pleated Skirts Ladies Fashion Swing Cotton T-Shirt Dress Plus Size S-6XL]",
    "x_57": "x_vars[2018 Women fashion Summer Lace Patchwork Tank Tops Casual Sleeveless Tops Vest Blouse (S-5XL) Plus Size]",
    "x_58": "x_vars[2019 Fashion Family matching swimwear beachwear mommy and me swimsuit mother daughter father son clothes dresses high waist bikini look mum]",
    "x_59": "x_vars[2019 Fashion Women Summer V-Neck Sleeveless Collect Waist Boho Print Maxi Long Dress S-XXXXXL]",
    "x_60": "x_vars[2019 Men's Summer Cool Jogging Shorts]",
    "x_61": "x_vars[2019 NEW Men's Short Sleeve T-Shirt Fitness Round Neck Casual Men's Zippers Summer T-Shirt]",
    "x_62": "x_vars[2019 New Fashion Spring Summer Women's Long Pants Short Sleeves Off Shoulder Jumpsuit]",
    "x_63": "x_vars[2019 New Fashion Summer Women Casual Dress Round Neck Loose Big Swing Skirt Sleeveless Soild Color Beach dress]",
    "x_64": "x_vars[2019 New Fashion Women Blouse Tops Sleeveless Spaghetti Strap Criss Cross V-neck T Shirt Tops Summer Casual Blouse Shirts]",
    "x_65": "x_vars[2019 New Fashion Women Casual Shorts Suit Summer Tie-Dye Print Halter Bandage Sleeveless Backless Crop Top And Elastic Waist Shorts Pants Two Piece Set]",
    "x_66": "x_vars[2019 New Fashion Women Loose Round Neck Dress Short Sleeve Solid Color Casual Dress]",
    "x_67": "x_vars[2019 New Fashion Women Summer Funny Penguin print T-Shirts Casual Short Sleeve O-Neck Tops Loose Cotton Cute Cool Tee Plus Size S-5XL 4 Colors]",
    "x_68": "x_vars[2019 New Fashion Women's Slim Plus Size Maxi Dress Ink Printing Spaghetti Strap Dress]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-022": {
   "source_code_sha256": "79cd440ac5a8fdfcbc7151a5be0da9ea0d80b55258030b36d0fc687185146700",
   "original_solution_sha256": "e83759268b35c4c0697e96d5065e61abf933c0b0d55ec07a8cfeec9bd1abe122",
   "bindings": [
    [
     "C0",
     "x_vars[27in 4K Gaming Monitor]"
    ],
    [
     "C1",
     "x_vars[27in FHD Monitor]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[27in 4K Gaming Monitor]",
    "x_2": "x_vars[27in FHD Monitor]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-023": {
   "source_code_sha256": "58deca8e32cea92763504563d02fc551a7611d940fe3e5519d87a163f1e81008",
   "original_solution_sha256": "857b7c8f4516bfd435af112fc052f95ad5f99355ede98eb706dee8cd9328b565",
   "bindings": [
    [
     "C0",
     "x_vars[ELE-SMA-10000463]"
    ],
    [
     "C1",
     "x_vars[ELE-SMA-10000487]"
    ],
    [
     "C2",
     "x_vars[ELE-SMA-10003333]"
    ],
    [
     "C3",
     "x_vars[ELE-SMA-10009012]"
    ],
    [
     "C4",
     "x_vars[ELE-SMA-10009999]"
    ],
    [
     "C5",
     "x_vars[ELE-SMA-10011234]"
    ],
    [
     "C6",
     "x_vars[ELE-SMA-10027456]"
    ],
    [
     "C7",
     "x_vars[ELE-SMA-10028567]"
    ],
    [
     "C8",
     "x_vars[ELE-SPE-10000484]"
    ],
    [
     "C9",
     "x_vars[ELE-SPE-10003030]"
    ],
    [
     "C10",
     "x_vars[ELE-SPE-10024123]"
    ],
    [
     "C11",
     "x_vars[ELE-SPE-10025234]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[ELE-SMA-10000463]",
    "x_2": "x_vars[ELE-SMA-10000487]",
    "x_3": "x_vars[ELE-SMA-10003333]",
    "x_4": "x_vars[ELE-SMA-10009012]",
    "x_5": "x_vars[ELE-SMA-10009999]",
    "x_6": "x_vars[ELE-SMA-10011234]",
    "x_7": "x_vars[ELE-SMA-10027456]",
    "x_8": "x_vars[ELE-SMA-10028567]",
    "x_9": "x_vars[ELE-SPE-10000484]",
    "x_10": "x_vars[ELE-SPE-10003030]",
    "x_11": "x_vars[ELE-SPE-10024123]",
    "x_12": "x_vars[ELE-SPE-10025234]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-024": {
   "source_code_sha256": "0a08955d6ede0a5926d92d76d2a639077262a61a9ca06832d1d03174750c6df2",
   "original_solution_sha256": "7d7acdf448c3687be7923bf582fa7e480bada5f0038ec1298fe4d05bf00de380",
   "bindings": [
    [
     "C0",
     "x_vars[S700_1138]"
    ],
    [
     "C1",
     "x_vars[S700_1691]"
    ],
    [
     "C2",
     "x_vars[S700_1938]"
    ],
    [
     "C3",
     "x_vars[S700_2047]"
    ],
    [
     "C4",
     "x_vars[S700_2466]"
    ],
    [
     "C5",
     "x_vars[S700_2610]"
    ],
    [
     "C6",
     "x_vars[S700_2824]"
    ],
    [
     "C7",
     "x_vars[S700_2834]"
    ],
    [
     "C8",
     "x_vars[S700_3167]"
    ],
    [
     "C9",
     "x_vars[S700_3505]"
    ],
    [
     "C10",
     "x_vars[S700_3962]"
    ],
    [
     "C11",
     "x_vars[S700_4002]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[S700_1138]",
    "x_2": "x_vars[S700_1691]",
    "x_3": "x_vars[S700_1938]",
    "x_4": "x_vars[S700_2047]",
    "x_5": "x_vars[S700_2466]",
    "x_6": "x_vars[S700_2610]",
    "x_7": "x_vars[S700_2824]",
    "x_8": "x_vars[S700_2834]",
    "x_9": "x_vars[S700_3167]",
    "x_10": "x_vars[S700_3505]",
    "x_11": "x_vars[S700_3962]",
    "x_12": "x_vars[S700_4002]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-025": {
   "source_code_sha256": "f5cc5924e305172fc0890811b0003edd55ff371c2e381190b0f6c34f5234cc6a",
   "original_solution_sha256": "bea986bfc281338dcb4cdd25bd07ead2ec842c1b016df01a8522411c797e9871",
   "bindings": [
    [
     "C0",
     "x_vars[TABLET_10084.74]"
    ],
    [
     "C1",
     "x_vars[TABLET_12211.86]"
    ],
    [
     "C2",
     "x_vars[TABLET_14669.5]"
    ],
    [
     "C3",
     "x_vars[TABLET_14745.76]"
    ],
    [
     "C4",
     "x_vars[TABLET_14754.24]"
    ],
    [
     "C5",
     "x_vars[TABLET_16448.3]"
    ],
    [
     "C6",
     "x_vars[TABLET_16448.31]"
    ],
    [
     "C7",
     "x_vars[TABLET_20143.22]"
    ],
    [
     "C8",
     "x_vars[TABLET_2042.38]"
    ],
    [
     "C9",
     "x_vars[TABLET_24915.25]"
    ],
    [
     "C10",
     "x_vars[TABLET_24915.26]"
    ],
    [
     "C11",
     "x_vars[TABLET_26448.3]"
    ],
    [
     "C12",
     "x_vars[TABLET_27042.38]"
    ],
    [
     "C13",
     "x_vars[TABLET_30000.0]"
    ],
    [
     "C14",
     "x_vars[TABLET_33397.46]"
    ],
    [
     "C15",
     "x_vars[TABLET_33398.3]"
    ],
    [
     "C16",
     "x_vars[TABLET_48567.8]"
    ],
    [
     "C17",
     "x_vars[TABLET_48644.07]"
    ],
    [
     "C18",
     "x_vars[TABLET_50262.72]"
    ],
    [
     "C19",
     "x_vars[TABLET_53736.44]"
    ],
    [
     "C20",
     "x_vars[TABLET_6957.62]"
    ],
    [
     "C21",
     "x_vars[TABLET_6957.63]"
    ],
    [
     "C22",
     "x_vars[TABLET_7550.84]"
    ],
    [
     "C23",
     "x_vars[TABLET_7550.85]"
    ],
    [
     "C24",
     "x_vars[TABLET_9584.74]"
    ],
    [
     "C25",
     "x_vars[TABLET_9661.02]"
    ],
    [
     "C26",
     "x_vars[TABLET_9669.5]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[TABLET_10084.74]",
    "x_2": "x_vars[TABLET_12211.86]",
    "x_3": "x_vars[TABLET_14669.5]",
    "x_4": "x_vars[TABLET_14745.76]",
    "x_5": "x_vars[TABLET_14754.24]",
    "x_6": "x_vars[TABLET_16448.3]",
    "x_7": "x_vars[TABLET_16448.31]",
    "x_8": "x_vars[TABLET_20143.22]",
    "x_9": "x_vars[TABLET_2042.38]",
    "x_10": "x_vars[TABLET_24915.25]",
    "x_11": "x_vars[TABLET_24915.26]",
    "x_12": "x_vars[TABLET_26448.3]",
    "x_13": "x_vars[TABLET_27042.38]",
    "x_14": "x_vars[TABLET_30000.0]",
    "x_15": "x_vars[TABLET_33397.46]",
    "x_16": "x_vars[TABLET_33398.3]",
    "x_17": "x_vars[TABLET_48567.8]",
    "x_18": "x_vars[TABLET_48644.07]",
    "x_19": "x_vars[TABLET_50262.72]",
    "x_20": "x_vars[TABLET_53736.44]",
    "x_21": "x_vars[TABLET_6957.62]",
    "x_22": "x_vars[TABLET_6957.63]",
    "x_23": "x_vars[TABLET_7550.84]",
    "x_24": "x_vars[TABLET_7550.85]",
    "x_25": "x_vars[TABLET_9584.74]",
    "x_26": "x_vars[TABLET_9661.02]",
    "x_27": "x_vars[TABLET_9669.5]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-026": {
   "source_code_sha256": "384b3ae402340587b30928823ca109a67fcdfa3ef34a6ab4cfb9489fa730a59c",
   "original_solution_sha256": "98299db46bc1cda46ec192ef54092dba5fd6595a39d763406f43818c2fabc865",
   "bindings": [
    [
     "C0",
     "x_vars[Fashion accessories_10.18]"
    ],
    [
     "C1",
     "x_vars[Fashion accessories_12.09]"
    ],
    [
     "C2",
     "x_vars[Fashion accessories_12.19]"
    ],
    [
     "C3",
     "x_vars[Fashion accessories_12.54]"
    ],
    [
     "C4",
     "x_vars[Fashion accessories_12.78]"
    ],
    [
     "C5",
     "x_vars[Fashion accessories_14.48]"
    ],
    [
     "C6",
     "x_vars[Fashion accessories_15.43]"
    ],
    [
     "C7",
     "x_vars[Fashion accessories_15.5]"
    ],
    [
     "C8",
     "x_vars[Fashion accessories_15.62]"
    ],
    [
     "C9",
     "x_vars[Fashion accessories_16.28]"
    ],
    [
     "C10",
     "x_vars[Fashion accessories_16.45]"
    ],
    [
     "C11",
     "x_vars[Fashion accessories_17.48]"
    ],
    [
     "C12",
     "x_vars[Fashion accessories_17.49]"
    ],
    [
     "C13",
     "x_vars[Fashion accessories_17.87]"
    ],
    [
     "C14",
     "x_vars[Fashion accessories_17.94]"
    ],
    [
     "C15",
     "x_vars[Fashion accessories_18.08]"
    ],
    [
     "C16",
     "x_vars[Fashion accessories_19.66]"
    ],
    [
     "C17",
     "x_vars[Fashion accessories_19.7]"
    ],
    [
     "C18",
     "x_vars[Fashion accessories_19.77]"
    ],
    [
     "C19",
     "x_vars[Fashion accessories_20.01]"
    ],
    [
     "C20",
     "x_vars[Fashion accessories_21.32]"
    ],
    [
     "C21",
     "x_vars[Fashion accessories_21.48]"
    ],
    [
     "C22",
     "x_vars[Fashion accessories_21.94]"
    ],
    [
     "C23",
     "x_vars[Fashion accessories_22.32]"
    ],
    [
     "C24",
     "x_vars[Fashion accessories_22.51]"
    ],
    [
     "C25",
     "x_vars[Fashion accessories_23.82]"
    ],
    [
     "C26",
     "x_vars[Fashion accessories_25.42]"
    ],
    [
     "C27",
     "x_vars[Fashion accessories_25.56]"
    ],
    [
     "C28",
     "x_vars[Fashion accessories_27.02]"
    ],
    [
     "C29",
     "x_vars[Fashion accessories_27.18]"
    ],
    [
     "C30",
     "x_vars[Fashion accessories_27.38]"
    ],
    [
     "C31",
     "x_vars[Fashion accessories_29.42]"
    ],
    [
     "C32",
     "x_vars[Fashion accessories_29.56]"
    ],
    [
     "C33",
     "x_vars[Fashion accessories_30.14]"
    ],
    [
     "C34",
     "x_vars[Fashion accessories_30.37]"
    ],
    [
     "C35",
     "x_vars[Fashion accessories_30.61]"
    ],
    [
     "C36",
     "x_vars[Fashion accessories_30.62]"
    ],
    [
     "C37",
     "x_vars[Fashion accessories_31.73]"
    ],
    [
     "C38",
     "x_vars[Fashion accessories_31.9]"
    ],
    [
     "C39",
     "x_vars[Fashion accessories_32.62]"
    ],
    [
     "C40",
     "x_vars[Fashion accessories_33.52]"
    ],
    [
     "C41",
     "x_vars[Fashion accessories_33.63]"
    ],
    [
     "C42",
     "x_vars[Fashion accessories_34.7]"
    ],
    [
     "C43",
     "x_vars[Fashion accessories_35.19]"
    ],
    [
     "C44",
     "x_vars[Fashion accessories_36.51]"
    ],
    [
     "C45",
     "x_vars[Fashion accessories_36.85]"
    ],
    [
     "C46",
     "x_vars[Fashion accessories_37.15]"
    ],
    [
     "C47",
     "x_vars[Fashion accessories_37.55]"
    ],
    [
     "C48",
     "x_vars[Fashion accessories_37.95]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Fashion accessories_10.18]",
    "x_2": "x_vars[Fashion accessories_12.09]",
    "x_3": "x_vars[Fashion accessories_12.19]",
    "x_4": "x_vars[Fashion accessories_12.54]",
    "x_5": "x_vars[Fashion accessories_12.78]",
    "x_6": "x_vars[Fashion accessories_14.48]",
    "x_7": "x_vars[Fashion accessories_15.43]",
    "x_8": "x_vars[Fashion accessories_15.5]",
    "x_9": "x_vars[Fashion accessories_15.62]",
    "x_10": "x_vars[Fashion accessories_16.28]",
    "x_11": "x_vars[Fashion accessories_16.45]",
    "x_12": "x_vars[Fashion accessories_17.48]",
    "x_13": "x_vars[Fashion accessories_17.49]",
    "x_14": "x_vars[Fashion accessories_17.87]",
    "x_15": "x_vars[Fashion accessories_17.94]",
    "x_16": "x_vars[Fashion accessories_18.08]",
    "x_17": "x_vars[Fashion accessories_19.66]",
    "x_18": "x_vars[Fashion accessories_19.7]",
    "x_19": "x_vars[Fashion accessories_19.77]",
    "x_20": "x_vars[Fashion accessories_20.01]",
    "x_21": "x_vars[Fashion accessories_21.32]",
    "x_22": "x_vars[Fashion accessories_21.48]",
    "x_23": "x_vars[Fashion accessories_21.94]",
    "x_24": "x_vars[Fashion accessories_22.32]",
    "x_25": "x_vars[Fashion accessories_22.51]",
    "x_26": "x_vars[Fashion accessories_23.82]",
    "x_27": "x_vars[Fashion accessories_25.42]",
    "x_28": "x_vars[Fashion accessories_25.56]",
    "x_29": "x_vars[Fashion accessories_27.02]",
    "x_30": "x_vars[Fashion accessories_27.18]",
    "x_31": "x_vars[Fashion accessories_27.38]",
    "x_32": "x_vars[Fashion accessories_29.42]",
    "x_33": "x_vars[Fashion accessories_29.56]",
    "x_34": "x_vars[Fashion accessories_30.14]",
    "x_35": "x_vars[Fashion accessories_30.37]",
    "x_36": "x_vars[Fashion accessories_30.61]",
    "x_37": "x_vars[Fashion accessories_30.62]",
    "x_38": "x_vars[Fashion accessories_31.73]",
    "x_39": "x_vars[Fashion accessories_31.9]",
    "x_40": "x_vars[Fashion accessories_32.62]",
    "x_41": "x_vars[Fashion accessories_33.52]",
    "x_42": "x_vars[Fashion accessories_33.63]",
    "x_43": "x_vars[Fashion accessories_34.7]",
    "x_44": "x_vars[Fashion accessories_35.19]",
    "x_45": "x_vars[Fashion accessories_36.51]",
    "x_46": "x_vars[Fashion accessories_36.85]",
    "x_47": "x_vars[Fashion accessories_37.15]",
    "x_48": "x_vars[Fashion accessories_37.55]",
    "x_49": "x_vars[Fashion accessories_37.95]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-027": {
   "source_code_sha256": "12060e75d5ab526795799ac165e828e7432ff1349941d659851d9221a70aaa73",
   "original_solution_sha256": "6f02b80d9ed8484982472ade0bbe3da295492220b93f445a34743a4fdd6c3fd0",
   "bindings": [
    [
     "C0",
     "x_vars[Organic Fruits]"
    ],
    [
     "C1",
     "x_vars[Organic Staples]"
    ],
    [
     "C2",
     "x_vars[Organic Vegetables]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Organic Fruits]",
    "x_2": "x_vars[Organic Staples]",
    "x_3": "x_vars[Organic Vegetables]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-028": {
   "source_code_sha256": "83f681bdb5d79a9340083e62fa381c69e169c01eb62df10e46005d097d0a1411",
   "original_solution_sha256": "c342ff4f837337896a13f2e71a8dd1634f4e6432b9786821ebf1ef1a84c3f22f",
   "bindings": [
    [
     "C0",
     "x_vars[sku_I27]"
    ],
    [
     "C1",
     "x_vars[sku_I499]"
    ],
    [
     "C2",
     "x_vars[sku_I719]"
    ],
    [
     "C3",
     "x_vars[sku_T18]"
    ],
    [
     "C4",
     "x_vars[sku_T29]"
    ],
    [
     "C5",
     "x_vars[sku_T39]"
    ],
    [
     "C6",
     "x_vars[sku_T499]"
    ],
    [
     "C7",
     "x_vars[sku_T9]"
    ],
    [
     "C8",
     "x_vars[sku_3081]"
    ],
    [
     "C9",
     "x_vars[sku_339]"
    ],
    [
     "C10",
     "x_vars[sku_3799]"
    ],
    [
     "C11",
     "x_vars[sku_439]"
    ],
    [
     "C12",
     "x_vars[sku_539]"
    ],
    [
     "C13",
     "x_vars[sku_61399]"
    ],
    [
     "C14",
     "x_vars[sku_628]"
    ],
    [
     "C15",
     "x_vars[sku_708]"
    ],
    [
     "C16",
     "x_vars[sku_77]"
    ],
    [
     "C17",
     "x_vars[sku_79]"
    ],
    [
     "C18",
     "x_vars[sku_799]"
    ],
    [
     "C19",
     "x_vars[sku_8499]"
    ],
    [
     "C20",
     "x_vars[sku_89]"
    ],
    [
     "C21",
     "x_vars[sku_897]"
    ],
    [
     "C22",
     "x_vars[sku_9699]"
    ],
    [
     "C23",
     "x_vars[sku_bobo]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[sku_I27]",
    "x_2": "x_vars[sku_I499]",
    "x_3": "x_vars[sku_I719]",
    "x_4": "x_vars[sku_T18]",
    "x_5": "x_vars[sku_T29]",
    "x_6": "x_vars[sku_T39]",
    "x_7": "x_vars[sku_T499]",
    "x_8": "x_vars[sku_T9]",
    "x_9": "x_vars[sku_3081]",
    "x_10": "x_vars[sku_339]",
    "x_11": "x_vars[sku_3799]",
    "x_12": "x_vars[sku_439]",
    "x_13": "x_vars[sku_539]",
    "x_14": "x_vars[sku_61399]",
    "x_15": "x_vars[sku_628]",
    "x_16": "x_vars[sku_708]",
    "x_17": "x_vars[sku_77]",
    "x_18": "x_vars[sku_79]",
    "x_19": "x_vars[sku_799]",
    "x_20": "x_vars[sku_8499]",
    "x_21": "x_vars[sku_89]",
    "x_22": "x_vars[sku_897]",
    "x_23": "x_vars[sku_9699]",
    "x_24": "x_vars[sku_bobo]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-029": {
   "source_code_sha256": "aa8d4ff041c6c22d154c9223fe1b384253babc9220e4a2f485dc55904d0ff207",
   "original_solution_sha256": "fbf70fd20dab297bb8104fa3e6880c9ec863d39164c4fa923526f21164bb9ba5",
   "bindings": [
    [
     "C0",
     "x_vars[FAUX FUR JEWEL SWEATER]"
    ],
    [
     "C1",
     "x_vars[FAUX LEATHER BOMBER JACKET]"
    ],
    [
     "C2",
     "x_vars[FAUX LEATHER BOXY FIT JACKET]"
    ],
    [
     "C3",
     "x_vars[FAUX LEATHER JACKET]"
    ],
    [
     "C4",
     "x_vars[FAUX LEATHER OVERSIZED JACKET LIMITED EDITION]"
    ],
    [
     "C5",
     "x_vars[FAUX LEATHER PUFFER JACKET]"
    ],
    [
     "C6",
     "x_vars[FAUX SHEARLING LINED SUEDE BOOTS]"
    ],
    [
     "C7",
     "x_vars[FAUX SHEARLING PLAID JACKET]"
    ],
    [
     "C8",
     "x_vars[FAUX SUEDE BOMBER JACKET]"
    ],
    [
     "C9",
     "x_vars[FAUX SUEDE JACKET]"
    ],
    [
     "C10",
     "x_vars[FAUX SUEDE OVERSHIRT]"
    ],
    [
     "C11",
     "x_vars[FAUX SUEDE PATCH JACKET]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[FAUX FUR JEWEL SWEATER]",
    "x_2": "x_vars[FAUX LEATHER BOMBER JACKET]",
    "x_3": "x_vars[FAUX LEATHER BOXY FIT JACKET]",
    "x_4": "x_vars[FAUX LEATHER JACKET]",
    "x_5": "x_vars[FAUX LEATHER OVERSIZED JACKET LIMITED EDITION]",
    "x_6": "x_vars[FAUX LEATHER PUFFER JACKET]",
    "x_7": "x_vars[FAUX SHEARLING LINED SUEDE BOOTS]",
    "x_8": "x_vars[FAUX SHEARLING PLAID JACKET]",
    "x_9": "x_vars[FAUX SUEDE BOMBER JACKET]",
    "x_10": "x_vars[FAUX SUEDE JACKET]",
    "x_11": "x_vars[FAUX SUEDE OVERSHIRT]",
    "x_12": "x_vars[FAUX SUEDE PATCH JACKET]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-030": {
   "source_code_sha256": "c197e197b87d5f830a00043936d51c1f70fb4acac20bd49a8f7f8f83c43b2ac8",
   "original_solution_sha256": "e2de0211b4d242a6080cee5137f39314222fa17c6d1c709dfd4b32df81c15aeb",
   "bindings": [
    [
     "C0",
     "x_vars[1163]"
    ],
    [
     "C1",
     "x_vars[1501]"
    ],
    [
     "C2",
     "x_vars[1576]"
    ],
    [
     "C3",
     "x_vars[1793]"
    ],
    [
     "C4",
     "x_vars[2438]"
    ],
    [
     "C5",
     "x_vars[4297]"
    ],
    [
     "C6",
     "x_vars[4806]"
    ],
    [
     "C7",
     "x_vars[4942]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[1163]",
    "x_2": "x_vars[1501]",
    "x_3": "x_vars[1576]",
    "x_4": "x_vars[1793]",
    "x_5": "x_vars[2438]",
    "x_6": "x_vars[4297]",
    "x_7": "x_vars[4806]",
    "x_8": "x_vars[4942]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-031": {
   "source_code_sha256": "9e76d9780ad0a43bea4239cf0584d0fd4b30bc4c8e0d08fa009cb0745867f8ac",
   "original_solution_sha256": "93b28149b40f21c890f3bde2c27821a82abcd65216ecf90b8e40455845517b81",
   "bindings": [
    [
     "C0",
     "x_vars[Butter_Amul]"
    ],
    [
     "C1",
     "x_vars[Butter_Mother Dairy]"
    ],
    [
     "C2",
     "x_vars[Butter_Parag Milk Foods]"
    ],
    [
     "C3",
     "x_vars[Butter_Warana]"
    ],
    [
     "C4",
     "x_vars[Buttermilk_Amul]"
    ],
    [
     "C5",
     "x_vars[Buttermilk_Mother Dairy]"
    ],
    [
     "C6",
     "x_vars[Buttermilk_Raj]"
    ],
    [
     "C7",
     "x_vars[Buttermilk_Sudha]"
    ],
    [
     "C8",
     "x_vars[Cheese_Amul]"
    ],
    [
     "C9",
     "x_vars[Cheese_Britannia Industries]"
    ],
    [
     "C10",
     "x_vars[Cheese_Dynamix Dairies]"
    ],
    [
     "C11",
     "x_vars[Cheese_Passion Cheese]"
    ],
    [
     "C12",
     "x_vars[Curd_Amul]"
    ],
    [
     "C13",
     "x_vars[Curd_Mother Dairy]"
    ],
    [
     "C14",
     "x_vars[Curd_Raj]"
    ],
    [
     "C15",
     "x_vars[Curd_Sudha]"
    ],
    [
     "C16",
     "x_vars[Ghee_Amul]"
    ],
    [
     "C17",
     "x_vars[Ghee_Mother Dairy]"
    ],
    [
     "C18",
     "x_vars[Ghee_Raj]"
    ],
    [
     "C19",
     "x_vars[Ghee_Sudha]"
    ],
    [
     "C20",
     "x_vars[Ice Cream_Amul]"
    ],
    [
     "C21",
     "x_vars[Ice Cream_Dodla Dairy]"
    ],
    [
     "C22",
     "x_vars[Ice Cream_Mother Dairy]"
    ],
    [
     "C23",
     "x_vars[Ice Cream_Palle2patnam]"
    ],
    [
     "C24",
     "x_vars[Lassi_Amul]"
    ],
    [
     "C25",
     "x_vars[Lassi_Mother Dairy]"
    ],
    [
     "C26",
     "x_vars[Lassi_Raj]"
    ],
    [
     "C27",
     "x_vars[Lassi_Sudha]"
    ],
    [
     "C28",
     "x_vars[Milk_Amul]"
    ],
    [
     "C29",
     "x_vars[Milk_Mother Dairy]"
    ],
    [
     "C30",
     "x_vars[Milk_Raj]"
    ],
    [
     "C31",
     "x_vars[Milk_Sudha]"
    ],
    [
     "C32",
     "x_vars[Paneer_Amul]"
    ],
    [
     "C33",
     "x_vars[Paneer_Mother Dairy]"
    ],
    [
     "C34",
     "x_vars[Paneer_Raj]"
    ],
    [
     "C35",
     "x_vars[Paneer_Sudha]"
    ],
    [
     "C36",
     "x_vars[Yogurt_Amul]"
    ],
    [
     "C37",
     "x_vars[Yogurt_Dodla Dairy]"
    ],
    [
     "C38",
     "x_vars[Yogurt_Mother Dairy]"
    ],
    [
     "C39",
     "x_vars[Yogurt_Palle2patnam]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Butter_Amul]",
    "x_2": "x_vars[Butter_Mother Dairy]",
    "x_3": "x_vars[Butter_Parag Milk Foods]",
    "x_4": "x_vars[Butter_Warana]",
    "x_5": "x_vars[Buttermilk_Amul]",
    "x_6": "x_vars[Buttermilk_Mother Dairy]",
    "x_7": "x_vars[Buttermilk_Raj]",
    "x_8": "x_vars[Buttermilk_Sudha]",
    "x_9": "x_vars[Cheese_Amul]",
    "x_10": "x_vars[Cheese_Britannia Industries]",
    "x_11": "x_vars[Cheese_Dynamix Dairies]",
    "x_12": "x_vars[Cheese_Passion Cheese]",
    "x_13": "x_vars[Curd_Amul]",
    "x_14": "x_vars[Curd_Mother Dairy]",
    "x_15": "x_vars[Curd_Raj]",
    "x_16": "x_vars[Curd_Sudha]",
    "x_17": "x_vars[Ghee_Amul]",
    "x_18": "x_vars[Ghee_Mother Dairy]",
    "x_19": "x_vars[Ghee_Raj]",
    "x_20": "x_vars[Ghee_Sudha]",
    "x_21": "x_vars[Ice Cream_Amul]",
    "x_22": "x_vars[Ice Cream_Dodla Dairy]",
    "x_23": "x_vars[Ice Cream_Mother Dairy]",
    "x_24": "x_vars[Ice Cream_Palle2patnam]",
    "x_25": "x_vars[Lassi_Amul]",
    "x_26": "x_vars[Lassi_Mother Dairy]",
    "x_27": "x_vars[Lassi_Raj]",
    "x_28": "x_vars[Lassi_Sudha]",
    "x_29": "x_vars[Milk_Amul]",
    "x_30": "x_vars[Milk_Mother Dairy]",
    "x_31": "x_vars[Milk_Raj]",
    "x_32": "x_vars[Milk_Sudha]",
    "x_33": "x_vars[Paneer_Amul]",
    "x_34": "x_vars[Paneer_Mother Dairy]",
    "x_35": "x_vars[Paneer_Raj]",
    "x_36": "x_vars[Paneer_Sudha]",
    "x_37": "x_vars[Yogurt_Amul]",
    "x_38": "x_vars[Yogurt_Dodla Dairy]",
    "x_39": "x_vars[Yogurt_Mother Dairy]",
    "x_40": "x_vars[Yogurt_Palle2patnam]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-032": {
   "source_code_sha256": "d7cb29419a8e0b39c1ddc7310046e1a3cc8c7d64fde5320c53eadf69a084cc43",
   "original_solution_sha256": "1b784364c94bf18cad5b7749fd9105fc0236256f7fe15effa75af73e97f94e3d",
   "bindings": [
    [
     "C0",
     "x_vars[Books_15.15]"
    ],
    [
     "C1",
     "x_vars[Books_30.3]"
    ],
    [
     "C2",
     "x_vars[Books_45.45]"
    ],
    [
     "C3",
     "x_vars[Books_60.6]"
    ],
    [
     "C4",
     "x_vars[Books_75.75]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Books_15.15]",
    "x_2": "x_vars[Books_30.3]",
    "x_3": "x_vars[Books_45.45]",
    "x_4": "x_vars[Books_60.6]",
    "x_5": "x_vars[Books_75.75]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-033": {
   "source_code_sha256": "55614195f29ad3ed6fab5fd3be4af5a572728fb29a3d3f00dc177fc7e0c9f977",
   "original_solution_sha256": "2821aefe9bfe2db49216fd701a4a17406bc1586d54ed2c33e130cc524de64cfa",
   "bindings": [
    [
     "C0",
     "x_vars[Baby Food_255.28]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[Baby Food_255.28]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-034": {
   "source_code_sha256": "73f053b4657f4aaf81e3b816cc7734189bbc5b61abd3be18c18daad516f3b715",
   "original_solution_sha256": "145176984d504aea75da64f4e7b42ab4811fa4fe2ddc58e0ad23b914c9fff8f9",
   "bindings": [
    [
     "C0",
     "x_vars[12 MACARON]"
    ],
    [
     "C1",
     "x_vars[ARMORICAIN]"
    ],
    [
     "C2",
     "x_vars[ARTICLE 295]"
    ],
    [
     "C3",
     "x_vars[BAGUETTE]"
    ],
    [
     "C4",
     "x_vars[BAGUETTE APERO]"
    ],
    [
     "C5",
     "x_vars[BAGUETTE GRAINE]"
    ],
    [
     "C6",
     "x_vars[BANETTE]"
    ],
    [
     "C7",
     "x_vars[BANETTINE]"
    ],
    [
     "C8",
     "x_vars[BOISSON 33CL]"
    ],
    [
     "C9",
     "x_vars[BOTTEREAU]"
    ],
    [
     "C10",
     "x_vars[BOULE 200G]"
    ],
    [
     "C11",
     "x_vars[BOULE 400G]"
    ],
    [
     "C12",
     "x_vars[BOULE POLKA]"
    ],
    [
     "C13",
     "x_vars[BRIOCHE]"
    ],
    [
     "C14",
     "x_vars[BRIOCHE DE NOEL]"
    ],
    [
     "C15",
     "x_vars[BRIOCHETTE]"
    ],
    [
     "C16",
     "x_vars[BROWNIES]"
    ],
    [
     "C17",
     "x_vars[BUCHE 4PERS]"
    ],
    [
     "C18",
     "x_vars[BUCHE 6PERS]"
    ],
    [
     "C19",
     "x_vars[BUCHE 8PERS]"
    ],
    [
     "C20",
     "x_vars[CAFE OU EAU]"
    ],
    [
     "C21",
     "x_vars[CAKE]"
    ],
    [
     "C22",
     "x_vars[CAMPAGNE]"
    ],
    [
     "C23",
     "x_vars[CARAMEL NOIX]"
    ],
    [
     "C24",
     "x_vars[CEREAL BAGUETTE]"
    ],
    [
     "C25",
     "x_vars[CHAUSSON AUX POMMES]"
    ],
    [
     "C26",
     "x_vars[CHOCOLAT]"
    ],
    [
     "C27",
     "x_vars[CHOU CHANTILLY]"
    ],
    [
     "C28",
     "x_vars[COMPLET]"
    ],
    [
     "C29",
     "x_vars[COOKIE]"
    ],
    [
     "C30",
     "x_vars[COUPE]"
    ],
    [
     "C31",
     "x_vars[CROISSANT]"
    ],
    [
     "C32",
     "x_vars[CROISSANT AMANDES]"
    ],
    [
     "C33",
     "x_vars[CRUMBLE]"
    ],
    [
     "C34",
     "x_vars[CRUMBLECARAMEL OU PISTAE]"
    ],
    [
     "C35",
     "x_vars[DELICETROPICAL]"
    ],
    [
     "C36",
     "x_vars[DEMI BAGUETTE]"
    ],
    [
     "C37",
     "x_vars[DEMI PAIN]"
    ],
    [
     "C38",
     "x_vars[DIVERS BOISSONS]"
    ],
    [
     "C39",
     "x_vars[DIVERS BOULANGERIE]"
    ],
    [
     "C40",
     "x_vars[DIVERS CONFISERIE]"
    ],
    [
     "C41",
     "x_vars[DIVERS PATISSERIE]"
    ],
    [
     "C42",
     "x_vars[DIVERS SANDWICHS]"
    ],
    [
     "C43",
     "x_vars[DIVERS VIENNOISERIE]"
    ],
    [
     "C44",
     "x_vars[DOUCEUR D HIVER]"
    ],
    [
     "C45",
     "x_vars[ECLAIR]"
    ],
    [
     "C46",
     "x_vars[ECLAIR FRAISE PISTACHE]"
    ],
    [
     "C47",
     "x_vars[ENTREMETS]"
    ],
    [
     "C48",
     "x_vars[FICELLE]"
    ],
    [
     "C49",
     "x_vars[FINANCIER]"
    ],
    [
     "C50",
     "x_vars[FINANCIER X5]"
    ],
    [
     "C51",
     "x_vars[FLAN]"
    ],
    [
     "C52",
     "x_vars[FLAN ABRICOT]"
    ],
    [
     "C53",
     "x_vars[FONDANT CHOCOLAT]"
    ],
    [
     "C54",
     "x_vars[FORMULE PATE]"
    ],
    [
     "C55",
     "x_vars[FORMULE PLAT PREPARE]"
    ],
    [
     "C56",
     "x_vars[FORMULE SANDWICH]"
    ],
    [
     "C57",
     "x_vars[FRAISIER]"
    ],
    [
     "C58",
     "x_vars[FRAMBOISIER]"
    ],
    [
     "C59",
     "x_vars[GACHE]"
    ],
    [
     "C60",
     "x_vars[GAL FRANGIPANE 4P]"
    ],
    [
     "C61",
     "x_vars[GAL FRANGIPANE 6P]"
    ],
    [
     "C62",
     "x_vars[GAL POIRE CHOCO 4P]"
    ],
    [
     "C63",
     "x_vars[GAL POIRE CHOCO 6P]"
    ],
    [
     "C64",
     "x_vars[GAL POMME 4P]"
    ],
    [
     "C65",
     "x_vars[GAL POMME 6P]"
    ],
    [
     "C66",
     "x_vars[GALETTE 8 PERS]"
    ],
    [
     "C67",
     "x_vars[GD FAR BRETON]"
    ],
    [
     "C68",
     "x_vars[GD KOUIGN AMANN]"
    ],
    [
     "C69",
     "x_vars[GD NANTAIS]"
    ],
    [
     "C70",
     "x_vars[GD PLATEAU SALE]"
    ],
    [
     "C71",
     "x_vars[GRAND FAR BRETON]"
    ],
    [
     "C72",
     "x_vars[GRANDE SUCETTE]"
    ],
    [
     "C73",
     "x_vars[GUERANDAIS]"
    ],
    [
     "C74",
     "x_vars[KOUIGN AMANN]"
    ],
    [
     "C75",
     "x_vars[MACARON]"
    ],
    [
     "C76",
     "x_vars[MERINGUE]"
    ],
    [
     "C77",
     "x_vars[MILLES FEUILLES]"
    ],
    [
     "C78",
     "x_vars[MOISSON]"
    ],
    [
     "C79",
     "x_vars[NANTAIS]"
    ],
    [
     "C80",
     "x_vars[NID DE POULE]"
    ],
    [
     "C81",
     "x_vars[NOIX JAPONAISE]"
    ],
    [
     "C82",
     "x_vars[PAILLE]"
    ],
    [
     "C83",
     "x_vars[PAIN]"
    ],
    [
     "C84",
     "x_vars[PAIN AU CHOCOLAT]"
    ],
    [
     "C85",
     "x_vars[PAIN AUX RAISINS]"
    ],
    [
     "C86",
     "x_vars[PAIN BANETTE]"
    ],
    [
     "C87",
     "x_vars[PAIN CHOCO AMANDES]"
    ],
    [
     "C88",
     "x_vars[PAIN DE MIE]"
    ],
    [
     "C89",
     "x_vars[PAIN GRAINES]"
    ],
    [
     "C90",
     "x_vars[PAIN NOIR]"
    ],
    [
     "C91",
     "x_vars[PAIN S/SEL]"
    ],
    [
     "C92",
     "x_vars[PAIN SUISSE PEPITO]"
    ],
    [
     "C93",
     "x_vars[PALET BRETON]"
    ],
    [
     "C94",
     "x_vars[PALMIER]"
    ],
    [
     "C95",
     "x_vars[PARIS BREST]"
    ],
    [
     "C96",
     "x_vars[PATES]"
    ],
    [
     "C97",
     "x_vars[PLAQUE TARTE 25P]"
    ],
    [
     "C98",
     "x_vars[PLAT]"
    ],
    [
     "C99",
     "x_vars[PLAT 6.50E]"
    ],
    [
     "C100",
     "x_vars[PLAT 7.00]"
    ],
    [
     "C101",
     "x_vars[PLAT 7.60E]"
    ],
    [
     "C102",
     "x_vars[PLAT 8.30E]"
    ],
    [
     "C103",
     "x_vars[PLATPREPARE5,50]"
    ],
    [
     "C104",
     "x_vars[PLATPREPARE6,00]"
    ],
    [
     "C105",
     "x_vars[PLATPREPARE6,50]"
    ],
    [
     "C106",
     "x_vars[PLATPREPARE7,00]"
    ],
    [
     "C107",
     "x_vars[PT NANTAIS]"
    ],
    [
     "C108",
     "x_vars[PT PLATEAU SALE]"
    ],
    [
     "C109",
     "x_vars[QUIM BREAD]"
    ],
    [
     "C110",
     "x_vars[REDUCTION SUCREES 12]"
    ],
    [
     "C111",
     "x_vars[REDUCTION SUCREES 24]"
    ],
    [
     "C112",
     "x_vars[RELIGIEUSE]"
    ],
    [
     "C113",
     "x_vars[ROYAL]"
    ],
    [
     "C114",
     "x_vars[ROYAL 4P]"
    ],
    [
     "C115",
     "x_vars[ROYAL 6P]"
    ],
    [
     "C116",
     "x_vars[SABLE F  P]"
    ],
    [
     "C117",
     "x_vars[SACHET DE CROUTON]"
    ],
    [
     "C118",
     "x_vars[SACHET DE VIENNOISERIE]"
    ],
    [
     "C119",
     "x_vars[SACHET VIENNOISERIE]"
    ],
    [
     "C120",
     "x_vars[SAND JB]"
    ],
    [
     "C121",
     "x_vars[SAND JB EMMENTAL]"
    ],
    [
     "C122",
     "x_vars[SANDWICH COMPLET]"
    ],
    [
     "C123",
     "x_vars[SAVARIN]"
    ],
    [
     "C124",
     "x_vars[SEIGLE]"
    ],
    [
     "C125",
     "x_vars[SPECIAL BREAD]"
    ],
    [
     "C126",
     "x_vars[SPECIAL BREAD KG]"
    ],
    [
     "C127",
     "x_vars[ST HONORE]"
    ],
    [
     "C128",
     "x_vars[SUCETTE]"
    ],
    [
     "C129",
     "x_vars[TARTE FINE]"
    ],
    [
     "C130",
     "x_vars[TARTE FRAISE 4PER]"
    ],
    [
     "C131",
     "x_vars[TARTE FRAISE 6P]"
    ],
    [
     "C132",
     "x_vars[TARTE FRUITS 4P]"
    ],
    [
     "C133",
     "x_vars[TARTE FRUITS 6P]"
    ],
    [
     "C134",
     "x_vars[TARTELETTE]"
    ],
    [
     "C135",
     "x_vars[TARTELETTE CHOC]"
    ],
    [
     "C136",
     "x_vars[TARTELETTE COCKTAIL]"
    ],
    [
     "C137",
     "x_vars[TARTELETTE FRAISE]"
    ],
    [
     "C138",
     "x_vars[THE]"
    ],
    [
     "C139",
     "x_vars[TRADITIONAL BAGUETTE]"
    ],
    [
     "C140",
     "x_vars[TRAITEUR]"
    ],
    [
     "C141",
     "x_vars[TRIANGLES]"
    ],
    [
     "C142",
     "x_vars[TROIS CHOCOLAT]"
    ],
    [
     "C143",
     "x_vars[TROPEZIENNE]"
    ],
    [
     "C144",
     "x_vars[TROPEZIENNE FRAMBOISE]"
    ],
    [
     "C145",
     "x_vars[TULIPE]"
    ],
    [
     "C146",
     "x_vars[VIENNOISE]"
    ],
    [
     "C147",
     "x_vars[VIK BREAD]"
    ]
   ],
   "primary_mapping": {
    "x_1": "x_vars[12 MACARON]",
    "x_2": "x_vars[ARMORICAIN]",
    "x_3": "x_vars[ARTICLE 295]",
    "x_4": "x_vars[BAGUETTE]",
    "x_5": "x_vars[BAGUETTE APERO]",
    "x_6": "x_vars[BAGUETTE GRAINE]",
    "x_7": "x_vars[BANETTE]",
    "x_8": "x_vars[BANETTINE]",
    "x_9": "x_vars[BOISSON 33CL]",
    "x_10": "x_vars[BOTTEREAU]",
    "x_11": "x_vars[BOULE 200G]",
    "x_12": "x_vars[BOULE 400G]",
    "x_13": "x_vars[BOULE POLKA]",
    "x_14": "x_vars[BRIOCHE]",
    "x_15": "x_vars[BRIOCHE DE NOEL]",
    "x_16": "x_vars[BRIOCHETTE]",
    "x_17": "x_vars[BROWNIES]",
    "x_18": "x_vars[BUCHE 4PERS]",
    "x_19": "x_vars[BUCHE 6PERS]",
    "x_20": "x_vars[BUCHE 8PERS]",
    "x_21": "x_vars[CAFE OU EAU]",
    "x_22": "x_vars[CAKE]",
    "x_23": "x_vars[CAMPAGNE]",
    "x_24": "x_vars[CARAMEL NOIX]",
    "x_25": "x_vars[CEREAL BAGUETTE]",
    "x_26": "x_vars[CHAUSSON AUX POMMES]",
    "x_27": "x_vars[CHOCOLAT]",
    "x_28": "x_vars[CHOU CHANTILLY]",
    "x_29": "x_vars[COMPLET]",
    "x_30": "x_vars[COOKIE]",
    "x_31": "x_vars[COUPE]",
    "x_32": "x_vars[CROISSANT]",
    "x_33": "x_vars[CROISSANT AMANDES]",
    "x_34": "x_vars[CRUMBLE]",
    "x_35": "x_vars[CRUMBLECARAMEL OU PISTAE]",
    "x_36": "x_vars[DELICETROPICAL]",
    "x_37": "x_vars[DEMI BAGUETTE]",
    "x_38": "x_vars[DEMI PAIN]",
    "x_39": "x_vars[DIVERS BOISSONS]",
    "x_40": "x_vars[DIVERS BOULANGERIE]",
    "x_41": "x_vars[DIVERS CONFISERIE]",
    "x_42": "x_vars[DIVERS PATISSERIE]",
    "x_43": "x_vars[DIVERS SANDWICHS]",
    "x_44": "x_vars[DIVERS VIENNOISERIE]",
    "x_45": "x_vars[DOUCEUR D HIVER]",
    "x_46": "x_vars[ECLAIR]",
    "x_47": "x_vars[ECLAIR FRAISE PISTACHE]",
    "x_48": "x_vars[ENTREMETS]",
    "x_49": "x_vars[FICELLE]",
    "x_50": "x_vars[FINANCIER]",
    "x_51": "x_vars[FINANCIER X5]",
    "x_52": "x_vars[FLAN]",
    "x_53": "x_vars[FLAN ABRICOT]",
    "x_54": "x_vars[FONDANT CHOCOLAT]",
    "x_55": "x_vars[FORMULE PATE]",
    "x_56": "x_vars[FORMULE PLAT PREPARE]",
    "x_57": "x_vars[FORMULE SANDWICH]",
    "x_58": "x_vars[FRAISIER]",
    "x_59": "x_vars[FRAMBOISIER]",
    "x_60": "x_vars[GACHE]",
    "x_61": "x_vars[GAL FRANGIPANE 4P]",
    "x_62": "x_vars[GAL FRANGIPANE 6P]",
    "x_63": "x_vars[GAL POIRE CHOCO 4P]",
    "x_64": "x_vars[GAL POIRE CHOCO 6P]",
    "x_65": "x_vars[GAL POMME 4P]",
    "x_66": "x_vars[GAL POMME 6P]",
    "x_67": "x_vars[GALETTE 8 PERS]",
    "x_68": "x_vars[GD FAR BRETON]",
    "x_69": "x_vars[GD KOUIGN AMANN]",
    "x_70": "x_vars[GD NANTAIS]",
    "x_71": "x_vars[GD PLATEAU SALE]",
    "x_72": "x_vars[GRAND FAR BRETON]",
    "x_73": "x_vars[GRANDE SUCETTE]",
    "x_74": "x_vars[GUERANDAIS]",
    "x_75": "x_vars[KOUIGN AMANN]",
    "x_76": "x_vars[MACARON]",
    "x_77": "x_vars[MERINGUE]",
    "x_78": "x_vars[MILLES FEUILLES]",
    "x_79": "x_vars[MOISSON]",
    "x_80": "x_vars[NANTAIS]",
    "x_81": "x_vars[NID DE POULE]",
    "x_82": "x_vars[NOIX JAPONAISE]",
    "x_83": "x_vars[PAILLE]",
    "x_84": "x_vars[PAIN]",
    "x_85": "x_vars[PAIN AU CHOCOLAT]",
    "x_86": "x_vars[PAIN AUX RAISINS]",
    "x_87": "x_vars[PAIN BANETTE]",
    "x_88": "x_vars[PAIN CHOCO AMANDES]",
    "x_89": "x_vars[PAIN DE MIE]",
    "x_90": "x_vars[PAIN GRAINES]",
    "x_91": "x_vars[PAIN NOIR]",
    "x_92": "x_vars[PAIN S/SEL]",
    "x_93": "x_vars[PAIN SUISSE PEPITO]",
    "x_94": "x_vars[PALET BRETON]",
    "x_95": "x_vars[PALMIER]",
    "x_96": "x_vars[PARIS BREST]",
    "x_97": "x_vars[PATES]",
    "x_98": "x_vars[PLAQUE TARTE 25P]",
    "x_99": "x_vars[PLAT]",
    "x_100": "x_vars[PLAT 6.50E]",
    "x_101": "x_vars[PLAT 7.00]",
    "x_102": "x_vars[PLAT 7.60E]",
    "x_103": "x_vars[PLAT 8.30E]",
    "x_104": "x_vars[PLATPREPARE5,50]",
    "x_105": "x_vars[PLATPREPARE6,00]",
    "x_106": "x_vars[PLATPREPARE6,50]",
    "x_107": "x_vars[PLATPREPARE7,00]",
    "x_108": "x_vars[PT NANTAIS]",
    "x_109": "x_vars[PT PLATEAU SALE]",
    "x_110": "x_vars[QUIM BREAD]",
    "x_111": "x_vars[REDUCTION SUCREES 12]",
    "x_112": "x_vars[REDUCTION SUCREES 24]",
    "x_113": "x_vars[RELIGIEUSE]",
    "x_114": "x_vars[ROYAL]",
    "x_115": "x_vars[ROYAL 4P]",
    "x_116": "x_vars[ROYAL 6P]",
    "x_117": "x_vars[SABLE F  P]",
    "x_118": "x_vars[SACHET DE CROUTON]",
    "x_119": "x_vars[SACHET DE VIENNOISERIE]",
    "x_120": "x_vars[SACHET VIENNOISERIE]",
    "x_121": "x_vars[SAND JB]",
    "x_122": "x_vars[SAND JB EMMENTAL]",
    "x_123": "x_vars[SANDWICH COMPLET]",
    "x_124": "x_vars[SAVARIN]",
    "x_125": "x_vars[SEIGLE]",
    "x_126": "x_vars[SPECIAL BREAD]",
    "x_127": "x_vars[SPECIAL BREAD KG]",
    "x_128": "x_vars[ST HONORE]",
    "x_129": "x_vars[SUCETTE]",
    "x_130": "x_vars[TARTE FINE]",
    "x_131": "x_vars[TARTE FRAISE 4PER]",
    "x_132": "x_vars[TARTE FRAISE 6P]",
    "x_133": "x_vars[TARTE FRUITS 4P]",
    "x_134": "x_vars[TARTE FRUITS 6P]",
    "x_135": "x_vars[TARTELETTE]",
    "x_136": "x_vars[TARTELETTE CHOC]",
    "x_137": "x_vars[TARTELETTE COCKTAIL]",
    "x_138": "x_vars[TARTELETTE FRAISE]",
    "x_139": "x_vars[THE]",
    "x_140": "x_vars[TRADITIONAL BAGUETTE]",
    "x_141": "x_vars[TRAITEUR]",
    "x_142": "x_vars[TRIANGLES]",
    "x_143": "x_vars[TROIS CHOCOLAT]",
    "x_144": "x_vars[TROPEZIENNE]",
    "x_145": "x_vars[TROPEZIENNE FRAMBOISE]",
    "x_146": "x_vars[TULIPE]",
    "x_147": "x_vars[VIENNOISE]",
    "x_148": "x_vars[VIK BREAD]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-035": {
   "source_code_sha256": "aee4d55a7f3b49975dff8ca3d0eab60f722567ed4ce95f7aeb2df74c1d179084",
   "original_solution_sha256": "b077b5e16efbc30347743a558640144bcbe4a1446d5008e3425b98b9088bd16d",
   "bindings": [
    [
     "C0",
     "x_vars[Baguette]"
    ],
    [
     "C1",
     "x_vars[Croissant]"
    ],
    [
     "C2",
     "x_vars[Sourdough]"
    ],
    [
     "C3",
     "x_vars[Rye Bread]"
    ],
    [
     "C4",
     "x_vars[Brioche]"
    ],
    [
     "C5",
     "x_vars[Focaccia]"
    ],
    [
     "C6",
     "x_vars[Ciabatta]"
    ],
    [
     "C7",
     "x_vars[Pita]"
    ],
    [
     "C8",
     "x_vars[Bagel]"
    ],
    [
     "C9",
     "x_vars[English Muffin]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Baguette]",
    "x_1_0": "x_vars[Croissant]",
    "x_2_0": "x_vars[Sourdough]",
    "x_3_0": "x_vars[Rye Bread]",
    "x_4_0": "x_vars[Brioche]",
    "x_5_0": "x_vars[Focaccia]",
    "x_6_0": "x_vars[Ciabatta]",
    "x_7_0": "x_vars[Pita]",
    "x_8_0": "x_vars[Bagel]",
    "x_9_0": "x_vars[English Muffin]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-036": {
   "source_code_sha256": "a82ee0127ed68cbe14fdf12e23bea52c10b5621d4528e66736d54add6bd7ae1e",
   "original_solution_sha256": "c0bd95f1b2dba8a9d9976975ad9d8d84c31a63b4b6a521197d5f32239c9b2ef1",
   "bindings": [
    [
     "C0",
     "x_vars[Sedan]"
    ],
    [
     "C1",
     "x_vars[SUV]"
    ],
    [
     "C2",
     "x_vars[Truck]"
    ],
    [
     "C3",
     "x_vars[Convertible]"
    ],
    [
     "C4",
     "x_vars[Minivan]"
    ],
    [
     "C5",
     "x_vars[Coupe]"
    ],
    [
     "C6",
     "x_vars[Hatchback]"
    ],
    [
     "C7",
     "x_vars[Station Wagon]"
    ],
    [
     "C8",
     "x_vars[Electric Car]"
    ],
    [
     "C9",
     "x_vars[Hybrid Car]"
    ],
    [
     "C10",
     "x_vars[Luxury Sedan]"
    ],
    [
     "C11",
     "x_vars[Sports Car]"
    ],
    [
     "C12",
     "x_vars[Crossover]"
    ],
    [
     "C13",
     "x_vars[Diesel Truck]"
    ],
    [
     "C14",
     "x_vars[Compact SUV]"
    ],
    [
     "C15",
     "x_vars[Luxury SUV]"
    ],
    [
     "C16",
     "x_vars[Cargo Van]"
    ],
    [
     "C17",
     "x_vars[Pickup Truck]"
    ],
    [
     "C18",
     "x_vars[Roadster]"
    ],
    [
     "C19",
     "x_vars[Muscle Car]"
    ],
    [
     "C20",
     "x_vars[Off-road Vehicle]"
    ],
    [
     "C21",
     "x_vars[Camper Van]"
    ],
    [
     "C22",
     "x_vars[Compact Car]"
    ],
    [
     "C23",
     "x_vars[Motorcycle]"
    ],
    [
     "C24",
     "x_vars[Electric SUV]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Sedan]",
    "x_1_0": "x_vars[SUV]",
    "x_2_0": "x_vars[Truck]",
    "x_3_0": "x_vars[Convertible]",
    "x_4_0": "x_vars[Minivan]",
    "x_5_0": "x_vars[Coupe]",
    "x_6_0": "x_vars[Hatchback]",
    "x_7_0": "x_vars[Station Wagon]",
    "x_8_0": "x_vars[Electric Car]",
    "x_9_0": "x_vars[Hybrid Car]",
    "x_10_0": "x_vars[Luxury Sedan]",
    "x_11_0": "x_vars[Sports Car]",
    "x_12_0": "x_vars[Crossover]",
    "x_13_0": "x_vars[Diesel Truck]",
    "x_14_0": "x_vars[Compact SUV]",
    "x_15_0": "x_vars[Luxury SUV]",
    "x_16_0": "x_vars[Cargo Van]",
    "x_17_0": "x_vars[Pickup Truck]",
    "x_18_0": "x_vars[Roadster]",
    "x_19_0": "x_vars[Muscle Car]",
    "x_20_0": "x_vars[Off-road Vehicle]",
    "x_21_0": "x_vars[Camper Van]",
    "x_22_0": "x_vars[Compact Car]",
    "x_23_0": "x_vars[Motorcycle]",
    "x_24_0": "x_vars[Electric SUV]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-037": {
   "source_code_sha256": "112df4885ff519f58365a6d7de154009e6ec45746c5eeeb0c28f588749f6996b",
   "original_solution_sha256": "db6fd2358b41a29d98d26925c158d5d25e4e0bbed638ab55b89fc326eb5c9d58",
   "bindings": [
    [
     "C0",
     "x_vars[Sedan]"
    ],
    [
     "C1",
     "x_vars[SUV]"
    ],
    [
     "C2",
     "x_vars[Truck]"
    ],
    [
     "C3",
     "x_vars[Convertible]"
    ],
    [
     "C4",
     "x_vars[Minivan]"
    ],
    [
     "C5",
     "x_vars[Coupe]"
    ],
    [
     "C6",
     "x_vars[Hatchback]"
    ],
    [
     "C7",
     "x_vars[Station Wagon]"
    ],
    [
     "C8",
     "x_vars[Electric Car]"
    ],
    [
     "C9",
     "x_vars[Hybrid Car]"
    ],
    [
     "C10",
     "x_vars[Luxury Sedan]"
    ],
    [
     "C11",
     "x_vars[Sports Car]"
    ],
    [
     "C12",
     "x_vars[Crossover]"
    ],
    [
     "C13",
     "x_vars[Diesel Truck]"
    ],
    [
     "C14",
     "x_vars[Compact SUV]"
    ],
    [
     "C15",
     "x_vars[Luxury SUV]"
    ],
    [
     "C16",
     "x_vars[Cargo Van]"
    ],
    [
     "C17",
     "x_vars[Pickup Truck]"
    ],
    [
     "C18",
     "x_vars[Roadster]"
    ],
    [
     "C19",
     "x_vars[Muscle Car]"
    ],
    [
     "C20",
     "x_vars[Off-road Vehicle]"
    ],
    [
     "C21",
     "x_vars[Camper Van]"
    ],
    [
     "C22",
     "x_vars[Compact Car]"
    ],
    [
     "C23",
     "x_vars[Motorcycle]"
    ],
    [
     "C24",
     "x_vars[Electric SUV]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Sedan]",
    "x_1_0": "x_vars[SUV]",
    "x_2_0": "x_vars[Truck]",
    "x_3_0": "x_vars[Convertible]",
    "x_4_0": "x_vars[Minivan]",
    "x_5_0": "x_vars[Coupe]",
    "x_6_0": "x_vars[Hatchback]",
    "x_7_0": "x_vars[Station Wagon]",
    "x_8_0": "x_vars[Electric Car]",
    "x_9_0": "x_vars[Hybrid Car]",
    "x_10_0": "x_vars[Luxury Sedan]",
    "x_11_0": "x_vars[Sports Car]",
    "x_12_0": "x_vars[Crossover]",
    "x_13_0": "x_vars[Diesel Truck]",
    "x_14_0": "x_vars[Compact SUV]",
    "x_15_0": "x_vars[Luxury SUV]",
    "x_16_0": "x_vars[Cargo Van]",
    "x_17_0": "x_vars[Pickup Truck]",
    "x_18_0": "x_vars[Roadster]",
    "x_19_0": "x_vars[Muscle Car]",
    "x_20_0": "x_vars[Off-road Vehicle]",
    "x_21_0": "x_vars[Camper Van]",
    "x_22_0": "x_vars[Compact Car]",
    "x_23_0": "x_vars[Motorcycle]",
    "x_24_0": "x_vars[Electric SUV]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-038": {
   "source_code_sha256": "7c13d10148469ccc42da39330f2c35c5260fcffff4894a4d520b47f3a3da2e55",
   "original_solution_sha256": "d91e2401f20bd532a42c6b47d15b2d22a6d2483cc2165324083a670901940969",
   "bindings": [
    [
     "C0",
     "x_vars[1]"
    ],
    [
     "C1",
     "x_vars[2]"
    ],
    [
     "C2",
     "x_vars[3]"
    ],
    [
     "C3",
     "x_vars[4]"
    ],
    [
     "C4",
     "x_vars[5]"
    ],
    [
     "C5",
     "x_vars[6]"
    ],
    [
     "C6",
     "x_vars[7]"
    ],
    [
     "C7",
     "x_vars[8]"
    ],
    [
     "C8",
     "x_vars[9]"
    ],
    [
     "C9",
     "x_vars[10]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1]",
    "x_1_0": "x_vars[2]",
    "x_2_0": "x_vars[3]",
    "x_3_0": "x_vars[4]",
    "x_4_0": "x_vars[5]",
    "x_5_0": "x_vars[6]",
    "x_6_0": "x_vars[7]",
    "x_7_0": "x_vars[8]",
    "x_8_0": "x_vars[9]",
    "x_9_0": "x_vars[10]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-039": {
   "source_code_sha256": "e92f9f1e92bd013597931ecd189946723d6d1526fa75a78b67f94d75c43283e1",
   "original_solution_sha256": "2c3b16c7c15bb291e2d1947e015a04a3ad7555fd579ee2bff5f046c27b39fc81",
   "bindings": [
    [
     "C0",
     "x_vars[Sedans,Warehouse 1]"
    ],
    [
     "C1",
     "x_vars[Sedans,Warehouse 2]"
    ],
    [
     "C2",
     "x_vars[Sedans,Warehouse 3]"
    ],
    [
     "C3",
     "x_vars[Sedans,Warehouse 4]"
    ],
    [
     "C4",
     "x_vars[Sedans,Warehouse 5]"
    ],
    [
     "C5",
     "x_vars[Sedans,Warehouse 6]"
    ],
    [
     "C6",
     "x_vars[Sedans,Warehouse 7]"
    ],
    [
     "C7",
     "x_vars[Sedans,Warehouse 8]"
    ],
    [
     "C8",
     "x_vars[Sedans,Warehouse 9]"
    ],
    [
     "C9",
     "x_vars[Sedans,Warehouse 10]"
    ],
    [
     "C10",
     "x_vars[SUVs,Warehouse 1]"
    ],
    [
     "C11",
     "x_vars[SUVs,Warehouse 2]"
    ],
    [
     "C12",
     "x_vars[SUVs,Warehouse 3]"
    ],
    [
     "C13",
     "x_vars[SUVs,Warehouse 4]"
    ],
    [
     "C14",
     "x_vars[SUVs,Warehouse 5]"
    ],
    [
     "C15",
     "x_vars[SUVs,Warehouse 6]"
    ],
    [
     "C16",
     "x_vars[SUVs,Warehouse 7]"
    ],
    [
     "C17",
     "x_vars[SUVs,Warehouse 8]"
    ],
    [
     "C18",
     "x_vars[SUVs,Warehouse 9]"
    ],
    [
     "C19",
     "x_vars[SUVs,Warehouse 10]"
    ],
    [
     "C20",
     "x_vars[Electric Vehicles,Warehouse 1]"
    ],
    [
     "C21",
     "x_vars[Electric Vehicles,Warehouse 2]"
    ],
    [
     "C22",
     "x_vars[Electric Vehicles,Warehouse 3]"
    ],
    [
     "C23",
     "x_vars[Electric Vehicles,Warehouse 4]"
    ],
    [
     "C24",
     "x_vars[Electric Vehicles,Warehouse 5]"
    ],
    [
     "C25",
     "x_vars[Electric Vehicles,Warehouse 6]"
    ],
    [
     "C26",
     "x_vars[Electric Vehicles,Warehouse 7]"
    ],
    [
     "C27",
     "x_vars[Electric Vehicles,Warehouse 8]"
    ],
    [
     "C28",
     "x_vars[Electric Vehicles,Warehouse 9]"
    ],
    [
     "C29",
     "x_vars[Electric Vehicles,Warehouse 10]"
    ],
    [
     "C30",
     "x_vars[Hybrid Vehicles,Warehouse 1]"
    ],
    [
     "C31",
     "x_vars[Hybrid Vehicles,Warehouse 2]"
    ],
    [
     "C32",
     "x_vars[Hybrid Vehicles,Warehouse 3]"
    ],
    [
     "C33",
     "x_vars[Hybrid Vehicles,Warehouse 4]"
    ],
    [
     "C34",
     "x_vars[Hybrid Vehicles,Warehouse 5]"
    ],
    [
     "C35",
     "x_vars[Hybrid Vehicles,Warehouse 6]"
    ],
    [
     "C36",
     "x_vars[Hybrid Vehicles,Warehouse 7]"
    ],
    [
     "C37",
     "x_vars[Hybrid Vehicles,Warehouse 8]"
    ],
    [
     "C38",
     "x_vars[Hybrid Vehicles,Warehouse 9]"
    ],
    [
     "C39",
     "x_vars[Hybrid Vehicles,Warehouse 10]"
    ],
    [
     "C40",
     "x_vars[Trucks,Warehouse 1]"
    ],
    [
     "C41",
     "x_vars[Trucks,Warehouse 2]"
    ],
    [
     "C42",
     "x_vars[Trucks,Warehouse 3]"
    ],
    [
     "C43",
     "x_vars[Trucks,Warehouse 4]"
    ],
    [
     "C44",
     "x_vars[Trucks,Warehouse 5]"
    ],
    [
     "C45",
     "x_vars[Trucks,Warehouse 6]"
    ],
    [
     "C46",
     "x_vars[Trucks,Warehouse 7]"
    ],
    [
     "C47",
     "x_vars[Trucks,Warehouse 8]"
    ],
    [
     "C48",
     "x_vars[Trucks,Warehouse 9]"
    ],
    [
     "C49",
     "x_vars[Trucks,Warehouse 10]"
    ],
    [
     "C50",
     "x_vars[Sports Cars,Warehouse 1]"
    ],
    [
     "C51",
     "x_vars[Sports Cars,Warehouse 2]"
    ],
    [
     "C52",
     "x_vars[Sports Cars,Warehouse 3]"
    ],
    [
     "C53",
     "x_vars[Sports Cars,Warehouse 4]"
    ],
    [
     "C54",
     "x_vars[Sports Cars,Warehouse 5]"
    ],
    [
     "C55",
     "x_vars[Sports Cars,Warehouse 6]"
    ],
    [
     "C56",
     "x_vars[Sports Cars,Warehouse 7]"
    ],
    [
     "C57",
     "x_vars[Sports Cars,Warehouse 8]"
    ],
    [
     "C58",
     "x_vars[Sports Cars,Warehouse 9]"
    ],
    [
     "C59",
     "x_vars[Sports Cars,Warehouse 10]"
    ],
    [
     "C60",
     "x_vars[Compact Cars,Warehouse 1]"
    ],
    [
     "C61",
     "x_vars[Compact Cars,Warehouse 2]"
    ],
    [
     "C62",
     "x_vars[Compact Cars,Warehouse 3]"
    ],
    [
     "C63",
     "x_vars[Compact Cars,Warehouse 4]"
    ],
    [
     "C64",
     "x_vars[Compact Cars,Warehouse 5]"
    ],
    [
     "C65",
     "x_vars[Compact Cars,Warehouse 6]"
    ],
    [
     "C66",
     "x_vars[Compact Cars,Warehouse 7]"
    ],
    [
     "C67",
     "x_vars[Compact Cars,Warehouse 8]"
    ],
    [
     "C68",
     "x_vars[Compact Cars,Warehouse 9]"
    ],
    [
     "C69",
     "x_vars[Compact Cars,Warehouse 10]"
    ],
    [
     "C70",
     "x_vars[Luxury Sedans,Warehouse 1]"
    ],
    [
     "C71",
     "x_vars[Luxury Sedans,Warehouse 2]"
    ],
    [
     "C72",
     "x_vars[Luxury Sedans,Warehouse 3]"
    ],
    [
     "C73",
     "x_vars[Luxury Sedans,Warehouse 4]"
    ],
    [
     "C74",
     "x_vars[Luxury Sedans,Warehouse 5]"
    ],
    [
     "C75",
     "x_vars[Luxury Sedans,Warehouse 6]"
    ],
    [
     "C76",
     "x_vars[Luxury Sedans,Warehouse 7]"
    ],
    [
     "C77",
     "x_vars[Luxury Sedans,Warehouse 8]"
    ],
    [
     "C78",
     "x_vars[Luxury Sedans,Warehouse 9]"
    ],
    [
     "C79",
     "x_vars[Luxury Sedans,Warehouse 10]"
    ],
    [
     "C80",
     "x_vars[Vans,Warehouse 1]"
    ],
    [
     "C81",
     "x_vars[Vans,Warehouse 2]"
    ],
    [
     "C82",
     "x_vars[Vans,Warehouse 3]"
    ],
    [
     "C83",
     "x_vars[Vans,Warehouse 4]"
    ],
    [
     "C84",
     "x_vars[Vans,Warehouse 5]"
    ],
    [
     "C85",
     "x_vars[Vans,Warehouse 6]"
    ],
    [
     "C86",
     "x_vars[Vans,Warehouse 7]"
    ],
    [
     "C87",
     "x_vars[Vans,Warehouse 8]"
    ],
    [
     "C88",
     "x_vars[Vans,Warehouse 9]"
    ],
    [
     "C89",
     "x_vars[Vans,Warehouse 10]"
    ],
    [
     "C90",
     "x_vars[Pickup Trucks,Warehouse 1]"
    ],
    [
     "C91",
     "x_vars[Pickup Trucks,Warehouse 2]"
    ],
    [
     "C92",
     "x_vars[Pickup Trucks,Warehouse 3]"
    ],
    [
     "C93",
     "x_vars[Pickup Trucks,Warehouse 4]"
    ],
    [
     "C94",
     "x_vars[Pickup Trucks,Warehouse 5]"
    ],
    [
     "C95",
     "x_vars[Pickup Trucks,Warehouse 6]"
    ],
    [
     "C96",
     "x_vars[Pickup Trucks,Warehouse 7]"
    ],
    [
     "C97",
     "x_vars[Pickup Trucks,Warehouse 8]"
    ],
    [
     "C98",
     "x_vars[Pickup Trucks,Warehouse 9]"
    ],
    [
     "C99",
     "x_vars[Pickup Trucks,Warehouse 10]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Sedans,Warehouse 1]",
    "x_0_1": "x_vars[Sedans,Warehouse 2]",
    "x_0_2": "x_vars[Sedans,Warehouse 3]",
    "x_0_3": "x_vars[Sedans,Warehouse 4]",
    "x_0_4": "x_vars[Sedans,Warehouse 5]",
    "x_0_5": "x_vars[Sedans,Warehouse 6]",
    "x_0_6": "x_vars[Sedans,Warehouse 7]",
    "x_0_7": "x_vars[Sedans,Warehouse 8]",
    "x_0_8": "x_vars[Sedans,Warehouse 9]",
    "x_0_9": "x_vars[Sedans,Warehouse 10]",
    "x_1_0": "x_vars[SUVs,Warehouse 1]",
    "x_1_1": "x_vars[SUVs,Warehouse 2]",
    "x_1_2": "x_vars[SUVs,Warehouse 3]",
    "x_1_3": "x_vars[SUVs,Warehouse 4]",
    "x_1_4": "x_vars[SUVs,Warehouse 5]",
    "x_1_5": "x_vars[SUVs,Warehouse 6]",
    "x_1_6": "x_vars[SUVs,Warehouse 7]",
    "x_1_7": "x_vars[SUVs,Warehouse 8]",
    "x_1_8": "x_vars[SUVs,Warehouse 9]",
    "x_1_9": "x_vars[SUVs,Warehouse 10]",
    "x_2_0": "x_vars[Electric Vehicles,Warehouse 1]",
    "x_2_1": "x_vars[Electric Vehicles,Warehouse 2]",
    "x_2_2": "x_vars[Electric Vehicles,Warehouse 3]",
    "x_2_3": "x_vars[Electric Vehicles,Warehouse 4]",
    "x_2_4": "x_vars[Electric Vehicles,Warehouse 5]",
    "x_2_5": "x_vars[Electric Vehicles,Warehouse 6]",
    "x_2_6": "x_vars[Electric Vehicles,Warehouse 7]",
    "x_2_7": "x_vars[Electric Vehicles,Warehouse 8]",
    "x_2_8": "x_vars[Electric Vehicles,Warehouse 9]",
    "x_2_9": "x_vars[Electric Vehicles,Warehouse 10]",
    "x_3_0": "x_vars[Hybrid Vehicles,Warehouse 1]",
    "x_3_1": "x_vars[Hybrid Vehicles,Warehouse 2]",
    "x_3_2": "x_vars[Hybrid Vehicles,Warehouse 3]",
    "x_3_3": "x_vars[Hybrid Vehicles,Warehouse 4]",
    "x_3_4": "x_vars[Hybrid Vehicles,Warehouse 5]",
    "x_3_5": "x_vars[Hybrid Vehicles,Warehouse 6]",
    "x_3_6": "x_vars[Hybrid Vehicles,Warehouse 7]",
    "x_3_7": "x_vars[Hybrid Vehicles,Warehouse 8]",
    "x_3_8": "x_vars[Hybrid Vehicles,Warehouse 9]",
    "x_3_9": "x_vars[Hybrid Vehicles,Warehouse 10]",
    "x_4_0": "x_vars[Trucks,Warehouse 1]",
    "x_4_1": "x_vars[Trucks,Warehouse 2]",
    "x_4_2": "x_vars[Trucks,Warehouse 3]",
    "x_4_3": "x_vars[Trucks,Warehouse 4]",
    "x_4_4": "x_vars[Trucks,Warehouse 5]",
    "x_4_5": "x_vars[Trucks,Warehouse 6]",
    "x_4_6": "x_vars[Trucks,Warehouse 7]",
    "x_4_7": "x_vars[Trucks,Warehouse 8]",
    "x_4_8": "x_vars[Trucks,Warehouse 9]",
    "x_4_9": "x_vars[Trucks,Warehouse 10]",
    "x_5_0": "x_vars[Sports Cars,Warehouse 1]",
    "x_5_1": "x_vars[Sports Cars,Warehouse 2]",
    "x_5_2": "x_vars[Sports Cars,Warehouse 3]",
    "x_5_3": "x_vars[Sports Cars,Warehouse 4]",
    "x_5_4": "x_vars[Sports Cars,Warehouse 5]",
    "x_5_5": "x_vars[Sports Cars,Warehouse 6]",
    "x_5_6": "x_vars[Sports Cars,Warehouse 7]",
    "x_5_7": "x_vars[Sports Cars,Warehouse 8]",
    "x_5_8": "x_vars[Sports Cars,Warehouse 9]",
    "x_5_9": "x_vars[Sports Cars,Warehouse 10]",
    "x_6_0": "x_vars[Compact Cars,Warehouse 1]",
    "x_6_1": "x_vars[Compact Cars,Warehouse 2]",
    "x_6_2": "x_vars[Compact Cars,Warehouse 3]",
    "x_6_3": "x_vars[Compact Cars,Warehouse 4]",
    "x_6_4": "x_vars[Compact Cars,Warehouse 5]",
    "x_6_5": "x_vars[Compact Cars,Warehouse 6]",
    "x_6_6": "x_vars[Compact Cars,Warehouse 7]",
    "x_6_7": "x_vars[Compact Cars,Warehouse 8]",
    "x_6_8": "x_vars[Compact Cars,Warehouse 9]",
    "x_6_9": "x_vars[Compact Cars,Warehouse 10]",
    "x_7_0": "x_vars[Luxury Sedans,Warehouse 1]",
    "x_7_1": "x_vars[Luxury Sedans,Warehouse 2]",
    "x_7_2": "x_vars[Luxury Sedans,Warehouse 3]",
    "x_7_3": "x_vars[Luxury Sedans,Warehouse 4]",
    "x_7_4": "x_vars[Luxury Sedans,Warehouse 5]",
    "x_7_5": "x_vars[Luxury Sedans,Warehouse 6]",
    "x_7_6": "x_vars[Luxury Sedans,Warehouse 7]",
    "x_7_7": "x_vars[Luxury Sedans,Warehouse 8]",
    "x_7_8": "x_vars[Luxury Sedans,Warehouse 9]",
    "x_7_9": "x_vars[Luxury Sedans,Warehouse 10]",
    "x_8_0": "x_vars[Vans,Warehouse 1]",
    "x_8_1": "x_vars[Vans,Warehouse 2]",
    "x_8_2": "x_vars[Vans,Warehouse 3]",
    "x_8_3": "x_vars[Vans,Warehouse 4]",
    "x_8_4": "x_vars[Vans,Warehouse 5]",
    "x_8_5": "x_vars[Vans,Warehouse 6]",
    "x_8_6": "x_vars[Vans,Warehouse 7]",
    "x_8_7": "x_vars[Vans,Warehouse 8]",
    "x_8_8": "x_vars[Vans,Warehouse 9]",
    "x_8_9": "x_vars[Vans,Warehouse 10]",
    "x_9_0": "x_vars[Pickup Trucks,Warehouse 1]",
    "x_9_1": "x_vars[Pickup Trucks,Warehouse 2]",
    "x_9_2": "x_vars[Pickup Trucks,Warehouse 3]",
    "x_9_3": "x_vars[Pickup Trucks,Warehouse 4]",
    "x_9_4": "x_vars[Pickup Trucks,Warehouse 5]",
    "x_9_5": "x_vars[Pickup Trucks,Warehouse 6]",
    "x_9_6": "x_vars[Pickup Trucks,Warehouse 7]",
    "x_9_7": "x_vars[Pickup Trucks,Warehouse 8]",
    "x_9_8": "x_vars[Pickup Trucks,Warehouse 9]",
    "x_9_9": "x_vars[Pickup Trucks,Warehouse 10]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-040": {
   "source_code_sha256": "c5158d1378d0e7f096f0e260988db03b81d5dddc0e24d7af935b9be37668b956",
   "original_solution_sha256": "8e2d8c41e214b874c850caf35709ac7f67d1362246b12fd810ddc360af914459",
   "bindings": [
    [
     "C0",
     "x_vars[Queens]"
    ],
    [
     "C1",
     "x_vars[Brooklyn]"
    ],
    [
     "C2",
     "x_vars[Manhattan]"
    ],
    [
     "C3",
     "x_vars[Bronx]"
    ],
    [
     "C4",
     "x_vars[Staten Island]"
    ],
    [
     "C5",
     "x_vars[Harlem]"
    ],
    [
     "C6",
     "x_vars[Upper East Side]"
    ],
    [
     "C7",
     "x_vars[Lower Manhattan]"
    ],
    [
     "C8",
     "x_vars[Midtown]"
    ],
    [
     "C9",
     "x_vars[Long Island City]"
    ],
    [
     "C10",
     "x_vars[Williamsburg]"
    ],
    [
     "C11",
     "x_vars[Bushwick]"
    ],
    [
     "C12",
     "x_vars[Flatbush]"
    ],
    [
     "C13",
     "x_vars[Greenpoint]"
    ],
    [
     "C14",
     "x_vars[Park Slope]"
    ],
    [
     "C15",
     "x_vars[Astoria]"
    ],
    [
     "C16",
     "x_vars[Jackson Heights]"
    ],
    [
     "C17",
     "x_vars[Flushing]"
    ],
    [
     "C18",
     "x_vars[Sunnyside]"
    ],
    [
     "C19",
     "x_vars[Ditmars]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Queens]",
    "x_1_0": "x_vars[Brooklyn]",
    "x_2_0": "x_vars[Manhattan]",
    "x_3_0": "x_vars[Bronx]",
    "x_4_0": "x_vars[Staten Island]",
    "x_5_0": "x_vars[Harlem]",
    "x_6_0": "x_vars[Upper East Side]",
    "x_7_0": "x_vars[Lower Manhattan]",
    "x_8_0": "x_vars[Midtown]",
    "x_9_0": "x_vars[Long Island City]",
    "x_10_0": "x_vars[Williamsburg]",
    "x_11_0": "x_vars[Bushwick]",
    "x_12_0": "x_vars[Flatbush]",
    "x_13_0": "x_vars[Greenpoint]",
    "x_14_0": "x_vars[Park Slope]",
    "x_15_0": "x_vars[Astoria]",
    "x_16_0": "x_vars[Jackson Heights]",
    "x_17_0": "x_vars[Flushing]",
    "x_18_0": "x_vars[Sunnyside]",
    "x_19_0": "x_vars[Ditmars]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-041": {
   "source_code_sha256": "9db3cd4d0597b2e36d2a2b8126a28d64151a48b635c03cd366a770d49077eb38",
   "original_solution_sha256": "a4bdf58012c54ed1aaf2710641de7d7d7ee530d669cb76c72c6dcc3685f5bd85",
   "bindings": [
    [
     "C0",
     "x_vars[Queens]"
    ],
    [
     "C1",
     "x_vars[Brooklyn]"
    ],
    [
     "C2",
     "x_vars[Manhattan]"
    ],
    [
     "C3",
     "x_vars[Bronx]"
    ],
    [
     "C4",
     "x_vars[Staten Island]"
    ],
    [
     "C5",
     "x_vars[Harlem]"
    ],
    [
     "C6",
     "x_vars[Upper East Side]"
    ],
    [
     "C7",
     "x_vars[Lower Manhattan]"
    ],
    [
     "C8",
     "x_vars[Midtown]"
    ],
    [
     "C9",
     "x_vars[Long Island City]"
    ],
    [
     "C10",
     "x_vars[Williamsburg]"
    ],
    [
     "C11",
     "x_vars[Bushwick]"
    ],
    [
     "C12",
     "x_vars[Flatbush]"
    ],
    [
     "C13",
     "x_vars[Greenpoint]"
    ],
    [
     "C14",
     "x_vars[Park Slope]"
    ],
    [
     "C15",
     "x_vars[Astoria]"
    ],
    [
     "C16",
     "x_vars[Jackson Heights]"
    ],
    [
     "C17",
     "x_vars[Flushing]"
    ],
    [
     "C18",
     "x_vars[Sunnyside]"
    ],
    [
     "C19",
     "x_vars[Ditmars]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Queens]",
    "x_1_0": "x_vars[Brooklyn]",
    "x_2_0": "x_vars[Manhattan]",
    "x_3_0": "x_vars[Bronx]",
    "x_4_0": "x_vars[Staten Island]",
    "x_5_0": "x_vars[Harlem]",
    "x_6_0": "x_vars[Upper East Side]",
    "x_7_0": "x_vars[Lower Manhattan]",
    "x_8_0": "x_vars[Midtown]",
    "x_9_0": "x_vars[Long Island City]",
    "x_10_0": "x_vars[Williamsburg]",
    "x_11_0": "x_vars[Bushwick]",
    "x_12_0": "x_vars[Flatbush]",
    "x_13_0": "x_vars[Greenpoint]",
    "x_14_0": "x_vars[Park Slope]",
    "x_15_0": "x_vars[Astoria]",
    "x_16_0": "x_vars[Jackson Heights]",
    "x_17_0": "x_vars[Flushing]",
    "x_18_0": "x_vars[Sunnyside]",
    "x_19_0": "x_vars[Ditmars]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-042": {
   "source_code_sha256": "b4c6064022ccb6dffb9bd270deefffc92e3def4d2d2a0204fad44d0bc5ad7285",
   "original_solution_sha256": "e1410690e75d6bd907ab2e270b0c40b33929ecd14066b4710cc1031e9e6ecbc6",
   "bindings": [
    [
     "C0",
     "x_vars[NSAIDs]"
    ],
    [
     "C1",
     "x_vars[Antirheumatic Drugs]"
    ],
    [
     "C2",
     "x_vars[Acetic Acid Derivatives]"
    ],
    [
     "C3",
     "x_vars[Antibiotics]"
    ],
    [
     "C4",
     "x_vars[Antiviral Drugs]"
    ],
    [
     "C5",
     "x_vars[Antifungal Agents]"
    ],
    [
     "C6",
     "x_vars[Antidepressants]"
    ],
    [
     "C7",
     "x_vars[Antipsychotics]"
    ],
    [
     "C8",
     "x_vars[Antihistamines]"
    ],
    [
     "C9",
     "x_vars[Corticosteroids]"
    ],
    [
     "C10",
     "x_vars[Beta Blockers]"
    ],
    [
     "C11",
     "x_vars[Calcium Channel Blockers]"
    ],
    [
     "C12",
     "x_vars[ACE Inhibitors]"
    ],
    [
     "C13",
     "x_vars[Angiotensin II Receptor Blockers]"
    ],
    [
     "C14",
     "x_vars[Diuretics]"
    ],
    [
     "C15",
     "x_vars[Statins]"
    ],
    [
     "C16",
     "x_vars[Insulin]"
    ],
    [
     "C17",
     "x_vars[Anticoagulants]"
    ],
    [
     "C18",
     "x_vars[Antiepileptic Drugs]"
    ],
    [
     "C19",
     "x_vars[Antiemetics]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[NSAIDs]",
    "x_1_0": "x_vars[Antirheumatic Drugs]",
    "x_2_0": "x_vars[Acetic Acid Derivatives]",
    "x_3_0": "x_vars[Antibiotics]",
    "x_4_0": "x_vars[Antiviral Drugs]",
    "x_5_0": "x_vars[Antifungal Agents]",
    "x_6_0": "x_vars[Antidepressants]",
    "x_7_0": "x_vars[Antipsychotics]",
    "x_8_0": "x_vars[Antihistamines]",
    "x_9_0": "x_vars[Corticosteroids]",
    "x_10_0": "x_vars[Beta Blockers]",
    "x_11_0": "x_vars[Calcium Channel Blockers]",
    "x_12_0": "x_vars[ACE Inhibitors]",
    "x_13_0": "x_vars[Angiotensin II Receptor Blockers]",
    "x_14_0": "x_vars[Diuretics]",
    "x_15_0": "x_vars[Statins]",
    "x_16_0": "x_vars[Insulin]",
    "x_17_0": "x_vars[Anticoagulants]",
    "x_18_0": "x_vars[Antiepileptic Drugs]",
    "x_19_0": "x_vars[Antiemetics]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-043": {
   "source_code_sha256": "3f7c887823006aadc575bb8bcd88f3b49c9591c23f489def73912dca599e5ff1",
   "original_solution_sha256": "8c88ac0a4b8930b744f571f5e2f1727cc706e816f62bc63416286f852c99225d",
   "bindings": [
    [
     "C0",
     "x_vars[NSAIDs]"
    ],
    [
     "C1",
     "x_vars[Antirheumatic Drugs]"
    ],
    [
     "C2",
     "x_vars[Acetic Acid Derivatives]"
    ],
    [
     "C3",
     "x_vars[Antibiotics]"
    ],
    [
     "C4",
     "x_vars[Antiviral Drugs]"
    ],
    [
     "C5",
     "x_vars[Antifungal Agents]"
    ],
    [
     "C6",
     "x_vars[Antidepressants]"
    ],
    [
     "C7",
     "x_vars[Antipsychotics]"
    ],
    [
     "C8",
     "x_vars[Antihistamines]"
    ],
    [
     "C9",
     "x_vars[Corticosteroids]"
    ],
    [
     "C10",
     "x_vars[Beta Blockers]"
    ],
    [
     "C11",
     "x_vars[Calcium Channel Blockers]"
    ],
    [
     "C12",
     "x_vars[ACE Inhibitors]"
    ],
    [
     "C13",
     "x_vars[Angiotensin II Receptor Blockers]"
    ],
    [
     "C14",
     "x_vars[Diuretics]"
    ],
    [
     "C15",
     "x_vars[Statins]"
    ],
    [
     "C16",
     "x_vars[Insulin]"
    ],
    [
     "C17",
     "x_vars[Anticoagulants]"
    ],
    [
     "C18",
     "x_vars[Antiepileptic Drugs]"
    ],
    [
     "C19",
     "x_vars[Antiemetics]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[NSAIDs]",
    "x_1_0": "x_vars[Antirheumatic Drugs]",
    "x_2_0": "x_vars[Acetic Acid Derivatives]",
    "x_3_0": "x_vars[Antibiotics]",
    "x_4_0": "x_vars[Antiviral Drugs]",
    "x_5_0": "x_vars[Antifungal Agents]",
    "x_6_0": "x_vars[Antidepressants]",
    "x_7_0": "x_vars[Antipsychotics]",
    "x_8_0": "x_vars[Antihistamines]",
    "x_9_0": "x_vars[Corticosteroids]",
    "x_10_0": "x_vars[Beta Blockers]",
    "x_11_0": "x_vars[Calcium Channel Blockers]",
    "x_12_0": "x_vars[ACE Inhibitors]",
    "x_13_0": "x_vars[Angiotensin II Receptor Blockers]",
    "x_14_0": "x_vars[Diuretics]",
    "x_15_0": "x_vars[Statins]",
    "x_16_0": "x_vars[Insulin]",
    "x_17_0": "x_vars[Anticoagulants]",
    "x_18_0": "x_vars[Antiepileptic Drugs]",
    "x_19_0": "x_vars[Antiemetics]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-044": {
   "source_code_sha256": "31e6743b9f2362456c36a284c1a18b0a2579940fb7a33fa924914c6adbac722a",
   "original_solution_sha256": "e6dd56492f5eda5827745e102076a2166e1d9b1d6f5fceca0c0297bac094bac9",
   "bindings": [
    [
     "C0",
     "x_vars[1,1]"
    ],
    [
     "C1",
     "x_vars[1,2]"
    ],
    [
     "C2",
     "x_vars[1,3]"
    ],
    [
     "C3",
     "x_vars[1,4]"
    ],
    [
     "C4",
     "x_vars[1,5]"
    ],
    [
     "C5",
     "x_vars[1,6]"
    ],
    [
     "C6",
     "x_vars[1,7]"
    ],
    [
     "C7",
     "x_vars[1,8]"
    ],
    [
     "C8",
     "x_vars[1,9]"
    ],
    [
     "C9",
     "x_vars[1,10]"
    ],
    [
     "C10",
     "x_vars[2,1]"
    ],
    [
     "C11",
     "x_vars[2,2]"
    ],
    [
     "C12",
     "x_vars[2,3]"
    ],
    [
     "C13",
     "x_vars[2,4]"
    ],
    [
     "C14",
     "x_vars[2,5]"
    ],
    [
     "C15",
     "x_vars[2,6]"
    ],
    [
     "C16",
     "x_vars[2,7]"
    ],
    [
     "C17",
     "x_vars[2,8]"
    ],
    [
     "C18",
     "x_vars[2,9]"
    ],
    [
     "C19",
     "x_vars[2,10]"
    ],
    [
     "C20",
     "x_vars[3,1]"
    ],
    [
     "C21",
     "x_vars[3,2]"
    ],
    [
     "C22",
     "x_vars[3,3]"
    ],
    [
     "C23",
     "x_vars[3,4]"
    ],
    [
     "C24",
     "x_vars[3,5]"
    ],
    [
     "C25",
     "x_vars[3,6]"
    ],
    [
     "C26",
     "x_vars[3,7]"
    ],
    [
     "C27",
     "x_vars[3,8]"
    ],
    [
     "C28",
     "x_vars[3,9]"
    ],
    [
     "C29",
     "x_vars[3,10]"
    ],
    [
     "C30",
     "x_vars[4,1]"
    ],
    [
     "C31",
     "x_vars[4,2]"
    ],
    [
     "C32",
     "x_vars[4,3]"
    ],
    [
     "C33",
     "x_vars[4,4]"
    ],
    [
     "C34",
     "x_vars[4,5]"
    ],
    [
     "C35",
     "x_vars[4,6]"
    ],
    [
     "C36",
     "x_vars[4,7]"
    ],
    [
     "C37",
     "x_vars[4,8]"
    ],
    [
     "C38",
     "x_vars[4,9]"
    ],
    [
     "C39",
     "x_vars[4,10]"
    ],
    [
     "C40",
     "x_vars[5,1]"
    ],
    [
     "C41",
     "x_vars[5,2]"
    ],
    [
     "C42",
     "x_vars[5,3]"
    ],
    [
     "C43",
     "x_vars[5,4]"
    ],
    [
     "C44",
     "x_vars[5,5]"
    ],
    [
     "C45",
     "x_vars[5,6]"
    ],
    [
     "C46",
     "x_vars[5,7]"
    ],
    [
     "C47",
     "x_vars[5,8]"
    ],
    [
     "C48",
     "x_vars[5,9]"
    ],
    [
     "C49",
     "x_vars[5,10]"
    ],
    [
     "C50",
     "x_vars[6,1]"
    ],
    [
     "C51",
     "x_vars[6,2]"
    ],
    [
     "C52",
     "x_vars[6,3]"
    ],
    [
     "C53",
     "x_vars[6,4]"
    ],
    [
     "C54",
     "x_vars[6,5]"
    ],
    [
     "C55",
     "x_vars[6,6]"
    ],
    [
     "C56",
     "x_vars[6,7]"
    ],
    [
     "C57",
     "x_vars[6,8]"
    ],
    [
     "C58",
     "x_vars[6,9]"
    ],
    [
     "C59",
     "x_vars[6,10]"
    ],
    [
     "C60",
     "x_vars[7,1]"
    ],
    [
     "C61",
     "x_vars[7,2]"
    ],
    [
     "C62",
     "x_vars[7,3]"
    ],
    [
     "C63",
     "x_vars[7,4]"
    ],
    [
     "C64",
     "x_vars[7,5]"
    ],
    [
     "C65",
     "x_vars[7,6]"
    ],
    [
     "C66",
     "x_vars[7,7]"
    ],
    [
     "C67",
     "x_vars[7,8]"
    ],
    [
     "C68",
     "x_vars[7,9]"
    ],
    [
     "C69",
     "x_vars[7,10]"
    ],
    [
     "C70",
     "x_vars[8,1]"
    ],
    [
     "C71",
     "x_vars[8,2]"
    ],
    [
     "C72",
     "x_vars[8,3]"
    ],
    [
     "C73",
     "x_vars[8,4]"
    ],
    [
     "C74",
     "x_vars[8,5]"
    ],
    [
     "C75",
     "x_vars[8,6]"
    ],
    [
     "C76",
     "x_vars[8,7]"
    ],
    [
     "C77",
     "x_vars[8,8]"
    ],
    [
     "C78",
     "x_vars[8,9]"
    ],
    [
     "C79",
     "x_vars[8,10]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,1]",
    "x_0_1": "x_vars[2,1]",
    "x_0_2": "x_vars[3,1]",
    "x_0_3": "x_vars[4,1]",
    "x_0_4": "x_vars[5,1]",
    "x_0_5": "x_vars[6,1]",
    "x_0_6": "x_vars[7,1]",
    "x_0_7": "x_vars[8,1]",
    "x_1_0": "x_vars[1,2]",
    "x_1_1": "x_vars[2,2]",
    "x_1_2": "x_vars[3,2]",
    "x_1_3": "x_vars[4,2]",
    "x_1_4": "x_vars[5,2]",
    "x_1_5": "x_vars[6,2]",
    "x_1_6": "x_vars[7,2]",
    "x_1_7": "x_vars[8,2]",
    "x_2_0": "x_vars[1,3]",
    "x_2_1": "x_vars[2,3]",
    "x_2_2": "x_vars[3,3]",
    "x_2_3": "x_vars[4,3]",
    "x_2_4": "x_vars[5,3]",
    "x_2_5": "x_vars[6,3]",
    "x_2_6": "x_vars[7,3]",
    "x_2_7": "x_vars[8,3]",
    "x_3_0": "x_vars[1,4]",
    "x_3_1": "x_vars[2,4]",
    "x_3_2": "x_vars[3,4]",
    "x_3_3": "x_vars[4,4]",
    "x_3_4": "x_vars[5,4]",
    "x_3_5": "x_vars[6,4]",
    "x_3_6": "x_vars[7,4]",
    "x_3_7": "x_vars[8,4]",
    "x_4_0": "x_vars[1,5]",
    "x_4_1": "x_vars[2,5]",
    "x_4_2": "x_vars[3,5]",
    "x_4_3": "x_vars[4,5]",
    "x_4_4": "x_vars[5,5]",
    "x_4_5": "x_vars[6,5]",
    "x_4_6": "x_vars[7,5]",
    "x_4_7": "x_vars[8,5]",
    "x_5_0": "x_vars[1,6]",
    "x_5_1": "x_vars[2,6]",
    "x_5_2": "x_vars[3,6]",
    "x_5_3": "x_vars[4,6]",
    "x_5_4": "x_vars[5,6]",
    "x_5_5": "x_vars[6,6]",
    "x_5_6": "x_vars[7,6]",
    "x_5_7": "x_vars[8,6]",
    "x_6_0": "x_vars[1,7]",
    "x_6_1": "x_vars[2,7]",
    "x_6_2": "x_vars[3,7]",
    "x_6_3": "x_vars[4,7]",
    "x_6_4": "x_vars[5,7]",
    "x_6_5": "x_vars[6,7]",
    "x_6_6": "x_vars[7,7]",
    "x_6_7": "x_vars[8,7]",
    "x_7_0": "x_vars[1,8]",
    "x_7_1": "x_vars[2,8]",
    "x_7_2": "x_vars[3,8]",
    "x_7_3": "x_vars[4,8]",
    "x_7_4": "x_vars[5,8]",
    "x_7_5": "x_vars[6,8]",
    "x_7_6": "x_vars[7,8]",
    "x_7_7": "x_vars[8,8]",
    "x_8_0": "x_vars[1,9]",
    "x_8_1": "x_vars[2,9]",
    "x_8_2": "x_vars[3,9]",
    "x_8_3": "x_vars[4,9]",
    "x_8_4": "x_vars[5,9]",
    "x_8_5": "x_vars[6,9]",
    "x_8_6": "x_vars[7,9]",
    "x_8_7": "x_vars[8,9]",
    "x_9_0": "x_vars[1,10]",
    "x_9_1": "x_vars[2,10]",
    "x_9_2": "x_vars[3,10]",
    "x_9_3": "x_vars[4,10]",
    "x_9_4": "x_vars[5,10]",
    "x_9_5": "x_vars[6,10]",
    "x_9_6": "x_vars[7,10]",
    "x_9_7": "x_vars[8,10]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-045": {
   "source_code_sha256": "3789a81a04013749d5c596929ed8b3f3be420ac9ca7f03e8b32b3c8fb69426d4",
   "original_solution_sha256": "66a82d3222d1f4c64117422a0b64c6440b3775c98c5e351a4c80ae4b4888420c",
   "bindings": [
    [
     "C0",
     "x_vars[Spinach]"
    ],
    [
     "C1",
     "x_vars[Shiitake Mushrooms]"
    ],
    [
     "C2",
     "x_vars[Apples]"
    ],
    [
     "C3",
     "x_vars[Carrots]"
    ],
    [
     "C4",
     "x_vars[Basil]"
    ],
    [
     "C5",
     "x_vars[Potatoes]"
    ],
    [
     "C6",
     "x_vars[Green Beans]"
    ],
    [
     "C7",
     "x_vars[Blueberries]"
    ],
    [
     "C8",
     "x_vars[Oranges]"
    ],
    [
     "C9",
     "x_vars[Watermelons]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Spinach]",
    "x_1_0": "x_vars[Shiitake Mushrooms]",
    "x_2_0": "x_vars[Apples]",
    "x_3_0": "x_vars[Carrots]",
    "x_4_0": "x_vars[Basil]",
    "x_5_0": "x_vars[Potatoes]",
    "x_6_0": "x_vars[Green Beans]",
    "x_7_0": "x_vars[Blueberries]",
    "x_8_0": "x_vars[Oranges]",
    "x_9_0": "x_vars[Watermelons]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-046": {
   "source_code_sha256": "ddf26f4e16c4b289cf100704d9dc88943b5bd9288b7f2a213727da281aea2204",
   "original_solution_sha256": "d920562de6283a4e2a31bc0cf85f74ac501f1b025af2b0aa47247e72425f650c",
   "bindings": [
    [
     "C0",
     "x_vars[Spinach]"
    ],
    [
     "C1",
     "x_vars[Shiitake Mushrooms]"
    ],
    [
     "C2",
     "x_vars[Apples]"
    ],
    [
     "C3",
     "x_vars[Carrots]"
    ],
    [
     "C4",
     "x_vars[Basil]"
    ],
    [
     "C5",
     "x_vars[Potatoes]"
    ],
    [
     "C6",
     "x_vars[Green Beans]"
    ],
    [
     "C7",
     "x_vars[Blueberries]"
    ],
    [
     "C8",
     "x_vars[Oranges]"
    ],
    [
     "C9",
     "x_vars[Watermelons]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Spinach]",
    "x_1_0": "x_vars[Shiitake Mushrooms]",
    "x_2_0": "x_vars[Apples]",
    "x_3_0": "x_vars[Carrots]",
    "x_4_0": "x_vars[Basil]",
    "x_5_0": "x_vars[Potatoes]",
    "x_6_0": "x_vars[Green Beans]",
    "x_7_0": "x_vars[Blueberries]",
    "x_8_0": "x_vars[Oranges]",
    "x_9_0": "x_vars[Watermelons]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-047": {
   "source_code_sha256": "b30435285256c9ef9b696168741c1094442daf7017ac23105a887978ee99467d",
   "original_solution_sha256": "4c6caf5504ac4b216a3bf536e68d9bd7090d83099c026319d8b816b8ef107d34",
   "bindings": [
    [
     "C0",
     "x_vars[1,Racing]"
    ],
    [
     "C1",
     "x_vars[1,Sports]"
    ],
    [
     "C2",
     "x_vars[1,Action]"
    ],
    [
     "C3",
     "x_vars[1,Adventure]"
    ],
    [
     "C4",
     "x_vars[1,RPG]"
    ],
    [
     "C5",
     "x_vars[1,Shooter]"
    ],
    [
     "C6",
     "x_vars[1,Strategy]"
    ],
    [
     "C7",
     "x_vars[1,Simulation]"
    ],
    [
     "C8",
     "x_vars[1,Puzzle]"
    ],
    [
     "C9",
     "x_vars[1,Fighting]"
    ],
    [
     "C10",
     "x_vars[1,Platformer]"
    ],
    [
     "C11",
     "x_vars[1,Survival]"
    ],
    [
     "C12",
     "x_vars[1,Horror]"
    ],
    [
     "C13",
     "x_vars[1,Sandbox]"
    ],
    [
     "C14",
     "x_vars[1,MMO]"
    ],
    [
     "C15",
     "x_vars[2,Racing]"
    ],
    [
     "C16",
     "x_vars[2,Sports]"
    ],
    [
     "C17",
     "x_vars[2,Action]"
    ],
    [
     "C18",
     "x_vars[2,Adventure]"
    ],
    [
     "C19",
     "x_vars[2,RPG]"
    ],
    [
     "C20",
     "x_vars[2,Shooter]"
    ],
    [
     "C21",
     "x_vars[2,Strategy]"
    ],
    [
     "C22",
     "x_vars[2,Simulation]"
    ],
    [
     "C23",
     "x_vars[2,Puzzle]"
    ],
    [
     "C24",
     "x_vars[2,Fighting]"
    ],
    [
     "C25",
     "x_vars[2,Platformer]"
    ],
    [
     "C26",
     "x_vars[2,Survival]"
    ],
    [
     "C27",
     "x_vars[2,Horror]"
    ],
    [
     "C28",
     "x_vars[2,Sandbox]"
    ],
    [
     "C29",
     "x_vars[2,MMO]"
    ],
    [
     "C30",
     "x_vars[3,Racing]"
    ],
    [
     "C31",
     "x_vars[3,Sports]"
    ],
    [
     "C32",
     "x_vars[3,Action]"
    ],
    [
     "C33",
     "x_vars[3,Adventure]"
    ],
    [
     "C34",
     "x_vars[3,RPG]"
    ],
    [
     "C35",
     "x_vars[3,Shooter]"
    ],
    [
     "C36",
     "x_vars[3,Strategy]"
    ],
    [
     "C37",
     "x_vars[3,Simulation]"
    ],
    [
     "C38",
     "x_vars[3,Puzzle]"
    ],
    [
     "C39",
     "x_vars[3,Fighting]"
    ],
    [
     "C40",
     "x_vars[3,Platformer]"
    ],
    [
     "C41",
     "x_vars[3,Survival]"
    ],
    [
     "C42",
     "x_vars[3,Horror]"
    ],
    [
     "C43",
     "x_vars[3,Sandbox]"
    ],
    [
     "C44",
     "x_vars[3,MMO]"
    ],
    [
     "C45",
     "x_vars[4,Racing]"
    ],
    [
     "C46",
     "x_vars[4,Sports]"
    ],
    [
     "C47",
     "x_vars[4,Action]"
    ],
    [
     "C48",
     "x_vars[4,Adventure]"
    ],
    [
     "C49",
     "x_vars[4,RPG]"
    ],
    [
     "C50",
     "x_vars[4,Shooter]"
    ],
    [
     "C51",
     "x_vars[4,Strategy]"
    ],
    [
     "C52",
     "x_vars[4,Simulation]"
    ],
    [
     "C53",
     "x_vars[4,Puzzle]"
    ],
    [
     "C54",
     "x_vars[4,Fighting]"
    ],
    [
     "C55",
     "x_vars[4,Platformer]"
    ],
    [
     "C56",
     "x_vars[4,Survival]"
    ],
    [
     "C57",
     "x_vars[4,Horror]"
    ],
    [
     "C58",
     "x_vars[4,Sandbox]"
    ],
    [
     "C59",
     "x_vars[4,MMO]"
    ],
    [
     "C60",
     "x_vars[5,Racing]"
    ],
    [
     "C61",
     "x_vars[5,Sports]"
    ],
    [
     "C62",
     "x_vars[5,Action]"
    ],
    [
     "C63",
     "x_vars[5,Adventure]"
    ],
    [
     "C64",
     "x_vars[5,RPG]"
    ],
    [
     "C65",
     "x_vars[5,Shooter]"
    ],
    [
     "C66",
     "x_vars[5,Strategy]"
    ],
    [
     "C67",
     "x_vars[5,Simulation]"
    ],
    [
     "C68",
     "x_vars[5,Puzzle]"
    ],
    [
     "C69",
     "x_vars[5,Fighting]"
    ],
    [
     "C70",
     "x_vars[5,Platformer]"
    ],
    [
     "C71",
     "x_vars[5,Survival]"
    ],
    [
     "C72",
     "x_vars[5,Horror]"
    ],
    [
     "C73",
     "x_vars[5,Sandbox]"
    ],
    [
     "C74",
     "x_vars[5,MMO]"
    ],
    [
     "C75",
     "x_vars[6,Racing]"
    ],
    [
     "C76",
     "x_vars[6,Sports]"
    ],
    [
     "C77",
     "x_vars[6,Action]"
    ],
    [
     "C78",
     "x_vars[6,Adventure]"
    ],
    [
     "C79",
     "x_vars[6,RPG]"
    ],
    [
     "C80",
     "x_vars[6,Shooter]"
    ],
    [
     "C81",
     "x_vars[6,Strategy]"
    ],
    [
     "C82",
     "x_vars[6,Simulation]"
    ],
    [
     "C83",
     "x_vars[6,Puzzle]"
    ],
    [
     "C84",
     "x_vars[6,Fighting]"
    ],
    [
     "C85",
     "x_vars[6,Platformer]"
    ],
    [
     "C86",
     "x_vars[6,Survival]"
    ],
    [
     "C87",
     "x_vars[6,Horror]"
    ],
    [
     "C88",
     "x_vars[6,Sandbox]"
    ],
    [
     "C89",
     "x_vars[6,MMO]"
    ],
    [
     "C90",
     "x_vars[7,Racing]"
    ],
    [
     "C91",
     "x_vars[7,Sports]"
    ],
    [
     "C92",
     "x_vars[7,Action]"
    ],
    [
     "C93",
     "x_vars[7,Adventure]"
    ],
    [
     "C94",
     "x_vars[7,RPG]"
    ],
    [
     "C95",
     "x_vars[7,Shooter]"
    ],
    [
     "C96",
     "x_vars[7,Strategy]"
    ],
    [
     "C97",
     "x_vars[7,Simulation]"
    ],
    [
     "C98",
     "x_vars[7,Puzzle]"
    ],
    [
     "C99",
     "x_vars[7,Fighting]"
    ],
    [
     "C100",
     "x_vars[7,Platformer]"
    ],
    [
     "C101",
     "x_vars[7,Survival]"
    ],
    [
     "C102",
     "x_vars[7,Horror]"
    ],
    [
     "C103",
     "x_vars[7,Sandbox]"
    ],
    [
     "C104",
     "x_vars[7,MMO]"
    ],
    [
     "C105",
     "x_vars[8,Racing]"
    ],
    [
     "C106",
     "x_vars[8,Sports]"
    ],
    [
     "C107",
     "x_vars[8,Action]"
    ],
    [
     "C108",
     "x_vars[8,Adventure]"
    ],
    [
     "C109",
     "x_vars[8,RPG]"
    ],
    [
     "C110",
     "x_vars[8,Shooter]"
    ],
    [
     "C111",
     "x_vars[8,Strategy]"
    ],
    [
     "C112",
     "x_vars[8,Simulation]"
    ],
    [
     "C113",
     "x_vars[8,Puzzle]"
    ],
    [
     "C114",
     "x_vars[8,Fighting]"
    ],
    [
     "C115",
     "x_vars[8,Platformer]"
    ],
    [
     "C116",
     "x_vars[8,Survival]"
    ],
    [
     "C117",
     "x_vars[8,Horror]"
    ],
    [
     "C118",
     "x_vars[8,Sandbox]"
    ],
    [
     "C119",
     "x_vars[8,MMO]"
    ],
    [
     "C120",
     "x_vars[9,Racing]"
    ],
    [
     "C121",
     "x_vars[9,Sports]"
    ],
    [
     "C122",
     "x_vars[9,Action]"
    ],
    [
     "C123",
     "x_vars[9,Adventure]"
    ],
    [
     "C124",
     "x_vars[9,RPG]"
    ],
    [
     "C125",
     "x_vars[9,Shooter]"
    ],
    [
     "C126",
     "x_vars[9,Strategy]"
    ],
    [
     "C127",
     "x_vars[9,Simulation]"
    ],
    [
     "C128",
     "x_vars[9,Puzzle]"
    ],
    [
     "C129",
     "x_vars[9,Fighting]"
    ],
    [
     "C130",
     "x_vars[9,Platformer]"
    ],
    [
     "C131",
     "x_vars[9,Survival]"
    ],
    [
     "C132",
     "x_vars[9,Horror]"
    ],
    [
     "C133",
     "x_vars[9,Sandbox]"
    ],
    [
     "C134",
     "x_vars[9,MMO]"
    ],
    [
     "C135",
     "x_vars[10,Racing]"
    ],
    [
     "C136",
     "x_vars[10,Sports]"
    ],
    [
     "C137",
     "x_vars[10,Action]"
    ],
    [
     "C138",
     "x_vars[10,Adventure]"
    ],
    [
     "C139",
     "x_vars[10,RPG]"
    ],
    [
     "C140",
     "x_vars[10,Shooter]"
    ],
    [
     "C141",
     "x_vars[10,Strategy]"
    ],
    [
     "C142",
     "x_vars[10,Simulation]"
    ],
    [
     "C143",
     "x_vars[10,Puzzle]"
    ],
    [
     "C144",
     "x_vars[10,Fighting]"
    ],
    [
     "C145",
     "x_vars[10,Platformer]"
    ],
    [
     "C146",
     "x_vars[10,Survival]"
    ],
    [
     "C147",
     "x_vars[10,Horror]"
    ],
    [
     "C148",
     "x_vars[10,Sandbox]"
    ],
    [
     "C149",
     "x_vars[10,MMO]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Racing]",
    "x_0_1": "x_vars[2,Racing]",
    "x_0_2": "x_vars[3,Racing]",
    "x_0_3": "x_vars[4,Racing]",
    "x_0_4": "x_vars[5,Racing]",
    "x_0_5": "x_vars[6,Racing]",
    "x_0_6": "x_vars[7,Racing]",
    "x_0_7": "x_vars[8,Racing]",
    "x_0_8": "x_vars[9,Racing]",
    "x_0_9": "x_vars[10,Racing]",
    "x_1_0": "x_vars[1,Sports]",
    "x_1_1": "x_vars[2,Sports]",
    "x_1_2": "x_vars[3,Sports]",
    "x_1_3": "x_vars[4,Sports]",
    "x_1_4": "x_vars[5,Sports]",
    "x_1_5": "x_vars[6,Sports]",
    "x_1_6": "x_vars[7,Sports]",
    "x_1_7": "x_vars[8,Sports]",
    "x_1_8": "x_vars[9,Sports]",
    "x_1_9": "x_vars[10,Sports]",
    "x_2_0": "x_vars[1,Action]",
    "x_2_1": "x_vars[2,Action]",
    "x_2_2": "x_vars[3,Action]",
    "x_2_3": "x_vars[4,Action]",
    "x_2_4": "x_vars[5,Action]",
    "x_2_5": "x_vars[6,Action]",
    "x_2_6": "x_vars[7,Action]",
    "x_2_7": "x_vars[8,Action]",
    "x_2_8": "x_vars[9,Action]",
    "x_2_9": "x_vars[10,Action]",
    "x_3_0": "x_vars[1,Adventure]",
    "x_3_1": "x_vars[2,Adventure]",
    "x_3_2": "x_vars[3,Adventure]",
    "x_3_3": "x_vars[4,Adventure]",
    "x_3_4": "x_vars[5,Adventure]",
    "x_3_5": "x_vars[6,Adventure]",
    "x_3_6": "x_vars[7,Adventure]",
    "x_3_7": "x_vars[8,Adventure]",
    "x_3_8": "x_vars[9,Adventure]",
    "x_3_9": "x_vars[10,Adventure]",
    "x_4_0": "x_vars[1,RPG]",
    "x_4_1": "x_vars[2,RPG]",
    "x_4_2": "x_vars[3,RPG]",
    "x_4_3": "x_vars[4,RPG]",
    "x_4_4": "x_vars[5,RPG]",
    "x_4_5": "x_vars[6,RPG]",
    "x_4_6": "x_vars[7,RPG]",
    "x_4_7": "x_vars[8,RPG]",
    "x_4_8": "x_vars[9,RPG]",
    "x_4_9": "x_vars[10,RPG]",
    "x_5_0": "x_vars[1,Shooter]",
    "x_5_1": "x_vars[2,Shooter]",
    "x_5_2": "x_vars[3,Shooter]",
    "x_5_3": "x_vars[4,Shooter]",
    "x_5_4": "x_vars[5,Shooter]",
    "x_5_5": "x_vars[6,Shooter]",
    "x_5_6": "x_vars[7,Shooter]",
    "x_5_7": "x_vars[8,Shooter]",
    "x_5_8": "x_vars[9,Shooter]",
    "x_5_9": "x_vars[10,Shooter]",
    "x_6_0": "x_vars[1,Strategy]",
    "x_6_1": "x_vars[2,Strategy]",
    "x_6_2": "x_vars[3,Strategy]",
    "x_6_3": "x_vars[4,Strategy]",
    "x_6_4": "x_vars[5,Strategy]",
    "x_6_5": "x_vars[6,Strategy]",
    "x_6_6": "x_vars[7,Strategy]",
    "x_6_7": "x_vars[8,Strategy]",
    "x_6_8": "x_vars[9,Strategy]",
    "x_6_9": "x_vars[10,Strategy]",
    "x_7_0": "x_vars[1,Simulation]",
    "x_7_1": "x_vars[2,Simulation]",
    "x_7_2": "x_vars[3,Simulation]",
    "x_7_3": "x_vars[4,Simulation]",
    "x_7_4": "x_vars[5,Simulation]",
    "x_7_5": "x_vars[6,Simulation]",
    "x_7_6": "x_vars[7,Simulation]",
    "x_7_7": "x_vars[8,Simulation]",
    "x_7_8": "x_vars[9,Simulation]",
    "x_7_9": "x_vars[10,Simulation]",
    "x_8_0": "x_vars[1,Puzzle]",
    "x_8_1": "x_vars[2,Puzzle]",
    "x_8_2": "x_vars[3,Puzzle]",
    "x_8_3": "x_vars[4,Puzzle]",
    "x_8_4": "x_vars[5,Puzzle]",
    "x_8_5": "x_vars[6,Puzzle]",
    "x_8_6": "x_vars[7,Puzzle]",
    "x_8_7": "x_vars[8,Puzzle]",
    "x_8_8": "x_vars[9,Puzzle]",
    "x_8_9": "x_vars[10,Puzzle]",
    "x_9_0": "x_vars[1,Fighting]",
    "x_9_1": "x_vars[2,Fighting]",
    "x_9_2": "x_vars[3,Fighting]",
    "x_9_3": "x_vars[4,Fighting]",
    "x_9_4": "x_vars[5,Fighting]",
    "x_9_5": "x_vars[6,Fighting]",
    "x_9_6": "x_vars[7,Fighting]",
    "x_9_7": "x_vars[8,Fighting]",
    "x_9_8": "x_vars[9,Fighting]",
    "x_9_9": "x_vars[10,Fighting]",
    "x_10_0": "x_vars[1,Platformer]",
    "x_10_1": "x_vars[2,Platformer]",
    "x_10_2": "x_vars[3,Platformer]",
    "x_10_3": "x_vars[4,Platformer]",
    "x_10_4": "x_vars[5,Platformer]",
    "x_10_5": "x_vars[6,Platformer]",
    "x_10_6": "x_vars[7,Platformer]",
    "x_10_7": "x_vars[8,Platformer]",
    "x_10_8": "x_vars[9,Platformer]",
    "x_10_9": "x_vars[10,Platformer]",
    "x_11_0": "x_vars[1,Survival]",
    "x_11_1": "x_vars[2,Survival]",
    "x_11_2": "x_vars[3,Survival]",
    "x_11_3": "x_vars[4,Survival]",
    "x_11_4": "x_vars[5,Survival]",
    "x_11_5": "x_vars[6,Survival]",
    "x_11_6": "x_vars[7,Survival]",
    "x_11_7": "x_vars[8,Survival]",
    "x_11_8": "x_vars[9,Survival]",
    "x_11_9": "x_vars[10,Survival]",
    "x_12_0": "x_vars[1,Horror]",
    "x_12_1": "x_vars[2,Horror]",
    "x_12_2": "x_vars[3,Horror]",
    "x_12_3": "x_vars[4,Horror]",
    "x_12_4": "x_vars[5,Horror]",
    "x_12_5": "x_vars[6,Horror]",
    "x_12_6": "x_vars[7,Horror]",
    "x_12_7": "x_vars[8,Horror]",
    "x_12_8": "x_vars[9,Horror]",
    "x_12_9": "x_vars[10,Horror]",
    "x_13_0": "x_vars[1,Sandbox]",
    "x_13_1": "x_vars[2,Sandbox]",
    "x_13_2": "x_vars[3,Sandbox]",
    "x_13_3": "x_vars[4,Sandbox]",
    "x_13_4": "x_vars[5,Sandbox]",
    "x_13_5": "x_vars[6,Sandbox]",
    "x_13_6": "x_vars[7,Sandbox]",
    "x_13_7": "x_vars[8,Sandbox]",
    "x_13_8": "x_vars[9,Sandbox]",
    "x_13_9": "x_vars[10,Sandbox]",
    "x_14_0": "x_vars[1,MMO]",
    "x_14_1": "x_vars[2,MMO]",
    "x_14_2": "x_vars[3,MMO]",
    "x_14_3": "x_vars[4,MMO]",
    "x_14_4": "x_vars[5,MMO]",
    "x_14_5": "x_vars[6,MMO]",
    "x_14_6": "x_vars[7,MMO]",
    "x_14_7": "x_vars[8,MMO]",
    "x_14_8": "x_vars[9,MMO]",
    "x_14_9": "x_vars[10,MMO]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-048": {
   "source_code_sha256": "9fab04ca10893631e1f9c96dcb82ed4e9302150615b4e4f5a961549062ea28d4",
   "original_solution_sha256": "ce3b40ea1ba2404b7a327ea54b547984f6ec09b5d8edd15a1f03969efbb848f0",
   "bindings": [
    [
     "C0",
     "x_vars[1,Window Unit]"
    ],
    [
     "C1",
     "x_vars[1,Portable Unit]"
    ],
    [
     "C2",
     "x_vars[1,Split System]"
    ],
    [
     "C3",
     "x_vars[1,Ductless System]"
    ],
    [
     "C4",
     "x_vars[1,Central AC]"
    ],
    [
     "C5",
     "x_vars[1,Hybrid AC]"
    ],
    [
     "C6",
     "x_vars[1,Geothermal AC]"
    ],
    [
     "C7",
     "x_vars[1,Smart AC]"
    ],
    [
     "C8",
     "x_vars[1,Evaporative Cooler]"
    ],
    [
     "C9",
     "x_vars[1,Package Unit]"
    ],
    [
     "C10",
     "x_vars[2,Window Unit]"
    ],
    [
     "C11",
     "x_vars[2,Portable Unit]"
    ],
    [
     "C12",
     "x_vars[2,Split System]"
    ],
    [
     "C13",
     "x_vars[2,Ductless System]"
    ],
    [
     "C14",
     "x_vars[2,Central AC]"
    ],
    [
     "C15",
     "x_vars[2,Hybrid AC]"
    ],
    [
     "C16",
     "x_vars[2,Geothermal AC]"
    ],
    [
     "C17",
     "x_vars[2,Smart AC]"
    ],
    [
     "C18",
     "x_vars[2,Evaporative Cooler]"
    ],
    [
     "C19",
     "x_vars[2,Package Unit]"
    ],
    [
     "C20",
     "x_vars[3,Window Unit]"
    ],
    [
     "C21",
     "x_vars[3,Portable Unit]"
    ],
    [
     "C22",
     "x_vars[3,Split System]"
    ],
    [
     "C23",
     "x_vars[3,Ductless System]"
    ],
    [
     "C24",
     "x_vars[3,Central AC]"
    ],
    [
     "C25",
     "x_vars[3,Hybrid AC]"
    ],
    [
     "C26",
     "x_vars[3,Geothermal AC]"
    ],
    [
     "C27",
     "x_vars[3,Smart AC]"
    ],
    [
     "C28",
     "x_vars[3,Evaporative Cooler]"
    ],
    [
     "C29",
     "x_vars[3,Package Unit]"
    ],
    [
     "C30",
     "x_vars[4,Window Unit]"
    ],
    [
     "C31",
     "x_vars[4,Portable Unit]"
    ],
    [
     "C32",
     "x_vars[4,Split System]"
    ],
    [
     "C33",
     "x_vars[4,Ductless System]"
    ],
    [
     "C34",
     "x_vars[4,Central AC]"
    ],
    [
     "C35",
     "x_vars[4,Hybrid AC]"
    ],
    [
     "C36",
     "x_vars[4,Geothermal AC]"
    ],
    [
     "C37",
     "x_vars[4,Smart AC]"
    ],
    [
     "C38",
     "x_vars[4,Evaporative Cooler]"
    ],
    [
     "C39",
     "x_vars[4,Package Unit]"
    ],
    [
     "C40",
     "x_vars[5,Window Unit]"
    ],
    [
     "C41",
     "x_vars[5,Portable Unit]"
    ],
    [
     "C42",
     "x_vars[5,Split System]"
    ],
    [
     "C43",
     "x_vars[5,Ductless System]"
    ],
    [
     "C44",
     "x_vars[5,Central AC]"
    ],
    [
     "C45",
     "x_vars[5,Hybrid AC]"
    ],
    [
     "C46",
     "x_vars[5,Geothermal AC]"
    ],
    [
     "C47",
     "x_vars[5,Smart AC]"
    ],
    [
     "C48",
     "x_vars[5,Evaporative Cooler]"
    ],
    [
     "C49",
     "x_vars[5,Package Unit]"
    ],
    [
     "C50",
     "x_vars[6,Window Unit]"
    ],
    [
     "C51",
     "x_vars[6,Portable Unit]"
    ],
    [
     "C52",
     "x_vars[6,Split System]"
    ],
    [
     "C53",
     "x_vars[6,Ductless System]"
    ],
    [
     "C54",
     "x_vars[6,Central AC]"
    ],
    [
     "C55",
     "x_vars[6,Hybrid AC]"
    ],
    [
     "C56",
     "x_vars[6,Geothermal AC]"
    ],
    [
     "C57",
     "x_vars[6,Smart AC]"
    ],
    [
     "C58",
     "x_vars[6,Evaporative Cooler]"
    ],
    [
     "C59",
     "x_vars[6,Package Unit]"
    ],
    [
     "C60",
     "x_vars[7,Window Unit]"
    ],
    [
     "C61",
     "x_vars[7,Portable Unit]"
    ],
    [
     "C62",
     "x_vars[7,Split System]"
    ],
    [
     "C63",
     "x_vars[7,Ductless System]"
    ],
    [
     "C64",
     "x_vars[7,Central AC]"
    ],
    [
     "C65",
     "x_vars[7,Hybrid AC]"
    ],
    [
     "C66",
     "x_vars[7,Geothermal AC]"
    ],
    [
     "C67",
     "x_vars[7,Smart AC]"
    ],
    [
     "C68",
     "x_vars[7,Evaporative Cooler]"
    ],
    [
     "C69",
     "x_vars[7,Package Unit]"
    ],
    [
     "C70",
     "x_vars[8,Window Unit]"
    ],
    [
     "C71",
     "x_vars[8,Portable Unit]"
    ],
    [
     "C72",
     "x_vars[8,Split System]"
    ],
    [
     "C73",
     "x_vars[8,Ductless System]"
    ],
    [
     "C74",
     "x_vars[8,Central AC]"
    ],
    [
     "C75",
     "x_vars[8,Hybrid AC]"
    ],
    [
     "C76",
     "x_vars[8,Geothermal AC]"
    ],
    [
     "C77",
     "x_vars[8,Smart AC]"
    ],
    [
     "C78",
     "x_vars[8,Evaporative Cooler]"
    ],
    [
     "C79",
     "x_vars[8,Package Unit]"
    ],
    [
     "C80",
     "x_vars[9,Window Unit]"
    ],
    [
     "C81",
     "x_vars[9,Portable Unit]"
    ],
    [
     "C82",
     "x_vars[9,Split System]"
    ],
    [
     "C83",
     "x_vars[9,Ductless System]"
    ],
    [
     "C84",
     "x_vars[9,Central AC]"
    ],
    [
     "C85",
     "x_vars[9,Hybrid AC]"
    ],
    [
     "C86",
     "x_vars[9,Geothermal AC]"
    ],
    [
     "C87",
     "x_vars[9,Smart AC]"
    ],
    [
     "C88",
     "x_vars[9,Evaporative Cooler]"
    ],
    [
     "C89",
     "x_vars[9,Package Unit]"
    ],
    [
     "C90",
     "x_vars[10,Window Unit]"
    ],
    [
     "C91",
     "x_vars[10,Portable Unit]"
    ],
    [
     "C92",
     "x_vars[10,Split System]"
    ],
    [
     "C93",
     "x_vars[10,Ductless System]"
    ],
    [
     "C94",
     "x_vars[10,Central AC]"
    ],
    [
     "C95",
     "x_vars[10,Hybrid AC]"
    ],
    [
     "C96",
     "x_vars[10,Geothermal AC]"
    ],
    [
     "C97",
     "x_vars[10,Smart AC]"
    ],
    [
     "C98",
     "x_vars[10,Evaporative Cooler]"
    ],
    [
     "C99",
     "x_vars[10,Package Unit]"
    ],
    [
     "C100",
     "x_vars[11,Window Unit]"
    ],
    [
     "C101",
     "x_vars[11,Portable Unit]"
    ],
    [
     "C102",
     "x_vars[11,Split System]"
    ],
    [
     "C103",
     "x_vars[11,Ductless System]"
    ],
    [
     "C104",
     "x_vars[11,Central AC]"
    ],
    [
     "C105",
     "x_vars[11,Hybrid AC]"
    ],
    [
     "C106",
     "x_vars[11,Geothermal AC]"
    ],
    [
     "C107",
     "x_vars[11,Smart AC]"
    ],
    [
     "C108",
     "x_vars[11,Evaporative Cooler]"
    ],
    [
     "C109",
     "x_vars[11,Package Unit]"
    ],
    [
     "C110",
     "x_vars[12,Window Unit]"
    ],
    [
     "C111",
     "x_vars[12,Portable Unit]"
    ],
    [
     "C112",
     "x_vars[12,Split System]"
    ],
    [
     "C113",
     "x_vars[12,Ductless System]"
    ],
    [
     "C114",
     "x_vars[12,Central AC]"
    ],
    [
     "C115",
     "x_vars[12,Hybrid AC]"
    ],
    [
     "C116",
     "x_vars[12,Geothermal AC]"
    ],
    [
     "C117",
     "x_vars[12,Smart AC]"
    ],
    [
     "C118",
     "x_vars[12,Evaporative Cooler]"
    ],
    [
     "C119",
     "x_vars[12,Package Unit]"
    ],
    [
     "C120",
     "x_vars[13,Window Unit]"
    ],
    [
     "C121",
     "x_vars[13,Portable Unit]"
    ],
    [
     "C122",
     "x_vars[13,Split System]"
    ],
    [
     "C123",
     "x_vars[13,Ductless System]"
    ],
    [
     "C124",
     "x_vars[13,Central AC]"
    ],
    [
     "C125",
     "x_vars[13,Hybrid AC]"
    ],
    [
     "C126",
     "x_vars[13,Geothermal AC]"
    ],
    [
     "C127",
     "x_vars[13,Smart AC]"
    ],
    [
     "C128",
     "x_vars[13,Evaporative Cooler]"
    ],
    [
     "C129",
     "x_vars[13,Package Unit]"
    ],
    [
     "C130",
     "x_vars[14,Window Unit]"
    ],
    [
     "C131",
     "x_vars[14,Portable Unit]"
    ],
    [
     "C132",
     "x_vars[14,Split System]"
    ],
    [
     "C133",
     "x_vars[14,Ductless System]"
    ],
    [
     "C134",
     "x_vars[14,Central AC]"
    ],
    [
     "C135",
     "x_vars[14,Hybrid AC]"
    ],
    [
     "C136",
     "x_vars[14,Geothermal AC]"
    ],
    [
     "C137",
     "x_vars[14,Smart AC]"
    ],
    [
     "C138",
     "x_vars[14,Evaporative Cooler]"
    ],
    [
     "C139",
     "x_vars[14,Package Unit]"
    ],
    [
     "C140",
     "x_vars[15,Window Unit]"
    ],
    [
     "C141",
     "x_vars[15,Portable Unit]"
    ],
    [
     "C142",
     "x_vars[15,Split System]"
    ],
    [
     "C143",
     "x_vars[15,Ductless System]"
    ],
    [
     "C144",
     "x_vars[15,Central AC]"
    ],
    [
     "C145",
     "x_vars[15,Hybrid AC]"
    ],
    [
     "C146",
     "x_vars[15,Geothermal AC]"
    ],
    [
     "C147",
     "x_vars[15,Smart AC]"
    ],
    [
     "C148",
     "x_vars[15,Evaporative Cooler]"
    ],
    [
     "C149",
     "x_vars[15,Package Unit]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Window Unit]",
    "x_0_1": "x_vars[2,Window Unit]",
    "x_0_2": "x_vars[3,Window Unit]",
    "x_0_3": "x_vars[4,Window Unit]",
    "x_0_4": "x_vars[5,Window Unit]",
    "x_0_5": "x_vars[6,Window Unit]",
    "x_0_6": "x_vars[7,Window Unit]",
    "x_0_7": "x_vars[8,Window Unit]",
    "x_0_8": "x_vars[9,Window Unit]",
    "x_0_9": "x_vars[10,Window Unit]",
    "x_0_10": "x_vars[11,Window Unit]",
    "x_0_11": "x_vars[12,Window Unit]",
    "x_0_12": "x_vars[13,Window Unit]",
    "x_0_13": "x_vars[14,Window Unit]",
    "x_0_14": "x_vars[15,Window Unit]",
    "x_1_0": "x_vars[1,Portable Unit]",
    "x_1_1": "x_vars[2,Portable Unit]",
    "x_1_2": "x_vars[3,Portable Unit]",
    "x_1_3": "x_vars[4,Portable Unit]",
    "x_1_4": "x_vars[5,Portable Unit]",
    "x_1_5": "x_vars[6,Portable Unit]",
    "x_1_6": "x_vars[7,Portable Unit]",
    "x_1_7": "x_vars[8,Portable Unit]",
    "x_1_8": "x_vars[9,Portable Unit]",
    "x_1_9": "x_vars[10,Portable Unit]",
    "x_1_10": "x_vars[11,Portable Unit]",
    "x_1_11": "x_vars[12,Portable Unit]",
    "x_1_12": "x_vars[13,Portable Unit]",
    "x_1_13": "x_vars[14,Portable Unit]",
    "x_1_14": "x_vars[15,Portable Unit]",
    "x_2_0": "x_vars[1,Split System]",
    "x_2_1": "x_vars[2,Split System]",
    "x_2_2": "x_vars[3,Split System]",
    "x_2_3": "x_vars[4,Split System]",
    "x_2_4": "x_vars[5,Split System]",
    "x_2_5": "x_vars[6,Split System]",
    "x_2_6": "x_vars[7,Split System]",
    "x_2_7": "x_vars[8,Split System]",
    "x_2_8": "x_vars[9,Split System]",
    "x_2_9": "x_vars[10,Split System]",
    "x_2_10": "x_vars[11,Split System]",
    "x_2_11": "x_vars[12,Split System]",
    "x_2_12": "x_vars[13,Split System]",
    "x_2_13": "x_vars[14,Split System]",
    "x_2_14": "x_vars[15,Split System]",
    "x_3_0": "x_vars[1,Ductless System]",
    "x_3_1": "x_vars[2,Ductless System]",
    "x_3_2": "x_vars[3,Ductless System]",
    "x_3_3": "x_vars[4,Ductless System]",
    "x_3_4": "x_vars[5,Ductless System]",
    "x_3_5": "x_vars[6,Ductless System]",
    "x_3_6": "x_vars[7,Ductless System]",
    "x_3_7": "x_vars[8,Ductless System]",
    "x_3_8": "x_vars[9,Ductless System]",
    "x_3_9": "x_vars[10,Ductless System]",
    "x_3_10": "x_vars[11,Ductless System]",
    "x_3_11": "x_vars[12,Ductless System]",
    "x_3_12": "x_vars[13,Ductless System]",
    "x_3_13": "x_vars[14,Ductless System]",
    "x_3_14": "x_vars[15,Ductless System]",
    "x_4_0": "x_vars[1,Central AC]",
    "x_4_1": "x_vars[2,Central AC]",
    "x_4_2": "x_vars[3,Central AC]",
    "x_4_3": "x_vars[4,Central AC]",
    "x_4_4": "x_vars[5,Central AC]",
    "x_4_5": "x_vars[6,Central AC]",
    "x_4_6": "x_vars[7,Central AC]",
    "x_4_7": "x_vars[8,Central AC]",
    "x_4_8": "x_vars[9,Central AC]",
    "x_4_9": "x_vars[10,Central AC]",
    "x_4_10": "x_vars[11,Central AC]",
    "x_4_11": "x_vars[12,Central AC]",
    "x_4_12": "x_vars[13,Central AC]",
    "x_4_13": "x_vars[14,Central AC]",
    "x_4_14": "x_vars[15,Central AC]",
    "x_5_0": "x_vars[1,Hybrid AC]",
    "x_5_1": "x_vars[2,Hybrid AC]",
    "x_5_2": "x_vars[3,Hybrid AC]",
    "x_5_3": "x_vars[4,Hybrid AC]",
    "x_5_4": "x_vars[5,Hybrid AC]",
    "x_5_5": "x_vars[6,Hybrid AC]",
    "x_5_6": "x_vars[7,Hybrid AC]",
    "x_5_7": "x_vars[8,Hybrid AC]",
    "x_5_8": "x_vars[9,Hybrid AC]",
    "x_5_9": "x_vars[10,Hybrid AC]",
    "x_5_10": "x_vars[11,Hybrid AC]",
    "x_5_11": "x_vars[12,Hybrid AC]",
    "x_5_12": "x_vars[13,Hybrid AC]",
    "x_5_13": "x_vars[14,Hybrid AC]",
    "x_5_14": "x_vars[15,Hybrid AC]",
    "x_6_0": "x_vars[1,Geothermal AC]",
    "x_6_1": "x_vars[2,Geothermal AC]",
    "x_6_2": "x_vars[3,Geothermal AC]",
    "x_6_3": "x_vars[4,Geothermal AC]",
    "x_6_4": "x_vars[5,Geothermal AC]",
    "x_6_5": "x_vars[6,Geothermal AC]",
    "x_6_6": "x_vars[7,Geothermal AC]",
    "x_6_7": "x_vars[8,Geothermal AC]",
    "x_6_8": "x_vars[9,Geothermal AC]",
    "x_6_9": "x_vars[10,Geothermal AC]",
    "x_6_10": "x_vars[11,Geothermal AC]",
    "x_6_11": "x_vars[12,Geothermal AC]",
    "x_6_12": "x_vars[13,Geothermal AC]",
    "x_6_13": "x_vars[14,Geothermal AC]",
    "x_6_14": "x_vars[15,Geothermal AC]",
    "x_7_0": "x_vars[1,Smart AC]",
    "x_7_1": "x_vars[2,Smart AC]",
    "x_7_2": "x_vars[3,Smart AC]",
    "x_7_3": "x_vars[4,Smart AC]",
    "x_7_4": "x_vars[5,Smart AC]",
    "x_7_5": "x_vars[6,Smart AC]",
    "x_7_6": "x_vars[7,Smart AC]",
    "x_7_7": "x_vars[8,Smart AC]",
    "x_7_8": "x_vars[9,Smart AC]",
    "x_7_9": "x_vars[10,Smart AC]",
    "x_7_10": "x_vars[11,Smart AC]",
    "x_7_11": "x_vars[12,Smart AC]",
    "x_7_12": "x_vars[13,Smart AC]",
    "x_7_13": "x_vars[14,Smart AC]",
    "x_7_14": "x_vars[15,Smart AC]",
    "x_8_0": "x_vars[1,Evaporative Cooler]",
    "x_8_1": "x_vars[2,Evaporative Cooler]",
    "x_8_2": "x_vars[3,Evaporative Cooler]",
    "x_8_3": "x_vars[4,Evaporative Cooler]",
    "x_8_4": "x_vars[5,Evaporative Cooler]",
    "x_8_5": "x_vars[6,Evaporative Cooler]",
    "x_8_6": "x_vars[7,Evaporative Cooler]",
    "x_8_7": "x_vars[8,Evaporative Cooler]",
    "x_8_8": "x_vars[9,Evaporative Cooler]",
    "x_8_9": "x_vars[10,Evaporative Cooler]",
    "x_8_10": "x_vars[11,Evaporative Cooler]",
    "x_8_11": "x_vars[12,Evaporative Cooler]",
    "x_8_12": "x_vars[13,Evaporative Cooler]",
    "x_8_13": "x_vars[14,Evaporative Cooler]",
    "x_8_14": "x_vars[15,Evaporative Cooler]",
    "x_9_0": "x_vars[1,Package Unit]",
    "x_9_1": "x_vars[2,Package Unit]",
    "x_9_2": "x_vars[3,Package Unit]",
    "x_9_3": "x_vars[4,Package Unit]",
    "x_9_4": "x_vars[5,Package Unit]",
    "x_9_5": "x_vars[6,Package Unit]",
    "x_9_6": "x_vars[7,Package Unit]",
    "x_9_7": "x_vars[8,Package Unit]",
    "x_9_8": "x_vars[9,Package Unit]",
    "x_9_9": "x_vars[10,Package Unit]",
    "x_9_10": "x_vars[11,Package Unit]",
    "x_9_11": "x_vars[12,Package Unit]",
    "x_9_12": "x_vars[13,Package Unit]",
    "x_9_13": "x_vars[14,Package Unit]",
    "x_9_14": "x_vars[15,Package Unit]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-049": {
   "source_code_sha256": "008ef5dbbf431a7860d2d3f665d81ef826825b35636a6b9200426a3f66521631",
   "original_solution_sha256": "818d23acaffa9267df75706492cd632d5f60fe8b7f08eacec77592296254f75a",
   "bindings": [
    [
     "C0",
     "x_vars[1,Smartphone]"
    ],
    [
     "C1",
     "x_vars[1,Laptop]"
    ],
    [
     "C2",
     "x_vars[1,Headphones]"
    ],
    [
     "C3",
     "x_vars[1,Camera]"
    ],
    [
     "C4",
     "x_vars[1,Smartwatch]"
    ],
    [
     "C5",
     "x_vars[1,Tablet]"
    ],
    [
     "C6",
     "x_vars[1,Bluetooth Speaker]"
    ],
    [
     "C7",
     "x_vars[1,Keyboard]"
    ],
    [
     "C8",
     "x_vars[1,Mouse]"
    ],
    [
     "C9",
     "x_vars[1,Monitor]"
    ],
    [
     "C10",
     "x_vars[1,Printer]"
    ],
    [
     "C11",
     "x_vars[1,External Hard Drive]"
    ],
    [
     "C12",
     "x_vars[1,Router]"
    ],
    [
     "C13",
     "x_vars[1,Power Bank]"
    ],
    [
     "C14",
     "x_vars[1,Memory Card]"
    ],
    [
     "C15",
     "x_vars[1,USB Flash Drive]"
    ],
    [
     "C16",
     "x_vars[1,Smart Home Hub]"
    ],
    [
     "C17",
     "x_vars[1,Gaming Console]"
    ],
    [
     "C18",
     "x_vars[1,Fitness Tracker]"
    ],
    [
     "C19",
     "x_vars[1,E-Reader]"
    ],
    [
     "C20",
     "x_vars[2,Smartphone]"
    ],
    [
     "C21",
     "x_vars[2,Laptop]"
    ],
    [
     "C22",
     "x_vars[2,Headphones]"
    ],
    [
     "C23",
     "x_vars[2,Camera]"
    ],
    [
     "C24",
     "x_vars[2,Smartwatch]"
    ],
    [
     "C25",
     "x_vars[2,Tablet]"
    ],
    [
     "C26",
     "x_vars[2,Bluetooth Speaker]"
    ],
    [
     "C27",
     "x_vars[2,Keyboard]"
    ],
    [
     "C28",
     "x_vars[2,Mouse]"
    ],
    [
     "C29",
     "x_vars[2,Monitor]"
    ],
    [
     "C30",
     "x_vars[2,Printer]"
    ],
    [
     "C31",
     "x_vars[2,External Hard Drive]"
    ],
    [
     "C32",
     "x_vars[2,Router]"
    ],
    [
     "C33",
     "x_vars[2,Power Bank]"
    ],
    [
     "C34",
     "x_vars[2,Memory Card]"
    ],
    [
     "C35",
     "x_vars[2,USB Flash Drive]"
    ],
    [
     "C36",
     "x_vars[2,Smart Home Hub]"
    ],
    [
     "C37",
     "x_vars[2,Gaming Console]"
    ],
    [
     "C38",
     "x_vars[2,Fitness Tracker]"
    ],
    [
     "C39",
     "x_vars[2,E-Reader]"
    ],
    [
     "C40",
     "x_vars[3,Smartphone]"
    ],
    [
     "C41",
     "x_vars[3,Laptop]"
    ],
    [
     "C42",
     "x_vars[3,Headphones]"
    ],
    [
     "C43",
     "x_vars[3,Camera]"
    ],
    [
     "C44",
     "x_vars[3,Smartwatch]"
    ],
    [
     "C45",
     "x_vars[3,Tablet]"
    ],
    [
     "C46",
     "x_vars[3,Bluetooth Speaker]"
    ],
    [
     "C47",
     "x_vars[3,Keyboard]"
    ],
    [
     "C48",
     "x_vars[3,Mouse]"
    ],
    [
     "C49",
     "x_vars[3,Monitor]"
    ],
    [
     "C50",
     "x_vars[3,Printer]"
    ],
    [
     "C51",
     "x_vars[3,External Hard Drive]"
    ],
    [
     "C52",
     "x_vars[3,Router]"
    ],
    [
     "C53",
     "x_vars[3,Power Bank]"
    ],
    [
     "C54",
     "x_vars[3,Memory Card]"
    ],
    [
     "C55",
     "x_vars[3,USB Flash Drive]"
    ],
    [
     "C56",
     "x_vars[3,Smart Home Hub]"
    ],
    [
     "C57",
     "x_vars[3,Gaming Console]"
    ],
    [
     "C58",
     "x_vars[3,Fitness Tracker]"
    ],
    [
     "C59",
     "x_vars[3,E-Reader]"
    ],
    [
     "C60",
     "x_vars[4,Smartphone]"
    ],
    [
     "C61",
     "x_vars[4,Laptop]"
    ],
    [
     "C62",
     "x_vars[4,Headphones]"
    ],
    [
     "C63",
     "x_vars[4,Camera]"
    ],
    [
     "C64",
     "x_vars[4,Smartwatch]"
    ],
    [
     "C65",
     "x_vars[4,Tablet]"
    ],
    [
     "C66",
     "x_vars[4,Bluetooth Speaker]"
    ],
    [
     "C67",
     "x_vars[4,Keyboard]"
    ],
    [
     "C68",
     "x_vars[4,Mouse]"
    ],
    [
     "C69",
     "x_vars[4,Monitor]"
    ],
    [
     "C70",
     "x_vars[4,Printer]"
    ],
    [
     "C71",
     "x_vars[4,External Hard Drive]"
    ],
    [
     "C72",
     "x_vars[4,Router]"
    ],
    [
     "C73",
     "x_vars[4,Power Bank]"
    ],
    [
     "C74",
     "x_vars[4,Memory Card]"
    ],
    [
     "C75",
     "x_vars[4,USB Flash Drive]"
    ],
    [
     "C76",
     "x_vars[4,Smart Home Hub]"
    ],
    [
     "C77",
     "x_vars[4,Gaming Console]"
    ],
    [
     "C78",
     "x_vars[4,Fitness Tracker]"
    ],
    [
     "C79",
     "x_vars[4,E-Reader]"
    ],
    [
     "C80",
     "x_vars[5,Smartphone]"
    ],
    [
     "C81",
     "x_vars[5,Laptop]"
    ],
    [
     "C82",
     "x_vars[5,Headphones]"
    ],
    [
     "C83",
     "x_vars[5,Camera]"
    ],
    [
     "C84",
     "x_vars[5,Smartwatch]"
    ],
    [
     "C85",
     "x_vars[5,Tablet]"
    ],
    [
     "C86",
     "x_vars[5,Bluetooth Speaker]"
    ],
    [
     "C87",
     "x_vars[5,Keyboard]"
    ],
    [
     "C88",
     "x_vars[5,Mouse]"
    ],
    [
     "C89",
     "x_vars[5,Monitor]"
    ],
    [
     "C90",
     "x_vars[5,Printer]"
    ],
    [
     "C91",
     "x_vars[5,External Hard Drive]"
    ],
    [
     "C92",
     "x_vars[5,Router]"
    ],
    [
     "C93",
     "x_vars[5,Power Bank]"
    ],
    [
     "C94",
     "x_vars[5,Memory Card]"
    ],
    [
     "C95",
     "x_vars[5,USB Flash Drive]"
    ],
    [
     "C96",
     "x_vars[5,Smart Home Hub]"
    ],
    [
     "C97",
     "x_vars[5,Gaming Console]"
    ],
    [
     "C98",
     "x_vars[5,Fitness Tracker]"
    ],
    [
     "C99",
     "x_vars[5,E-Reader]"
    ],
    [
     "C100",
     "x_vars[6,Smartphone]"
    ],
    [
     "C101",
     "x_vars[6,Laptop]"
    ],
    [
     "C102",
     "x_vars[6,Headphones]"
    ],
    [
     "C103",
     "x_vars[6,Camera]"
    ],
    [
     "C104",
     "x_vars[6,Smartwatch]"
    ],
    [
     "C105",
     "x_vars[6,Tablet]"
    ],
    [
     "C106",
     "x_vars[6,Bluetooth Speaker]"
    ],
    [
     "C107",
     "x_vars[6,Keyboard]"
    ],
    [
     "C108",
     "x_vars[6,Mouse]"
    ],
    [
     "C109",
     "x_vars[6,Monitor]"
    ],
    [
     "C110",
     "x_vars[6,Printer]"
    ],
    [
     "C111",
     "x_vars[6,External Hard Drive]"
    ],
    [
     "C112",
     "x_vars[6,Router]"
    ],
    [
     "C113",
     "x_vars[6,Power Bank]"
    ],
    [
     "C114",
     "x_vars[6,Memory Card]"
    ],
    [
     "C115",
     "x_vars[6,USB Flash Drive]"
    ],
    [
     "C116",
     "x_vars[6,Smart Home Hub]"
    ],
    [
     "C117",
     "x_vars[6,Gaming Console]"
    ],
    [
     "C118",
     "x_vars[6,Fitness Tracker]"
    ],
    [
     "C119",
     "x_vars[6,E-Reader]"
    ],
    [
     "C120",
     "x_vars[7,Smartphone]"
    ],
    [
     "C121",
     "x_vars[7,Laptop]"
    ],
    [
     "C122",
     "x_vars[7,Headphones]"
    ],
    [
     "C123",
     "x_vars[7,Camera]"
    ],
    [
     "C124",
     "x_vars[7,Smartwatch]"
    ],
    [
     "C125",
     "x_vars[7,Tablet]"
    ],
    [
     "C126",
     "x_vars[7,Bluetooth Speaker]"
    ],
    [
     "C127",
     "x_vars[7,Keyboard]"
    ],
    [
     "C128",
     "x_vars[7,Mouse]"
    ],
    [
     "C129",
     "x_vars[7,Monitor]"
    ],
    [
     "C130",
     "x_vars[7,Printer]"
    ],
    [
     "C131",
     "x_vars[7,External Hard Drive]"
    ],
    [
     "C132",
     "x_vars[7,Router]"
    ],
    [
     "C133",
     "x_vars[7,Power Bank]"
    ],
    [
     "C134",
     "x_vars[7,Memory Card]"
    ],
    [
     "C135",
     "x_vars[7,USB Flash Drive]"
    ],
    [
     "C136",
     "x_vars[7,Smart Home Hub]"
    ],
    [
     "C137",
     "x_vars[7,Gaming Console]"
    ],
    [
     "C138",
     "x_vars[7,Fitness Tracker]"
    ],
    [
     "C139",
     "x_vars[7,E-Reader]"
    ],
    [
     "C140",
     "x_vars[8,Smartphone]"
    ],
    [
     "C141",
     "x_vars[8,Laptop]"
    ],
    [
     "C142",
     "x_vars[8,Headphones]"
    ],
    [
     "C143",
     "x_vars[8,Camera]"
    ],
    [
     "C144",
     "x_vars[8,Smartwatch]"
    ],
    [
     "C145",
     "x_vars[8,Tablet]"
    ],
    [
     "C146",
     "x_vars[8,Bluetooth Speaker]"
    ],
    [
     "C147",
     "x_vars[8,Keyboard]"
    ],
    [
     "C148",
     "x_vars[8,Mouse]"
    ],
    [
     "C149",
     "x_vars[8,Monitor]"
    ],
    [
     "C150",
     "x_vars[8,Printer]"
    ],
    [
     "C151",
     "x_vars[8,External Hard Drive]"
    ],
    [
     "C152",
     "x_vars[8,Router]"
    ],
    [
     "C153",
     "x_vars[8,Power Bank]"
    ],
    [
     "C154",
     "x_vars[8,Memory Card]"
    ],
    [
     "C155",
     "x_vars[8,USB Flash Drive]"
    ],
    [
     "C156",
     "x_vars[8,Smart Home Hub]"
    ],
    [
     "C157",
     "x_vars[8,Gaming Console]"
    ],
    [
     "C158",
     "x_vars[8,Fitness Tracker]"
    ],
    [
     "C159",
     "x_vars[8,E-Reader]"
    ],
    [
     "C160",
     "x_vars[9,Smartphone]"
    ],
    [
     "C161",
     "x_vars[9,Laptop]"
    ],
    [
     "C162",
     "x_vars[9,Headphones]"
    ],
    [
     "C163",
     "x_vars[9,Camera]"
    ],
    [
     "C164",
     "x_vars[9,Smartwatch]"
    ],
    [
     "C165",
     "x_vars[9,Tablet]"
    ],
    [
     "C166",
     "x_vars[9,Bluetooth Speaker]"
    ],
    [
     "C167",
     "x_vars[9,Keyboard]"
    ],
    [
     "C168",
     "x_vars[9,Mouse]"
    ],
    [
     "C169",
     "x_vars[9,Monitor]"
    ],
    [
     "C170",
     "x_vars[9,Printer]"
    ],
    [
     "C171",
     "x_vars[9,External Hard Drive]"
    ],
    [
     "C172",
     "x_vars[9,Router]"
    ],
    [
     "C173",
     "x_vars[9,Power Bank]"
    ],
    [
     "C174",
     "x_vars[9,Memory Card]"
    ],
    [
     "C175",
     "x_vars[9,USB Flash Drive]"
    ],
    [
     "C176",
     "x_vars[9,Smart Home Hub]"
    ],
    [
     "C177",
     "x_vars[9,Gaming Console]"
    ],
    [
     "C178",
     "x_vars[9,Fitness Tracker]"
    ],
    [
     "C179",
     "x_vars[9,E-Reader]"
    ],
    [
     "C180",
     "x_vars[10,Smartphone]"
    ],
    [
     "C181",
     "x_vars[10,Laptop]"
    ],
    [
     "C182",
     "x_vars[10,Headphones]"
    ],
    [
     "C183",
     "x_vars[10,Camera]"
    ],
    [
     "C184",
     "x_vars[10,Smartwatch]"
    ],
    [
     "C185",
     "x_vars[10,Tablet]"
    ],
    [
     "C186",
     "x_vars[10,Bluetooth Speaker]"
    ],
    [
     "C187",
     "x_vars[10,Keyboard]"
    ],
    [
     "C188",
     "x_vars[10,Mouse]"
    ],
    [
     "C189",
     "x_vars[10,Monitor]"
    ],
    [
     "C190",
     "x_vars[10,Printer]"
    ],
    [
     "C191",
     "x_vars[10,External Hard Drive]"
    ],
    [
     "C192",
     "x_vars[10,Router]"
    ],
    [
     "C193",
     "x_vars[10,Power Bank]"
    ],
    [
     "C194",
     "x_vars[10,Memory Card]"
    ],
    [
     "C195",
     "x_vars[10,USB Flash Drive]"
    ],
    [
     "C196",
     "x_vars[10,Smart Home Hub]"
    ],
    [
     "C197",
     "x_vars[10,Gaming Console]"
    ],
    [
     "C198",
     "x_vars[10,Fitness Tracker]"
    ],
    [
     "C199",
     "x_vars[10,E-Reader]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Smartphone]",
    "x_0_1": "x_vars[2,Smartphone]",
    "x_0_2": "x_vars[3,Smartphone]",
    "x_0_3": "x_vars[4,Smartphone]",
    "x_0_4": "x_vars[5,Smartphone]",
    "x_0_5": "x_vars[6,Smartphone]",
    "x_0_6": "x_vars[7,Smartphone]",
    "x_0_7": "x_vars[8,Smartphone]",
    "x_0_8": "x_vars[9,Smartphone]",
    "x_0_9": "x_vars[10,Smartphone]",
    "x_1_0": "x_vars[1,Laptop]",
    "x_1_1": "x_vars[2,Laptop]",
    "x_1_2": "x_vars[3,Laptop]",
    "x_1_3": "x_vars[4,Laptop]",
    "x_1_4": "x_vars[5,Laptop]",
    "x_1_5": "x_vars[6,Laptop]",
    "x_1_6": "x_vars[7,Laptop]",
    "x_1_7": "x_vars[8,Laptop]",
    "x_1_8": "x_vars[9,Laptop]",
    "x_1_9": "x_vars[10,Laptop]",
    "x_2_0": "x_vars[1,Headphones]",
    "x_2_1": "x_vars[2,Headphones]",
    "x_2_2": "x_vars[3,Headphones]",
    "x_2_3": "x_vars[4,Headphones]",
    "x_2_4": "x_vars[5,Headphones]",
    "x_2_5": "x_vars[6,Headphones]",
    "x_2_6": "x_vars[7,Headphones]",
    "x_2_7": "x_vars[8,Headphones]",
    "x_2_8": "x_vars[9,Headphones]",
    "x_2_9": "x_vars[10,Headphones]",
    "x_3_0": "x_vars[1,Camera]",
    "x_3_1": "x_vars[2,Camera]",
    "x_3_2": "x_vars[3,Camera]",
    "x_3_3": "x_vars[4,Camera]",
    "x_3_4": "x_vars[5,Camera]",
    "x_3_5": "x_vars[6,Camera]",
    "x_3_6": "x_vars[7,Camera]",
    "x_3_7": "x_vars[8,Camera]",
    "x_3_8": "x_vars[9,Camera]",
    "x_3_9": "x_vars[10,Camera]",
    "x_4_0": "x_vars[1,Smartwatch]",
    "x_4_1": "x_vars[2,Smartwatch]",
    "x_4_2": "x_vars[3,Smartwatch]",
    "x_4_3": "x_vars[4,Smartwatch]",
    "x_4_4": "x_vars[5,Smartwatch]",
    "x_4_5": "x_vars[6,Smartwatch]",
    "x_4_6": "x_vars[7,Smartwatch]",
    "x_4_7": "x_vars[8,Smartwatch]",
    "x_4_8": "x_vars[9,Smartwatch]",
    "x_4_9": "x_vars[10,Smartwatch]",
    "x_5_0": "x_vars[1,Tablet]",
    "x_5_1": "x_vars[2,Tablet]",
    "x_5_2": "x_vars[3,Tablet]",
    "x_5_3": "x_vars[4,Tablet]",
    "x_5_4": "x_vars[5,Tablet]",
    "x_5_5": "x_vars[6,Tablet]",
    "x_5_6": "x_vars[7,Tablet]",
    "x_5_7": "x_vars[8,Tablet]",
    "x_5_8": "x_vars[9,Tablet]",
    "x_5_9": "x_vars[10,Tablet]",
    "x_6_0": "x_vars[1,Bluetooth Speaker]",
    "x_6_1": "x_vars[2,Bluetooth Speaker]",
    "x_6_2": "x_vars[3,Bluetooth Speaker]",
    "x_6_3": "x_vars[4,Bluetooth Speaker]",
    "x_6_4": "x_vars[5,Bluetooth Speaker]",
    "x_6_5": "x_vars[6,Bluetooth Speaker]",
    "x_6_6": "x_vars[7,Bluetooth Speaker]",
    "x_6_7": "x_vars[8,Bluetooth Speaker]",
    "x_6_8": "x_vars[9,Bluetooth Speaker]",
    "x_6_9": "x_vars[10,Bluetooth Speaker]",
    "x_7_0": "x_vars[1,Keyboard]",
    "x_7_1": "x_vars[2,Keyboard]",
    "x_7_2": "x_vars[3,Keyboard]",
    "x_7_3": "x_vars[4,Keyboard]",
    "x_7_4": "x_vars[5,Keyboard]",
    "x_7_5": "x_vars[6,Keyboard]",
    "x_7_6": "x_vars[7,Keyboard]",
    "x_7_7": "x_vars[8,Keyboard]",
    "x_7_8": "x_vars[9,Keyboard]",
    "x_7_9": "x_vars[10,Keyboard]",
    "x_8_0": "x_vars[1,Mouse]",
    "x_8_1": "x_vars[2,Mouse]",
    "x_8_2": "x_vars[3,Mouse]",
    "x_8_3": "x_vars[4,Mouse]",
    "x_8_4": "x_vars[5,Mouse]",
    "x_8_5": "x_vars[6,Mouse]",
    "x_8_6": "x_vars[7,Mouse]",
    "x_8_7": "x_vars[8,Mouse]",
    "x_8_8": "x_vars[9,Mouse]",
    "x_8_9": "x_vars[10,Mouse]",
    "x_9_0": "x_vars[1,Monitor]",
    "x_9_1": "x_vars[2,Monitor]",
    "x_9_2": "x_vars[3,Monitor]",
    "x_9_3": "x_vars[4,Monitor]",
    "x_9_4": "x_vars[5,Monitor]",
    "x_9_5": "x_vars[6,Monitor]",
    "x_9_6": "x_vars[7,Monitor]",
    "x_9_7": "x_vars[8,Monitor]",
    "x_9_8": "x_vars[9,Monitor]",
    "x_9_9": "x_vars[10,Monitor]",
    "x_10_0": "x_vars[1,Printer]",
    "x_10_1": "x_vars[2,Printer]",
    "x_10_2": "x_vars[3,Printer]",
    "x_10_3": "x_vars[4,Printer]",
    "x_10_4": "x_vars[5,Printer]",
    "x_10_5": "x_vars[6,Printer]",
    "x_10_6": "x_vars[7,Printer]",
    "x_10_7": "x_vars[8,Printer]",
    "x_10_8": "x_vars[9,Printer]",
    "x_10_9": "x_vars[10,Printer]",
    "x_11_0": "x_vars[1,External Hard Drive]",
    "x_11_1": "x_vars[2,External Hard Drive]",
    "x_11_2": "x_vars[3,External Hard Drive]",
    "x_11_3": "x_vars[4,External Hard Drive]",
    "x_11_4": "x_vars[5,External Hard Drive]",
    "x_11_5": "x_vars[6,External Hard Drive]",
    "x_11_6": "x_vars[7,External Hard Drive]",
    "x_11_7": "x_vars[8,External Hard Drive]",
    "x_11_8": "x_vars[9,External Hard Drive]",
    "x_11_9": "x_vars[10,External Hard Drive]",
    "x_12_0": "x_vars[1,Router]",
    "x_12_1": "x_vars[2,Router]",
    "x_12_2": "x_vars[3,Router]",
    "x_12_3": "x_vars[4,Router]",
    "x_12_4": "x_vars[5,Router]",
    "x_12_5": "x_vars[6,Router]",
    "x_12_6": "x_vars[7,Router]",
    "x_12_7": "x_vars[8,Router]",
    "x_12_8": "x_vars[9,Router]",
    "x_12_9": "x_vars[10,Router]",
    "x_13_0": "x_vars[1,Power Bank]",
    "x_13_1": "x_vars[2,Power Bank]",
    "x_13_2": "x_vars[3,Power Bank]",
    "x_13_3": "x_vars[4,Power Bank]",
    "x_13_4": "x_vars[5,Power Bank]",
    "x_13_5": "x_vars[6,Power Bank]",
    "x_13_6": "x_vars[7,Power Bank]",
    "x_13_7": "x_vars[8,Power Bank]",
    "x_13_8": "x_vars[9,Power Bank]",
    "x_13_9": "x_vars[10,Power Bank]",
    "x_14_0": "x_vars[1,Memory Card]",
    "x_14_1": "x_vars[2,Memory Card]",
    "x_14_2": "x_vars[3,Memory Card]",
    "x_14_3": "x_vars[4,Memory Card]",
    "x_14_4": "x_vars[5,Memory Card]",
    "x_14_5": "x_vars[6,Memory Card]",
    "x_14_6": "x_vars[7,Memory Card]",
    "x_14_7": "x_vars[8,Memory Card]",
    "x_14_8": "x_vars[9,Memory Card]",
    "x_14_9": "x_vars[10,Memory Card]",
    "x_15_0": "x_vars[1,USB Flash Drive]",
    "x_15_1": "x_vars[2,USB Flash Drive]",
    "x_15_2": "x_vars[3,USB Flash Drive]",
    "x_15_3": "x_vars[4,USB Flash Drive]",
    "x_15_4": "x_vars[5,USB Flash Drive]",
    "x_15_5": "x_vars[6,USB Flash Drive]",
    "x_15_6": "x_vars[7,USB Flash Drive]",
    "x_15_7": "x_vars[8,USB Flash Drive]",
    "x_15_8": "x_vars[9,USB Flash Drive]",
    "x_15_9": "x_vars[10,USB Flash Drive]",
    "x_16_0": "x_vars[1,Smart Home Hub]",
    "x_16_1": "x_vars[2,Smart Home Hub]",
    "x_16_2": "x_vars[3,Smart Home Hub]",
    "x_16_3": "x_vars[4,Smart Home Hub]",
    "x_16_4": "x_vars[5,Smart Home Hub]",
    "x_16_5": "x_vars[6,Smart Home Hub]",
    "x_16_6": "x_vars[7,Smart Home Hub]",
    "x_16_7": "x_vars[8,Smart Home Hub]",
    "x_16_8": "x_vars[9,Smart Home Hub]",
    "x_16_9": "x_vars[10,Smart Home Hub]",
    "x_17_0": "x_vars[1,Gaming Console]",
    "x_17_1": "x_vars[2,Gaming Console]",
    "x_17_2": "x_vars[3,Gaming Console]",
    "x_17_3": "x_vars[4,Gaming Console]",
    "x_17_4": "x_vars[5,Gaming Console]",
    "x_17_5": "x_vars[6,Gaming Console]",
    "x_17_6": "x_vars[7,Gaming Console]",
    "x_17_7": "x_vars[8,Gaming Console]",
    "x_17_8": "x_vars[9,Gaming Console]",
    "x_17_9": "x_vars[10,Gaming Console]",
    "x_18_0": "x_vars[1,Fitness Tracker]",
    "x_18_1": "x_vars[2,Fitness Tracker]",
    "x_18_2": "x_vars[3,Fitness Tracker]",
    "x_18_3": "x_vars[4,Fitness Tracker]",
    "x_18_4": "x_vars[5,Fitness Tracker]",
    "x_18_5": "x_vars[6,Fitness Tracker]",
    "x_18_6": "x_vars[7,Fitness Tracker]",
    "x_18_7": "x_vars[8,Fitness Tracker]",
    "x_18_8": "x_vars[9,Fitness Tracker]",
    "x_18_9": "x_vars[10,Fitness Tracker]",
    "x_19_0": "x_vars[1,E-Reader]",
    "x_19_1": "x_vars[2,E-Reader]",
    "x_19_2": "x_vars[3,E-Reader]",
    "x_19_3": "x_vars[4,E-Reader]",
    "x_19_4": "x_vars[5,E-Reader]",
    "x_19_5": "x_vars[6,E-Reader]",
    "x_19_6": "x_vars[7,E-Reader]",
    "x_19_7": "x_vars[8,E-Reader]",
    "x_19_8": "x_vars[9,E-Reader]",
    "x_19_9": "x_vars[10,E-Reader]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-050": {
   "source_code_sha256": "6b90d8c6fd59f1d5820a9cba49f4eafc62f2738bf2b0ee0e32a0768dd98b5c65",
   "original_solution_sha256": "4458cfa2d051a53ebd5570ee89439918e3b04bacb4e394fdbfa4783439747b31",
   "bindings": [
    [
     "C0",
     "x_vars[1,Smartphone]"
    ],
    [
     "C1",
     "x_vars[1,Laptop]"
    ],
    [
     "C2",
     "x_vars[1,Headphones]"
    ],
    [
     "C3",
     "x_vars[1,Camera]"
    ],
    [
     "C4",
     "x_vars[1,Smartwatch]"
    ],
    [
     "C5",
     "x_vars[1,Tablet]"
    ],
    [
     "C6",
     "x_vars[1,Bluetooth Speaker]"
    ],
    [
     "C7",
     "x_vars[1,Keyboard]"
    ],
    [
     "C8",
     "x_vars[1,Mouse]"
    ],
    [
     "C9",
     "x_vars[1,Monitor]"
    ],
    [
     "C10",
     "x_vars[1,Printer]"
    ],
    [
     "C11",
     "x_vars[1,External Hard Drive]"
    ],
    [
     "C12",
     "x_vars[1,Router]"
    ],
    [
     "C13",
     "x_vars[1,Power Bank]"
    ],
    [
     "C14",
     "x_vars[1,Memory Card]"
    ],
    [
     "C15",
     "x_vars[1,USB Flash Drive]"
    ],
    [
     "C16",
     "x_vars[1,Smart Home Hub]"
    ],
    [
     "C17",
     "x_vars[1,Gaming Console]"
    ],
    [
     "C18",
     "x_vars[1,Fitness Tracker]"
    ],
    [
     "C19",
     "x_vars[1,E-Reader]"
    ],
    [
     "C20",
     "x_vars[2,Smartphone]"
    ],
    [
     "C21",
     "x_vars[2,Laptop]"
    ],
    [
     "C22",
     "x_vars[2,Headphones]"
    ],
    [
     "C23",
     "x_vars[2,Camera]"
    ],
    [
     "C24",
     "x_vars[2,Smartwatch]"
    ],
    [
     "C25",
     "x_vars[2,Tablet]"
    ],
    [
     "C26",
     "x_vars[2,Bluetooth Speaker]"
    ],
    [
     "C27",
     "x_vars[2,Keyboard]"
    ],
    [
     "C28",
     "x_vars[2,Mouse]"
    ],
    [
     "C29",
     "x_vars[2,Monitor]"
    ],
    [
     "C30",
     "x_vars[2,Printer]"
    ],
    [
     "C31",
     "x_vars[2,External Hard Drive]"
    ],
    [
     "C32",
     "x_vars[2,Router]"
    ],
    [
     "C33",
     "x_vars[2,Power Bank]"
    ],
    [
     "C34",
     "x_vars[2,Memory Card]"
    ],
    [
     "C35",
     "x_vars[2,USB Flash Drive]"
    ],
    [
     "C36",
     "x_vars[2,Smart Home Hub]"
    ],
    [
     "C37",
     "x_vars[2,Gaming Console]"
    ],
    [
     "C38",
     "x_vars[2,Fitness Tracker]"
    ],
    [
     "C39",
     "x_vars[2,E-Reader]"
    ],
    [
     "C40",
     "x_vars[3,Smartphone]"
    ],
    [
     "C41",
     "x_vars[3,Laptop]"
    ],
    [
     "C42",
     "x_vars[3,Headphones]"
    ],
    [
     "C43",
     "x_vars[3,Camera]"
    ],
    [
     "C44",
     "x_vars[3,Smartwatch]"
    ],
    [
     "C45",
     "x_vars[3,Tablet]"
    ],
    [
     "C46",
     "x_vars[3,Bluetooth Speaker]"
    ],
    [
     "C47",
     "x_vars[3,Keyboard]"
    ],
    [
     "C48",
     "x_vars[3,Mouse]"
    ],
    [
     "C49",
     "x_vars[3,Monitor]"
    ],
    [
     "C50",
     "x_vars[3,Printer]"
    ],
    [
     "C51",
     "x_vars[3,External Hard Drive]"
    ],
    [
     "C52",
     "x_vars[3,Router]"
    ],
    [
     "C53",
     "x_vars[3,Power Bank]"
    ],
    [
     "C54",
     "x_vars[3,Memory Card]"
    ],
    [
     "C55",
     "x_vars[3,USB Flash Drive]"
    ],
    [
     "C56",
     "x_vars[3,Smart Home Hub]"
    ],
    [
     "C57",
     "x_vars[3,Gaming Console]"
    ],
    [
     "C58",
     "x_vars[3,Fitness Tracker]"
    ],
    [
     "C59",
     "x_vars[3,E-Reader]"
    ],
    [
     "C60",
     "x_vars[4,Smartphone]"
    ],
    [
     "C61",
     "x_vars[4,Laptop]"
    ],
    [
     "C62",
     "x_vars[4,Headphones]"
    ],
    [
     "C63",
     "x_vars[4,Camera]"
    ],
    [
     "C64",
     "x_vars[4,Smartwatch]"
    ],
    [
     "C65",
     "x_vars[4,Tablet]"
    ],
    [
     "C66",
     "x_vars[4,Bluetooth Speaker]"
    ],
    [
     "C67",
     "x_vars[4,Keyboard]"
    ],
    [
     "C68",
     "x_vars[4,Mouse]"
    ],
    [
     "C69",
     "x_vars[4,Monitor]"
    ],
    [
     "C70",
     "x_vars[4,Printer]"
    ],
    [
     "C71",
     "x_vars[4,External Hard Drive]"
    ],
    [
     "C72",
     "x_vars[4,Router]"
    ],
    [
     "C73",
     "x_vars[4,Power Bank]"
    ],
    [
     "C74",
     "x_vars[4,Memory Card]"
    ],
    [
     "C75",
     "x_vars[4,USB Flash Drive]"
    ],
    [
     "C76",
     "x_vars[4,Smart Home Hub]"
    ],
    [
     "C77",
     "x_vars[4,Gaming Console]"
    ],
    [
     "C78",
     "x_vars[4,Fitness Tracker]"
    ],
    [
     "C79",
     "x_vars[4,E-Reader]"
    ],
    [
     "C80",
     "x_vars[5,Smartphone]"
    ],
    [
     "C81",
     "x_vars[5,Laptop]"
    ],
    [
     "C82",
     "x_vars[5,Headphones]"
    ],
    [
     "C83",
     "x_vars[5,Camera]"
    ],
    [
     "C84",
     "x_vars[5,Smartwatch]"
    ],
    [
     "C85",
     "x_vars[5,Tablet]"
    ],
    [
     "C86",
     "x_vars[5,Bluetooth Speaker]"
    ],
    [
     "C87",
     "x_vars[5,Keyboard]"
    ],
    [
     "C88",
     "x_vars[5,Mouse]"
    ],
    [
     "C89",
     "x_vars[5,Monitor]"
    ],
    [
     "C90",
     "x_vars[5,Printer]"
    ],
    [
     "C91",
     "x_vars[5,External Hard Drive]"
    ],
    [
     "C92",
     "x_vars[5,Router]"
    ],
    [
     "C93",
     "x_vars[5,Power Bank]"
    ],
    [
     "C94",
     "x_vars[5,Memory Card]"
    ],
    [
     "C95",
     "x_vars[5,USB Flash Drive]"
    ],
    [
     "C96",
     "x_vars[5,Smart Home Hub]"
    ],
    [
     "C97",
     "x_vars[5,Gaming Console]"
    ],
    [
     "C98",
     "x_vars[5,Fitness Tracker]"
    ],
    [
     "C99",
     "x_vars[5,E-Reader]"
    ],
    [
     "C100",
     "x_vars[6,Smartphone]"
    ],
    [
     "C101",
     "x_vars[6,Laptop]"
    ],
    [
     "C102",
     "x_vars[6,Headphones]"
    ],
    [
     "C103",
     "x_vars[6,Camera]"
    ],
    [
     "C104",
     "x_vars[6,Smartwatch]"
    ],
    [
     "C105",
     "x_vars[6,Tablet]"
    ],
    [
     "C106",
     "x_vars[6,Bluetooth Speaker]"
    ],
    [
     "C107",
     "x_vars[6,Keyboard]"
    ],
    [
     "C108",
     "x_vars[6,Mouse]"
    ],
    [
     "C109",
     "x_vars[6,Monitor]"
    ],
    [
     "C110",
     "x_vars[6,Printer]"
    ],
    [
     "C111",
     "x_vars[6,External Hard Drive]"
    ],
    [
     "C112",
     "x_vars[6,Router]"
    ],
    [
     "C113",
     "x_vars[6,Power Bank]"
    ],
    [
     "C114",
     "x_vars[6,Memory Card]"
    ],
    [
     "C115",
     "x_vars[6,USB Flash Drive]"
    ],
    [
     "C116",
     "x_vars[6,Smart Home Hub]"
    ],
    [
     "C117",
     "x_vars[6,Gaming Console]"
    ],
    [
     "C118",
     "x_vars[6,Fitness Tracker]"
    ],
    [
     "C119",
     "x_vars[6,E-Reader]"
    ],
    [
     "C120",
     "x_vars[7,Smartphone]"
    ],
    [
     "C121",
     "x_vars[7,Laptop]"
    ],
    [
     "C122",
     "x_vars[7,Headphones]"
    ],
    [
     "C123",
     "x_vars[7,Camera]"
    ],
    [
     "C124",
     "x_vars[7,Smartwatch]"
    ],
    [
     "C125",
     "x_vars[7,Tablet]"
    ],
    [
     "C126",
     "x_vars[7,Bluetooth Speaker]"
    ],
    [
     "C127",
     "x_vars[7,Keyboard]"
    ],
    [
     "C128",
     "x_vars[7,Mouse]"
    ],
    [
     "C129",
     "x_vars[7,Monitor]"
    ],
    [
     "C130",
     "x_vars[7,Printer]"
    ],
    [
     "C131",
     "x_vars[7,External Hard Drive]"
    ],
    [
     "C132",
     "x_vars[7,Router]"
    ],
    [
     "C133",
     "x_vars[7,Power Bank]"
    ],
    [
     "C134",
     "x_vars[7,Memory Card]"
    ],
    [
     "C135",
     "x_vars[7,USB Flash Drive]"
    ],
    [
     "C136",
     "x_vars[7,Smart Home Hub]"
    ],
    [
     "C137",
     "x_vars[7,Gaming Console]"
    ],
    [
     "C138",
     "x_vars[7,Fitness Tracker]"
    ],
    [
     "C139",
     "x_vars[7,E-Reader]"
    ],
    [
     "C140",
     "x_vars[8,Smartphone]"
    ],
    [
     "C141",
     "x_vars[8,Laptop]"
    ],
    [
     "C142",
     "x_vars[8,Headphones]"
    ],
    [
     "C143",
     "x_vars[8,Camera]"
    ],
    [
     "C144",
     "x_vars[8,Smartwatch]"
    ],
    [
     "C145",
     "x_vars[8,Tablet]"
    ],
    [
     "C146",
     "x_vars[8,Bluetooth Speaker]"
    ],
    [
     "C147",
     "x_vars[8,Keyboard]"
    ],
    [
     "C148",
     "x_vars[8,Mouse]"
    ],
    [
     "C149",
     "x_vars[8,Monitor]"
    ],
    [
     "C150",
     "x_vars[8,Printer]"
    ],
    [
     "C151",
     "x_vars[8,External Hard Drive]"
    ],
    [
     "C152",
     "x_vars[8,Router]"
    ],
    [
     "C153",
     "x_vars[8,Power Bank]"
    ],
    [
     "C154",
     "x_vars[8,Memory Card]"
    ],
    [
     "C155",
     "x_vars[8,USB Flash Drive]"
    ],
    [
     "C156",
     "x_vars[8,Smart Home Hub]"
    ],
    [
     "C157",
     "x_vars[8,Gaming Console]"
    ],
    [
     "C158",
     "x_vars[8,Fitness Tracker]"
    ],
    [
     "C159",
     "x_vars[8,E-Reader]"
    ],
    [
     "C160",
     "x_vars[9,Smartphone]"
    ],
    [
     "C161",
     "x_vars[9,Laptop]"
    ],
    [
     "C162",
     "x_vars[9,Headphones]"
    ],
    [
     "C163",
     "x_vars[9,Camera]"
    ],
    [
     "C164",
     "x_vars[9,Smartwatch]"
    ],
    [
     "C165",
     "x_vars[9,Tablet]"
    ],
    [
     "C166",
     "x_vars[9,Bluetooth Speaker]"
    ],
    [
     "C167",
     "x_vars[9,Keyboard]"
    ],
    [
     "C168",
     "x_vars[9,Mouse]"
    ],
    [
     "C169",
     "x_vars[9,Monitor]"
    ],
    [
     "C170",
     "x_vars[9,Printer]"
    ],
    [
     "C171",
     "x_vars[9,External Hard Drive]"
    ],
    [
     "C172",
     "x_vars[9,Router]"
    ],
    [
     "C173",
     "x_vars[9,Power Bank]"
    ],
    [
     "C174",
     "x_vars[9,Memory Card]"
    ],
    [
     "C175",
     "x_vars[9,USB Flash Drive]"
    ],
    [
     "C176",
     "x_vars[9,Smart Home Hub]"
    ],
    [
     "C177",
     "x_vars[9,Gaming Console]"
    ],
    [
     "C178",
     "x_vars[9,Fitness Tracker]"
    ],
    [
     "C179",
     "x_vars[9,E-Reader]"
    ],
    [
     "C180",
     "x_vars[10,Smartphone]"
    ],
    [
     "C181",
     "x_vars[10,Laptop]"
    ],
    [
     "C182",
     "x_vars[10,Headphones]"
    ],
    [
     "C183",
     "x_vars[10,Camera]"
    ],
    [
     "C184",
     "x_vars[10,Smartwatch]"
    ],
    [
     "C185",
     "x_vars[10,Tablet]"
    ],
    [
     "C186",
     "x_vars[10,Bluetooth Speaker]"
    ],
    [
     "C187",
     "x_vars[10,Keyboard]"
    ],
    [
     "C188",
     "x_vars[10,Mouse]"
    ],
    [
     "C189",
     "x_vars[10,Monitor]"
    ],
    [
     "C190",
     "x_vars[10,Printer]"
    ],
    [
     "C191",
     "x_vars[10,External Hard Drive]"
    ],
    [
     "C192",
     "x_vars[10,Router]"
    ],
    [
     "C193",
     "x_vars[10,Power Bank]"
    ],
    [
     "C194",
     "x_vars[10,Memory Card]"
    ],
    [
     "C195",
     "x_vars[10,USB Flash Drive]"
    ],
    [
     "C196",
     "x_vars[10,Smart Home Hub]"
    ],
    [
     "C197",
     "x_vars[10,Gaming Console]"
    ],
    [
     "C198",
     "x_vars[10,Fitness Tracker]"
    ],
    [
     "C199",
     "x_vars[10,E-Reader]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Smartphone]",
    "x_1_0": "x_vars[2,Smartphone]",
    "x_2_0": "x_vars[3,Smartphone]",
    "x_3_0": "x_vars[4,Smartphone]",
    "x_4_0": "x_vars[5,Smartphone]",
    "x_5_0": "x_vars[6,Smartphone]",
    "x_6_0": "x_vars[7,Smartphone]",
    "x_7_0": "x_vars[8,Smartphone]",
    "x_8_0": "x_vars[9,Smartphone]",
    "x_9_0": "x_vars[10,Smartphone]",
    "x_0_1": "x_vars[1,Laptop]",
    "x_1_1": "x_vars[2,Laptop]",
    "x_2_1": "x_vars[3,Laptop]",
    "x_3_1": "x_vars[4,Laptop]",
    "x_4_1": "x_vars[5,Laptop]",
    "x_5_1": "x_vars[6,Laptop]",
    "x_6_1": "x_vars[7,Laptop]",
    "x_7_1": "x_vars[8,Laptop]",
    "x_8_1": "x_vars[9,Laptop]",
    "x_9_1": "x_vars[10,Laptop]",
    "x_0_2": "x_vars[1,Headphones]",
    "x_1_2": "x_vars[2,Headphones]",
    "x_2_2": "x_vars[3,Headphones]",
    "x_3_2": "x_vars[4,Headphones]",
    "x_4_2": "x_vars[5,Headphones]",
    "x_5_2": "x_vars[6,Headphones]",
    "x_6_2": "x_vars[7,Headphones]",
    "x_7_2": "x_vars[8,Headphones]",
    "x_8_2": "x_vars[9,Headphones]",
    "x_9_2": "x_vars[10,Headphones]",
    "x_0_3": "x_vars[1,Camera]",
    "x_1_3": "x_vars[2,Camera]",
    "x_2_3": "x_vars[3,Camera]",
    "x_3_3": "x_vars[4,Camera]",
    "x_4_3": "x_vars[5,Camera]",
    "x_5_3": "x_vars[6,Camera]",
    "x_6_3": "x_vars[7,Camera]",
    "x_7_3": "x_vars[8,Camera]",
    "x_8_3": "x_vars[9,Camera]",
    "x_9_3": "x_vars[10,Camera]",
    "x_0_4": "x_vars[1,Smartwatch]",
    "x_1_4": "x_vars[2,Smartwatch]",
    "x_2_4": "x_vars[3,Smartwatch]",
    "x_3_4": "x_vars[4,Smartwatch]",
    "x_4_4": "x_vars[5,Smartwatch]",
    "x_5_4": "x_vars[6,Smartwatch]",
    "x_6_4": "x_vars[7,Smartwatch]",
    "x_7_4": "x_vars[8,Smartwatch]",
    "x_8_4": "x_vars[9,Smartwatch]",
    "x_9_4": "x_vars[10,Smartwatch]",
    "x_0_5": "x_vars[1,Tablet]",
    "x_1_5": "x_vars[2,Tablet]",
    "x_2_5": "x_vars[3,Tablet]",
    "x_3_5": "x_vars[4,Tablet]",
    "x_4_5": "x_vars[5,Tablet]",
    "x_5_5": "x_vars[6,Tablet]",
    "x_6_5": "x_vars[7,Tablet]",
    "x_7_5": "x_vars[8,Tablet]",
    "x_8_5": "x_vars[9,Tablet]",
    "x_9_5": "x_vars[10,Tablet]",
    "x_0_6": "x_vars[1,Bluetooth Speaker]",
    "x_1_6": "x_vars[2,Bluetooth Speaker]",
    "x_2_6": "x_vars[3,Bluetooth Speaker]",
    "x_3_6": "x_vars[4,Bluetooth Speaker]",
    "x_4_6": "x_vars[5,Bluetooth Speaker]",
    "x_5_6": "x_vars[6,Bluetooth Speaker]",
    "x_6_6": "x_vars[7,Bluetooth Speaker]",
    "x_7_6": "x_vars[8,Bluetooth Speaker]",
    "x_8_6": "x_vars[9,Bluetooth Speaker]",
    "x_9_6": "x_vars[10,Bluetooth Speaker]",
    "x_0_7": "x_vars[1,Keyboard]",
    "x_1_7": "x_vars[2,Keyboard]",
    "x_2_7": "x_vars[3,Keyboard]",
    "x_3_7": "x_vars[4,Keyboard]",
    "x_4_7": "x_vars[5,Keyboard]",
    "x_5_7": "x_vars[6,Keyboard]",
    "x_6_7": "x_vars[7,Keyboard]",
    "x_7_7": "x_vars[8,Keyboard]",
    "x_8_7": "x_vars[9,Keyboard]",
    "x_9_7": "x_vars[10,Keyboard]",
    "x_0_8": "x_vars[1,Mouse]",
    "x_1_8": "x_vars[2,Mouse]",
    "x_2_8": "x_vars[3,Mouse]",
    "x_3_8": "x_vars[4,Mouse]",
    "x_4_8": "x_vars[5,Mouse]",
    "x_5_8": "x_vars[6,Mouse]",
    "x_6_8": "x_vars[7,Mouse]",
    "x_7_8": "x_vars[8,Mouse]",
    "x_8_8": "x_vars[9,Mouse]",
    "x_9_8": "x_vars[10,Mouse]",
    "x_0_9": "x_vars[1,Monitor]",
    "x_1_9": "x_vars[2,Monitor]",
    "x_2_9": "x_vars[3,Monitor]",
    "x_3_9": "x_vars[4,Monitor]",
    "x_4_9": "x_vars[5,Monitor]",
    "x_5_9": "x_vars[6,Monitor]",
    "x_6_9": "x_vars[7,Monitor]",
    "x_7_9": "x_vars[8,Monitor]",
    "x_8_9": "x_vars[9,Monitor]",
    "x_9_9": "x_vars[10,Monitor]",
    "x_0_10": "x_vars[1,Printer]",
    "x_1_10": "x_vars[2,Printer]",
    "x_2_10": "x_vars[3,Printer]",
    "x_3_10": "x_vars[4,Printer]",
    "x_4_10": "x_vars[5,Printer]",
    "x_5_10": "x_vars[6,Printer]",
    "x_6_10": "x_vars[7,Printer]",
    "x_7_10": "x_vars[8,Printer]",
    "x_8_10": "x_vars[9,Printer]",
    "x_9_10": "x_vars[10,Printer]",
    "x_0_11": "x_vars[1,External Hard Drive]",
    "x_1_11": "x_vars[2,External Hard Drive]",
    "x_2_11": "x_vars[3,External Hard Drive]",
    "x_3_11": "x_vars[4,External Hard Drive]",
    "x_4_11": "x_vars[5,External Hard Drive]",
    "x_5_11": "x_vars[6,External Hard Drive]",
    "x_6_11": "x_vars[7,External Hard Drive]",
    "x_7_11": "x_vars[8,External Hard Drive]",
    "x_8_11": "x_vars[9,External Hard Drive]",
    "x_9_11": "x_vars[10,External Hard Drive]",
    "x_0_12": "x_vars[1,Router]",
    "x_1_12": "x_vars[2,Router]",
    "x_2_12": "x_vars[3,Router]",
    "x_3_12": "x_vars[4,Router]",
    "x_4_12": "x_vars[5,Router]",
    "x_5_12": "x_vars[6,Router]",
    "x_6_12": "x_vars[7,Router]",
    "x_7_12": "x_vars[8,Router]",
    "x_8_12": "x_vars[9,Router]",
    "x_9_12": "x_vars[10,Router]",
    "x_0_13": "x_vars[1,Power Bank]",
    "x_1_13": "x_vars[2,Power Bank]",
    "x_2_13": "x_vars[3,Power Bank]",
    "x_3_13": "x_vars[4,Power Bank]",
    "x_4_13": "x_vars[5,Power Bank]",
    "x_5_13": "x_vars[6,Power Bank]",
    "x_6_13": "x_vars[7,Power Bank]",
    "x_7_13": "x_vars[8,Power Bank]",
    "x_8_13": "x_vars[9,Power Bank]",
    "x_9_13": "x_vars[10,Power Bank]",
    "x_0_14": "x_vars[1,Memory Card]",
    "x_1_14": "x_vars[2,Memory Card]",
    "x_2_14": "x_vars[3,Memory Card]",
    "x_3_14": "x_vars[4,Memory Card]",
    "x_4_14": "x_vars[5,Memory Card]",
    "x_5_14": "x_vars[6,Memory Card]",
    "x_6_14": "x_vars[7,Memory Card]",
    "x_7_14": "x_vars[8,Memory Card]",
    "x_8_14": "x_vars[9,Memory Card]",
    "x_9_14": "x_vars[10,Memory Card]",
    "x_0_15": "x_vars[1,USB Flash Drive]",
    "x_1_15": "x_vars[2,USB Flash Drive]",
    "x_2_15": "x_vars[3,USB Flash Drive]",
    "x_3_15": "x_vars[4,USB Flash Drive]",
    "x_4_15": "x_vars[5,USB Flash Drive]",
    "x_5_15": "x_vars[6,USB Flash Drive]",
    "x_6_15": "x_vars[7,USB Flash Drive]",
    "x_7_15": "x_vars[8,USB Flash Drive]",
    "x_8_15": "x_vars[9,USB Flash Drive]",
    "x_9_15": "x_vars[10,USB Flash Drive]",
    "x_0_16": "x_vars[1,Smart Home Hub]",
    "x_1_16": "x_vars[2,Smart Home Hub]",
    "x_2_16": "x_vars[3,Smart Home Hub]",
    "x_3_16": "x_vars[4,Smart Home Hub]",
    "x_4_16": "x_vars[5,Smart Home Hub]",
    "x_5_16": "x_vars[6,Smart Home Hub]",
    "x_6_16": "x_vars[7,Smart Home Hub]",
    "x_7_16": "x_vars[8,Smart Home Hub]",
    "x_8_16": "x_vars[9,Smart Home Hub]",
    "x_9_16": "x_vars[10,Smart Home Hub]",
    "x_0_17": "x_vars[1,Gaming Console]",
    "x_1_17": "x_vars[2,Gaming Console]",
    "x_2_17": "x_vars[3,Gaming Console]",
    "x_3_17": "x_vars[4,Gaming Console]",
    "x_4_17": "x_vars[5,Gaming Console]",
    "x_5_17": "x_vars[6,Gaming Console]",
    "x_6_17": "x_vars[7,Gaming Console]",
    "x_7_17": "x_vars[8,Gaming Console]",
    "x_8_17": "x_vars[9,Gaming Console]",
    "x_9_17": "x_vars[10,Gaming Console]",
    "x_0_18": "x_vars[1,Fitness Tracker]",
    "x_1_18": "x_vars[2,Fitness Tracker]",
    "x_2_18": "x_vars[3,Fitness Tracker]",
    "x_3_18": "x_vars[4,Fitness Tracker]",
    "x_4_18": "x_vars[5,Fitness Tracker]",
    "x_5_18": "x_vars[6,Fitness Tracker]",
    "x_6_18": "x_vars[7,Fitness Tracker]",
    "x_7_18": "x_vars[8,Fitness Tracker]",
    "x_8_18": "x_vars[9,Fitness Tracker]",
    "x_9_18": "x_vars[10,Fitness Tracker]",
    "x_0_19": "x_vars[1,E-Reader]",
    "x_1_19": "x_vars[2,E-Reader]",
    "x_2_19": "x_vars[3,E-Reader]",
    "x_3_19": "x_vars[4,E-Reader]",
    "x_4_19": "x_vars[5,E-Reader]",
    "x_5_19": "x_vars[6,E-Reader]",
    "x_6_19": "x_vars[7,E-Reader]",
    "x_7_19": "x_vars[8,E-Reader]",
    "x_8_19": "x_vars[9,E-Reader]",
    "x_9_19": "x_vars[10,E-Reader]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-051": {
   "source_code_sha256": "221c0f9612edfa3df95a800b08de7f269582de0e3edb94e94278c3c84319eff1",
   "original_solution_sha256": "d409487192057476c48b668d5d60ef7ac7a9f285eaab11a48b88c135ec76b999",
   "bindings": [
    [
     "C0",
     "x_vars[1,Espresso Beans]"
    ],
    [
     "C1",
     "x_vars[1,Colombian Roast]"
    ],
    [
     "C2",
     "x_vars[1,Arabica Blend]"
    ],
    [
     "C3",
     "x_vars[1,French Roast]"
    ],
    [
     "C4",
     "x_vars[1,Italian Roast]"
    ],
    [
     "C5",
     "x_vars[1,House Blend]"
    ],
    [
     "C6",
     "x_vars[1,Sumatra Coffee]"
    ],
    [
     "C7",
     "x_vars[1,Mocha Java]"
    ],
    [
     "C8",
     "x_vars[1,Hazelnut Flavor]"
    ],
    [
     "C9",
     "x_vars[1,Caramel Blend]"
    ],
    [
     "C10",
     "x_vars[1,Vanilla Flavor]"
    ],
    [
     "C11",
     "x_vars[1,Cappuccino Mix]"
    ],
    [
     "C12",
     "x_vars[1,Pumpkin Spice]"
    ],
    [
     "C13",
     "x_vars[1,Decaf Roast]"
    ],
    [
     "C14",
     "x_vars[1,Organic Roast]"
    ],
    [
     "C15",
     "x_vars[1,Cold Brew]"
    ],
    [
     "C16",
     "x_vars[1,Peruvian Blend]"
    ],
    [
     "C17",
     "x_vars[1,Kenyan AA]"
    ],
    [
     "C18",
     "x_vars[2,Espresso Beans]"
    ],
    [
     "C19",
     "x_vars[2,Colombian Roast]"
    ],
    [
     "C20",
     "x_vars[2,Arabica Blend]"
    ],
    [
     "C21",
     "x_vars[2,French Roast]"
    ],
    [
     "C22",
     "x_vars[2,Italian Roast]"
    ],
    [
     "C23",
     "x_vars[2,House Blend]"
    ],
    [
     "C24",
     "x_vars[2,Sumatra Coffee]"
    ],
    [
     "C25",
     "x_vars[2,Mocha Java]"
    ],
    [
     "C26",
     "x_vars[2,Hazelnut Flavor]"
    ],
    [
     "C27",
     "x_vars[2,Caramel Blend]"
    ],
    [
     "C28",
     "x_vars[2,Vanilla Flavor]"
    ],
    [
     "C29",
     "x_vars[2,Cappuccino Mix]"
    ],
    [
     "C30",
     "x_vars[2,Pumpkin Spice]"
    ],
    [
     "C31",
     "x_vars[2,Decaf Roast]"
    ],
    [
     "C32",
     "x_vars[2,Organic Roast]"
    ],
    [
     "C33",
     "x_vars[2,Cold Brew]"
    ],
    [
     "C34",
     "x_vars[2,Peruvian Blend]"
    ],
    [
     "C35",
     "x_vars[2,Kenyan AA]"
    ],
    [
     "C36",
     "x_vars[3,Espresso Beans]"
    ],
    [
     "C37",
     "x_vars[3,Colombian Roast]"
    ],
    [
     "C38",
     "x_vars[3,Arabica Blend]"
    ],
    [
     "C39",
     "x_vars[3,French Roast]"
    ],
    [
     "C40",
     "x_vars[3,Italian Roast]"
    ],
    [
     "C41",
     "x_vars[3,House Blend]"
    ],
    [
     "C42",
     "x_vars[3,Sumatra Coffee]"
    ],
    [
     "C43",
     "x_vars[3,Mocha Java]"
    ],
    [
     "C44",
     "x_vars[3,Hazelnut Flavor]"
    ],
    [
     "C45",
     "x_vars[3,Caramel Blend]"
    ],
    [
     "C46",
     "x_vars[3,Vanilla Flavor]"
    ],
    [
     "C47",
     "x_vars[3,Cappuccino Mix]"
    ],
    [
     "C48",
     "x_vars[3,Pumpkin Spice]"
    ],
    [
     "C49",
     "x_vars[3,Decaf Roast]"
    ],
    [
     "C50",
     "x_vars[3,Organic Roast]"
    ],
    [
     "C51",
     "x_vars[3,Cold Brew]"
    ],
    [
     "C52",
     "x_vars[3,Peruvian Blend]"
    ],
    [
     "C53",
     "x_vars[3,Kenyan AA]"
    ],
    [
     "C54",
     "x_vars[4,Espresso Beans]"
    ],
    [
     "C55",
     "x_vars[4,Colombian Roast]"
    ],
    [
     "C56",
     "x_vars[4,Arabica Blend]"
    ],
    [
     "C57",
     "x_vars[4,French Roast]"
    ],
    [
     "C58",
     "x_vars[4,Italian Roast]"
    ],
    [
     "C59",
     "x_vars[4,House Blend]"
    ],
    [
     "C60",
     "x_vars[4,Sumatra Coffee]"
    ],
    [
     "C61",
     "x_vars[4,Mocha Java]"
    ],
    [
     "C62",
     "x_vars[4,Hazelnut Flavor]"
    ],
    [
     "C63",
     "x_vars[4,Caramel Blend]"
    ],
    [
     "C64",
     "x_vars[4,Vanilla Flavor]"
    ],
    [
     "C65",
     "x_vars[4,Cappuccino Mix]"
    ],
    [
     "C66",
     "x_vars[4,Pumpkin Spice]"
    ],
    [
     "C67",
     "x_vars[4,Decaf Roast]"
    ],
    [
     "C68",
     "x_vars[4,Organic Roast]"
    ],
    [
     "C69",
     "x_vars[4,Cold Brew]"
    ],
    [
     "C70",
     "x_vars[4,Peruvian Blend]"
    ],
    [
     "C71",
     "x_vars[4,Kenyan AA]"
    ],
    [
     "C72",
     "x_vars[5,Espresso Beans]"
    ],
    [
     "C73",
     "x_vars[5,Colombian Roast]"
    ],
    [
     "C74",
     "x_vars[5,Arabica Blend]"
    ],
    [
     "C75",
     "x_vars[5,French Roast]"
    ],
    [
     "C76",
     "x_vars[5,Italian Roast]"
    ],
    [
     "C77",
     "x_vars[5,House Blend]"
    ],
    [
     "C78",
     "x_vars[5,Sumatra Coffee]"
    ],
    [
     "C79",
     "x_vars[5,Mocha Java]"
    ],
    [
     "C80",
     "x_vars[5,Hazelnut Flavor]"
    ],
    [
     "C81",
     "x_vars[5,Caramel Blend]"
    ],
    [
     "C82",
     "x_vars[5,Vanilla Flavor]"
    ],
    [
     "C83",
     "x_vars[5,Cappuccino Mix]"
    ],
    [
     "C84",
     "x_vars[5,Pumpkin Spice]"
    ],
    [
     "C85",
     "x_vars[5,Decaf Roast]"
    ],
    [
     "C86",
     "x_vars[5,Organic Roast]"
    ],
    [
     "C87",
     "x_vars[5,Cold Brew]"
    ],
    [
     "C88",
     "x_vars[5,Peruvian Blend]"
    ],
    [
     "C89",
     "x_vars[5,Kenyan AA]"
    ],
    [
     "C90",
     "x_vars[6,Espresso Beans]"
    ],
    [
     "C91",
     "x_vars[6,Colombian Roast]"
    ],
    [
     "C92",
     "x_vars[6,Arabica Blend]"
    ],
    [
     "C93",
     "x_vars[6,French Roast]"
    ],
    [
     "C94",
     "x_vars[6,Italian Roast]"
    ],
    [
     "C95",
     "x_vars[6,House Blend]"
    ],
    [
     "C96",
     "x_vars[6,Sumatra Coffee]"
    ],
    [
     "C97",
     "x_vars[6,Mocha Java]"
    ],
    [
     "C98",
     "x_vars[6,Hazelnut Flavor]"
    ],
    [
     "C99",
     "x_vars[6,Caramel Blend]"
    ],
    [
     "C100",
     "x_vars[6,Vanilla Flavor]"
    ],
    [
     "C101",
     "x_vars[6,Cappuccino Mix]"
    ],
    [
     "C102",
     "x_vars[6,Pumpkin Spice]"
    ],
    [
     "C103",
     "x_vars[6,Decaf Roast]"
    ],
    [
     "C104",
     "x_vars[6,Organic Roast]"
    ],
    [
     "C105",
     "x_vars[6,Cold Brew]"
    ],
    [
     "C106",
     "x_vars[6,Peruvian Blend]"
    ],
    [
     "C107",
     "x_vars[6,Kenyan AA]"
    ],
    [
     "C108",
     "x_vars[7,Espresso Beans]"
    ],
    [
     "C109",
     "x_vars[7,Colombian Roast]"
    ],
    [
     "C110",
     "x_vars[7,Arabica Blend]"
    ],
    [
     "C111",
     "x_vars[7,French Roast]"
    ],
    [
     "C112",
     "x_vars[7,Italian Roast]"
    ],
    [
     "C113",
     "x_vars[7,House Blend]"
    ],
    [
     "C114",
     "x_vars[7,Sumatra Coffee]"
    ],
    [
     "C115",
     "x_vars[7,Mocha Java]"
    ],
    [
     "C116",
     "x_vars[7,Hazelnut Flavor]"
    ],
    [
     "C117",
     "x_vars[7,Caramel Blend]"
    ],
    [
     "C118",
     "x_vars[7,Vanilla Flavor]"
    ],
    [
     "C119",
     "x_vars[7,Cappuccino Mix]"
    ],
    [
     "C120",
     "x_vars[7,Pumpkin Spice]"
    ],
    [
     "C121",
     "x_vars[7,Decaf Roast]"
    ],
    [
     "C122",
     "x_vars[7,Organic Roast]"
    ],
    [
     "C123",
     "x_vars[7,Cold Brew]"
    ],
    [
     "C124",
     "x_vars[7,Peruvian Blend]"
    ],
    [
     "C125",
     "x_vars[7,Kenyan AA]"
    ],
    [
     "C126",
     "x_vars[8,Espresso Beans]"
    ],
    [
     "C127",
     "x_vars[8,Colombian Roast]"
    ],
    [
     "C128",
     "x_vars[8,Arabica Blend]"
    ],
    [
     "C129",
     "x_vars[8,French Roast]"
    ],
    [
     "C130",
     "x_vars[8,Italian Roast]"
    ],
    [
     "C131",
     "x_vars[8,House Blend]"
    ],
    [
     "C132",
     "x_vars[8,Sumatra Coffee]"
    ],
    [
     "C133",
     "x_vars[8,Mocha Java]"
    ],
    [
     "C134",
     "x_vars[8,Hazelnut Flavor]"
    ],
    [
     "C135",
     "x_vars[8,Caramel Blend]"
    ],
    [
     "C136",
     "x_vars[8,Vanilla Flavor]"
    ],
    [
     "C137",
     "x_vars[8,Cappuccino Mix]"
    ],
    [
     "C138",
     "x_vars[8,Pumpkin Spice]"
    ],
    [
     "C139",
     "x_vars[8,Decaf Roast]"
    ],
    [
     "C140",
     "x_vars[8,Organic Roast]"
    ],
    [
     "C141",
     "x_vars[8,Cold Brew]"
    ],
    [
     "C142",
     "x_vars[8,Peruvian Blend]"
    ],
    [
     "C143",
     "x_vars[8,Kenyan AA]"
    ],
    [
     "C144",
     "x_vars[9,Espresso Beans]"
    ],
    [
     "C145",
     "x_vars[9,Colombian Roast]"
    ],
    [
     "C146",
     "x_vars[9,Arabica Blend]"
    ],
    [
     "C147",
     "x_vars[9,French Roast]"
    ],
    [
     "C148",
     "x_vars[9,Italian Roast]"
    ],
    [
     "C149",
     "x_vars[9,House Blend]"
    ],
    [
     "C150",
     "x_vars[9,Sumatra Coffee]"
    ],
    [
     "C151",
     "x_vars[9,Mocha Java]"
    ],
    [
     "C152",
     "x_vars[9,Hazelnut Flavor]"
    ],
    [
     "C153",
     "x_vars[9,Caramel Blend]"
    ],
    [
     "C154",
     "x_vars[9,Vanilla Flavor]"
    ],
    [
     "C155",
     "x_vars[9,Cappuccino Mix]"
    ],
    [
     "C156",
     "x_vars[9,Pumpkin Spice]"
    ],
    [
     "C157",
     "x_vars[9,Decaf Roast]"
    ],
    [
     "C158",
     "x_vars[9,Organic Roast]"
    ],
    [
     "C159",
     "x_vars[9,Cold Brew]"
    ],
    [
     "C160",
     "x_vars[9,Peruvian Blend]"
    ],
    [
     "C161",
     "x_vars[9,Kenyan AA]"
    ],
    [
     "C162",
     "x_vars[10,Espresso Beans]"
    ],
    [
     "C163",
     "x_vars[10,Colombian Roast]"
    ],
    [
     "C164",
     "x_vars[10,Arabica Blend]"
    ],
    [
     "C165",
     "x_vars[10,French Roast]"
    ],
    [
     "C166",
     "x_vars[10,Italian Roast]"
    ],
    [
     "C167",
     "x_vars[10,House Blend]"
    ],
    [
     "C168",
     "x_vars[10,Sumatra Coffee]"
    ],
    [
     "C169",
     "x_vars[10,Mocha Java]"
    ],
    [
     "C170",
     "x_vars[10,Hazelnut Flavor]"
    ],
    [
     "C171",
     "x_vars[10,Caramel Blend]"
    ],
    [
     "C172",
     "x_vars[10,Vanilla Flavor]"
    ],
    [
     "C173",
     "x_vars[10,Cappuccino Mix]"
    ],
    [
     "C174",
     "x_vars[10,Pumpkin Spice]"
    ],
    [
     "C175",
     "x_vars[10,Decaf Roast]"
    ],
    [
     "C176",
     "x_vars[10,Organic Roast]"
    ],
    [
     "C177",
     "x_vars[10,Cold Brew]"
    ],
    [
     "C178",
     "x_vars[10,Peruvian Blend]"
    ],
    [
     "C179",
     "x_vars[10,Kenyan AA]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Espresso Beans]",
    "x_0_1": "x_vars[2,Espresso Beans]",
    "x_0_2": "x_vars[3,Espresso Beans]",
    "x_0_3": "x_vars[4,Espresso Beans]",
    "x_0_4": "x_vars[5,Espresso Beans]",
    "x_0_5": "x_vars[6,Espresso Beans]",
    "x_0_6": "x_vars[7,Espresso Beans]",
    "x_0_7": "x_vars[8,Espresso Beans]",
    "x_0_8": "x_vars[9,Espresso Beans]",
    "x_0_9": "x_vars[10,Espresso Beans]",
    "x_1_0": "x_vars[1,Colombian Roast]",
    "x_1_1": "x_vars[2,Colombian Roast]",
    "x_1_2": "x_vars[3,Colombian Roast]",
    "x_1_3": "x_vars[4,Colombian Roast]",
    "x_1_4": "x_vars[5,Colombian Roast]",
    "x_1_5": "x_vars[6,Colombian Roast]",
    "x_1_6": "x_vars[7,Colombian Roast]",
    "x_1_7": "x_vars[8,Colombian Roast]",
    "x_1_8": "x_vars[9,Colombian Roast]",
    "x_1_9": "x_vars[10,Colombian Roast]",
    "x_2_0": "x_vars[1,Arabica Blend]",
    "x_2_1": "x_vars[2,Arabica Blend]",
    "x_2_2": "x_vars[3,Arabica Blend]",
    "x_2_3": "x_vars[4,Arabica Blend]",
    "x_2_4": "x_vars[5,Arabica Blend]",
    "x_2_5": "x_vars[6,Arabica Blend]",
    "x_2_6": "x_vars[7,Arabica Blend]",
    "x_2_7": "x_vars[8,Arabica Blend]",
    "x_2_8": "x_vars[9,Arabica Blend]",
    "x_2_9": "x_vars[10,Arabica Blend]",
    "x_3_0": "x_vars[1,French Roast]",
    "x_3_1": "x_vars[2,French Roast]",
    "x_3_2": "x_vars[3,French Roast]",
    "x_3_3": "x_vars[4,French Roast]",
    "x_3_4": "x_vars[5,French Roast]",
    "x_3_5": "x_vars[6,French Roast]",
    "x_3_6": "x_vars[7,French Roast]",
    "x_3_7": "x_vars[8,French Roast]",
    "x_3_8": "x_vars[9,French Roast]",
    "x_3_9": "x_vars[10,French Roast]",
    "x_4_0": "x_vars[1,Italian Roast]",
    "x_4_1": "x_vars[2,Italian Roast]",
    "x_4_2": "x_vars[3,Italian Roast]",
    "x_4_3": "x_vars[4,Italian Roast]",
    "x_4_4": "x_vars[5,Italian Roast]",
    "x_4_5": "x_vars[6,Italian Roast]",
    "x_4_6": "x_vars[7,Italian Roast]",
    "x_4_7": "x_vars[8,Italian Roast]",
    "x_4_8": "x_vars[9,Italian Roast]",
    "x_4_9": "x_vars[10,Italian Roast]",
    "x_5_0": "x_vars[1,House Blend]",
    "x_5_1": "x_vars[2,House Blend]",
    "x_5_2": "x_vars[3,House Blend]",
    "x_5_3": "x_vars[4,House Blend]",
    "x_5_4": "x_vars[5,House Blend]",
    "x_5_5": "x_vars[6,House Blend]",
    "x_5_6": "x_vars[7,House Blend]",
    "x_5_7": "x_vars[8,House Blend]",
    "x_5_8": "x_vars[9,House Blend]",
    "x_5_9": "x_vars[10,House Blend]",
    "x_6_0": "x_vars[1,Sumatra Coffee]",
    "x_6_1": "x_vars[2,Sumatra Coffee]",
    "x_6_2": "x_vars[3,Sumatra Coffee]",
    "x_6_3": "x_vars[4,Sumatra Coffee]",
    "x_6_4": "x_vars[5,Sumatra Coffee]",
    "x_6_5": "x_vars[6,Sumatra Coffee]",
    "x_6_6": "x_vars[7,Sumatra Coffee]",
    "x_6_7": "x_vars[8,Sumatra Coffee]",
    "x_6_8": "x_vars[9,Sumatra Coffee]",
    "x_6_9": "x_vars[10,Sumatra Coffee]",
    "x_7_0": "x_vars[1,Mocha Java]",
    "x_7_1": "x_vars[2,Mocha Java]",
    "x_7_2": "x_vars[3,Mocha Java]",
    "x_7_3": "x_vars[4,Mocha Java]",
    "x_7_4": "x_vars[5,Mocha Java]",
    "x_7_5": "x_vars[6,Mocha Java]",
    "x_7_6": "x_vars[7,Mocha Java]",
    "x_7_7": "x_vars[8,Mocha Java]",
    "x_7_8": "x_vars[9,Mocha Java]",
    "x_7_9": "x_vars[10,Mocha Java]",
    "x_8_0": "x_vars[1,Hazelnut Flavor]",
    "x_8_1": "x_vars[2,Hazelnut Flavor]",
    "x_8_2": "x_vars[3,Hazelnut Flavor]",
    "x_8_3": "x_vars[4,Hazelnut Flavor]",
    "x_8_4": "x_vars[5,Hazelnut Flavor]",
    "x_8_5": "x_vars[6,Hazelnut Flavor]",
    "x_8_6": "x_vars[7,Hazelnut Flavor]",
    "x_8_7": "x_vars[8,Hazelnut Flavor]",
    "x_8_8": "x_vars[9,Hazelnut Flavor]",
    "x_8_9": "x_vars[10,Hazelnut Flavor]",
    "x_9_0": "x_vars[1,Caramel Blend]",
    "x_9_1": "x_vars[2,Caramel Blend]",
    "x_9_2": "x_vars[3,Caramel Blend]",
    "x_9_3": "x_vars[4,Caramel Blend]",
    "x_9_4": "x_vars[5,Caramel Blend]",
    "x_9_5": "x_vars[6,Caramel Blend]",
    "x_9_6": "x_vars[7,Caramel Blend]",
    "x_9_7": "x_vars[8,Caramel Blend]",
    "x_9_8": "x_vars[9,Caramel Blend]",
    "x_9_9": "x_vars[10,Caramel Blend]",
    "x_10_0": "x_vars[1,Vanilla Flavor]",
    "x_10_1": "x_vars[2,Vanilla Flavor]",
    "x_10_2": "x_vars[3,Vanilla Flavor]",
    "x_10_3": "x_vars[4,Vanilla Flavor]",
    "x_10_4": "x_vars[5,Vanilla Flavor]",
    "x_10_5": "x_vars[6,Vanilla Flavor]",
    "x_10_6": "x_vars[7,Vanilla Flavor]",
    "x_10_7": "x_vars[8,Vanilla Flavor]",
    "x_10_8": "x_vars[9,Vanilla Flavor]",
    "x_10_9": "x_vars[10,Vanilla Flavor]",
    "x_11_0": "x_vars[1,Cappuccino Mix]",
    "x_11_1": "x_vars[2,Cappuccino Mix]",
    "x_11_2": "x_vars[3,Cappuccino Mix]",
    "x_11_3": "x_vars[4,Cappuccino Mix]",
    "x_11_4": "x_vars[5,Cappuccino Mix]",
    "x_11_5": "x_vars[6,Cappuccino Mix]",
    "x_11_6": "x_vars[7,Cappuccino Mix]",
    "x_11_7": "x_vars[8,Cappuccino Mix]",
    "x_11_8": "x_vars[9,Cappuccino Mix]",
    "x_11_9": "x_vars[10,Cappuccino Mix]",
    "x_12_0": "x_vars[1,Pumpkin Spice]",
    "x_12_1": "x_vars[2,Pumpkin Spice]",
    "x_12_2": "x_vars[3,Pumpkin Spice]",
    "x_12_3": "x_vars[4,Pumpkin Spice]",
    "x_12_4": "x_vars[5,Pumpkin Spice]",
    "x_12_5": "x_vars[6,Pumpkin Spice]",
    "x_12_6": "x_vars[7,Pumpkin Spice]",
    "x_12_7": "x_vars[8,Pumpkin Spice]",
    "x_12_8": "x_vars[9,Pumpkin Spice]",
    "x_12_9": "x_vars[10,Pumpkin Spice]",
    "x_13_0": "x_vars[1,Decaf Roast]",
    "x_13_1": "x_vars[2,Decaf Roast]",
    "x_13_2": "x_vars[3,Decaf Roast]",
    "x_13_3": "x_vars[4,Decaf Roast]",
    "x_13_4": "x_vars[5,Decaf Roast]",
    "x_13_5": "x_vars[6,Decaf Roast]",
    "x_13_6": "x_vars[7,Decaf Roast]",
    "x_13_7": "x_vars[8,Decaf Roast]",
    "x_13_8": "x_vars[9,Decaf Roast]",
    "x_13_9": "x_vars[10,Decaf Roast]",
    "x_14_0": "x_vars[1,Organic Roast]",
    "x_14_1": "x_vars[2,Organic Roast]",
    "x_14_2": "x_vars[3,Organic Roast]",
    "x_14_3": "x_vars[4,Organic Roast]",
    "x_14_4": "x_vars[5,Organic Roast]",
    "x_14_5": "x_vars[6,Organic Roast]",
    "x_14_6": "x_vars[7,Organic Roast]",
    "x_14_7": "x_vars[8,Organic Roast]",
    "x_14_8": "x_vars[9,Organic Roast]",
    "x_14_9": "x_vars[10,Organic Roast]",
    "x_15_0": "x_vars[1,Cold Brew]",
    "x_15_1": "x_vars[2,Cold Brew]",
    "x_15_2": "x_vars[3,Cold Brew]",
    "x_15_3": "x_vars[4,Cold Brew]",
    "x_15_4": "x_vars[5,Cold Brew]",
    "x_15_5": "x_vars[6,Cold Brew]",
    "x_15_6": "x_vars[7,Cold Brew]",
    "x_15_7": "x_vars[8,Cold Brew]",
    "x_15_8": "x_vars[9,Cold Brew]",
    "x_15_9": "x_vars[10,Cold Brew]",
    "x_16_0": "x_vars[1,Peruvian Blend]",
    "x_16_1": "x_vars[2,Peruvian Blend]",
    "x_16_2": "x_vars[3,Peruvian Blend]",
    "x_16_3": "x_vars[4,Peruvian Blend]",
    "x_16_4": "x_vars[5,Peruvian Blend]",
    "x_16_5": "x_vars[6,Peruvian Blend]",
    "x_16_6": "x_vars[7,Peruvian Blend]",
    "x_16_7": "x_vars[8,Peruvian Blend]",
    "x_16_8": "x_vars[9,Peruvian Blend]",
    "x_16_9": "x_vars[10,Peruvian Blend]",
    "x_17_0": "x_vars[1,Kenyan AA]",
    "x_17_1": "x_vars[2,Kenyan AA]",
    "x_17_2": "x_vars[3,Kenyan AA]",
    "x_17_3": "x_vars[4,Kenyan AA]",
    "x_17_4": "x_vars[5,Kenyan AA]",
    "x_17_5": "x_vars[6,Kenyan AA]",
    "x_17_6": "x_vars[7,Kenyan AA]",
    "x_17_7": "x_vars[8,Kenyan AA]",
    "x_17_8": "x_vars[9,Kenyan AA]",
    "x_17_9": "x_vars[10,Kenyan AA]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-052": {
   "source_code_sha256": "73a40d2c8cf07dcc730905539446e46ce08503cd9d1e270f6bf0d4833dc50c35",
   "original_solution_sha256": "85dfeb28f05861c79314d7ffa365df698153e8a4c2b779e8a2b4e7a4336e5a16",
   "bindings": [
    [
     "C0",
     "x_vars[1,The Great Gatsby]"
    ],
    [
     "C1",
     "x_vars[1,To Kill a Mockingbird]"
    ],
    [
     "C2",
     "x_vars[1,1984]"
    ],
    [
     "C3",
     "x_vars[1,Pride and Prejudice]"
    ],
    [
     "C4",
     "x_vars[1,The Catcher in the Rye]"
    ],
    [
     "C5",
     "x_vars[1,Moby Dick]"
    ],
    [
     "C6",
     "x_vars[1,Jane Eyre]"
    ],
    [
     "C7",
     "x_vars[1,War and Peace]"
    ],
    [
     "C8",
     "x_vars[1,The Odyssey]"
    ],
    [
     "C9",
     "x_vars[1,Crime and Punishment]"
    ],
    [
     "C10",
     "x_vars[1,The Hobbit]"
    ],
    [
     "C11",
     "x_vars[1,Brave New World]"
    ],
    [
     "C12",
     "x_vars[1,Anna Karenina]"
    ],
    [
     "C13",
     "x_vars[1,Wuthering Heights]"
    ],
    [
     "C14",
     "x_vars[1,The Divine Comedy]"
    ],
    [
     "C15",
     "x_vars[1,The Iliad]"
    ],
    [
     "C16",
     "x_vars[1,Les Misérables]"
    ],
    [
     "C17",
     "x_vars[1,Dracula]"
    ],
    [
     "C18",
     "x_vars[1,Frankenstein]"
    ],
    [
     "C19",
     "x_vars[1,The Brothers Karamazov]"
    ],
    [
     "C20",
     "x_vars[1,Don Quixote]"
    ],
    [
     "C21",
     "x_vars[1,One Hundred Years of Solitude]"
    ],
    [
     "C22",
     "x_vars[1,Ulysses]"
    ],
    [
     "C23",
     "x_vars[1,The Alchemist]"
    ],
    [
     "C24",
     "x_vars[1,Meditations]"
    ],
    [
     "C25",
     "x_vars[2,The Great Gatsby]"
    ],
    [
     "C26",
     "x_vars[2,To Kill a Mockingbird]"
    ],
    [
     "C27",
     "x_vars[2,1984]"
    ],
    [
     "C28",
     "x_vars[2,Pride and Prejudice]"
    ],
    [
     "C29",
     "x_vars[2,The Catcher in the Rye]"
    ],
    [
     "C30",
     "x_vars[2,Moby Dick]"
    ],
    [
     "C31",
     "x_vars[2,Jane Eyre]"
    ],
    [
     "C32",
     "x_vars[2,War and Peace]"
    ],
    [
     "C33",
     "x_vars[2,The Odyssey]"
    ],
    [
     "C34",
     "x_vars[2,Crime and Punishment]"
    ],
    [
     "C35",
     "x_vars[2,The Hobbit]"
    ],
    [
     "C36",
     "x_vars[2,Brave New World]"
    ],
    [
     "C37",
     "x_vars[2,Anna Karenina]"
    ],
    [
     "C38",
     "x_vars[2,Wuthering Heights]"
    ],
    [
     "C39",
     "x_vars[2,The Divine Comedy]"
    ],
    [
     "C40",
     "x_vars[2,The Iliad]"
    ],
    [
     "C41",
     "x_vars[2,Les Misérables]"
    ],
    [
     "C42",
     "x_vars[2,Dracula]"
    ],
    [
     "C43",
     "x_vars[2,Frankenstein]"
    ],
    [
     "C44",
     "x_vars[2,The Brothers Karamazov]"
    ],
    [
     "C45",
     "x_vars[2,Don Quixote]"
    ],
    [
     "C46",
     "x_vars[2,One Hundred Years of Solitude]"
    ],
    [
     "C47",
     "x_vars[2,Ulysses]"
    ],
    [
     "C48",
     "x_vars[2,The Alchemist]"
    ],
    [
     "C49",
     "x_vars[2,Meditations]"
    ],
    [
     "C50",
     "x_vars[3,The Great Gatsby]"
    ],
    [
     "C51",
     "x_vars[3,To Kill a Mockingbird]"
    ],
    [
     "C52",
     "x_vars[3,1984]"
    ],
    [
     "C53",
     "x_vars[3,Pride and Prejudice]"
    ],
    [
     "C54",
     "x_vars[3,The Catcher in the Rye]"
    ],
    [
     "C55",
     "x_vars[3,Moby Dick]"
    ],
    [
     "C56",
     "x_vars[3,Jane Eyre]"
    ],
    [
     "C57",
     "x_vars[3,War and Peace]"
    ],
    [
     "C58",
     "x_vars[3,The Odyssey]"
    ],
    [
     "C59",
     "x_vars[3,Crime and Punishment]"
    ],
    [
     "C60",
     "x_vars[3,The Hobbit]"
    ],
    [
     "C61",
     "x_vars[3,Brave New World]"
    ],
    [
     "C62",
     "x_vars[3,Anna Karenina]"
    ],
    [
     "C63",
     "x_vars[3,Wuthering Heights]"
    ],
    [
     "C64",
     "x_vars[3,The Divine Comedy]"
    ],
    [
     "C65",
     "x_vars[3,The Iliad]"
    ],
    [
     "C66",
     "x_vars[3,Les Misérables]"
    ],
    [
     "C67",
     "x_vars[3,Dracula]"
    ],
    [
     "C68",
     "x_vars[3,Frankenstein]"
    ],
    [
     "C69",
     "x_vars[3,The Brothers Karamazov]"
    ],
    [
     "C70",
     "x_vars[3,Don Quixote]"
    ],
    [
     "C71",
     "x_vars[3,One Hundred Years of Solitude]"
    ],
    [
     "C72",
     "x_vars[3,Ulysses]"
    ],
    [
     "C73",
     "x_vars[3,The Alchemist]"
    ],
    [
     "C74",
     "x_vars[3,Meditations]"
    ],
    [
     "C75",
     "x_vars[4,The Great Gatsby]"
    ],
    [
     "C76",
     "x_vars[4,To Kill a Mockingbird]"
    ],
    [
     "C77",
     "x_vars[4,1984]"
    ],
    [
     "C78",
     "x_vars[4,Pride and Prejudice]"
    ],
    [
     "C79",
     "x_vars[4,The Catcher in the Rye]"
    ],
    [
     "C80",
     "x_vars[4,Moby Dick]"
    ],
    [
     "C81",
     "x_vars[4,Jane Eyre]"
    ],
    [
     "C82",
     "x_vars[4,War and Peace]"
    ],
    [
     "C83",
     "x_vars[4,The Odyssey]"
    ],
    [
     "C84",
     "x_vars[4,Crime and Punishment]"
    ],
    [
     "C85",
     "x_vars[4,The Hobbit]"
    ],
    [
     "C86",
     "x_vars[4,Brave New World]"
    ],
    [
     "C87",
     "x_vars[4,Anna Karenina]"
    ],
    [
     "C88",
     "x_vars[4,Wuthering Heights]"
    ],
    [
     "C89",
     "x_vars[4,The Divine Comedy]"
    ],
    [
     "C90",
     "x_vars[4,The Iliad]"
    ],
    [
     "C91",
     "x_vars[4,Les Misérables]"
    ],
    [
     "C92",
     "x_vars[4,Dracula]"
    ],
    [
     "C93",
     "x_vars[4,Frankenstein]"
    ],
    [
     "C94",
     "x_vars[4,The Brothers Karamazov]"
    ],
    [
     "C95",
     "x_vars[4,Don Quixote]"
    ],
    [
     "C96",
     "x_vars[4,One Hundred Years of Solitude]"
    ],
    [
     "C97",
     "x_vars[4,Ulysses]"
    ],
    [
     "C98",
     "x_vars[4,The Alchemist]"
    ],
    [
     "C99",
     "x_vars[4,Meditations]"
    ],
    [
     "C100",
     "x_vars[5,The Great Gatsby]"
    ],
    [
     "C101",
     "x_vars[5,To Kill a Mockingbird]"
    ],
    [
     "C102",
     "x_vars[5,1984]"
    ],
    [
     "C103",
     "x_vars[5,Pride and Prejudice]"
    ],
    [
     "C104",
     "x_vars[5,The Catcher in the Rye]"
    ],
    [
     "C105",
     "x_vars[5,Moby Dick]"
    ],
    [
     "C106",
     "x_vars[5,Jane Eyre]"
    ],
    [
     "C107",
     "x_vars[5,War and Peace]"
    ],
    [
     "C108",
     "x_vars[5,The Odyssey]"
    ],
    [
     "C109",
     "x_vars[5,Crime and Punishment]"
    ],
    [
     "C110",
     "x_vars[5,The Hobbit]"
    ],
    [
     "C111",
     "x_vars[5,Brave New World]"
    ],
    [
     "C112",
     "x_vars[5,Anna Karenina]"
    ],
    [
     "C113",
     "x_vars[5,Wuthering Heights]"
    ],
    [
     "C114",
     "x_vars[5,The Divine Comedy]"
    ],
    [
     "C115",
     "x_vars[5,The Iliad]"
    ],
    [
     "C116",
     "x_vars[5,Les Misérables]"
    ],
    [
     "C117",
     "x_vars[5,Dracula]"
    ],
    [
     "C118",
     "x_vars[5,Frankenstein]"
    ],
    [
     "C119",
     "x_vars[5,The Brothers Karamazov]"
    ],
    [
     "C120",
     "x_vars[5,Don Quixote]"
    ],
    [
     "C121",
     "x_vars[5,One Hundred Years of Solitude]"
    ],
    [
     "C122",
     "x_vars[5,Ulysses]"
    ],
    [
     "C123",
     "x_vars[5,The Alchemist]"
    ],
    [
     "C124",
     "x_vars[5,Meditations]"
    ],
    [
     "C125",
     "x_vars[6,The Great Gatsby]"
    ],
    [
     "C126",
     "x_vars[6,To Kill a Mockingbird]"
    ],
    [
     "C127",
     "x_vars[6,1984]"
    ],
    [
     "C128",
     "x_vars[6,Pride and Prejudice]"
    ],
    [
     "C129",
     "x_vars[6,The Catcher in the Rye]"
    ],
    [
     "C130",
     "x_vars[6,Moby Dick]"
    ],
    [
     "C131",
     "x_vars[6,Jane Eyre]"
    ],
    [
     "C132",
     "x_vars[6,War and Peace]"
    ],
    [
     "C133",
     "x_vars[6,The Odyssey]"
    ],
    [
     "C134",
     "x_vars[6,Crime and Punishment]"
    ],
    [
     "C135",
     "x_vars[6,The Hobbit]"
    ],
    [
     "C136",
     "x_vars[6,Brave New World]"
    ],
    [
     "C137",
     "x_vars[6,Anna Karenina]"
    ],
    [
     "C138",
     "x_vars[6,Wuthering Heights]"
    ],
    [
     "C139",
     "x_vars[6,The Divine Comedy]"
    ],
    [
     "C140",
     "x_vars[6,The Iliad]"
    ],
    [
     "C141",
     "x_vars[6,Les Misérables]"
    ],
    [
     "C142",
     "x_vars[6,Dracula]"
    ],
    [
     "C143",
     "x_vars[6,Frankenstein]"
    ],
    [
     "C144",
     "x_vars[6,The Brothers Karamazov]"
    ],
    [
     "C145",
     "x_vars[6,Don Quixote]"
    ],
    [
     "C146",
     "x_vars[6,One Hundred Years of Solitude]"
    ],
    [
     "C147",
     "x_vars[6,Ulysses]"
    ],
    [
     "C148",
     "x_vars[6,The Alchemist]"
    ],
    [
     "C149",
     "x_vars[6,Meditations]"
    ],
    [
     "C150",
     "x_vars[7,The Great Gatsby]"
    ],
    [
     "C151",
     "x_vars[7,To Kill a Mockingbird]"
    ],
    [
     "C152",
     "x_vars[7,1984]"
    ],
    [
     "C153",
     "x_vars[7,Pride and Prejudice]"
    ],
    [
     "C154",
     "x_vars[7,The Catcher in the Rye]"
    ],
    [
     "C155",
     "x_vars[7,Moby Dick]"
    ],
    [
     "C156",
     "x_vars[7,Jane Eyre]"
    ],
    [
     "C157",
     "x_vars[7,War and Peace]"
    ],
    [
     "C158",
     "x_vars[7,The Odyssey]"
    ],
    [
     "C159",
     "x_vars[7,Crime and Punishment]"
    ],
    [
     "C160",
     "x_vars[7,The Hobbit]"
    ],
    [
     "C161",
     "x_vars[7,Brave New World]"
    ],
    [
     "C162",
     "x_vars[7,Anna Karenina]"
    ],
    [
     "C163",
     "x_vars[7,Wuthering Heights]"
    ],
    [
     "C164",
     "x_vars[7,The Divine Comedy]"
    ],
    [
     "C165",
     "x_vars[7,The Iliad]"
    ],
    [
     "C166",
     "x_vars[7,Les Misérables]"
    ],
    [
     "C167",
     "x_vars[7,Dracula]"
    ],
    [
     "C168",
     "x_vars[7,Frankenstein]"
    ],
    [
     "C169",
     "x_vars[7,The Brothers Karamazov]"
    ],
    [
     "C170",
     "x_vars[7,Don Quixote]"
    ],
    [
     "C171",
     "x_vars[7,One Hundred Years of Solitude]"
    ],
    [
     "C172",
     "x_vars[7,Ulysses]"
    ],
    [
     "C173",
     "x_vars[7,The Alchemist]"
    ],
    [
     "C174",
     "x_vars[7,Meditations]"
    ],
    [
     "C175",
     "x_vars[8,The Great Gatsby]"
    ],
    [
     "C176",
     "x_vars[8,To Kill a Mockingbird]"
    ],
    [
     "C177",
     "x_vars[8,1984]"
    ],
    [
     "C178",
     "x_vars[8,Pride and Prejudice]"
    ],
    [
     "C179",
     "x_vars[8,The Catcher in the Rye]"
    ],
    [
     "C180",
     "x_vars[8,Moby Dick]"
    ],
    [
     "C181",
     "x_vars[8,Jane Eyre]"
    ],
    [
     "C182",
     "x_vars[8,War and Peace]"
    ],
    [
     "C183",
     "x_vars[8,The Odyssey]"
    ],
    [
     "C184",
     "x_vars[8,Crime and Punishment]"
    ],
    [
     "C185",
     "x_vars[8,The Hobbit]"
    ],
    [
     "C186",
     "x_vars[8,Brave New World]"
    ],
    [
     "C187",
     "x_vars[8,Anna Karenina]"
    ],
    [
     "C188",
     "x_vars[8,Wuthering Heights]"
    ],
    [
     "C189",
     "x_vars[8,The Divine Comedy]"
    ],
    [
     "C190",
     "x_vars[8,The Iliad]"
    ],
    [
     "C191",
     "x_vars[8,Les Misérables]"
    ],
    [
     "C192",
     "x_vars[8,Dracula]"
    ],
    [
     "C193",
     "x_vars[8,Frankenstein]"
    ],
    [
     "C194",
     "x_vars[8,The Brothers Karamazov]"
    ],
    [
     "C195",
     "x_vars[8,Don Quixote]"
    ],
    [
     "C196",
     "x_vars[8,One Hundred Years of Solitude]"
    ],
    [
     "C197",
     "x_vars[8,Ulysses]"
    ],
    [
     "C198",
     "x_vars[8,The Alchemist]"
    ],
    [
     "C199",
     "x_vars[8,Meditations]"
    ],
    [
     "C200",
     "x_vars[9,The Great Gatsby]"
    ],
    [
     "C201",
     "x_vars[9,To Kill a Mockingbird]"
    ],
    [
     "C202",
     "x_vars[9,1984]"
    ],
    [
     "C203",
     "x_vars[9,Pride and Prejudice]"
    ],
    [
     "C204",
     "x_vars[9,The Catcher in the Rye]"
    ],
    [
     "C205",
     "x_vars[9,Moby Dick]"
    ],
    [
     "C206",
     "x_vars[9,Jane Eyre]"
    ],
    [
     "C207",
     "x_vars[9,War and Peace]"
    ],
    [
     "C208",
     "x_vars[9,The Odyssey]"
    ],
    [
     "C209",
     "x_vars[9,Crime and Punishment]"
    ],
    [
     "C210",
     "x_vars[9,The Hobbit]"
    ],
    [
     "C211",
     "x_vars[9,Brave New World]"
    ],
    [
     "C212",
     "x_vars[9,Anna Karenina]"
    ],
    [
     "C213",
     "x_vars[9,Wuthering Heights]"
    ],
    [
     "C214",
     "x_vars[9,The Divine Comedy]"
    ],
    [
     "C215",
     "x_vars[9,The Iliad]"
    ],
    [
     "C216",
     "x_vars[9,Les Misérables]"
    ],
    [
     "C217",
     "x_vars[9,Dracula]"
    ],
    [
     "C218",
     "x_vars[9,Frankenstein]"
    ],
    [
     "C219",
     "x_vars[9,The Brothers Karamazov]"
    ],
    [
     "C220",
     "x_vars[9,Don Quixote]"
    ],
    [
     "C221",
     "x_vars[9,One Hundred Years of Solitude]"
    ],
    [
     "C222",
     "x_vars[9,Ulysses]"
    ],
    [
     "C223",
     "x_vars[9,The Alchemist]"
    ],
    [
     "C224",
     "x_vars[9,Meditations]"
    ],
    [
     "C225",
     "x_vars[10,The Great Gatsby]"
    ],
    [
     "C226",
     "x_vars[10,To Kill a Mockingbird]"
    ],
    [
     "C227",
     "x_vars[10,1984]"
    ],
    [
     "C228",
     "x_vars[10,Pride and Prejudice]"
    ],
    [
     "C229",
     "x_vars[10,The Catcher in the Rye]"
    ],
    [
     "C230",
     "x_vars[10,Moby Dick]"
    ],
    [
     "C231",
     "x_vars[10,Jane Eyre]"
    ],
    [
     "C232",
     "x_vars[10,War and Peace]"
    ],
    [
     "C233",
     "x_vars[10,The Odyssey]"
    ],
    [
     "C234",
     "x_vars[10,Crime and Punishment]"
    ],
    [
     "C235",
     "x_vars[10,The Hobbit]"
    ],
    [
     "C236",
     "x_vars[10,Brave New World]"
    ],
    [
     "C237",
     "x_vars[10,Anna Karenina]"
    ],
    [
     "C238",
     "x_vars[10,Wuthering Heights]"
    ],
    [
     "C239",
     "x_vars[10,The Divine Comedy]"
    ],
    [
     "C240",
     "x_vars[10,The Iliad]"
    ],
    [
     "C241",
     "x_vars[10,Les Misérables]"
    ],
    [
     "C242",
     "x_vars[10,Dracula]"
    ],
    [
     "C243",
     "x_vars[10,Frankenstein]"
    ],
    [
     "C244",
     "x_vars[10,The Brothers Karamazov]"
    ],
    [
     "C245",
     "x_vars[10,Don Quixote]"
    ],
    [
     "C246",
     "x_vars[10,One Hundred Years of Solitude]"
    ],
    [
     "C247",
     "x_vars[10,Ulysses]"
    ],
    [
     "C248",
     "x_vars[10,The Alchemist]"
    ],
    [
     "C249",
     "x_vars[10,Meditations]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,The Great Gatsby]",
    "x_0_1": "x_vars[2,The Great Gatsby]",
    "x_0_2": "x_vars[3,The Great Gatsby]",
    "x_0_3": "x_vars[4,The Great Gatsby]",
    "x_0_4": "x_vars[5,The Great Gatsby]",
    "x_0_5": "x_vars[6,The Great Gatsby]",
    "x_0_6": "x_vars[7,The Great Gatsby]",
    "x_0_7": "x_vars[8,The Great Gatsby]",
    "x_0_8": "x_vars[9,The Great Gatsby]",
    "x_0_9": "x_vars[10,The Great Gatsby]",
    "x_1_0": "x_vars[1,To Kill a Mockingbird]",
    "x_1_1": "x_vars[2,To Kill a Mockingbird]",
    "x_1_2": "x_vars[3,To Kill a Mockingbird]",
    "x_1_3": "x_vars[4,To Kill a Mockingbird]",
    "x_1_4": "x_vars[5,To Kill a Mockingbird]",
    "x_1_5": "x_vars[6,To Kill a Mockingbird]",
    "x_1_6": "x_vars[7,To Kill a Mockingbird]",
    "x_1_7": "x_vars[8,To Kill a Mockingbird]",
    "x_1_8": "x_vars[9,To Kill a Mockingbird]",
    "x_1_9": "x_vars[10,To Kill a Mockingbird]",
    "x_2_0": "x_vars[1,1984]",
    "x_2_1": "x_vars[2,1984]",
    "x_2_2": "x_vars[3,1984]",
    "x_2_3": "x_vars[4,1984]",
    "x_2_4": "x_vars[5,1984]",
    "x_2_5": "x_vars[6,1984]",
    "x_2_6": "x_vars[7,1984]",
    "x_2_7": "x_vars[8,1984]",
    "x_2_8": "x_vars[9,1984]",
    "x_2_9": "x_vars[10,1984]",
    "x_3_0": "x_vars[1,Pride and Prejudice]",
    "x_3_1": "x_vars[2,Pride and Prejudice]",
    "x_3_2": "x_vars[3,Pride and Prejudice]",
    "x_3_3": "x_vars[4,Pride and Prejudice]",
    "x_3_4": "x_vars[5,Pride and Prejudice]",
    "x_3_5": "x_vars[6,Pride and Prejudice]",
    "x_3_6": "x_vars[7,Pride and Prejudice]",
    "x_3_7": "x_vars[8,Pride and Prejudice]",
    "x_3_8": "x_vars[9,Pride and Prejudice]",
    "x_3_9": "x_vars[10,Pride and Prejudice]",
    "x_4_0": "x_vars[1,The Catcher in the Rye]",
    "x_4_1": "x_vars[2,The Catcher in the Rye]",
    "x_4_2": "x_vars[3,The Catcher in the Rye]",
    "x_4_3": "x_vars[4,The Catcher in the Rye]",
    "x_4_4": "x_vars[5,The Catcher in the Rye]",
    "x_4_5": "x_vars[6,The Catcher in the Rye]",
    "x_4_6": "x_vars[7,The Catcher in the Rye]",
    "x_4_7": "x_vars[8,The Catcher in the Rye]",
    "x_4_8": "x_vars[9,The Catcher in the Rye]",
    "x_4_9": "x_vars[10,The Catcher in the Rye]",
    "x_5_0": "x_vars[1,Moby Dick]",
    "x_5_1": "x_vars[2,Moby Dick]",
    "x_5_2": "x_vars[3,Moby Dick]",
    "x_5_3": "x_vars[4,Moby Dick]",
    "x_5_4": "x_vars[5,Moby Dick]",
    "x_5_5": "x_vars[6,Moby Dick]",
    "x_5_6": "x_vars[7,Moby Dick]",
    "x_5_7": "x_vars[8,Moby Dick]",
    "x_5_8": "x_vars[9,Moby Dick]",
    "x_5_9": "x_vars[10,Moby Dick]",
    "x_6_0": "x_vars[1,Jane Eyre]",
    "x_6_1": "x_vars[2,Jane Eyre]",
    "x_6_2": "x_vars[3,Jane Eyre]",
    "x_6_3": "x_vars[4,Jane Eyre]",
    "x_6_4": "x_vars[5,Jane Eyre]",
    "x_6_5": "x_vars[6,Jane Eyre]",
    "x_6_6": "x_vars[7,Jane Eyre]",
    "x_6_7": "x_vars[8,Jane Eyre]",
    "x_6_8": "x_vars[9,Jane Eyre]",
    "x_6_9": "x_vars[10,Jane Eyre]",
    "x_7_0": "x_vars[1,War and Peace]",
    "x_7_1": "x_vars[2,War and Peace]",
    "x_7_2": "x_vars[3,War and Peace]",
    "x_7_3": "x_vars[4,War and Peace]",
    "x_7_4": "x_vars[5,War and Peace]",
    "x_7_5": "x_vars[6,War and Peace]",
    "x_7_6": "x_vars[7,War and Peace]",
    "x_7_7": "x_vars[8,War and Peace]",
    "x_7_8": "x_vars[9,War and Peace]",
    "x_7_9": "x_vars[10,War and Peace]",
    "x_8_0": "x_vars[1,The Odyssey]",
    "x_8_1": "x_vars[2,The Odyssey]",
    "x_8_2": "x_vars[3,The Odyssey]",
    "x_8_3": "x_vars[4,The Odyssey]",
    "x_8_4": "x_vars[5,The Odyssey]",
    "x_8_5": "x_vars[6,The Odyssey]",
    "x_8_6": "x_vars[7,The Odyssey]",
    "x_8_7": "x_vars[8,The Odyssey]",
    "x_8_8": "x_vars[9,The Odyssey]",
    "x_8_9": "x_vars[10,The Odyssey]",
    "x_9_0": "x_vars[1,Crime and Punishment]",
    "x_9_1": "x_vars[2,Crime and Punishment]",
    "x_9_2": "x_vars[3,Crime and Punishment]",
    "x_9_3": "x_vars[4,Crime and Punishment]",
    "x_9_4": "x_vars[5,Crime and Punishment]",
    "x_9_5": "x_vars[6,Crime and Punishment]",
    "x_9_6": "x_vars[7,Crime and Punishment]",
    "x_9_7": "x_vars[8,Crime and Punishment]",
    "x_9_8": "x_vars[9,Crime and Punishment]",
    "x_9_9": "x_vars[10,Crime and Punishment]",
    "x_10_0": "x_vars[1,The Hobbit]",
    "x_10_1": "x_vars[2,The Hobbit]",
    "x_10_2": "x_vars[3,The Hobbit]",
    "x_10_3": "x_vars[4,The Hobbit]",
    "x_10_4": "x_vars[5,The Hobbit]",
    "x_10_5": "x_vars[6,The Hobbit]",
    "x_10_6": "x_vars[7,The Hobbit]",
    "x_10_7": "x_vars[8,The Hobbit]",
    "x_10_8": "x_vars[9,The Hobbit]",
    "x_10_9": "x_vars[10,The Hobbit]",
    "x_11_0": "x_vars[1,Brave New World]",
    "x_11_1": "x_vars[2,Brave New World]",
    "x_11_2": "x_vars[3,Brave New World]",
    "x_11_3": "x_vars[4,Brave New World]",
    "x_11_4": "x_vars[5,Brave New World]",
    "x_11_5": "x_vars[6,Brave New World]",
    "x_11_6": "x_vars[7,Brave New World]",
    "x_11_7": "x_vars[8,Brave New World]",
    "x_11_8": "x_vars[9,Brave New World]",
    "x_11_9": "x_vars[10,Brave New World]",
    "x_12_0": "x_vars[1,Anna Karenina]",
    "x_12_1": "x_vars[2,Anna Karenina]",
    "x_12_2": "x_vars[3,Anna Karenina]",
    "x_12_3": "x_vars[4,Anna Karenina]",
    "x_12_4": "x_vars[5,Anna Karenina]",
    "x_12_5": "x_vars[6,Anna Karenina]",
    "x_12_6": "x_vars[7,Anna Karenina]",
    "x_12_7": "x_vars[8,Anna Karenina]",
    "x_12_8": "x_vars[9,Anna Karenina]",
    "x_12_9": "x_vars[10,Anna Karenina]",
    "x_13_0": "x_vars[1,Wuthering Heights]",
    "x_13_1": "x_vars[2,Wuthering Heights]",
    "x_13_2": "x_vars[3,Wuthering Heights]",
    "x_13_3": "x_vars[4,Wuthering Heights]",
    "x_13_4": "x_vars[5,Wuthering Heights]",
    "x_13_5": "x_vars[6,Wuthering Heights]",
    "x_13_6": "x_vars[7,Wuthering Heights]",
    "x_13_7": "x_vars[8,Wuthering Heights]",
    "x_13_8": "x_vars[9,Wuthering Heights]",
    "x_13_9": "x_vars[10,Wuthering Heights]",
    "x_14_0": "x_vars[1,The Divine Comedy]",
    "x_14_1": "x_vars[2,The Divine Comedy]",
    "x_14_2": "x_vars[3,The Divine Comedy]",
    "x_14_3": "x_vars[4,The Divine Comedy]",
    "x_14_4": "x_vars[5,The Divine Comedy]",
    "x_14_5": "x_vars[6,The Divine Comedy]",
    "x_14_6": "x_vars[7,The Divine Comedy]",
    "x_14_7": "x_vars[8,The Divine Comedy]",
    "x_14_8": "x_vars[9,The Divine Comedy]",
    "x_14_9": "x_vars[10,The Divine Comedy]",
    "x_15_0": "x_vars[1,The Iliad]",
    "x_15_1": "x_vars[2,The Iliad]",
    "x_15_2": "x_vars[3,The Iliad]",
    "x_15_3": "x_vars[4,The Iliad]",
    "x_15_4": "x_vars[5,The Iliad]",
    "x_15_5": "x_vars[6,The Iliad]",
    "x_15_6": "x_vars[7,The Iliad]",
    "x_15_7": "x_vars[8,The Iliad]",
    "x_15_8": "x_vars[9,The Iliad]",
    "x_15_9": "x_vars[10,The Iliad]",
    "x_16_0": "x_vars[1,Les Misérables]",
    "x_16_1": "x_vars[2,Les Misérables]",
    "x_16_2": "x_vars[3,Les Misérables]",
    "x_16_3": "x_vars[4,Les Misérables]",
    "x_16_4": "x_vars[5,Les Misérables]",
    "x_16_5": "x_vars[6,Les Misérables]",
    "x_16_6": "x_vars[7,Les Misérables]",
    "x_16_7": "x_vars[8,Les Misérables]",
    "x_16_8": "x_vars[9,Les Misérables]",
    "x_16_9": "x_vars[10,Les Misérables]",
    "x_17_0": "x_vars[1,Dracula]",
    "x_17_1": "x_vars[2,Dracula]",
    "x_17_2": "x_vars[3,Dracula]",
    "x_17_3": "x_vars[4,Dracula]",
    "x_17_4": "x_vars[5,Dracula]",
    "x_17_5": "x_vars[6,Dracula]",
    "x_17_6": "x_vars[7,Dracula]",
    "x_17_7": "x_vars[8,Dracula]",
    "x_17_8": "x_vars[9,Dracula]",
    "x_17_9": "x_vars[10,Dracula]",
    "x_18_0": "x_vars[1,Frankenstein]",
    "x_18_1": "x_vars[2,Frankenstein]",
    "x_18_2": "x_vars[3,Frankenstein]",
    "x_18_3": "x_vars[4,Frankenstein]",
    "x_18_4": "x_vars[5,Frankenstein]",
    "x_18_5": "x_vars[6,Frankenstein]",
    "x_18_6": "x_vars[7,Frankenstein]",
    "x_18_7": "x_vars[8,Frankenstein]",
    "x_18_8": "x_vars[9,Frankenstein]",
    "x_18_9": "x_vars[10,Frankenstein]",
    "x_19_0": "x_vars[1,The Brothers Karamazov]",
    "x_19_1": "x_vars[2,The Brothers Karamazov]",
    "x_19_2": "x_vars[3,The Brothers Karamazov]",
    "x_19_3": "x_vars[4,The Brothers Karamazov]",
    "x_19_4": "x_vars[5,The Brothers Karamazov]",
    "x_19_5": "x_vars[6,The Brothers Karamazov]",
    "x_19_6": "x_vars[7,The Brothers Karamazov]",
    "x_19_7": "x_vars[8,The Brothers Karamazov]",
    "x_19_8": "x_vars[9,The Brothers Karamazov]",
    "x_19_9": "x_vars[10,The Brothers Karamazov]",
    "x_20_0": "x_vars[1,Don Quixote]",
    "x_20_1": "x_vars[2,Don Quixote]",
    "x_20_2": "x_vars[3,Don Quixote]",
    "x_20_3": "x_vars[4,Don Quixote]",
    "x_20_4": "x_vars[5,Don Quixote]",
    "x_20_5": "x_vars[6,Don Quixote]",
    "x_20_6": "x_vars[7,Don Quixote]",
    "x_20_7": "x_vars[8,Don Quixote]",
    "x_20_8": "x_vars[9,Don Quixote]",
    "x_20_9": "x_vars[10,Don Quixote]",
    "x_21_0": "x_vars[1,One Hundred Years of Solitude]",
    "x_21_1": "x_vars[2,One Hundred Years of Solitude]",
    "x_21_2": "x_vars[3,One Hundred Years of Solitude]",
    "x_21_3": "x_vars[4,One Hundred Years of Solitude]",
    "x_21_4": "x_vars[5,One Hundred Years of Solitude]",
    "x_21_5": "x_vars[6,One Hundred Years of Solitude]",
    "x_21_6": "x_vars[7,One Hundred Years of Solitude]",
    "x_21_7": "x_vars[8,One Hundred Years of Solitude]",
    "x_21_8": "x_vars[9,One Hundred Years of Solitude]",
    "x_21_9": "x_vars[10,One Hundred Years of Solitude]",
    "x_22_0": "x_vars[1,Ulysses]",
    "x_22_1": "x_vars[2,Ulysses]",
    "x_22_2": "x_vars[3,Ulysses]",
    "x_22_3": "x_vars[4,Ulysses]",
    "x_22_4": "x_vars[5,Ulysses]",
    "x_22_5": "x_vars[6,Ulysses]",
    "x_22_6": "x_vars[7,Ulysses]",
    "x_22_7": "x_vars[8,Ulysses]",
    "x_22_8": "x_vars[9,Ulysses]",
    "x_22_9": "x_vars[10,Ulysses]",
    "x_23_0": "x_vars[1,The Alchemist]",
    "x_23_1": "x_vars[2,The Alchemist]",
    "x_23_2": "x_vars[3,The Alchemist]",
    "x_23_3": "x_vars[4,The Alchemist]",
    "x_23_4": "x_vars[5,The Alchemist]",
    "x_23_5": "x_vars[6,The Alchemist]",
    "x_23_6": "x_vars[7,The Alchemist]",
    "x_23_7": "x_vars[8,The Alchemist]",
    "x_23_8": "x_vars[9,The Alchemist]",
    "x_23_9": "x_vars[10,The Alchemist]",
    "x_24_0": "x_vars[1,Meditations]",
    "x_24_1": "x_vars[2,Meditations]",
    "x_24_2": "x_vars[3,Meditations]",
    "x_24_3": "x_vars[4,Meditations]",
    "x_24_4": "x_vars[5,Meditations]",
    "x_24_5": "x_vars[6,Meditations]",
    "x_24_6": "x_vars[7,Meditations]",
    "x_24_7": "x_vars[8,Meditations]",
    "x_24_8": "x_vars[9,Meditations]",
    "x_24_9": "x_vars[10,Meditations]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-053": {
   "source_code_sha256": "5054e084e70052be69b65b32ae7fdf0d3e99e143765691261ebef29c315bc0f8",
   "original_solution_sha256": "85b8313391ec386654edd65ce2bbafc051bf7523976cc1e79a8eeafa9757ffe8",
   "bindings": [
    [
     "C0",
     "x_vars[1,1]"
    ],
    [
     "C1",
     "x_vars[1,2]"
    ],
    [
     "C2",
     "x_vars[1,3]"
    ],
    [
     "C3",
     "x_vars[1,4]"
    ],
    [
     "C4",
     "x_vars[1,5]"
    ],
    [
     "C5",
     "x_vars[1,6]"
    ],
    [
     "C6",
     "x_vars[1,7]"
    ],
    [
     "C7",
     "x_vars[1,8]"
    ],
    [
     "C8",
     "x_vars[1,9]"
    ],
    [
     "C9",
     "x_vars[1,10]"
    ],
    [
     "C10",
     "x_vars[1,11]"
    ],
    [
     "C11",
     "x_vars[1,12]"
    ],
    [
     "C12",
     "x_vars[1,13]"
    ],
    [
     "C13",
     "x_vars[1,14]"
    ],
    [
     "C14",
     "x_vars[1,15]"
    ],
    [
     "C15",
     "x_vars[1,16]"
    ],
    [
     "C16",
     "x_vars[1,17]"
    ],
    [
     "C17",
     "x_vars[1,18]"
    ],
    [
     "C18",
     "x_vars[1,19]"
    ],
    [
     "C19",
     "x_vars[1,20]"
    ],
    [
     "C20",
     "x_vars[2,1]"
    ],
    [
     "C21",
     "x_vars[2,2]"
    ],
    [
     "C22",
     "x_vars[2,3]"
    ],
    [
     "C23",
     "x_vars[2,4]"
    ],
    [
     "C24",
     "x_vars[2,5]"
    ],
    [
     "C25",
     "x_vars[2,6]"
    ],
    [
     "C26",
     "x_vars[2,7]"
    ],
    [
     "C27",
     "x_vars[2,8]"
    ],
    [
     "C28",
     "x_vars[2,9]"
    ],
    [
     "C29",
     "x_vars[2,10]"
    ],
    [
     "C30",
     "x_vars[2,11]"
    ],
    [
     "C31",
     "x_vars[2,12]"
    ],
    [
     "C32",
     "x_vars[2,13]"
    ],
    [
     "C33",
     "x_vars[2,14]"
    ],
    [
     "C34",
     "x_vars[2,15]"
    ],
    [
     "C35",
     "x_vars[2,16]"
    ],
    [
     "C36",
     "x_vars[2,17]"
    ],
    [
     "C37",
     "x_vars[2,18]"
    ],
    [
     "C38",
     "x_vars[2,19]"
    ],
    [
     "C39",
     "x_vars[2,20]"
    ],
    [
     "C40",
     "x_vars[3,1]"
    ],
    [
     "C41",
     "x_vars[3,2]"
    ],
    [
     "C42",
     "x_vars[3,3]"
    ],
    [
     "C43",
     "x_vars[3,4]"
    ],
    [
     "C44",
     "x_vars[3,5]"
    ],
    [
     "C45",
     "x_vars[3,6]"
    ],
    [
     "C46",
     "x_vars[3,7]"
    ],
    [
     "C47",
     "x_vars[3,8]"
    ],
    [
     "C48",
     "x_vars[3,9]"
    ],
    [
     "C49",
     "x_vars[3,10]"
    ],
    [
     "C50",
     "x_vars[3,11]"
    ],
    [
     "C51",
     "x_vars[3,12]"
    ],
    [
     "C52",
     "x_vars[3,13]"
    ],
    [
     "C53",
     "x_vars[3,14]"
    ],
    [
     "C54",
     "x_vars[3,15]"
    ],
    [
     "C55",
     "x_vars[3,16]"
    ],
    [
     "C56",
     "x_vars[3,17]"
    ],
    [
     "C57",
     "x_vars[3,18]"
    ],
    [
     "C58",
     "x_vars[3,19]"
    ],
    [
     "C59",
     "x_vars[3,20]"
    ],
    [
     "C60",
     "x_vars[4,1]"
    ],
    [
     "C61",
     "x_vars[4,2]"
    ],
    [
     "C62",
     "x_vars[4,3]"
    ],
    [
     "C63",
     "x_vars[4,4]"
    ],
    [
     "C64",
     "x_vars[4,5]"
    ],
    [
     "C65",
     "x_vars[4,6]"
    ],
    [
     "C66",
     "x_vars[4,7]"
    ],
    [
     "C67",
     "x_vars[4,8]"
    ],
    [
     "C68",
     "x_vars[4,9]"
    ],
    [
     "C69",
     "x_vars[4,10]"
    ],
    [
     "C70",
     "x_vars[4,11]"
    ],
    [
     "C71",
     "x_vars[4,12]"
    ],
    [
     "C72",
     "x_vars[4,13]"
    ],
    [
     "C73",
     "x_vars[4,14]"
    ],
    [
     "C74",
     "x_vars[4,15]"
    ],
    [
     "C75",
     "x_vars[4,16]"
    ],
    [
     "C76",
     "x_vars[4,17]"
    ],
    [
     "C77",
     "x_vars[4,18]"
    ],
    [
     "C78",
     "x_vars[4,19]"
    ],
    [
     "C79",
     "x_vars[4,20]"
    ],
    [
     "C80",
     "x_vars[5,1]"
    ],
    [
     "C81",
     "x_vars[5,2]"
    ],
    [
     "C82",
     "x_vars[5,3]"
    ],
    [
     "C83",
     "x_vars[5,4]"
    ],
    [
     "C84",
     "x_vars[5,5]"
    ],
    [
     "C85",
     "x_vars[5,6]"
    ],
    [
     "C86",
     "x_vars[5,7]"
    ],
    [
     "C87",
     "x_vars[5,8]"
    ],
    [
     "C88",
     "x_vars[5,9]"
    ],
    [
     "C89",
     "x_vars[5,10]"
    ],
    [
     "C90",
     "x_vars[5,11]"
    ],
    [
     "C91",
     "x_vars[5,12]"
    ],
    [
     "C92",
     "x_vars[5,13]"
    ],
    [
     "C93",
     "x_vars[5,14]"
    ],
    [
     "C94",
     "x_vars[5,15]"
    ],
    [
     "C95",
     "x_vars[5,16]"
    ],
    [
     "C96",
     "x_vars[5,17]"
    ],
    [
     "C97",
     "x_vars[5,18]"
    ],
    [
     "C98",
     "x_vars[5,19]"
    ],
    [
     "C99",
     "x_vars[5,20]"
    ],
    [
     "C100",
     "x_vars[6,1]"
    ],
    [
     "C101",
     "x_vars[6,2]"
    ],
    [
     "C102",
     "x_vars[6,3]"
    ],
    [
     "C103",
     "x_vars[6,4]"
    ],
    [
     "C104",
     "x_vars[6,5]"
    ],
    [
     "C105",
     "x_vars[6,6]"
    ],
    [
     "C106",
     "x_vars[6,7]"
    ],
    [
     "C107",
     "x_vars[6,8]"
    ],
    [
     "C108",
     "x_vars[6,9]"
    ],
    [
     "C109",
     "x_vars[6,10]"
    ],
    [
     "C110",
     "x_vars[6,11]"
    ],
    [
     "C111",
     "x_vars[6,12]"
    ],
    [
     "C112",
     "x_vars[6,13]"
    ],
    [
     "C113",
     "x_vars[6,14]"
    ],
    [
     "C114",
     "x_vars[6,15]"
    ],
    [
     "C115",
     "x_vars[6,16]"
    ],
    [
     "C116",
     "x_vars[6,17]"
    ],
    [
     "C117",
     "x_vars[6,18]"
    ],
    [
     "C118",
     "x_vars[6,19]"
    ],
    [
     "C119",
     "x_vars[6,20]"
    ],
    [
     "C120",
     "x_vars[7,1]"
    ],
    [
     "C121",
     "x_vars[7,2]"
    ],
    [
     "C122",
     "x_vars[7,3]"
    ],
    [
     "C123",
     "x_vars[7,4]"
    ],
    [
     "C124",
     "x_vars[7,5]"
    ],
    [
     "C125",
     "x_vars[7,6]"
    ],
    [
     "C126",
     "x_vars[7,7]"
    ],
    [
     "C127",
     "x_vars[7,8]"
    ],
    [
     "C128",
     "x_vars[7,9]"
    ],
    [
     "C129",
     "x_vars[7,10]"
    ],
    [
     "C130",
     "x_vars[7,11]"
    ],
    [
     "C131",
     "x_vars[7,12]"
    ],
    [
     "C132",
     "x_vars[7,13]"
    ],
    [
     "C133",
     "x_vars[7,14]"
    ],
    [
     "C134",
     "x_vars[7,15]"
    ],
    [
     "C135",
     "x_vars[7,16]"
    ],
    [
     "C136",
     "x_vars[7,17]"
    ],
    [
     "C137",
     "x_vars[7,18]"
    ],
    [
     "C138",
     "x_vars[7,19]"
    ],
    [
     "C139",
     "x_vars[7,20]"
    ],
    [
     "C140",
     "x_vars[8,1]"
    ],
    [
     "C141",
     "x_vars[8,2]"
    ],
    [
     "C142",
     "x_vars[8,3]"
    ],
    [
     "C143",
     "x_vars[8,4]"
    ],
    [
     "C144",
     "x_vars[8,5]"
    ],
    [
     "C145",
     "x_vars[8,6]"
    ],
    [
     "C146",
     "x_vars[8,7]"
    ],
    [
     "C147",
     "x_vars[8,8]"
    ],
    [
     "C148",
     "x_vars[8,9]"
    ],
    [
     "C149",
     "x_vars[8,10]"
    ],
    [
     "C150",
     "x_vars[8,11]"
    ],
    [
     "C151",
     "x_vars[8,12]"
    ],
    [
     "C152",
     "x_vars[8,13]"
    ],
    [
     "C153",
     "x_vars[8,14]"
    ],
    [
     "C154",
     "x_vars[8,15]"
    ],
    [
     "C155",
     "x_vars[8,16]"
    ],
    [
     "C156",
     "x_vars[8,17]"
    ],
    [
     "C157",
     "x_vars[8,18]"
    ],
    [
     "C158",
     "x_vars[8,19]"
    ],
    [
     "C159",
     "x_vars[8,20]"
    ],
    [
     "C160",
     "x_vars[9,1]"
    ],
    [
     "C161",
     "x_vars[9,2]"
    ],
    [
     "C162",
     "x_vars[9,3]"
    ],
    [
     "C163",
     "x_vars[9,4]"
    ],
    [
     "C164",
     "x_vars[9,5]"
    ],
    [
     "C165",
     "x_vars[9,6]"
    ],
    [
     "C166",
     "x_vars[9,7]"
    ],
    [
     "C167",
     "x_vars[9,8]"
    ],
    [
     "C168",
     "x_vars[9,9]"
    ],
    [
     "C169",
     "x_vars[9,10]"
    ],
    [
     "C170",
     "x_vars[9,11]"
    ],
    [
     "C171",
     "x_vars[9,12]"
    ],
    [
     "C172",
     "x_vars[9,13]"
    ],
    [
     "C173",
     "x_vars[9,14]"
    ],
    [
     "C174",
     "x_vars[9,15]"
    ],
    [
     "C175",
     "x_vars[9,16]"
    ],
    [
     "C176",
     "x_vars[9,17]"
    ],
    [
     "C177",
     "x_vars[9,18]"
    ],
    [
     "C178",
     "x_vars[9,19]"
    ],
    [
     "C179",
     "x_vars[9,20]"
    ],
    [
     "C180",
     "x_vars[10,1]"
    ],
    [
     "C181",
     "x_vars[10,2]"
    ],
    [
     "C182",
     "x_vars[10,3]"
    ],
    [
     "C183",
     "x_vars[10,4]"
    ],
    [
     "C184",
     "x_vars[10,5]"
    ],
    [
     "C185",
     "x_vars[10,6]"
    ],
    [
     "C186",
     "x_vars[10,7]"
    ],
    [
     "C187",
     "x_vars[10,8]"
    ],
    [
     "C188",
     "x_vars[10,9]"
    ],
    [
     "C189",
     "x_vars[10,10]"
    ],
    [
     "C190",
     "x_vars[10,11]"
    ],
    [
     "C191",
     "x_vars[10,12]"
    ],
    [
     "C192",
     "x_vars[10,13]"
    ],
    [
     "C193",
     "x_vars[10,14]"
    ],
    [
     "C194",
     "x_vars[10,15]"
    ],
    [
     "C195",
     "x_vars[10,16]"
    ],
    [
     "C196",
     "x_vars[10,17]"
    ],
    [
     "C197",
     "x_vars[10,18]"
    ],
    [
     "C198",
     "x_vars[10,19]"
    ],
    [
     "C199",
     "x_vars[10,20]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,1]",
    "x_0_1": "x_vars[2,1]",
    "x_0_2": "x_vars[3,1]",
    "x_0_3": "x_vars[4,1]",
    "x_0_4": "x_vars[5,1]",
    "x_0_5": "x_vars[6,1]",
    "x_0_6": "x_vars[7,1]",
    "x_0_7": "x_vars[8,1]",
    "x_0_8": "x_vars[9,1]",
    "x_0_9": "x_vars[10,1]",
    "x_1_0": "x_vars[1,2]",
    "x_1_1": "x_vars[2,2]",
    "x_1_2": "x_vars[3,2]",
    "x_1_3": "x_vars[4,2]",
    "x_1_4": "x_vars[5,2]",
    "x_1_5": "x_vars[6,2]",
    "x_1_6": "x_vars[7,2]",
    "x_1_7": "x_vars[8,2]",
    "x_1_8": "x_vars[9,2]",
    "x_1_9": "x_vars[10,2]",
    "x_2_0": "x_vars[1,3]",
    "x_2_1": "x_vars[2,3]",
    "x_2_2": "x_vars[3,3]",
    "x_2_3": "x_vars[4,3]",
    "x_2_4": "x_vars[5,3]",
    "x_2_5": "x_vars[6,3]",
    "x_2_6": "x_vars[7,3]",
    "x_2_7": "x_vars[8,3]",
    "x_2_8": "x_vars[9,3]",
    "x_2_9": "x_vars[10,3]",
    "x_3_0": "x_vars[1,4]",
    "x_3_1": "x_vars[2,4]",
    "x_3_2": "x_vars[3,4]",
    "x_3_3": "x_vars[4,4]",
    "x_3_4": "x_vars[5,4]",
    "x_3_5": "x_vars[6,4]",
    "x_3_6": "x_vars[7,4]",
    "x_3_7": "x_vars[8,4]",
    "x_3_8": "x_vars[9,4]",
    "x_3_9": "x_vars[10,4]",
    "x_4_0": "x_vars[1,5]",
    "x_4_1": "x_vars[2,5]",
    "x_4_2": "x_vars[3,5]",
    "x_4_3": "x_vars[4,5]",
    "x_4_4": "x_vars[5,5]",
    "x_4_5": "x_vars[6,5]",
    "x_4_6": "x_vars[7,5]",
    "x_4_7": "x_vars[8,5]",
    "x_4_8": "x_vars[9,5]",
    "x_4_9": "x_vars[10,5]",
    "x_5_0": "x_vars[1,6]",
    "x_5_1": "x_vars[2,6]",
    "x_5_2": "x_vars[3,6]",
    "x_5_3": "x_vars[4,6]",
    "x_5_4": "x_vars[5,6]",
    "x_5_5": "x_vars[6,6]",
    "x_5_6": "x_vars[7,6]",
    "x_5_7": "x_vars[8,6]",
    "x_5_8": "x_vars[9,6]",
    "x_5_9": "x_vars[10,6]",
    "x_6_0": "x_vars[1,7]",
    "x_6_1": "x_vars[2,7]",
    "x_6_2": "x_vars[3,7]",
    "x_6_3": "x_vars[4,7]",
    "x_6_4": "x_vars[5,7]",
    "x_6_5": "x_vars[6,7]",
    "x_6_6": "x_vars[7,7]",
    "x_6_7": "x_vars[8,7]",
    "x_6_8": "x_vars[9,7]",
    "x_6_9": "x_vars[10,7]",
    "x_7_0": "x_vars[1,8]",
    "x_7_1": "x_vars[2,8]",
    "x_7_2": "x_vars[3,8]",
    "x_7_3": "x_vars[4,8]",
    "x_7_4": "x_vars[5,8]",
    "x_7_5": "x_vars[6,8]",
    "x_7_6": "x_vars[7,8]",
    "x_7_7": "x_vars[8,8]",
    "x_7_8": "x_vars[9,8]",
    "x_7_9": "x_vars[10,8]",
    "x_8_0": "x_vars[1,9]",
    "x_8_1": "x_vars[2,9]",
    "x_8_2": "x_vars[3,9]",
    "x_8_3": "x_vars[4,9]",
    "x_8_4": "x_vars[5,9]",
    "x_8_5": "x_vars[6,9]",
    "x_8_6": "x_vars[7,9]",
    "x_8_7": "x_vars[8,9]",
    "x_8_8": "x_vars[9,9]",
    "x_8_9": "x_vars[10,9]",
    "x_9_0": "x_vars[1,10]",
    "x_9_1": "x_vars[2,10]",
    "x_9_2": "x_vars[3,10]",
    "x_9_3": "x_vars[4,10]",
    "x_9_4": "x_vars[5,10]",
    "x_9_5": "x_vars[6,10]",
    "x_9_6": "x_vars[7,10]",
    "x_9_7": "x_vars[8,10]",
    "x_9_8": "x_vars[9,10]",
    "x_9_9": "x_vars[10,10]",
    "x_10_0": "x_vars[1,11]",
    "x_10_1": "x_vars[2,11]",
    "x_10_2": "x_vars[3,11]",
    "x_10_3": "x_vars[4,11]",
    "x_10_4": "x_vars[5,11]",
    "x_10_5": "x_vars[6,11]",
    "x_10_6": "x_vars[7,11]",
    "x_10_7": "x_vars[8,11]",
    "x_10_8": "x_vars[9,11]",
    "x_10_9": "x_vars[10,11]",
    "x_11_0": "x_vars[1,12]",
    "x_11_1": "x_vars[2,12]",
    "x_11_2": "x_vars[3,12]",
    "x_11_3": "x_vars[4,12]",
    "x_11_4": "x_vars[5,12]",
    "x_11_5": "x_vars[6,12]",
    "x_11_6": "x_vars[7,12]",
    "x_11_7": "x_vars[8,12]",
    "x_11_8": "x_vars[9,12]",
    "x_11_9": "x_vars[10,12]",
    "x_12_0": "x_vars[1,13]",
    "x_12_1": "x_vars[2,13]",
    "x_12_2": "x_vars[3,13]",
    "x_12_3": "x_vars[4,13]",
    "x_12_4": "x_vars[5,13]",
    "x_12_5": "x_vars[6,13]",
    "x_12_6": "x_vars[7,13]",
    "x_12_7": "x_vars[8,13]",
    "x_12_8": "x_vars[9,13]",
    "x_12_9": "x_vars[10,13]",
    "x_13_0": "x_vars[1,14]",
    "x_13_1": "x_vars[2,14]",
    "x_13_2": "x_vars[3,14]",
    "x_13_3": "x_vars[4,14]",
    "x_13_4": "x_vars[5,14]",
    "x_13_5": "x_vars[6,14]",
    "x_13_6": "x_vars[7,14]",
    "x_13_7": "x_vars[8,14]",
    "x_13_8": "x_vars[9,14]",
    "x_13_9": "x_vars[10,14]",
    "x_14_0": "x_vars[1,15]",
    "x_14_1": "x_vars[2,15]",
    "x_14_2": "x_vars[3,15]",
    "x_14_3": "x_vars[4,15]",
    "x_14_4": "x_vars[5,15]",
    "x_14_5": "x_vars[6,15]",
    "x_14_6": "x_vars[7,15]",
    "x_14_7": "x_vars[8,15]",
    "x_14_8": "x_vars[9,15]",
    "x_14_9": "x_vars[10,15]",
    "x_15_0": "x_vars[1,16]",
    "x_15_1": "x_vars[2,16]",
    "x_15_2": "x_vars[3,16]",
    "x_15_3": "x_vars[4,16]",
    "x_15_4": "x_vars[5,16]",
    "x_15_5": "x_vars[6,16]",
    "x_15_6": "x_vars[7,16]",
    "x_15_7": "x_vars[8,16]",
    "x_15_8": "x_vars[9,16]",
    "x_15_9": "x_vars[10,16]",
    "x_16_0": "x_vars[1,17]",
    "x_16_1": "x_vars[2,17]",
    "x_16_2": "x_vars[3,17]",
    "x_16_3": "x_vars[4,17]",
    "x_16_4": "x_vars[5,17]",
    "x_16_5": "x_vars[6,17]",
    "x_16_6": "x_vars[7,17]",
    "x_16_7": "x_vars[8,17]",
    "x_16_8": "x_vars[9,17]",
    "x_16_9": "x_vars[10,17]",
    "x_17_0": "x_vars[1,18]",
    "x_17_1": "x_vars[2,18]",
    "x_17_2": "x_vars[3,18]",
    "x_17_3": "x_vars[4,18]",
    "x_17_4": "x_vars[5,18]",
    "x_17_5": "x_vars[6,18]",
    "x_17_6": "x_vars[7,18]",
    "x_17_7": "x_vars[8,18]",
    "x_17_8": "x_vars[9,18]",
    "x_17_9": "x_vars[10,18]",
    "x_18_0": "x_vars[1,19]",
    "x_18_1": "x_vars[2,19]",
    "x_18_2": "x_vars[3,19]",
    "x_18_3": "x_vars[4,19]",
    "x_18_4": "x_vars[5,19]",
    "x_18_5": "x_vars[6,19]",
    "x_18_6": "x_vars[7,19]",
    "x_18_7": "x_vars[8,19]",
    "x_18_8": "x_vars[9,19]",
    "x_18_9": "x_vars[10,19]",
    "x_19_0": "x_vars[1,20]",
    "x_19_1": "x_vars[2,20]",
    "x_19_2": "x_vars[3,20]",
    "x_19_3": "x_vars[4,20]",
    "x_19_4": "x_vars[5,20]",
    "x_19_5": "x_vars[6,20]",
    "x_19_6": "x_vars[7,20]",
    "x_19_7": "x_vars[8,20]",
    "x_19_8": "x_vars[9,20]",
    "x_19_9": "x_vars[10,20]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-054": {
   "source_code_sha256": "81e0bbfd73b9af932b890e8e41d0c525a802c6ebca0eefb5a588dc1be5c5e240",
   "original_solution_sha256": "763cc8a10acb85fbf69cbc0d68bc505975588bdb7752fae71b76faf7e61c1ba9",
   "bindings": [
    [
     "C0",
     "x_vars[1,1]"
    ],
    [
     "C1",
     "x_vars[1,2]"
    ],
    [
     "C2",
     "x_vars[1,3]"
    ],
    [
     "C3",
     "x_vars[1,4]"
    ],
    [
     "C4",
     "x_vars[1,5]"
    ],
    [
     "C5",
     "x_vars[1,6]"
    ],
    [
     "C6",
     "x_vars[1,7]"
    ],
    [
     "C7",
     "x_vars[1,8]"
    ],
    [
     "C8",
     "x_vars[1,9]"
    ],
    [
     "C9",
     "x_vars[1,10]"
    ],
    [
     "C10",
     "x_vars[1,11]"
    ],
    [
     "C11",
     "x_vars[1,12]"
    ],
    [
     "C12",
     "x_vars[1,13]"
    ],
    [
     "C13",
     "x_vars[1,14]"
    ],
    [
     "C14",
     "x_vars[1,15]"
    ],
    [
     "C15",
     "x_vars[1,16]"
    ],
    [
     "C16",
     "x_vars[1,17]"
    ],
    [
     "C17",
     "x_vars[1,18]"
    ],
    [
     "C18",
     "x_vars[1,19]"
    ],
    [
     "C19",
     "x_vars[1,20]"
    ],
    [
     "C20",
     "x_vars[2,1]"
    ],
    [
     "C21",
     "x_vars[2,2]"
    ],
    [
     "C22",
     "x_vars[2,3]"
    ],
    [
     "C23",
     "x_vars[2,4]"
    ],
    [
     "C24",
     "x_vars[2,5]"
    ],
    [
     "C25",
     "x_vars[2,6]"
    ],
    [
     "C26",
     "x_vars[2,7]"
    ],
    [
     "C27",
     "x_vars[2,8]"
    ],
    [
     "C28",
     "x_vars[2,9]"
    ],
    [
     "C29",
     "x_vars[2,10]"
    ],
    [
     "C30",
     "x_vars[2,11]"
    ],
    [
     "C31",
     "x_vars[2,12]"
    ],
    [
     "C32",
     "x_vars[2,13]"
    ],
    [
     "C33",
     "x_vars[2,14]"
    ],
    [
     "C34",
     "x_vars[2,15]"
    ],
    [
     "C35",
     "x_vars[2,16]"
    ],
    [
     "C36",
     "x_vars[2,17]"
    ],
    [
     "C37",
     "x_vars[2,18]"
    ],
    [
     "C38",
     "x_vars[2,19]"
    ],
    [
     "C39",
     "x_vars[2,20]"
    ],
    [
     "C40",
     "x_vars[3,1]"
    ],
    [
     "C41",
     "x_vars[3,2]"
    ],
    [
     "C42",
     "x_vars[3,3]"
    ],
    [
     "C43",
     "x_vars[3,4]"
    ],
    [
     "C44",
     "x_vars[3,5]"
    ],
    [
     "C45",
     "x_vars[3,6]"
    ],
    [
     "C46",
     "x_vars[3,7]"
    ],
    [
     "C47",
     "x_vars[3,8]"
    ],
    [
     "C48",
     "x_vars[3,9]"
    ],
    [
     "C49",
     "x_vars[3,10]"
    ],
    [
     "C50",
     "x_vars[3,11]"
    ],
    [
     "C51",
     "x_vars[3,12]"
    ],
    [
     "C52",
     "x_vars[3,13]"
    ],
    [
     "C53",
     "x_vars[3,14]"
    ],
    [
     "C54",
     "x_vars[3,15]"
    ],
    [
     "C55",
     "x_vars[3,16]"
    ],
    [
     "C56",
     "x_vars[3,17]"
    ],
    [
     "C57",
     "x_vars[3,18]"
    ],
    [
     "C58",
     "x_vars[3,19]"
    ],
    [
     "C59",
     "x_vars[3,20]"
    ],
    [
     "C60",
     "x_vars[4,1]"
    ],
    [
     "C61",
     "x_vars[4,2]"
    ],
    [
     "C62",
     "x_vars[4,3]"
    ],
    [
     "C63",
     "x_vars[4,4]"
    ],
    [
     "C64",
     "x_vars[4,5]"
    ],
    [
     "C65",
     "x_vars[4,6]"
    ],
    [
     "C66",
     "x_vars[4,7]"
    ],
    [
     "C67",
     "x_vars[4,8]"
    ],
    [
     "C68",
     "x_vars[4,9]"
    ],
    [
     "C69",
     "x_vars[4,10]"
    ],
    [
     "C70",
     "x_vars[4,11]"
    ],
    [
     "C71",
     "x_vars[4,12]"
    ],
    [
     "C72",
     "x_vars[4,13]"
    ],
    [
     "C73",
     "x_vars[4,14]"
    ],
    [
     "C74",
     "x_vars[4,15]"
    ],
    [
     "C75",
     "x_vars[4,16]"
    ],
    [
     "C76",
     "x_vars[4,17]"
    ],
    [
     "C77",
     "x_vars[4,18]"
    ],
    [
     "C78",
     "x_vars[4,19]"
    ],
    [
     "C79",
     "x_vars[4,20]"
    ],
    [
     "C80",
     "x_vars[5,1]"
    ],
    [
     "C81",
     "x_vars[5,2]"
    ],
    [
     "C82",
     "x_vars[5,3]"
    ],
    [
     "C83",
     "x_vars[5,4]"
    ],
    [
     "C84",
     "x_vars[5,5]"
    ],
    [
     "C85",
     "x_vars[5,6]"
    ],
    [
     "C86",
     "x_vars[5,7]"
    ],
    [
     "C87",
     "x_vars[5,8]"
    ],
    [
     "C88",
     "x_vars[5,9]"
    ],
    [
     "C89",
     "x_vars[5,10]"
    ],
    [
     "C90",
     "x_vars[5,11]"
    ],
    [
     "C91",
     "x_vars[5,12]"
    ],
    [
     "C92",
     "x_vars[5,13]"
    ],
    [
     "C93",
     "x_vars[5,14]"
    ],
    [
     "C94",
     "x_vars[5,15]"
    ],
    [
     "C95",
     "x_vars[5,16]"
    ],
    [
     "C96",
     "x_vars[5,17]"
    ],
    [
     "C97",
     "x_vars[5,18]"
    ],
    [
     "C98",
     "x_vars[5,19]"
    ],
    [
     "C99",
     "x_vars[5,20]"
    ],
    [
     "C100",
     "x_vars[6,1]"
    ],
    [
     "C101",
     "x_vars[6,2]"
    ],
    [
     "C102",
     "x_vars[6,3]"
    ],
    [
     "C103",
     "x_vars[6,4]"
    ],
    [
     "C104",
     "x_vars[6,5]"
    ],
    [
     "C105",
     "x_vars[6,6]"
    ],
    [
     "C106",
     "x_vars[6,7]"
    ],
    [
     "C107",
     "x_vars[6,8]"
    ],
    [
     "C108",
     "x_vars[6,9]"
    ],
    [
     "C109",
     "x_vars[6,10]"
    ],
    [
     "C110",
     "x_vars[6,11]"
    ],
    [
     "C111",
     "x_vars[6,12]"
    ],
    [
     "C112",
     "x_vars[6,13]"
    ],
    [
     "C113",
     "x_vars[6,14]"
    ],
    [
     "C114",
     "x_vars[6,15]"
    ],
    [
     "C115",
     "x_vars[6,16]"
    ],
    [
     "C116",
     "x_vars[6,17]"
    ],
    [
     "C117",
     "x_vars[6,18]"
    ],
    [
     "C118",
     "x_vars[6,19]"
    ],
    [
     "C119",
     "x_vars[6,20]"
    ],
    [
     "C120",
     "x_vars[7,1]"
    ],
    [
     "C121",
     "x_vars[7,2]"
    ],
    [
     "C122",
     "x_vars[7,3]"
    ],
    [
     "C123",
     "x_vars[7,4]"
    ],
    [
     "C124",
     "x_vars[7,5]"
    ],
    [
     "C125",
     "x_vars[7,6]"
    ],
    [
     "C126",
     "x_vars[7,7]"
    ],
    [
     "C127",
     "x_vars[7,8]"
    ],
    [
     "C128",
     "x_vars[7,9]"
    ],
    [
     "C129",
     "x_vars[7,10]"
    ],
    [
     "C130",
     "x_vars[7,11]"
    ],
    [
     "C131",
     "x_vars[7,12]"
    ],
    [
     "C132",
     "x_vars[7,13]"
    ],
    [
     "C133",
     "x_vars[7,14]"
    ],
    [
     "C134",
     "x_vars[7,15]"
    ],
    [
     "C135",
     "x_vars[7,16]"
    ],
    [
     "C136",
     "x_vars[7,17]"
    ],
    [
     "C137",
     "x_vars[7,18]"
    ],
    [
     "C138",
     "x_vars[7,19]"
    ],
    [
     "C139",
     "x_vars[7,20]"
    ],
    [
     "C140",
     "x_vars[8,1]"
    ],
    [
     "C141",
     "x_vars[8,2]"
    ],
    [
     "C142",
     "x_vars[8,3]"
    ],
    [
     "C143",
     "x_vars[8,4]"
    ],
    [
     "C144",
     "x_vars[8,5]"
    ],
    [
     "C145",
     "x_vars[8,6]"
    ],
    [
     "C146",
     "x_vars[8,7]"
    ],
    [
     "C147",
     "x_vars[8,8]"
    ],
    [
     "C148",
     "x_vars[8,9]"
    ],
    [
     "C149",
     "x_vars[8,10]"
    ],
    [
     "C150",
     "x_vars[8,11]"
    ],
    [
     "C151",
     "x_vars[8,12]"
    ],
    [
     "C152",
     "x_vars[8,13]"
    ],
    [
     "C153",
     "x_vars[8,14]"
    ],
    [
     "C154",
     "x_vars[8,15]"
    ],
    [
     "C155",
     "x_vars[8,16]"
    ],
    [
     "C156",
     "x_vars[8,17]"
    ],
    [
     "C157",
     "x_vars[8,18]"
    ],
    [
     "C158",
     "x_vars[8,19]"
    ],
    [
     "C159",
     "x_vars[8,20]"
    ],
    [
     "C160",
     "x_vars[9,1]"
    ],
    [
     "C161",
     "x_vars[9,2]"
    ],
    [
     "C162",
     "x_vars[9,3]"
    ],
    [
     "C163",
     "x_vars[9,4]"
    ],
    [
     "C164",
     "x_vars[9,5]"
    ],
    [
     "C165",
     "x_vars[9,6]"
    ],
    [
     "C166",
     "x_vars[9,7]"
    ],
    [
     "C167",
     "x_vars[9,8]"
    ],
    [
     "C168",
     "x_vars[9,9]"
    ],
    [
     "C169",
     "x_vars[9,10]"
    ],
    [
     "C170",
     "x_vars[9,11]"
    ],
    [
     "C171",
     "x_vars[9,12]"
    ],
    [
     "C172",
     "x_vars[9,13]"
    ],
    [
     "C173",
     "x_vars[9,14]"
    ],
    [
     "C174",
     "x_vars[9,15]"
    ],
    [
     "C175",
     "x_vars[9,16]"
    ],
    [
     "C176",
     "x_vars[9,17]"
    ],
    [
     "C177",
     "x_vars[9,18]"
    ],
    [
     "C178",
     "x_vars[9,19]"
    ],
    [
     "C179",
     "x_vars[9,20]"
    ],
    [
     "C180",
     "x_vars[10,1]"
    ],
    [
     "C181",
     "x_vars[10,2]"
    ],
    [
     "C182",
     "x_vars[10,3]"
    ],
    [
     "C183",
     "x_vars[10,4]"
    ],
    [
     "C184",
     "x_vars[10,5]"
    ],
    [
     "C185",
     "x_vars[10,6]"
    ],
    [
     "C186",
     "x_vars[10,7]"
    ],
    [
     "C187",
     "x_vars[10,8]"
    ],
    [
     "C188",
     "x_vars[10,9]"
    ],
    [
     "C189",
     "x_vars[10,10]"
    ],
    [
     "C190",
     "x_vars[10,11]"
    ],
    [
     "C191",
     "x_vars[10,12]"
    ],
    [
     "C192",
     "x_vars[10,13]"
    ],
    [
     "C193",
     "x_vars[10,14]"
    ],
    [
     "C194",
     "x_vars[10,15]"
    ],
    [
     "C195",
     "x_vars[10,16]"
    ],
    [
     "C196",
     "x_vars[10,17]"
    ],
    [
     "C197",
     "x_vars[10,18]"
    ],
    [
     "C198",
     "x_vars[10,19]"
    ],
    [
     "C199",
     "x_vars[10,20]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,1]",
    "x_0_1": "x_vars[2,1]",
    "x_0_2": "x_vars[3,1]",
    "x_0_3": "x_vars[4,1]",
    "x_0_4": "x_vars[5,1]",
    "x_0_5": "x_vars[6,1]",
    "x_0_6": "x_vars[7,1]",
    "x_0_7": "x_vars[8,1]",
    "x_0_8": "x_vars[9,1]",
    "x_0_9": "x_vars[10,1]",
    "x_1_0": "x_vars[1,2]",
    "x_1_1": "x_vars[2,2]",
    "x_1_2": "x_vars[3,2]",
    "x_1_3": "x_vars[4,2]",
    "x_1_4": "x_vars[5,2]",
    "x_1_5": "x_vars[6,2]",
    "x_1_6": "x_vars[7,2]",
    "x_1_7": "x_vars[8,2]",
    "x_1_8": "x_vars[9,2]",
    "x_1_9": "x_vars[10,2]",
    "x_2_0": "x_vars[1,3]",
    "x_2_1": "x_vars[2,3]",
    "x_2_2": "x_vars[3,3]",
    "x_2_3": "x_vars[4,3]",
    "x_2_4": "x_vars[5,3]",
    "x_2_5": "x_vars[6,3]",
    "x_2_6": "x_vars[7,3]",
    "x_2_7": "x_vars[8,3]",
    "x_2_8": "x_vars[9,3]",
    "x_2_9": "x_vars[10,3]",
    "x_3_0": "x_vars[1,4]",
    "x_3_1": "x_vars[2,4]",
    "x_3_2": "x_vars[3,4]",
    "x_3_3": "x_vars[4,4]",
    "x_3_4": "x_vars[5,4]",
    "x_3_5": "x_vars[6,4]",
    "x_3_6": "x_vars[7,4]",
    "x_3_7": "x_vars[8,4]",
    "x_3_8": "x_vars[9,4]",
    "x_3_9": "x_vars[10,4]",
    "x_4_0": "x_vars[1,5]",
    "x_4_1": "x_vars[2,5]",
    "x_4_2": "x_vars[3,5]",
    "x_4_3": "x_vars[4,5]",
    "x_4_4": "x_vars[5,5]",
    "x_4_5": "x_vars[6,5]",
    "x_4_6": "x_vars[7,5]",
    "x_4_7": "x_vars[8,5]",
    "x_4_8": "x_vars[9,5]",
    "x_4_9": "x_vars[10,5]",
    "x_5_0": "x_vars[1,6]",
    "x_5_1": "x_vars[2,6]",
    "x_5_2": "x_vars[3,6]",
    "x_5_3": "x_vars[4,6]",
    "x_5_4": "x_vars[5,6]",
    "x_5_5": "x_vars[6,6]",
    "x_5_6": "x_vars[7,6]",
    "x_5_7": "x_vars[8,6]",
    "x_5_8": "x_vars[9,6]",
    "x_5_9": "x_vars[10,6]",
    "x_6_0": "x_vars[1,7]",
    "x_6_1": "x_vars[2,7]",
    "x_6_2": "x_vars[3,7]",
    "x_6_3": "x_vars[4,7]",
    "x_6_4": "x_vars[5,7]",
    "x_6_5": "x_vars[6,7]",
    "x_6_6": "x_vars[7,7]",
    "x_6_7": "x_vars[8,7]",
    "x_6_8": "x_vars[9,7]",
    "x_6_9": "x_vars[10,7]",
    "x_7_0": "x_vars[1,8]",
    "x_7_1": "x_vars[2,8]",
    "x_7_2": "x_vars[3,8]",
    "x_7_3": "x_vars[4,8]",
    "x_7_4": "x_vars[5,8]",
    "x_7_5": "x_vars[6,8]",
    "x_7_6": "x_vars[7,8]",
    "x_7_7": "x_vars[8,8]",
    "x_7_8": "x_vars[9,8]",
    "x_7_9": "x_vars[10,8]",
    "x_8_0": "x_vars[1,9]",
    "x_8_1": "x_vars[2,9]",
    "x_8_2": "x_vars[3,9]",
    "x_8_3": "x_vars[4,9]",
    "x_8_4": "x_vars[5,9]",
    "x_8_5": "x_vars[6,9]",
    "x_8_6": "x_vars[7,9]",
    "x_8_7": "x_vars[8,9]",
    "x_8_8": "x_vars[9,9]",
    "x_8_9": "x_vars[10,9]",
    "x_9_0": "x_vars[1,10]",
    "x_9_1": "x_vars[2,10]",
    "x_9_2": "x_vars[3,10]",
    "x_9_3": "x_vars[4,10]",
    "x_9_4": "x_vars[5,10]",
    "x_9_5": "x_vars[6,10]",
    "x_9_6": "x_vars[7,10]",
    "x_9_7": "x_vars[8,10]",
    "x_9_8": "x_vars[9,10]",
    "x_9_9": "x_vars[10,10]",
    "x_10_0": "x_vars[1,11]",
    "x_10_1": "x_vars[2,11]",
    "x_10_2": "x_vars[3,11]",
    "x_10_3": "x_vars[4,11]",
    "x_10_4": "x_vars[5,11]",
    "x_10_5": "x_vars[6,11]",
    "x_10_6": "x_vars[7,11]",
    "x_10_7": "x_vars[8,11]",
    "x_10_8": "x_vars[9,11]",
    "x_10_9": "x_vars[10,11]",
    "x_11_0": "x_vars[1,12]",
    "x_11_1": "x_vars[2,12]",
    "x_11_2": "x_vars[3,12]",
    "x_11_3": "x_vars[4,12]",
    "x_11_4": "x_vars[5,12]",
    "x_11_5": "x_vars[6,12]",
    "x_11_6": "x_vars[7,12]",
    "x_11_7": "x_vars[8,12]",
    "x_11_8": "x_vars[9,12]",
    "x_11_9": "x_vars[10,12]",
    "x_12_0": "x_vars[1,13]",
    "x_12_1": "x_vars[2,13]",
    "x_12_2": "x_vars[3,13]",
    "x_12_3": "x_vars[4,13]",
    "x_12_4": "x_vars[5,13]",
    "x_12_5": "x_vars[6,13]",
    "x_12_6": "x_vars[7,13]",
    "x_12_7": "x_vars[8,13]",
    "x_12_8": "x_vars[9,13]",
    "x_12_9": "x_vars[10,13]",
    "x_13_0": "x_vars[1,14]",
    "x_13_1": "x_vars[2,14]",
    "x_13_2": "x_vars[3,14]",
    "x_13_3": "x_vars[4,14]",
    "x_13_4": "x_vars[5,14]",
    "x_13_5": "x_vars[6,14]",
    "x_13_6": "x_vars[7,14]",
    "x_13_7": "x_vars[8,14]",
    "x_13_8": "x_vars[9,14]",
    "x_13_9": "x_vars[10,14]",
    "x_14_0": "x_vars[1,15]",
    "x_14_1": "x_vars[2,15]",
    "x_14_2": "x_vars[3,15]",
    "x_14_3": "x_vars[4,15]",
    "x_14_4": "x_vars[5,15]",
    "x_14_5": "x_vars[6,15]",
    "x_14_6": "x_vars[7,15]",
    "x_14_7": "x_vars[8,15]",
    "x_14_8": "x_vars[9,15]",
    "x_14_9": "x_vars[10,15]",
    "x_15_0": "x_vars[1,16]",
    "x_15_1": "x_vars[2,16]",
    "x_15_2": "x_vars[3,16]",
    "x_15_3": "x_vars[4,16]",
    "x_15_4": "x_vars[5,16]",
    "x_15_5": "x_vars[6,16]",
    "x_15_6": "x_vars[7,16]",
    "x_15_7": "x_vars[8,16]",
    "x_15_8": "x_vars[9,16]",
    "x_15_9": "x_vars[10,16]",
    "x_16_0": "x_vars[1,17]",
    "x_16_1": "x_vars[2,17]",
    "x_16_2": "x_vars[3,17]",
    "x_16_3": "x_vars[4,17]",
    "x_16_4": "x_vars[5,17]",
    "x_16_5": "x_vars[6,17]",
    "x_16_6": "x_vars[7,17]",
    "x_16_7": "x_vars[8,17]",
    "x_16_8": "x_vars[9,17]",
    "x_16_9": "x_vars[10,17]",
    "x_17_0": "x_vars[1,18]",
    "x_17_1": "x_vars[2,18]",
    "x_17_2": "x_vars[3,18]",
    "x_17_3": "x_vars[4,18]",
    "x_17_4": "x_vars[5,18]",
    "x_17_5": "x_vars[6,18]",
    "x_17_6": "x_vars[7,18]",
    "x_17_7": "x_vars[8,18]",
    "x_17_8": "x_vars[9,18]",
    "x_17_9": "x_vars[10,18]",
    "x_18_0": "x_vars[1,19]",
    "x_18_1": "x_vars[2,19]",
    "x_18_2": "x_vars[3,19]",
    "x_18_3": "x_vars[4,19]",
    "x_18_4": "x_vars[5,19]",
    "x_18_5": "x_vars[6,19]",
    "x_18_6": "x_vars[7,19]",
    "x_18_7": "x_vars[8,19]",
    "x_18_8": "x_vars[9,19]",
    "x_18_9": "x_vars[10,19]",
    "x_19_0": "x_vars[1,20]",
    "x_19_1": "x_vars[2,20]",
    "x_19_2": "x_vars[3,20]",
    "x_19_3": "x_vars[4,20]",
    "x_19_4": "x_vars[5,20]",
    "x_19_5": "x_vars[6,20]",
    "x_19_6": "x_vars[7,20]",
    "x_19_7": "x_vars[8,20]",
    "x_19_8": "x_vars[9,20]",
    "x_19_9": "x_vars[10,20]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-055": {
   "source_code_sha256": "e4a0b8ce7cb1744ab2cfce6787b468cdef57273c21eac12506ecc49d0e5afc51",
   "original_solution_sha256": "78f1a854ceafe2eef30ae1d5063b9311ce898d0af26b10742e6b9875eeb2117a",
   "bindings": [
    [
     "C0",
     "x_vars[1,Speedboat]"
    ],
    [
     "C1",
     "x_vars[1,Fishing Boat]"
    ],
    [
     "C2",
     "x_vars[1,Catamaran]"
    ],
    [
     "C3",
     "x_vars[1,Yacht]"
    ],
    [
     "C4",
     "x_vars[1,Sailboat]"
    ],
    [
     "C5",
     "x_vars[1,Kayak]"
    ],
    [
     "C6",
     "x_vars[1,Canoe]"
    ],
    [
     "C7",
     "x_vars[1,Houseboat]"
    ],
    [
     "C8",
     "x_vars[1,Pontoon]"
    ],
    [
     "C9",
     "x_vars[1,Jet Ski]"
    ],
    [
     "C10",
     "x_vars[1,Rowboat]"
    ],
    [
     "C11",
     "x_vars[1,Hovercraft]"
    ],
    [
     "C12",
     "x_vars[1,Cabin Cruiser]"
    ],
    [
     "C13",
     "x_vars[1,Wakeboard Boat]"
    ],
    [
     "C14",
     "x_vars[1,Dinghy]"
    ],
    [
     "C15",
     "x_vars[1,Trawler]"
    ],
    [
     "C16",
     "x_vars[1,Paddle Boat]"
    ],
    [
     "C17",
     "x_vars[1,Submarine]"
    ],
    [
     "C18",
     "x_vars[1,RIB]"
    ],
    [
     "C19",
     "x_vars[1,Skiff]"
    ],
    [
     "C20",
     "x_vars[2,Speedboat]"
    ],
    [
     "C21",
     "x_vars[2,Fishing Boat]"
    ],
    [
     "C22",
     "x_vars[2,Catamaran]"
    ],
    [
     "C23",
     "x_vars[2,Yacht]"
    ],
    [
     "C24",
     "x_vars[2,Sailboat]"
    ],
    [
     "C25",
     "x_vars[2,Kayak]"
    ],
    [
     "C26",
     "x_vars[2,Canoe]"
    ],
    [
     "C27",
     "x_vars[2,Houseboat]"
    ],
    [
     "C28",
     "x_vars[2,Pontoon]"
    ],
    [
     "C29",
     "x_vars[2,Jet Ski]"
    ],
    [
     "C30",
     "x_vars[2,Rowboat]"
    ],
    [
     "C31",
     "x_vars[2,Hovercraft]"
    ],
    [
     "C32",
     "x_vars[2,Cabin Cruiser]"
    ],
    [
     "C33",
     "x_vars[2,Wakeboard Boat]"
    ],
    [
     "C34",
     "x_vars[2,Dinghy]"
    ],
    [
     "C35",
     "x_vars[2,Trawler]"
    ],
    [
     "C36",
     "x_vars[2,Paddle Boat]"
    ],
    [
     "C37",
     "x_vars[2,Submarine]"
    ],
    [
     "C38",
     "x_vars[2,RIB]"
    ],
    [
     "C39",
     "x_vars[2,Skiff]"
    ],
    [
     "C40",
     "x_vars[3,Speedboat]"
    ],
    [
     "C41",
     "x_vars[3,Fishing Boat]"
    ],
    [
     "C42",
     "x_vars[3,Catamaran]"
    ],
    [
     "C43",
     "x_vars[3,Yacht]"
    ],
    [
     "C44",
     "x_vars[3,Sailboat]"
    ],
    [
     "C45",
     "x_vars[3,Kayak]"
    ],
    [
     "C46",
     "x_vars[3,Canoe]"
    ],
    [
     "C47",
     "x_vars[3,Houseboat]"
    ],
    [
     "C48",
     "x_vars[3,Pontoon]"
    ],
    [
     "C49",
     "x_vars[3,Jet Ski]"
    ],
    [
     "C50",
     "x_vars[3,Rowboat]"
    ],
    [
     "C51",
     "x_vars[3,Hovercraft]"
    ],
    [
     "C52",
     "x_vars[3,Cabin Cruiser]"
    ],
    [
     "C53",
     "x_vars[3,Wakeboard Boat]"
    ],
    [
     "C54",
     "x_vars[3,Dinghy]"
    ],
    [
     "C55",
     "x_vars[3,Trawler]"
    ],
    [
     "C56",
     "x_vars[3,Paddle Boat]"
    ],
    [
     "C57",
     "x_vars[3,Submarine]"
    ],
    [
     "C58",
     "x_vars[3,RIB]"
    ],
    [
     "C59",
     "x_vars[3,Skiff]"
    ],
    [
     "C60",
     "x_vars[4,Speedboat]"
    ],
    [
     "C61",
     "x_vars[4,Fishing Boat]"
    ],
    [
     "C62",
     "x_vars[4,Catamaran]"
    ],
    [
     "C63",
     "x_vars[4,Yacht]"
    ],
    [
     "C64",
     "x_vars[4,Sailboat]"
    ],
    [
     "C65",
     "x_vars[4,Kayak]"
    ],
    [
     "C66",
     "x_vars[4,Canoe]"
    ],
    [
     "C67",
     "x_vars[4,Houseboat]"
    ],
    [
     "C68",
     "x_vars[4,Pontoon]"
    ],
    [
     "C69",
     "x_vars[4,Jet Ski]"
    ],
    [
     "C70",
     "x_vars[4,Rowboat]"
    ],
    [
     "C71",
     "x_vars[4,Hovercraft]"
    ],
    [
     "C72",
     "x_vars[4,Cabin Cruiser]"
    ],
    [
     "C73",
     "x_vars[4,Wakeboard Boat]"
    ],
    [
     "C74",
     "x_vars[4,Dinghy]"
    ],
    [
     "C75",
     "x_vars[4,Trawler]"
    ],
    [
     "C76",
     "x_vars[4,Paddle Boat]"
    ],
    [
     "C77",
     "x_vars[4,Submarine]"
    ],
    [
     "C78",
     "x_vars[4,RIB]"
    ],
    [
     "C79",
     "x_vars[4,Skiff]"
    ],
    [
     "C80",
     "x_vars[5,Speedboat]"
    ],
    [
     "C81",
     "x_vars[5,Fishing Boat]"
    ],
    [
     "C82",
     "x_vars[5,Catamaran]"
    ],
    [
     "C83",
     "x_vars[5,Yacht]"
    ],
    [
     "C84",
     "x_vars[5,Sailboat]"
    ],
    [
     "C85",
     "x_vars[5,Kayak]"
    ],
    [
     "C86",
     "x_vars[5,Canoe]"
    ],
    [
     "C87",
     "x_vars[5,Houseboat]"
    ],
    [
     "C88",
     "x_vars[5,Pontoon]"
    ],
    [
     "C89",
     "x_vars[5,Jet Ski]"
    ],
    [
     "C90",
     "x_vars[5,Rowboat]"
    ],
    [
     "C91",
     "x_vars[5,Hovercraft]"
    ],
    [
     "C92",
     "x_vars[5,Cabin Cruiser]"
    ],
    [
     "C93",
     "x_vars[5,Wakeboard Boat]"
    ],
    [
     "C94",
     "x_vars[5,Dinghy]"
    ],
    [
     "C95",
     "x_vars[5,Trawler]"
    ],
    [
     "C96",
     "x_vars[5,Paddle Boat]"
    ],
    [
     "C97",
     "x_vars[5,Submarine]"
    ],
    [
     "C98",
     "x_vars[5,RIB]"
    ],
    [
     "C99",
     "x_vars[5,Skiff]"
    ],
    [
     "C100",
     "x_vars[6,Speedboat]"
    ],
    [
     "C101",
     "x_vars[6,Fishing Boat]"
    ],
    [
     "C102",
     "x_vars[6,Catamaran]"
    ],
    [
     "C103",
     "x_vars[6,Yacht]"
    ],
    [
     "C104",
     "x_vars[6,Sailboat]"
    ],
    [
     "C105",
     "x_vars[6,Kayak]"
    ],
    [
     "C106",
     "x_vars[6,Canoe]"
    ],
    [
     "C107",
     "x_vars[6,Houseboat]"
    ],
    [
     "C108",
     "x_vars[6,Pontoon]"
    ],
    [
     "C109",
     "x_vars[6,Jet Ski]"
    ],
    [
     "C110",
     "x_vars[6,Rowboat]"
    ],
    [
     "C111",
     "x_vars[6,Hovercraft]"
    ],
    [
     "C112",
     "x_vars[6,Cabin Cruiser]"
    ],
    [
     "C113",
     "x_vars[6,Wakeboard Boat]"
    ],
    [
     "C114",
     "x_vars[6,Dinghy]"
    ],
    [
     "C115",
     "x_vars[6,Trawler]"
    ],
    [
     "C116",
     "x_vars[6,Paddle Boat]"
    ],
    [
     "C117",
     "x_vars[6,Submarine]"
    ],
    [
     "C118",
     "x_vars[6,RIB]"
    ],
    [
     "C119",
     "x_vars[6,Skiff]"
    ],
    [
     "C120",
     "x_vars[7,Speedboat]"
    ],
    [
     "C121",
     "x_vars[7,Fishing Boat]"
    ],
    [
     "C122",
     "x_vars[7,Catamaran]"
    ],
    [
     "C123",
     "x_vars[7,Yacht]"
    ],
    [
     "C124",
     "x_vars[7,Sailboat]"
    ],
    [
     "C125",
     "x_vars[7,Kayak]"
    ],
    [
     "C126",
     "x_vars[7,Canoe]"
    ],
    [
     "C127",
     "x_vars[7,Houseboat]"
    ],
    [
     "C128",
     "x_vars[7,Pontoon]"
    ],
    [
     "C129",
     "x_vars[7,Jet Ski]"
    ],
    [
     "C130",
     "x_vars[7,Rowboat]"
    ],
    [
     "C131",
     "x_vars[7,Hovercraft]"
    ],
    [
     "C132",
     "x_vars[7,Cabin Cruiser]"
    ],
    [
     "C133",
     "x_vars[7,Wakeboard Boat]"
    ],
    [
     "C134",
     "x_vars[7,Dinghy]"
    ],
    [
     "C135",
     "x_vars[7,Trawler]"
    ],
    [
     "C136",
     "x_vars[7,Paddle Boat]"
    ],
    [
     "C137",
     "x_vars[7,Submarine]"
    ],
    [
     "C138",
     "x_vars[7,RIB]"
    ],
    [
     "C139",
     "x_vars[7,Skiff]"
    ],
    [
     "C140",
     "x_vars[8,Speedboat]"
    ],
    [
     "C141",
     "x_vars[8,Fishing Boat]"
    ],
    [
     "C142",
     "x_vars[8,Catamaran]"
    ],
    [
     "C143",
     "x_vars[8,Yacht]"
    ],
    [
     "C144",
     "x_vars[8,Sailboat]"
    ],
    [
     "C145",
     "x_vars[8,Kayak]"
    ],
    [
     "C146",
     "x_vars[8,Canoe]"
    ],
    [
     "C147",
     "x_vars[8,Houseboat]"
    ],
    [
     "C148",
     "x_vars[8,Pontoon]"
    ],
    [
     "C149",
     "x_vars[8,Jet Ski]"
    ],
    [
     "C150",
     "x_vars[8,Rowboat]"
    ],
    [
     "C151",
     "x_vars[8,Hovercraft]"
    ],
    [
     "C152",
     "x_vars[8,Cabin Cruiser]"
    ],
    [
     "C153",
     "x_vars[8,Wakeboard Boat]"
    ],
    [
     "C154",
     "x_vars[8,Dinghy]"
    ],
    [
     "C155",
     "x_vars[8,Trawler]"
    ],
    [
     "C156",
     "x_vars[8,Paddle Boat]"
    ],
    [
     "C157",
     "x_vars[8,Submarine]"
    ],
    [
     "C158",
     "x_vars[8,RIB]"
    ],
    [
     "C159",
     "x_vars[8,Skiff]"
    ],
    [
     "C160",
     "x_vars[9,Speedboat]"
    ],
    [
     "C161",
     "x_vars[9,Fishing Boat]"
    ],
    [
     "C162",
     "x_vars[9,Catamaran]"
    ],
    [
     "C163",
     "x_vars[9,Yacht]"
    ],
    [
     "C164",
     "x_vars[9,Sailboat]"
    ],
    [
     "C165",
     "x_vars[9,Kayak]"
    ],
    [
     "C166",
     "x_vars[9,Canoe]"
    ],
    [
     "C167",
     "x_vars[9,Houseboat]"
    ],
    [
     "C168",
     "x_vars[9,Pontoon]"
    ],
    [
     "C169",
     "x_vars[9,Jet Ski]"
    ],
    [
     "C170",
     "x_vars[9,Rowboat]"
    ],
    [
     "C171",
     "x_vars[9,Hovercraft]"
    ],
    [
     "C172",
     "x_vars[9,Cabin Cruiser]"
    ],
    [
     "C173",
     "x_vars[9,Wakeboard Boat]"
    ],
    [
     "C174",
     "x_vars[9,Dinghy]"
    ],
    [
     "C175",
     "x_vars[9,Trawler]"
    ],
    [
     "C176",
     "x_vars[9,Paddle Boat]"
    ],
    [
     "C177",
     "x_vars[9,Submarine]"
    ],
    [
     "C178",
     "x_vars[9,RIB]"
    ],
    [
     "C179",
     "x_vars[9,Skiff]"
    ],
    [
     "C180",
     "x_vars[10,Speedboat]"
    ],
    [
     "C181",
     "x_vars[10,Fishing Boat]"
    ],
    [
     "C182",
     "x_vars[10,Catamaran]"
    ],
    [
     "C183",
     "x_vars[10,Yacht]"
    ],
    [
     "C184",
     "x_vars[10,Sailboat]"
    ],
    [
     "C185",
     "x_vars[10,Kayak]"
    ],
    [
     "C186",
     "x_vars[10,Canoe]"
    ],
    [
     "C187",
     "x_vars[10,Houseboat]"
    ],
    [
     "C188",
     "x_vars[10,Pontoon]"
    ],
    [
     "C189",
     "x_vars[10,Jet Ski]"
    ],
    [
     "C190",
     "x_vars[10,Rowboat]"
    ],
    [
     "C191",
     "x_vars[10,Hovercraft]"
    ],
    [
     "C192",
     "x_vars[10,Cabin Cruiser]"
    ],
    [
     "C193",
     "x_vars[10,Wakeboard Boat]"
    ],
    [
     "C194",
     "x_vars[10,Dinghy]"
    ],
    [
     "C195",
     "x_vars[10,Trawler]"
    ],
    [
     "C196",
     "x_vars[10,Paddle Boat]"
    ],
    [
     "C197",
     "x_vars[10,Submarine]"
    ],
    [
     "C198",
     "x_vars[10,RIB]"
    ],
    [
     "C199",
     "x_vars[10,Skiff]"
    ],
    [
     "C200",
     "x_vars[11,Speedboat]"
    ],
    [
     "C201",
     "x_vars[11,Fishing Boat]"
    ],
    [
     "C202",
     "x_vars[11,Catamaran]"
    ],
    [
     "C203",
     "x_vars[11,Yacht]"
    ],
    [
     "C204",
     "x_vars[11,Sailboat]"
    ],
    [
     "C205",
     "x_vars[11,Kayak]"
    ],
    [
     "C206",
     "x_vars[11,Canoe]"
    ],
    [
     "C207",
     "x_vars[11,Houseboat]"
    ],
    [
     "C208",
     "x_vars[11,Pontoon]"
    ],
    [
     "C209",
     "x_vars[11,Jet Ski]"
    ],
    [
     "C210",
     "x_vars[11,Rowboat]"
    ],
    [
     "C211",
     "x_vars[11,Hovercraft]"
    ],
    [
     "C212",
     "x_vars[11,Cabin Cruiser]"
    ],
    [
     "C213",
     "x_vars[11,Wakeboard Boat]"
    ],
    [
     "C214",
     "x_vars[11,Dinghy]"
    ],
    [
     "C215",
     "x_vars[11,Trawler]"
    ],
    [
     "C216",
     "x_vars[11,Paddle Boat]"
    ],
    [
     "C217",
     "x_vars[11,Submarine]"
    ],
    [
     "C218",
     "x_vars[11,RIB]"
    ],
    [
     "C219",
     "x_vars[11,Skiff]"
    ],
    [
     "C220",
     "x_vars[12,Speedboat]"
    ],
    [
     "C221",
     "x_vars[12,Fishing Boat]"
    ],
    [
     "C222",
     "x_vars[12,Catamaran]"
    ],
    [
     "C223",
     "x_vars[12,Yacht]"
    ],
    [
     "C224",
     "x_vars[12,Sailboat]"
    ],
    [
     "C225",
     "x_vars[12,Kayak]"
    ],
    [
     "C226",
     "x_vars[12,Canoe]"
    ],
    [
     "C227",
     "x_vars[12,Houseboat]"
    ],
    [
     "C228",
     "x_vars[12,Pontoon]"
    ],
    [
     "C229",
     "x_vars[12,Jet Ski]"
    ],
    [
     "C230",
     "x_vars[12,Rowboat]"
    ],
    [
     "C231",
     "x_vars[12,Hovercraft]"
    ],
    [
     "C232",
     "x_vars[12,Cabin Cruiser]"
    ],
    [
     "C233",
     "x_vars[12,Wakeboard Boat]"
    ],
    [
     "C234",
     "x_vars[12,Dinghy]"
    ],
    [
     "C235",
     "x_vars[12,Trawler]"
    ],
    [
     "C236",
     "x_vars[12,Paddle Boat]"
    ],
    [
     "C237",
     "x_vars[12,Submarine]"
    ],
    [
     "C238",
     "x_vars[12,RIB]"
    ],
    [
     "C239",
     "x_vars[12,Skiff]"
    ],
    [
     "C240",
     "x_vars[13,Speedboat]"
    ],
    [
     "C241",
     "x_vars[13,Fishing Boat]"
    ],
    [
     "C242",
     "x_vars[13,Catamaran]"
    ],
    [
     "C243",
     "x_vars[13,Yacht]"
    ],
    [
     "C244",
     "x_vars[13,Sailboat]"
    ],
    [
     "C245",
     "x_vars[13,Kayak]"
    ],
    [
     "C246",
     "x_vars[13,Canoe]"
    ],
    [
     "C247",
     "x_vars[13,Houseboat]"
    ],
    [
     "C248",
     "x_vars[13,Pontoon]"
    ],
    [
     "C249",
     "x_vars[13,Jet Ski]"
    ],
    [
     "C250",
     "x_vars[13,Rowboat]"
    ],
    [
     "C251",
     "x_vars[13,Hovercraft]"
    ],
    [
     "C252",
     "x_vars[13,Cabin Cruiser]"
    ],
    [
     "C253",
     "x_vars[13,Wakeboard Boat]"
    ],
    [
     "C254",
     "x_vars[13,Dinghy]"
    ],
    [
     "C255",
     "x_vars[13,Trawler]"
    ],
    [
     "C256",
     "x_vars[13,Paddle Boat]"
    ],
    [
     "C257",
     "x_vars[13,Submarine]"
    ],
    [
     "C258",
     "x_vars[13,RIB]"
    ],
    [
     "C259",
     "x_vars[13,Skiff]"
    ],
    [
     "C260",
     "x_vars[14,Speedboat]"
    ],
    [
     "C261",
     "x_vars[14,Fishing Boat]"
    ],
    [
     "C262",
     "x_vars[14,Catamaran]"
    ],
    [
     "C263",
     "x_vars[14,Yacht]"
    ],
    [
     "C264",
     "x_vars[14,Sailboat]"
    ],
    [
     "C265",
     "x_vars[14,Kayak]"
    ],
    [
     "C266",
     "x_vars[14,Canoe]"
    ],
    [
     "C267",
     "x_vars[14,Houseboat]"
    ],
    [
     "C268",
     "x_vars[14,Pontoon]"
    ],
    [
     "C269",
     "x_vars[14,Jet Ski]"
    ],
    [
     "C270",
     "x_vars[14,Rowboat]"
    ],
    [
     "C271",
     "x_vars[14,Hovercraft]"
    ],
    [
     "C272",
     "x_vars[14,Cabin Cruiser]"
    ],
    [
     "C273",
     "x_vars[14,Wakeboard Boat]"
    ],
    [
     "C274",
     "x_vars[14,Dinghy]"
    ],
    [
     "C275",
     "x_vars[14,Trawler]"
    ],
    [
     "C276",
     "x_vars[14,Paddle Boat]"
    ],
    [
     "C277",
     "x_vars[14,Submarine]"
    ],
    [
     "C278",
     "x_vars[14,RIB]"
    ],
    [
     "C279",
     "x_vars[14,Skiff]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Speedboat]",
    "x_0_1": "x_vars[2,Speedboat]",
    "x_0_2": "x_vars[3,Speedboat]",
    "x_0_3": "x_vars[4,Speedboat]",
    "x_0_4": "x_vars[5,Speedboat]",
    "x_0_5": "x_vars[6,Speedboat]",
    "x_0_6": "x_vars[7,Speedboat]",
    "x_0_7": "x_vars[8,Speedboat]",
    "x_0_8": "x_vars[9,Speedboat]",
    "x_0_9": "x_vars[10,Speedboat]",
    "x_0_10": "x_vars[11,Speedboat]",
    "x_0_11": "x_vars[12,Speedboat]",
    "x_0_12": "x_vars[13,Speedboat]",
    "x_0_13": "x_vars[14,Speedboat]",
    "x_1_0": "x_vars[1,Fishing Boat]",
    "x_1_1": "x_vars[2,Fishing Boat]",
    "x_1_2": "x_vars[3,Fishing Boat]",
    "x_1_3": "x_vars[4,Fishing Boat]",
    "x_1_4": "x_vars[5,Fishing Boat]",
    "x_1_5": "x_vars[6,Fishing Boat]",
    "x_1_6": "x_vars[7,Fishing Boat]",
    "x_1_7": "x_vars[8,Fishing Boat]",
    "x_1_8": "x_vars[9,Fishing Boat]",
    "x_1_9": "x_vars[10,Fishing Boat]",
    "x_1_10": "x_vars[11,Fishing Boat]",
    "x_1_11": "x_vars[12,Fishing Boat]",
    "x_1_12": "x_vars[13,Fishing Boat]",
    "x_1_13": "x_vars[14,Fishing Boat]",
    "x_2_0": "x_vars[1,Catamaran]",
    "x_2_1": "x_vars[2,Catamaran]",
    "x_2_2": "x_vars[3,Catamaran]",
    "x_2_3": "x_vars[4,Catamaran]",
    "x_2_4": "x_vars[5,Catamaran]",
    "x_2_5": "x_vars[6,Catamaran]",
    "x_2_6": "x_vars[7,Catamaran]",
    "x_2_7": "x_vars[8,Catamaran]",
    "x_2_8": "x_vars[9,Catamaran]",
    "x_2_9": "x_vars[10,Catamaran]",
    "x_2_10": "x_vars[11,Catamaran]",
    "x_2_11": "x_vars[12,Catamaran]",
    "x_2_12": "x_vars[13,Catamaran]",
    "x_2_13": "x_vars[14,Catamaran]",
    "x_3_0": "x_vars[1,Yacht]",
    "x_3_1": "x_vars[2,Yacht]",
    "x_3_2": "x_vars[3,Yacht]",
    "x_3_3": "x_vars[4,Yacht]",
    "x_3_4": "x_vars[5,Yacht]",
    "x_3_5": "x_vars[6,Yacht]",
    "x_3_6": "x_vars[7,Yacht]",
    "x_3_7": "x_vars[8,Yacht]",
    "x_3_8": "x_vars[9,Yacht]",
    "x_3_9": "x_vars[10,Yacht]",
    "x_3_10": "x_vars[11,Yacht]",
    "x_3_11": "x_vars[12,Yacht]",
    "x_3_12": "x_vars[13,Yacht]",
    "x_3_13": "x_vars[14,Yacht]",
    "x_4_0": "x_vars[1,Sailboat]",
    "x_4_1": "x_vars[2,Sailboat]",
    "x_4_2": "x_vars[3,Sailboat]",
    "x_4_3": "x_vars[4,Sailboat]",
    "x_4_4": "x_vars[5,Sailboat]",
    "x_4_5": "x_vars[6,Sailboat]",
    "x_4_6": "x_vars[7,Sailboat]",
    "x_4_7": "x_vars[8,Sailboat]",
    "x_4_8": "x_vars[9,Sailboat]",
    "x_4_9": "x_vars[10,Sailboat]",
    "x_4_10": "x_vars[11,Sailboat]",
    "x_4_11": "x_vars[12,Sailboat]",
    "x_4_12": "x_vars[13,Sailboat]",
    "x_4_13": "x_vars[14,Sailboat]",
    "x_5_0": "x_vars[1,Kayak]",
    "x_5_1": "x_vars[2,Kayak]",
    "x_5_2": "x_vars[3,Kayak]",
    "x_5_3": "x_vars[4,Kayak]",
    "x_5_4": "x_vars[5,Kayak]",
    "x_5_5": "x_vars[6,Kayak]",
    "x_5_6": "x_vars[7,Kayak]",
    "x_5_7": "x_vars[8,Kayak]",
    "x_5_8": "x_vars[9,Kayak]",
    "x_5_9": "x_vars[10,Kayak]",
    "x_5_10": "x_vars[11,Kayak]",
    "x_5_11": "x_vars[12,Kayak]",
    "x_5_12": "x_vars[13,Kayak]",
    "x_5_13": "x_vars[14,Kayak]",
    "x_6_0": "x_vars[1,Canoe]",
    "x_6_1": "x_vars[2,Canoe]",
    "x_6_2": "x_vars[3,Canoe]",
    "x_6_3": "x_vars[4,Canoe]",
    "x_6_4": "x_vars[5,Canoe]",
    "x_6_5": "x_vars[6,Canoe]",
    "x_6_6": "x_vars[7,Canoe]",
    "x_6_7": "x_vars[8,Canoe]",
    "x_6_8": "x_vars[9,Canoe]",
    "x_6_9": "x_vars[10,Canoe]",
    "x_6_10": "x_vars[11,Canoe]",
    "x_6_11": "x_vars[12,Canoe]",
    "x_6_12": "x_vars[13,Canoe]",
    "x_6_13": "x_vars[14,Canoe]",
    "x_7_0": "x_vars[1,Houseboat]",
    "x_7_1": "x_vars[2,Houseboat]",
    "x_7_2": "x_vars[3,Houseboat]",
    "x_7_3": "x_vars[4,Houseboat]",
    "x_7_4": "x_vars[5,Houseboat]",
    "x_7_5": "x_vars[6,Houseboat]",
    "x_7_6": "x_vars[7,Houseboat]",
    "x_7_7": "x_vars[8,Houseboat]",
    "x_7_8": "x_vars[9,Houseboat]",
    "x_7_9": "x_vars[10,Houseboat]",
    "x_7_10": "x_vars[11,Houseboat]",
    "x_7_11": "x_vars[12,Houseboat]",
    "x_7_12": "x_vars[13,Houseboat]",
    "x_7_13": "x_vars[14,Houseboat]",
    "x_8_0": "x_vars[1,Pontoon]",
    "x_8_1": "x_vars[2,Pontoon]",
    "x_8_2": "x_vars[3,Pontoon]",
    "x_8_3": "x_vars[4,Pontoon]",
    "x_8_4": "x_vars[5,Pontoon]",
    "x_8_5": "x_vars[6,Pontoon]",
    "x_8_6": "x_vars[7,Pontoon]",
    "x_8_7": "x_vars[8,Pontoon]",
    "x_8_8": "x_vars[9,Pontoon]",
    "x_8_9": "x_vars[10,Pontoon]",
    "x_8_10": "x_vars[11,Pontoon]",
    "x_8_11": "x_vars[12,Pontoon]",
    "x_8_12": "x_vars[13,Pontoon]",
    "x_8_13": "x_vars[14,Pontoon]",
    "x_9_0": "x_vars[1,Jet Ski]",
    "x_9_1": "x_vars[2,Jet Ski]",
    "x_9_2": "x_vars[3,Jet Ski]",
    "x_9_3": "x_vars[4,Jet Ski]",
    "x_9_4": "x_vars[5,Jet Ski]",
    "x_9_5": "x_vars[6,Jet Ski]",
    "x_9_6": "x_vars[7,Jet Ski]",
    "x_9_7": "x_vars[8,Jet Ski]",
    "x_9_8": "x_vars[9,Jet Ski]",
    "x_9_9": "x_vars[10,Jet Ski]",
    "x_9_10": "x_vars[11,Jet Ski]",
    "x_9_11": "x_vars[12,Jet Ski]",
    "x_9_12": "x_vars[13,Jet Ski]",
    "x_9_13": "x_vars[14,Jet Ski]",
    "x_10_0": "x_vars[1,Rowboat]",
    "x_10_1": "x_vars[2,Rowboat]",
    "x_10_2": "x_vars[3,Rowboat]",
    "x_10_3": "x_vars[4,Rowboat]",
    "x_10_4": "x_vars[5,Rowboat]",
    "x_10_5": "x_vars[6,Rowboat]",
    "x_10_6": "x_vars[7,Rowboat]",
    "x_10_7": "x_vars[8,Rowboat]",
    "x_10_8": "x_vars[9,Rowboat]",
    "x_10_9": "x_vars[10,Rowboat]",
    "x_10_10": "x_vars[11,Rowboat]",
    "x_10_11": "x_vars[12,Rowboat]",
    "x_10_12": "x_vars[13,Rowboat]",
    "x_10_13": "x_vars[14,Rowboat]",
    "x_11_0": "x_vars[1,Hovercraft]",
    "x_11_1": "x_vars[2,Hovercraft]",
    "x_11_2": "x_vars[3,Hovercraft]",
    "x_11_3": "x_vars[4,Hovercraft]",
    "x_11_4": "x_vars[5,Hovercraft]",
    "x_11_5": "x_vars[6,Hovercraft]",
    "x_11_6": "x_vars[7,Hovercraft]",
    "x_11_7": "x_vars[8,Hovercraft]",
    "x_11_8": "x_vars[9,Hovercraft]",
    "x_11_9": "x_vars[10,Hovercraft]",
    "x_11_10": "x_vars[11,Hovercraft]",
    "x_11_11": "x_vars[12,Hovercraft]",
    "x_11_12": "x_vars[13,Hovercraft]",
    "x_11_13": "x_vars[14,Hovercraft]",
    "x_12_0": "x_vars[1,Cabin Cruiser]",
    "x_12_1": "x_vars[2,Cabin Cruiser]",
    "x_12_2": "x_vars[3,Cabin Cruiser]",
    "x_12_3": "x_vars[4,Cabin Cruiser]",
    "x_12_4": "x_vars[5,Cabin Cruiser]",
    "x_12_5": "x_vars[6,Cabin Cruiser]",
    "x_12_6": "x_vars[7,Cabin Cruiser]",
    "x_12_7": "x_vars[8,Cabin Cruiser]",
    "x_12_8": "x_vars[9,Cabin Cruiser]",
    "x_12_9": "x_vars[10,Cabin Cruiser]",
    "x_12_10": "x_vars[11,Cabin Cruiser]",
    "x_12_11": "x_vars[12,Cabin Cruiser]",
    "x_12_12": "x_vars[13,Cabin Cruiser]",
    "x_12_13": "x_vars[14,Cabin Cruiser]",
    "x_13_0": "x_vars[1,Wakeboard Boat]",
    "x_13_1": "x_vars[2,Wakeboard Boat]",
    "x_13_2": "x_vars[3,Wakeboard Boat]",
    "x_13_3": "x_vars[4,Wakeboard Boat]",
    "x_13_4": "x_vars[5,Wakeboard Boat]",
    "x_13_5": "x_vars[6,Wakeboard Boat]",
    "x_13_6": "x_vars[7,Wakeboard Boat]",
    "x_13_7": "x_vars[8,Wakeboard Boat]",
    "x_13_8": "x_vars[9,Wakeboard Boat]",
    "x_13_9": "x_vars[10,Wakeboard Boat]",
    "x_13_10": "x_vars[11,Wakeboard Boat]",
    "x_13_11": "x_vars[12,Wakeboard Boat]",
    "x_13_12": "x_vars[13,Wakeboard Boat]",
    "x_13_13": "x_vars[14,Wakeboard Boat]",
    "x_14_0": "x_vars[1,Dinghy]",
    "x_14_1": "x_vars[2,Dinghy]",
    "x_14_2": "x_vars[3,Dinghy]",
    "x_14_3": "x_vars[4,Dinghy]",
    "x_14_4": "x_vars[5,Dinghy]",
    "x_14_5": "x_vars[6,Dinghy]",
    "x_14_6": "x_vars[7,Dinghy]",
    "x_14_7": "x_vars[8,Dinghy]",
    "x_14_8": "x_vars[9,Dinghy]",
    "x_14_9": "x_vars[10,Dinghy]",
    "x_14_10": "x_vars[11,Dinghy]",
    "x_14_11": "x_vars[12,Dinghy]",
    "x_14_12": "x_vars[13,Dinghy]",
    "x_14_13": "x_vars[14,Dinghy]",
    "x_15_0": "x_vars[1,Trawler]",
    "x_15_1": "x_vars[2,Trawler]",
    "x_15_2": "x_vars[3,Trawler]",
    "x_15_3": "x_vars[4,Trawler]",
    "x_15_4": "x_vars[5,Trawler]",
    "x_15_5": "x_vars[6,Trawler]",
    "x_15_6": "x_vars[7,Trawler]",
    "x_15_7": "x_vars[8,Trawler]",
    "x_15_8": "x_vars[9,Trawler]",
    "x_15_9": "x_vars[10,Trawler]",
    "x_15_10": "x_vars[11,Trawler]",
    "x_15_11": "x_vars[12,Trawler]",
    "x_15_12": "x_vars[13,Trawler]",
    "x_15_13": "x_vars[14,Trawler]",
    "x_16_0": "x_vars[1,Paddle Boat]",
    "x_16_1": "x_vars[2,Paddle Boat]",
    "x_16_2": "x_vars[3,Paddle Boat]",
    "x_16_3": "x_vars[4,Paddle Boat]",
    "x_16_4": "x_vars[5,Paddle Boat]",
    "x_16_5": "x_vars[6,Paddle Boat]",
    "x_16_6": "x_vars[7,Paddle Boat]",
    "x_16_7": "x_vars[8,Paddle Boat]",
    "x_16_8": "x_vars[9,Paddle Boat]",
    "x_16_9": "x_vars[10,Paddle Boat]",
    "x_16_10": "x_vars[11,Paddle Boat]",
    "x_16_11": "x_vars[12,Paddle Boat]",
    "x_16_12": "x_vars[13,Paddle Boat]",
    "x_16_13": "x_vars[14,Paddle Boat]",
    "x_17_0": "x_vars[1,Submarine]",
    "x_17_1": "x_vars[2,Submarine]",
    "x_17_2": "x_vars[3,Submarine]",
    "x_17_3": "x_vars[4,Submarine]",
    "x_17_4": "x_vars[5,Submarine]",
    "x_17_5": "x_vars[6,Submarine]",
    "x_17_6": "x_vars[7,Submarine]",
    "x_17_7": "x_vars[8,Submarine]",
    "x_17_8": "x_vars[9,Submarine]",
    "x_17_9": "x_vars[10,Submarine]",
    "x_17_10": "x_vars[11,Submarine]",
    "x_17_11": "x_vars[12,Submarine]",
    "x_17_12": "x_vars[13,Submarine]",
    "x_17_13": "x_vars[14,Submarine]",
    "x_18_0": "x_vars[1,RIB]",
    "x_18_1": "x_vars[2,RIB]",
    "x_18_2": "x_vars[3,RIB]",
    "x_18_3": "x_vars[4,RIB]",
    "x_18_4": "x_vars[5,RIB]",
    "x_18_5": "x_vars[6,RIB]",
    "x_18_6": "x_vars[7,RIB]",
    "x_18_7": "x_vars[8,RIB]",
    "x_18_8": "x_vars[9,RIB]",
    "x_18_9": "x_vars[10,RIB]",
    "x_18_10": "x_vars[11,RIB]",
    "x_18_11": "x_vars[12,RIB]",
    "x_18_12": "x_vars[13,RIB]",
    "x_18_13": "x_vars[14,RIB]",
    "x_19_0": "x_vars[1,Skiff]",
    "x_19_1": "x_vars[2,Skiff]",
    "x_19_2": "x_vars[3,Skiff]",
    "x_19_3": "x_vars[4,Skiff]",
    "x_19_4": "x_vars[5,Skiff]",
    "x_19_5": "x_vars[6,Skiff]",
    "x_19_6": "x_vars[7,Skiff]",
    "x_19_7": "x_vars[8,Skiff]",
    "x_19_8": "x_vars[9,Skiff]",
    "x_19_9": "x_vars[10,Skiff]",
    "x_19_10": "x_vars[11,Skiff]",
    "x_19_11": "x_vars[12,Skiff]",
    "x_19_12": "x_vars[13,Skiff]",
    "x_19_13": "x_vars[14,Skiff]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-056": {
   "source_code_sha256": "81fb344dd553bc3998bf445b285e67f14a36d4af23e2c26334211c257c11420c",
   "original_solution_sha256": "ec3f37c07a4743a464dfaab2feb60b9233d476a401f284ea59d71dd41cdbf9ac",
   "bindings": [
    [
     "C0",
     "x_vars[1,Speedboat]"
    ],
    [
     "C1",
     "x_vars[1,Fishing Boat]"
    ],
    [
     "C2",
     "x_vars[1,Catamaran]"
    ],
    [
     "C3",
     "x_vars[1,Yacht]"
    ],
    [
     "C4",
     "x_vars[1,Sailboat]"
    ],
    [
     "C5",
     "x_vars[1,Kayak]"
    ],
    [
     "C6",
     "x_vars[1,Canoe]"
    ],
    [
     "C7",
     "x_vars[1,Houseboat]"
    ],
    [
     "C8",
     "x_vars[1,Pontoon]"
    ],
    [
     "C9",
     "x_vars[1,Jet Ski]"
    ],
    [
     "C10",
     "x_vars[1,Rowboat]"
    ],
    [
     "C11",
     "x_vars[1,Hovercraft]"
    ],
    [
     "C12",
     "x_vars[1,Cabin Cruiser]"
    ],
    [
     "C13",
     "x_vars[1,Wakeboard Boat]"
    ],
    [
     "C14",
     "x_vars[1,Dinghy]"
    ],
    [
     "C15",
     "x_vars[1,Trawler]"
    ],
    [
     "C16",
     "x_vars[1,Paddle Boat]"
    ],
    [
     "C17",
     "x_vars[1,Submarine]"
    ],
    [
     "C18",
     "x_vars[1,RIB]"
    ],
    [
     "C19",
     "x_vars[1,Skiff]"
    ],
    [
     "C20",
     "x_vars[2,Speedboat]"
    ],
    [
     "C21",
     "x_vars[2,Fishing Boat]"
    ],
    [
     "C22",
     "x_vars[2,Catamaran]"
    ],
    [
     "C23",
     "x_vars[2,Yacht]"
    ],
    [
     "C24",
     "x_vars[2,Sailboat]"
    ],
    [
     "C25",
     "x_vars[2,Kayak]"
    ],
    [
     "C26",
     "x_vars[2,Canoe]"
    ],
    [
     "C27",
     "x_vars[2,Houseboat]"
    ],
    [
     "C28",
     "x_vars[2,Pontoon]"
    ],
    [
     "C29",
     "x_vars[2,Jet Ski]"
    ],
    [
     "C30",
     "x_vars[2,Rowboat]"
    ],
    [
     "C31",
     "x_vars[2,Hovercraft]"
    ],
    [
     "C32",
     "x_vars[2,Cabin Cruiser]"
    ],
    [
     "C33",
     "x_vars[2,Wakeboard Boat]"
    ],
    [
     "C34",
     "x_vars[2,Dinghy]"
    ],
    [
     "C35",
     "x_vars[2,Trawler]"
    ],
    [
     "C36",
     "x_vars[2,Paddle Boat]"
    ],
    [
     "C37",
     "x_vars[2,Submarine]"
    ],
    [
     "C38",
     "x_vars[2,RIB]"
    ],
    [
     "C39",
     "x_vars[2,Skiff]"
    ],
    [
     "C40",
     "x_vars[3,Speedboat]"
    ],
    [
     "C41",
     "x_vars[3,Fishing Boat]"
    ],
    [
     "C42",
     "x_vars[3,Catamaran]"
    ],
    [
     "C43",
     "x_vars[3,Yacht]"
    ],
    [
     "C44",
     "x_vars[3,Sailboat]"
    ],
    [
     "C45",
     "x_vars[3,Kayak]"
    ],
    [
     "C46",
     "x_vars[3,Canoe]"
    ],
    [
     "C47",
     "x_vars[3,Houseboat]"
    ],
    [
     "C48",
     "x_vars[3,Pontoon]"
    ],
    [
     "C49",
     "x_vars[3,Jet Ski]"
    ],
    [
     "C50",
     "x_vars[3,Rowboat]"
    ],
    [
     "C51",
     "x_vars[3,Hovercraft]"
    ],
    [
     "C52",
     "x_vars[3,Cabin Cruiser]"
    ],
    [
     "C53",
     "x_vars[3,Wakeboard Boat]"
    ],
    [
     "C54",
     "x_vars[3,Dinghy]"
    ],
    [
     "C55",
     "x_vars[3,Trawler]"
    ],
    [
     "C56",
     "x_vars[3,Paddle Boat]"
    ],
    [
     "C57",
     "x_vars[3,Submarine]"
    ],
    [
     "C58",
     "x_vars[3,RIB]"
    ],
    [
     "C59",
     "x_vars[3,Skiff]"
    ],
    [
     "C60",
     "x_vars[4,Speedboat]"
    ],
    [
     "C61",
     "x_vars[4,Fishing Boat]"
    ],
    [
     "C62",
     "x_vars[4,Catamaran]"
    ],
    [
     "C63",
     "x_vars[4,Yacht]"
    ],
    [
     "C64",
     "x_vars[4,Sailboat]"
    ],
    [
     "C65",
     "x_vars[4,Kayak]"
    ],
    [
     "C66",
     "x_vars[4,Canoe]"
    ],
    [
     "C67",
     "x_vars[4,Houseboat]"
    ],
    [
     "C68",
     "x_vars[4,Pontoon]"
    ],
    [
     "C69",
     "x_vars[4,Jet Ski]"
    ],
    [
     "C70",
     "x_vars[4,Rowboat]"
    ],
    [
     "C71",
     "x_vars[4,Hovercraft]"
    ],
    [
     "C72",
     "x_vars[4,Cabin Cruiser]"
    ],
    [
     "C73",
     "x_vars[4,Wakeboard Boat]"
    ],
    [
     "C74",
     "x_vars[4,Dinghy]"
    ],
    [
     "C75",
     "x_vars[4,Trawler]"
    ],
    [
     "C76",
     "x_vars[4,Paddle Boat]"
    ],
    [
     "C77",
     "x_vars[4,Submarine]"
    ],
    [
     "C78",
     "x_vars[4,RIB]"
    ],
    [
     "C79",
     "x_vars[4,Skiff]"
    ],
    [
     "C80",
     "x_vars[5,Speedboat]"
    ],
    [
     "C81",
     "x_vars[5,Fishing Boat]"
    ],
    [
     "C82",
     "x_vars[5,Catamaran]"
    ],
    [
     "C83",
     "x_vars[5,Yacht]"
    ],
    [
     "C84",
     "x_vars[5,Sailboat]"
    ],
    [
     "C85",
     "x_vars[5,Kayak]"
    ],
    [
     "C86",
     "x_vars[5,Canoe]"
    ],
    [
     "C87",
     "x_vars[5,Houseboat]"
    ],
    [
     "C88",
     "x_vars[5,Pontoon]"
    ],
    [
     "C89",
     "x_vars[5,Jet Ski]"
    ],
    [
     "C90",
     "x_vars[5,Rowboat]"
    ],
    [
     "C91",
     "x_vars[5,Hovercraft]"
    ],
    [
     "C92",
     "x_vars[5,Cabin Cruiser]"
    ],
    [
     "C93",
     "x_vars[5,Wakeboard Boat]"
    ],
    [
     "C94",
     "x_vars[5,Dinghy]"
    ],
    [
     "C95",
     "x_vars[5,Trawler]"
    ],
    [
     "C96",
     "x_vars[5,Paddle Boat]"
    ],
    [
     "C97",
     "x_vars[5,Submarine]"
    ],
    [
     "C98",
     "x_vars[5,RIB]"
    ],
    [
     "C99",
     "x_vars[5,Skiff]"
    ],
    [
     "C100",
     "x_vars[6,Speedboat]"
    ],
    [
     "C101",
     "x_vars[6,Fishing Boat]"
    ],
    [
     "C102",
     "x_vars[6,Catamaran]"
    ],
    [
     "C103",
     "x_vars[6,Yacht]"
    ],
    [
     "C104",
     "x_vars[6,Sailboat]"
    ],
    [
     "C105",
     "x_vars[6,Kayak]"
    ],
    [
     "C106",
     "x_vars[6,Canoe]"
    ],
    [
     "C107",
     "x_vars[6,Houseboat]"
    ],
    [
     "C108",
     "x_vars[6,Pontoon]"
    ],
    [
     "C109",
     "x_vars[6,Jet Ski]"
    ],
    [
     "C110",
     "x_vars[6,Rowboat]"
    ],
    [
     "C111",
     "x_vars[6,Hovercraft]"
    ],
    [
     "C112",
     "x_vars[6,Cabin Cruiser]"
    ],
    [
     "C113",
     "x_vars[6,Wakeboard Boat]"
    ],
    [
     "C114",
     "x_vars[6,Dinghy]"
    ],
    [
     "C115",
     "x_vars[6,Trawler]"
    ],
    [
     "C116",
     "x_vars[6,Paddle Boat]"
    ],
    [
     "C117",
     "x_vars[6,Submarine]"
    ],
    [
     "C118",
     "x_vars[6,RIB]"
    ],
    [
     "C119",
     "x_vars[6,Skiff]"
    ],
    [
     "C120",
     "x_vars[7,Speedboat]"
    ],
    [
     "C121",
     "x_vars[7,Fishing Boat]"
    ],
    [
     "C122",
     "x_vars[7,Catamaran]"
    ],
    [
     "C123",
     "x_vars[7,Yacht]"
    ],
    [
     "C124",
     "x_vars[7,Sailboat]"
    ],
    [
     "C125",
     "x_vars[7,Kayak]"
    ],
    [
     "C126",
     "x_vars[7,Canoe]"
    ],
    [
     "C127",
     "x_vars[7,Houseboat]"
    ],
    [
     "C128",
     "x_vars[7,Pontoon]"
    ],
    [
     "C129",
     "x_vars[7,Jet Ski]"
    ],
    [
     "C130",
     "x_vars[7,Rowboat]"
    ],
    [
     "C131",
     "x_vars[7,Hovercraft]"
    ],
    [
     "C132",
     "x_vars[7,Cabin Cruiser]"
    ],
    [
     "C133",
     "x_vars[7,Wakeboard Boat]"
    ],
    [
     "C134",
     "x_vars[7,Dinghy]"
    ],
    [
     "C135",
     "x_vars[7,Trawler]"
    ],
    [
     "C136",
     "x_vars[7,Paddle Boat]"
    ],
    [
     "C137",
     "x_vars[7,Submarine]"
    ],
    [
     "C138",
     "x_vars[7,RIB]"
    ],
    [
     "C139",
     "x_vars[7,Skiff]"
    ],
    [
     "C140",
     "x_vars[8,Speedboat]"
    ],
    [
     "C141",
     "x_vars[8,Fishing Boat]"
    ],
    [
     "C142",
     "x_vars[8,Catamaran]"
    ],
    [
     "C143",
     "x_vars[8,Yacht]"
    ],
    [
     "C144",
     "x_vars[8,Sailboat]"
    ],
    [
     "C145",
     "x_vars[8,Kayak]"
    ],
    [
     "C146",
     "x_vars[8,Canoe]"
    ],
    [
     "C147",
     "x_vars[8,Houseboat]"
    ],
    [
     "C148",
     "x_vars[8,Pontoon]"
    ],
    [
     "C149",
     "x_vars[8,Jet Ski]"
    ],
    [
     "C150",
     "x_vars[8,Rowboat]"
    ],
    [
     "C151",
     "x_vars[8,Hovercraft]"
    ],
    [
     "C152",
     "x_vars[8,Cabin Cruiser]"
    ],
    [
     "C153",
     "x_vars[8,Wakeboard Boat]"
    ],
    [
     "C154",
     "x_vars[8,Dinghy]"
    ],
    [
     "C155",
     "x_vars[8,Trawler]"
    ],
    [
     "C156",
     "x_vars[8,Paddle Boat]"
    ],
    [
     "C157",
     "x_vars[8,Submarine]"
    ],
    [
     "C158",
     "x_vars[8,RIB]"
    ],
    [
     "C159",
     "x_vars[8,Skiff]"
    ],
    [
     "C160",
     "x_vars[9,Speedboat]"
    ],
    [
     "C161",
     "x_vars[9,Fishing Boat]"
    ],
    [
     "C162",
     "x_vars[9,Catamaran]"
    ],
    [
     "C163",
     "x_vars[9,Yacht]"
    ],
    [
     "C164",
     "x_vars[9,Sailboat]"
    ],
    [
     "C165",
     "x_vars[9,Kayak]"
    ],
    [
     "C166",
     "x_vars[9,Canoe]"
    ],
    [
     "C167",
     "x_vars[9,Houseboat]"
    ],
    [
     "C168",
     "x_vars[9,Pontoon]"
    ],
    [
     "C169",
     "x_vars[9,Jet Ski]"
    ],
    [
     "C170",
     "x_vars[9,Rowboat]"
    ],
    [
     "C171",
     "x_vars[9,Hovercraft]"
    ],
    [
     "C172",
     "x_vars[9,Cabin Cruiser]"
    ],
    [
     "C173",
     "x_vars[9,Wakeboard Boat]"
    ],
    [
     "C174",
     "x_vars[9,Dinghy]"
    ],
    [
     "C175",
     "x_vars[9,Trawler]"
    ],
    [
     "C176",
     "x_vars[9,Paddle Boat]"
    ],
    [
     "C177",
     "x_vars[9,Submarine]"
    ],
    [
     "C178",
     "x_vars[9,RIB]"
    ],
    [
     "C179",
     "x_vars[9,Skiff]"
    ],
    [
     "C180",
     "x_vars[10,Speedboat]"
    ],
    [
     "C181",
     "x_vars[10,Fishing Boat]"
    ],
    [
     "C182",
     "x_vars[10,Catamaran]"
    ],
    [
     "C183",
     "x_vars[10,Yacht]"
    ],
    [
     "C184",
     "x_vars[10,Sailboat]"
    ],
    [
     "C185",
     "x_vars[10,Kayak]"
    ],
    [
     "C186",
     "x_vars[10,Canoe]"
    ],
    [
     "C187",
     "x_vars[10,Houseboat]"
    ],
    [
     "C188",
     "x_vars[10,Pontoon]"
    ],
    [
     "C189",
     "x_vars[10,Jet Ski]"
    ],
    [
     "C190",
     "x_vars[10,Rowboat]"
    ],
    [
     "C191",
     "x_vars[10,Hovercraft]"
    ],
    [
     "C192",
     "x_vars[10,Cabin Cruiser]"
    ],
    [
     "C193",
     "x_vars[10,Wakeboard Boat]"
    ],
    [
     "C194",
     "x_vars[10,Dinghy]"
    ],
    [
     "C195",
     "x_vars[10,Trawler]"
    ],
    [
     "C196",
     "x_vars[10,Paddle Boat]"
    ],
    [
     "C197",
     "x_vars[10,Submarine]"
    ],
    [
     "C198",
     "x_vars[10,RIB]"
    ],
    [
     "C199",
     "x_vars[10,Skiff]"
    ],
    [
     "C200",
     "x_vars[11,Speedboat]"
    ],
    [
     "C201",
     "x_vars[11,Fishing Boat]"
    ],
    [
     "C202",
     "x_vars[11,Catamaran]"
    ],
    [
     "C203",
     "x_vars[11,Yacht]"
    ],
    [
     "C204",
     "x_vars[11,Sailboat]"
    ],
    [
     "C205",
     "x_vars[11,Kayak]"
    ],
    [
     "C206",
     "x_vars[11,Canoe]"
    ],
    [
     "C207",
     "x_vars[11,Houseboat]"
    ],
    [
     "C208",
     "x_vars[11,Pontoon]"
    ],
    [
     "C209",
     "x_vars[11,Jet Ski]"
    ],
    [
     "C210",
     "x_vars[11,Rowboat]"
    ],
    [
     "C211",
     "x_vars[11,Hovercraft]"
    ],
    [
     "C212",
     "x_vars[11,Cabin Cruiser]"
    ],
    [
     "C213",
     "x_vars[11,Wakeboard Boat]"
    ],
    [
     "C214",
     "x_vars[11,Dinghy]"
    ],
    [
     "C215",
     "x_vars[11,Trawler]"
    ],
    [
     "C216",
     "x_vars[11,Paddle Boat]"
    ],
    [
     "C217",
     "x_vars[11,Submarine]"
    ],
    [
     "C218",
     "x_vars[11,RIB]"
    ],
    [
     "C219",
     "x_vars[11,Skiff]"
    ],
    [
     "C220",
     "x_vars[12,Speedboat]"
    ],
    [
     "C221",
     "x_vars[12,Fishing Boat]"
    ],
    [
     "C222",
     "x_vars[12,Catamaran]"
    ],
    [
     "C223",
     "x_vars[12,Yacht]"
    ],
    [
     "C224",
     "x_vars[12,Sailboat]"
    ],
    [
     "C225",
     "x_vars[12,Kayak]"
    ],
    [
     "C226",
     "x_vars[12,Canoe]"
    ],
    [
     "C227",
     "x_vars[12,Houseboat]"
    ],
    [
     "C228",
     "x_vars[12,Pontoon]"
    ],
    [
     "C229",
     "x_vars[12,Jet Ski]"
    ],
    [
     "C230",
     "x_vars[12,Rowboat]"
    ],
    [
     "C231",
     "x_vars[12,Hovercraft]"
    ],
    [
     "C232",
     "x_vars[12,Cabin Cruiser]"
    ],
    [
     "C233",
     "x_vars[12,Wakeboard Boat]"
    ],
    [
     "C234",
     "x_vars[12,Dinghy]"
    ],
    [
     "C235",
     "x_vars[12,Trawler]"
    ],
    [
     "C236",
     "x_vars[12,Paddle Boat]"
    ],
    [
     "C237",
     "x_vars[12,Submarine]"
    ],
    [
     "C238",
     "x_vars[12,RIB]"
    ],
    [
     "C239",
     "x_vars[12,Skiff]"
    ],
    [
     "C240",
     "x_vars[13,Speedboat]"
    ],
    [
     "C241",
     "x_vars[13,Fishing Boat]"
    ],
    [
     "C242",
     "x_vars[13,Catamaran]"
    ],
    [
     "C243",
     "x_vars[13,Yacht]"
    ],
    [
     "C244",
     "x_vars[13,Sailboat]"
    ],
    [
     "C245",
     "x_vars[13,Kayak]"
    ],
    [
     "C246",
     "x_vars[13,Canoe]"
    ],
    [
     "C247",
     "x_vars[13,Houseboat]"
    ],
    [
     "C248",
     "x_vars[13,Pontoon]"
    ],
    [
     "C249",
     "x_vars[13,Jet Ski]"
    ],
    [
     "C250",
     "x_vars[13,Rowboat]"
    ],
    [
     "C251",
     "x_vars[13,Hovercraft]"
    ],
    [
     "C252",
     "x_vars[13,Cabin Cruiser]"
    ],
    [
     "C253",
     "x_vars[13,Wakeboard Boat]"
    ],
    [
     "C254",
     "x_vars[13,Dinghy]"
    ],
    [
     "C255",
     "x_vars[13,Trawler]"
    ],
    [
     "C256",
     "x_vars[13,Paddle Boat]"
    ],
    [
     "C257",
     "x_vars[13,Submarine]"
    ],
    [
     "C258",
     "x_vars[13,RIB]"
    ],
    [
     "C259",
     "x_vars[13,Skiff]"
    ],
    [
     "C260",
     "x_vars[14,Speedboat]"
    ],
    [
     "C261",
     "x_vars[14,Fishing Boat]"
    ],
    [
     "C262",
     "x_vars[14,Catamaran]"
    ],
    [
     "C263",
     "x_vars[14,Yacht]"
    ],
    [
     "C264",
     "x_vars[14,Sailboat]"
    ],
    [
     "C265",
     "x_vars[14,Kayak]"
    ],
    [
     "C266",
     "x_vars[14,Canoe]"
    ],
    [
     "C267",
     "x_vars[14,Houseboat]"
    ],
    [
     "C268",
     "x_vars[14,Pontoon]"
    ],
    [
     "C269",
     "x_vars[14,Jet Ski]"
    ],
    [
     "C270",
     "x_vars[14,Rowboat]"
    ],
    [
     "C271",
     "x_vars[14,Hovercraft]"
    ],
    [
     "C272",
     "x_vars[14,Cabin Cruiser]"
    ],
    [
     "C273",
     "x_vars[14,Wakeboard Boat]"
    ],
    [
     "C274",
     "x_vars[14,Dinghy]"
    ],
    [
     "C275",
     "x_vars[14,Trawler]"
    ],
    [
     "C276",
     "x_vars[14,Paddle Boat]"
    ],
    [
     "C277",
     "x_vars[14,Submarine]"
    ],
    [
     "C278",
     "x_vars[14,RIB]"
    ],
    [
     "C279",
     "x_vars[14,Skiff]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Speedboat]",
    "x_0_1": "x_vars[2,Speedboat]",
    "x_0_2": "x_vars[3,Speedboat]",
    "x_0_3": "x_vars[4,Speedboat]",
    "x_0_4": "x_vars[5,Speedboat]",
    "x_0_5": "x_vars[6,Speedboat]",
    "x_0_6": "x_vars[7,Speedboat]",
    "x_0_7": "x_vars[8,Speedboat]",
    "x_0_8": "x_vars[9,Speedboat]",
    "x_0_9": "x_vars[10,Speedboat]",
    "x_0_10": "x_vars[11,Speedboat]",
    "x_0_11": "x_vars[12,Speedboat]",
    "x_0_12": "x_vars[13,Speedboat]",
    "x_0_13": "x_vars[14,Speedboat]",
    "x_1_0": "x_vars[1,Fishing Boat]",
    "x_1_1": "x_vars[2,Fishing Boat]",
    "x_1_2": "x_vars[3,Fishing Boat]",
    "x_1_3": "x_vars[4,Fishing Boat]",
    "x_1_4": "x_vars[5,Fishing Boat]",
    "x_1_5": "x_vars[6,Fishing Boat]",
    "x_1_6": "x_vars[7,Fishing Boat]",
    "x_1_7": "x_vars[8,Fishing Boat]",
    "x_1_8": "x_vars[9,Fishing Boat]",
    "x_1_9": "x_vars[10,Fishing Boat]",
    "x_1_10": "x_vars[11,Fishing Boat]",
    "x_1_11": "x_vars[12,Fishing Boat]",
    "x_1_12": "x_vars[13,Fishing Boat]",
    "x_1_13": "x_vars[14,Fishing Boat]",
    "x_2_0": "x_vars[1,Catamaran]",
    "x_2_1": "x_vars[2,Catamaran]",
    "x_2_2": "x_vars[3,Catamaran]",
    "x_2_3": "x_vars[4,Catamaran]",
    "x_2_4": "x_vars[5,Catamaran]",
    "x_2_5": "x_vars[6,Catamaran]",
    "x_2_6": "x_vars[7,Catamaran]",
    "x_2_7": "x_vars[8,Catamaran]",
    "x_2_8": "x_vars[9,Catamaran]",
    "x_2_9": "x_vars[10,Catamaran]",
    "x_2_10": "x_vars[11,Catamaran]",
    "x_2_11": "x_vars[12,Catamaran]",
    "x_2_12": "x_vars[13,Catamaran]",
    "x_2_13": "x_vars[14,Catamaran]",
    "x_3_0": "x_vars[1,Yacht]",
    "x_3_1": "x_vars[2,Yacht]",
    "x_3_2": "x_vars[3,Yacht]",
    "x_3_3": "x_vars[4,Yacht]",
    "x_3_4": "x_vars[5,Yacht]",
    "x_3_5": "x_vars[6,Yacht]",
    "x_3_6": "x_vars[7,Yacht]",
    "x_3_7": "x_vars[8,Yacht]",
    "x_3_8": "x_vars[9,Yacht]",
    "x_3_9": "x_vars[10,Yacht]",
    "x_3_10": "x_vars[11,Yacht]",
    "x_3_11": "x_vars[12,Yacht]",
    "x_3_12": "x_vars[13,Yacht]",
    "x_3_13": "x_vars[14,Yacht]",
    "x_4_0": "x_vars[1,Sailboat]",
    "x_4_1": "x_vars[2,Sailboat]",
    "x_4_2": "x_vars[3,Sailboat]",
    "x_4_3": "x_vars[4,Sailboat]",
    "x_4_4": "x_vars[5,Sailboat]",
    "x_4_5": "x_vars[6,Sailboat]",
    "x_4_6": "x_vars[7,Sailboat]",
    "x_4_7": "x_vars[8,Sailboat]",
    "x_4_8": "x_vars[9,Sailboat]",
    "x_4_9": "x_vars[10,Sailboat]",
    "x_4_10": "x_vars[11,Sailboat]",
    "x_4_11": "x_vars[12,Sailboat]",
    "x_4_12": "x_vars[13,Sailboat]",
    "x_4_13": "x_vars[14,Sailboat]",
    "x_5_0": "x_vars[1,Kayak]",
    "x_5_1": "x_vars[2,Kayak]",
    "x_5_2": "x_vars[3,Kayak]",
    "x_5_3": "x_vars[4,Kayak]",
    "x_5_4": "x_vars[5,Kayak]",
    "x_5_5": "x_vars[6,Kayak]",
    "x_5_6": "x_vars[7,Kayak]",
    "x_5_7": "x_vars[8,Kayak]",
    "x_5_8": "x_vars[9,Kayak]",
    "x_5_9": "x_vars[10,Kayak]",
    "x_5_10": "x_vars[11,Kayak]",
    "x_5_11": "x_vars[12,Kayak]",
    "x_5_12": "x_vars[13,Kayak]",
    "x_5_13": "x_vars[14,Kayak]",
    "x_6_0": "x_vars[1,Canoe]",
    "x_6_1": "x_vars[2,Canoe]",
    "x_6_2": "x_vars[3,Canoe]",
    "x_6_3": "x_vars[4,Canoe]",
    "x_6_4": "x_vars[5,Canoe]",
    "x_6_5": "x_vars[6,Canoe]",
    "x_6_6": "x_vars[7,Canoe]",
    "x_6_7": "x_vars[8,Canoe]",
    "x_6_8": "x_vars[9,Canoe]",
    "x_6_9": "x_vars[10,Canoe]",
    "x_6_10": "x_vars[11,Canoe]",
    "x_6_11": "x_vars[12,Canoe]",
    "x_6_12": "x_vars[13,Canoe]",
    "x_6_13": "x_vars[14,Canoe]",
    "x_7_0": "x_vars[1,Houseboat]",
    "x_7_1": "x_vars[2,Houseboat]",
    "x_7_2": "x_vars[3,Houseboat]",
    "x_7_3": "x_vars[4,Houseboat]",
    "x_7_4": "x_vars[5,Houseboat]",
    "x_7_5": "x_vars[6,Houseboat]",
    "x_7_6": "x_vars[7,Houseboat]",
    "x_7_7": "x_vars[8,Houseboat]",
    "x_7_8": "x_vars[9,Houseboat]",
    "x_7_9": "x_vars[10,Houseboat]",
    "x_7_10": "x_vars[11,Houseboat]",
    "x_7_11": "x_vars[12,Houseboat]",
    "x_7_12": "x_vars[13,Houseboat]",
    "x_7_13": "x_vars[14,Houseboat]",
    "x_8_0": "x_vars[1,Pontoon]",
    "x_8_1": "x_vars[2,Pontoon]",
    "x_8_2": "x_vars[3,Pontoon]",
    "x_8_3": "x_vars[4,Pontoon]",
    "x_8_4": "x_vars[5,Pontoon]",
    "x_8_5": "x_vars[6,Pontoon]",
    "x_8_6": "x_vars[7,Pontoon]",
    "x_8_7": "x_vars[8,Pontoon]",
    "x_8_8": "x_vars[9,Pontoon]",
    "x_8_9": "x_vars[10,Pontoon]",
    "x_8_10": "x_vars[11,Pontoon]",
    "x_8_11": "x_vars[12,Pontoon]",
    "x_8_12": "x_vars[13,Pontoon]",
    "x_8_13": "x_vars[14,Pontoon]",
    "x_9_0": "x_vars[1,Jet Ski]",
    "x_9_1": "x_vars[2,Jet Ski]",
    "x_9_2": "x_vars[3,Jet Ski]",
    "x_9_3": "x_vars[4,Jet Ski]",
    "x_9_4": "x_vars[5,Jet Ski]",
    "x_9_5": "x_vars[6,Jet Ski]",
    "x_9_6": "x_vars[7,Jet Ski]",
    "x_9_7": "x_vars[8,Jet Ski]",
    "x_9_8": "x_vars[9,Jet Ski]",
    "x_9_9": "x_vars[10,Jet Ski]",
    "x_9_10": "x_vars[11,Jet Ski]",
    "x_9_11": "x_vars[12,Jet Ski]",
    "x_9_12": "x_vars[13,Jet Ski]",
    "x_9_13": "x_vars[14,Jet Ski]",
    "x_10_0": "x_vars[1,Rowboat]",
    "x_10_1": "x_vars[2,Rowboat]",
    "x_10_2": "x_vars[3,Rowboat]",
    "x_10_3": "x_vars[4,Rowboat]",
    "x_10_4": "x_vars[5,Rowboat]",
    "x_10_5": "x_vars[6,Rowboat]",
    "x_10_6": "x_vars[7,Rowboat]",
    "x_10_7": "x_vars[8,Rowboat]",
    "x_10_8": "x_vars[9,Rowboat]",
    "x_10_9": "x_vars[10,Rowboat]",
    "x_10_10": "x_vars[11,Rowboat]",
    "x_10_11": "x_vars[12,Rowboat]",
    "x_10_12": "x_vars[13,Rowboat]",
    "x_10_13": "x_vars[14,Rowboat]",
    "x_11_0": "x_vars[1,Hovercraft]",
    "x_11_1": "x_vars[2,Hovercraft]",
    "x_11_2": "x_vars[3,Hovercraft]",
    "x_11_3": "x_vars[4,Hovercraft]",
    "x_11_4": "x_vars[5,Hovercraft]",
    "x_11_5": "x_vars[6,Hovercraft]",
    "x_11_6": "x_vars[7,Hovercraft]",
    "x_11_7": "x_vars[8,Hovercraft]",
    "x_11_8": "x_vars[9,Hovercraft]",
    "x_11_9": "x_vars[10,Hovercraft]",
    "x_11_10": "x_vars[11,Hovercraft]",
    "x_11_11": "x_vars[12,Hovercraft]",
    "x_11_12": "x_vars[13,Hovercraft]",
    "x_11_13": "x_vars[14,Hovercraft]",
    "x_12_0": "x_vars[1,Cabin Cruiser]",
    "x_12_1": "x_vars[2,Cabin Cruiser]",
    "x_12_2": "x_vars[3,Cabin Cruiser]",
    "x_12_3": "x_vars[4,Cabin Cruiser]",
    "x_12_4": "x_vars[5,Cabin Cruiser]",
    "x_12_5": "x_vars[6,Cabin Cruiser]",
    "x_12_6": "x_vars[7,Cabin Cruiser]",
    "x_12_7": "x_vars[8,Cabin Cruiser]",
    "x_12_8": "x_vars[9,Cabin Cruiser]",
    "x_12_9": "x_vars[10,Cabin Cruiser]",
    "x_12_10": "x_vars[11,Cabin Cruiser]",
    "x_12_11": "x_vars[12,Cabin Cruiser]",
    "x_12_12": "x_vars[13,Cabin Cruiser]",
    "x_12_13": "x_vars[14,Cabin Cruiser]",
    "x_13_0": "x_vars[1,Wakeboard Boat]",
    "x_13_1": "x_vars[2,Wakeboard Boat]",
    "x_13_2": "x_vars[3,Wakeboard Boat]",
    "x_13_3": "x_vars[4,Wakeboard Boat]",
    "x_13_4": "x_vars[5,Wakeboard Boat]",
    "x_13_5": "x_vars[6,Wakeboard Boat]",
    "x_13_6": "x_vars[7,Wakeboard Boat]",
    "x_13_7": "x_vars[8,Wakeboard Boat]",
    "x_13_8": "x_vars[9,Wakeboard Boat]",
    "x_13_9": "x_vars[10,Wakeboard Boat]",
    "x_13_10": "x_vars[11,Wakeboard Boat]",
    "x_13_11": "x_vars[12,Wakeboard Boat]",
    "x_13_12": "x_vars[13,Wakeboard Boat]",
    "x_13_13": "x_vars[14,Wakeboard Boat]",
    "x_14_0": "x_vars[1,Dinghy]",
    "x_14_1": "x_vars[2,Dinghy]",
    "x_14_2": "x_vars[3,Dinghy]",
    "x_14_3": "x_vars[4,Dinghy]",
    "x_14_4": "x_vars[5,Dinghy]",
    "x_14_5": "x_vars[6,Dinghy]",
    "x_14_6": "x_vars[7,Dinghy]",
    "x_14_7": "x_vars[8,Dinghy]",
    "x_14_8": "x_vars[9,Dinghy]",
    "x_14_9": "x_vars[10,Dinghy]",
    "x_14_10": "x_vars[11,Dinghy]",
    "x_14_11": "x_vars[12,Dinghy]",
    "x_14_12": "x_vars[13,Dinghy]",
    "x_14_13": "x_vars[14,Dinghy]",
    "x_15_0": "x_vars[1,Trawler]",
    "x_15_1": "x_vars[2,Trawler]",
    "x_15_2": "x_vars[3,Trawler]",
    "x_15_3": "x_vars[4,Trawler]",
    "x_15_4": "x_vars[5,Trawler]",
    "x_15_5": "x_vars[6,Trawler]",
    "x_15_6": "x_vars[7,Trawler]",
    "x_15_7": "x_vars[8,Trawler]",
    "x_15_8": "x_vars[9,Trawler]",
    "x_15_9": "x_vars[10,Trawler]",
    "x_15_10": "x_vars[11,Trawler]",
    "x_15_11": "x_vars[12,Trawler]",
    "x_15_12": "x_vars[13,Trawler]",
    "x_15_13": "x_vars[14,Trawler]",
    "x_16_0": "x_vars[1,Paddle Boat]",
    "x_16_1": "x_vars[2,Paddle Boat]",
    "x_16_2": "x_vars[3,Paddle Boat]",
    "x_16_3": "x_vars[4,Paddle Boat]",
    "x_16_4": "x_vars[5,Paddle Boat]",
    "x_16_5": "x_vars[6,Paddle Boat]",
    "x_16_6": "x_vars[7,Paddle Boat]",
    "x_16_7": "x_vars[8,Paddle Boat]",
    "x_16_8": "x_vars[9,Paddle Boat]",
    "x_16_9": "x_vars[10,Paddle Boat]",
    "x_16_10": "x_vars[11,Paddle Boat]",
    "x_16_11": "x_vars[12,Paddle Boat]",
    "x_16_12": "x_vars[13,Paddle Boat]",
    "x_16_13": "x_vars[14,Paddle Boat]",
    "x_17_0": "x_vars[1,Submarine]",
    "x_17_1": "x_vars[2,Submarine]",
    "x_17_2": "x_vars[3,Submarine]",
    "x_17_3": "x_vars[4,Submarine]",
    "x_17_4": "x_vars[5,Submarine]",
    "x_17_5": "x_vars[6,Submarine]",
    "x_17_6": "x_vars[7,Submarine]",
    "x_17_7": "x_vars[8,Submarine]",
    "x_17_8": "x_vars[9,Submarine]",
    "x_17_9": "x_vars[10,Submarine]",
    "x_17_10": "x_vars[11,Submarine]",
    "x_17_11": "x_vars[12,Submarine]",
    "x_17_12": "x_vars[13,Submarine]",
    "x_17_13": "x_vars[14,Submarine]",
    "x_18_0": "x_vars[1,RIB]",
    "x_18_1": "x_vars[2,RIB]",
    "x_18_2": "x_vars[3,RIB]",
    "x_18_3": "x_vars[4,RIB]",
    "x_18_4": "x_vars[5,RIB]",
    "x_18_5": "x_vars[6,RIB]",
    "x_18_6": "x_vars[7,RIB]",
    "x_18_7": "x_vars[8,RIB]",
    "x_18_8": "x_vars[9,RIB]",
    "x_18_9": "x_vars[10,RIB]",
    "x_18_10": "x_vars[11,RIB]",
    "x_18_11": "x_vars[12,RIB]",
    "x_18_12": "x_vars[13,RIB]",
    "x_18_13": "x_vars[14,RIB]",
    "x_19_0": "x_vars[1,Skiff]",
    "x_19_1": "x_vars[2,Skiff]",
    "x_19_2": "x_vars[3,Skiff]",
    "x_19_3": "x_vars[4,Skiff]",
    "x_19_4": "x_vars[5,Skiff]",
    "x_19_5": "x_vars[6,Skiff]",
    "x_19_6": "x_vars[7,Skiff]",
    "x_19_7": "x_vars[8,Skiff]",
    "x_19_8": "x_vars[9,Skiff]",
    "x_19_9": "x_vars[10,Skiff]",
    "x_19_10": "x_vars[11,Skiff]",
    "x_19_11": "x_vars[12,Skiff]",
    "x_19_12": "x_vars[13,Skiff]",
    "x_19_13": "x_vars[14,Skiff]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-057": {
   "source_code_sha256": "63e3a2fe8288cda6f5f77cd047f1e29bccbaafcdccdfb04fb9930de429fc0759",
   "original_solution_sha256": "2a181e28654bd02a714bfbb8e8ca9ce75d92efa42887dd7e48b5d52c1f84b74e",
   "bindings": [
    [
     "C0",
     "x_vars[1,Racing]"
    ],
    [
     "C1",
     "x_vars[1,Sports]"
    ],
    [
     "C2",
     "x_vars[1,Action]"
    ],
    [
     "C3",
     "x_vars[1,Adventure]"
    ],
    [
     "C4",
     "x_vars[1,RPG]"
    ],
    [
     "C5",
     "x_vars[1,Shooter]"
    ],
    [
     "C6",
     "x_vars[1,Strategy]"
    ],
    [
     "C7",
     "x_vars[1,Simulation]"
    ],
    [
     "C8",
     "x_vars[1,Puzzle]"
    ],
    [
     "C9",
     "x_vars[1,Fighting]"
    ],
    [
     "C10",
     "x_vars[1,Platformer]"
    ],
    [
     "C11",
     "x_vars[1,Survival]"
    ],
    [
     "C12",
     "x_vars[1,Horror]"
    ],
    [
     "C13",
     "x_vars[1,Sandbox]"
    ],
    [
     "C14",
     "x_vars[1,MMO]"
    ],
    [
     "C15",
     "x_vars[2,Racing]"
    ],
    [
     "C16",
     "x_vars[2,Sports]"
    ],
    [
     "C17",
     "x_vars[2,Action]"
    ],
    [
     "C18",
     "x_vars[2,Adventure]"
    ],
    [
     "C19",
     "x_vars[2,RPG]"
    ],
    [
     "C20",
     "x_vars[2,Shooter]"
    ],
    [
     "C21",
     "x_vars[2,Strategy]"
    ],
    [
     "C22",
     "x_vars[2,Simulation]"
    ],
    [
     "C23",
     "x_vars[2,Puzzle]"
    ],
    [
     "C24",
     "x_vars[2,Fighting]"
    ],
    [
     "C25",
     "x_vars[2,Platformer]"
    ],
    [
     "C26",
     "x_vars[2,Survival]"
    ],
    [
     "C27",
     "x_vars[2,Horror]"
    ],
    [
     "C28",
     "x_vars[2,Sandbox]"
    ],
    [
     "C29",
     "x_vars[2,MMO]"
    ],
    [
     "C30",
     "x_vars[3,Racing]"
    ],
    [
     "C31",
     "x_vars[3,Sports]"
    ],
    [
     "C32",
     "x_vars[3,Action]"
    ],
    [
     "C33",
     "x_vars[3,Adventure]"
    ],
    [
     "C34",
     "x_vars[3,RPG]"
    ],
    [
     "C35",
     "x_vars[3,Shooter]"
    ],
    [
     "C36",
     "x_vars[3,Strategy]"
    ],
    [
     "C37",
     "x_vars[3,Simulation]"
    ],
    [
     "C38",
     "x_vars[3,Puzzle]"
    ],
    [
     "C39",
     "x_vars[3,Fighting]"
    ],
    [
     "C40",
     "x_vars[3,Platformer]"
    ],
    [
     "C41",
     "x_vars[3,Survival]"
    ],
    [
     "C42",
     "x_vars[3,Horror]"
    ],
    [
     "C43",
     "x_vars[3,Sandbox]"
    ],
    [
     "C44",
     "x_vars[3,MMO]"
    ],
    [
     "C45",
     "x_vars[4,Racing]"
    ],
    [
     "C46",
     "x_vars[4,Sports]"
    ],
    [
     "C47",
     "x_vars[4,Action]"
    ],
    [
     "C48",
     "x_vars[4,Adventure]"
    ],
    [
     "C49",
     "x_vars[4,RPG]"
    ],
    [
     "C50",
     "x_vars[4,Shooter]"
    ],
    [
     "C51",
     "x_vars[4,Strategy]"
    ],
    [
     "C52",
     "x_vars[4,Simulation]"
    ],
    [
     "C53",
     "x_vars[4,Puzzle]"
    ],
    [
     "C54",
     "x_vars[4,Fighting]"
    ],
    [
     "C55",
     "x_vars[4,Platformer]"
    ],
    [
     "C56",
     "x_vars[4,Survival]"
    ],
    [
     "C57",
     "x_vars[4,Horror]"
    ],
    [
     "C58",
     "x_vars[4,Sandbox]"
    ],
    [
     "C59",
     "x_vars[4,MMO]"
    ],
    [
     "C60",
     "x_vars[5,Racing]"
    ],
    [
     "C61",
     "x_vars[5,Sports]"
    ],
    [
     "C62",
     "x_vars[5,Action]"
    ],
    [
     "C63",
     "x_vars[5,Adventure]"
    ],
    [
     "C64",
     "x_vars[5,RPG]"
    ],
    [
     "C65",
     "x_vars[5,Shooter]"
    ],
    [
     "C66",
     "x_vars[5,Strategy]"
    ],
    [
     "C67",
     "x_vars[5,Simulation]"
    ],
    [
     "C68",
     "x_vars[5,Puzzle]"
    ],
    [
     "C69",
     "x_vars[5,Fighting]"
    ],
    [
     "C70",
     "x_vars[5,Platformer]"
    ],
    [
     "C71",
     "x_vars[5,Survival]"
    ],
    [
     "C72",
     "x_vars[5,Horror]"
    ],
    [
     "C73",
     "x_vars[5,Sandbox]"
    ],
    [
     "C74",
     "x_vars[5,MMO]"
    ],
    [
     "C75",
     "x_vars[6,Racing]"
    ],
    [
     "C76",
     "x_vars[6,Sports]"
    ],
    [
     "C77",
     "x_vars[6,Action]"
    ],
    [
     "C78",
     "x_vars[6,Adventure]"
    ],
    [
     "C79",
     "x_vars[6,RPG]"
    ],
    [
     "C80",
     "x_vars[6,Shooter]"
    ],
    [
     "C81",
     "x_vars[6,Strategy]"
    ],
    [
     "C82",
     "x_vars[6,Simulation]"
    ],
    [
     "C83",
     "x_vars[6,Puzzle]"
    ],
    [
     "C84",
     "x_vars[6,Fighting]"
    ],
    [
     "C85",
     "x_vars[6,Platformer]"
    ],
    [
     "C86",
     "x_vars[6,Survival]"
    ],
    [
     "C87",
     "x_vars[6,Horror]"
    ],
    [
     "C88",
     "x_vars[6,Sandbox]"
    ],
    [
     "C89",
     "x_vars[6,MMO]"
    ],
    [
     "C90",
     "x_vars[7,Racing]"
    ],
    [
     "C91",
     "x_vars[7,Sports]"
    ],
    [
     "C92",
     "x_vars[7,Action]"
    ],
    [
     "C93",
     "x_vars[7,Adventure]"
    ],
    [
     "C94",
     "x_vars[7,RPG]"
    ],
    [
     "C95",
     "x_vars[7,Shooter]"
    ],
    [
     "C96",
     "x_vars[7,Strategy]"
    ],
    [
     "C97",
     "x_vars[7,Simulation]"
    ],
    [
     "C98",
     "x_vars[7,Puzzle]"
    ],
    [
     "C99",
     "x_vars[7,Fighting]"
    ],
    [
     "C100",
     "x_vars[7,Platformer]"
    ],
    [
     "C101",
     "x_vars[7,Survival]"
    ],
    [
     "C102",
     "x_vars[7,Horror]"
    ],
    [
     "C103",
     "x_vars[7,Sandbox]"
    ],
    [
     "C104",
     "x_vars[7,MMO]"
    ],
    [
     "C105",
     "x_vars[8,Racing]"
    ],
    [
     "C106",
     "x_vars[8,Sports]"
    ],
    [
     "C107",
     "x_vars[8,Action]"
    ],
    [
     "C108",
     "x_vars[8,Adventure]"
    ],
    [
     "C109",
     "x_vars[8,RPG]"
    ],
    [
     "C110",
     "x_vars[8,Shooter]"
    ],
    [
     "C111",
     "x_vars[8,Strategy]"
    ],
    [
     "C112",
     "x_vars[8,Simulation]"
    ],
    [
     "C113",
     "x_vars[8,Puzzle]"
    ],
    [
     "C114",
     "x_vars[8,Fighting]"
    ],
    [
     "C115",
     "x_vars[8,Platformer]"
    ],
    [
     "C116",
     "x_vars[8,Survival]"
    ],
    [
     "C117",
     "x_vars[8,Horror]"
    ],
    [
     "C118",
     "x_vars[8,Sandbox]"
    ],
    [
     "C119",
     "x_vars[8,MMO]"
    ],
    [
     "C120",
     "x_vars[9,Racing]"
    ],
    [
     "C121",
     "x_vars[9,Sports]"
    ],
    [
     "C122",
     "x_vars[9,Action]"
    ],
    [
     "C123",
     "x_vars[9,Adventure]"
    ],
    [
     "C124",
     "x_vars[9,RPG]"
    ],
    [
     "C125",
     "x_vars[9,Shooter]"
    ],
    [
     "C126",
     "x_vars[9,Strategy]"
    ],
    [
     "C127",
     "x_vars[9,Simulation]"
    ],
    [
     "C128",
     "x_vars[9,Puzzle]"
    ],
    [
     "C129",
     "x_vars[9,Fighting]"
    ],
    [
     "C130",
     "x_vars[9,Platformer]"
    ],
    [
     "C131",
     "x_vars[9,Survival]"
    ],
    [
     "C132",
     "x_vars[9,Horror]"
    ],
    [
     "C133",
     "x_vars[9,Sandbox]"
    ],
    [
     "C134",
     "x_vars[9,MMO]"
    ],
    [
     "C135",
     "x_vars[10,Racing]"
    ],
    [
     "C136",
     "x_vars[10,Sports]"
    ],
    [
     "C137",
     "x_vars[10,Action]"
    ],
    [
     "C138",
     "x_vars[10,Adventure]"
    ],
    [
     "C139",
     "x_vars[10,RPG]"
    ],
    [
     "C140",
     "x_vars[10,Shooter]"
    ],
    [
     "C141",
     "x_vars[10,Strategy]"
    ],
    [
     "C142",
     "x_vars[10,Simulation]"
    ],
    [
     "C143",
     "x_vars[10,Puzzle]"
    ],
    [
     "C144",
     "x_vars[10,Fighting]"
    ],
    [
     "C145",
     "x_vars[10,Platformer]"
    ],
    [
     "C146",
     "x_vars[10,Survival]"
    ],
    [
     "C147",
     "x_vars[10,Horror]"
    ],
    [
     "C148",
     "x_vars[10,Sandbox]"
    ],
    [
     "C149",
     "x_vars[10,MMO]"
    ],
    [
     "C150",
     "x_vars[11,Racing]"
    ],
    [
     "C151",
     "x_vars[11,Sports]"
    ],
    [
     "C152",
     "x_vars[11,Action]"
    ],
    [
     "C153",
     "x_vars[11,Adventure]"
    ],
    [
     "C154",
     "x_vars[11,RPG]"
    ],
    [
     "C155",
     "x_vars[11,Shooter]"
    ],
    [
     "C156",
     "x_vars[11,Strategy]"
    ],
    [
     "C157",
     "x_vars[11,Simulation]"
    ],
    [
     "C158",
     "x_vars[11,Puzzle]"
    ],
    [
     "C159",
     "x_vars[11,Fighting]"
    ],
    [
     "C160",
     "x_vars[11,Platformer]"
    ],
    [
     "C161",
     "x_vars[11,Survival]"
    ],
    [
     "C162",
     "x_vars[11,Horror]"
    ],
    [
     "C163",
     "x_vars[11,Sandbox]"
    ],
    [
     "C164",
     "x_vars[11,MMO]"
    ],
    [
     "C165",
     "x_vars[12,Racing]"
    ],
    [
     "C166",
     "x_vars[12,Sports]"
    ],
    [
     "C167",
     "x_vars[12,Action]"
    ],
    [
     "C168",
     "x_vars[12,Adventure]"
    ],
    [
     "C169",
     "x_vars[12,RPG]"
    ],
    [
     "C170",
     "x_vars[12,Shooter]"
    ],
    [
     "C171",
     "x_vars[12,Strategy]"
    ],
    [
     "C172",
     "x_vars[12,Simulation]"
    ],
    [
     "C173",
     "x_vars[12,Puzzle]"
    ],
    [
     "C174",
     "x_vars[12,Fighting]"
    ],
    [
     "C175",
     "x_vars[12,Platformer]"
    ],
    [
     "C176",
     "x_vars[12,Survival]"
    ],
    [
     "C177",
     "x_vars[12,Horror]"
    ],
    [
     "C178",
     "x_vars[12,Sandbox]"
    ],
    [
     "C179",
     "x_vars[12,MMO]"
    ],
    [
     "C180",
     "x_vars[13,Racing]"
    ],
    [
     "C181",
     "x_vars[13,Sports]"
    ],
    [
     "C182",
     "x_vars[13,Action]"
    ],
    [
     "C183",
     "x_vars[13,Adventure]"
    ],
    [
     "C184",
     "x_vars[13,RPG]"
    ],
    [
     "C185",
     "x_vars[13,Shooter]"
    ],
    [
     "C186",
     "x_vars[13,Strategy]"
    ],
    [
     "C187",
     "x_vars[13,Simulation]"
    ],
    [
     "C188",
     "x_vars[13,Puzzle]"
    ],
    [
     "C189",
     "x_vars[13,Fighting]"
    ],
    [
     "C190",
     "x_vars[13,Platformer]"
    ],
    [
     "C191",
     "x_vars[13,Survival]"
    ],
    [
     "C192",
     "x_vars[13,Horror]"
    ],
    [
     "C193",
     "x_vars[13,Sandbox]"
    ],
    [
     "C194",
     "x_vars[13,MMO]"
    ],
    [
     "C195",
     "x_vars[14,Racing]"
    ],
    [
     "C196",
     "x_vars[14,Sports]"
    ],
    [
     "C197",
     "x_vars[14,Action]"
    ],
    [
     "C198",
     "x_vars[14,Adventure]"
    ],
    [
     "C199",
     "x_vars[14,RPG]"
    ],
    [
     "C200",
     "x_vars[14,Shooter]"
    ],
    [
     "C201",
     "x_vars[14,Strategy]"
    ],
    [
     "C202",
     "x_vars[14,Simulation]"
    ],
    [
     "C203",
     "x_vars[14,Puzzle]"
    ],
    [
     "C204",
     "x_vars[14,Fighting]"
    ],
    [
     "C205",
     "x_vars[14,Platformer]"
    ],
    [
     "C206",
     "x_vars[14,Survival]"
    ],
    [
     "C207",
     "x_vars[14,Horror]"
    ],
    [
     "C208",
     "x_vars[14,Sandbox]"
    ],
    [
     "C209",
     "x_vars[14,MMO]"
    ],
    [
     "C210",
     "x_vars[15,Racing]"
    ],
    [
     "C211",
     "x_vars[15,Sports]"
    ],
    [
     "C212",
     "x_vars[15,Action]"
    ],
    [
     "C213",
     "x_vars[15,Adventure]"
    ],
    [
     "C214",
     "x_vars[15,RPG]"
    ],
    [
     "C215",
     "x_vars[15,Shooter]"
    ],
    [
     "C216",
     "x_vars[15,Strategy]"
    ],
    [
     "C217",
     "x_vars[15,Simulation]"
    ],
    [
     "C218",
     "x_vars[15,Puzzle]"
    ],
    [
     "C219",
     "x_vars[15,Fighting]"
    ],
    [
     "C220",
     "x_vars[15,Platformer]"
    ],
    [
     "C221",
     "x_vars[15,Survival]"
    ],
    [
     "C222",
     "x_vars[15,Horror]"
    ],
    [
     "C223",
     "x_vars[15,Sandbox]"
    ],
    [
     "C224",
     "x_vars[15,MMO]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[1,Racing]",
    "x_0_1": "x_vars[2,Racing]",
    "x_0_2": "x_vars[3,Racing]",
    "x_0_3": "x_vars[4,Racing]",
    "x_0_4": "x_vars[5,Racing]",
    "x_0_5": "x_vars[6,Racing]",
    "x_0_6": "x_vars[7,Racing]",
    "x_0_7": "x_vars[8,Racing]",
    "x_0_8": "x_vars[9,Racing]",
    "x_0_9": "x_vars[10,Racing]",
    "x_0_10": "x_vars[11,Racing]",
    "x_0_11": "x_vars[12,Racing]",
    "x_0_12": "x_vars[13,Racing]",
    "x_0_13": "x_vars[14,Racing]",
    "x_0_14": "x_vars[15,Racing]",
    "x_1_0": "x_vars[1,Sports]",
    "x_1_1": "x_vars[2,Sports]",
    "x_1_2": "x_vars[3,Sports]",
    "x_1_3": "x_vars[4,Sports]",
    "x_1_4": "x_vars[5,Sports]",
    "x_1_5": "x_vars[6,Sports]",
    "x_1_6": "x_vars[7,Sports]",
    "x_1_7": "x_vars[8,Sports]",
    "x_1_8": "x_vars[9,Sports]",
    "x_1_9": "x_vars[10,Sports]",
    "x_1_10": "x_vars[11,Sports]",
    "x_1_11": "x_vars[12,Sports]",
    "x_1_12": "x_vars[13,Sports]",
    "x_1_13": "x_vars[14,Sports]",
    "x_1_14": "x_vars[15,Sports]",
    "x_2_0": "x_vars[1,Action]",
    "x_2_1": "x_vars[2,Action]",
    "x_2_2": "x_vars[3,Action]",
    "x_2_3": "x_vars[4,Action]",
    "x_2_4": "x_vars[5,Action]",
    "x_2_5": "x_vars[6,Action]",
    "x_2_6": "x_vars[7,Action]",
    "x_2_7": "x_vars[8,Action]",
    "x_2_8": "x_vars[9,Action]",
    "x_2_9": "x_vars[10,Action]",
    "x_2_10": "x_vars[11,Action]",
    "x_2_11": "x_vars[12,Action]",
    "x_2_12": "x_vars[13,Action]",
    "x_2_13": "x_vars[14,Action]",
    "x_2_14": "x_vars[15,Action]",
    "x_3_0": "x_vars[1,Adventure]",
    "x_3_1": "x_vars[2,Adventure]",
    "x_3_2": "x_vars[3,Adventure]",
    "x_3_3": "x_vars[4,Adventure]",
    "x_3_4": "x_vars[5,Adventure]",
    "x_3_5": "x_vars[6,Adventure]",
    "x_3_6": "x_vars[7,Adventure]",
    "x_3_7": "x_vars[8,Adventure]",
    "x_3_8": "x_vars[9,Adventure]",
    "x_3_9": "x_vars[10,Adventure]",
    "x_3_10": "x_vars[11,Adventure]",
    "x_3_11": "x_vars[12,Adventure]",
    "x_3_12": "x_vars[13,Adventure]",
    "x_3_13": "x_vars[14,Adventure]",
    "x_3_14": "x_vars[15,Adventure]",
    "x_4_0": "x_vars[1,RPG]",
    "x_4_1": "x_vars[2,RPG]",
    "x_4_2": "x_vars[3,RPG]",
    "x_4_3": "x_vars[4,RPG]",
    "x_4_4": "x_vars[5,RPG]",
    "x_4_5": "x_vars[6,RPG]",
    "x_4_6": "x_vars[7,RPG]",
    "x_4_7": "x_vars[8,RPG]",
    "x_4_8": "x_vars[9,RPG]",
    "x_4_9": "x_vars[10,RPG]",
    "x_4_10": "x_vars[11,RPG]",
    "x_4_11": "x_vars[12,RPG]",
    "x_4_12": "x_vars[13,RPG]",
    "x_4_13": "x_vars[14,RPG]",
    "x_4_14": "x_vars[15,RPG]",
    "x_5_0": "x_vars[1,Shooter]",
    "x_5_1": "x_vars[2,Shooter]",
    "x_5_2": "x_vars[3,Shooter]",
    "x_5_3": "x_vars[4,Shooter]",
    "x_5_4": "x_vars[5,Shooter]",
    "x_5_5": "x_vars[6,Shooter]",
    "x_5_6": "x_vars[7,Shooter]",
    "x_5_7": "x_vars[8,Shooter]",
    "x_5_8": "x_vars[9,Shooter]",
    "x_5_9": "x_vars[10,Shooter]",
    "x_5_10": "x_vars[11,Shooter]",
    "x_5_11": "x_vars[12,Shooter]",
    "x_5_12": "x_vars[13,Shooter]",
    "x_5_13": "x_vars[14,Shooter]",
    "x_5_14": "x_vars[15,Shooter]",
    "x_6_0": "x_vars[1,Strategy]",
    "x_6_1": "x_vars[2,Strategy]",
    "x_6_2": "x_vars[3,Strategy]",
    "x_6_3": "x_vars[4,Strategy]",
    "x_6_4": "x_vars[5,Strategy]",
    "x_6_5": "x_vars[6,Strategy]",
    "x_6_6": "x_vars[7,Strategy]",
    "x_6_7": "x_vars[8,Strategy]",
    "x_6_8": "x_vars[9,Strategy]",
    "x_6_9": "x_vars[10,Strategy]",
    "x_6_10": "x_vars[11,Strategy]",
    "x_6_11": "x_vars[12,Strategy]",
    "x_6_12": "x_vars[13,Strategy]",
    "x_6_13": "x_vars[14,Strategy]",
    "x_6_14": "x_vars[15,Strategy]",
    "x_7_0": "x_vars[1,Simulation]",
    "x_7_1": "x_vars[2,Simulation]",
    "x_7_2": "x_vars[3,Simulation]",
    "x_7_3": "x_vars[4,Simulation]",
    "x_7_4": "x_vars[5,Simulation]",
    "x_7_5": "x_vars[6,Simulation]",
    "x_7_6": "x_vars[7,Simulation]",
    "x_7_7": "x_vars[8,Simulation]",
    "x_7_8": "x_vars[9,Simulation]",
    "x_7_9": "x_vars[10,Simulation]",
    "x_7_10": "x_vars[11,Simulation]",
    "x_7_11": "x_vars[12,Simulation]",
    "x_7_12": "x_vars[13,Simulation]",
    "x_7_13": "x_vars[14,Simulation]",
    "x_7_14": "x_vars[15,Simulation]",
    "x_8_0": "x_vars[1,Puzzle]",
    "x_8_1": "x_vars[2,Puzzle]",
    "x_8_2": "x_vars[3,Puzzle]",
    "x_8_3": "x_vars[4,Puzzle]",
    "x_8_4": "x_vars[5,Puzzle]",
    "x_8_5": "x_vars[6,Puzzle]",
    "x_8_6": "x_vars[7,Puzzle]",
    "x_8_7": "x_vars[8,Puzzle]",
    "x_8_8": "x_vars[9,Puzzle]",
    "x_8_9": "x_vars[10,Puzzle]",
    "x_8_10": "x_vars[11,Puzzle]",
    "x_8_11": "x_vars[12,Puzzle]",
    "x_8_12": "x_vars[13,Puzzle]",
    "x_8_13": "x_vars[14,Puzzle]",
    "x_8_14": "x_vars[15,Puzzle]",
    "x_9_0": "x_vars[1,Fighting]",
    "x_9_1": "x_vars[2,Fighting]",
    "x_9_2": "x_vars[3,Fighting]",
    "x_9_3": "x_vars[4,Fighting]",
    "x_9_4": "x_vars[5,Fighting]",
    "x_9_5": "x_vars[6,Fighting]",
    "x_9_6": "x_vars[7,Fighting]",
    "x_9_7": "x_vars[8,Fighting]",
    "x_9_8": "x_vars[9,Fighting]",
    "x_9_9": "x_vars[10,Fighting]",
    "x_9_10": "x_vars[11,Fighting]",
    "x_9_11": "x_vars[12,Fighting]",
    "x_9_12": "x_vars[13,Fighting]",
    "x_9_13": "x_vars[14,Fighting]",
    "x_9_14": "x_vars[15,Fighting]",
    "x_10_0": "x_vars[1,Platformer]",
    "x_10_1": "x_vars[2,Platformer]",
    "x_10_2": "x_vars[3,Platformer]",
    "x_10_3": "x_vars[4,Platformer]",
    "x_10_4": "x_vars[5,Platformer]",
    "x_10_5": "x_vars[6,Platformer]",
    "x_10_6": "x_vars[7,Platformer]",
    "x_10_7": "x_vars[8,Platformer]",
    "x_10_8": "x_vars[9,Platformer]",
    "x_10_9": "x_vars[10,Platformer]",
    "x_10_10": "x_vars[11,Platformer]",
    "x_10_11": "x_vars[12,Platformer]",
    "x_10_12": "x_vars[13,Platformer]",
    "x_10_13": "x_vars[14,Platformer]",
    "x_10_14": "x_vars[15,Platformer]",
    "x_11_0": "x_vars[1,Survival]",
    "x_11_1": "x_vars[2,Survival]",
    "x_11_2": "x_vars[3,Survival]",
    "x_11_3": "x_vars[4,Survival]",
    "x_11_4": "x_vars[5,Survival]",
    "x_11_5": "x_vars[6,Survival]",
    "x_11_6": "x_vars[7,Survival]",
    "x_11_7": "x_vars[8,Survival]",
    "x_11_8": "x_vars[9,Survival]",
    "x_11_9": "x_vars[10,Survival]",
    "x_11_10": "x_vars[11,Survival]",
    "x_11_11": "x_vars[12,Survival]",
    "x_11_12": "x_vars[13,Survival]",
    "x_11_13": "x_vars[14,Survival]",
    "x_11_14": "x_vars[15,Survival]",
    "x_12_0": "x_vars[1,Horror]",
    "x_12_1": "x_vars[2,Horror]",
    "x_12_2": "x_vars[3,Horror]",
    "x_12_3": "x_vars[4,Horror]",
    "x_12_4": "x_vars[5,Horror]",
    "x_12_5": "x_vars[6,Horror]",
    "x_12_6": "x_vars[7,Horror]",
    "x_12_7": "x_vars[8,Horror]",
    "x_12_8": "x_vars[9,Horror]",
    "x_12_9": "x_vars[10,Horror]",
    "x_12_10": "x_vars[11,Horror]",
    "x_12_11": "x_vars[12,Horror]",
    "x_12_12": "x_vars[13,Horror]",
    "x_12_13": "x_vars[14,Horror]",
    "x_12_14": "x_vars[15,Horror]",
    "x_13_0": "x_vars[1,Sandbox]",
    "x_13_1": "x_vars[2,Sandbox]",
    "x_13_2": "x_vars[3,Sandbox]",
    "x_13_3": "x_vars[4,Sandbox]",
    "x_13_4": "x_vars[5,Sandbox]",
    "x_13_5": "x_vars[6,Sandbox]",
    "x_13_6": "x_vars[7,Sandbox]",
    "x_13_7": "x_vars[8,Sandbox]",
    "x_13_8": "x_vars[9,Sandbox]",
    "x_13_9": "x_vars[10,Sandbox]",
    "x_13_10": "x_vars[11,Sandbox]",
    "x_13_11": "x_vars[12,Sandbox]",
    "x_13_12": "x_vars[13,Sandbox]",
    "x_13_13": "x_vars[14,Sandbox]",
    "x_13_14": "x_vars[15,Sandbox]",
    "x_14_0": "x_vars[1,MMO]",
    "x_14_1": "x_vars[2,MMO]",
    "x_14_2": "x_vars[3,MMO]",
    "x_14_3": "x_vars[4,MMO]",
    "x_14_4": "x_vars[5,MMO]",
    "x_14_5": "x_vars[6,MMO]",
    "x_14_6": "x_vars[7,MMO]",
    "x_14_7": "x_vars[8,MMO]",
    "x_14_8": "x_vars[9,MMO]",
    "x_14_9": "x_vars[10,MMO]",
    "x_14_10": "x_vars[11,MMO]",
    "x_14_11": "x_vars[12,MMO]",
    "x_14_12": "x_vars[13,MMO]",
    "x_14_13": "x_vars[14,MMO]",
    "x_14_14": "x_vars[15,MMO]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-058": {
   "source_code_sha256": "d924abab236ff14cc02f3029bb0b85e19b4387ccdfee0694e3ac876f0415cad5",
   "original_solution_sha256": "d9571a7b201f3122db5bb762c3bc642e8babe053e853c962fddf1209556316d3",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S2,C1]"
    ],
    [
     "C7",
     "x_vars[S2,C2]"
    ],
    [
     "C8",
     "x_vars[S2,C3]"
    ],
    [
     "C9",
     "x_vars[S2,C4]"
    ],
    [
     "C10",
     "x_vars[S2,C5]"
    ],
    [
     "C11",
     "x_vars[S2,C6]"
    ],
    [
     "C12",
     "x_vars[S3,C1]"
    ],
    [
     "C13",
     "x_vars[S3,C2]"
    ],
    [
     "C14",
     "x_vars[S3,C3]"
    ],
    [
     "C15",
     "x_vars[S3,C4]"
    ],
    [
     "C16",
     "x_vars[S3,C5]"
    ],
    [
     "C17",
     "x_vars[S3,C6]"
    ],
    [
     "C18",
     "x_vars[S4,C1]"
    ],
    [
     "C19",
     "x_vars[S4,C2]"
    ],
    [
     "C20",
     "x_vars[S4,C3]"
    ],
    [
     "C21",
     "x_vars[S4,C4]"
    ],
    [
     "C22",
     "x_vars[S4,C5]"
    ],
    [
     "C23",
     "x_vars[S4,C6]"
    ],
    [
     "C24",
     "x_vars[S5,C1]"
    ],
    [
     "C25",
     "x_vars[S5,C2]"
    ],
    [
     "C26",
     "x_vars[S5,C3]"
    ],
    [
     "C27",
     "x_vars[S5,C4]"
    ],
    [
     "C28",
     "x_vars[S5,C5]"
    ],
    [
     "C29",
     "x_vars[S5,C6]"
    ],
    [
     "C30",
     "x_vars[S6,C1]"
    ],
    [
     "C31",
     "x_vars[S6,C2]"
    ],
    [
     "C32",
     "x_vars[S6,C3]"
    ],
    [
     "C33",
     "x_vars[S6,C4]"
    ],
    [
     "C34",
     "x_vars[S6,C5]"
    ],
    [
     "C35",
     "x_vars[S6,C6]"
    ],
    [
     "C36",
     "y_vars[S1]"
    ],
    [
     "C37",
     "y_vars[S2]"
    ],
    [
     "C38",
     "y_vars[S3]"
    ],
    [
     "C39",
     "y_vars[S4]"
    ],
    [
     "C40",
     "y_vars[S5]"
    ],
    [
     "C41",
     "y_vars[S6]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]",
    "y_2": "y_vars[S3]",
    "y_3": "y_vars[S4]",
    "y_4": "y_vars[S5]",
    "y_5": "y_vars[S6]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-059": {
   "source_code_sha256": "6d751ebc41dac7059b5570f781f9eb05c9fd42efc9cd9a43a4c6c75c76a4ccb8",
   "original_solution_sha256": "d73d6e47c6829a10dc707a0371cc253309b1fddf7a98aaa46d80a80ca88ea938",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S2,C1]"
    ],
    [
     "C10",
     "x_vars[S2,C2]"
    ],
    [
     "C11",
     "x_vars[S2,C3]"
    ],
    [
     "C12",
     "x_vars[S2,C4]"
    ],
    [
     "C13",
     "x_vars[S2,C5]"
    ],
    [
     "C14",
     "x_vars[S2,C6]"
    ],
    [
     "C15",
     "x_vars[S2,C7]"
    ],
    [
     "C16",
     "x_vars[S2,C8]"
    ],
    [
     "C17",
     "x_vars[S2,C9]"
    ],
    [
     "C18",
     "x_vars[S3,C1]"
    ],
    [
     "C19",
     "x_vars[S3,C2]"
    ],
    [
     "C20",
     "x_vars[S3,C3]"
    ],
    [
     "C21",
     "x_vars[S3,C4]"
    ],
    [
     "C22",
     "x_vars[S3,C5]"
    ],
    [
     "C23",
     "x_vars[S3,C6]"
    ],
    [
     "C24",
     "x_vars[S3,C7]"
    ],
    [
     "C25",
     "x_vars[S3,C8]"
    ],
    [
     "C26",
     "x_vars[S3,C9]"
    ],
    [
     "C27",
     "x_vars[S4,C1]"
    ],
    [
     "C28",
     "x_vars[S4,C2]"
    ],
    [
     "C29",
     "x_vars[S4,C3]"
    ],
    [
     "C30",
     "x_vars[S4,C4]"
    ],
    [
     "C31",
     "x_vars[S4,C5]"
    ],
    [
     "C32",
     "x_vars[S4,C6]"
    ],
    [
     "C33",
     "x_vars[S4,C7]"
    ],
    [
     "C34",
     "x_vars[S4,C8]"
    ],
    [
     "C35",
     "x_vars[S4,C9]"
    ],
    [
     "C36",
     "x_vars[S5,C1]"
    ],
    [
     "C37",
     "x_vars[S5,C2]"
    ],
    [
     "C38",
     "x_vars[S5,C3]"
    ],
    [
     "C39",
     "x_vars[S5,C4]"
    ],
    [
     "C40",
     "x_vars[S5,C5]"
    ],
    [
     "C41",
     "x_vars[S5,C6]"
    ],
    [
     "C42",
     "x_vars[S5,C7]"
    ],
    [
     "C43",
     "x_vars[S5,C8]"
    ],
    [
     "C44",
     "x_vars[S5,C9]"
    ],
    [
     "C45",
     "x_vars[S6,C1]"
    ],
    [
     "C46",
     "x_vars[S6,C2]"
    ],
    [
     "C47",
     "x_vars[S6,C3]"
    ],
    [
     "C48",
     "x_vars[S6,C4]"
    ],
    [
     "C49",
     "x_vars[S6,C5]"
    ],
    [
     "C50",
     "x_vars[S6,C6]"
    ],
    [
     "C51",
     "x_vars[S6,C7]"
    ],
    [
     "C52",
     "x_vars[S6,C8]"
    ],
    [
     "C53",
     "x_vars[S6,C9]"
    ],
    [
     "C54",
     "x_vars[S7,C1]"
    ],
    [
     "C55",
     "x_vars[S7,C2]"
    ],
    [
     "C56",
     "x_vars[S7,C3]"
    ],
    [
     "C57",
     "x_vars[S7,C4]"
    ],
    [
     "C58",
     "x_vars[S7,C5]"
    ],
    [
     "C59",
     "x_vars[S7,C6]"
    ],
    [
     "C60",
     "x_vars[S7,C7]"
    ],
    [
     "C61",
     "x_vars[S7,C8]"
    ],
    [
     "C62",
     "x_vars[S7,C9]"
    ],
    [
     "C63",
     "x_vars[S8,C1]"
    ],
    [
     "C64",
     "x_vars[S8,C2]"
    ],
    [
     "C65",
     "x_vars[S8,C3]"
    ],
    [
     "C66",
     "x_vars[S8,C4]"
    ],
    [
     "C67",
     "x_vars[S8,C5]"
    ],
    [
     "C68",
     "x_vars[S8,C6]"
    ],
    [
     "C69",
     "x_vars[S8,C7]"
    ],
    [
     "C70",
     "x_vars[S8,C8]"
    ],
    [
     "C71",
     "x_vars[S8,C9]"
    ],
    [
     "C72",
     "y_vars[S1]"
    ],
    [
     "C73",
     "y_vars[S2]"
    ],
    [
     "C74",
     "y_vars[S3]"
    ],
    [
     "C75",
     "y_vars[S4]"
    ],
    [
     "C76",
     "y_vars[S5]"
    ],
    [
     "C77",
     "y_vars[S6]"
    ],
    [
     "C78",
     "y_vars[S7]"
    ],
    [
     "C79",
     "y_vars[S8]"
    ]
   ],
   "primary_mapping": {
    "y[S1]": "y_vars[S1]",
    "y[S2]": "y_vars[S2]",
    "y[S3]": "y_vars[S3]",
    "y[S4]": "y_vars[S4]",
    "y[S5]": "y_vars[S5]",
    "y[S6]": "y_vars[S6]",
    "y[S7]": "y_vars[S7]",
    "y[S8]": "y_vars[S8]",
    "x[S1,C1]": "x_vars[S1,C1]",
    "x[S1,C9]": "x_vars[S1,C9]",
    "x[S2,C2]": "x_vars[S2,C2]",
    "x[S2,C9]": "x_vars[S2,C9]",
    "x[S3,C3]": "x_vars[S3,C3]",
    "x[S3,C9]": "x_vars[S3,C9]",
    "x[S4,C4]": "x_vars[S4,C4]",
    "x[S4,C9]": "x_vars[S4,C9]",
    "x[S5,C5]": "x_vars[S5,C5]",
    "x[S5,C9]": "x_vars[S5,C9]",
    "x[S6,C6]": "x_vars[S6,C6]",
    "x[S6,C9]": "x_vars[S6,C9]",
    "x[S7,C7]": "x_vars[S7,C7]",
    "x[S7,C9]": "x_vars[S7,C9]",
    "x[S8,C8]": "x_vars[S8,C8]",
    "x[S8,C9]": "x_vars[S8,C9]",
    "x[S1,C2]": "x_vars[S1,C2]",
    "x[S1,C3]": "x_vars[S1,C3]",
    "x[S1,C4]": "x_vars[S1,C4]",
    "x[S1,C5]": "x_vars[S1,C5]",
    "x[S1,C6]": "x_vars[S1,C6]",
    "x[S1,C7]": "x_vars[S1,C7]",
    "x[S1,C8]": "x_vars[S1,C8]",
    "x[S2,C1]": "x_vars[S2,C1]",
    "x[S2,C3]": "x_vars[S2,C3]",
    "x[S2,C4]": "x_vars[S2,C4]",
    "x[S2,C5]": "x_vars[S2,C5]",
    "x[S2,C6]": "x_vars[S2,C6]",
    "x[S2,C7]": "x_vars[S2,C7]",
    "x[S2,C8]": "x_vars[S2,C8]",
    "x[S3,C1]": "x_vars[S3,C1]",
    "x[S3,C2]": "x_vars[S3,C2]",
    "x[S3,C4]": "x_vars[S3,C4]",
    "x[S3,C5]": "x_vars[S3,C5]",
    "x[S3,C6]": "x_vars[S3,C6]",
    "x[S3,C7]": "x_vars[S3,C7]",
    "x[S3,C8]": "x_vars[S3,C8]",
    "x[S4,C1]": "x_vars[S4,C1]",
    "x[S4,C2]": "x_vars[S4,C2]",
    "x[S4,C3]": "x_vars[S4,C3]",
    "x[S4,C5]": "x_vars[S4,C5]",
    "x[S4,C6]": "x_vars[S4,C6]",
    "x[S4,C7]": "x_vars[S4,C7]",
    "x[S4,C8]": "x_vars[S4,C8]",
    "x[S5,C1]": "x_vars[S5,C1]",
    "x[S5,C2]": "x_vars[S5,C2]",
    "x[S5,C3]": "x_vars[S5,C3]",
    "x[S5,C4]": "x_vars[S5,C4]",
    "x[S5,C6]": "x_vars[S5,C6]",
    "x[S5,C7]": "x_vars[S5,C7]",
    "x[S5,C8]": "x_vars[S5,C8]",
    "x[S6,C1]": "x_vars[S6,C1]",
    "x[S6,C2]": "x_vars[S6,C2]",
    "x[S6,C3]": "x_vars[S6,C3]",
    "x[S6,C4]": "x_vars[S6,C4]",
    "x[S6,C5]": "x_vars[S6,C5]",
    "x[S6,C7]": "x_vars[S6,C7]",
    "x[S6,C8]": "x_vars[S6,C8]",
    "x[S7,C1]": "x_vars[S7,C1]",
    "x[S7,C2]": "x_vars[S7,C2]",
    "x[S7,C3]": "x_vars[S7,C3]",
    "x[S7,C4]": "x_vars[S7,C4]",
    "x[S7,C5]": "x_vars[S7,C5]",
    "x[S7,C6]": "x_vars[S7,C6]",
    "x[S7,C8]": "x_vars[S7,C8]",
    "x[S8,C1]": "x_vars[S8,C1]",
    "x[S8,C2]": "x_vars[S8,C2]",
    "x[S8,C3]": "x_vars[S8,C3]",
    "x[S8,C4]": "x_vars[S8,C4]",
    "x[S8,C5]": "x_vars[S8,C5]",
    "x[S8,C6]": "x_vars[S8,C6]",
    "x[S8,C7]": "x_vars[S8,C7]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-060": {
   "source_code_sha256": "212baa67ec97c85376da297b26dc576ba2426d0ae6f95f0addf4f6bd76eaa6d7",
   "original_solution_sha256": "334a6abd03ef3e20cb28ac19f741564c4af2d12d989ed5a2585613be90a35b29",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S1,C11]"
    ],
    [
     "C11",
     "x_vars[S1,C12]"
    ],
    [
     "C12",
     "x_vars[S2,C1]"
    ],
    [
     "C13",
     "x_vars[S2,C2]"
    ],
    [
     "C14",
     "x_vars[S2,C3]"
    ],
    [
     "C15",
     "x_vars[S2,C4]"
    ],
    [
     "C16",
     "x_vars[S2,C5]"
    ],
    [
     "C17",
     "x_vars[S2,C6]"
    ],
    [
     "C18",
     "x_vars[S2,C7]"
    ],
    [
     "C19",
     "x_vars[S2,C8]"
    ],
    [
     "C20",
     "x_vars[S2,C9]"
    ],
    [
     "C21",
     "x_vars[S2,C10]"
    ],
    [
     "C22",
     "x_vars[S2,C11]"
    ],
    [
     "C23",
     "x_vars[S2,C12]"
    ],
    [
     "C24",
     "x_vars[S3,C1]"
    ],
    [
     "C25",
     "x_vars[S3,C2]"
    ],
    [
     "C26",
     "x_vars[S3,C3]"
    ],
    [
     "C27",
     "x_vars[S3,C4]"
    ],
    [
     "C28",
     "x_vars[S3,C5]"
    ],
    [
     "C29",
     "x_vars[S3,C6]"
    ],
    [
     "C30",
     "x_vars[S3,C7]"
    ],
    [
     "C31",
     "x_vars[S3,C8]"
    ],
    [
     "C32",
     "x_vars[S3,C9]"
    ],
    [
     "C33",
     "x_vars[S3,C10]"
    ],
    [
     "C34",
     "x_vars[S3,C11]"
    ],
    [
     "C35",
     "x_vars[S3,C12]"
    ],
    [
     "C36",
     "x_vars[S4,C1]"
    ],
    [
     "C37",
     "x_vars[S4,C2]"
    ],
    [
     "C38",
     "x_vars[S4,C3]"
    ],
    [
     "C39",
     "x_vars[S4,C4]"
    ],
    [
     "C40",
     "x_vars[S4,C5]"
    ],
    [
     "C41",
     "x_vars[S4,C6]"
    ],
    [
     "C42",
     "x_vars[S4,C7]"
    ],
    [
     "C43",
     "x_vars[S4,C8]"
    ],
    [
     "C44",
     "x_vars[S4,C9]"
    ],
    [
     "C45",
     "x_vars[S4,C10]"
    ],
    [
     "C46",
     "x_vars[S4,C11]"
    ],
    [
     "C47",
     "x_vars[S4,C12]"
    ],
    [
     "C48",
     "x_vars[S5,C1]"
    ],
    [
     "C49",
     "x_vars[S5,C2]"
    ],
    [
     "C50",
     "x_vars[S5,C3]"
    ],
    [
     "C51",
     "x_vars[S5,C4]"
    ],
    [
     "C52",
     "x_vars[S5,C5]"
    ],
    [
     "C53",
     "x_vars[S5,C6]"
    ],
    [
     "C54",
     "x_vars[S5,C7]"
    ],
    [
     "C55",
     "x_vars[S5,C8]"
    ],
    [
     "C56",
     "x_vars[S5,C9]"
    ],
    [
     "C57",
     "x_vars[S5,C10]"
    ],
    [
     "C58",
     "x_vars[S5,C11]"
    ],
    [
     "C59",
     "x_vars[S5,C12]"
    ],
    [
     "C60",
     "x_vars[S6,C1]"
    ],
    [
     "C61",
     "x_vars[S6,C2]"
    ],
    [
     "C62",
     "x_vars[S6,C3]"
    ],
    [
     "C63",
     "x_vars[S6,C4]"
    ],
    [
     "C64",
     "x_vars[S6,C5]"
    ],
    [
     "C65",
     "x_vars[S6,C6]"
    ],
    [
     "C66",
     "x_vars[S6,C7]"
    ],
    [
     "C67",
     "x_vars[S6,C8]"
    ],
    [
     "C68",
     "x_vars[S6,C9]"
    ],
    [
     "C69",
     "x_vars[S6,C10]"
    ],
    [
     "C70",
     "x_vars[S6,C11]"
    ],
    [
     "C71",
     "x_vars[S6,C12]"
    ],
    [
     "C72",
     "x_vars[S7,C1]"
    ],
    [
     "C73",
     "x_vars[S7,C2]"
    ],
    [
     "C74",
     "x_vars[S7,C3]"
    ],
    [
     "C75",
     "x_vars[S7,C4]"
    ],
    [
     "C76",
     "x_vars[S7,C5]"
    ],
    [
     "C77",
     "x_vars[S7,C6]"
    ],
    [
     "C78",
     "x_vars[S7,C7]"
    ],
    [
     "C79",
     "x_vars[S7,C8]"
    ],
    [
     "C80",
     "x_vars[S7,C9]"
    ],
    [
     "C81",
     "x_vars[S7,C10]"
    ],
    [
     "C82",
     "x_vars[S7,C11]"
    ],
    [
     "C83",
     "x_vars[S7,C12]"
    ],
    [
     "C84",
     "x_vars[S8,C1]"
    ],
    [
     "C85",
     "x_vars[S8,C2]"
    ],
    [
     "C86",
     "x_vars[S8,C3]"
    ],
    [
     "C87",
     "x_vars[S8,C4]"
    ],
    [
     "C88",
     "x_vars[S8,C5]"
    ],
    [
     "C89",
     "x_vars[S8,C6]"
    ],
    [
     "C90",
     "x_vars[S8,C7]"
    ],
    [
     "C91",
     "x_vars[S8,C8]"
    ],
    [
     "C92",
     "x_vars[S8,C9]"
    ],
    [
     "C93",
     "x_vars[S8,C10]"
    ],
    [
     "C94",
     "x_vars[S8,C11]"
    ],
    [
     "C95",
     "x_vars[S8,C12]"
    ],
    [
     "C96",
     "x_vars[S9,C1]"
    ],
    [
     "C97",
     "x_vars[S9,C2]"
    ],
    [
     "C98",
     "x_vars[S9,C3]"
    ],
    [
     "C99",
     "x_vars[S9,C4]"
    ],
    [
     "C100",
     "x_vars[S9,C5]"
    ],
    [
     "C101",
     "x_vars[S9,C6]"
    ],
    [
     "C102",
     "x_vars[S9,C7]"
    ],
    [
     "C103",
     "x_vars[S9,C8]"
    ],
    [
     "C104",
     "x_vars[S9,C9]"
    ],
    [
     "C105",
     "x_vars[S9,C10]"
    ],
    [
     "C106",
     "x_vars[S9,C11]"
    ],
    [
     "C107",
     "x_vars[S9,C12]"
    ],
    [
     "C108",
     "x_vars[S10,C1]"
    ],
    [
     "C109",
     "x_vars[S10,C2]"
    ],
    [
     "C110",
     "x_vars[S10,C3]"
    ],
    [
     "C111",
     "x_vars[S10,C4]"
    ],
    [
     "C112",
     "x_vars[S10,C5]"
    ],
    [
     "C113",
     "x_vars[S10,C6]"
    ],
    [
     "C114",
     "x_vars[S10,C7]"
    ],
    [
     "C115",
     "x_vars[S10,C8]"
    ],
    [
     "C116",
     "x_vars[S10,C9]"
    ],
    [
     "C117",
     "x_vars[S10,C10]"
    ],
    [
     "C118",
     "x_vars[S10,C11]"
    ],
    [
     "C119",
     "x_vars[S10,C12]"
    ],
    [
     "C120",
     "x_vars[S11,C1]"
    ],
    [
     "C121",
     "x_vars[S11,C2]"
    ],
    [
     "C122",
     "x_vars[S11,C3]"
    ],
    [
     "C123",
     "x_vars[S11,C4]"
    ],
    [
     "C124",
     "x_vars[S11,C5]"
    ],
    [
     "C125",
     "x_vars[S11,C6]"
    ],
    [
     "C126",
     "x_vars[S11,C7]"
    ],
    [
     "C127",
     "x_vars[S11,C8]"
    ],
    [
     "C128",
     "x_vars[S11,C9]"
    ],
    [
     "C129",
     "x_vars[S11,C10]"
    ],
    [
     "C130",
     "x_vars[S11,C11]"
    ],
    [
     "C131",
     "x_vars[S11,C12]"
    ],
    [
     "C132",
     "x_vars[S12,C1]"
    ],
    [
     "C133",
     "x_vars[S12,C2]"
    ],
    [
     "C134",
     "x_vars[S12,C3]"
    ],
    [
     "C135",
     "x_vars[S12,C4]"
    ],
    [
     "C136",
     "x_vars[S12,C5]"
    ],
    [
     "C137",
     "x_vars[S12,C6]"
    ],
    [
     "C138",
     "x_vars[S12,C7]"
    ],
    [
     "C139",
     "x_vars[S12,C8]"
    ],
    [
     "C140",
     "x_vars[S12,C9]"
    ],
    [
     "C141",
     "x_vars[S12,C10]"
    ],
    [
     "C142",
     "x_vars[S12,C11]"
    ],
    [
     "C143",
     "x_vars[S12,C12]"
    ],
    [
     "C144",
     "y_vars[S1]"
    ],
    [
     "C145",
     "y_vars[S2]"
    ],
    [
     "C146",
     "y_vars[S3]"
    ],
    [
     "C147",
     "y_vars[S4]"
    ],
    [
     "C148",
     "y_vars[S5]"
    ],
    [
     "C149",
     "y_vars[S6]"
    ],
    [
     "C150",
     "y_vars[S7]"
    ],
    [
     "C151",
     "y_vars[S8]"
    ],
    [
     "C152",
     "y_vars[S9]"
    ],
    [
     "C153",
     "y_vars[S10]"
    ],
    [
     "C154",
     "y_vars[S11]"
    ],
    [
     "C155",
     "y_vars[S12]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_0_8": "x_vars[S1,C9]",
    "x_0_9": "x_vars[S1,C10]",
    "x_0_10": "x_vars[S1,C11]",
    "x_0_11": "x_vars[S1,C12]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_1_7": "x_vars[S2,C8]",
    "x_1_8": "x_vars[S2,C9]",
    "x_1_9": "x_vars[S2,C10]",
    "x_1_10": "x_vars[S2,C11]",
    "x_1_11": "x_vars[S2,C12]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_2_10": "x_vars[S3,C11]",
    "x_2_11": "x_vars[S3,C12]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_3_10": "x_vars[S4,C11]",
    "x_3_11": "x_vars[S4,C12]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_4_10": "x_vars[S5,C11]",
    "x_4_11": "x_vars[S5,C12]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_5_10": "x_vars[S6,C11]",
    "x_5_11": "x_vars[S6,C12]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "x_6_7": "x_vars[S7,C8]",
    "x_6_9": "x_vars[S7,C10]",
    "x_6_10": "x_vars[S7,C11]",
    "x_6_11": "x_vars[S7,C12]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_7_10": "x_vars[S8,C11]",
    "x_7_11": "x_vars[S8,C12]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_1": "x_vars[S9,C2]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_3": "x_vars[S9,C4]",
    "x_8_4": "x_vars[S9,C5]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_6": "x_vars[S9,C7]",
    "x_8_7": "x_vars[S9,C8]",
    "x_8_8": "x_vars[S9,C9]",
    "x_8_9": "x_vars[S9,C10]",
    "x_8_10": "x_vars[S9,C11]",
    "x_8_11": "x_vars[S9,C12]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_1": "x_vars[S10,C2]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_3": "x_vars[S10,C4]",
    "x_9_4": "x_vars[S10,C5]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_6": "x_vars[S10,C7]",
    "x_9_7": "x_vars[S10,C8]",
    "x_9_8": "x_vars[S10,C9]",
    "x_9_9": "x_vars[S10,C10]",
    "x_9_10": "x_vars[S10,C11]",
    "x_9_11": "x_vars[S10,C12]",
    "x_10_0": "x_vars[S11,C1]",
    "x_10_2": "x_vars[S11,C3]",
    "x_10_3": "x_vars[S11,C4]",
    "x_10_4": "x_vars[S11,C5]",
    "x_10_5": "x_vars[S11,C6]",
    "x_10_6": "x_vars[S11,C7]",
    "x_10_7": "x_vars[S11,C8]",
    "x_10_8": "x_vars[S11,C9]",
    "x_10_9": "x_vars[S11,C10]",
    "x_10_10": "x_vars[S11,C11]",
    "x_10_11": "x_vars[S11,C12]",
    "x_11_0": "x_vars[S12,C1]",
    "x_11_1": "x_vars[S12,C2]",
    "x_11_2": "x_vars[S12,C3]",
    "x_11_3": "x_vars[S12,C4]",
    "x_11_4": "x_vars[S12,C5]",
    "x_11_5": "x_vars[S12,C6]",
    "x_11_6": "x_vars[S12,C7]",
    "x_11_7": "x_vars[S12,C8]",
    "x_11_8": "x_vars[S12,C9]",
    "x_11_9": "x_vars[S12,C10]",
    "x_11_10": "x_vars[S12,C11]",
    "x_11_11": "x_vars[S12,C12]",
    "x_10_1": "x_vars[S11,C2]",
    "x_6_8": "x_vars[S7,C9]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]",
    "y_2": "y_vars[S3]",
    "y_3": "y_vars[S4]",
    "y_4": "y_vars[S5]",
    "y_5": "y_vars[S6]",
    "y_6": "y_vars[S7]",
    "y_7": "y_vars[S8]",
    "y_8": "y_vars[S9]",
    "y_9": "y_vars[S10]",
    "y_10": "y_vars[S11]",
    "y_11": "y_vars[S12]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-061": {
   "source_code_sha256": "f070b38c2715b953fc8327d15d08d484560f0ca7f9791d77ff0aabcbeb49a02c",
   "original_solution_sha256": "a6f83c019b93e1892fcbd4e2e29ccaf3cb82fdca7020bcdb0c7205ab0df04f6b",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S2,C1]"
    ],
    [
     "C6",
     "x_vars[S2,C2]"
    ],
    [
     "C7",
     "x_vars[S2,C3]"
    ],
    [
     "C8",
     "x_vars[S2,C4]"
    ],
    [
     "C9",
     "x_vars[S2,C5]"
    ],
    [
     "C10",
     "x_vars[S3,C1]"
    ],
    [
     "C11",
     "x_vars[S3,C2]"
    ],
    [
     "C12",
     "x_vars[S3,C3]"
    ],
    [
     "C13",
     "x_vars[S3,C4]"
    ],
    [
     "C14",
     "x_vars[S3,C5]"
    ],
    [
     "C15",
     "x_vars[S4,C1]"
    ],
    [
     "C16",
     "x_vars[S4,C2]"
    ],
    [
     "C17",
     "x_vars[S4,C3]"
    ],
    [
     "C18",
     "x_vars[S4,C4]"
    ],
    [
     "C19",
     "x_vars[S4,C5]"
    ],
    [
     "C20",
     "x_vars[S5,C1]"
    ],
    [
     "C21",
     "x_vars[S5,C2]"
    ],
    [
     "C22",
     "x_vars[S5,C3]"
    ],
    [
     "C23",
     "x_vars[S5,C4]"
    ],
    [
     "C24",
     "x_vars[S5,C5]"
    ],
    [
     "C25",
     "y_vars[S1]"
    ],
    [
     "C26",
     "y_vars[S2]"
    ],
    [
     "C27",
     "y_vars[S3]"
    ],
    [
     "C28",
     "y_vars[S4]"
    ],
    [
     "C29",
     "y_vars[S5]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]",
    "y_2": "y_vars[S3]",
    "y_3": "y_vars[S4]",
    "y_4": "y_vars[S5]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-062": {
   "source_code_sha256": "e10da2df4bcee4e4daf57b4be3b96fffa3280fa3e5387126b80e7e6d29ace5bf",
   "original_solution_sha256": "9fc0b60fb12e3e839b4b5f0d76c98060f636b4173bf5294b76ff2915ee0c3f34",
   "bindings": [
    [
     "C0",
     "x_vars[MOUNT AYR,CLARINDA]"
    ],
    [
     "C1",
     "x_vars[MOUNT AYR,FORT MADISON]"
    ],
    [
     "C2",
     "x_vars[MOUNT AYR,SIOUX CITY]"
    ],
    [
     "C3",
     "x_vars[MOUNT AYR,TOLEDO]"
    ],
    [
     "C4",
     "x_vars[MOUNT AYR,BANCROFT]"
    ],
    [
     "C5",
     "x_vars[WAUKEE,CLARINDA]"
    ],
    [
     "C6",
     "x_vars[WAUKEE,FORT MADISON]"
    ],
    [
     "C7",
     "x_vars[WAUKEE,SIOUX CITY]"
    ],
    [
     "C8",
     "x_vars[WAUKEE,TOLEDO]"
    ],
    [
     "C9",
     "x_vars[WAUKEE,BANCROFT]"
    ],
    [
     "C10",
     "x_vars[WAVERLY,CLARINDA]"
    ],
    [
     "C11",
     "x_vars[WAVERLY,FORT MADISON]"
    ],
    [
     "C12",
     "x_vars[WAVERLY,SIOUX CITY]"
    ],
    [
     "C13",
     "x_vars[WAVERLY,TOLEDO]"
    ],
    [
     "C14",
     "x_vars[WAVERLY,BANCROFT]"
    ],
    [
     "C15",
     "x_vars[PELLA,CLARINDA]"
    ],
    [
     "C16",
     "x_vars[PELLA,FORT MADISON]"
    ],
    [
     "C17",
     "x_vars[PELLA,SIOUX CITY]"
    ],
    [
     "C18",
     "x_vars[PELLA,TOLEDO]"
    ],
    [
     "C19",
     "x_vars[PELLA,BANCROFT]"
    ],
    [
     "C20",
     "x_vars[DES MOINES,CLARINDA]"
    ],
    [
     "C21",
     "x_vars[DES MOINES,FORT MADISON]"
    ],
    [
     "C22",
     "x_vars[DES MOINES,SIOUX CITY]"
    ],
    [
     "C23",
     "x_vars[DES MOINES,TOLEDO]"
    ],
    [
     "C24",
     "x_vars[DES MOINES,BANCROFT]"
    ],
    [
     "C25",
     "y_vars[MOUNT AYR]"
    ],
    [
     "C26",
     "y_vars[WAUKEE]"
    ],
    [
     "C27",
     "y_vars[WAVERLY]"
    ],
    [
     "C28",
     "y_vars[PELLA]"
    ],
    [
     "C29",
     "y_vars[DES MOINES]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[MOUNT AYR,CLARINDA]",
    "x_0_1": "x_vars[MOUNT AYR,FORT MADISON]",
    "x_0_2": "x_vars[MOUNT AYR,SIOUX CITY]",
    "x_0_3": "x_vars[MOUNT AYR,TOLEDO]",
    "x_0_4": "x_vars[MOUNT AYR,BANCROFT]",
    "x_1_0": "x_vars[WAUKEE,CLARINDA]",
    "x_1_1": "x_vars[WAUKEE,FORT MADISON]",
    "x_1_2": "x_vars[WAUKEE,SIOUX CITY]",
    "x_1_3": "x_vars[WAUKEE,TOLEDO]",
    "x_1_4": "x_vars[WAUKEE,BANCROFT]",
    "x_2_0": "x_vars[WAVERLY,CLARINDA]",
    "x_2_1": "x_vars[WAVERLY,FORT MADISON]",
    "x_2_2": "x_vars[WAVERLY,SIOUX CITY]",
    "x_2_3": "x_vars[WAVERLY,TOLEDO]",
    "x_2_4": "x_vars[WAVERLY,BANCROFT]",
    "x_3_0": "x_vars[PELLA,CLARINDA]",
    "x_3_1": "x_vars[PELLA,FORT MADISON]",
    "x_3_2": "x_vars[PELLA,SIOUX CITY]",
    "x_3_3": "x_vars[PELLA,TOLEDO]",
    "x_3_4": "x_vars[PELLA,BANCROFT]",
    "x_4_0": "x_vars[DES MOINES,CLARINDA]",
    "x_4_1": "x_vars[DES MOINES,FORT MADISON]",
    "x_4_2": "x_vars[DES MOINES,SIOUX CITY]",
    "x_4_3": "x_vars[DES MOINES,TOLEDO]",
    "x_4_4": "x_vars[DES MOINES,BANCROFT]",
    "y_0": "y_vars[MOUNT AYR]",
    "y_1": "y_vars[WAUKEE]",
    "y_2": "y_vars[WAVERLY]",
    "y_3": "y_vars[PELLA]",
    "y_4": "y_vars[DES MOINES]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-063": {
   "source_code_sha256": "392a8b1180565ca8931497efff6bf9504d61c0f57758a506fa9bce2ff8cc37c3",
   "original_solution_sha256": "a39be2de0325f735fde048339e8c8e345a246d4f85babdb82675cb794178b88b",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S2,C1]"
    ],
    [
     "C8",
     "x_vars[S2,C2]"
    ],
    [
     "C9",
     "x_vars[S2,C3]"
    ],
    [
     "C10",
     "x_vars[S2,C4]"
    ],
    [
     "C11",
     "x_vars[S2,C5]"
    ],
    [
     "C12",
     "x_vars[S2,C6]"
    ],
    [
     "C13",
     "x_vars[S2,C7]"
    ],
    [
     "C14",
     "x_vars[S3,C1]"
    ],
    [
     "C15",
     "x_vars[S3,C2]"
    ],
    [
     "C16",
     "x_vars[S3,C3]"
    ],
    [
     "C17",
     "x_vars[S3,C4]"
    ],
    [
     "C18",
     "x_vars[S3,C5]"
    ],
    [
     "C19",
     "x_vars[S3,C6]"
    ],
    [
     "C20",
     "x_vars[S3,C7]"
    ],
    [
     "C21",
     "x_vars[S4,C1]"
    ],
    [
     "C22",
     "x_vars[S4,C2]"
    ],
    [
     "C23",
     "x_vars[S4,C3]"
    ],
    [
     "C24",
     "x_vars[S4,C4]"
    ],
    [
     "C25",
     "x_vars[S4,C5]"
    ],
    [
     "C26",
     "x_vars[S4,C6]"
    ],
    [
     "C27",
     "x_vars[S4,C7]"
    ],
    [
     "C28",
     "x_vars[S5,C1]"
    ],
    [
     "C29",
     "x_vars[S5,C2]"
    ],
    [
     "C30",
     "x_vars[S5,C3]"
    ],
    [
     "C31",
     "x_vars[S5,C4]"
    ],
    [
     "C32",
     "x_vars[S5,C5]"
    ],
    [
     "C33",
     "x_vars[S5,C6]"
    ],
    [
     "C34",
     "x_vars[S5,C7]"
    ],
    [
     "C35",
     "x_vars[S6,C1]"
    ],
    [
     "C36",
     "x_vars[S6,C2]"
    ],
    [
     "C37",
     "x_vars[S6,C3]"
    ],
    [
     "C38",
     "x_vars[S6,C4]"
    ],
    [
     "C39",
     "x_vars[S6,C5]"
    ],
    [
     "C40",
     "x_vars[S6,C6]"
    ],
    [
     "C41",
     "x_vars[S6,C7]"
    ],
    [
     "C42",
     "x_vars[S7,C1]"
    ],
    [
     "C43",
     "x_vars[S7,C2]"
    ],
    [
     "C44",
     "x_vars[S7,C3]"
    ],
    [
     "C45",
     "x_vars[S7,C4]"
    ],
    [
     "C46",
     "x_vars[S7,C5]"
    ],
    [
     "C47",
     "x_vars[S7,C6]"
    ],
    [
     "C48",
     "x_vars[S7,C7]"
    ],
    [
     "C49",
     "y_vars[S1]"
    ],
    [
     "C50",
     "y_vars[S2]"
    ],
    [
     "C51",
     "y_vars[S3]"
    ],
    [
     "C52",
     "y_vars[S4]"
    ],
    [
     "C53",
     "y_vars[S5]"
    ],
    [
     "C54",
     "y_vars[S6]"
    ],
    [
     "C55",
     "y_vars[S7]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]",
    "y_2": "y_vars[S3]",
    "y_3": "y_vars[S4]",
    "y_4": "y_vars[S5]",
    "y_5": "y_vars[S6]",
    "y_6": "y_vars[S7]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-064": {
   "source_code_sha256": "cf0983725953965d33dcc0abc336691e3cdf817e061ae244bb08c9506dcfc8de",
   "original_solution_sha256": "7f587853ca7491fac2946e4b22fb28f74fd7f469402e67365bc2ab96207a63ec",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S1,C4]"
    ],
    [
     "C4",
     "x_vars[S1,C5]"
    ],
    [
     "C5",
     "x_vars[S1,C6]"
    ],
    [
     "C6",
     "x_vars[S1,C7]"
    ],
    [
     "C7",
     "x_vars[S1,C8]"
    ],
    [
     "C8",
     "x_vars[S1,C9]"
    ],
    [
     "C9",
     "x_vars[S1,C10]"
    ],
    [
     "C10",
     "x_vars[S1,C11]"
    ],
    [
     "C11",
     "x_vars[S1,C12]"
    ],
    [
     "C12",
     "x_vars[S1,C13]"
    ],
    [
     "C13",
     "x_vars[S1,C14]"
    ],
    [
     "C14",
     "x_vars[S1,C15]"
    ],
    [
     "C15",
     "x_vars[S1,C16]"
    ],
    [
     "C16",
     "x_vars[S1,C17]"
    ],
    [
     "C17",
     "x_vars[S1,C18]"
    ],
    [
     "C18",
     "x_vars[S1,C19]"
    ],
    [
     "C19",
     "x_vars[S1,C20]"
    ],
    [
     "C20",
     "x_vars[S1,C21]"
    ],
    [
     "C21",
     "x_vars[S1,C22]"
    ],
    [
     "C22",
     "x_vars[S1,C23]"
    ],
    [
     "C23",
     "x_vars[S1,C24]"
    ],
    [
     "C24",
     "x_vars[S1,C25]"
    ],
    [
     "C25",
     "x_vars[S2,C1]"
    ],
    [
     "C26",
     "x_vars[S2,C2]"
    ],
    [
     "C27",
     "x_vars[S2,C3]"
    ],
    [
     "C28",
     "x_vars[S2,C4]"
    ],
    [
     "C29",
     "x_vars[S2,C5]"
    ],
    [
     "C30",
     "x_vars[S2,C6]"
    ],
    [
     "C31",
     "x_vars[S2,C7]"
    ],
    [
     "C32",
     "x_vars[S2,C8]"
    ],
    [
     "C33",
     "x_vars[S2,C9]"
    ],
    [
     "C34",
     "x_vars[S2,C10]"
    ],
    [
     "C35",
     "x_vars[S2,C11]"
    ],
    [
     "C36",
     "x_vars[S2,C12]"
    ],
    [
     "C37",
     "x_vars[S2,C13]"
    ],
    [
     "C38",
     "x_vars[S2,C14]"
    ],
    [
     "C39",
     "x_vars[S2,C15]"
    ],
    [
     "C40",
     "x_vars[S2,C16]"
    ],
    [
     "C41",
     "x_vars[S2,C17]"
    ],
    [
     "C42",
     "x_vars[S2,C18]"
    ],
    [
     "C43",
     "x_vars[S2,C19]"
    ],
    [
     "C44",
     "x_vars[S2,C20]"
    ],
    [
     "C45",
     "x_vars[S2,C21]"
    ],
    [
     "C46",
     "x_vars[S2,C22]"
    ],
    [
     "C47",
     "x_vars[S2,C23]"
    ],
    [
     "C48",
     "x_vars[S2,C24]"
    ],
    [
     "C49",
     "x_vars[S2,C25]"
    ],
    [
     "C50",
     "x_vars[S3,C1]"
    ],
    [
     "C51",
     "x_vars[S3,C2]"
    ],
    [
     "C52",
     "x_vars[S3,C3]"
    ],
    [
     "C53",
     "x_vars[S3,C4]"
    ],
    [
     "C54",
     "x_vars[S3,C5]"
    ],
    [
     "C55",
     "x_vars[S3,C6]"
    ],
    [
     "C56",
     "x_vars[S3,C7]"
    ],
    [
     "C57",
     "x_vars[S3,C8]"
    ],
    [
     "C58",
     "x_vars[S3,C9]"
    ],
    [
     "C59",
     "x_vars[S3,C10]"
    ],
    [
     "C60",
     "x_vars[S3,C11]"
    ],
    [
     "C61",
     "x_vars[S3,C12]"
    ],
    [
     "C62",
     "x_vars[S3,C13]"
    ],
    [
     "C63",
     "x_vars[S3,C14]"
    ],
    [
     "C64",
     "x_vars[S3,C15]"
    ],
    [
     "C65",
     "x_vars[S3,C16]"
    ],
    [
     "C66",
     "x_vars[S3,C17]"
    ],
    [
     "C67",
     "x_vars[S3,C18]"
    ],
    [
     "C68",
     "x_vars[S3,C19]"
    ],
    [
     "C69",
     "x_vars[S3,C20]"
    ],
    [
     "C70",
     "x_vars[S3,C21]"
    ],
    [
     "C71",
     "x_vars[S3,C22]"
    ],
    [
     "C72",
     "x_vars[S3,C23]"
    ],
    [
     "C73",
     "x_vars[S3,C24]"
    ],
    [
     "C74",
     "x_vars[S3,C25]"
    ],
    [
     "C75",
     "x_vars[S4,C1]"
    ],
    [
     "C76",
     "x_vars[S4,C2]"
    ],
    [
     "C77",
     "x_vars[S4,C3]"
    ],
    [
     "C78",
     "x_vars[S4,C4]"
    ],
    [
     "C79",
     "x_vars[S4,C5]"
    ],
    [
     "C80",
     "x_vars[S4,C6]"
    ],
    [
     "C81",
     "x_vars[S4,C7]"
    ],
    [
     "C82",
     "x_vars[S4,C8]"
    ],
    [
     "C83",
     "x_vars[S4,C9]"
    ],
    [
     "C84",
     "x_vars[S4,C10]"
    ],
    [
     "C85",
     "x_vars[S4,C11]"
    ],
    [
     "C86",
     "x_vars[S4,C12]"
    ],
    [
     "C87",
     "x_vars[S4,C13]"
    ],
    [
     "C88",
     "x_vars[S4,C14]"
    ],
    [
     "C89",
     "x_vars[S4,C15]"
    ],
    [
     "C90",
     "x_vars[S4,C16]"
    ],
    [
     "C91",
     "x_vars[S4,C17]"
    ],
    [
     "C92",
     "x_vars[S4,C18]"
    ],
    [
     "C93",
     "x_vars[S4,C19]"
    ],
    [
     "C94",
     "x_vars[S4,C20]"
    ],
    [
     "C95",
     "x_vars[S4,C21]"
    ],
    [
     "C96",
     "x_vars[S4,C22]"
    ],
    [
     "C97",
     "x_vars[S4,C23]"
    ],
    [
     "C98",
     "x_vars[S4,C24]"
    ],
    [
     "C99",
     "x_vars[S4,C25]"
    ],
    [
     "C100",
     "x_vars[S5,C1]"
    ],
    [
     "C101",
     "x_vars[S5,C2]"
    ],
    [
     "C102",
     "x_vars[S5,C3]"
    ],
    [
     "C103",
     "x_vars[S5,C4]"
    ],
    [
     "C104",
     "x_vars[S5,C5]"
    ],
    [
     "C105",
     "x_vars[S5,C6]"
    ],
    [
     "C106",
     "x_vars[S5,C7]"
    ],
    [
     "C107",
     "x_vars[S5,C8]"
    ],
    [
     "C108",
     "x_vars[S5,C9]"
    ],
    [
     "C109",
     "x_vars[S5,C10]"
    ],
    [
     "C110",
     "x_vars[S5,C11]"
    ],
    [
     "C111",
     "x_vars[S5,C12]"
    ],
    [
     "C112",
     "x_vars[S5,C13]"
    ],
    [
     "C113",
     "x_vars[S5,C14]"
    ],
    [
     "C114",
     "x_vars[S5,C15]"
    ],
    [
     "C115",
     "x_vars[S5,C16]"
    ],
    [
     "C116",
     "x_vars[S5,C17]"
    ],
    [
     "C117",
     "x_vars[S5,C18]"
    ],
    [
     "C118",
     "x_vars[S5,C19]"
    ],
    [
     "C119",
     "x_vars[S5,C20]"
    ],
    [
     "C120",
     "x_vars[S5,C21]"
    ],
    [
     "C121",
     "x_vars[S5,C22]"
    ],
    [
     "C122",
     "x_vars[S5,C23]"
    ],
    [
     "C123",
     "x_vars[S5,C24]"
    ],
    [
     "C124",
     "x_vars[S5,C25]"
    ],
    [
     "C125",
     "x_vars[S6,C1]"
    ],
    [
     "C126",
     "x_vars[S6,C2]"
    ],
    [
     "C127",
     "x_vars[S6,C3]"
    ],
    [
     "C128",
     "x_vars[S6,C4]"
    ],
    [
     "C129",
     "x_vars[S6,C5]"
    ],
    [
     "C130",
     "x_vars[S6,C6]"
    ],
    [
     "C131",
     "x_vars[S6,C7]"
    ],
    [
     "C132",
     "x_vars[S6,C8]"
    ],
    [
     "C133",
     "x_vars[S6,C9]"
    ],
    [
     "C134",
     "x_vars[S6,C10]"
    ],
    [
     "C135",
     "x_vars[S6,C11]"
    ],
    [
     "C136",
     "x_vars[S6,C12]"
    ],
    [
     "C137",
     "x_vars[S6,C13]"
    ],
    [
     "C138",
     "x_vars[S6,C14]"
    ],
    [
     "C139",
     "x_vars[S6,C15]"
    ],
    [
     "C140",
     "x_vars[S6,C16]"
    ],
    [
     "C141",
     "x_vars[S6,C17]"
    ],
    [
     "C142",
     "x_vars[S6,C18]"
    ],
    [
     "C143",
     "x_vars[S6,C19]"
    ],
    [
     "C144",
     "x_vars[S6,C20]"
    ],
    [
     "C145",
     "x_vars[S6,C21]"
    ],
    [
     "C146",
     "x_vars[S6,C22]"
    ],
    [
     "C147",
     "x_vars[S6,C23]"
    ],
    [
     "C148",
     "x_vars[S6,C24]"
    ],
    [
     "C149",
     "x_vars[S6,C25]"
    ],
    [
     "C150",
     "x_vars[S7,C1]"
    ],
    [
     "C151",
     "x_vars[S7,C2]"
    ],
    [
     "C152",
     "x_vars[S7,C3]"
    ],
    [
     "C153",
     "x_vars[S7,C4]"
    ],
    [
     "C154",
     "x_vars[S7,C5]"
    ],
    [
     "C155",
     "x_vars[S7,C6]"
    ],
    [
     "C156",
     "x_vars[S7,C7]"
    ],
    [
     "C157",
     "x_vars[S7,C8]"
    ],
    [
     "C158",
     "x_vars[S7,C9]"
    ],
    [
     "C159",
     "x_vars[S7,C10]"
    ],
    [
     "C160",
     "x_vars[S7,C11]"
    ],
    [
     "C161",
     "x_vars[S7,C12]"
    ],
    [
     "C162",
     "x_vars[S7,C13]"
    ],
    [
     "C163",
     "x_vars[S7,C14]"
    ],
    [
     "C164",
     "x_vars[S7,C15]"
    ],
    [
     "C165",
     "x_vars[S7,C16]"
    ],
    [
     "C166",
     "x_vars[S7,C17]"
    ],
    [
     "C167",
     "x_vars[S7,C18]"
    ],
    [
     "C168",
     "x_vars[S7,C19]"
    ],
    [
     "C169",
     "x_vars[S7,C20]"
    ],
    [
     "C170",
     "x_vars[S7,C21]"
    ],
    [
     "C171",
     "x_vars[S7,C22]"
    ],
    [
     "C172",
     "x_vars[S7,C23]"
    ],
    [
     "C173",
     "x_vars[S7,C24]"
    ],
    [
     "C174",
     "x_vars[S7,C25]"
    ],
    [
     "C175",
     "x_vars[S8,C1]"
    ],
    [
     "C176",
     "x_vars[S8,C2]"
    ],
    [
     "C177",
     "x_vars[S8,C3]"
    ],
    [
     "C178",
     "x_vars[S8,C4]"
    ],
    [
     "C179",
     "x_vars[S8,C5]"
    ],
    [
     "C180",
     "x_vars[S8,C6]"
    ],
    [
     "C181",
     "x_vars[S8,C7]"
    ],
    [
     "C182",
     "x_vars[S8,C8]"
    ],
    [
     "C183",
     "x_vars[S8,C9]"
    ],
    [
     "C184",
     "x_vars[S8,C10]"
    ],
    [
     "C185",
     "x_vars[S8,C11]"
    ],
    [
     "C186",
     "x_vars[S8,C12]"
    ],
    [
     "C187",
     "x_vars[S8,C13]"
    ],
    [
     "C188",
     "x_vars[S8,C14]"
    ],
    [
     "C189",
     "x_vars[S8,C15]"
    ],
    [
     "C190",
     "x_vars[S8,C16]"
    ],
    [
     "C191",
     "x_vars[S8,C17]"
    ],
    [
     "C192",
     "x_vars[S8,C18]"
    ],
    [
     "C193",
     "x_vars[S8,C19]"
    ],
    [
     "C194",
     "x_vars[S8,C20]"
    ],
    [
     "C195",
     "x_vars[S8,C21]"
    ],
    [
     "C196",
     "x_vars[S8,C22]"
    ],
    [
     "C197",
     "x_vars[S8,C23]"
    ],
    [
     "C198",
     "x_vars[S8,C24]"
    ],
    [
     "C199",
     "x_vars[S8,C25]"
    ],
    [
     "C200",
     "x_vars[S9,C1]"
    ],
    [
     "C201",
     "x_vars[S9,C2]"
    ],
    [
     "C202",
     "x_vars[S9,C3]"
    ],
    [
     "C203",
     "x_vars[S9,C4]"
    ],
    [
     "C204",
     "x_vars[S9,C5]"
    ],
    [
     "C205",
     "x_vars[S9,C6]"
    ],
    [
     "C206",
     "x_vars[S9,C7]"
    ],
    [
     "C207",
     "x_vars[S9,C8]"
    ],
    [
     "C208",
     "x_vars[S9,C9]"
    ],
    [
     "C209",
     "x_vars[S9,C10]"
    ],
    [
     "C210",
     "x_vars[S9,C11]"
    ],
    [
     "C211",
     "x_vars[S9,C12]"
    ],
    [
     "C212",
     "x_vars[S9,C13]"
    ],
    [
     "C213",
     "x_vars[S9,C14]"
    ],
    [
     "C214",
     "x_vars[S9,C15]"
    ],
    [
     "C215",
     "x_vars[S9,C16]"
    ],
    [
     "C216",
     "x_vars[S9,C17]"
    ],
    [
     "C217",
     "x_vars[S9,C18]"
    ],
    [
     "C218",
     "x_vars[S9,C19]"
    ],
    [
     "C219",
     "x_vars[S9,C20]"
    ],
    [
     "C220",
     "x_vars[S9,C21]"
    ],
    [
     "C221",
     "x_vars[S9,C22]"
    ],
    [
     "C222",
     "x_vars[S9,C23]"
    ],
    [
     "C223",
     "x_vars[S9,C24]"
    ],
    [
     "C224",
     "x_vars[S9,C25]"
    ],
    [
     "C225",
     "x_vars[S10,C1]"
    ],
    [
     "C226",
     "x_vars[S10,C2]"
    ],
    [
     "C227",
     "x_vars[S10,C3]"
    ],
    [
     "C228",
     "x_vars[S10,C4]"
    ],
    [
     "C229",
     "x_vars[S10,C5]"
    ],
    [
     "C230",
     "x_vars[S10,C6]"
    ],
    [
     "C231",
     "x_vars[S10,C7]"
    ],
    [
     "C232",
     "x_vars[S10,C8]"
    ],
    [
     "C233",
     "x_vars[S10,C9]"
    ],
    [
     "C234",
     "x_vars[S10,C10]"
    ],
    [
     "C235",
     "x_vars[S10,C11]"
    ],
    [
     "C236",
     "x_vars[S10,C12]"
    ],
    [
     "C237",
     "x_vars[S10,C13]"
    ],
    [
     "C238",
     "x_vars[S10,C14]"
    ],
    [
     "C239",
     "x_vars[S10,C15]"
    ],
    [
     "C240",
     "x_vars[S10,C16]"
    ],
    [
     "C241",
     "x_vars[S10,C17]"
    ],
    [
     "C242",
     "x_vars[S10,C18]"
    ],
    [
     "C243",
     "x_vars[S10,C19]"
    ],
    [
     "C244",
     "x_vars[S10,C20]"
    ],
    [
     "C245",
     "x_vars[S10,C21]"
    ],
    [
     "C246",
     "x_vars[S10,C22]"
    ],
    [
     "C247",
     "x_vars[S10,C23]"
    ],
    [
     "C248",
     "x_vars[S10,C24]"
    ],
    [
     "C249",
     "x_vars[S10,C25]"
    ],
    [
     "C250",
     "x_vars[S11,C1]"
    ],
    [
     "C251",
     "x_vars[S11,C2]"
    ],
    [
     "C252",
     "x_vars[S11,C3]"
    ],
    [
     "C253",
     "x_vars[S11,C4]"
    ],
    [
     "C254",
     "x_vars[S11,C5]"
    ],
    [
     "C255",
     "x_vars[S11,C6]"
    ],
    [
     "C256",
     "x_vars[S11,C7]"
    ],
    [
     "C257",
     "x_vars[S11,C8]"
    ],
    [
     "C258",
     "x_vars[S11,C9]"
    ],
    [
     "C259",
     "x_vars[S11,C10]"
    ],
    [
     "C260",
     "x_vars[S11,C11]"
    ],
    [
     "C261",
     "x_vars[S11,C12]"
    ],
    [
     "C262",
     "x_vars[S11,C13]"
    ],
    [
     "C263",
     "x_vars[S11,C14]"
    ],
    [
     "C264",
     "x_vars[S11,C15]"
    ],
    [
     "C265",
     "x_vars[S11,C16]"
    ],
    [
     "C266",
     "x_vars[S11,C17]"
    ],
    [
     "C267",
     "x_vars[S11,C18]"
    ],
    [
     "C268",
     "x_vars[S11,C19]"
    ],
    [
     "C269",
     "x_vars[S11,C20]"
    ],
    [
     "C270",
     "x_vars[S11,C21]"
    ],
    [
     "C271",
     "x_vars[S11,C22]"
    ],
    [
     "C272",
     "x_vars[S11,C23]"
    ],
    [
     "C273",
     "x_vars[S11,C24]"
    ],
    [
     "C274",
     "x_vars[S11,C25]"
    ],
    [
     "C275",
     "x_vars[S12,C1]"
    ],
    [
     "C276",
     "x_vars[S12,C2]"
    ],
    [
     "C277",
     "x_vars[S12,C3]"
    ],
    [
     "C278",
     "x_vars[S12,C4]"
    ],
    [
     "C279",
     "x_vars[S12,C5]"
    ],
    [
     "C280",
     "x_vars[S12,C6]"
    ],
    [
     "C281",
     "x_vars[S12,C7]"
    ],
    [
     "C282",
     "x_vars[S12,C8]"
    ],
    [
     "C283",
     "x_vars[S12,C9]"
    ],
    [
     "C284",
     "x_vars[S12,C10]"
    ],
    [
     "C285",
     "x_vars[S12,C11]"
    ],
    [
     "C286",
     "x_vars[S12,C12]"
    ],
    [
     "C287",
     "x_vars[S12,C13]"
    ],
    [
     "C288",
     "x_vars[S12,C14]"
    ],
    [
     "C289",
     "x_vars[S12,C15]"
    ],
    [
     "C290",
     "x_vars[S12,C16]"
    ],
    [
     "C291",
     "x_vars[S12,C17]"
    ],
    [
     "C292",
     "x_vars[S12,C18]"
    ],
    [
     "C293",
     "x_vars[S12,C19]"
    ],
    [
     "C294",
     "x_vars[S12,C20]"
    ],
    [
     "C295",
     "x_vars[S12,C21]"
    ],
    [
     "C296",
     "x_vars[S12,C22]"
    ],
    [
     "C297",
     "x_vars[S12,C23]"
    ],
    [
     "C298",
     "x_vars[S12,C24]"
    ],
    [
     "C299",
     "x_vars[S12,C25]"
    ],
    [
     "C300",
     "x_vars[S13,C1]"
    ],
    [
     "C301",
     "x_vars[S13,C2]"
    ],
    [
     "C302",
     "x_vars[S13,C3]"
    ],
    [
     "C303",
     "x_vars[S13,C4]"
    ],
    [
     "C304",
     "x_vars[S13,C5]"
    ],
    [
     "C305",
     "x_vars[S13,C6]"
    ],
    [
     "C306",
     "x_vars[S13,C7]"
    ],
    [
     "C307",
     "x_vars[S13,C8]"
    ],
    [
     "C308",
     "x_vars[S13,C9]"
    ],
    [
     "C309",
     "x_vars[S13,C10]"
    ],
    [
     "C310",
     "x_vars[S13,C11]"
    ],
    [
     "C311",
     "x_vars[S13,C12]"
    ],
    [
     "C312",
     "x_vars[S13,C13]"
    ],
    [
     "C313",
     "x_vars[S13,C14]"
    ],
    [
     "C314",
     "x_vars[S13,C15]"
    ],
    [
     "C315",
     "x_vars[S13,C16]"
    ],
    [
     "C316",
     "x_vars[S13,C17]"
    ],
    [
     "C317",
     "x_vars[S13,C18]"
    ],
    [
     "C318",
     "x_vars[S13,C19]"
    ],
    [
     "C319",
     "x_vars[S13,C20]"
    ],
    [
     "C320",
     "x_vars[S13,C21]"
    ],
    [
     "C321",
     "x_vars[S13,C22]"
    ],
    [
     "C322",
     "x_vars[S13,C23]"
    ],
    [
     "C323",
     "x_vars[S13,C24]"
    ],
    [
     "C324",
     "x_vars[S13,C25]"
    ],
    [
     "C325",
     "x_vars[S14,C1]"
    ],
    [
     "C326",
     "x_vars[S14,C2]"
    ],
    [
     "C327",
     "x_vars[S14,C3]"
    ],
    [
     "C328",
     "x_vars[S14,C4]"
    ],
    [
     "C329",
     "x_vars[S14,C5]"
    ],
    [
     "C330",
     "x_vars[S14,C6]"
    ],
    [
     "C331",
     "x_vars[S14,C7]"
    ],
    [
     "C332",
     "x_vars[S14,C8]"
    ],
    [
     "C333",
     "x_vars[S14,C9]"
    ],
    [
     "C334",
     "x_vars[S14,C10]"
    ],
    [
     "C335",
     "x_vars[S14,C11]"
    ],
    [
     "C336",
     "x_vars[S14,C12]"
    ],
    [
     "C337",
     "x_vars[S14,C13]"
    ],
    [
     "C338",
     "x_vars[S14,C14]"
    ],
    [
     "C339",
     "x_vars[S14,C15]"
    ],
    [
     "C340",
     "x_vars[S14,C16]"
    ],
    [
     "C341",
     "x_vars[S14,C17]"
    ],
    [
     "C342",
     "x_vars[S14,C18]"
    ],
    [
     "C343",
     "x_vars[S14,C19]"
    ],
    [
     "C344",
     "x_vars[S14,C20]"
    ],
    [
     "C345",
     "x_vars[S14,C21]"
    ],
    [
     "C346",
     "x_vars[S14,C22]"
    ],
    [
     "C347",
     "x_vars[S14,C23]"
    ],
    [
     "C348",
     "x_vars[S14,C24]"
    ],
    [
     "C349",
     "x_vars[S14,C25]"
    ],
    [
     "C350",
     "x_vars[S15,C1]"
    ],
    [
     "C351",
     "x_vars[S15,C2]"
    ],
    [
     "C352",
     "x_vars[S15,C3]"
    ],
    [
     "C353",
     "x_vars[S15,C4]"
    ],
    [
     "C354",
     "x_vars[S15,C5]"
    ],
    [
     "C355",
     "x_vars[S15,C6]"
    ],
    [
     "C356",
     "x_vars[S15,C7]"
    ],
    [
     "C357",
     "x_vars[S15,C8]"
    ],
    [
     "C358",
     "x_vars[S15,C9]"
    ],
    [
     "C359",
     "x_vars[S15,C10]"
    ],
    [
     "C360",
     "x_vars[S15,C11]"
    ],
    [
     "C361",
     "x_vars[S15,C12]"
    ],
    [
     "C362",
     "x_vars[S15,C13]"
    ],
    [
     "C363",
     "x_vars[S15,C14]"
    ],
    [
     "C364",
     "x_vars[S15,C15]"
    ],
    [
     "C365",
     "x_vars[S15,C16]"
    ],
    [
     "C366",
     "x_vars[S15,C17]"
    ],
    [
     "C367",
     "x_vars[S15,C18]"
    ],
    [
     "C368",
     "x_vars[S15,C19]"
    ],
    [
     "C369",
     "x_vars[S15,C20]"
    ],
    [
     "C370",
     "x_vars[S15,C21]"
    ],
    [
     "C371",
     "x_vars[S15,C22]"
    ],
    [
     "C372",
     "x_vars[S15,C23]"
    ],
    [
     "C373",
     "x_vars[S15,C24]"
    ],
    [
     "C374",
     "x_vars[S15,C25]"
    ],
    [
     "C375",
     "x_vars[S16,C1]"
    ],
    [
     "C376",
     "x_vars[S16,C2]"
    ],
    [
     "C377",
     "x_vars[S16,C3]"
    ],
    [
     "C378",
     "x_vars[S16,C4]"
    ],
    [
     "C379",
     "x_vars[S16,C5]"
    ],
    [
     "C380",
     "x_vars[S16,C6]"
    ],
    [
     "C381",
     "x_vars[S16,C7]"
    ],
    [
     "C382",
     "x_vars[S16,C8]"
    ],
    [
     "C383",
     "x_vars[S16,C9]"
    ],
    [
     "C384",
     "x_vars[S16,C10]"
    ],
    [
     "C385",
     "x_vars[S16,C11]"
    ],
    [
     "C386",
     "x_vars[S16,C12]"
    ],
    [
     "C387",
     "x_vars[S16,C13]"
    ],
    [
     "C388",
     "x_vars[S16,C14]"
    ],
    [
     "C389",
     "x_vars[S16,C15]"
    ],
    [
     "C390",
     "x_vars[S16,C16]"
    ],
    [
     "C391",
     "x_vars[S16,C17]"
    ],
    [
     "C392",
     "x_vars[S16,C18]"
    ],
    [
     "C393",
     "x_vars[S16,C19]"
    ],
    [
     "C394",
     "x_vars[S16,C20]"
    ],
    [
     "C395",
     "x_vars[S16,C21]"
    ],
    [
     "C396",
     "x_vars[S16,C22]"
    ],
    [
     "C397",
     "x_vars[S16,C23]"
    ],
    [
     "C398",
     "x_vars[S16,C24]"
    ],
    [
     "C399",
     "x_vars[S16,C25]"
    ],
    [
     "C400",
     "x_vars[S17,C1]"
    ],
    [
     "C401",
     "x_vars[S17,C2]"
    ],
    [
     "C402",
     "x_vars[S17,C3]"
    ],
    [
     "C403",
     "x_vars[S17,C4]"
    ],
    [
     "C404",
     "x_vars[S17,C5]"
    ],
    [
     "C405",
     "x_vars[S17,C6]"
    ],
    [
     "C406",
     "x_vars[S17,C7]"
    ],
    [
     "C407",
     "x_vars[S17,C8]"
    ],
    [
     "C408",
     "x_vars[S17,C9]"
    ],
    [
     "C409",
     "x_vars[S17,C10]"
    ],
    [
     "C410",
     "x_vars[S17,C11]"
    ],
    [
     "C411",
     "x_vars[S17,C12]"
    ],
    [
     "C412",
     "x_vars[S17,C13]"
    ],
    [
     "C413",
     "x_vars[S17,C14]"
    ],
    [
     "C414",
     "x_vars[S17,C15]"
    ],
    [
     "C415",
     "x_vars[S17,C16]"
    ],
    [
     "C416",
     "x_vars[S17,C17]"
    ],
    [
     "C417",
     "x_vars[S17,C18]"
    ],
    [
     "C418",
     "x_vars[S17,C19]"
    ],
    [
     "C419",
     "x_vars[S17,C20]"
    ],
    [
     "C420",
     "x_vars[S17,C21]"
    ],
    [
     "C421",
     "x_vars[S17,C22]"
    ],
    [
     "C422",
     "x_vars[S17,C23]"
    ],
    [
     "C423",
     "x_vars[S17,C24]"
    ],
    [
     "C424",
     "x_vars[S17,C25]"
    ],
    [
     "C425",
     "x_vars[S18,C1]"
    ],
    [
     "C426",
     "x_vars[S18,C2]"
    ],
    [
     "C427",
     "x_vars[S18,C3]"
    ],
    [
     "C428",
     "x_vars[S18,C4]"
    ],
    [
     "C429",
     "x_vars[S18,C5]"
    ],
    [
     "C430",
     "x_vars[S18,C6]"
    ],
    [
     "C431",
     "x_vars[S18,C7]"
    ],
    [
     "C432",
     "x_vars[S18,C8]"
    ],
    [
     "C433",
     "x_vars[S18,C9]"
    ],
    [
     "C434",
     "x_vars[S18,C10]"
    ],
    [
     "C435",
     "x_vars[S18,C11]"
    ],
    [
     "C436",
     "x_vars[S18,C12]"
    ],
    [
     "C437",
     "x_vars[S18,C13]"
    ],
    [
     "C438",
     "x_vars[S18,C14]"
    ],
    [
     "C439",
     "x_vars[S18,C15]"
    ],
    [
     "C440",
     "x_vars[S18,C16]"
    ],
    [
     "C441",
     "x_vars[S18,C17]"
    ],
    [
     "C442",
     "x_vars[S18,C18]"
    ],
    [
     "C443",
     "x_vars[S18,C19]"
    ],
    [
     "C444",
     "x_vars[S18,C20]"
    ],
    [
     "C445",
     "x_vars[S18,C21]"
    ],
    [
     "C446",
     "x_vars[S18,C22]"
    ],
    [
     "C447",
     "x_vars[S18,C23]"
    ],
    [
     "C448",
     "x_vars[S18,C24]"
    ],
    [
     "C449",
     "x_vars[S18,C25]"
    ],
    [
     "C450",
     "x_vars[S19,C1]"
    ],
    [
     "C451",
     "x_vars[S19,C2]"
    ],
    [
     "C452",
     "x_vars[S19,C3]"
    ],
    [
     "C453",
     "x_vars[S19,C4]"
    ],
    [
     "C454",
     "x_vars[S19,C5]"
    ],
    [
     "C455",
     "x_vars[S19,C6]"
    ],
    [
     "C456",
     "x_vars[S19,C7]"
    ],
    [
     "C457",
     "x_vars[S19,C8]"
    ],
    [
     "C458",
     "x_vars[S19,C9]"
    ],
    [
     "C459",
     "x_vars[S19,C10]"
    ],
    [
     "C460",
     "x_vars[S19,C11]"
    ],
    [
     "C461",
     "x_vars[S19,C12]"
    ],
    [
     "C462",
     "x_vars[S19,C13]"
    ],
    [
     "C463",
     "x_vars[S19,C14]"
    ],
    [
     "C464",
     "x_vars[S19,C15]"
    ],
    [
     "C465",
     "x_vars[S19,C16]"
    ],
    [
     "C466",
     "x_vars[S19,C17]"
    ],
    [
     "C467",
     "x_vars[S19,C18]"
    ],
    [
     "C468",
     "x_vars[S19,C19]"
    ],
    [
     "C469",
     "x_vars[S19,C20]"
    ],
    [
     "C470",
     "x_vars[S19,C21]"
    ],
    [
     "C471",
     "x_vars[S19,C22]"
    ],
    [
     "C472",
     "x_vars[S19,C23]"
    ],
    [
     "C473",
     "x_vars[S19,C24]"
    ],
    [
     "C474",
     "x_vars[S19,C25]"
    ],
    [
     "C475",
     "x_vars[S20,C1]"
    ],
    [
     "C476",
     "x_vars[S20,C2]"
    ],
    [
     "C477",
     "x_vars[S20,C3]"
    ],
    [
     "C478",
     "x_vars[S20,C4]"
    ],
    [
     "C479",
     "x_vars[S20,C5]"
    ],
    [
     "C480",
     "x_vars[S20,C6]"
    ],
    [
     "C481",
     "x_vars[S20,C7]"
    ],
    [
     "C482",
     "x_vars[S20,C8]"
    ],
    [
     "C483",
     "x_vars[S20,C9]"
    ],
    [
     "C484",
     "x_vars[S20,C10]"
    ],
    [
     "C485",
     "x_vars[S20,C11]"
    ],
    [
     "C486",
     "x_vars[S20,C12]"
    ],
    [
     "C487",
     "x_vars[S20,C13]"
    ],
    [
     "C488",
     "x_vars[S20,C14]"
    ],
    [
     "C489",
     "x_vars[S20,C15]"
    ],
    [
     "C490",
     "x_vars[S20,C16]"
    ],
    [
     "C491",
     "x_vars[S20,C17]"
    ],
    [
     "C492",
     "x_vars[S20,C18]"
    ],
    [
     "C493",
     "x_vars[S20,C19]"
    ],
    [
     "C494",
     "x_vars[S20,C20]"
    ],
    [
     "C495",
     "x_vars[S20,C21]"
    ],
    [
     "C496",
     "x_vars[S20,C22]"
    ],
    [
     "C497",
     "x_vars[S20,C23]"
    ],
    [
     "C498",
     "x_vars[S20,C24]"
    ],
    [
     "C499",
     "x_vars[S20,C25]"
    ],
    [
     "C500",
     "x_vars[S21,C1]"
    ],
    [
     "C501",
     "x_vars[S21,C2]"
    ],
    [
     "C502",
     "x_vars[S21,C3]"
    ],
    [
     "C503",
     "x_vars[S21,C4]"
    ],
    [
     "C504",
     "x_vars[S21,C5]"
    ],
    [
     "C505",
     "x_vars[S21,C6]"
    ],
    [
     "C506",
     "x_vars[S21,C7]"
    ],
    [
     "C507",
     "x_vars[S21,C8]"
    ],
    [
     "C508",
     "x_vars[S21,C9]"
    ],
    [
     "C509",
     "x_vars[S21,C10]"
    ],
    [
     "C510",
     "x_vars[S21,C11]"
    ],
    [
     "C511",
     "x_vars[S21,C12]"
    ],
    [
     "C512",
     "x_vars[S21,C13]"
    ],
    [
     "C513",
     "x_vars[S21,C14]"
    ],
    [
     "C514",
     "x_vars[S21,C15]"
    ],
    [
     "C515",
     "x_vars[S21,C16]"
    ],
    [
     "C516",
     "x_vars[S21,C17]"
    ],
    [
     "C517",
     "x_vars[S21,C18]"
    ],
    [
     "C518",
     "x_vars[S21,C19]"
    ],
    [
     "C519",
     "x_vars[S21,C20]"
    ],
    [
     "C520",
     "x_vars[S21,C21]"
    ],
    [
     "C521",
     "x_vars[S21,C22]"
    ],
    [
     "C522",
     "x_vars[S21,C23]"
    ],
    [
     "C523",
     "x_vars[S21,C24]"
    ],
    [
     "C524",
     "x_vars[S21,C25]"
    ],
    [
     "C525",
     "x_vars[S22,C1]"
    ],
    [
     "C526",
     "x_vars[S22,C2]"
    ],
    [
     "C527",
     "x_vars[S22,C3]"
    ],
    [
     "C528",
     "x_vars[S22,C4]"
    ],
    [
     "C529",
     "x_vars[S22,C5]"
    ],
    [
     "C530",
     "x_vars[S22,C6]"
    ],
    [
     "C531",
     "x_vars[S22,C7]"
    ],
    [
     "C532",
     "x_vars[S22,C8]"
    ],
    [
     "C533",
     "x_vars[S22,C9]"
    ],
    [
     "C534",
     "x_vars[S22,C10]"
    ],
    [
     "C535",
     "x_vars[S22,C11]"
    ],
    [
     "C536",
     "x_vars[S22,C12]"
    ],
    [
     "C537",
     "x_vars[S22,C13]"
    ],
    [
     "C538",
     "x_vars[S22,C14]"
    ],
    [
     "C539",
     "x_vars[S22,C15]"
    ],
    [
     "C540",
     "x_vars[S22,C16]"
    ],
    [
     "C541",
     "x_vars[S22,C17]"
    ],
    [
     "C542",
     "x_vars[S22,C18]"
    ],
    [
     "C543",
     "x_vars[S22,C19]"
    ],
    [
     "C544",
     "x_vars[S22,C20]"
    ],
    [
     "C545",
     "x_vars[S22,C21]"
    ],
    [
     "C546",
     "x_vars[S22,C22]"
    ],
    [
     "C547",
     "x_vars[S22,C23]"
    ],
    [
     "C548",
     "x_vars[S22,C24]"
    ],
    [
     "C549",
     "x_vars[S22,C25]"
    ],
    [
     "C550",
     "x_vars[S23,C1]"
    ],
    [
     "C551",
     "x_vars[S23,C2]"
    ],
    [
     "C552",
     "x_vars[S23,C3]"
    ],
    [
     "C553",
     "x_vars[S23,C4]"
    ],
    [
     "C554",
     "x_vars[S23,C5]"
    ],
    [
     "C555",
     "x_vars[S23,C6]"
    ],
    [
     "C556",
     "x_vars[S23,C7]"
    ],
    [
     "C557",
     "x_vars[S23,C8]"
    ],
    [
     "C558",
     "x_vars[S23,C9]"
    ],
    [
     "C559",
     "x_vars[S23,C10]"
    ],
    [
     "C560",
     "x_vars[S23,C11]"
    ],
    [
     "C561",
     "x_vars[S23,C12]"
    ],
    [
     "C562",
     "x_vars[S23,C13]"
    ],
    [
     "C563",
     "x_vars[S23,C14]"
    ],
    [
     "C564",
     "x_vars[S23,C15]"
    ],
    [
     "C565",
     "x_vars[S23,C16]"
    ],
    [
     "C566",
     "x_vars[S23,C17]"
    ],
    [
     "C567",
     "x_vars[S23,C18]"
    ],
    [
     "C568",
     "x_vars[S23,C19]"
    ],
    [
     "C569",
     "x_vars[S23,C20]"
    ],
    [
     "C570",
     "x_vars[S23,C21]"
    ],
    [
     "C571",
     "x_vars[S23,C22]"
    ],
    [
     "C572",
     "x_vars[S23,C23]"
    ],
    [
     "C573",
     "x_vars[S23,C24]"
    ],
    [
     "C574",
     "x_vars[S23,C25]"
    ],
    [
     "C575",
     "x_vars[S24,C1]"
    ],
    [
     "C576",
     "x_vars[S24,C2]"
    ],
    [
     "C577",
     "x_vars[S24,C3]"
    ],
    [
     "C578",
     "x_vars[S24,C4]"
    ],
    [
     "C579",
     "x_vars[S24,C5]"
    ],
    [
     "C580",
     "x_vars[S24,C6]"
    ],
    [
     "C581",
     "x_vars[S24,C7]"
    ],
    [
     "C582",
     "x_vars[S24,C8]"
    ],
    [
     "C583",
     "x_vars[S24,C9]"
    ],
    [
     "C584",
     "x_vars[S24,C10]"
    ],
    [
     "C585",
     "x_vars[S24,C11]"
    ],
    [
     "C586",
     "x_vars[S24,C12]"
    ],
    [
     "C587",
     "x_vars[S24,C13]"
    ],
    [
     "C588",
     "x_vars[S24,C14]"
    ],
    [
     "C589",
     "x_vars[S24,C15]"
    ],
    [
     "C590",
     "x_vars[S24,C16]"
    ],
    [
     "C591",
     "x_vars[S24,C17]"
    ],
    [
     "C592",
     "x_vars[S24,C18]"
    ],
    [
     "C593",
     "x_vars[S24,C19]"
    ],
    [
     "C594",
     "x_vars[S24,C20]"
    ],
    [
     "C595",
     "x_vars[S24,C21]"
    ],
    [
     "C596",
     "x_vars[S24,C22]"
    ],
    [
     "C597",
     "x_vars[S24,C23]"
    ],
    [
     "C598",
     "x_vars[S24,C24]"
    ],
    [
     "C599",
     "x_vars[S24,C25]"
    ],
    [
     "C600",
     "y_vars[S1]"
    ],
    [
     "C601",
     "y_vars[S2]"
    ],
    [
     "C602",
     "y_vars[S3]"
    ],
    [
     "C603",
     "y_vars[S4]"
    ],
    [
     "C604",
     "y_vars[S5]"
    ],
    [
     "C605",
     "y_vars[S6]"
    ],
    [
     "C606",
     "y_vars[S7]"
    ],
    [
     "C607",
     "y_vars[S8]"
    ],
    [
     "C608",
     "y_vars[S9]"
    ],
    [
     "C609",
     "y_vars[S10]"
    ],
    [
     "C610",
     "y_vars[S11]"
    ],
    [
     "C611",
     "y_vars[S12]"
    ],
    [
     "C612",
     "y_vars[S13]"
    ],
    [
     "C613",
     "y_vars[S14]"
    ],
    [
     "C614",
     "y_vars[S15]"
    ],
    [
     "C615",
     "y_vars[S16]"
    ],
    [
     "C616",
     "y_vars[S17]"
    ],
    [
     "C617",
     "y_vars[S18]"
    ],
    [
     "C618",
     "y_vars[S19]"
    ],
    [
     "C619",
     "y_vars[S20]"
    ],
    [
     "C620",
     "y_vars[S21]"
    ],
    [
     "C621",
     "y_vars[S22]"
    ],
    [
     "C622",
     "y_vars[S23]"
    ],
    [
     "C623",
     "y_vars[S24]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_0_3": "x_vars[S1,C4]",
    "x_0_4": "x_vars[S1,C5]",
    "x_0_5": "x_vars[S1,C6]",
    "x_0_6": "x_vars[S1,C7]",
    "x_0_7": "x_vars[S1,C8]",
    "x_0_8": "x_vars[S1,C9]",
    "x_0_9": "x_vars[S1,C10]",
    "x_0_10": "x_vars[S1,C11]",
    "x_0_11": "x_vars[S1,C12]",
    "x_0_12": "x_vars[S1,C13]",
    "x_0_13": "x_vars[S1,C14]",
    "x_0_14": "x_vars[S1,C15]",
    "x_0_15": "x_vars[S1,C16]",
    "x_0_16": "x_vars[S1,C17]",
    "x_0_17": "x_vars[S1,C18]",
    "x_0_18": "x_vars[S1,C19]",
    "x_0_19": "x_vars[S1,C20]",
    "x_0_20": "x_vars[S1,C21]",
    "x_0_21": "x_vars[S1,C22]",
    "x_0_22": "x_vars[S1,C23]",
    "x_0_23": "x_vars[S1,C24]",
    "x_0_24": "x_vars[S1,C25]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_1_3": "x_vars[S2,C4]",
    "x_1_4": "x_vars[S2,C5]",
    "x_1_5": "x_vars[S2,C6]",
    "x_1_6": "x_vars[S2,C7]",
    "x_1_7": "x_vars[S2,C8]",
    "x_1_8": "x_vars[S2,C9]",
    "x_1_9": "x_vars[S2,C10]",
    "x_1_10": "x_vars[S2,C11]",
    "x_1_11": "x_vars[S2,C12]",
    "x_1_12": "x_vars[S2,C13]",
    "x_1_13": "x_vars[S2,C14]",
    "x_1_14": "x_vars[S2,C15]",
    "x_1_15": "x_vars[S2,C16]",
    "x_1_16": "x_vars[S2,C17]",
    "x_1_17": "x_vars[S2,C18]",
    "x_1_18": "x_vars[S2,C19]",
    "x_1_19": "x_vars[S2,C20]",
    "x_1_21": "x_vars[S2,C22]",
    "x_1_22": "x_vars[S2,C23]",
    "x_1_23": "x_vars[S2,C24]",
    "x_1_24": "x_vars[S2,C25]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "x_2_3": "x_vars[S3,C4]",
    "x_2_4": "x_vars[S3,C5]",
    "x_2_5": "x_vars[S3,C6]",
    "x_2_6": "x_vars[S3,C7]",
    "x_2_7": "x_vars[S3,C8]",
    "x_2_8": "x_vars[S3,C9]",
    "x_2_9": "x_vars[S3,C10]",
    "x_2_10": "x_vars[S3,C11]",
    "x_2_11": "x_vars[S3,C12]",
    "x_2_12": "x_vars[S3,C13]",
    "x_2_13": "x_vars[S3,C14]",
    "x_2_14": "x_vars[S3,C15]",
    "x_2_15": "x_vars[S3,C16]",
    "x_2_16": "x_vars[S3,C17]",
    "x_2_17": "x_vars[S3,C18]",
    "x_2_18": "x_vars[S3,C19]",
    "x_2_19": "x_vars[S3,C20]",
    "x_2_21": "x_vars[S3,C22]",
    "x_2_22": "x_vars[S3,C23]",
    "x_2_23": "x_vars[S3,C24]",
    "x_2_24": "x_vars[S3,C25]",
    "x_3_0": "x_vars[S4,C1]",
    "x_3_1": "x_vars[S4,C2]",
    "x_3_2": "x_vars[S4,C3]",
    "x_3_3": "x_vars[S4,C4]",
    "x_3_4": "x_vars[S4,C5]",
    "x_3_5": "x_vars[S4,C6]",
    "x_3_6": "x_vars[S4,C7]",
    "x_3_7": "x_vars[S4,C8]",
    "x_3_8": "x_vars[S4,C9]",
    "x_3_9": "x_vars[S4,C10]",
    "x_3_10": "x_vars[S4,C11]",
    "x_3_11": "x_vars[S4,C12]",
    "x_3_12": "x_vars[S4,C13]",
    "x_3_13": "x_vars[S4,C14]",
    "x_3_14": "x_vars[S4,C15]",
    "x_3_15": "x_vars[S4,C16]",
    "x_3_16": "x_vars[S4,C17]",
    "x_3_17": "x_vars[S4,C18]",
    "x_3_18": "x_vars[S4,C19]",
    "x_3_19": "x_vars[S4,C20]",
    "x_3_20": "x_vars[S4,C21]",
    "x_3_21": "x_vars[S4,C22]",
    "x_3_22": "x_vars[S4,C23]",
    "x_3_23": "x_vars[S4,C24]",
    "x_3_24": "x_vars[S4,C25]",
    "x_4_0": "x_vars[S5,C1]",
    "x_4_1": "x_vars[S5,C2]",
    "x_4_2": "x_vars[S5,C3]",
    "x_4_3": "x_vars[S5,C4]",
    "x_4_4": "x_vars[S5,C5]",
    "x_4_5": "x_vars[S5,C6]",
    "x_4_6": "x_vars[S5,C7]",
    "x_4_7": "x_vars[S5,C8]",
    "x_4_8": "x_vars[S5,C9]",
    "x_4_9": "x_vars[S5,C10]",
    "x_4_10": "x_vars[S5,C11]",
    "x_4_11": "x_vars[S5,C12]",
    "x_4_12": "x_vars[S5,C13]",
    "x_4_13": "x_vars[S5,C14]",
    "x_4_14": "x_vars[S5,C15]",
    "x_4_15": "x_vars[S5,C16]",
    "x_4_16": "x_vars[S5,C17]",
    "x_4_17": "x_vars[S5,C18]",
    "x_4_18": "x_vars[S5,C19]",
    "x_4_19": "x_vars[S5,C20]",
    "x_4_20": "x_vars[S5,C21]",
    "x_4_21": "x_vars[S5,C22]",
    "x_4_22": "x_vars[S5,C23]",
    "x_4_23": "x_vars[S5,C24]",
    "x_4_24": "x_vars[S5,C25]",
    "x_5_0": "x_vars[S6,C1]",
    "x_5_1": "x_vars[S6,C2]",
    "x_5_2": "x_vars[S6,C3]",
    "x_5_3": "x_vars[S6,C4]",
    "x_5_4": "x_vars[S6,C5]",
    "x_5_5": "x_vars[S6,C6]",
    "x_5_6": "x_vars[S6,C7]",
    "x_5_7": "x_vars[S6,C8]",
    "x_5_8": "x_vars[S6,C9]",
    "x_5_9": "x_vars[S6,C10]",
    "x_5_10": "x_vars[S6,C11]",
    "x_5_11": "x_vars[S6,C12]",
    "x_5_12": "x_vars[S6,C13]",
    "x_5_13": "x_vars[S6,C14]",
    "x_5_14": "x_vars[S6,C15]",
    "x_5_15": "x_vars[S6,C16]",
    "x_5_16": "x_vars[S6,C17]",
    "x_5_17": "x_vars[S6,C18]",
    "x_5_18": "x_vars[S6,C19]",
    "x_5_19": "x_vars[S6,C20]",
    "x_5_20": "x_vars[S6,C21]",
    "x_5_21": "x_vars[S6,C22]",
    "x_5_22": "x_vars[S6,C23]",
    "x_5_23": "x_vars[S6,C24]",
    "x_5_24": "x_vars[S6,C25]",
    "x_6_0": "x_vars[S7,C1]",
    "x_6_1": "x_vars[S7,C2]",
    "x_6_2": "x_vars[S7,C3]",
    "x_6_3": "x_vars[S7,C4]",
    "x_6_4": "x_vars[S7,C5]",
    "x_6_5": "x_vars[S7,C6]",
    "x_6_6": "x_vars[S7,C7]",
    "x_6_7": "x_vars[S7,C8]",
    "x_6_9": "x_vars[S7,C10]",
    "x_6_10": "x_vars[S7,C11]",
    "x_6_11": "x_vars[S7,C12]",
    "x_6_12": "x_vars[S7,C13]",
    "x_6_13": "x_vars[S7,C14]",
    "x_6_14": "x_vars[S7,C15]",
    "x_6_15": "x_vars[S7,C16]",
    "x_6_16": "x_vars[S7,C17]",
    "x_6_17": "x_vars[S7,C18]",
    "x_6_18": "x_vars[S7,C19]",
    "x_6_19": "x_vars[S7,C20]",
    "x_6_20": "x_vars[S7,C21]",
    "x_6_21": "x_vars[S7,C22]",
    "x_6_22": "x_vars[S7,C23]",
    "x_6_23": "x_vars[S7,C24]",
    "x_6_24": "x_vars[S7,C25]",
    "x_7_0": "x_vars[S8,C1]",
    "x_7_1": "x_vars[S8,C2]",
    "x_7_2": "x_vars[S8,C3]",
    "x_7_3": "x_vars[S8,C4]",
    "x_7_4": "x_vars[S8,C5]",
    "x_7_5": "x_vars[S8,C6]",
    "x_7_6": "x_vars[S8,C7]",
    "x_7_7": "x_vars[S8,C8]",
    "x_7_8": "x_vars[S8,C9]",
    "x_7_9": "x_vars[S8,C10]",
    "x_7_10": "x_vars[S8,C11]",
    "x_7_11": "x_vars[S8,C12]",
    "x_7_12": "x_vars[S8,C13]",
    "x_7_13": "x_vars[S8,C14]",
    "x_7_14": "x_vars[S8,C15]",
    "x_7_15": "x_vars[S8,C16]",
    "x_7_16": "x_vars[S8,C17]",
    "x_7_17": "x_vars[S8,C18]",
    "x_7_18": "x_vars[S8,C19]",
    "x_7_19": "x_vars[S8,C20]",
    "x_7_21": "x_vars[S8,C22]",
    "x_7_22": "x_vars[S8,C23]",
    "x_7_23": "x_vars[S8,C24]",
    "x_7_24": "x_vars[S8,C25]",
    "x_8_0": "x_vars[S9,C1]",
    "x_8_1": "x_vars[S9,C2]",
    "x_8_2": "x_vars[S9,C3]",
    "x_8_3": "x_vars[S9,C4]",
    "x_8_4": "x_vars[S9,C5]",
    "x_8_5": "x_vars[S9,C6]",
    "x_8_6": "x_vars[S9,C7]",
    "x_8_7": "x_vars[S9,C8]",
    "x_8_8": "x_vars[S9,C9]",
    "x_8_9": "x_vars[S9,C10]",
    "x_8_10": "x_vars[S9,C11]",
    "x_8_11": "x_vars[S9,C12]",
    "x_8_12": "x_vars[S9,C13]",
    "x_8_13": "x_vars[S9,C14]",
    "x_8_14": "x_vars[S9,C15]",
    "x_8_15": "x_vars[S9,C16]",
    "x_8_16": "x_vars[S9,C17]",
    "x_8_17": "x_vars[S9,C18]",
    "x_8_18": "x_vars[S9,C19]",
    "x_8_19": "x_vars[S9,C20]",
    "x_8_20": "x_vars[S9,C21]",
    "x_8_21": "x_vars[S9,C22]",
    "x_8_22": "x_vars[S9,C23]",
    "x_8_23": "x_vars[S9,C24]",
    "x_8_24": "x_vars[S9,C25]",
    "x_9_0": "x_vars[S10,C1]",
    "x_9_1": "x_vars[S10,C2]",
    "x_9_2": "x_vars[S10,C3]",
    "x_9_3": "x_vars[S10,C4]",
    "x_9_4": "x_vars[S10,C5]",
    "x_9_5": "x_vars[S10,C6]",
    "x_9_6": "x_vars[S10,C7]",
    "x_9_7": "x_vars[S10,C8]",
    "x_9_8": "x_vars[S10,C9]",
    "x_9_9": "x_vars[S10,C10]",
    "x_9_10": "x_vars[S10,C11]",
    "x_9_11": "x_vars[S10,C12]",
    "x_9_12": "x_vars[S10,C13]",
    "x_9_13": "x_vars[S10,C14]",
    "x_9_14": "x_vars[S10,C15]",
    "x_9_15": "x_vars[S10,C16]",
    "x_9_16": "x_vars[S10,C17]",
    "x_9_17": "x_vars[S10,C18]",
    "x_9_18": "x_vars[S10,C19]",
    "x_9_19": "x_vars[S10,C20]",
    "x_9_20": "x_vars[S10,C21]",
    "x_9_21": "x_vars[S10,C22]",
    "x_9_22": "x_vars[S10,C23]",
    "x_9_23": "x_vars[S10,C24]",
    "x_9_24": "x_vars[S10,C25]",
    "x_10_0": "x_vars[S11,C1]",
    "x_10_2": "x_vars[S11,C3]",
    "x_10_3": "x_vars[S11,C4]",
    "x_10_4": "x_vars[S11,C5]",
    "x_10_5": "x_vars[S11,C6]",
    "x_10_6": "x_vars[S11,C7]",
    "x_10_7": "x_vars[S11,C8]",
    "x_10_8": "x_vars[S11,C9]",
    "x_10_9": "x_vars[S11,C10]",
    "x_10_10": "x_vars[S11,C11]",
    "x_10_11": "x_vars[S11,C12]",
    "x_10_12": "x_vars[S11,C13]",
    "x_10_13": "x_vars[S11,C14]",
    "x_10_14": "x_vars[S11,C15]",
    "x_10_15": "x_vars[S11,C16]",
    "x_10_16": "x_vars[S11,C17]",
    "x_10_17": "x_vars[S11,C18]",
    "x_10_18": "x_vars[S11,C19]",
    "x_10_19": "x_vars[S11,C20]",
    "x_10_20": "x_vars[S11,C21]",
    "x_10_21": "x_vars[S11,C22]",
    "x_10_22": "x_vars[S11,C23]",
    "x_10_23": "x_vars[S11,C24]",
    "x_10_24": "x_vars[S11,C25]",
    "x_11_0": "x_vars[S12,C1]",
    "x_11_1": "x_vars[S12,C2]",
    "x_11_2": "x_vars[S12,C3]",
    "x_11_3": "x_vars[S12,C4]",
    "x_11_4": "x_vars[S12,C5]",
    "x_11_5": "x_vars[S12,C6]",
    "x_11_6": "x_vars[S12,C7]",
    "x_11_7": "x_vars[S12,C8]",
    "x_11_8": "x_vars[S12,C9]",
    "x_11_9": "x_vars[S12,C10]",
    "x_11_10": "x_vars[S12,C11]",
    "x_11_11": "x_vars[S12,C12]",
    "x_11_12": "x_vars[S12,C13]",
    "x_11_13": "x_vars[S12,C14]",
    "x_11_14": "x_vars[S12,C15]",
    "x_11_15": "x_vars[S12,C16]",
    "x_11_16": "x_vars[S12,C17]",
    "x_11_17": "x_vars[S12,C18]",
    "x_11_18": "x_vars[S12,C19]",
    "x_11_19": "x_vars[S12,C20]",
    "x_11_20": "x_vars[S12,C21]",
    "x_11_21": "x_vars[S12,C22]",
    "x_11_22": "x_vars[S12,C23]",
    "x_11_23": "x_vars[S12,C24]",
    "x_11_24": "x_vars[S12,C25]",
    "x_12_0": "x_vars[S13,C1]",
    "x_12_1": "x_vars[S13,C2]",
    "x_12_2": "x_vars[S13,C3]",
    "x_12_3": "x_vars[S13,C4]",
    "x_12_4": "x_vars[S13,C5]",
    "x_12_5": "x_vars[S13,C6]",
    "x_12_6": "x_vars[S13,C7]",
    "x_12_7": "x_vars[S13,C8]",
    "x_12_8": "x_vars[S13,C9]",
    "x_12_9": "x_vars[S13,C10]",
    "x_12_10": "x_vars[S13,C11]",
    "x_12_11": "x_vars[S13,C12]",
    "x_12_12": "x_vars[S13,C13]",
    "x_12_13": "x_vars[S13,C14]",
    "x_12_14": "x_vars[S13,C15]",
    "x_12_15": "x_vars[S13,C16]",
    "x_12_16": "x_vars[S13,C17]",
    "x_12_17": "x_vars[S13,C18]",
    "x_12_18": "x_vars[S13,C19]",
    "x_12_19": "x_vars[S13,C20]",
    "x_12_20": "x_vars[S13,C21]",
    "x_12_21": "x_vars[S13,C22]",
    "x_12_22": "x_vars[S13,C23]",
    "x_12_23": "x_vars[S13,C24]",
    "x_12_24": "x_vars[S13,C25]",
    "x_13_0": "x_vars[S14,C1]",
    "x_13_1": "x_vars[S14,C2]",
    "x_13_2": "x_vars[S14,C3]",
    "x_13_3": "x_vars[S14,C4]",
    "x_13_4": "x_vars[S14,C5]",
    "x_13_5": "x_vars[S14,C6]",
    "x_13_6": "x_vars[S14,C7]",
    "x_13_7": "x_vars[S14,C8]",
    "x_13_8": "x_vars[S14,C9]",
    "x_13_9": "x_vars[S14,C10]",
    "x_13_10": "x_vars[S14,C11]",
    "x_13_11": "x_vars[S14,C12]",
    "x_13_12": "x_vars[S14,C13]",
    "x_13_13": "x_vars[S14,C14]",
    "x_13_14": "x_vars[S14,C15]",
    "x_13_15": "x_vars[S14,C16]",
    "x_13_16": "x_vars[S14,C17]",
    "x_13_17": "x_vars[S14,C18]",
    "x_13_18": "x_vars[S14,C19]",
    "x_13_19": "x_vars[S14,C20]",
    "x_13_20": "x_vars[S14,C21]",
    "x_13_21": "x_vars[S14,C22]",
    "x_13_22": "x_vars[S14,C23]",
    "x_13_23": "x_vars[S14,C24]",
    "x_13_24": "x_vars[S14,C25]",
    "x_14_0": "x_vars[S15,C1]",
    "x_14_1": "x_vars[S15,C2]",
    "x_14_2": "x_vars[S15,C3]",
    "x_14_3": "x_vars[S15,C4]",
    "x_14_4": "x_vars[S15,C5]",
    "x_14_5": "x_vars[S15,C6]",
    "x_14_6": "x_vars[S15,C7]",
    "x_14_7": "x_vars[S15,C8]",
    "x_14_8": "x_vars[S15,C9]",
    "x_14_9": "x_vars[S15,C10]",
    "x_14_10": "x_vars[S15,C11]",
    "x_14_11": "x_vars[S15,C12]",
    "x_14_12": "x_vars[S15,C13]",
    "x_14_13": "x_vars[S15,C14]",
    "x_14_14": "x_vars[S15,C15]",
    "x_14_15": "x_vars[S15,C16]",
    "x_14_16": "x_vars[S15,C17]",
    "x_14_17": "x_vars[S15,C18]",
    "x_14_18": "x_vars[S15,C19]",
    "x_14_19": "x_vars[S15,C20]",
    "x_14_20": "x_vars[S15,C21]",
    "x_14_21": "x_vars[S15,C22]",
    "x_14_22": "x_vars[S15,C23]",
    "x_14_23": "x_vars[S15,C24]",
    "x_14_24": "x_vars[S15,C25]",
    "x_15_0": "x_vars[S16,C1]",
    "x_15_1": "x_vars[S16,C2]",
    "x_15_2": "x_vars[S16,C3]",
    "x_15_3": "x_vars[S16,C4]",
    "x_15_4": "x_vars[S16,C5]",
    "x_15_5": "x_vars[S16,C6]",
    "x_15_6": "x_vars[S16,C7]",
    "x_15_7": "x_vars[S16,C8]",
    "x_15_8": "x_vars[S16,C9]",
    "x_15_9": "x_vars[S16,C10]",
    "x_15_10": "x_vars[S16,C11]",
    "x_15_11": "x_vars[S16,C12]",
    "x_15_12": "x_vars[S16,C13]",
    "x_15_13": "x_vars[S16,C14]",
    "x_15_14": "x_vars[S16,C15]",
    "x_15_15": "x_vars[S16,C16]",
    "x_15_16": "x_vars[S16,C17]",
    "x_15_17": "x_vars[S16,C18]",
    "x_15_18": "x_vars[S16,C19]",
    "x_15_19": "x_vars[S16,C20]",
    "x_15_20": "x_vars[S16,C21]",
    "x_15_21": "x_vars[S16,C22]",
    "x_15_22": "x_vars[S16,C23]",
    "x_15_23": "x_vars[S16,C24]",
    "x_15_24": "x_vars[S16,C25]",
    "x_16_0": "x_vars[S17,C1]",
    "x_16_1": "x_vars[S17,C2]",
    "x_16_2": "x_vars[S17,C3]",
    "x_16_3": "x_vars[S17,C4]",
    "x_16_4": "x_vars[S17,C5]",
    "x_16_5": "x_vars[S17,C6]",
    "x_16_6": "x_vars[S17,C7]",
    "x_16_7": "x_vars[S17,C8]",
    "x_16_8": "x_vars[S17,C9]",
    "x_16_9": "x_vars[S17,C10]",
    "x_16_10": "x_vars[S17,C11]",
    "x_16_11": "x_vars[S17,C12]",
    "x_16_12": "x_vars[S17,C13]",
    "x_16_13": "x_vars[S17,C14]",
    "x_16_14": "x_vars[S17,C15]",
    "x_16_15": "x_vars[S17,C16]",
    "x_16_16": "x_vars[S17,C17]",
    "x_16_17": "x_vars[S17,C18]",
    "x_16_18": "x_vars[S17,C19]",
    "x_16_19": "x_vars[S17,C20]",
    "x_16_20": "x_vars[S17,C21]",
    "x_16_21": "x_vars[S17,C22]",
    "x_16_22": "x_vars[S17,C23]",
    "x_16_23": "x_vars[S17,C24]",
    "x_16_24": "x_vars[S17,C25]",
    "x_17_0": "x_vars[S18,C1]",
    "x_17_1": "x_vars[S18,C2]",
    "x_17_2": "x_vars[S18,C3]",
    "x_17_3": "x_vars[S18,C4]",
    "x_17_4": "x_vars[S18,C5]",
    "x_17_5": "x_vars[S18,C6]",
    "x_17_6": "x_vars[S18,C7]",
    "x_17_7": "x_vars[S18,C8]",
    "x_17_8": "x_vars[S18,C9]",
    "x_17_9": "x_vars[S18,C10]",
    "x_17_10": "x_vars[S18,C11]",
    "x_17_11": "x_vars[S18,C12]",
    "x_17_12": "x_vars[S18,C13]",
    "x_17_13": "x_vars[S18,C14]",
    "x_17_14": "x_vars[S18,C15]",
    "x_17_15": "x_vars[S18,C16]",
    "x_17_16": "x_vars[S18,C17]",
    "x_17_17": "x_vars[S18,C18]",
    "x_17_18": "x_vars[S18,C19]",
    "x_17_19": "x_vars[S18,C20]",
    "x_17_20": "x_vars[S18,C21]",
    "x_17_21": "x_vars[S18,C22]",
    "x_17_22": "x_vars[S18,C23]",
    "x_17_23": "x_vars[S18,C24]",
    "x_17_24": "x_vars[S18,C25]",
    "x_18_0": "x_vars[S19,C1]",
    "x_18_1": "x_vars[S19,C2]",
    "x_18_2": "x_vars[S19,C3]",
    "x_18_3": "x_vars[S19,C4]",
    "x_18_4": "x_vars[S19,C5]",
    "x_18_5": "x_vars[S19,C6]",
    "x_18_6": "x_vars[S19,C7]",
    "x_18_7": "x_vars[S19,C8]",
    "x_18_8": "x_vars[S19,C9]",
    "x_18_9": "x_vars[S19,C10]",
    "x_18_10": "x_vars[S19,C11]",
    "x_18_11": "x_vars[S19,C12]",
    "x_18_12": "x_vars[S19,C13]",
    "x_18_13": "x_vars[S19,C14]",
    "x_18_14": "x_vars[S19,C15]",
    "x_18_15": "x_vars[S19,C16]",
    "x_18_16": "x_vars[S19,C17]",
    "x_18_17": "x_vars[S19,C18]",
    "x_18_18": "x_vars[S19,C19]",
    "x_18_19": "x_vars[S19,C20]",
    "x_18_20": "x_vars[S19,C21]",
    "x_18_21": "x_vars[S19,C22]",
    "x_18_22": "x_vars[S19,C23]",
    "x_18_23": "x_vars[S19,C24]",
    "x_18_24": "x_vars[S19,C25]",
    "x_19_0": "x_vars[S20,C1]",
    "x_19_1": "x_vars[S20,C2]",
    "x_19_2": "x_vars[S20,C3]",
    "x_19_3": "x_vars[S20,C4]",
    "x_19_4": "x_vars[S20,C5]",
    "x_19_5": "x_vars[S20,C6]",
    "x_19_6": "x_vars[S20,C7]",
    "x_19_7": "x_vars[S20,C8]",
    "x_19_8": "x_vars[S20,C9]",
    "x_19_9": "x_vars[S20,C10]",
    "x_19_10": "x_vars[S20,C11]",
    "x_19_11": "x_vars[S20,C12]",
    "x_19_12": "x_vars[S20,C13]",
    "x_19_13": "x_vars[S20,C14]",
    "x_19_14": "x_vars[S20,C15]",
    "x_19_15": "x_vars[S20,C16]",
    "x_19_16": "x_vars[S20,C17]",
    "x_19_17": "x_vars[S20,C18]",
    "x_19_18": "x_vars[S20,C19]",
    "x_19_19": "x_vars[S20,C20]",
    "x_19_20": "x_vars[S20,C21]",
    "x_19_21": "x_vars[S20,C22]",
    "x_19_22": "x_vars[S20,C23]",
    "x_19_23": "x_vars[S20,C24]",
    "x_19_24": "x_vars[S20,C25]",
    "x_20_0": "x_vars[S21,C1]",
    "x_20_1": "x_vars[S21,C2]",
    "x_20_2": "x_vars[S21,C3]",
    "x_20_3": "x_vars[S21,C4]",
    "x_20_4": "x_vars[S21,C5]",
    "x_20_5": "x_vars[S21,C6]",
    "x_20_6": "x_vars[S21,C7]",
    "x_20_7": "x_vars[S21,C8]",
    "x_20_8": "x_vars[S21,C9]",
    "x_20_9": "x_vars[S21,C10]",
    "x_20_10": "x_vars[S21,C11]",
    "x_20_11": "x_vars[S21,C12]",
    "x_20_12": "x_vars[S21,C13]",
    "x_20_13": "x_vars[S21,C14]",
    "x_20_14": "x_vars[S21,C15]",
    "x_20_15": "x_vars[S21,C16]",
    "x_20_16": "x_vars[S21,C17]",
    "x_20_17": "x_vars[S21,C18]",
    "x_20_18": "x_vars[S21,C19]",
    "x_20_19": "x_vars[S21,C20]",
    "x_20_20": "x_vars[S21,C21]",
    "x_20_21": "x_vars[S21,C22]",
    "x_20_22": "x_vars[S21,C23]",
    "x_20_23": "x_vars[S21,C24]",
    "x_20_24": "x_vars[S21,C25]",
    "x_21_0": "x_vars[S22,C1]",
    "x_21_1": "x_vars[S22,C2]",
    "x_21_2": "x_vars[S22,C3]",
    "x_21_3": "x_vars[S22,C4]",
    "x_21_4": "x_vars[S22,C5]",
    "x_21_5": "x_vars[S22,C6]",
    "x_21_6": "x_vars[S22,C7]",
    "x_21_7": "x_vars[S22,C8]",
    "x_21_8": "x_vars[S22,C9]",
    "x_21_9": "x_vars[S22,C10]",
    "x_21_10": "x_vars[S22,C11]",
    "x_21_11": "x_vars[S22,C12]",
    "x_21_12": "x_vars[S22,C13]",
    "x_21_13": "x_vars[S22,C14]",
    "x_21_14": "x_vars[S22,C15]",
    "x_21_15": "x_vars[S22,C16]",
    "x_21_16": "x_vars[S22,C17]",
    "x_21_17": "x_vars[S22,C18]",
    "x_21_18": "x_vars[S22,C19]",
    "x_21_19": "x_vars[S22,C20]",
    "x_21_20": "x_vars[S22,C21]",
    "x_21_21": "x_vars[S22,C22]",
    "x_21_22": "x_vars[S22,C23]",
    "x_21_23": "x_vars[S22,C24]",
    "x_21_24": "x_vars[S22,C25]",
    "x_22_0": "x_vars[S23,C1]",
    "x_22_1": "x_vars[S23,C2]",
    "x_22_2": "x_vars[S23,C3]",
    "x_22_3": "x_vars[S23,C4]",
    "x_22_4": "x_vars[S23,C5]",
    "x_22_5": "x_vars[S23,C6]",
    "x_22_6": "x_vars[S23,C7]",
    "x_22_7": "x_vars[S23,C8]",
    "x_22_8": "x_vars[S23,C9]",
    "x_22_9": "x_vars[S23,C10]",
    "x_22_10": "x_vars[S23,C11]",
    "x_22_11": "x_vars[S23,C12]",
    "x_22_12": "x_vars[S23,C13]",
    "x_22_13": "x_vars[S23,C14]",
    "x_22_14": "x_vars[S23,C15]",
    "x_22_15": "x_vars[S23,C16]",
    "x_22_16": "x_vars[S23,C17]",
    "x_22_17": "x_vars[S23,C18]",
    "x_22_18": "x_vars[S23,C19]",
    "x_22_19": "x_vars[S23,C20]",
    "x_22_20": "x_vars[S23,C21]",
    "x_22_21": "x_vars[S23,C22]",
    "x_22_22": "x_vars[S23,C23]",
    "x_22_23": "x_vars[S23,C24]",
    "x_22_24": "x_vars[S23,C25]",
    "x_23_0": "x_vars[S24,C1]",
    "x_23_1": "x_vars[S24,C2]",
    "x_23_2": "x_vars[S24,C3]",
    "x_23_3": "x_vars[S24,C4]",
    "x_23_4": "x_vars[S24,C5]",
    "x_23_5": "x_vars[S24,C6]",
    "x_23_6": "x_vars[S24,C7]",
    "x_23_7": "x_vars[S24,C8]",
    "x_23_8": "x_vars[S24,C9]",
    "x_23_9": "x_vars[S24,C10]",
    "x_23_10": "x_vars[S24,C11]",
    "x_23_11": "x_vars[S24,C12]",
    "x_23_12": "x_vars[S24,C13]",
    "x_23_13": "x_vars[S24,C14]",
    "x_23_14": "x_vars[S24,C15]",
    "x_23_15": "x_vars[S24,C16]",
    "x_23_16": "x_vars[S24,C17]",
    "x_23_17": "x_vars[S24,C18]",
    "x_23_18": "x_vars[S24,C19]",
    "x_23_19": "x_vars[S24,C20]",
    "x_23_20": "x_vars[S24,C21]",
    "x_23_21": "x_vars[S24,C22]",
    "x_23_22": "x_vars[S24,C23]",
    "x_23_23": "x_vars[S24,C24]",
    "x_23_24": "x_vars[S24,C25]",
    "x_10_1": "x_vars[S11,C2]",
    "x_6_8": "x_vars[S7,C9]",
    "x_1_20": "x_vars[S2,C21]",
    "x_2_20": "x_vars[S3,C21]",
    "x_7_20": "x_vars[S8,C21]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]",
    "y_2": "y_vars[S3]",
    "y_3": "y_vars[S4]",
    "y_4": "y_vars[S5]",
    "y_5": "y_vars[S6]",
    "y_6": "y_vars[S7]",
    "y_7": "y_vars[S8]",
    "y_8": "y_vars[S9]",
    "y_9": "y_vars[S10]",
    "y_10": "y_vars[S11]",
    "y_11": "y_vars[S12]",
    "y_12": "y_vars[S13]",
    "y_13": "y_vars[S14]",
    "y_14": "y_vars[S15]",
    "y_15": "y_vars[S16]",
    "y_16": "y_vars[S17]",
    "y_17": "y_vars[S18]",
    "y_18": "y_vars[S19]",
    "y_19": "y_vars[S20]",
    "y_20": "y_vars[S21]",
    "y_21": "y_vars[S22]",
    "y_22": "y_vars[S23]",
    "y_23": "y_vars[S24]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-065": {
   "source_code_sha256": "fc7cba4f5e9e1e3434ff092bf4d9a5db0fce297efef4020cd0e7c9f0d1294b18",
   "original_solution_sha256": "9f10625464ea5be243cf19fd8acc413397e7ca0cdeb26a47524b048d53db8c12",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S1,C3]"
    ],
    [
     "C3",
     "x_vars[S2,C1]"
    ],
    [
     "C4",
     "x_vars[S2,C2]"
    ],
    [
     "C5",
     "x_vars[S2,C3]"
    ],
    [
     "C6",
     "x_vars[S3,C1]"
    ],
    [
     "C7",
     "x_vars[S3,C2]"
    ],
    [
     "C8",
     "x_vars[S3,C3]"
    ],
    [
     "C9",
     "y_vars[S1]"
    ],
    [
     "C10",
     "y_vars[S2]"
    ],
    [
     "C11",
     "y_vars[S3]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_0_2": "x_vars[S1,C3]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "x_1_2": "x_vars[S2,C3]",
    "x_2_0": "x_vars[S3,C1]",
    "x_2_1": "x_vars[S3,C2]",
    "x_2_2": "x_vars[S3,C3]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]",
    "y_2": "y_vars[S3]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-066": {
   "source_code_sha256": "575ab02c9b2815b76b695a6992f132543dc85a77d69ad147eb66d9a079e36653",
   "original_solution_sha256": "feac0468c67fc04e0149f83bb4701b7245d0e0629e4616625b44ec9051a5abcb",
   "bindings": [
    [
     "C0",
     "x_vars[S1,C1]"
    ],
    [
     "C1",
     "x_vars[S1,C2]"
    ],
    [
     "C2",
     "x_vars[S2,C1]"
    ],
    [
     "C3",
     "x_vars[S2,C2]"
    ],
    [
     "C4",
     "y_vars[S1]"
    ],
    [
     "C5",
     "y_vars[S2]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[S1,C1]",
    "x_0_1": "x_vars[S1,C2]",
    "x_1_0": "x_vars[S2,C1]",
    "x_1_1": "x_vars[S2,C2]",
    "y_0": "y_vars[S1]",
    "y_1": "y_vars[S2]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-067": {
   "source_code_sha256": "8c17d2ef67a16f9b4fb59e40ed016ff61f77256b850e5ff4352e560a9b35d0a0",
   "original_solution_sha256": "80b7db5f84ce9becda4807c8f2c70bc69aab9561f86bc8bec9e138f6dbae405f",
   "bindings": [
    [
     "C0",
     "x_vars[MA,P1]"
    ],
    [
     "C1",
     "x_vars[MA,P2]"
    ],
    [
     "C2",
     "x_vars[MA,P3]"
    ],
    [
     "C3",
     "x_vars[MB,P1]"
    ],
    [
     "C4",
     "x_vars[MB,P2]"
    ],
    [
     "C5",
     "x_vars[MB,P3]"
    ],
    [
     "C6",
     "x_vars[MC,P1]"
    ],
    [
     "C7",
     "x_vars[MC,P2]"
    ],
    [
     "C8",
     "x_vars[MC,P3]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[MA,P1]",
    "x_0_1": "x_vars[MA,P2]",
    "x_0_2": "x_vars[MA,P3]",
    "x_1_0": "x_vars[MB,P1]",
    "x_1_1": "x_vars[MB,P2]",
    "x_1_2": "x_vars[MB,P3]",
    "x_2_0": "x_vars[MC,P1]",
    "x_2_1": "x_vars[MC,P2]",
    "x_2_2": "x_vars[MC,P3]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-068": {
   "source_code_sha256": "125f8748df7cea328edae247f906aae879208d1aaba598a51392d2f0a8b9f195",
   "original_solution_sha256": "42bd7e848e91f292b559ceffbe2a5b282b53743efada860871afd630b4ef5fa3",
   "bindings": [
    [
     "C0",
     "x_vars[MA,P1]"
    ],
    [
     "C1",
     "x_vars[MA,P2]"
    ],
    [
     "C2",
     "x_vars[MA,P3]"
    ],
    [
     "C3",
     "x_vars[MA,P4]"
    ],
    [
     "C4",
     "x_vars[MA,P5]"
    ],
    [
     "C5",
     "x_vars[MA,P6]"
    ],
    [
     "C6",
     "x_vars[MB,P1]"
    ],
    [
     "C7",
     "x_vars[MB,P2]"
    ],
    [
     "C8",
     "x_vars[MB,P3]"
    ],
    [
     "C9",
     "x_vars[MB,P4]"
    ],
    [
     "C10",
     "x_vars[MB,P5]"
    ],
    [
     "C11",
     "x_vars[MB,P6]"
    ],
    [
     "C12",
     "x_vars[MC,P1]"
    ],
    [
     "C13",
     "x_vars[MC,P2]"
    ],
    [
     "C14",
     "x_vars[MC,P3]"
    ],
    [
     "C15",
     "x_vars[MC,P4]"
    ],
    [
     "C16",
     "x_vars[MC,P5]"
    ],
    [
     "C17",
     "x_vars[MC,P6]"
    ],
    [
     "C18",
     "x_vars[MD,P1]"
    ],
    [
     "C19",
     "x_vars[MD,P2]"
    ],
    [
     "C20",
     "x_vars[MD,P3]"
    ],
    [
     "C21",
     "x_vars[MD,P4]"
    ],
    [
     "C22",
     "x_vars[MD,P5]"
    ],
    [
     "C23",
     "x_vars[MD,P6]"
    ],
    [
     "C24",
     "x_vars[ME,P1]"
    ],
    [
     "C25",
     "x_vars[ME,P2]"
    ],
    [
     "C26",
     "x_vars[ME,P3]"
    ],
    [
     "C27",
     "x_vars[ME,P4]"
    ],
    [
     "C28",
     "x_vars[ME,P5]"
    ],
    [
     "C29",
     "x_vars[ME,P6]"
    ],
    [
     "C30",
     "x_vars[MF,P1]"
    ],
    [
     "C31",
     "x_vars[MF,P2]"
    ],
    [
     "C32",
     "x_vars[MF,P3]"
    ],
    [
     "C33",
     "x_vars[MF,P4]"
    ],
    [
     "C34",
     "x_vars[MF,P5]"
    ],
    [
     "C35",
     "x_vars[MF,P6]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[MA,P1]",
    "x_0_1": "x_vars[MA,P2]",
    "x_0_2": "x_vars[MA,P3]",
    "x_0_3": "x_vars[MA,P4]",
    "x_0_4": "x_vars[MA,P5]",
    "x_0_5": "x_vars[MA,P6]",
    "x_1_0": "x_vars[MB,P1]",
    "x_1_1": "x_vars[MB,P2]",
    "x_1_2": "x_vars[MB,P3]",
    "x_1_3": "x_vars[MB,P4]",
    "x_1_4": "x_vars[MB,P5]",
    "x_1_5": "x_vars[MB,P6]",
    "x_2_0": "x_vars[MC,P1]",
    "x_2_1": "x_vars[MC,P2]",
    "x_2_2": "x_vars[MC,P3]",
    "x_2_3": "x_vars[MC,P4]",
    "x_2_4": "x_vars[MC,P5]",
    "x_2_5": "x_vars[MC,P6]",
    "x_3_0": "x_vars[MD,P1]",
    "x_3_1": "x_vars[MD,P2]",
    "x_3_2": "x_vars[MD,P3]",
    "x_3_3": "x_vars[MD,P4]",
    "x_3_4": "x_vars[MD,P5]",
    "x_3_5": "x_vars[MD,P6]",
    "x_4_0": "x_vars[ME,P1]",
    "x_4_1": "x_vars[ME,P2]",
    "x_4_2": "x_vars[ME,P3]",
    "x_4_3": "x_vars[ME,P4]",
    "x_4_4": "x_vars[ME,P5]",
    "x_4_5": "x_vars[ME,P6]",
    "x_5_0": "x_vars[MF,P1]",
    "x_5_1": "x_vars[MF,P2]",
    "x_5_2": "x_vars[MF,P3]",
    "x_5_3": "x_vars[MF,P4]",
    "x_5_4": "x_vars[MF,P5]",
    "x_5_5": "x_vars[MF,P6]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-069": {
   "source_code_sha256": "d2d0010533cdb2628c59062429b5b71928fec25cae81109271977a846d83367a",
   "original_solution_sha256": "167914019b6a244d702ecd4d123caf4300e526dfc1991d0e0048c5d4fcc580b7",
   "bindings": [
    [
     "C0",
     "x_vars[Manager 1,Project 1]"
    ],
    [
     "C1",
     "x_vars[Manager 1,Project 2]"
    ],
    [
     "C2",
     "x_vars[Manager 1,Project 3]"
    ],
    [
     "C3",
     "x_vars[Manager 1,Project 4]"
    ],
    [
     "C4",
     "x_vars[Manager 1,Project 5]"
    ],
    [
     "C5",
     "x_vars[Manager 1,Project 6]"
    ],
    [
     "C6",
     "x_vars[Manager 1,Project 7]"
    ],
    [
     "C7",
     "x_vars[Manager 1,Project 8]"
    ],
    [
     "C8",
     "x_vars[Manager 1,Project 9]"
    ],
    [
     "C9",
     "x_vars[Manager 1,Project 10]"
    ],
    [
     "C10",
     "x_vars[Manager 1,Project 11]"
    ],
    [
     "C11",
     "x_vars[Manager 2,Project 1]"
    ],
    [
     "C12",
     "x_vars[Manager 2,Project 2]"
    ],
    [
     "C13",
     "x_vars[Manager 2,Project 3]"
    ],
    [
     "C14",
     "x_vars[Manager 2,Project 4]"
    ],
    [
     "C15",
     "x_vars[Manager 2,Project 5]"
    ],
    [
     "C16",
     "x_vars[Manager 2,Project 6]"
    ],
    [
     "C17",
     "x_vars[Manager 2,Project 7]"
    ],
    [
     "C18",
     "x_vars[Manager 2,Project 8]"
    ],
    [
     "C19",
     "x_vars[Manager 2,Project 9]"
    ],
    [
     "C20",
     "x_vars[Manager 2,Project 10]"
    ],
    [
     "C21",
     "x_vars[Manager 2,Project 11]"
    ],
    [
     "C22",
     "x_vars[Manager 3,Project 1]"
    ],
    [
     "C23",
     "x_vars[Manager 3,Project 2]"
    ],
    [
     "C24",
     "x_vars[Manager 3,Project 3]"
    ],
    [
     "C25",
     "x_vars[Manager 3,Project 4]"
    ],
    [
     "C26",
     "x_vars[Manager 3,Project 5]"
    ],
    [
     "C27",
     "x_vars[Manager 3,Project 6]"
    ],
    [
     "C28",
     "x_vars[Manager 3,Project 7]"
    ],
    [
     "C29",
     "x_vars[Manager 3,Project 8]"
    ],
    [
     "C30",
     "x_vars[Manager 3,Project 9]"
    ],
    [
     "C31",
     "x_vars[Manager 3,Project 10]"
    ],
    [
     "C32",
     "x_vars[Manager 3,Project 11]"
    ],
    [
     "C33",
     "x_vars[Manager 4,Project 1]"
    ],
    [
     "C34",
     "x_vars[Manager 4,Project 2]"
    ],
    [
     "C35",
     "x_vars[Manager 4,Project 3]"
    ],
    [
     "C36",
     "x_vars[Manager 4,Project 4]"
    ],
    [
     "C37",
     "x_vars[Manager 4,Project 5]"
    ],
    [
     "C38",
     "x_vars[Manager 4,Project 6]"
    ],
    [
     "C39",
     "x_vars[Manager 4,Project 7]"
    ],
    [
     "C40",
     "x_vars[Manager 4,Project 8]"
    ],
    [
     "C41",
     "x_vars[Manager 4,Project 9]"
    ],
    [
     "C42",
     "x_vars[Manager 4,Project 10]"
    ],
    [
     "C43",
     "x_vars[Manager 4,Project 11]"
    ],
    [
     "C44",
     "x_vars[Manager 5,Project 1]"
    ],
    [
     "C45",
     "x_vars[Manager 5,Project 2]"
    ],
    [
     "C46",
     "x_vars[Manager 5,Project 3]"
    ],
    [
     "C47",
     "x_vars[Manager 5,Project 4]"
    ],
    [
     "C48",
     "x_vars[Manager 5,Project 5]"
    ],
    [
     "C49",
     "x_vars[Manager 5,Project 6]"
    ],
    [
     "C50",
     "x_vars[Manager 5,Project 7]"
    ],
    [
     "C51",
     "x_vars[Manager 5,Project 8]"
    ],
    [
     "C52",
     "x_vars[Manager 5,Project 9]"
    ],
    [
     "C53",
     "x_vars[Manager 5,Project 10]"
    ],
    [
     "C54",
     "x_vars[Manager 5,Project 11]"
    ],
    [
     "C55",
     "x_vars[Manager 6,Project 1]"
    ],
    [
     "C56",
     "x_vars[Manager 6,Project 2]"
    ],
    [
     "C57",
     "x_vars[Manager 6,Project 3]"
    ],
    [
     "C58",
     "x_vars[Manager 6,Project 4]"
    ],
    [
     "C59",
     "x_vars[Manager 6,Project 5]"
    ],
    [
     "C60",
     "x_vars[Manager 6,Project 6]"
    ],
    [
     "C61",
     "x_vars[Manager 6,Project 7]"
    ],
    [
     "C62",
     "x_vars[Manager 6,Project 8]"
    ],
    [
     "C63",
     "x_vars[Manager 6,Project 9]"
    ],
    [
     "C64",
     "x_vars[Manager 6,Project 10]"
    ],
    [
     "C65",
     "x_vars[Manager 6,Project 11]"
    ],
    [
     "C66",
     "x_vars[Manager 7,Project 1]"
    ],
    [
     "C67",
     "x_vars[Manager 7,Project 2]"
    ],
    [
     "C68",
     "x_vars[Manager 7,Project 3]"
    ],
    [
     "C69",
     "x_vars[Manager 7,Project 4]"
    ],
    [
     "C70",
     "x_vars[Manager 7,Project 5]"
    ],
    [
     "C71",
     "x_vars[Manager 7,Project 6]"
    ],
    [
     "C72",
     "x_vars[Manager 7,Project 7]"
    ],
    [
     "C73",
     "x_vars[Manager 7,Project 8]"
    ],
    [
     "C74",
     "x_vars[Manager 7,Project 9]"
    ],
    [
     "C75",
     "x_vars[Manager 7,Project 10]"
    ],
    [
     "C76",
     "x_vars[Manager 7,Project 11]"
    ],
    [
     "C77",
     "x_vars[Manager 8,Project 1]"
    ],
    [
     "C78",
     "x_vars[Manager 8,Project 2]"
    ],
    [
     "C79",
     "x_vars[Manager 8,Project 3]"
    ],
    [
     "C80",
     "x_vars[Manager 8,Project 4]"
    ],
    [
     "C81",
     "x_vars[Manager 8,Project 5]"
    ],
    [
     "C82",
     "x_vars[Manager 8,Project 6]"
    ],
    [
     "C83",
     "x_vars[Manager 8,Project 7]"
    ],
    [
     "C84",
     "x_vars[Manager 8,Project 8]"
    ],
    [
     "C85",
     "x_vars[Manager 8,Project 9]"
    ],
    [
     "C86",
     "x_vars[Manager 8,Project 10]"
    ],
    [
     "C87",
     "x_vars[Manager 8,Project 11]"
    ],
    [
     "C88",
     "x_vars[Manager 9,Project 1]"
    ],
    [
     "C89",
     "x_vars[Manager 9,Project 2]"
    ],
    [
     "C90",
     "x_vars[Manager 9,Project 3]"
    ],
    [
     "C91",
     "x_vars[Manager 9,Project 4]"
    ],
    [
     "C92",
     "x_vars[Manager 9,Project 5]"
    ],
    [
     "C93",
     "x_vars[Manager 9,Project 6]"
    ],
    [
     "C94",
     "x_vars[Manager 9,Project 7]"
    ],
    [
     "C95",
     "x_vars[Manager 9,Project 8]"
    ],
    [
     "C96",
     "x_vars[Manager 9,Project 9]"
    ],
    [
     "C97",
     "x_vars[Manager 9,Project 10]"
    ],
    [
     "C98",
     "x_vars[Manager 9,Project 11]"
    ],
    [
     "C99",
     "x_vars[Manager 10,Project 1]"
    ],
    [
     "C100",
     "x_vars[Manager 10,Project 2]"
    ],
    [
     "C101",
     "x_vars[Manager 10,Project 3]"
    ],
    [
     "C102",
     "x_vars[Manager 10,Project 4]"
    ],
    [
     "C103",
     "x_vars[Manager 10,Project 5]"
    ],
    [
     "C104",
     "x_vars[Manager 10,Project 6]"
    ],
    [
     "C105",
     "x_vars[Manager 10,Project 7]"
    ],
    [
     "C106",
     "x_vars[Manager 10,Project 8]"
    ],
    [
     "C107",
     "x_vars[Manager 10,Project 9]"
    ],
    [
     "C108",
     "x_vars[Manager 10,Project 10]"
    ],
    [
     "C109",
     "x_vars[Manager 10,Project 11]"
    ],
    [
     "C110",
     "x_vars[Manager 11,Project 1]"
    ],
    [
     "C111",
     "x_vars[Manager 11,Project 2]"
    ],
    [
     "C112",
     "x_vars[Manager 11,Project 3]"
    ],
    [
     "C113",
     "x_vars[Manager 11,Project 4]"
    ],
    [
     "C114",
     "x_vars[Manager 11,Project 5]"
    ],
    [
     "C115",
     "x_vars[Manager 11,Project 6]"
    ],
    [
     "C116",
     "x_vars[Manager 11,Project 7]"
    ],
    [
     "C117",
     "x_vars[Manager 11,Project 8]"
    ],
    [
     "C118",
     "x_vars[Manager 11,Project 9]"
    ],
    [
     "C119",
     "x_vars[Manager 11,Project 10]"
    ],
    [
     "C120",
     "x_vars[Manager 11,Project 11]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Manager 1,Project 1]",
    "x_0_1": "x_vars[Manager 1,Project 2]",
    "x_0_2": "x_vars[Manager 1,Project 3]",
    "x_0_3": "x_vars[Manager 1,Project 4]",
    "x_0_4": "x_vars[Manager 1,Project 5]",
    "x_0_5": "x_vars[Manager 1,Project 6]",
    "x_0_6": "x_vars[Manager 1,Project 7]",
    "x_0_7": "x_vars[Manager 1,Project 8]",
    "x_0_8": "x_vars[Manager 1,Project 9]",
    "x_0_9": "x_vars[Manager 1,Project 10]",
    "x_0_10": "x_vars[Manager 1,Project 11]",
    "x_1_0": "x_vars[Manager 2,Project 1]",
    "x_1_1": "x_vars[Manager 2,Project 2]",
    "x_1_2": "x_vars[Manager 2,Project 3]",
    "x_1_3": "x_vars[Manager 2,Project 4]",
    "x_1_4": "x_vars[Manager 2,Project 5]",
    "x_1_5": "x_vars[Manager 2,Project 6]",
    "x_1_6": "x_vars[Manager 2,Project 7]",
    "x_1_7": "x_vars[Manager 2,Project 8]",
    "x_1_8": "x_vars[Manager 2,Project 9]",
    "x_1_9": "x_vars[Manager 2,Project 10]",
    "x_1_10": "x_vars[Manager 2,Project 11]",
    "x_2_0": "x_vars[Manager 3,Project 1]",
    "x_2_1": "x_vars[Manager 3,Project 2]",
    "x_2_2": "x_vars[Manager 3,Project 3]",
    "x_2_3": "x_vars[Manager 3,Project 4]",
    "x_2_4": "x_vars[Manager 3,Project 5]",
    "x_2_5": "x_vars[Manager 3,Project 6]",
    "x_2_6": "x_vars[Manager 3,Project 7]",
    "x_2_7": "x_vars[Manager 3,Project 8]",
    "x_2_8": "x_vars[Manager 3,Project 9]",
    "x_2_9": "x_vars[Manager 3,Project 10]",
    "x_2_10": "x_vars[Manager 3,Project 11]",
    "x_3_0": "x_vars[Manager 4,Project 1]",
    "x_3_1": "x_vars[Manager 4,Project 2]",
    "x_3_2": "x_vars[Manager 4,Project 3]",
    "x_3_3": "x_vars[Manager 4,Project 4]",
    "x_3_4": "x_vars[Manager 4,Project 5]",
    "x_3_5": "x_vars[Manager 4,Project 6]",
    "x_3_6": "x_vars[Manager 4,Project 7]",
    "x_3_7": "x_vars[Manager 4,Project 8]",
    "x_3_8": "x_vars[Manager 4,Project 9]",
    "x_3_9": "x_vars[Manager 4,Project 10]",
    "x_3_10": "x_vars[Manager 4,Project 11]",
    "x_4_0": "x_vars[Manager 5,Project 1]",
    "x_4_1": "x_vars[Manager 5,Project 2]",
    "x_4_2": "x_vars[Manager 5,Project 3]",
    "x_4_3": "x_vars[Manager 5,Project 4]",
    "x_4_4": "x_vars[Manager 5,Project 5]",
    "x_4_5": "x_vars[Manager 5,Project 6]",
    "x_4_6": "x_vars[Manager 5,Project 7]",
    "x_4_7": "x_vars[Manager 5,Project 8]",
    "x_4_8": "x_vars[Manager 5,Project 9]",
    "x_4_9": "x_vars[Manager 5,Project 10]",
    "x_4_10": "x_vars[Manager 5,Project 11]",
    "x_5_0": "x_vars[Manager 6,Project 1]",
    "x_5_1": "x_vars[Manager 6,Project 2]",
    "x_5_2": "x_vars[Manager 6,Project 3]",
    "x_5_3": "x_vars[Manager 6,Project 4]",
    "x_5_4": "x_vars[Manager 6,Project 5]",
    "x_5_5": "x_vars[Manager 6,Project 6]",
    "x_5_6": "x_vars[Manager 6,Project 7]",
    "x_5_7": "x_vars[Manager 6,Project 8]",
    "x_5_8": "x_vars[Manager 6,Project 9]",
    "x_5_9": "x_vars[Manager 6,Project 10]",
    "x_5_10": "x_vars[Manager 6,Project 11]",
    "x_6_0": "x_vars[Manager 7,Project 1]",
    "x_6_1": "x_vars[Manager 7,Project 2]",
    "x_6_2": "x_vars[Manager 7,Project 3]",
    "x_6_3": "x_vars[Manager 7,Project 4]",
    "x_6_4": "x_vars[Manager 7,Project 5]",
    "x_6_5": "x_vars[Manager 7,Project 6]",
    "x_6_6": "x_vars[Manager 7,Project 7]",
    "x_6_7": "x_vars[Manager 7,Project 8]",
    "x_6_8": "x_vars[Manager 7,Project 9]",
    "x_6_9": "x_vars[Manager 7,Project 10]",
    "x_6_10": "x_vars[Manager 7,Project 11]",
    "x_7_0": "x_vars[Manager 8,Project 1]",
    "x_7_1": "x_vars[Manager 8,Project 2]",
    "x_7_2": "x_vars[Manager 8,Project 3]",
    "x_7_3": "x_vars[Manager 8,Project 4]",
    "x_7_4": "x_vars[Manager 8,Project 5]",
    "x_7_5": "x_vars[Manager 8,Project 6]",
    "x_7_6": "x_vars[Manager 8,Project 7]",
    "x_7_7": "x_vars[Manager 8,Project 8]",
    "x_7_8": "x_vars[Manager 8,Project 9]",
    "x_7_9": "x_vars[Manager 8,Project 10]",
    "x_7_10": "x_vars[Manager 8,Project 11]",
    "x_8_0": "x_vars[Manager 9,Project 1]",
    "x_8_1": "x_vars[Manager 9,Project 2]",
    "x_8_2": "x_vars[Manager 9,Project 3]",
    "x_8_3": "x_vars[Manager 9,Project 4]",
    "x_8_4": "x_vars[Manager 9,Project 5]",
    "x_8_5": "x_vars[Manager 9,Project 6]",
    "x_8_6": "x_vars[Manager 9,Project 7]",
    "x_8_7": "x_vars[Manager 9,Project 8]",
    "x_8_8": "x_vars[Manager 9,Project 9]",
    "x_8_9": "x_vars[Manager 9,Project 10]",
    "x_8_10": "x_vars[Manager 9,Project 11]",
    "x_9_0": "x_vars[Manager 10,Project 1]",
    "x_9_1": "x_vars[Manager 10,Project 2]",
    "x_9_2": "x_vars[Manager 10,Project 3]",
    "x_9_3": "x_vars[Manager 10,Project 4]",
    "x_9_4": "x_vars[Manager 10,Project 5]",
    "x_9_5": "x_vars[Manager 10,Project 6]",
    "x_9_6": "x_vars[Manager 10,Project 7]",
    "x_9_7": "x_vars[Manager 10,Project 8]",
    "x_9_8": "x_vars[Manager 10,Project 9]",
    "x_9_9": "x_vars[Manager 10,Project 10]",
    "x_9_10": "x_vars[Manager 10,Project 11]",
    "x_10_0": "x_vars[Manager 11,Project 1]",
    "x_10_1": "x_vars[Manager 11,Project 2]",
    "x_10_2": "x_vars[Manager 11,Project 3]",
    "x_10_3": "x_vars[Manager 11,Project 4]",
    "x_10_4": "x_vars[Manager 11,Project 5]",
    "x_10_5": "x_vars[Manager 11,Project 6]",
    "x_10_6": "x_vars[Manager 11,Project 7]",
    "x_10_7": "x_vars[Manager 11,Project 8]",
    "x_10_8": "x_vars[Manager 11,Project 9]",
    "x_10_9": "x_vars[Manager 11,Project 10]",
    "x_10_10": "x_vars[Manager 11,Project 11]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-070": {
   "source_code_sha256": "3da613317286d4de7465cc95fc112ae1e67fe4844676efaf2a1c9b1f831a4da7",
   "original_solution_sha256": "810dcd637f4b50d39b4745b5791639fbba8c1b2c3186ab8bb472fc6112e3174d",
   "bindings": [
    [
     "C0",
     "x_vars[Manager 1,Project 1]"
    ],
    [
     "C1",
     "x_vars[Manager 1,Project 2]"
    ],
    [
     "C2",
     "x_vars[Manager 1,Project 3]"
    ],
    [
     "C3",
     "x_vars[Manager 1,Project 4]"
    ],
    [
     "C4",
     "x_vars[Manager 1,Project 5]"
    ],
    [
     "C5",
     "x_vars[Manager 1,Project 6]"
    ],
    [
     "C6",
     "x_vars[Manager 1,Project 7]"
    ],
    [
     "C7",
     "x_vars[Manager 2,Project 1]"
    ],
    [
     "C8",
     "x_vars[Manager 2,Project 2]"
    ],
    [
     "C9",
     "x_vars[Manager 2,Project 3]"
    ],
    [
     "C10",
     "x_vars[Manager 2,Project 4]"
    ],
    [
     "C11",
     "x_vars[Manager 2,Project 5]"
    ],
    [
     "C12",
     "x_vars[Manager 2,Project 6]"
    ],
    [
     "C13",
     "x_vars[Manager 2,Project 7]"
    ],
    [
     "C14",
     "x_vars[Manager 3,Project 1]"
    ],
    [
     "C15",
     "x_vars[Manager 3,Project 2]"
    ],
    [
     "C16",
     "x_vars[Manager 3,Project 3]"
    ],
    [
     "C17",
     "x_vars[Manager 3,Project 4]"
    ],
    [
     "C18",
     "x_vars[Manager 3,Project 5]"
    ],
    [
     "C19",
     "x_vars[Manager 3,Project 6]"
    ],
    [
     "C20",
     "x_vars[Manager 3,Project 7]"
    ],
    [
     "C21",
     "x_vars[Manager 4,Project 1]"
    ],
    [
     "C22",
     "x_vars[Manager 4,Project 2]"
    ],
    [
     "C23",
     "x_vars[Manager 4,Project 3]"
    ],
    [
     "C24",
     "x_vars[Manager 4,Project 4]"
    ],
    [
     "C25",
     "x_vars[Manager 4,Project 5]"
    ],
    [
     "C26",
     "x_vars[Manager 4,Project 6]"
    ],
    [
     "C27",
     "x_vars[Manager 4,Project 7]"
    ],
    [
     "C28",
     "x_vars[Manager 5,Project 1]"
    ],
    [
     "C29",
     "x_vars[Manager 5,Project 2]"
    ],
    [
     "C30",
     "x_vars[Manager 5,Project 3]"
    ],
    [
     "C31",
     "x_vars[Manager 5,Project 4]"
    ],
    [
     "C32",
     "x_vars[Manager 5,Project 5]"
    ],
    [
     "C33",
     "x_vars[Manager 5,Project 6]"
    ],
    [
     "C34",
     "x_vars[Manager 5,Project 7]"
    ],
    [
     "C35",
     "x_vars[Manager 6,Project 1]"
    ],
    [
     "C36",
     "x_vars[Manager 6,Project 2]"
    ],
    [
     "C37",
     "x_vars[Manager 6,Project 3]"
    ],
    [
     "C38",
     "x_vars[Manager 6,Project 4]"
    ],
    [
     "C39",
     "x_vars[Manager 6,Project 5]"
    ],
    [
     "C40",
     "x_vars[Manager 6,Project 6]"
    ],
    [
     "C41",
     "x_vars[Manager 6,Project 7]"
    ],
    [
     "C42",
     "x_vars[Manager 7,Project 1]"
    ],
    [
     "C43",
     "x_vars[Manager 7,Project 2]"
    ],
    [
     "C44",
     "x_vars[Manager 7,Project 3]"
    ],
    [
     "C45",
     "x_vars[Manager 7,Project 4]"
    ],
    [
     "C46",
     "x_vars[Manager 7,Project 5]"
    ],
    [
     "C47",
     "x_vars[Manager 7,Project 6]"
    ],
    [
     "C48",
     "x_vars[Manager 7,Project 7]"
    ]
   ],
   "primary_mapping": {
    "x_0_0": "x_vars[Manager 1,Project 1]",
    "x_0_1": "x_vars[Manager 1,Project 2]",
    "x_0_2": "x_vars[Manager 1,Project 3]",
    "x_0_3": "x_vars[Manager 1,Project 4]",
    "x_0_4": "x_vars[Manager 1,Project 5]",
    "x_0_5": "x_vars[Manager 1,Project 6]",
    "x_0_6": "x_vars[Manager 1,Project 7]",
    "x_1_0": "x_vars[Manager 2,Project 1]",
    "x_1_1": "x_vars[Manager 2,Project 2]",
    "x_1_2": "x_vars[Manager 2,Project 3]",
    "x_1_3": "x_vars[Manager 2,Project 4]",
    "x_1_4": "x_vars[Manager 2,Project 5]",
    "x_1_5": "x_vars[Manager 2,Project 6]",
    "x_1_6": "x_vars[Manager 2,Project 7]",
    "x_2_0": "x_vars[Manager 3,Project 1]",
    "x_2_1": "x_vars[Manager 3,Project 2]",
    "x_2_2": "x_vars[Manager 3,Project 3]",
    "x_2_3": "x_vars[Manager 3,Project 4]",
    "x_2_4": "x_vars[Manager 3,Project 5]",
    "x_2_5": "x_vars[Manager 3,Project 6]",
    "x_2_6": "x_vars[Manager 3,Project 7]",
    "x_3_0": "x_vars[Manager 4,Project 1]",
    "x_3_1": "x_vars[Manager 4,Project 2]",
    "x_3_2": "x_vars[Manager 4,Project 3]",
    "x_3_3": "x_vars[Manager 4,Project 4]",
    "x_3_4": "x_vars[Manager 4,Project 5]",
    "x_3_5": "x_vars[Manager 4,Project 6]",
    "x_3_6": "x_vars[Manager 4,Project 7]",
    "x_4_0": "x_vars[Manager 5,Project 1]",
    "x_4_1": "x_vars[Manager 5,Project 2]",
    "x_4_2": "x_vars[Manager 5,Project 3]",
    "x_4_3": "x_vars[Manager 5,Project 4]",
    "x_4_4": "x_vars[Manager 5,Project 5]",
    "x_4_5": "x_vars[Manager 5,Project 6]",
    "x_4_6": "x_vars[Manager 5,Project 7]",
    "x_5_0": "x_vars[Manager 6,Project 1]",
    "x_5_1": "x_vars[Manager 6,Project 2]",
    "x_5_2": "x_vars[Manager 6,Project 3]",
    "x_5_3": "x_vars[Manager 6,Project 4]",
    "x_5_4": "x_vars[Manager 6,Project 5]",
    "x_5_5": "x_vars[Manager 6,Project 6]",
    "x_5_6": "x_vars[Manager 6,Project 7]",
    "x_6_0": "x_vars[Manager 7,Project 1]",
    "x_6_1": "x_vars[Manager 7,Project 2]",
    "x_6_2": "x_vars[Manager 7,Project 3]",
    "x_6_3": "x_vars[Manager 7,Project 4]",
    "x_6_4": "x_vars[Manager 7,Project 5]",
    "x_6_5": "x_vars[Manager 7,Project 6]",
    "x_6_6": "x_vars[Manager 7,Project 7]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-071": {
   "source_code_sha256": "12d645bbab01d777e14f4e53b74c5a861f1f9d228642b90653641e57da3c3aac",
   "original_solution_sha256": "bd2c67c5b4f0f21edf7640d760524a54c53fc5a9b6b33f19acb6495c0b9e16bb",
   "bindings": [
    [
     "C0",
     "x_vars[Classic Oxford Shirt]"
    ],
    [
     "C1",
     "x_vars[Cotton Crew Neck T-Shirt]"
    ],
    [
     "C2",
     "x_vars[French Terry Hoodie]"
    ],
    [
     "C3",
     "x_vars[Denim Snap Shirt]"
    ],
    [
     "C4",
     "x_vars[Jersey V-Neck T-Shirt]"
    ],
    [
     "C5",
     "x_vars[Unisex Jogger Pants]"
    ],
    [
     "C6",
     "x_vars[Flannel Plaid Shirt]"
    ],
    [
     "C7",
     "x_vars[Performance Polo Shirt]"
    ],
    [
     "C8",
     "x_vars[Long-sleeve Henley]"
    ],
    [
     "C9",
     "x_vars[Heavyweight Sweatshirt]"
    ],
    [
     "C10",
     "x_vars[Linen Blend Shirt]"
    ],
    [
     "C11",
     "x_vars[Graphic Print Tee]"
    ],
    [
     "C12",
     "x_vars[Quilted Vest]"
    ],
    [
     "C13",
     "x_vars[Chambray Work Shirt]"
    ],
    [
     "C14",
     "x_vars[Microfiber Sport Shirt]"
    ],
    [
     "C15",
     "x_vars[Wool Blend Peacoat]"
    ],
    [
     "C16",
     "x_vars[Seersucker Short-sleeve]"
    ],
    [
     "C17",
     "x_vars[Twill Utility Shirt]"
    ],
    [
     "C18",
     "x_vars[Basic Scoop Neck Tee]"
    ],
    [
     "C19",
     "x_vars[Waffle Knit Pullover]"
    ],
    [
     "C20",
     "x_vars[Button-Down Poplin Shirt]"
    ],
    [
     "C21",
     "x_vars[Slub Cotton T-Shirt]"
    ],
    [
     "C22",
     "x_vars[Drawstring Cargo Pants]"
    ],
    [
     "C23",
     "x_vars[Pima Cotton Dress Shirt]"
    ],
    [
     "C24",
     "x_vars[Pocket T-Shirt]"
    ],
    [
     "C25",
     "x_vars[Fleece-Lined Track Suit]"
    ],
    [
     "C26",
     "x_vars[Printed Camp Collar Shirt]"
    ],
    [
     "C27",
     "x_vars[Performance Dry-Fit Tee]"
    ],
    [
     "C28",
     "x_vars[Corduroy Shacket]"
    ],
    [
     "C29",
     "x_vars[Herringbone Casual Shirt]"
    ],
    [
     "C30",
     "x_vars[Organic Cotton Tee]"
    ],
    [
     "C31",
     "x_vars[Cashmere Blend Sweater]"
    ],
    [
     "C32",
     "x_vars[Striped Rugby Shirt]"
    ],
    [
     "C33",
     "x_vars[Raglan Sleeve Tee]"
    ],
    [
     "C34",
     "x_vars[Waterproof Rain Jacket]"
    ],
    [
     "C35",
     "x_vars[Micro-check Business Shirt]"
    ],
    [
     "C36",
     "x_vars[Heather Grey T-Shirt]"
    ],
    [
     "C37",
     "x_vars[Slim Fit Chinos]"
    ],
    [
     "C38",
     "x_vars[Band Collar Shirt]"
    ],
    [
     "C39",
     "x_vars[Triblend Jersey Tee]"
    ],
    [
     "C40",
     "x_vars[Sherpa Lined Pullover]"
    ],
    [
     "C41",
     "x_vars[Western Style Shirt]"
    ],
    [
     "C42",
     "x_vars[Recycled Poly T-Shirt]"
    ],
    [
     "C43",
     "x_vars[Insulated Puffer Jacket]"
    ],
    [
     "C44",
     "x_vars[End-on-End Fabric Shirt]"
    ],
    [
     "C45",
     "x_vars[Oversized Graphic Tee]"
    ],
    [
     "C46",
     "x_vars[Stretch Hybrid Shorts]"
    ],
    [
     "C47",
     "x_vars[Non-Iron Travel Shirt]"
    ],
    [
     "C48",
     "x_vars[Sun-Protective T-Shirt]"
    ],
    [
     "C49",
     "x_vars[Velvet Bomber Jacket]"
    ],
    [
     "C50",
     "x_vars[Madras Plaid Shirt]"
    ],
    [
     "C51",
     "x_vars[Tie-Dye T-Shirt]"
    ],
    [
     "C52",
     "x_vars[Lightweight Windbreaker]"
    ],
    [
     "C53",
     "x_vars[Pinpoint Oxford Shirt]"
    ],
    [
     "C54",
     "x_vars[Bamboo Viscose Tee]"
    ],
    [
     "C55",
     "x_vars[Tapered Denim Jeans]"
    ],
    [
     "C56",
     "x_vars[Contrast Collar Shirt]"
    ],
    [
     "C57",
     "x_vars[Mock Neck Long-sleeve]"
    ],
    [
     "C58",
     "x_vars[Down-filled Parka]"
    ],
    [
     "C59",
     "x_vars[Textured Grid Shirt]"
    ],
    [
     "C60",
     "x_vars[Heavy Cotton T-Shirt]"
    ],
    [
     "C61",
     "x_vars[Convertible Zip-Off Trousers]"
    ],
    [
     "C62",
     "x_vars[Tencel Blend Casual Shirt]"
    ],
    [
     "C63",
     "x_vars[Relaxed Fit Tee]"
    ],
    [
     "C64",
     "x_vars[Faux Leather Jacket]"
    ],
    [
     "C65",
     "x_vars[Speckled Slub Shirt]"
    ],
    [
     "C66",
     "x_vars[Embroidered T-Shirt]"
    ],
    [
     "C67",
     "x_vars[Knit Cardigan Sweater]"
    ],
    [
     "C68",
     "x_vars[Two-Pocket Utility Shirt]"
    ],
    [
     "C69",
     "x_vars[Performance Stretch Tee]"
    ],
    [
     "C70",
     "x_vars[Technical Shell Jacket]"
    ],
    [
     "C71",
     "x_vars[Semi-Spread Collar Shirt]"
    ],
    [
     "C72",
     "x_vars[Washed Pigment Dye Tee]"
    ],
    [
     "C73",
     "x_vars[Twill Blazer (Unlined)]"
    ],
    [
     "C74",
     "x_vars[Brushed Twill Shirt]"
    ],
    [
     "C75",
     "x_vars[Supima Cotton Tee]"
    ],
    [
     "C76",
     "x_vars[High-Waist Trousers]"
    ],
    [
     "C77",
     "x_vars[Printed Geometric Shirt]"
    ],
    [
     "C78",
     "x_vars[Minimalist Plain Tee]"
    ],
    [
     "C79",
     "x_vars[Reversible Quilted Jacket]"
    ],
    [
     "C80",
     "x_vars[Dobby Weave Shirt]"
    ],
    [
     "C81",
     "x_vars[Heavyweight Boxy Tee]"
    ],
    [
     "C82",
     "x_vars[Casual Wool Overcoat]"
    ],
    [
     "C83",
     "x_vars[Half-Zip Pullover Shirt]"
    ],
    [
     "C84",
     "x_vars[Drop Shoulder T-Shirt]"
    ],
    [
     "C85",
     "x_vars[Straight Leg Corduroys]"
    ],
    [
     "C86",
     "x_vars[Camp Collar Floral Shirt]"
    ],
    [
     "C87",
     "x_vars[Garment Dyed Tee]"
    ],
    [
     "C88",
     "x_vars[Suede Bomber Jacket]"
    ],
    [
     "C89",
     "x_vars[Melange Knit Shirt]"
    ],
    [
     "C90",
     "x_vars[Longline T-Shirt]"
    ],
    [
     "C91",
     "x_vars[Wide Leg Culottes]"
    ],
    [
     "C92",
     "x_vars[Tropical Print Shirt]"
    ],
    [
     "C93",
     "x_vars[Distressed Cotton Tee]"
    ],
    [
     "C94",
     "x_vars[Patchwork Denim Jacket]"
    ],
    [
     "C95",
     "x_vars[Micro-Dot Shirt]"
    ],
    [
     "C96",
     "x_vars[Muscle Fit T-Shirt]"
    ],
    [
     "C97",
     "x_vars[Woven Utility Vest]"
    ],
    [
     "C98",
     "x_vars[Silk Blend Dress Shirt]"
    ],
    [
     "C99",
     "x_vars[Heavyweight Jersey T-Shirt]"
    ],
    [
     "C100",
     "x_vars[Slim Fit Ankle Grazer]"
    ],
    [
     "C101",
     "x_vars[Lattice Weave Shirt]"
    ],
    [
     "C102",
     "x_vars[Boxy Cropped Tee]"
    ],
    [
     "C103",
     "x_vars[Reversible Bomber Jacket]"
    ],
    [
     "C104",
     "x_vars[Stretch Performance Shirt]"
    ],
    [
     "C105",
     "x_vars[Sublimation Print Tee]"
    ],
    [
     "C106",
     "x_vars[Brushed Cotton Trousers]"
    ],
    [
     "C107",
     "x_vars[Micro-Waffle Knit Shirt]"
    ],
    [
     "C108",
     "x_vars[Reflective Detail Tee]"
    ],
    [
     "C109",
     "x_vars[Shearling Lined Coat]"
    ],
    [
     "C110",
     "x_vars[Printed Patchwork Shirt]"
    ],
    [
     "C111",
     "x_vars[Washed Effect T-Shirt]"
    ],
    [
     "C112",
     "x_vars[Velour Track Pants]"
    ],
    [
     "C113",
     "x_vars[Fine Wale Corduroy Shirt]"
    ],
    [
     "C114",
     "x_vars[Organic Hemp T-Shirt]"
    ],
    [
     "C115",
     "x_vars[Leather Biker Jacket]"
    ],
    [
     "C116",
     "x_vars[Supersoft Modal Shirt]"
    ],
    [
     "C117",
     "x_vars[Deep V-Neck Tee]"
    ],
    [
     "C118",
     "x_vars[Wool Blend Trousers]"
    ],
    [
     "C119",
     "x_vars[Contrast Stitch Shirt]"
    ],
    [
     "C120",
     "x_vars[Embroidered Pocket Tee]"
    ],
    [
     "C121",
     "x_vars[Camo Print Hoodie]"
    ],
    [
     "C122",
     "x_vars[Silk-Touch Poplin Shirt]"
    ],
    [
     "C123",
     "x_vars[Slim Fit Pique Polo]"
    ],
    [
     "C124",
     "x_vars[Technical Cargo Vest]"
    ],
    [
     "C125",
     "x_vars[Cutaway Collar Shirt]"
    ],
    [
     "C126",
     "x_vars[Lycra Blend Active Tee]"
    ],
    [
     "C127",
     "x_vars[Heavy Knit Sweater]"
    ],
    [
     "C128",
     "x_vars[Textured Stripe Shirt]"
    ],
    [
     "C129",
     "x_vars[Bio-Washed T-Shirt]"
    ],
    [
     "C130",
     "x_vars[Mohair Blend Cardigan]"
    ],
    [
     "C131",
     "x_vars[Shantung Silk Shirt]"
    ],
    [
     "C132",
     "x_vars[Distressed Denim Tee]"
    ],
    [
     "C133",
     "x_vars[High-Tech Ski Jacket]"
    ],
    [
     "C134",
     "x_vars[Fine Poplin Check Shirt]"
    ],
    [
     "C135",
     "x_vars[Slim Fit Linen T-Shirt]"
    ],
    [
     "C136",
     "x_vars[Ripstop Utility Pants]"
    ],
    [
     "C137",
     "x_vars[Classic Barrel Cuff Shirt]"
    ],
    [
     "C138",
     "x_vars[Bamboo Fiber Tee]"
    ],
    [
     "C139",
     "x_vars[Shearling Trench Coat]"
    ],
    [
     "C140",
     "x_vars[Slim Fit Chambray]"
    ],
    [
     "C141",
     "x_vars[Printed Geometric Tee]"
    ],
    [
     "C142",
     "x_vars[Wool Cargo Trousers]"
    ],
    [
     "C143",
     "x_vars[Two-Tone Oxford Shirt]"
    ],
    [
     "C144",
     "x_vars[Slogan Print T-Shirt]"
    ],
    [
     "C145",
     "x_vars[Water-Resistant Shell]"
    ],
    [
     "C146",
     "x_vars[Grandad Collar Shirt]"
    ],
    [
     "C147",
     "x_vars[Embroidered Logo Tee]"
    ],
    [
     "C148",
     "x_vars[Casual Knit Blazer]"
    ],
    [
     "C149",
     "x_vars[Eco-Friendly Hemp Shirt]"
    ],
    [
     "C150",
     "x_vars[Cotton Slub Polo]"
    ],
    [
     "C151",
     "x_vars[Corduroy Carpenter Pants]"
    ],
    [
     "C152",
     "x_vars[Linen Safari Shirt]"
    ],
    [
     "C153",
     "x_vars[Ribbed Crew Neck Tee]"
    ],
    [
     "C154",
     "x_vars[Hybrid Hiking Trousers]"
    ],
    [
     "C155",
     "x_vars[Brushed Flannel Shirt]"
    ],
    [
     "C156",
     "x_vars[Microfiber V-Neck]"
    ],
    [
     "C157",
     "x_vars[Down-Alternative Vest]"
    ],
    [
     "C158",
     "x_vars[Checked Seersucker Shirt]"
    ],
    [
     "C159",
     "x_vars[Fitted Basic Tee]"
    ],
    [
     "C160",
     "x_vars[Heavyweight Denim Jacket]"
    ],
    [
     "C161",
     "x_vars[Woven Stripe Shirt]"
    ],
    [
     "C162",
     "x_vars[Tech Fleece Pullover]"
    ],
    [
     "C163",
     "x_vars[Pin Dot Print Shirt]"
    ],
    [
     "C164",
     "x_vars[Supersoft Modal T-Shirt]"
    ],
    [
     "C165",
     "x_vars[Water-Resistant Chinos]"
    ],
    [
     "C166",
     "x_vars[Tailored Fit Dress Shirt]"
    ],
    [
     "C167",
     "x_vars[Printed Art Graphic Tee]"
    ],
    [
     "C168",
     "x_vars[Cashmere Cable Knit]"
    ],
    [
     "C169",
     "x_vars[Plaid Flannel Overshirt]"
    ],
    [
     "C170",
     "x_vars[Moisture-Wicking Polo]"
    ],
    [
     "C171",
     "x_vars[Slim Tapered Jeans]"
    ],
    [
     "C172",
     "x_vars[Contrast Cuff Shirt]"
    ],
    [
     "C173",
     "x_vars[Sustainable Cotton Tee]"
    ],
    [
     "C174",
     "x_vars[Trench Coat (Lined)]"
    ],
    [
     "C175",
     "x_vars[Spread Collar Shirt]"
    ],
    [
     "C176",
     "x_vars[Printed Stripe Tee]"
    ],
    [
     "C177",
     "x_vars[Non-Iron Twill Shirt]"
    ],
    [
     "C178",
     "x_vars[Recycled Material Tee]"
    ],
    [
     "C179",
     "x_vars[Faux Fur Coat]"
    ],
    [
     "C180",
     "x_vars[Seersucker Popover Shirt]"
    ],
    [
     "C181",
     "x_vars[Graphic Back Print Tee]"
    ],
    [
     "C182",
     "x_vars[Corduroy Field Jacket]"
    ],
    [
     "C183",
     "x_vars[Linen/Cotton Blend Shirt]"
    ],
    [
     "C184",
     "x_vars[Lightweight Jersey Tee]"
    ],
    [
     "C185",
     "x_vars[High Performance Hiking Pants]"
    ],
    [
     "C186",
     "x_vars[Cuban Collar Shirt]"
    ],
    [
     "C187",
     "x_vars[Faded Pigment Tee]"
    ],
    [
     "C188",
     "x_vars[Canvas Chore Coat]"
    ],
    [
     "C189",
     "x_vars[Brushed Cotton Shirt]"
    ],
    [
     "C190",
     "x_vars[Ribbed Knit T-Shirt]"
    ],
    [
     "C191",
     "x_vars[Wide Leg Trousers]"
    ],
    [
     "C192",
     "x_vars[Patchwork Denim Shirt]"
    ],
    [
     "C193",
     "x_vars[Basic White T-Shirt]"
    ],
    [
     "C194",
     "x_vars[Mohair Knit Sweater]"
    ],
    [
     "C195",
     "x_vars[Contrast Stitch Polo]"
    ],
    [
     "C196",
     "x_vars[Tailored Wool Trousers]"
    ],
    [
     "C197",
     "x_vars[Heavyweight Denim Shirt]"
    ]
   ],
   "primary_mapping": {
    "x[Classic_Oxford_Shirt]": "x_vars[Classic Oxford Shirt]",
    "x[Cotton_Crew_Neck_T-Shirt]": "x_vars[Cotton Crew Neck T-Shirt]",
    "x[French_Terry_Hoodie]": "x_vars[French Terry Hoodie]",
    "x[Denim_Snap_Shirt]": "x_vars[Denim Snap Shirt]",
    "x[Jersey_V-Neck_T-Shirt]": "x_vars[Jersey V-Neck T-Shirt]",
    "x[Unisex_Jogger_Pants]": "x_vars[Unisex Jogger Pants]",
    "x[Flannel_Plaid_Shirt]": "x_vars[Flannel Plaid Shirt]",
    "x[Performance_Polo_Shirt]": "x_vars[Performance Polo Shirt]",
    "x[Long-sleeve_Henley]": "x_vars[Long-sleeve Henley]",
    "x[Heavyweight_Sweatshirt]": "x_vars[Heavyweight Sweatshirt]",
    "x[Linen_Blend_Shirt]": "x_vars[Linen Blend Shirt]",
    "x[Graphic_Print_Tee]": "x_vars[Graphic Print Tee]",
    "x[Quilted_Vest]": "x_vars[Quilted Vest]",
    "x[Chambray_Work_Shirt]": "x_vars[Chambray Work Shirt]",
    "x[Microfiber_Sport_Shirt]": "x_vars[Microfiber Sport Shirt]",
    "x[Wool_Blend_Peacoat]": "x_vars[Wool Blend Peacoat]",
    "x[Seersucker_Short-sleeve]": "x_vars[Seersucker Short-sleeve]",
    "x[Twill_Utility_Shirt]": "x_vars[Twill Utility Shirt]",
    "x[Basic_Scoop_Neck_Tee]": "x_vars[Basic Scoop Neck Tee]",
    "x[Waffle_Knit_Pullover]": "x_vars[Waffle Knit Pullover]",
    "x[Button-Down_Poplin_Shirt]": "x_vars[Button-Down Poplin Shirt]",
    "x[Slub_Cotton_T-Shirt]": "x_vars[Slub Cotton T-Shirt]",
    "x[Drawstring_Cargo_Pants]": "x_vars[Drawstring Cargo Pants]",
    "x[Pima_Cotton_Dress_Shirt]": "x_vars[Pima Cotton Dress Shirt]",
    "x[Pocket_T-Shirt]": "x_vars[Pocket T-Shirt]",
    "x[Fleece-Lined_Track_Suit]": "x_vars[Fleece-Lined Track Suit]",
    "x[Printed_Camp_Collar_Shirt]": "x_vars[Printed Camp Collar Shirt]",
    "x[Performance_Dry-Fit_Tee]": "x_vars[Performance Dry-Fit Tee]",
    "x[Corduroy_Shacket]": "x_vars[Corduroy Shacket]",
    "x[Herringbone_Casual_Shirt]": "x_vars[Herringbone Casual Shirt]",
    "x[Organic_Cotton_Tee]": "x_vars[Organic Cotton Tee]",
    "x[Cashmere_Blend_Sweater]": "x_vars[Cashmere Blend Sweater]",
    "x[Striped_Rugby_Shirt]": "x_vars[Striped Rugby Shirt]",
    "x[Raglan_Sleeve_Tee]": "x_vars[Raglan Sleeve Tee]",
    "x[Waterproof_Rain_Jacket]": "x_vars[Waterproof Rain Jacket]",
    "x[Micro-check_Business_Shirt]": "x_vars[Micro-check Business Shirt]",
    "x[Heather_Grey_T-Shirt]": "x_vars[Heather Grey T-Shirt]",
    "x[Slim_Fit_Chinos]": "x_vars[Slim Fit Chinos]",
    "x[Band_Collar_Shirt]": "x_vars[Band Collar Shirt]",
    "x[Triblend_Jersey_Tee]": "x_vars[Triblend Jersey Tee]",
    "x[Sherpa_Lined_Pullover]": "x_vars[Sherpa Lined Pullover]",
    "x[Western_Style_Shirt]": "x_vars[Western Style Shirt]",
    "x[Recycled_Poly_T-Shirt]": "x_vars[Recycled Poly T-Shirt]",
    "x[Insulated_Puffer_Jacket]": "x_vars[Insulated Puffer Jacket]",
    "x[End-on-End_Fabric_Shirt]": "x_vars[End-on-End Fabric Shirt]",
    "x[Oversized_Graphic_Tee]": "x_vars[Oversized Graphic Tee]",
    "x[Stretch_Hybrid_Shorts]": "x_vars[Stretch Hybrid Shorts]",
    "x[Non-Iron_Travel_Shirt]": "x_vars[Non-Iron Travel Shirt]",
    "x[Sun-Protective_T-Shirt]": "x_vars[Sun-Protective T-Shirt]",
    "x[Velvet_Bomber_Jacket]": "x_vars[Velvet Bomber Jacket]",
    "x[Madras_Plaid_Shirt]": "x_vars[Madras Plaid Shirt]",
    "x[Tie-Dye_T-Shirt]": "x_vars[Tie-Dye T-Shirt]",
    "x[Lightweight_Windbreaker]": "x_vars[Lightweight Windbreaker]",
    "x[Pinpoint_Oxford_Shirt]": "x_vars[Pinpoint Oxford Shirt]",
    "x[Bamboo_Viscose_Tee]": "x_vars[Bamboo Viscose Tee]",
    "x[Tapered_Denim_Jeans]": "x_vars[Tapered Denim Jeans]",
    "x[Contrast_Collar_Shirt]": "x_vars[Contrast Collar Shirt]",
    "x[Mock_Neck_Long-sleeve]": "x_vars[Mock Neck Long-sleeve]",
    "x[Down-filled_Parka]": "x_vars[Down-filled Parka]",
    "x[Textured_Grid_Shirt]": "x_vars[Textured Grid Shirt]",
    "x[Heavy_Cotton_T-Shirt]": "x_vars[Heavy Cotton T-Shirt]",
    "x[Convertible_Zip-Off_Trousers]": "x_vars[Convertible Zip-Off Trousers]",
    "x[Tencel_Blend_Casual_Shirt]": "x_vars[Tencel Blend Casual Shirt]",
    "x[Relaxed_Fit_Tee]": "x_vars[Relaxed Fit Tee]",
    "x[Faux_Leather_Jacket]": "x_vars[Faux Leather Jacket]",
    "x[Speckled_Slub_Shirt]": "x_vars[Speckled Slub Shirt]",
    "x[Embroidered_T-Shirt]": "x_vars[Embroidered T-Shirt]",
    "x[Knit_Cardigan_Sweater]": "x_vars[Knit Cardigan Sweater]",
    "x[Two-Pocket_Utility_Shirt]": "x_vars[Two-Pocket Utility Shirt]",
    "x[Performance_Stretch_Tee]": "x_vars[Performance Stretch Tee]",
    "x[Technical_Shell_Jacket]": "x_vars[Technical Shell Jacket]",
    "x[Semi-Spread_Collar_Shirt]": "x_vars[Semi-Spread Collar Shirt]",
    "x[Washed_Pigment_Dye_Tee]": "x_vars[Washed Pigment Dye Tee]",
    "x[Twill_Blazer_(Unlined)]": "x_vars[Twill Blazer (Unlined)]",
    "x[Brushed_Twill_Shirt]": "x_vars[Brushed Twill Shirt]",
    "x[Supima_Cotton_Tee]": "x_vars[Supima Cotton Tee]",
    "x[High-Waist_Trousers]": "x_vars[High-Waist Trousers]",
    "x[Printed_Geometric_Shirt]": "x_vars[Printed Geometric Shirt]",
    "x[Minimalist_Plain_Tee]": "x_vars[Minimalist Plain Tee]",
    "x[Reversible_Quilted_Jacket]": "x_vars[Reversible Quilted Jacket]",
    "x[Dobby_Weave_Shirt]": "x_vars[Dobby Weave Shirt]",
    "x[Heavyweight_Boxy_Tee]": "x_vars[Heavyweight Boxy Tee]",
    "x[Casual_Wool_Overcoat]": "x_vars[Casual Wool Overcoat]",
    "x[Half-Zip_Pullover_Shirt]": "x_vars[Half-Zip Pullover Shirt]",
    "x[Drop_Shoulder_T-Shirt]": "x_vars[Drop Shoulder T-Shirt]",
    "x[Straight_Leg_Corduroys]": "x_vars[Straight Leg Corduroys]",
    "x[Camp_Collar_Floral_Shirt]": "x_vars[Camp Collar Floral Shirt]",
    "x[Garment_Dyed_Tee]": "x_vars[Garment Dyed Tee]",
    "x[Suede_Bomber_Jacket]": "x_vars[Suede Bomber Jacket]",
    "x[Melange_Knit_Shirt]": "x_vars[Melange Knit Shirt]",
    "x[Longline_T-Shirt]": "x_vars[Longline T-Shirt]",
    "x[Wide_Leg_Culottes]": "x_vars[Wide Leg Culottes]",
    "x[Tropical_Print_Shirt]": "x_vars[Tropical Print Shirt]",
    "x[Distressed_Cotton_Tee]": "x_vars[Distressed Cotton Tee]",
    "x[Patchwork_Denim_Jacket]": "x_vars[Patchwork Denim Jacket]",
    "x[Micro-Dot_Shirt]": "x_vars[Micro-Dot Shirt]",
    "x[Muscle_Fit_T-Shirt]": "x_vars[Muscle Fit T-Shirt]",
    "x[Woven_Utility_Vest]": "x_vars[Woven Utility Vest]",
    "x[Silk_Blend_Dress_Shirt]": "x_vars[Silk Blend Dress Shirt]",
    "x[Heavyweight_Jersey_T-Shirt]": "x_vars[Heavyweight Jersey T-Shirt]",
    "x[Slim_Fit_Ankle_Grazer]": "x_vars[Slim Fit Ankle Grazer]",
    "x[Lattice_Weave_Shirt]": "x_vars[Lattice Weave Shirt]",
    "x[Boxy_Cropped_Tee]": "x_vars[Boxy Cropped Tee]",
    "x[Reversible_Bomber_Jacket]": "x_vars[Reversible Bomber Jacket]",
    "x[Stretch_Performance_Shirt]": "x_vars[Stretch Performance Shirt]",
    "x[Sublimation_Print_Tee]": "x_vars[Sublimation Print Tee]",
    "x[Brushed_Cotton_Trousers]": "x_vars[Brushed Cotton Trousers]",
    "x[Micro-Waffle_Knit_Shirt]": "x_vars[Micro-Waffle Knit Shirt]",
    "x[Reflective_Detail_Tee]": "x_vars[Reflective Detail Tee]",
    "x[Shearling_Lined_Coat]": "x_vars[Shearling Lined Coat]",
    "x[Printed_Patchwork_Shirt]": "x_vars[Printed Patchwork Shirt]",
    "x[Washed_Effect_T-Shirt]": "x_vars[Washed Effect T-Shirt]",
    "x[Velour_Track_Pants]": "x_vars[Velour Track Pants]",
    "x[Fine_Wale_Corduroy_Shirt]": "x_vars[Fine Wale Corduroy Shirt]",
    "x[Organic_Hemp_T-Shirt]": "x_vars[Organic Hemp T-Shirt]",
    "x[Leather_Biker_Jacket]": "x_vars[Leather Biker Jacket]",
    "x[Supersoft_Modal_Shirt]": "x_vars[Supersoft Modal Shirt]",
    "x[Deep_V-Neck_Tee]": "x_vars[Deep V-Neck Tee]",
    "x[Wool_Blend_Trousers]": "x_vars[Wool Blend Trousers]",
    "x[Contrast_Stitch_Shirt]": "x_vars[Contrast Stitch Shirt]",
    "x[Embroidered_Pocket_Tee]": "x_vars[Embroidered Pocket Tee]",
    "x[Camo_Print_Hoodie]": "x_vars[Camo Print Hoodie]",
    "x[Silk-Touch_Poplin_Shirt]": "x_vars[Silk-Touch Poplin Shirt]",
    "x[Slim_Fit_Pique_Polo]": "x_vars[Slim Fit Pique Polo]",
    "x[Technical_Cargo_Vest]": "x_vars[Technical Cargo Vest]",
    "x[Cutaway_Collar_Shirt]": "x_vars[Cutaway Collar Shirt]",
    "x[Lycra_Blend_Active_Tee]": "x_vars[Lycra Blend Active Tee]",
    "x[Heavy_Knit_Sweater]": "x_vars[Heavy Knit Sweater]",
    "x[Textured_Stripe_Shirt]": "x_vars[Textured Stripe Shirt]",
    "x[Bio-Washed_T-Shirt]": "x_vars[Bio-Washed T-Shirt]",
    "x[Mohair_Blend_Cardigan]": "x_vars[Mohair Blend Cardigan]",
    "x[Shantung_Silk_Shirt]": "x_vars[Shantung Silk Shirt]",
    "x[Distressed_Denim_Tee]": "x_vars[Distressed Denim Tee]",
    "x[High-Tech_Ski_Jacket]": "x_vars[High-Tech Ski Jacket]",
    "x[Fine_Poplin_Check_Shirt]": "x_vars[Fine Poplin Check Shirt]",
    "x[Slim_Fit_Linen_T-Shirt]": "x_vars[Slim Fit Linen T-Shirt]",
    "x[Ripstop_Utility_Pants]": "x_vars[Ripstop Utility Pants]",
    "x[Classic_Barrel_Cuff_Shirt]": "x_vars[Classic Barrel Cuff Shirt]",
    "x[Bamboo_Fiber_Tee]": "x_vars[Bamboo Fiber Tee]",
    "x[Shearling_Trench_Coat]": "x_vars[Shearling Trench Coat]",
    "x[Slim_Fit_Chambray]": "x_vars[Slim Fit Chambray]",
    "x[Printed_Geometric_Tee]": "x_vars[Printed Geometric Tee]",
    "x[Wool_Cargo_Trousers]": "x_vars[Wool Cargo Trousers]",
    "x[Two-Tone_Oxford_Shirt]": "x_vars[Two-Tone Oxford Shirt]",
    "x[Slogan_Print_T-Shirt]": "x_vars[Slogan Print T-Shirt]",
    "x[Water-Resistant_Shell]": "x_vars[Water-Resistant Shell]",
    "x[Grandad_Collar_Shirt]": "x_vars[Grandad Collar Shirt]",
    "x[Embroidered_Logo_Tee]": "x_vars[Embroidered Logo Tee]",
    "x[Casual_Knit_Blazer]": "x_vars[Casual Knit Blazer]",
    "x[Eco-Friendly_Hemp_Shirt]": "x_vars[Eco-Friendly Hemp Shirt]",
    "x[Cotton_Slub_Polo]": "x_vars[Cotton Slub Polo]",
    "x[Corduroy_Carpenter_Pants]": "x_vars[Corduroy Carpenter Pants]",
    "x[Linen_Safari_Shirt]": "x_vars[Linen Safari Shirt]",
    "x[Ribbed_Crew_Neck_Tee]": "x_vars[Ribbed Crew Neck Tee]",
    "x[Hybrid_Hiking_Trousers]": "x_vars[Hybrid Hiking Trousers]",
    "x[Brushed_Flannel_Shirt]": "x_vars[Brushed Flannel Shirt]",
    "x[Microfiber_V-Neck]": "x_vars[Microfiber V-Neck]",
    "x[Down-Alternative_Vest]": "x_vars[Down-Alternative Vest]",
    "x[Checked_Seersucker_Shirt]": "x_vars[Checked Seersucker Shirt]",
    "x[Fitted_Basic_Tee]": "x_vars[Fitted Basic Tee]",
    "x[Heavyweight_Denim_Jacket]": "x_vars[Heavyweight Denim Jacket]",
    "x[Woven_Stripe_Shirt]": "x_vars[Woven Stripe Shirt]",
    "x[Tech_Fleece_Pullover]": "x_vars[Tech Fleece Pullover]",
    "x[Pin_Dot_Print_Shirt]": "x_vars[Pin Dot Print Shirt]",
    "x[Supersoft_Modal_T-Shirt]": "x_vars[Supersoft Modal T-Shirt]",
    "x[Water-Resistant_Chinos]": "x_vars[Water-Resistant Chinos]",
    "x[Tailored_Fit_Dress_Shirt]": "x_vars[Tailored Fit Dress Shirt]",
    "x[Printed_Art_Graphic_Tee]": "x_vars[Printed Art Graphic Tee]",
    "x[Cashmere_Cable_Knit]": "x_vars[Cashmere Cable Knit]",
    "x[Plaid_Flannel_Overshirt]": "x_vars[Plaid Flannel Overshirt]",
    "x[Moisture-Wicking_Polo]": "x_vars[Moisture-Wicking Polo]",
    "x[Slim_Tapered_Jeans]": "x_vars[Slim Tapered Jeans]",
    "x[Contrast_Cuff_Shirt]": "x_vars[Contrast Cuff Shirt]",
    "x[Sustainable_Cotton_Tee]": "x_vars[Sustainable Cotton Tee]",
    "x[Trench_Coat_(Lined)]": "x_vars[Trench Coat (Lined)]",
    "x[Spread_Collar_Shirt]": "x_vars[Spread Collar Shirt]",
    "x[Printed_Stripe_Tee]": "x_vars[Printed Stripe Tee]",
    "x[Non-Iron_Twill_Shirt]": "x_vars[Non-Iron Twill Shirt]",
    "x[Recycled_Material_Tee]": "x_vars[Recycled Material Tee]",
    "x[Faux_Fur_Coat]": "x_vars[Faux Fur Coat]",
    "x[Seersucker_Popover_Shirt]": "x_vars[Seersucker Popover Shirt]",
    "x[Graphic_Back_Print_Tee]": "x_vars[Graphic Back Print Tee]",
    "x[Corduroy_Field_Jacket]": "x_vars[Corduroy Field Jacket]",
    "x[Linen/Cotton_Blend_Shirt]": "x_vars[Linen/Cotton Blend Shirt]",
    "x[Lightweight_Jersey_Tee]": "x_vars[Lightweight Jersey Tee]",
    "x[High_Performance_Hiking_Pants]": "x_vars[High Performance Hiking Pants]",
    "x[Cuban_Collar_Shirt]": "x_vars[Cuban Collar Shirt]",
    "x[Faded_Pigment_Tee]": "x_vars[Faded Pigment Tee]",
    "x[Canvas_Chore_Coat]": "x_vars[Canvas Chore Coat]",
    "x[Brushed_Cotton_Shirt]": "x_vars[Brushed Cotton Shirt]",
    "x[Ribbed_Knit_T-Shirt]": "x_vars[Ribbed Knit T-Shirt]",
    "x[Wide_Leg_Trousers]": "x_vars[Wide Leg Trousers]",
    "x[Patchwork_Denim_Shirt]": "x_vars[Patchwork Denim Shirt]",
    "x[Basic_White_T-Shirt]": "x_vars[Basic White T-Shirt]",
    "x[Mohair_Knit_Sweater]": "x_vars[Mohair Knit Sweater]",
    "x[Contrast_Stitch_Polo]": "x_vars[Contrast Stitch Polo]",
    "x[Tailored_Wool_Trousers]": "x_vars[Tailored Wool Trousers]",
    "x[Heavyweight_Denim_Shirt]": "x_vars[Heavyweight Denim Shirt]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-072": {
   "source_code_sha256": "972a3170510df3735060880ff1af7fd2b311ef229ef607033b389e5d1120df56",
   "original_solution_sha256": "6b9d488653a612575c58e5e3941364fb5f47c7b7a1cc8f9649a50a0372ff8594",
   "bindings": [
    [
     "C0",
     "x_vars[1]"
    ],
    [
     "C1",
     "x_vars[2]"
    ],
    [
     "C2",
     "x_vars[3]"
    ],
    [
     "C3",
     "x_vars[4]"
    ],
    [
     "C4",
     "x_vars[5]"
    ],
    [
     "C5",
     "x_vars[6]"
    ],
    [
     "C6",
     "x_vars[7]"
    ],
    [
     "C7",
     "x_vars[8]"
    ],
    [
     "C8",
     "x_vars[9]"
    ],
    [
     "C9",
     "x_vars[10]"
    ],
    [
     "C10",
     "x_vars[11]"
    ],
    [
     "C11",
     "x_vars[12]"
    ],
    [
     "C12",
     "x_vars[13]"
    ],
    [
     "C13",
     "x_vars[14]"
    ],
    [
     "C14",
     "x_vars[15]"
    ],
    [
     "C15",
     "x_vars[16]"
    ],
    [
     "C16",
     "x_vars[17]"
    ],
    [
     "C17",
     "x_vars[18]"
    ],
    [
     "C18",
     "x_vars[19]"
    ],
    [
     "C19",
     "x_vars[20]"
    ],
    [
     "C20",
     "x_vars[21]"
    ],
    [
     "C21",
     "x_vars[22]"
    ],
    [
     "C22",
     "x_vars[23]"
    ],
    [
     "C23",
     "x_vars[24]"
    ]
   ],
   "primary_mapping": {
    "x[0]": "x_vars[1]",
    "x[1]": "x_vars[2]",
    "x[2]": "x_vars[3]",
    "x[3]": "x_vars[4]",
    "x[4]": "x_vars[5]",
    "x[5]": "x_vars[6]",
    "x[6]": "x_vars[7]",
    "x[7]": "x_vars[8]",
    "x[8]": "x_vars[9]",
    "x[9]": "x_vars[10]",
    "x[10]": "x_vars[11]",
    "x[11]": "x_vars[12]",
    "x[12]": "x_vars[13]",
    "x[13]": "x_vars[14]",
    "x[14]": "x_vars[15]",
    "x[15]": "x_vars[16]",
    "x[16]": "x_vars[17]",
    "x[17]": "x_vars[18]",
    "x[18]": "x_vars[19]",
    "x[19]": "x_vars[20]",
    "x[20]": "x_vars[21]",
    "x[21]": "x_vars[22]",
    "x[22]": "x_vars[23]",
    "x[23]": "x_vars[24]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-073": {
   "source_code_sha256": "e5c5be6c1f6ab329e67243610a7c3036c0248147000ec8074b96e0ab0a67e16c",
   "original_solution_sha256": "0c319d4ad78c5c6f3124ac3159cfc97b0c236d834fee54f0fc96f2f74348c349",
   "bindings": [
    [
     "x_I_A2",
     "x_I_A2"
    ],
    [
     "x_I_A1",
     "x_I_A1"
    ],
    [
     "x_I_B1",
     "x_I_B1"
    ],
    [
     "x_I_B2",
     "x_I_B2"
    ],
    [
     "x_I_B3",
     "x_I_B3"
    ],
    [
     "x_II_A2",
     "x_II_A2"
    ],
    [
     "x_II_A1",
     "x_II_A1"
    ],
    [
     "x_II_B1",
     "x_II_B1"
    ],
    [
     "x_III_A2",
     "x_III_A2"
    ],
    [
     "x_III_B2",
     "x_III_B2"
    ]
   ],
   "primary_mapping": {
    "x[A1,I]": "x_I_A1",
    "x[A2,II]": "x_II_A2",
    "x[A2,III]": "x_III_A2",
    "x[B1,I]": "x_I_B1",
    "x[B2,III]": "x_III_B2",
    "x[B3,I]": "x_I_B3",
    "x[A1,II]": "x_II_A1",
    "x[A2,I]": "x_I_A2",
    "x[B1,II]": "x_II_B1",
    "x[B2,I]": "x_I_B2"
   },
   "mapping_basis": "frozen_automatic_name_axis_projection"
  },
  "OR-074": {
   "source_code_sha256": "40fd636eeb343143881ae3ab30c729844d61ce2e6149b3127ff433d55f1ce4b2",
   "original_solution_sha256": "bdecbccb5c8bc776e31b1be37e3606d424d7f2fa947ba920f4190f53b33bfa51",
   "bindings": [
    [
     "C0",
     "x_vars[0]"
    ],
    [
     "C1",
     "x_vars[1]"
    ],
    [
     "C2",
     "x_vars[2]"
    ],
    [
     "C3",
     "x_vars[3]"
    ],
    [
     "C4",
     "x_vars[4]"
    ],
    [
     "C5",
     "x_vars[5]"
    ],
    [
     "C6",
     "x_vars[6]"
    ],
    [
     "C7",
     "x_vars[7]"
    ],
    [
     "C8",
     "x_vars[8]"
    ],
    [
     "C9",
     "x_vars[9]"
    ],
    [
     "C10",
     "x_vars[10]"
    ],
    [
     "C11",
     "x_vars[11]"
    ],
    [
     "C12",
     "x_vars[12]"
    ],
    [
     "C13",
     "x_vars[13]"
    ],
    [
     "C14",
     "x_vars[14]"
    ],
    [
     "C15",
     "x_vars[15]"
    ],
    [
     "C16",
     "x_vars[16]"
    ],
    [
     "C17",
     "x_vars[17]"
    ],
    [
     "C18",
     "x_vars[18]"
    ],
    [
     "C19",
     "x_vars[19]"
    ],
    [
     "C20",
     "x_vars[20]"
    ],
    [
     "C21",
     "x_vars[21]"
    ],
    [
     "C22",
     "x_vars[22]"
    ],
    [
     "C23",
     "x_vars[23]"
    ],
    [
     "C24",
     "x_vars[24]"
    ],
    [
     "C25",
     "x_vars[25]"
    ],
    [
     "C26",
     "x_vars[26]"
    ],
    [
     "C27",
     "x_vars[27]"
    ],
    [
     "C28",
     "x_vars[28]"
    ],
    [
     "C29",
     "x_vars[29]"
    ],
    [
     "C30",
     "x_vars[30]"
    ],
    [
     "C31",
     "x_vars[31]"
    ],
    [
     "C32",
     "x_vars[32]"
    ],
    [
     "C33",
     "x_vars[33]"
    ],
    [
     "C34",
     "x_vars[34]"
    ],
    [
     "C35",
     "x_vars[35]"
    ],
    [
     "C36",
     "x_vars[36]"
    ],
    [
     "C37",
     "x_vars[37]"
    ],
    [
     "C38",
     "x_vars[38]"
    ],
    [
     "C39",
     "x_vars[39]"
    ],
    [
     "C40",
     "x_vars[40]"
    ],
    [
     "C41",
     "x_vars[41]"
    ],
    [
     "C42",
     "x_vars[42]"
    ],
    [
     "C43",
     "x_vars[43]"
    ],
    [
     "C44",
     "x_vars[44]"
    ],
    [
     "C45",
     "x_vars[45]"
    ],
    [
     "C46",
     "x_vars[46]"
    ],
    [
     "C47",
     "x_vars[47]"
    ]
   ],
   "primary_mapping": {
    "x[0]": "x_vars[0]",
    "x[1]": "x_vars[1]",
    "x[2]": "x_vars[2]",
    "x[3]": "x_vars[3]",
    "x[4]": "x_vars[4]",
    "x[5]": "x_vars[5]",
    "x[6]": "x_vars[6]",
    "x[7]": "x_vars[7]",
    "x[8]": "x_vars[8]",
    "x[9]": "x_vars[9]",
    "x[10]": "x_vars[10]",
    "x[11]": "x_vars[11]",
    "x[12]": "x_vars[12]",
    "x[13]": "x_vars[13]",
    "x[14]": "x_vars[14]",
    "x[15]": "x_vars[15]",
    "x[16]": "x_vars[16]",
    "x[17]": "x_vars[17]",
    "x[18]": "x_vars[18]",
    "x[19]": "x_vars[19]",
    "x[20]": "x_vars[20]",
    "x[21]": "x_vars[21]",
    "x[22]": "x_vars[22]",
    "x[23]": "x_vars[23]",
    "x[24]": "x_vars[24]",
    "x[25]": "x_vars[25]",
    "x[26]": "x_vars[26]",
    "x[27]": "x_vars[27]",
    "x[28]": "x_vars[28]",
    "x[29]": "x_vars[29]",
    "x[30]": "x_vars[30]",
    "x[31]": "x_vars[31]",
    "x[32]": "x_vars[32]",
    "x[33]": "x_vars[33]",
    "x[34]": "x_vars[34]",
    "x[35]": "x_vars[35]",
    "x[36]": "x_vars[36]",
    "x[37]": "x_vars[37]",
    "x[38]": "x_vars[38]",
    "x[39]": "x_vars[39]",
    "x[40]": "x_vars[40]",
    "x[41]": "x_vars[41]",
    "x[42]": "x_vars[42]",
    "x[43]": "x_vars[43]",
    "x[44]": "x_vars[44]",
    "x[45]": "x_vars[45]",
    "x[46]": "x_vars[46]",
    "x[47]": "x_vars[47]"
   },
   "mapping_basis": "frozen_automatic_name_axis_projection"
  },
  "OR-075": {
   "source_code_sha256": "dd2e4878261a9744a7b6584b146f18644cbc9d82d6cac898ef39a7dcc85b42e3",
   "original_solution_sha256": "f66d7578c64f979d4379b5f3211192585a5f4f68874a8edb433e6daef5bcc4cf",
   "bindings": [
    [
     "C0",
     "x_vars[1]"
    ],
    [
     "C1",
     "x_vars[2]"
    ],
    [
     "C2",
     "x_vars[3]"
    ],
    [
     "C3",
     "x_vars[4]"
    ],
    [
     "C4",
     "x_vars[5]"
    ],
    [
     "C5",
     "x_vars[6]"
    ],
    [
     "C6",
     "x_vars[7]"
    ],
    [
     "C7",
     "x_vars[8]"
    ],
    [
     "C8",
     "x_vars[9]"
    ],
    [
     "C9",
     "x_vars[10]"
    ],
    [
     "C10",
     "x_vars[11]"
    ],
    [
     "C11",
     "x_vars[12]"
    ],
    [
     "C12",
     "x_vars[13]"
    ],
    [
     "C13",
     "x_vars[14]"
    ],
    [
     "C14",
     "x_vars[15]"
    ],
    [
     "C15",
     "x_vars[16]"
    ],
    [
     "C16",
     "x_vars[17]"
    ],
    [
     "C17",
     "x_vars[18]"
    ],
    [
     "C18",
     "x_vars[19]"
    ],
    [
     "C19",
     "x_vars[20]"
    ],
    [
     "C20",
     "x_vars[21]"
    ],
    [
     "C21",
     "x_vars[22]"
    ],
    [
     "C22",
     "x_vars[23]"
    ],
    [
     "C23",
     "x_vars[24]"
    ],
    [
     "C24",
     "x_vars[25]"
    ],
    [
     "C25",
     "x_vars[26]"
    ],
    [
     "C26",
     "x_vars[27]"
    ],
    [
     "C27",
     "x_vars[28]"
    ],
    [
     "C28",
     "x_vars[29]"
    ],
    [
     "C29",
     "x_vars[30]"
    ],
    [
     "C30",
     "x_vars[31]"
    ],
    [
     "C31",
     "x_vars[32]"
    ],
    [
     "C32",
     "x_vars[33]"
    ],
    [
     "C33",
     "x_vars[34]"
    ],
    [
     "C34",
     "x_vars[35]"
    ],
    [
     "C35",
     "x_vars[36]"
    ],
    [
     "C36",
     "x_vars[37]"
    ],
    [
     "C37",
     "x_vars[38]"
    ],
    [
     "C38",
     "x_vars[39]"
    ],
    [
     "C39",
     "x_vars[40]"
    ],
    [
     "C40",
     "x_vars[41]"
    ],
    [
     "C41",
     "x_vars[42]"
    ],
    [
     "C42",
     "x_vars[43]"
    ],
    [
     "C43",
     "x_vars[44]"
    ],
    [
     "C44",
     "x_vars[45]"
    ],
    [
     "C45",
     "x_vars[46]"
    ],
    [
     "C46",
     "x_vars[47]"
    ],
    [
     "C47",
     "x_vars[48]"
    ],
    [
     "C48",
     "x_vars[49]"
    ],
    [
     "C49",
     "x_vars[50]"
    ],
    [
     "C50",
     "x_vars[51]"
    ],
    [
     "C51",
     "x_vars[52]"
    ],
    [
     "C52",
     "x_vars[53]"
    ],
    [
     "C53",
     "x_vars[54]"
    ],
    [
     "C54",
     "x_vars[55]"
    ],
    [
     "C55",
     "x_vars[56]"
    ],
    [
     "C56",
     "x_vars[57]"
    ],
    [
     "C57",
     "x_vars[58]"
    ],
    [
     "C58",
     "x_vars[59]"
    ],
    [
     "C59",
     "x_vars[60]"
    ],
    [
     "C60",
     "x_vars[61]"
    ],
    [
     "C61",
     "x_vars[62]"
    ],
    [
     "C62",
     "x_vars[63]"
    ],
    [
     "C63",
     "x_vars[64]"
    ],
    [
     "C64",
     "x_vars[65]"
    ],
    [
     "C65",
     "x_vars[66]"
    ],
    [
     "C66",
     "x_vars[67]"
    ],
    [
     "C67",
     "x_vars[68]"
    ],
    [
     "C68",
     "x_vars[69]"
    ],
    [
     "C69",
     "x_vars[70]"
    ],
    [
     "C70",
     "x_vars[71]"
    ],
    [
     "C71",
     "x_vars[72]"
    ],
    [
     "C72",
     "x_vars[73]"
    ],
    [
     "C73",
     "x_vars[74]"
    ],
    [
     "C74",
     "x_vars[75]"
    ],
    [
     "C75",
     "x_vars[76]"
    ],
    [
     "C76",
     "x_vars[77]"
    ],
    [
     "C77",
     "x_vars[78]"
    ],
    [
     "C78",
     "x_vars[79]"
    ],
    [
     "C79",
     "x_vars[80]"
    ],
    [
     "C80",
     "x_vars[81]"
    ],
    [
     "C81",
     "x_vars[82]"
    ],
    [
     "C82",
     "x_vars[83]"
    ],
    [
     "C83",
     "x_vars[84]"
    ],
    [
     "C84",
     "x_vars[85]"
    ],
    [
     "C85",
     "x_vars[86]"
    ],
    [
     "C86",
     "x_vars[87]"
    ],
    [
     "C87",
     "x_vars[88]"
    ],
    [
     "C88",
     "x_vars[89]"
    ],
    [
     "C89",
     "x_vars[90]"
    ],
    [
     "C90",
     "x_vars[91]"
    ],
    [
     "C91",
     "x_vars[92]"
    ],
    [
     "C92",
     "x_vars[93]"
    ],
    [
     "C93",
     "x_vars[94]"
    ],
    [
     "C94",
     "x_vars[95]"
    ],
    [
     "C95",
     "x_vars[96]"
    ],
    [
     "C96",
     "x_vars[97]"
    ],
    [
     "C97",
     "x_vars[98]"
    ],
    [
     "C98",
     "x_vars[99]"
    ],
    [
     "C99",
     "x_vars[100]"
    ],
    [
     "C100",
     "x_vars[101]"
    ],
    [
     "C101",
     "x_vars[102]"
    ],
    [
     "C102",
     "x_vars[103]"
    ],
    [
     "C103",
     "x_vars[104]"
    ],
    [
     "C104",
     "x_vars[105]"
    ],
    [
     "C105",
     "x_vars[106]"
    ],
    [
     "C106",
     "x_vars[107]"
    ],
    [
     "C107",
     "x_vars[108]"
    ],
    [
     "C108",
     "x_vars[109]"
    ],
    [
     "C109",
     "x_vars[110]"
    ]
   ],
   "primary_mapping": {
    "y[1]": "x_vars[1]",
    "y[2]": "x_vars[2]",
    "y[3]": "x_vars[3]",
    "y[4]": "x_vars[4]",
    "y[5]": "x_vars[5]",
    "y[6]": "x_vars[6]",
    "y[7]": "x_vars[7]",
    "y[8]": "x_vars[8]",
    "y[9]": "x_vars[9]",
    "y[10]": "x_vars[10]",
    "y[11]": "x_vars[11]",
    "y[12]": "x_vars[12]",
    "y[13]": "x_vars[13]",
    "y[14]": "x_vars[14]",
    "y[15]": "x_vars[15]",
    "y[16]": "x_vars[16]",
    "y[17]": "x_vars[17]",
    "y[18]": "x_vars[18]",
    "y[19]": "x_vars[19]",
    "y[20]": "x_vars[20]",
    "y[21]": "x_vars[21]",
    "y[22]": "x_vars[22]",
    "y[23]": "x_vars[23]",
    "y[24]": "x_vars[24]",
    "y[25]": "x_vars[25]",
    "y[26]": "x_vars[26]",
    "y[27]": "x_vars[27]",
    "y[28]": "x_vars[28]",
    "y[29]": "x_vars[29]",
    "y[30]": "x_vars[30]",
    "y[31]": "x_vars[31]",
    "y[32]": "x_vars[32]",
    "y[33]": "x_vars[33]",
    "y[34]": "x_vars[34]",
    "y[35]": "x_vars[35]",
    "y[36]": "x_vars[36]",
    "y[37]": "x_vars[37]",
    "y[38]": "x_vars[38]",
    "y[39]": "x_vars[39]",
    "y[40]": "x_vars[40]",
    "y[41]": "x_vars[41]",
    "y[42]": "x_vars[42]",
    "y[43]": "x_vars[43]",
    "y[44]": "x_vars[44]",
    "y[45]": "x_vars[45]",
    "y[46]": "x_vars[46]",
    "y[47]": "x_vars[47]",
    "y[48]": "x_vars[48]",
    "y[49]": "x_vars[49]",
    "y[50]": "x_vars[50]",
    "y[51]": "x_vars[51]",
    "y[52]": "x_vars[52]",
    "y[53]": "x_vars[53]",
    "y[54]": "x_vars[54]",
    "y[55]": "x_vars[55]",
    "y[56]": "x_vars[56]",
    "y[57]": "x_vars[57]",
    "y[58]": "x_vars[58]",
    "y[59]": "x_vars[59]",
    "y[60]": "x_vars[60]",
    "y[61]": "x_vars[61]",
    "y[62]": "x_vars[62]",
    "y[63]": "x_vars[63]",
    "y[64]": "x_vars[64]",
    "y[65]": "x_vars[65]",
    "y[66]": "x_vars[66]",
    "y[67]": "x_vars[67]",
    "y[68]": "x_vars[68]",
    "y[69]": "x_vars[69]",
    "y[70]": "x_vars[70]",
    "y[71]": "x_vars[71]",
    "y[72]": "x_vars[72]",
    "y[73]": "x_vars[73]",
    "y[74]": "x_vars[74]",
    "y[75]": "x_vars[75]",
    "y[76]": "x_vars[76]",
    "y[77]": "x_vars[77]",
    "y[78]": "x_vars[78]",
    "y[79]": "x_vars[79]",
    "y[80]": "x_vars[80]",
    "y[81]": "x_vars[81]",
    "y[82]": "x_vars[82]",
    "y[83]": "x_vars[83]",
    "y[84]": "x_vars[84]",
    "y[85]": "x_vars[85]",
    "y[86]": "x_vars[86]",
    "y[87]": "x_vars[87]",
    "y[88]": "x_vars[88]",
    "y[89]": "x_vars[89]",
    "y[90]": "x_vars[90]",
    "y[91]": "x_vars[91]",
    "y[92]": "x_vars[92]",
    "y[93]": "x_vars[93]",
    "y[94]": "x_vars[94]",
    "y[95]": "x_vars[95]",
    "y[96]": "x_vars[96]",
    "y[97]": "x_vars[97]",
    "y[98]": "x_vars[98]",
    "y[99]": "x_vars[99]",
    "y[100]": "x_vars[100]",
    "y[101]": "x_vars[101]",
    "y[102]": "x_vars[102]",
    "y[103]": "x_vars[103]",
    "y[104]": "x_vars[104]",
    "y[105]": "x_vars[105]",
    "y[106]": "x_vars[106]",
    "y[107]": "x_vars[107]",
    "y[108]": "x_vars[108]",
    "y[109]": "x_vars[109]",
    "y[110]": "x_vars[110]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-076": {
   "source_code_sha256": "8586b246c8b3735974414f9143dd6e45dd8894cc15fbbd79a8982f64c5faa447",
   "original_solution_sha256": "e6aa3a71e03ae9a888785a19494feb3d9c0eafeb3531f4229213a9738efe876f",
   "bindings": [
    [
     "C0",
     "x_vars[W1,C1]"
    ],
    [
     "C1",
     "x_vars[W1,C2]"
    ],
    [
     "C2",
     "x_vars[W1,C3]"
    ],
    [
     "C3",
     "x_vars[W1,C4]"
    ],
    [
     "C4",
     "x_vars[W1,C5]"
    ],
    [
     "C5",
     "x_vars[W1,C6]"
    ],
    [
     "C6",
     "x_vars[W1,C7]"
    ],
    [
     "C7",
     "x_vars[W1,C8]"
    ],
    [
     "C8",
     "x_vars[W1,C9]"
    ],
    [
     "C9",
     "x_vars[W1,C10]"
    ],
    [
     "C10",
     "x_vars[W1,C11]"
    ],
    [
     "C11",
     "x_vars[W1,C12]"
    ],
    [
     "C12",
     "x_vars[W1,C13]"
    ],
    [
     "C13",
     "x_vars[W1,C14]"
    ],
    [
     "C14",
     "x_vars[W1,C15]"
    ],
    [
     "C15",
     "x_vars[W1,C16]"
    ],
    [
     "C16",
     "x_vars[W1,C17]"
    ],
    [
     "C17",
     "x_vars[W1,C18]"
    ],
    [
     "C18",
     "x_vars[W1,C19]"
    ],
    [
     "C19",
     "x_vars[W1,C20]"
    ],
    [
     "C20",
     "x_vars[W2,C1]"
    ],
    [
     "C21",
     "x_vars[W2,C2]"
    ],
    [
     "C22",
     "x_vars[W2,C3]"
    ],
    [
     "C23",
     "x_vars[W2,C4]"
    ],
    [
     "C24",
     "x_vars[W2,C5]"
    ],
    [
     "C25",
     "x_vars[W2,C6]"
    ],
    [
     "C26",
     "x_vars[W2,C7]"
    ],
    [
     "C27",
     "x_vars[W2,C8]"
    ],
    [
     "C28",
     "x_vars[W2,C9]"
    ],
    [
     "C29",
     "x_vars[W2,C10]"
    ],
    [
     "C30",
     "x_vars[W2,C11]"
    ],
    [
     "C31",
     "x_vars[W2,C12]"
    ],
    [
     "C32",
     "x_vars[W2,C13]"
    ],
    [
     "C33",
     "x_vars[W2,C14]"
    ],
    [
     "C34",
     "x_vars[W2,C15]"
    ],
    [
     "C35",
     "x_vars[W2,C16]"
    ],
    [
     "C36",
     "x_vars[W2,C17]"
    ],
    [
     "C37",
     "x_vars[W2,C18]"
    ],
    [
     "C38",
     "x_vars[W2,C19]"
    ],
    [
     "C39",
     "x_vars[W2,C20]"
    ],
    [
     "C40",
     "x_vars[W3,C1]"
    ],
    [
     "C41",
     "x_vars[W3,C2]"
    ],
    [
     "C42",
     "x_vars[W3,C3]"
    ],
    [
     "C43",
     "x_vars[W3,C4]"
    ],
    [
     "C44",
     "x_vars[W3,C5]"
    ],
    [
     "C45",
     "x_vars[W3,C6]"
    ],
    [
     "C46",
     "x_vars[W3,C7]"
    ],
    [
     "C47",
     "x_vars[W3,C8]"
    ],
    [
     "C48",
     "x_vars[W3,C9]"
    ],
    [
     "C49",
     "x_vars[W3,C10]"
    ],
    [
     "C50",
     "x_vars[W3,C11]"
    ],
    [
     "C51",
     "x_vars[W3,C12]"
    ],
    [
     "C52",
     "x_vars[W3,C13]"
    ],
    [
     "C53",
     "x_vars[W3,C14]"
    ],
    [
     "C54",
     "x_vars[W3,C15]"
    ],
    [
     "C55",
     "x_vars[W3,C16]"
    ],
    [
     "C56",
     "x_vars[W3,C17]"
    ],
    [
     "C57",
     "x_vars[W3,C18]"
    ],
    [
     "C58",
     "x_vars[W3,C19]"
    ],
    [
     "C59",
     "x_vars[W3,C20]"
    ],
    [
     "C60",
     "x_vars[W4,C1]"
    ],
    [
     "C61",
     "x_vars[W4,C2]"
    ],
    [
     "C62",
     "x_vars[W4,C3]"
    ],
    [
     "C63",
     "x_vars[W4,C4]"
    ],
    [
     "C64",
     "x_vars[W4,C5]"
    ],
    [
     "C65",
     "x_vars[W4,C6]"
    ],
    [
     "C66",
     "x_vars[W4,C7]"
    ],
    [
     "C67",
     "x_vars[W4,C8]"
    ],
    [
     "C68",
     "x_vars[W4,C9]"
    ],
    [
     "C69",
     "x_vars[W4,C10]"
    ],
    [
     "C70",
     "x_vars[W4,C11]"
    ],
    [
     "C71",
     "x_vars[W4,C12]"
    ],
    [
     "C72",
     "x_vars[W4,C13]"
    ],
    [
     "C73",
     "x_vars[W4,C14]"
    ],
    [
     "C74",
     "x_vars[W4,C15]"
    ],
    [
     "C75",
     "x_vars[W4,C16]"
    ],
    [
     "C76",
     "x_vars[W4,C17]"
    ],
    [
     "C77",
     "x_vars[W4,C18]"
    ],
    [
     "C78",
     "x_vars[W4,C19]"
    ],
    [
     "C79",
     "x_vars[W4,C20]"
    ],
    [
     "C80",
     "x_vars[W5,C1]"
    ],
    [
     "C81",
     "x_vars[W5,C2]"
    ],
    [
     "C82",
     "x_vars[W5,C3]"
    ],
    [
     "C83",
     "x_vars[W5,C4]"
    ],
    [
     "C84",
     "x_vars[W5,C5]"
    ],
    [
     "C85",
     "x_vars[W5,C6]"
    ],
    [
     "C86",
     "x_vars[W5,C7]"
    ],
    [
     "C87",
     "x_vars[W5,C8]"
    ],
    [
     "C88",
     "x_vars[W5,C9]"
    ],
    [
     "C89",
     "x_vars[W5,C10]"
    ],
    [
     "C90",
     "x_vars[W5,C11]"
    ],
    [
     "C91",
     "x_vars[W5,C12]"
    ],
    [
     "C92",
     "x_vars[W5,C13]"
    ],
    [
     "C93",
     "x_vars[W5,C14]"
    ],
    [
     "C94",
     "x_vars[W5,C15]"
    ],
    [
     "C95",
     "x_vars[W5,C16]"
    ],
    [
     "C96",
     "x_vars[W5,C17]"
    ],
    [
     "C97",
     "x_vars[W5,C18]"
    ],
    [
     "C98",
     "x_vars[W5,C19]"
    ],
    [
     "C99",
     "x_vars[W5,C20]"
    ],
    [
     "C100",
     "x_vars[W6,C1]"
    ],
    [
     "C101",
     "x_vars[W6,C2]"
    ],
    [
     "C102",
     "x_vars[W6,C3]"
    ],
    [
     "C103",
     "x_vars[W6,C4]"
    ],
    [
     "C104",
     "x_vars[W6,C5]"
    ],
    [
     "C105",
     "x_vars[W6,C6]"
    ],
    [
     "C106",
     "x_vars[W6,C7]"
    ],
    [
     "C107",
     "x_vars[W6,C8]"
    ],
    [
     "C108",
     "x_vars[W6,C9]"
    ],
    [
     "C109",
     "x_vars[W6,C10]"
    ],
    [
     "C110",
     "x_vars[W6,C11]"
    ],
    [
     "C111",
     "x_vars[W6,C12]"
    ],
    [
     "C112",
     "x_vars[W6,C13]"
    ],
    [
     "C113",
     "x_vars[W6,C14]"
    ],
    [
     "C114",
     "x_vars[W6,C15]"
    ],
    [
     "C115",
     "x_vars[W6,C16]"
    ],
    [
     "C116",
     "x_vars[W6,C17]"
    ],
    [
     "C117",
     "x_vars[W6,C18]"
    ],
    [
     "C118",
     "x_vars[W6,C19]"
    ],
    [
     "C119",
     "x_vars[W6,C20]"
    ],
    [
     "C120",
     "x_vars[W7,C1]"
    ],
    [
     "C121",
     "x_vars[W7,C2]"
    ],
    [
     "C122",
     "x_vars[W7,C3]"
    ],
    [
     "C123",
     "x_vars[W7,C4]"
    ],
    [
     "C124",
     "x_vars[W7,C5]"
    ],
    [
     "C125",
     "x_vars[W7,C6]"
    ],
    [
     "C126",
     "x_vars[W7,C7]"
    ],
    [
     "C127",
     "x_vars[W7,C8]"
    ],
    [
     "C128",
     "x_vars[W7,C9]"
    ],
    [
     "C129",
     "x_vars[W7,C10]"
    ],
    [
     "C130",
     "x_vars[W7,C11]"
    ],
    [
     "C131",
     "x_vars[W7,C12]"
    ],
    [
     "C132",
     "x_vars[W7,C13]"
    ],
    [
     "C133",
     "x_vars[W7,C14]"
    ],
    [
     "C134",
     "x_vars[W7,C15]"
    ],
    [
     "C135",
     "x_vars[W7,C16]"
    ],
    [
     "C136",
     "x_vars[W7,C17]"
    ],
    [
     "C137",
     "x_vars[W7,C18]"
    ],
    [
     "C138",
     "x_vars[W7,C19]"
    ],
    [
     "C139",
     "x_vars[W7,C20]"
    ],
    [
     "C140",
     "x_vars[W8,C1]"
    ],
    [
     "C141",
     "x_vars[W8,C2]"
    ],
    [
     "C142",
     "x_vars[W8,C3]"
    ],
    [
     "C143",
     "x_vars[W8,C4]"
    ],
    [
     "C144",
     "x_vars[W8,C5]"
    ],
    [
     "C145",
     "x_vars[W8,C6]"
    ],
    [
     "C146",
     "x_vars[W8,C7]"
    ],
    [
     "C147",
     "x_vars[W8,C8]"
    ],
    [
     "C148",
     "x_vars[W8,C9]"
    ],
    [
     "C149",
     "x_vars[W8,C10]"
    ],
    [
     "C150",
     "x_vars[W8,C11]"
    ],
    [
     "C151",
     "x_vars[W8,C12]"
    ],
    [
     "C152",
     "x_vars[W8,C13]"
    ],
    [
     "C153",
     "x_vars[W8,C14]"
    ],
    [
     "C154",
     "x_vars[W8,C15]"
    ],
    [
     "C155",
     "x_vars[W8,C16]"
    ],
    [
     "C156",
     "x_vars[W8,C17]"
    ],
    [
     "C157",
     "x_vars[W8,C18]"
    ],
    [
     "C158",
     "x_vars[W8,C19]"
    ],
    [
     "C159",
     "x_vars[W8,C20]"
    ],
    [
     "C160",
     "x_vars[W9,C1]"
    ],
    [
     "C161",
     "x_vars[W9,C2]"
    ],
    [
     "C162",
     "x_vars[W9,C3]"
    ],
    [
     "C163",
     "x_vars[W9,C4]"
    ],
    [
     "C164",
     "x_vars[W9,C5]"
    ],
    [
     "C165",
     "x_vars[W9,C6]"
    ],
    [
     "C166",
     "x_vars[W9,C7]"
    ],
    [
     "C167",
     "x_vars[W9,C8]"
    ],
    [
     "C168",
     "x_vars[W9,C9]"
    ],
    [
     "C169",
     "x_vars[W9,C10]"
    ],
    [
     "C170",
     "x_vars[W9,C11]"
    ],
    [
     "C171",
     "x_vars[W9,C12]"
    ],
    [
     "C172",
     "x_vars[W9,C13]"
    ],
    [
     "C173",
     "x_vars[W9,C14]"
    ],
    [
     "C174",
     "x_vars[W9,C15]"
    ],
    [
     "C175",
     "x_vars[W9,C16]"
    ],
    [
     "C176",
     "x_vars[W9,C17]"
    ],
    [
     "C177",
     "x_vars[W9,C18]"
    ],
    [
     "C178",
     "x_vars[W9,C19]"
    ],
    [
     "C179",
     "x_vars[W9,C20]"
    ],
    [
     "C180",
     "x_vars[W10,C1]"
    ],
    [
     "C181",
     "x_vars[W10,C2]"
    ],
    [
     "C182",
     "x_vars[W10,C3]"
    ],
    [
     "C183",
     "x_vars[W10,C4]"
    ],
    [
     "C184",
     "x_vars[W10,C5]"
    ],
    [
     "C185",
     "x_vars[W10,C6]"
    ],
    [
     "C186",
     "x_vars[W10,C7]"
    ],
    [
     "C187",
     "x_vars[W10,C8]"
    ],
    [
     "C188",
     "x_vars[W10,C9]"
    ],
    [
     "C189",
     "x_vars[W10,C10]"
    ],
    [
     "C190",
     "x_vars[W10,C11]"
    ],
    [
     "C191",
     "x_vars[W10,C12]"
    ],
    [
     "C192",
     "x_vars[W10,C13]"
    ],
    [
     "C193",
     "x_vars[W10,C14]"
    ],
    [
     "C194",
     "x_vars[W10,C15]"
    ],
    [
     "C195",
     "x_vars[W10,C16]"
    ],
    [
     "C196",
     "x_vars[W10,C17]"
    ],
    [
     "C197",
     "x_vars[W10,C18]"
    ],
    [
     "C198",
     "x_vars[W10,C19]"
    ],
    [
     "C199",
     "x_vars[W10,C20]"
    ],
    [
     "C200",
     "y_vars[W1]"
    ],
    [
     "C201",
     "y_vars[W2]"
    ],
    [
     "C202",
     "y_vars[W3]"
    ],
    [
     "C203",
     "y_vars[W4]"
    ],
    [
     "C204",
     "y_vars[W5]"
    ],
    [
     "C205",
     "y_vars[W6]"
    ],
    [
     "C206",
     "y_vars[W7]"
    ],
    [
     "C207",
     "y_vars[W8]"
    ],
    [
     "C208",
     "y_vars[W9]"
    ],
    [
     "C209",
     "y_vars[W10]"
    ]
   ],
   "primary_mapping": {
    "y[W1]": "y_vars[W1]",
    "y[W2]": "y_vars[W2]",
    "y[W3]": "y_vars[W3]",
    "y[W4]": "y_vars[W4]",
    "y[W5]": "y_vars[W5]",
    "y[W6]": "y_vars[W6]",
    "y[W7]": "y_vars[W7]",
    "y[W8]": "y_vars[W8]",
    "y[W9]": "y_vars[W9]",
    "y[W10]": "y_vars[W10]",
    "s[W1,C1]": "x_vars[W1,C1]",
    "s[W1,C11]": "x_vars[W1,C11]",
    "s[W1,C12]": "x_vars[W1,C12]",
    "s[W1,C13]": "x_vars[W1,C13]",
    "s[W1,C14]": "x_vars[W1,C14]",
    "s[W1,C15]": "x_vars[W1,C15]",
    "s[W1,C16]": "x_vars[W1,C16]",
    "s[W1,C17]": "x_vars[W1,C17]",
    "s[W1,C18]": "x_vars[W1,C18]",
    "s[W1,C19]": "x_vars[W1,C19]",
    "s[W1,C20]": "x_vars[W1,C20]",
    "s[W2,C2]": "x_vars[W2,C2]",
    "s[W2,C11]": "x_vars[W2,C11]",
    "s[W2,C12]": "x_vars[W2,C12]",
    "s[W2,C13]": "x_vars[W2,C13]",
    "s[W2,C14]": "x_vars[W2,C14]",
    "s[W2,C15]": "x_vars[W2,C15]",
    "s[W2,C16]": "x_vars[W2,C16]",
    "s[W2,C17]": "x_vars[W2,C17]",
    "s[W2,C18]": "x_vars[W2,C18]",
    "s[W2,C19]": "x_vars[W2,C19]",
    "s[W2,C20]": "x_vars[W2,C20]",
    "s[W3,C3]": "x_vars[W3,C3]",
    "s[W3,C11]": "x_vars[W3,C11]",
    "s[W3,C12]": "x_vars[W3,C12]",
    "s[W3,C13]": "x_vars[W3,C13]",
    "s[W3,C14]": "x_vars[W3,C14]",
    "s[W3,C15]": "x_vars[W3,C15]",
    "s[W3,C16]": "x_vars[W3,C16]",
    "s[W3,C17]": "x_vars[W3,C17]",
    "s[W3,C18]": "x_vars[W3,C18]",
    "s[W3,C19]": "x_vars[W3,C19]",
    "s[W3,C20]": "x_vars[W3,C20]",
    "s[W4,C4]": "x_vars[W4,C4]",
    "s[W4,C11]": "x_vars[W4,C11]",
    "s[W4,C12]": "x_vars[W4,C12]",
    "s[W4,C13]": "x_vars[W4,C13]",
    "s[W4,C14]": "x_vars[W4,C14]",
    "s[W4,C15]": "x_vars[W4,C15]",
    "s[W4,C16]": "x_vars[W4,C16]",
    "s[W4,C17]": "x_vars[W4,C17]",
    "s[W4,C18]": "x_vars[W4,C18]",
    "s[W4,C19]": "x_vars[W4,C19]",
    "s[W4,C20]": "x_vars[W4,C20]",
    "s[W5,C5]": "x_vars[W5,C5]",
    "s[W5,C11]": "x_vars[W5,C11]",
    "s[W5,C12]": "x_vars[W5,C12]",
    "s[W5,C13]": "x_vars[W5,C13]",
    "s[W5,C14]": "x_vars[W5,C14]",
    "s[W5,C15]": "x_vars[W5,C15]",
    "s[W5,C16]": "x_vars[W5,C16]",
    "s[W5,C17]": "x_vars[W5,C17]",
    "s[W5,C18]": "x_vars[W5,C18]",
    "s[W5,C19]": "x_vars[W5,C19]",
    "s[W5,C20]": "x_vars[W5,C20]",
    "s[W6,C6]": "x_vars[W6,C6]",
    "s[W6,C11]": "x_vars[W6,C11]",
    "s[W6,C12]": "x_vars[W6,C12]",
    "s[W6,C13]": "x_vars[W6,C13]",
    "s[W6,C14]": "x_vars[W6,C14]",
    "s[W6,C15]": "x_vars[W6,C15]",
    "s[W6,C16]": "x_vars[W6,C16]",
    "s[W6,C17]": "x_vars[W6,C17]",
    "s[W6,C18]": "x_vars[W6,C18]",
    "s[W6,C19]": "x_vars[W6,C19]",
    "s[W6,C20]": "x_vars[W6,C20]",
    "s[W7,C7]": "x_vars[W7,C7]",
    "s[W7,C11]": "x_vars[W7,C11]",
    "s[W7,C12]": "x_vars[W7,C12]",
    "s[W7,C13]": "x_vars[W7,C13]",
    "s[W7,C14]": "x_vars[W7,C14]",
    "s[W7,C15]": "x_vars[W7,C15]",
    "s[W7,C16]": "x_vars[W7,C16]",
    "s[W7,C17]": "x_vars[W7,C17]",
    "s[W7,C18]": "x_vars[W7,C18]",
    "s[W7,C19]": "x_vars[W7,C19]",
    "s[W7,C20]": "x_vars[W7,C20]",
    "s[W8,C8]": "x_vars[W8,C8]",
    "s[W8,C11]": "x_vars[W8,C11]",
    "s[W8,C12]": "x_vars[W8,C12]",
    "s[W8,C13]": "x_vars[W8,C13]",
    "s[W8,C14]": "x_vars[W8,C14]",
    "s[W8,C15]": "x_vars[W8,C15]",
    "s[W8,C16]": "x_vars[W8,C16]",
    "s[W8,C17]": "x_vars[W8,C17]",
    "s[W8,C18]": "x_vars[W8,C18]",
    "s[W8,C19]": "x_vars[W8,C19]",
    "s[W8,C20]": "x_vars[W8,C20]",
    "s[W9,C9]": "x_vars[W9,C9]",
    "s[W9,C11]": "x_vars[W9,C11]",
    "s[W9,C12]": "x_vars[W9,C12]",
    "s[W9,C13]": "x_vars[W9,C13]",
    "s[W9,C14]": "x_vars[W9,C14]",
    "s[W9,C15]": "x_vars[W9,C15]",
    "s[W9,C16]": "x_vars[W9,C16]",
    "s[W9,C17]": "x_vars[W9,C17]",
    "s[W9,C18]": "x_vars[W9,C18]",
    "s[W9,C19]": "x_vars[W9,C19]",
    "s[W9,C20]": "x_vars[W9,C20]",
    "s[W10,C10]": "x_vars[W10,C10]",
    "s[W10,C11]": "x_vars[W10,C11]",
    "s[W10,C12]": "x_vars[W10,C12]",
    "s[W10,C13]": "x_vars[W10,C13]",
    "s[W10,C14]": "x_vars[W10,C14]",
    "s[W10,C15]": "x_vars[W10,C15]",
    "s[W10,C16]": "x_vars[W10,C16]",
    "s[W10,C17]": "x_vars[W10,C17]",
    "s[W10,C18]": "x_vars[W10,C18]",
    "s[W10,C19]": "x_vars[W10,C19]",
    "s[W10,C20]": "x_vars[W10,C20]",
    "s[W1,C2]": "x_vars[W1,C2]",
    "s[W1,C3]": "x_vars[W1,C3]",
    "s[W1,C4]": "x_vars[W1,C4]",
    "s[W1,C5]": "x_vars[W1,C5]",
    "s[W1,C6]": "x_vars[W1,C6]",
    "s[W1,C7]": "x_vars[W1,C7]",
    "s[W1,C8]": "x_vars[W1,C8]",
    "s[W1,C9]": "x_vars[W1,C9]",
    "s[W1,C10]": "x_vars[W1,C10]",
    "s[W2,C1]": "x_vars[W2,C1]",
    "s[W2,C3]": "x_vars[W2,C3]",
    "s[W2,C4]": "x_vars[W2,C4]",
    "s[W2,C5]": "x_vars[W2,C5]",
    "s[W2,C6]": "x_vars[W2,C6]",
    "s[W2,C7]": "x_vars[W2,C7]",
    "s[W2,C8]": "x_vars[W2,C8]",
    "s[W2,C9]": "x_vars[W2,C9]",
    "s[W2,C10]": "x_vars[W2,C10]",
    "s[W3,C1]": "x_vars[W3,C1]",
    "s[W3,C2]": "x_vars[W3,C2]",
    "s[W3,C4]": "x_vars[W3,C4]",
    "s[W3,C5]": "x_vars[W3,C5]",
    "s[W3,C6]": "x_vars[W3,C6]",
    "s[W3,C7]": "x_vars[W3,C7]",
    "s[W3,C8]": "x_vars[W3,C8]",
    "s[W3,C9]": "x_vars[W3,C9]",
    "s[W3,C10]": "x_vars[W3,C10]",
    "s[W4,C1]": "x_vars[W4,C1]",
    "s[W4,C2]": "x_vars[W4,C2]",
    "s[W4,C3]": "x_vars[W4,C3]",
    "s[W4,C5]": "x_vars[W4,C5]",
    "s[W4,C6]": "x_vars[W4,C6]",
    "s[W4,C7]": "x_vars[W4,C7]",
    "s[W4,C8]": "x_vars[W4,C8]",
    "s[W4,C9]": "x_vars[W4,C9]",
    "s[W4,C10]": "x_vars[W4,C10]",
    "s[W5,C1]": "x_vars[W5,C1]",
    "s[W5,C2]": "x_vars[W5,C2]",
    "s[W5,C3]": "x_vars[W5,C3]",
    "s[W5,C4]": "x_vars[W5,C4]",
    "s[W5,C6]": "x_vars[W5,C6]",
    "s[W5,C7]": "x_vars[W5,C7]",
    "s[W5,C8]": "x_vars[W5,C8]",
    "s[W5,C9]": "x_vars[W5,C9]",
    "s[W5,C10]": "x_vars[W5,C10]",
    "s[W6,C1]": "x_vars[W6,C1]",
    "s[W6,C2]": "x_vars[W6,C2]",
    "s[W6,C3]": "x_vars[W6,C3]",
    "s[W6,C4]": "x_vars[W6,C4]",
    "s[W6,C5]": "x_vars[W6,C5]",
    "s[W6,C7]": "x_vars[W6,C7]",
    "s[W6,C8]": "x_vars[W6,C8]",
    "s[W6,C9]": "x_vars[W6,C9]",
    "s[W6,C10]": "x_vars[W6,C10]",
    "s[W7,C1]": "x_vars[W7,C1]",
    "s[W7,C2]": "x_vars[W7,C2]",
    "s[W7,C3]": "x_vars[W7,C3]",
    "s[W7,C4]": "x_vars[W7,C4]",
    "s[W7,C5]": "x_vars[W7,C5]",
    "s[W7,C6]": "x_vars[W7,C6]",
    "s[W7,C8]": "x_vars[W7,C8]",
    "s[W7,C9]": "x_vars[W7,C9]",
    "s[W7,C10]": "x_vars[W7,C10]",
    "s[W8,C1]": "x_vars[W8,C1]",
    "s[W8,C2]": "x_vars[W8,C2]",
    "s[W8,C3]": "x_vars[W8,C3]",
    "s[W8,C4]": "x_vars[W8,C4]",
    "s[W8,C5]": "x_vars[W8,C5]",
    "s[W8,C6]": "x_vars[W8,C6]",
    "s[W8,C7]": "x_vars[W8,C7]",
    "s[W8,C9]": "x_vars[W8,C9]",
    "s[W8,C10]": "x_vars[W8,C10]",
    "s[W9,C1]": "x_vars[W9,C1]",
    "s[W9,C2]": "x_vars[W9,C2]",
    "s[W9,C3]": "x_vars[W9,C3]",
    "s[W9,C4]": "x_vars[W9,C4]",
    "s[W9,C5]": "x_vars[W9,C5]",
    "s[W9,C6]": "x_vars[W9,C6]",
    "s[W9,C7]": "x_vars[W9,C7]",
    "s[W9,C8]": "x_vars[W9,C8]",
    "s[W9,C10]": "x_vars[W9,C10]",
    "s[W10,C1]": "x_vars[W10,C1]",
    "s[W10,C2]": "x_vars[W10,C2]",
    "s[W10,C3]": "x_vars[W10,C3]",
    "s[W10,C4]": "x_vars[W10,C4]",
    "s[W10,C5]": "x_vars[W10,C5]",
    "s[W10,C6]": "x_vars[W10,C6]",
    "s[W10,C7]": "x_vars[W10,C7]",
    "s[W10,C8]": "x_vars[W10,C8]",
    "s[W10,C9]": "x_vars[W10,C9]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-077": {
   "source_code_sha256": "02c2d048b137a990def3a5b6aea55e97a0a13cc023b651a85ded2edbbad9cfa0",
   "original_solution_sha256": "2adcbede4949f9e7827335fe535c6007bd8de9da543c3e67286e38a1281ee354",
   "bindings": [
    [
     "C0",
     "x_vars[1]"
    ],
    [
     "C1",
     "x_vars[2]"
    ],
    [
     "C2",
     "x_vars[3]"
    ],
    [
     "C3",
     "x_vars[4]"
    ],
    [
     "C4",
     "x_vars[5]"
    ],
    [
     "C5",
     "x_vars[6]"
    ],
    [
     "C6",
     "x_vars[7]"
    ],
    [
     "C7",
     "x_vars[8]"
    ],
    [
     "C8",
     "x_vars[9]"
    ],
    [
     "C9",
     "x_vars[10]"
    ],
    [
     "C10",
     "x_vars[11]"
    ],
    [
     "C11",
     "x_vars[12]"
    ],
    [
     "C12",
     "x_vars[13]"
    ],
    [
     "C13",
     "x_vars[14]"
    ],
    [
     "C14",
     "x_vars[15]"
    ],
    [
     "C15",
     "x_vars[16]"
    ],
    [
     "C16",
     "x_vars[17]"
    ],
    [
     "C17",
     "x_vars[18]"
    ],
    [
     "C18",
     "x_vars[19]"
    ],
    [
     "C19",
     "x_vars[20]"
    ],
    [
     "C20",
     "x_vars[21]"
    ],
    [
     "C21",
     "x_vars[22]"
    ],
    [
     "C22",
     "x_vars[23]"
    ],
    [
     "C23",
     "x_vars[24]"
    ],
    [
     "C24",
     "x_vars[25]"
    ],
    [
     "C25",
     "x_vars[26]"
    ],
    [
     "C26",
     "x_vars[27]"
    ],
    [
     "C27",
     "x_vars[28]"
    ],
    [
     "C28",
     "x_vars[29]"
    ],
    [
     "C29",
     "x_vars[30]"
    ],
    [
     "C30",
     "x_vars[31]"
    ],
    [
     "C31",
     "x_vars[32]"
    ],
    [
     "C32",
     "x_vars[33]"
    ],
    [
     "C33",
     "x_vars[34]"
    ],
    [
     "C34",
     "x_vars[35]"
    ],
    [
     "C35",
     "x_vars[36]"
    ],
    [
     "C36",
     "x_vars[37]"
    ],
    [
     "C37",
     "x_vars[38]"
    ],
    [
     "C38",
     "x_vars[39]"
    ],
    [
     "C39",
     "x_vars[40]"
    ],
    [
     "C40",
     "x_vars[41]"
    ],
    [
     "C41",
     "x_vars[42]"
    ],
    [
     "C42",
     "x_vars[43]"
    ],
    [
     "C43",
     "x_vars[44]"
    ],
    [
     "C44",
     "x_vars[45]"
    ],
    [
     "C45",
     "x_vars[46]"
    ],
    [
     "C46",
     "x_vars[47]"
    ],
    [
     "C47",
     "x_vars[48]"
    ],
    [
     "C48",
     "x_vars[49]"
    ],
    [
     "C49",
     "x_vars[50]"
    ],
    [
     "C50",
     "x_vars[51]"
    ],
    [
     "C51",
     "x_vars[52]"
    ],
    [
     "C52",
     "x_vars[53]"
    ],
    [
     "C53",
     "x_vars[54]"
    ],
    [
     "C54",
     "x_vars[55]"
    ],
    [
     "C55",
     "x_vars[56]"
    ],
    [
     "C56",
     "x_vars[57]"
    ],
    [
     "C57",
     "x_vars[58]"
    ],
    [
     "C58",
     "x_vars[59]"
    ],
    [
     "C59",
     "x_vars[60]"
    ],
    [
     "C60",
     "x_vars[61]"
    ],
    [
     "C61",
     "x_vars[62]"
    ],
    [
     "C62",
     "x_vars[63]"
    ],
    [
     "C63",
     "x_vars[64]"
    ],
    [
     "C64",
     "x_vars[65]"
    ],
    [
     "C65",
     "x_vars[66]"
    ],
    [
     "C66",
     "x_vars[67]"
    ],
    [
     "C67",
     "x_vars[68]"
    ],
    [
     "C68",
     "x_vars[69]"
    ],
    [
     "C69",
     "x_vars[70]"
    ],
    [
     "C70",
     "x_vars[71]"
    ],
    [
     "C71",
     "x_vars[72]"
    ],
    [
     "C72",
     "x_vars[73]"
    ],
    [
     "C73",
     "x_vars[74]"
    ],
    [
     "C74",
     "x_vars[75]"
    ],
    [
     "C75",
     "x_vars[76]"
    ],
    [
     "C76",
     "x_vars[77]"
    ],
    [
     "C77",
     "x_vars[78]"
    ],
    [
     "C78",
     "x_vars[79]"
    ],
    [
     "C79",
     "x_vars[80]"
    ],
    [
     "C80",
     "x_vars[81]"
    ],
    [
     "C81",
     "x_vars[82]"
    ],
    [
     "C82",
     "x_vars[83]"
    ],
    [
     "C83",
     "x_vars[84]"
    ],
    [
     "C84",
     "x_vars[85]"
    ],
    [
     "C85",
     "x_vars[86]"
    ],
    [
     "C86",
     "x_vars[87]"
    ],
    [
     "C87",
     "x_vars[88]"
    ],
    [
     "C88",
     "x_vars[89]"
    ],
    [
     "C89",
     "x_vars[90]"
    ],
    [
     "C90",
     "x_vars[91]"
    ],
    [
     "C91",
     "x_vars[92]"
    ],
    [
     "C92",
     "x_vars[93]"
    ],
    [
     "C93",
     "x_vars[94]"
    ],
    [
     "C94",
     "x_vars[95]"
    ],
    [
     "C95",
     "x_vars[96]"
    ],
    [
     "C96",
     "x_vars[97]"
    ],
    [
     "C97",
     "x_vars[98]"
    ],
    [
     "C98",
     "x_vars[99]"
    ],
    [
     "C99",
     "x_vars[100]"
    ],
    [
     "C100",
     "x_vars[101]"
    ],
    [
     "C101",
     "x_vars[102]"
    ],
    [
     "C102",
     "x_vars[103]"
    ],
    [
     "C103",
     "x_vars[104]"
    ],
    [
     "C104",
     "x_vars[105]"
    ],
    [
     "C105",
     "x_vars[106]"
    ],
    [
     "C106",
     "x_vars[107]"
    ],
    [
     "C107",
     "x_vars[108]"
    ],
    [
     "C108",
     "x_vars[109]"
    ],
    [
     "C109",
     "x_vars[110]"
    ],
    [
     "C110",
     "x_vars[111]"
    ],
    [
     "C111",
     "x_vars[112]"
    ],
    [
     "C112",
     "x_vars[113]"
    ],
    [
     "C113",
     "x_vars[114]"
    ],
    [
     "C114",
     "x_vars[115]"
    ],
    [
     "C115",
     "x_vars[116]"
    ],
    [
     "C116",
     "x_vars[117]"
    ],
    [
     "C117",
     "x_vars[118]"
    ],
    [
     "C118",
     "x_vars[119]"
    ],
    [
     "C119",
     "x_vars[120]"
    ],
    [
     "C120",
     "x_vars[121]"
    ],
    [
     "C121",
     "x_vars[122]"
    ],
    [
     "C122",
     "x_vars[123]"
    ],
    [
     "C123",
     "x_vars[124]"
    ],
    [
     "C124",
     "x_vars[125]"
    ],
    [
     "C125",
     "x_vars[126]"
    ],
    [
     "C126",
     "x_vars[127]"
    ],
    [
     "C127",
     "x_vars[128]"
    ],
    [
     "C128",
     "x_vars[129]"
    ],
    [
     "C129",
     "x_vars[130]"
    ],
    [
     "C130",
     "x_vars[131]"
    ],
    [
     "C131",
     "x_vars[132]"
    ],
    [
     "C132",
     "x_vars[133]"
    ],
    [
     "C133",
     "x_vars[134]"
    ],
    [
     "C134",
     "x_vars[135]"
    ],
    [
     "C135",
     "x_vars[136]"
    ],
    [
     "C136",
     "x_vars[137]"
    ],
    [
     "C137",
     "x_vars[138]"
    ],
    [
     "C138",
     "x_vars[139]"
    ],
    [
     "C139",
     "x_vars[140]"
    ]
   ],
   "primary_mapping": {
    "x[0]": "x_vars[1]",
    "x[1]": "x_vars[2]",
    "x[2]": "x_vars[3]",
    "x[3]": "x_vars[4]",
    "x[4]": "x_vars[5]",
    "x[5]": "x_vars[6]",
    "x[6]": "x_vars[7]",
    "x[7]": "x_vars[8]",
    "x[8]": "x_vars[9]",
    "x[9]": "x_vars[10]",
    "x[10]": "x_vars[11]",
    "x[11]": "x_vars[12]",
    "x[12]": "x_vars[13]",
    "x[13]": "x_vars[14]",
    "x[14]": "x_vars[15]",
    "x[15]": "x_vars[16]",
    "x[16]": "x_vars[17]",
    "x[17]": "x_vars[18]",
    "x[18]": "x_vars[19]",
    "x[19]": "x_vars[20]",
    "x[20]": "x_vars[21]",
    "x[21]": "x_vars[22]",
    "x[22]": "x_vars[23]",
    "x[23]": "x_vars[24]",
    "x[24]": "x_vars[25]",
    "x[25]": "x_vars[26]",
    "x[26]": "x_vars[27]",
    "x[27]": "x_vars[28]",
    "x[28]": "x_vars[29]",
    "x[29]": "x_vars[30]",
    "x[30]": "x_vars[31]",
    "x[31]": "x_vars[32]",
    "x[32]": "x_vars[33]",
    "x[33]": "x_vars[34]",
    "x[34]": "x_vars[35]",
    "x[35]": "x_vars[36]",
    "x[36]": "x_vars[37]",
    "x[37]": "x_vars[38]",
    "x[38]": "x_vars[39]",
    "x[39]": "x_vars[40]",
    "x[40]": "x_vars[41]",
    "x[41]": "x_vars[42]",
    "x[42]": "x_vars[43]",
    "x[43]": "x_vars[44]",
    "x[44]": "x_vars[45]",
    "x[45]": "x_vars[46]",
    "x[46]": "x_vars[47]",
    "x[47]": "x_vars[48]",
    "x[48]": "x_vars[49]",
    "x[49]": "x_vars[50]",
    "x[50]": "x_vars[51]",
    "x[51]": "x_vars[52]",
    "x[52]": "x_vars[53]",
    "x[53]": "x_vars[54]",
    "x[54]": "x_vars[55]",
    "x[55]": "x_vars[56]",
    "x[56]": "x_vars[57]",
    "x[57]": "x_vars[58]",
    "x[58]": "x_vars[59]",
    "x[59]": "x_vars[60]",
    "x[60]": "x_vars[61]",
    "x[61]": "x_vars[62]",
    "x[62]": "x_vars[63]",
    "x[63]": "x_vars[64]",
    "x[64]": "x_vars[65]",
    "x[65]": "x_vars[66]",
    "x[66]": "x_vars[67]",
    "x[67]": "x_vars[68]",
    "x[68]": "x_vars[69]",
    "x[69]": "x_vars[70]",
    "x[70]": "x_vars[71]",
    "x[71]": "x_vars[72]",
    "x[72]": "x_vars[73]",
    "x[73]": "x_vars[74]",
    "x[74]": "x_vars[75]",
    "x[75]": "x_vars[76]",
    "x[76]": "x_vars[77]",
    "x[77]": "x_vars[78]",
    "x[78]": "x_vars[79]",
    "x[79]": "x_vars[80]",
    "x[80]": "x_vars[81]",
    "x[81]": "x_vars[82]",
    "x[82]": "x_vars[83]",
    "x[83]": "x_vars[84]",
    "x[84]": "x_vars[85]",
    "x[85]": "x_vars[86]",
    "x[86]": "x_vars[87]",
    "x[87]": "x_vars[88]",
    "x[88]": "x_vars[89]",
    "x[89]": "x_vars[90]",
    "x[90]": "x_vars[91]",
    "x[91]": "x_vars[92]",
    "x[92]": "x_vars[93]",
    "x[93]": "x_vars[94]",
    "x[94]": "x_vars[95]",
    "x[95]": "x_vars[96]",
    "x[96]": "x_vars[97]",
    "x[97]": "x_vars[98]",
    "x[98]": "x_vars[99]",
    "x[99]": "x_vars[100]",
    "x[100]": "x_vars[101]",
    "x[101]": "x_vars[102]",
    "x[102]": "x_vars[103]",
    "x[103]": "x_vars[104]",
    "x[104]": "x_vars[105]",
    "x[105]": "x_vars[106]",
    "x[106]": "x_vars[107]",
    "x[107]": "x_vars[108]",
    "x[108]": "x_vars[109]",
    "x[109]": "x_vars[110]",
    "x[110]": "x_vars[111]",
    "x[111]": "x_vars[112]",
    "x[112]": "x_vars[113]",
    "x[113]": "x_vars[114]",
    "x[114]": "x_vars[115]",
    "x[115]": "x_vars[116]",
    "x[116]": "x_vars[117]",
    "x[117]": "x_vars[118]",
    "x[118]": "x_vars[119]",
    "x[119]": "x_vars[120]",
    "x[120]": "x_vars[121]",
    "x[121]": "x_vars[122]",
    "x[122]": "x_vars[123]",
    "x[123]": "x_vars[124]",
    "x[124]": "x_vars[125]",
    "x[125]": "x_vars[126]",
    "x[126]": "x_vars[127]",
    "x[127]": "x_vars[128]",
    "x[128]": "x_vars[129]",
    "x[129]": "x_vars[130]",
    "x[130]": "x_vars[131]",
    "x[131]": "x_vars[132]",
    "x[132]": "x_vars[133]",
    "x[133]": "x_vars[134]",
    "x[134]": "x_vars[135]",
    "x[135]": "x_vars[136]",
    "x[136]": "x_vars[137]",
    "x[137]": "x_vars[138]",
    "x[138]": "x_vars[139]",
    "x[139]": "x_vars[140]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-078": {
   "source_code_sha256": "12e4ebd1bc5e63f1d47e031ec4ae2d700da01ef81c4e28012d223ec4a3112e2b",
   "original_solution_sha256": "51826467d314ad5a50f20f912475ba727b2ef2efbc573b5b2ed7dae16f4cd823",
   "bindings": [
    [
     "C0",
     "x_vars[coal_001]"
    ],
    [
     "C1",
     "x_vars[coal_002]"
    ],
    [
     "C2",
     "x_vars[coal_003]"
    ],
    [
     "C3",
     "x_vars[coal_004]"
    ],
    [
     "C4",
     "x_vars[coal_005]"
    ],
    [
     "C5",
     "x_vars[coal_006]"
    ],
    [
     "C6",
     "x_vars[coal_007]"
    ],
    [
     "C7",
     "x_vars[coal_008]"
    ],
    [
     "C8",
     "x_vars[coal_009]"
    ],
    [
     "C9",
     "x_vars[coal_010]"
    ],
    [
     "C10",
     "x_vars[coal_011]"
    ],
    [
     "C11",
     "x_vars[coal_012]"
    ],
    [
     "C12",
     "x_vars[coal_013]"
    ],
    [
     "C13",
     "x_vars[coal_014]"
    ],
    [
     "C14",
     "x_vars[coal_015]"
    ],
    [
     "C15",
     "x_vars[coal_043]"
    ],
    [
     "C16",
     "x_vars[coal_044]"
    ],
    [
     "C17",
     "x_vars[coal_045]"
    ],
    [
     "C18",
     "x_vars[coal_046]"
    ],
    [
     "C19",
     "x_vars[coal_047]"
    ],
    [
     "C20",
     "x_vars[coal_048]"
    ],
    [
     "C21",
     "x_vars[coal_049]"
    ],
    [
     "C22",
     "x_vars[coal_050]"
    ],
    [
     "C23",
     "x_vars[coal_051]"
    ],
    [
     "C24",
     "x_vars[coal_091]"
    ],
    [
     "C25",
     "x_vars[coal_092]"
    ],
    [
     "C26",
     "x_vars[coal_093]"
    ],
    [
     "C27",
     "x_vars[coal_094]"
    ],
    [
     "C28",
     "x_vars[coal_095]"
    ],
    [
     "C29",
     "x_vars[coal_096]"
    ],
    [
     "C30",
     "x_vars[coal_097]"
    ],
    [
     "C31",
     "x_vars[coal_098]"
    ],
    [
     "C32",
     "x_vars[coal_099]"
    ],
    [
     "C33",
     "x_vars[coal_100]"
    ],
    [
     "C34",
     "x_vars[gas_001]"
    ],
    [
     "C35",
     "x_vars[gas_002]"
    ],
    [
     "C36",
     "x_vars[gas_003]"
    ],
    [
     "C37",
     "x_vars[gas_004]"
    ],
    [
     "C38",
     "x_vars[gas_005]"
    ],
    [
     "C39",
     "x_vars[gas_006]"
    ],
    [
     "C40",
     "x_vars[gas_007]"
    ],
    [
     "C41",
     "x_vars[gas_008]"
    ],
    [
     "C42",
     "x_vars[gas_009]"
    ],
    [
     "C43",
     "x_vars[gas_010]"
    ],
    [
     "C44",
     "x_vars[gas_011]"
    ],
    [
     "C45",
     "x_vars[gas_012]"
    ],
    [
     "C46",
     "x_vars[gas_013]"
    ],
    [
     "C47",
     "x_vars[gas_014]"
    ],
    [
     "C48",
     "x_vars[gas_015]"
    ],
    [
     "C49",
     "x_vars[gas_016]"
    ],
    [
     "C50",
     "x_vars[gas_017]"
    ],
    [
     "C51",
     "x_vars[gas_018]"
    ],
    [
     "C52",
     "x_vars[gas_019]"
    ],
    [
     "C53",
     "x_vars[gas_020]"
    ],
    [
     "C54",
     "x_vars[gas_021]"
    ],
    [
     "C55",
     "x_vars[gas_022]"
    ],
    [
     "C56",
     "x_vars[gas_023]"
    ],
    [
     "C57",
     "x_vars[gas_024]"
    ],
    [
     "C58",
     "x_vars[gas_025]"
    ],
    [
     "C59",
     "x_vars[gas_026]"
    ],
    [
     "C60",
     "x_vars[gas_027]"
    ],
    [
     "C61",
     "x_vars[gas_028]"
    ],
    [
     "C62",
     "x_vars[gas_029]"
    ],
    [
     "C63",
     "x_vars[gas_030]"
    ],
    [
     "C64",
     "x_vars[gas_031]"
    ],
    [
     "C65",
     "x_vars[gas_032]"
    ],
    [
     "C66",
     "x_vars[gas_033]"
    ],
    [
     "C67",
     "x_vars[gas_034]"
    ],
    [
     "C68",
     "x_vars[gas_035]"
    ],
    [
     "C69",
     "x_vars[gas_036]"
    ],
    [
     "C70",
     "x_vars[gas_037]"
    ],
    [
     "C71",
     "x_vars[gas_038]"
    ],
    [
     "C72",
     "x_vars[gas_039]"
    ],
    [
     "C73",
     "x_vars[gas_040]"
    ],
    [
     "C74",
     "x_vars[gas_041]"
    ],
    [
     "C75",
     "x_vars[gas_042]"
    ],
    [
     "C76",
     "x_vars[gas_043]"
    ],
    [
     "C77",
     "x_vars[gas_044]"
    ],
    [
     "C78",
     "x_vars[gas_045]"
    ],
    [
     "C79",
     "x_vars[gas_062]"
    ],
    [
     "C80",
     "x_vars[gas_063]"
    ],
    [
     "C81",
     "x_vars[gas_064]"
    ],
    [
     "C82",
     "x_vars[gas_065]"
    ],
    [
     "C83",
     "x_vars[gas_066]"
    ],
    [
     "C84",
     "x_vars[gas_067]"
    ],
    [
     "C85",
     "x_vars[gas_096]"
    ],
    [
     "C86",
     "x_vars[gas_097]"
    ],
    [
     "C87",
     "x_vars[gas_098]"
    ],
    [
     "C88",
     "x_vars[gas_099]"
    ],
    [
     "C89",
     "x_vars[gas_100]"
    ],
    [
     "C90",
     "x_vars[renewables_001]"
    ],
    [
     "C91",
     "x_vars[renewables_002]"
    ],
    [
     "C92",
     "x_vars[renewables_003]"
    ],
    [
     "C93",
     "x_vars[renewables_004]"
    ],
    [
     "C94",
     "x_vars[renewables_005]"
    ],
    [
     "C95",
     "x_vars[renewables_021]"
    ],
    [
     "C96",
     "x_vars[renewables_022]"
    ],
    [
     "C97",
     "x_vars[renewables_023]"
    ],
    [
     "C98",
     "x_vars[renewables_024]"
    ],
    [
     "C99",
     "x_vars[renewables_025]"
    ],
    [
     "C100",
     "x_vars[renewables_026]"
    ],
    [
     "C101",
     "x_vars[renewables_027]"
    ],
    [
     "C102",
     "x_vars[renewables_028]"
    ],
    [
     "C103",
     "x_vars[renewables_029]"
    ],
    [
     "C104",
     "x_vars[renewables_030]"
    ],
    [
     "C105",
     "x_vars[renewables_031]"
    ],
    [
     "C106",
     "x_vars[renewables_032]"
    ],
    [
     "C107",
     "x_vars[renewables_033]"
    ],
    [
     "C108",
     "x_vars[renewables_034]"
    ],
    [
     "C109",
     "x_vars[renewables_035]"
    ],
    [
     "C110",
     "x_vars[renewables_036]"
    ],
    [
     "C111",
     "x_vars[renewables_037]"
    ],
    [
     "C112",
     "x_vars[renewables_038]"
    ],
    [
     "C113",
     "x_vars[renewables_039]"
    ],
    [
     "C114",
     "x_vars[renewables_040]"
    ],
    [
     "C115",
     "x_vars[renewables_041]"
    ],
    [
     "C116",
     "x_vars[renewables_042]"
    ],
    [
     "C117",
     "x_vars[renewables_043]"
    ],
    [
     "C118",
     "x_vars[renewables_044]"
    ],
    [
     "C119",
     "x_vars[renewables_045]"
    ],
    [
     "C120",
     "x_vars[renewables_046]"
    ],
    [
     "C121",
     "x_vars[renewables_047]"
    ],
    [
     "C122",
     "x_vars[renewables_048]"
    ],
    [
     "C123",
     "x_vars[renewables_049]"
    ],
    [
     "C124",
     "x_vars[renewables_050]"
    ],
    [
     "C125",
     "x_vars[renewables_051]"
    ],
    [
     "C126",
     "x_vars[renewables_052]"
    ],
    [
     "C127",
     "x_vars[renewables_053]"
    ],
    [
     "C128",
     "x_vars[renewables_054]"
    ],
    [
     "C129",
     "x_vars[renewables_055]"
    ],
    [
     "C130",
     "x_vars[renewables_056]"
    ]
   ],
   "primary_mapping": {
    "x[0]": "x_vars[coal_001]",
    "x[1]": "x_vars[coal_002]",
    "x[2]": "x_vars[coal_003]",
    "x[3]": "x_vars[coal_004]",
    "x[4]": "x_vars[coal_005]",
    "x[5]": "x_vars[coal_006]",
    "x[6]": "x_vars[coal_007]",
    "x[7]": "x_vars[coal_008]",
    "x[8]": "x_vars[coal_009]",
    "x[9]": "x_vars[coal_010]",
    "x[10]": "x_vars[coal_011]",
    "x[11]": "x_vars[coal_012]",
    "x[12]": "x_vars[coal_013]",
    "x[13]": "x_vars[coal_014]",
    "x[14]": "x_vars[coal_015]",
    "x[15]": "x_vars[coal_043]",
    "x[16]": "x_vars[coal_044]",
    "x[17]": "x_vars[coal_045]",
    "x[18]": "x_vars[coal_046]",
    "x[19]": "x_vars[coal_047]",
    "x[20]": "x_vars[coal_048]",
    "x[21]": "x_vars[coal_049]",
    "x[22]": "x_vars[coal_050]",
    "x[23]": "x_vars[coal_051]",
    "x[24]": "x_vars[coal_091]",
    "x[25]": "x_vars[coal_092]",
    "x[26]": "x_vars[coal_093]",
    "x[27]": "x_vars[coal_094]",
    "x[28]": "x_vars[coal_095]",
    "x[29]": "x_vars[coal_096]",
    "x[30]": "x_vars[coal_097]",
    "x[31]": "x_vars[coal_098]",
    "x[32]": "x_vars[coal_099]",
    "x[33]": "x_vars[coal_100]",
    "x[34]": "x_vars[gas_001]",
    "x[35]": "x_vars[gas_002]",
    "x[36]": "x_vars[gas_003]",
    "x[37]": "x_vars[gas_004]",
    "x[38]": "x_vars[gas_005]",
    "x[39]": "x_vars[gas_006]",
    "x[40]": "x_vars[gas_007]",
    "x[41]": "x_vars[gas_008]",
    "x[42]": "x_vars[gas_009]",
    "x[43]": "x_vars[gas_010]",
    "x[44]": "x_vars[gas_011]",
    "x[45]": "x_vars[gas_012]",
    "x[46]": "x_vars[gas_013]",
    "x[47]": "x_vars[gas_014]",
    "x[48]": "x_vars[gas_015]",
    "x[49]": "x_vars[gas_016]",
    "x[50]": "x_vars[gas_017]",
    "x[51]": "x_vars[gas_018]",
    "x[52]": "x_vars[gas_019]",
    "x[53]": "x_vars[gas_020]",
    "x[54]": "x_vars[gas_021]",
    "x[55]": "x_vars[gas_022]",
    "x[56]": "x_vars[gas_023]",
    "x[57]": "x_vars[gas_024]",
    "x[58]": "x_vars[gas_025]",
    "x[59]": "x_vars[gas_026]",
    "x[60]": "x_vars[gas_027]",
    "x[61]": "x_vars[gas_028]",
    "x[62]": "x_vars[gas_029]",
    "x[63]": "x_vars[gas_030]",
    "x[64]": "x_vars[gas_031]",
    "x[65]": "x_vars[gas_032]",
    "x[66]": "x_vars[gas_033]",
    "x[67]": "x_vars[gas_034]",
    "x[68]": "x_vars[gas_035]",
    "x[69]": "x_vars[gas_036]",
    "x[70]": "x_vars[gas_037]",
    "x[71]": "x_vars[gas_038]",
    "x[72]": "x_vars[gas_039]",
    "x[73]": "x_vars[gas_040]",
    "x[74]": "x_vars[gas_041]",
    "x[75]": "x_vars[gas_042]",
    "x[76]": "x_vars[gas_043]",
    "x[77]": "x_vars[gas_044]",
    "x[78]": "x_vars[gas_045]",
    "x[79]": "x_vars[gas_062]",
    "x[80]": "x_vars[gas_063]",
    "x[81]": "x_vars[gas_064]",
    "x[82]": "x_vars[gas_065]",
    "x[83]": "x_vars[gas_066]",
    "x[84]": "x_vars[gas_067]",
    "x[85]": "x_vars[gas_096]",
    "x[86]": "x_vars[gas_097]",
    "x[87]": "x_vars[gas_098]",
    "x[88]": "x_vars[gas_099]",
    "x[89]": "x_vars[gas_100]",
    "x[90]": "x_vars[renewables_001]",
    "x[91]": "x_vars[renewables_002]",
    "x[92]": "x_vars[renewables_003]",
    "x[93]": "x_vars[renewables_004]",
    "x[94]": "x_vars[renewables_005]",
    "x[95]": "x_vars[renewables_021]",
    "x[96]": "x_vars[renewables_022]",
    "x[97]": "x_vars[renewables_023]",
    "x[98]": "x_vars[renewables_024]",
    "x[99]": "x_vars[renewables_025]",
    "x[100]": "x_vars[renewables_026]",
    "x[101]": "x_vars[renewables_027]",
    "x[102]": "x_vars[renewables_028]",
    "x[103]": "x_vars[renewables_029]",
    "x[104]": "x_vars[renewables_030]",
    "x[105]": "x_vars[renewables_031]",
    "x[106]": "x_vars[renewables_032]",
    "x[107]": "x_vars[renewables_033]",
    "x[108]": "x_vars[renewables_034]",
    "x[109]": "x_vars[renewables_035]",
    "x[110]": "x_vars[renewables_036]",
    "x[111]": "x_vars[renewables_037]",
    "x[112]": "x_vars[renewables_038]",
    "x[113]": "x_vars[renewables_039]",
    "x[114]": "x_vars[renewables_040]",
    "x[115]": "x_vars[renewables_041]",
    "x[116]": "x_vars[renewables_042]",
    "x[117]": "x_vars[renewables_043]",
    "x[118]": "x_vars[renewables_044]",
    "x[119]": "x_vars[renewables_045]",
    "x[120]": "x_vars[renewables_046]",
    "x[121]": "x_vars[renewables_047]",
    "x[122]": "x_vars[renewables_048]",
    "x[123]": "x_vars[renewables_049]",
    "x[124]": "x_vars[renewables_050]",
    "x[125]": "x_vars[renewables_051]",
    "x[126]": "x_vars[renewables_052]",
    "x[127]": "x_vars[renewables_053]",
    "x[128]": "x_vars[renewables_054]",
    "x[129]": "x_vars[renewables_055]",
    "x[130]": "x_vars[renewables_056]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-079": {
   "source_code_sha256": "e2538f9026d3dae5f75e88fa32b8a1b274e73db1fc586526bc817975271c601b",
   "original_solution_sha256": "538c230448d7e17e03b916ac63f4a66f2cfa4b5605cc7c7f7274aa3085be61f5",
   "bindings": [
    [
     "C0",
     "x_vars[A1,B1]"
    ],
    [
     "C1",
     "x_vars[A1,B2]"
    ],
    [
     "C2",
     "x_vars[A1,B3]"
    ],
    [
     "C3",
     "x_vars[A1,B4]"
    ],
    [
     "C4",
     "x_vars[A1,B5]"
    ],
    [
     "C5",
     "x_vars[A1,B6]"
    ],
    [
     "C6",
     "x_vars[A1,B7]"
    ],
    [
     "C7",
     "x_vars[A1,B8]"
    ],
    [
     "C8",
     "x_vars[A2,B1]"
    ],
    [
     "C9",
     "x_vars[A2,B2]"
    ],
    [
     "C10",
     "x_vars[A2,B3]"
    ],
    [
     "C11",
     "x_vars[A2,B4]"
    ],
    [
     "C12",
     "x_vars[A2,B5]"
    ],
    [
     "C13",
     "x_vars[A2,B6]"
    ],
    [
     "C14",
     "x_vars[A2,B7]"
    ],
    [
     "C15",
     "x_vars[A2,B8]"
    ],
    [
     "C16",
     "x_vars[A3,B1]"
    ],
    [
     "C17",
     "x_vars[A3,B2]"
    ],
    [
     "C18",
     "x_vars[A3,B3]"
    ],
    [
     "C19",
     "x_vars[A3,B4]"
    ],
    [
     "C20",
     "x_vars[A3,B5]"
    ],
    [
     "C21",
     "x_vars[A3,B6]"
    ],
    [
     "C22",
     "x_vars[A3,B7]"
    ],
    [
     "C23",
     "x_vars[A3,B8]"
    ],
    [
     "C24",
     "x_vars[A4,B1]"
    ],
    [
     "C25",
     "x_vars[A4,B2]"
    ],
    [
     "C26",
     "x_vars[A4,B3]"
    ],
    [
     "C27",
     "x_vars[A4,B4]"
    ],
    [
     "C28",
     "x_vars[A4,B5]"
    ],
    [
     "C29",
     "x_vars[A4,B6]"
    ],
    [
     "C30",
     "x_vars[A4,B7]"
    ],
    [
     "C31",
     "x_vars[A4,B8]"
    ],
    [
     "C32",
     "x_vars[A5,B1]"
    ],
    [
     "C33",
     "x_vars[A5,B2]"
    ],
    [
     "C34",
     "x_vars[A5,B3]"
    ],
    [
     "C35",
     "x_vars[A5,B4]"
    ],
    [
     "C36",
     "x_vars[A5,B5]"
    ],
    [
     "C37",
     "x_vars[A5,B6]"
    ],
    [
     "C38",
     "x_vars[A5,B7]"
    ],
    [
     "C39",
     "x_vars[A5,B8]"
    ],
    [
     "C40",
     "x_vars[A6,B1]"
    ],
    [
     "C41",
     "x_vars[A6,B2]"
    ],
    [
     "C42",
     "x_vars[A6,B3]"
    ],
    [
     "C43",
     "x_vars[A6,B4]"
    ],
    [
     "C44",
     "x_vars[A6,B5]"
    ],
    [
     "C45",
     "x_vars[A6,B6]"
    ],
    [
     "C46",
     "x_vars[A6,B7]"
    ],
    [
     "C47",
     "x_vars[A6,B8]"
    ],
    [
     "C48",
     "x_vars[A7,B1]"
    ],
    [
     "C49",
     "x_vars[A7,B2]"
    ],
    [
     "C50",
     "x_vars[A7,B3]"
    ],
    [
     "C51",
     "x_vars[A7,B4]"
    ],
    [
     "C52",
     "x_vars[A7,B5]"
    ],
    [
     "C53",
     "x_vars[A7,B6]"
    ],
    [
     "C54",
     "x_vars[A7,B7]"
    ],
    [
     "C55",
     "x_vars[A7,B8]"
    ],
    [
     "C56",
     "x_vars[A8,B1]"
    ],
    [
     "C57",
     "x_vars[A8,B2]"
    ],
    [
     "C58",
     "x_vars[A8,B3]"
    ],
    [
     "C59",
     "x_vars[A8,B4]"
    ],
    [
     "C60",
     "x_vars[A8,B5]"
    ],
    [
     "C61",
     "x_vars[A8,B6]"
    ],
    [
     "C62",
     "x_vars[A8,B7]"
    ],
    [
     "C63",
     "x_vars[A8,B8]"
    ],
    [
     "C64",
     "x_vars[A9,B1]"
    ],
    [
     "C65",
     "x_vars[A9,B2]"
    ],
    [
     "C66",
     "x_vars[A9,B3]"
    ],
    [
     "C67",
     "x_vars[A9,B4]"
    ],
    [
     "C68",
     "x_vars[A9,B5]"
    ],
    [
     "C69",
     "x_vars[A9,B6]"
    ],
    [
     "C70",
     "x_vars[A9,B7]"
    ],
    [
     "C71",
     "x_vars[A9,B8]"
    ],
    [
     "C72",
     "x_vars[A10,B1]"
    ],
    [
     "C73",
     "x_vars[A10,B2]"
    ],
    [
     "C74",
     "x_vars[A10,B3]"
    ],
    [
     "C75",
     "x_vars[A10,B4]"
    ],
    [
     "C76",
     "x_vars[A10,B5]"
    ],
    [
     "C77",
     "x_vars[A10,B6]"
    ],
    [
     "C78",
     "x_vars[A10,B7]"
    ],
    [
     "C79",
     "x_vars[A10,B8]"
    ],
    [
     "C80",
     "x_vars[A11,B1]"
    ],
    [
     "C81",
     "x_vars[A11,B2]"
    ],
    [
     "C82",
     "x_vars[A11,B3]"
    ],
    [
     "C83",
     "x_vars[A11,B4]"
    ],
    [
     "C84",
     "x_vars[A11,B5]"
    ],
    [
     "C85",
     "x_vars[A11,B6]"
    ],
    [
     "C86",
     "x_vars[A11,B7]"
    ],
    [
     "C87",
     "x_vars[A11,B8]"
    ],
    [
     "C88",
     "x_vars[A12,B1]"
    ],
    [
     "C89",
     "x_vars[A12,B2]"
    ],
    [
     "C90",
     "x_vars[A12,B3]"
    ],
    [
     "C91",
     "x_vars[A12,B4]"
    ],
    [
     "C92",
     "x_vars[A12,B5]"
    ],
    [
     "C93",
     "x_vars[A12,B6]"
    ],
    [
     "C94",
     "x_vars[A12,B7]"
    ],
    [
     "C95",
     "x_vars[A12,B8]"
    ],
    [
     "C96",
     "x_vars[A13,B1]"
    ],
    [
     "C97",
     "x_vars[A13,B2]"
    ],
    [
     "C98",
     "x_vars[A13,B3]"
    ],
    [
     "C99",
     "x_vars[A13,B4]"
    ],
    [
     "C100",
     "x_vars[A13,B5]"
    ],
    [
     "C101",
     "x_vars[A13,B6]"
    ],
    [
     "C102",
     "x_vars[A13,B7]"
    ],
    [
     "C103",
     "x_vars[A13,B8]"
    ],
    [
     "C104",
     "x_vars[A14,B1]"
    ],
    [
     "C105",
     "x_vars[A14,B2]"
    ],
    [
     "C106",
     "x_vars[A14,B3]"
    ],
    [
     "C107",
     "x_vars[A14,B4]"
    ],
    [
     "C108",
     "x_vars[A14,B5]"
    ],
    [
     "C109",
     "x_vars[A14,B6]"
    ],
    [
     "C110",
     "x_vars[A14,B7]"
    ],
    [
     "C111",
     "x_vars[A14,B8]"
    ],
    [
     "C112",
     "x_vars[A15,B1]"
    ],
    [
     "C113",
     "x_vars[A15,B2]"
    ],
    [
     "C114",
     "x_vars[A15,B3]"
    ],
    [
     "C115",
     "x_vars[A15,B4]"
    ],
    [
     "C116",
     "x_vars[A15,B5]"
    ],
    [
     "C117",
     "x_vars[A15,B6]"
    ],
    [
     "C118",
     "x_vars[A15,B7]"
    ],
    [
     "C119",
     "x_vars[A15,B8]"
    ],
    [
     "C120",
     "y_vars[A1]"
    ],
    [
     "C121",
     "y_vars[A2]"
    ],
    [
     "C122",
     "y_vars[A3]"
    ],
    [
     "C123",
     "y_vars[A4]"
    ],
    [
     "C124",
     "y_vars[A5]"
    ],
    [
     "C125",
     "y_vars[A6]"
    ],
    [
     "C126",
     "y_vars[A7]"
    ],
    [
     "C127",
     "y_vars[A8]"
    ],
    [
     "C128",
     "y_vars[A9]"
    ],
    [
     "C129",
     "y_vars[A10]"
    ],
    [
     "C130",
     "y_vars[A11]"
    ],
    [
     "C131",
     "y_vars[A12]"
    ],
    [
     "C132",
     "y_vars[A13]"
    ],
    [
     "C133",
     "y_vars[A14]"
    ],
    [
     "C134",
     "y_vars[A15]"
    ]
   ],
   "primary_mapping": {
    "x[0,0]": "x_vars[A1,B1]",
    "x[0,1]": "x_vars[A1,B2]",
    "x[0,2]": "x_vars[A1,B3]",
    "x[0,3]": "x_vars[A1,B4]",
    "x[0,4]": "x_vars[A1,B5]",
    "x[0,5]": "x_vars[A1,B6]",
    "x[0,6]": "x_vars[A1,B7]",
    "x[0,7]": "x_vars[A1,B8]",
    "x[1,0]": "x_vars[A2,B1]",
    "x[1,1]": "x_vars[A2,B2]",
    "x[1,2]": "x_vars[A2,B3]",
    "x[1,3]": "x_vars[A2,B4]",
    "x[1,4]": "x_vars[A2,B5]",
    "x[1,5]": "x_vars[A2,B6]",
    "x[1,6]": "x_vars[A2,B7]",
    "x[1,7]": "x_vars[A2,B8]",
    "x[2,0]": "x_vars[A3,B1]",
    "x[2,1]": "x_vars[A3,B2]",
    "x[2,2]": "x_vars[A3,B3]",
    "x[2,3]": "x_vars[A3,B4]",
    "x[2,4]": "x_vars[A3,B5]",
    "x[2,5]": "x_vars[A3,B6]",
    "x[2,6]": "x_vars[A3,B7]",
    "x[2,7]": "x_vars[A3,B8]",
    "x[3,0]": "x_vars[A4,B1]",
    "x[3,1]": "x_vars[A4,B2]",
    "x[3,2]": "x_vars[A4,B3]",
    "x[3,3]": "x_vars[A4,B4]",
    "x[3,4]": "x_vars[A4,B5]",
    "x[3,5]": "x_vars[A4,B6]",
    "x[3,6]": "x_vars[A4,B7]",
    "x[3,7]": "x_vars[A4,B8]",
    "x[4,0]": "x_vars[A5,B1]",
    "x[4,1]": "x_vars[A5,B2]",
    "x[4,2]": "x_vars[A5,B3]",
    "x[4,3]": "x_vars[A5,B4]",
    "x[4,4]": "x_vars[A5,B5]",
    "x[4,5]": "x_vars[A5,B6]",
    "x[4,6]": "x_vars[A5,B7]",
    "x[4,7]": "x_vars[A5,B8]",
    "x[5,0]": "x_vars[A6,B1]",
    "x[5,1]": "x_vars[A6,B2]",
    "x[5,2]": "x_vars[A6,B3]",
    "x[5,3]": "x_vars[A6,B4]",
    "x[5,4]": "x_vars[A6,B5]",
    "x[5,5]": "x_vars[A6,B6]",
    "x[5,6]": "x_vars[A6,B7]",
    "x[5,7]": "x_vars[A6,B8]",
    "x[6,0]": "x_vars[A7,B1]",
    "x[6,1]": "x_vars[A7,B2]",
    "x[6,2]": "x_vars[A7,B3]",
    "x[6,3]": "x_vars[A7,B4]",
    "x[6,4]": "x_vars[A7,B5]",
    "x[6,5]": "x_vars[A7,B6]",
    "x[6,6]": "x_vars[A7,B7]",
    "x[6,7]": "x_vars[A7,B8]",
    "x[7,0]": "x_vars[A8,B1]",
    "x[7,1]": "x_vars[A8,B2]",
    "x[7,2]": "x_vars[A8,B3]",
    "x[7,3]": "x_vars[A8,B4]",
    "x[7,4]": "x_vars[A8,B5]",
    "x[7,5]": "x_vars[A8,B6]",
    "x[7,6]": "x_vars[A8,B7]",
    "x[7,7]": "x_vars[A8,B8]",
    "x[8,0]": "x_vars[A9,B1]",
    "x[8,1]": "x_vars[A9,B2]",
    "x[8,2]": "x_vars[A9,B3]",
    "x[8,3]": "x_vars[A9,B4]",
    "x[8,4]": "x_vars[A9,B5]",
    "x[8,5]": "x_vars[A9,B6]",
    "x[8,6]": "x_vars[A9,B7]",
    "x[8,7]": "x_vars[A9,B8]",
    "x[9,0]": "x_vars[A10,B1]",
    "x[9,1]": "x_vars[A10,B2]",
    "x[9,2]": "x_vars[A10,B3]",
    "x[9,3]": "x_vars[A10,B4]",
    "x[9,4]": "x_vars[A10,B5]",
    "x[9,5]": "x_vars[A10,B6]",
    "x[9,6]": "x_vars[A10,B7]",
    "x[9,7]": "x_vars[A10,B8]",
    "x[10,0]": "x_vars[A11,B1]",
    "x[10,1]": "x_vars[A11,B2]",
    "x[10,2]": "x_vars[A11,B3]",
    "x[10,3]": "x_vars[A11,B4]",
    "x[10,4]": "x_vars[A11,B5]",
    "x[10,5]": "x_vars[A11,B6]",
    "x[10,6]": "x_vars[A11,B7]",
    "x[10,7]": "x_vars[A11,B8]",
    "x[11,0]": "x_vars[A12,B1]",
    "x[11,1]": "x_vars[A12,B2]",
    "x[11,2]": "x_vars[A12,B3]",
    "x[11,3]": "x_vars[A12,B4]",
    "x[11,4]": "x_vars[A12,B5]",
    "x[11,5]": "x_vars[A12,B6]",
    "x[11,6]": "x_vars[A12,B7]",
    "x[11,7]": "x_vars[A12,B8]",
    "x[12,0]": "x_vars[A13,B1]",
    "x[12,1]": "x_vars[A13,B2]",
    "x[12,2]": "x_vars[A13,B3]",
    "x[12,3]": "x_vars[A13,B4]",
    "x[12,4]": "x_vars[A13,B5]",
    "x[12,5]": "x_vars[A13,B6]",
    "x[12,6]": "x_vars[A13,B7]",
    "x[12,7]": "x_vars[A13,B8]",
    "x[13,0]": "x_vars[A14,B1]",
    "x[13,1]": "x_vars[A14,B2]",
    "x[13,2]": "x_vars[A14,B3]",
    "x[13,3]": "x_vars[A14,B4]",
    "x[13,4]": "x_vars[A14,B5]",
    "x[13,5]": "x_vars[A14,B6]",
    "x[13,6]": "x_vars[A14,B7]",
    "x[13,7]": "x_vars[A14,B8]",
    "x[14,0]": "x_vars[A15,B1]",
    "x[14,1]": "x_vars[A15,B2]",
    "x[14,2]": "x_vars[A15,B3]",
    "x[14,3]": "x_vars[A15,B4]",
    "x[14,4]": "x_vars[A15,B5]",
    "x[14,5]": "x_vars[A15,B6]",
    "x[14,6]": "x_vars[A15,B7]",
    "x[14,7]": "x_vars[A15,B8]",
    "y[1]": "y_vars[A2]",
    "y[2]": "y_vars[A3]",
    "y[3]": "y_vars[A4]",
    "y[4]": "y_vars[A5]",
    "y[5]": "y_vars[A6]",
    "y[6]": "y_vars[A7]",
    "y[7]": "y_vars[A8]",
    "y[8]": "y_vars[A9]",
    "y[9]": "y_vars[A10]",
    "y[10]": "y_vars[A11]",
    "y[11]": "y_vars[A12]",
    "y[12]": "y_vars[A13]",
    "y[13]": "y_vars[A14]",
    "y[14]": "y_vars[A15]",
    "y[0]": "y_vars[A1]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-080": {
   "source_code_sha256": "e8db98f6f134db1cf4857758b4fc788c915e45387ef95a2f1ff849047d9bd13d",
   "original_solution_sha256": "00d8fd9923ac69218bd29f0a7546963723f227d2f32e3500ea1002a6c8d19242",
   "bindings": [
    [
     "C0",
     "x_vars[1,1]"
    ],
    [
     "C1",
     "x_vars[1,2]"
    ],
    [
     "C2",
     "x_vars[1,3]"
    ],
    [
     "C3",
     "x_vars[1,4]"
    ],
    [
     "C4",
     "x_vars[2,1]"
    ],
    [
     "C5",
     "x_vars[2,2]"
    ],
    [
     "C6",
     "x_vars[2,3]"
    ],
    [
     "C7",
     "x_vars[2,4]"
    ],
    [
     "C8",
     "x_vars[3,1]"
    ],
    [
     "C9",
     "x_vars[3,2]"
    ],
    [
     "C10",
     "x_vars[3,3]"
    ],
    [
     "C11",
     "x_vars[3,4]"
    ],
    [
     "C12",
     "x_vars[4,1]"
    ],
    [
     "C13",
     "x_vars[4,2]"
    ],
    [
     "C14",
     "x_vars[4,3]"
    ],
    [
     "C15",
     "x_vars[4,4]"
    ],
    [
     "C16",
     "x_vars[5,1]"
    ],
    [
     "C17",
     "x_vars[5,2]"
    ],
    [
     "C18",
     "x_vars[5,3]"
    ],
    [
     "C19",
     "x_vars[5,4]"
    ],
    [
     "C20",
     "x_vars[6,1]"
    ],
    [
     "C21",
     "x_vars[6,2]"
    ],
    [
     "C22",
     "x_vars[6,3]"
    ],
    [
     "C23",
     "x_vars[6,4]"
    ],
    [
     "C24",
     "x_vars[7,1]"
    ],
    [
     "C25",
     "x_vars[7,2]"
    ],
    [
     "C26",
     "x_vars[7,3]"
    ],
    [
     "C27",
     "x_vars[7,4]"
    ],
    [
     "C28",
     "x_vars[8,1]"
    ],
    [
     "C29",
     "x_vars[8,2]"
    ],
    [
     "C30",
     "x_vars[8,3]"
    ],
    [
     "C31",
     "x_vars[8,4]"
    ],
    [
     "C32",
     "x_vars[9,1]"
    ],
    [
     "C33",
     "x_vars[9,2]"
    ],
    [
     "C34",
     "x_vars[9,3]"
    ],
    [
     "C35",
     "x_vars[9,4]"
    ],
    [
     "C36",
     "x_vars[10,1]"
    ],
    [
     "C37",
     "x_vars[10,2]"
    ],
    [
     "C38",
     "x_vars[10,3]"
    ],
    [
     "C39",
     "x_vars[10,4]"
    ],
    [
     "C40",
     "y_vars[1,1]"
    ],
    [
     "C41",
     "y_vars[1,2]"
    ],
    [
     "C42",
     "y_vars[1,3]"
    ],
    [
     "C43",
     "y_vars[1,4]"
    ],
    [
     "C44",
     "y_vars[2,1]"
    ],
    [
     "C45",
     "y_vars[2,2]"
    ],
    [
     "C46",
     "y_vars[2,3]"
    ],
    [
     "C47",
     "y_vars[2,4]"
    ],
    [
     "C48",
     "y_vars[3,1]"
    ],
    [
     "C49",
     "y_vars[3,2]"
    ],
    [
     "C50",
     "y_vars[3,3]"
    ],
    [
     "C51",
     "y_vars[3,4]"
    ],
    [
     "C52",
     "y_vars[4,1]"
    ],
    [
     "C53",
     "y_vars[4,2]"
    ],
    [
     "C54",
     "y_vars[4,3]"
    ],
    [
     "C55",
     "y_vars[4,4]"
    ],
    [
     "C56",
     "y_vars[5,1]"
    ],
    [
     "C57",
     "y_vars[5,2]"
    ],
    [
     "C58",
     "y_vars[5,3]"
    ],
    [
     "C59",
     "y_vars[5,4]"
    ],
    [
     "C60",
     "y_vars[6,1]"
    ],
    [
     "C61",
     "y_vars[6,2]"
    ],
    [
     "C62",
     "y_vars[6,3]"
    ],
    [
     "C63",
     "y_vars[6,4]"
    ],
    [
     "C64",
     "y_vars[7,1]"
    ],
    [
     "C65",
     "y_vars[7,2]"
    ],
    [
     "C66",
     "y_vars[7,3]"
    ],
    [
     "C67",
     "y_vars[7,4]"
    ],
    [
     "C68",
     "y_vars[8,1]"
    ],
    [
     "C69",
     "y_vars[8,2]"
    ],
    [
     "C70",
     "y_vars[8,3]"
    ],
    [
     "C71",
     "y_vars[8,4]"
    ],
    [
     "C72",
     "y_vars[9,1]"
    ],
    [
     "C73",
     "y_vars[9,2]"
    ],
    [
     "C74",
     "y_vars[9,3]"
    ],
    [
     "C75",
     "y_vars[9,4]"
    ],
    [
     "C76",
     "y_vars[10,1]"
    ],
    [
     "C77",
     "y_vars[10,2]"
    ],
    [
     "C78",
     "y_vars[10,3]"
    ],
    [
     "C79",
     "y_vars[10,4]"
    ],
    [
     "C80",
     "s_vars[1,1]"
    ],
    [
     "C81",
     "s_vars[1,2]"
    ],
    [
     "C82",
     "s_vars[1,3]"
    ],
    [
     "C83",
     "s_vars[1,4]"
    ],
    [
     "C84",
     "s_vars[2,1]"
    ],
    [
     "C85",
     "s_vars[2,2]"
    ],
    [
     "C86",
     "s_vars[2,3]"
    ],
    [
     "C87",
     "s_vars[2,4]"
    ],
    [
     "C88",
     "s_vars[3,1]"
    ],
    [
     "C89",
     "s_vars[3,2]"
    ],
    [
     "C90",
     "s_vars[3,3]"
    ],
    [
     "C91",
     "s_vars[3,4]"
    ],
    [
     "C92",
     "s_vars[4,1]"
    ],
    [
     "C93",
     "s_vars[4,2]"
    ],
    [
     "C94",
     "s_vars[4,3]"
    ],
    [
     "C95",
     "s_vars[4,4]"
    ],
    [
     "C96",
     "s_vars[5,1]"
    ],
    [
     "C97",
     "s_vars[5,2]"
    ],
    [
     "C98",
     "s_vars[5,3]"
    ],
    [
     "C99",
     "s_vars[5,4]"
    ],
    [
     "C100",
     "s_vars[6,1]"
    ],
    [
     "C101",
     "s_vars[6,2]"
    ],
    [
     "C102",
     "s_vars[6,3]"
    ],
    [
     "C103",
     "s_vars[6,4]"
    ],
    [
     "C104",
     "s_vars[7,1]"
    ],
    [
     "C105",
     "s_vars[7,2]"
    ],
    [
     "C106",
     "s_vars[7,3]"
    ],
    [
     "C107",
     "s_vars[7,4]"
    ],
    [
     "C108",
     "s_vars[8,1]"
    ],
    [
     "C109",
     "s_vars[8,2]"
    ],
    [
     "C110",
     "s_vars[8,3]"
    ],
    [
     "C111",
     "s_vars[8,4]"
    ],
    [
     "C112",
     "s_vars[9,1]"
    ],
    [
     "C113",
     "s_vars[9,2]"
    ],
    [
     "C114",
     "s_vars[9,3]"
    ],
    [
     "C115",
     "s_vars[9,4]"
    ],
    [
     "C116",
     "s_vars[10,1]"
    ],
    [
     "C117",
     "s_vars[10,2]"
    ],
    [
     "C118",
     "s_vars[10,3]"
    ],
    [
     "C119",
     "s_vars[10,4]"
    ]
   ],
   "primary_mapping": {
    "w[1,1]": "x_vars[1,1]",
    "w[1,2]": "x_vars[1,2]",
    "w[1,3]": "x_vars[1,3]",
    "w[1,4]": "x_vars[1,4]",
    "w[2,1]": "x_vars[2,1]",
    "w[2,2]": "x_vars[2,2]",
    "w[2,3]": "x_vars[2,3]",
    "w[2,4]": "x_vars[2,4]",
    "w[3,1]": "x_vars[3,1]",
    "w[3,2]": "x_vars[3,2]",
    "w[3,3]": "x_vars[3,3]",
    "w[3,4]": "x_vars[3,4]",
    "w[4,1]": "x_vars[4,1]",
    "w[4,2]": "x_vars[4,2]",
    "w[4,3]": "x_vars[4,3]",
    "w[4,4]": "x_vars[4,4]",
    "w[5,1]": "x_vars[5,1]",
    "w[5,2]": "x_vars[5,2]",
    "w[5,3]": "x_vars[5,3]",
    "w[5,4]": "x_vars[5,4]",
    "w[6,1]": "x_vars[6,1]",
    "w[6,2]": "x_vars[6,2]",
    "w[6,3]": "x_vars[6,3]",
    "w[6,4]": "x_vars[6,4]",
    "w[7,1]": "x_vars[7,1]",
    "w[7,2]": "x_vars[7,2]",
    "w[7,3]": "x_vars[7,3]",
    "w[7,4]": "x_vars[7,4]",
    "w[8,1]": "x_vars[8,1]",
    "w[8,2]": "x_vars[8,2]",
    "w[8,3]": "x_vars[8,3]",
    "w[8,4]": "x_vars[8,4]",
    "w[9,1]": "x_vars[9,1]",
    "w[9,2]": "x_vars[9,2]",
    "w[9,3]": "x_vars[9,3]",
    "w[9,4]": "x_vars[9,4]",
    "w[10,1]": "x_vars[10,1]",
    "w[10,2]": "x_vars[10,2]",
    "w[10,3]": "x_vars[10,3]",
    "w[10,4]": "x_vars[10,4]",
    "y[1,1]": "y_vars[1,1]",
    "y[1,2]": "y_vars[1,2]",
    "y[1,3]": "y_vars[1,3]",
    "y[1,4]": "y_vars[1,4]",
    "y[2,1]": "y_vars[2,1]",
    "y[2,2]": "y_vars[2,2]",
    "y[2,3]": "y_vars[2,3]",
    "y[2,4]": "y_vars[2,4]",
    "y[3,1]": "y_vars[3,1]",
    "y[3,2]": "y_vars[3,2]",
    "y[3,3]": "y_vars[3,3]",
    "y[3,4]": "y_vars[3,4]",
    "y[4,1]": "y_vars[4,1]",
    "y[4,2]": "y_vars[4,2]",
    "y[4,3]": "y_vars[4,3]",
    "y[4,4]": "y_vars[4,4]",
    "y[5,1]": "y_vars[5,1]",
    "y[5,2]": "y_vars[5,2]",
    "y[5,3]": "y_vars[5,3]",
    "y[5,4]": "y_vars[5,4]",
    "y[6,1]": "y_vars[6,1]",
    "y[6,2]": "y_vars[6,2]",
    "y[6,3]": "y_vars[6,3]",
    "y[6,4]": "y_vars[6,4]",
    "y[7,1]": "y_vars[7,1]",
    "y[7,2]": "y_vars[7,2]",
    "y[7,3]": "y_vars[7,3]",
    "y[7,4]": "y_vars[7,4]",
    "y[8,1]": "y_vars[8,1]",
    "y[8,2]": "y_vars[8,2]",
    "y[8,3]": "y_vars[8,3]",
    "y[8,4]": "y_vars[8,4]",
    "y[9,1]": "y_vars[9,1]",
    "y[9,2]": "y_vars[9,2]",
    "y[9,3]": "y_vars[9,3]",
    "y[9,4]": "y_vars[9,4]",
    "y[10,1]": "y_vars[10,1]",
    "y[10,2]": "y_vars[10,2]",
    "y[10,3]": "y_vars[10,3]",
    "y[10,4]": "y_vars[10,4]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-081": {
   "source_code_sha256": "5180a1b19c2b7018182f74d99f6d8039aaca61e32d39d0bc218b34106723a03b",
   "original_solution_sha256": "8b4e8159dab27db39243def6d49337e88a112004cf85a5071f04281927c7bbd2",
   "bindings": [
    [
     "C0",
     "x_vars[Oatmeal]"
    ],
    [
     "C1",
     "x_vars[Milk]"
    ],
    [
     "C2",
     "x_vars[Egg]"
    ],
    [
     "C3",
     "x_vars[Banana]"
    ],
    [
     "C4",
     "x_vars[Grain_001]"
    ],
    [
     "C5",
     "x_vars[Grain_002]"
    ],
    [
     "C6",
     "x_vars[Grain_003]"
    ],
    [
     "C7",
     "x_vars[Grain_004]"
    ],
    [
     "C8",
     "x_vars[Grain_005]"
    ],
    [
     "C9",
     "x_vars[Grain_006]"
    ],
    [
     "C10",
     "x_vars[Grain_007]"
    ],
    [
     "C11",
     "x_vars[Grain_008]"
    ],
    [
     "C12",
     "x_vars[Grain_009]"
    ],
    [
     "C13",
     "x_vars[Grain_010]"
    ],
    [
     "C14",
     "x_vars[Grain_011]"
    ],
    [
     "C15",
     "x_vars[Grain_012]"
    ],
    [
     "C16",
     "x_vars[Grain_013]"
    ],
    [
     "C17",
     "x_vars[Grain_014]"
    ],
    [
     "C18",
     "x_vars[Grain_015]"
    ],
    [
     "C19",
     "x_vars[Grain_016]"
    ],
    [
     "C20",
     "x_vars[Grain_017]"
    ],
    [
     "C21",
     "x_vars[Grain_018]"
    ],
    [
     "C22",
     "x_vars[Grain_019]"
    ],
    [
     "C23",
     "x_vars[Grain_020]"
    ],
    [
     "C24",
     "x_vars[Grain_021]"
    ],
    [
     "C25",
     "x_vars[Grain_022]"
    ],
    [
     "C26",
     "x_vars[Grain_023]"
    ],
    [
     "C27",
     "x_vars[Grain_024]"
    ],
    [
     "C28",
     "x_vars[Grain_025]"
    ],
    [
     "C29",
     "x_vars[Dairy_001]"
    ],
    [
     "C30",
     "x_vars[Dairy_002]"
    ],
    [
     "C31",
     "x_vars[Dairy_003]"
    ],
    [
     "C32",
     "x_vars[Dairy_004]"
    ],
    [
     "C33",
     "x_vars[Dairy_005]"
    ],
    [
     "C34",
     "x_vars[Dairy_006]"
    ],
    [
     "C35",
     "x_vars[Dairy_007]"
    ],
    [
     "C36",
     "x_vars[Dairy_008]"
    ],
    [
     "C37",
     "x_vars[Dairy_009]"
    ],
    [
     "C38",
     "x_vars[Dairy_010]"
    ],
    [
     "C39",
     "x_vars[Dairy_011]"
    ],
    [
     "C40",
     "x_vars[Dairy_012]"
    ],
    [
     "C41",
     "x_vars[Dairy_013]"
    ],
    [
     "C42",
     "x_vars[Dairy_014]"
    ],
    [
     "C43",
     "x_vars[Dairy_015]"
    ],
    [
     "C44",
     "x_vars[Dairy_016]"
    ],
    [
     "C45",
     "x_vars[Dairy_017]"
    ],
    [
     "C46",
     "x_vars[Dairy_018]"
    ],
    [
     "C47",
     "x_vars[Dairy_019]"
    ],
    [
     "C48",
     "x_vars[Dairy_020]"
    ],
    [
     "C49",
     "x_vars[Dairy_021]"
    ],
    [
     "C50",
     "x_vars[Dairy_022]"
    ],
    [
     "C51",
     "x_vars[Dairy_023]"
    ],
    [
     "C52",
     "x_vars[Dairy_024]"
    ],
    [
     "C53",
     "x_vars[Dairy_025]"
    ],
    [
     "C54",
     "x_vars[Meat_001]"
    ],
    [
     "C55",
     "x_vars[Meat_002]"
    ],
    [
     "C56",
     "x_vars[Meat_003]"
    ],
    [
     "C57",
     "x_vars[Meat_004]"
    ],
    [
     "C58",
     "x_vars[Meat_005]"
    ],
    [
     "C59",
     "x_vars[Meat_006]"
    ],
    [
     "C60",
     "x_vars[Meat_007]"
    ],
    [
     "C61",
     "x_vars[Meat_008]"
    ],
    [
     "C62",
     "x_vars[Meat_009]"
    ],
    [
     "C63",
     "x_vars[Meat_010]"
    ],
    [
     "C64",
     "x_vars[Meat_011]"
    ],
    [
     "C65",
     "x_vars[Meat_012]"
    ],
    [
     "C66",
     "x_vars[Meat_013]"
    ],
    [
     "C67",
     "x_vars[Meat_014]"
    ],
    [
     "C68",
     "x_vars[Meat_015]"
    ],
    [
     "C69",
     "x_vars[Meat_016]"
    ],
    [
     "C70",
     "x_vars[Meat_017]"
    ],
    [
     "C71",
     "x_vars[Meat_018]"
    ],
    [
     "C72",
     "x_vars[Meat_019]"
    ],
    [
     "C73",
     "x_vars[Meat_020]"
    ],
    [
     "C74",
     "x_vars[Meat_021]"
    ],
    [
     "C75",
     "x_vars[Meat_022]"
    ],
    [
     "C76",
     "x_vars[Meat_023]"
    ],
    [
     "C77",
     "x_vars[Meat_024]"
    ],
    [
     "C78",
     "x_vars[Meat_025]"
    ],
    [
     "C79",
     "x_vars[Fruit_001]"
    ],
    [
     "C80",
     "x_vars[Fruit_002]"
    ],
    [
     "C81",
     "x_vars[Fruit_003]"
    ],
    [
     "C82",
     "x_vars[Fruit_004]"
    ],
    [
     "C83",
     "x_vars[Fruit_005]"
    ],
    [
     "C84",
     "x_vars[Fruit_006]"
    ],
    [
     "C85",
     "x_vars[Fruit_007]"
    ],
    [
     "C86",
     "x_vars[Fruit_008]"
    ],
    [
     "C87",
     "x_vars[Fruit_009]"
    ],
    [
     "C88",
     "x_vars[Fruit_010]"
    ],
    [
     "C89",
     "x_vars[Fruit_011]"
    ],
    [
     "C90",
     "x_vars[Fruit_012]"
    ],
    [
     "C91",
     "x_vars[Fruit_013]"
    ],
    [
     "C92",
     "x_vars[Fruit_014]"
    ],
    [
     "C93",
     "x_vars[Fruit_015]"
    ],
    [
     "C94",
     "x_vars[Fruit_016]"
    ],
    [
     "C95",
     "x_vars[Fruit_017]"
    ],
    [
     "C96",
     "x_vars[Fruit_018]"
    ],
    [
     "C97",
     "x_vars[Fruit_019]"
    ],
    [
     "C98",
     "x_vars[Fruit_020]"
    ],
    [
     "C99",
     "x_vars[Fruit_021]"
    ],
    [
     "C100",
     "x_vars[Fruit_022]"
    ],
    [
     "C101",
     "x_vars[Fruit_023]"
    ],
    [
     "C102",
     "x_vars[Fruit_024]"
    ],
    [
     "C103",
     "x_vars[Fruit_025]"
    ],
    [
     "C104",
     "x_vars[Veg_001]"
    ],
    [
     "C105",
     "x_vars[Veg_002]"
    ],
    [
     "C106",
     "x_vars[Veg_003]"
    ],
    [
     "C107",
     "x_vars[Veg_004]"
    ],
    [
     "C108",
     "x_vars[Veg_005]"
    ],
    [
     "C109",
     "x_vars[Veg_006]"
    ],
    [
     "C110",
     "x_vars[Veg_007]"
    ],
    [
     "C111",
     "x_vars[Veg_008]"
    ],
    [
     "C112",
     "x_vars[Veg_009]"
    ],
    [
     "C113",
     "x_vars[Veg_010]"
    ],
    [
     "C114",
     "x_vars[Veg_011]"
    ],
    [
     "C115",
     "x_vars[Veg_012]"
    ],
    [
     "C116",
     "x_vars[Veg_013]"
    ],
    [
     "C117",
     "x_vars[Veg_014]"
    ],
    [
     "C118",
     "x_vars[Veg_015]"
    ],
    [
     "C119",
     "x_vars[Veg_016]"
    ]
   ],
   "primary_mapping": {
    "x[Oatmeal]": "x_vars[Oatmeal]",
    "x[Milk]": "x_vars[Milk]",
    "x[Egg]": "x_vars[Egg]",
    "x[Banana]": "x_vars[Banana]",
    "x[Grain_001]": "x_vars[Grain_001]",
    "x[Grain_002]": "x_vars[Grain_002]",
    "x[Grain_003]": "x_vars[Grain_003]",
    "x[Grain_004]": "x_vars[Grain_004]",
    "x[Grain_005]": "x_vars[Grain_005]",
    "x[Grain_006]": "x_vars[Grain_006]",
    "x[Grain_007]": "x_vars[Grain_007]",
    "x[Grain_008]": "x_vars[Grain_008]",
    "x[Grain_009]": "x_vars[Grain_009]",
    "x[Grain_010]": "x_vars[Grain_010]",
    "x[Grain_011]": "x_vars[Grain_011]",
    "x[Grain_012]": "x_vars[Grain_012]",
    "x[Grain_013]": "x_vars[Grain_013]",
    "x[Grain_014]": "x_vars[Grain_014]",
    "x[Grain_015]": "x_vars[Grain_015]",
    "x[Grain_016]": "x_vars[Grain_016]",
    "x[Grain_017]": "x_vars[Grain_017]",
    "x[Grain_018]": "x_vars[Grain_018]",
    "x[Grain_019]": "x_vars[Grain_019]",
    "x[Grain_020]": "x_vars[Grain_020]",
    "x[Grain_021]": "x_vars[Grain_021]",
    "x[Grain_022]": "x_vars[Grain_022]",
    "x[Grain_023]": "x_vars[Grain_023]",
    "x[Grain_024]": "x_vars[Grain_024]",
    "x[Grain_025]": "x_vars[Grain_025]",
    "x[Dairy_001]": "x_vars[Dairy_001]",
    "x[Dairy_002]": "x_vars[Dairy_002]",
    "x[Dairy_003]": "x_vars[Dairy_003]",
    "x[Dairy_004]": "x_vars[Dairy_004]",
    "x[Dairy_005]": "x_vars[Dairy_005]",
    "x[Dairy_006]": "x_vars[Dairy_006]",
    "x[Dairy_007]": "x_vars[Dairy_007]",
    "x[Dairy_008]": "x_vars[Dairy_008]",
    "x[Dairy_009]": "x_vars[Dairy_009]",
    "x[Dairy_010]": "x_vars[Dairy_010]",
    "x[Dairy_011]": "x_vars[Dairy_011]",
    "x[Dairy_012]": "x_vars[Dairy_012]",
    "x[Dairy_013]": "x_vars[Dairy_013]",
    "x[Dairy_014]": "x_vars[Dairy_014]",
    "x[Dairy_015]": "x_vars[Dairy_015]",
    "x[Dairy_016]": "x_vars[Dairy_016]",
    "x[Dairy_017]": "x_vars[Dairy_017]",
    "x[Dairy_018]": "x_vars[Dairy_018]",
    "x[Dairy_019]": "x_vars[Dairy_019]",
    "x[Dairy_020]": "x_vars[Dairy_020]",
    "x[Dairy_021]": "x_vars[Dairy_021]",
    "x[Dairy_022]": "x_vars[Dairy_022]",
    "x[Dairy_023]": "x_vars[Dairy_023]",
    "x[Dairy_024]": "x_vars[Dairy_024]",
    "x[Dairy_025]": "x_vars[Dairy_025]",
    "x[Meat_001]": "x_vars[Meat_001]",
    "x[Meat_002]": "x_vars[Meat_002]",
    "x[Meat_003]": "x_vars[Meat_003]",
    "x[Meat_004]": "x_vars[Meat_004]",
    "x[Meat_005]": "x_vars[Meat_005]",
    "x[Meat_006]": "x_vars[Meat_006]",
    "x[Meat_007]": "x_vars[Meat_007]",
    "x[Meat_008]": "x_vars[Meat_008]",
    "x[Meat_009]": "x_vars[Meat_009]",
    "x[Meat_010]": "x_vars[Meat_010]",
    "x[Meat_011]": "x_vars[Meat_011]",
    "x[Meat_012]": "x_vars[Meat_012]",
    "x[Meat_013]": "x_vars[Meat_013]",
    "x[Meat_014]": "x_vars[Meat_014]",
    "x[Meat_015]": "x_vars[Meat_015]",
    "x[Meat_016]": "x_vars[Meat_016]",
    "x[Meat_017]": "x_vars[Meat_017]",
    "x[Meat_018]": "x_vars[Meat_018]",
    "x[Meat_019]": "x_vars[Meat_019]",
    "x[Meat_020]": "x_vars[Meat_020]",
    "x[Meat_021]": "x_vars[Meat_021]",
    "x[Meat_022]": "x_vars[Meat_022]",
    "x[Meat_023]": "x_vars[Meat_023]",
    "x[Meat_024]": "x_vars[Meat_024]",
    "x[Meat_025]": "x_vars[Meat_025]",
    "x[Fruit_001]": "x_vars[Fruit_001]",
    "x[Fruit_002]": "x_vars[Fruit_002]",
    "x[Fruit_003]": "x_vars[Fruit_003]",
    "x[Fruit_004]": "x_vars[Fruit_004]",
    "x[Fruit_005]": "x_vars[Fruit_005]",
    "x[Fruit_006]": "x_vars[Fruit_006]",
    "x[Fruit_007]": "x_vars[Fruit_007]",
    "x[Fruit_008]": "x_vars[Fruit_008]",
    "x[Fruit_009]": "x_vars[Fruit_009]",
    "x[Fruit_010]": "x_vars[Fruit_010]",
    "x[Fruit_011]": "x_vars[Fruit_011]",
    "x[Fruit_012]": "x_vars[Fruit_012]",
    "x[Fruit_013]": "x_vars[Fruit_013]",
    "x[Fruit_014]": "x_vars[Fruit_014]",
    "x[Fruit_015]": "x_vars[Fruit_015]",
    "x[Fruit_016]": "x_vars[Fruit_016]",
    "x[Fruit_017]": "x_vars[Fruit_017]",
    "x[Fruit_018]": "x_vars[Fruit_018]",
    "x[Fruit_019]": "x_vars[Fruit_019]",
    "x[Fruit_020]": "x_vars[Fruit_020]",
    "x[Fruit_021]": "x_vars[Fruit_021]",
    "x[Fruit_022]": "x_vars[Fruit_022]",
    "x[Fruit_023]": "x_vars[Fruit_023]",
    "x[Fruit_024]": "x_vars[Fruit_024]",
    "x[Fruit_025]": "x_vars[Fruit_025]",
    "x[Veg_001]": "x_vars[Veg_001]",
    "x[Veg_002]": "x_vars[Veg_002]",
    "x[Veg_003]": "x_vars[Veg_003]",
    "x[Veg_004]": "x_vars[Veg_004]",
    "x[Veg_005]": "x_vars[Veg_005]",
    "x[Veg_006]": "x_vars[Veg_006]",
    "x[Veg_007]": "x_vars[Veg_007]",
    "x[Veg_008]": "x_vars[Veg_008]",
    "x[Veg_009]": "x_vars[Veg_009]",
    "x[Veg_010]": "x_vars[Veg_010]",
    "x[Veg_011]": "x_vars[Veg_011]",
    "x[Veg_012]": "x_vars[Veg_012]",
    "x[Veg_013]": "x_vars[Veg_013]",
    "x[Veg_014]": "x_vars[Veg_014]",
    "x[Veg_015]": "x_vars[Veg_015]",
    "x[Veg_016]": "x_vars[Veg_016]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-082": {
   "source_code_sha256": "5ca59fe46a035b1a52d2e94b8ff98225a8ca24350cf2162d3f6f31a2de03313e",
   "original_solution_sha256": "87752db3814dc59eb7fe379d45bb8ac8fc68b7473525457aa99fd5c4aae8a956",
   "bindings": [
    [
     "C0",
     "x_vars[Depot,A]"
    ],
    [
     "C1",
     "x_vars[Depot,B]"
    ],
    [
     "C2",
     "x_vars[Depot,C]"
    ],
    [
     "C3",
     "x_vars[A,Depot]"
    ],
    [
     "C4",
     "x_vars[A,B]"
    ],
    [
     "C5",
     "x_vars[A,C]"
    ],
    [
     "C6",
     "x_vars[B,Depot]"
    ],
    [
     "C7",
     "x_vars[B,A]"
    ],
    [
     "C8",
     "x_vars[B,C]"
    ],
    [
     "C9",
     "x_vars[C,Depot]"
    ],
    [
     "C10",
     "x_vars[C,A]"
    ],
    [
     "C11",
     "x_vars[C,B]"
    ],
    [
     "C12",
     "u_vars[A]"
    ],
    [
     "C13",
     "u_vars[B]"
    ],
    [
     "C14",
     "u_vars[C]"
    ]
   ],
   "primary_mapping": {
    "X[0,1]": "x_vars[Depot,A]",
    "X[0,2]": "x_vars[Depot,B]",
    "X[0,3]": "x_vars[Depot,C]",
    "X[1,0]": "x_vars[A,Depot]",
    "X[1,2]": "x_vars[A,B]",
    "X[1,3]": "x_vars[A,C]",
    "X[2,0]": "x_vars[B,Depot]",
    "X[2,1]": "x_vars[B,A]",
    "X[2,3]": "x_vars[B,C]",
    "X[3,0]": "x_vars[C,Depot]",
    "X[3,1]": "x_vars[C,A]",
    "X[3,2]": "x_vars[C,B]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-083": {
   "source_code_sha256": "fc060baa88cc5e3045b01bd122a5c1d4ecb79e69d0b9083f75a79cdf06a4d0cc",
   "original_solution_sha256": "bf7686c5440b2ef508aba1076bfcd3fb03f88530ae24035395df41787f0787f1",
   "bindings": [
    [
     "C0",
     "x_vars[1,A]"
    ],
    [
     "C1",
     "x_vars[1,B]"
    ],
    [
     "C2",
     "x_vars[1,C]"
    ],
    [
     "C3",
     "x_vars[1,D]"
    ],
    [
     "C4",
     "x_vars[1,E]"
    ],
    [
     "C5",
     "x_vars[1,F]"
    ],
    [
     "C6",
     "x_vars[1,G]"
    ],
    [
     "C7",
     "x_vars[1,H]"
    ],
    [
     "C8",
     "x_vars[1,I]"
    ],
    [
     "C9",
     "x_vars[1,J]"
    ],
    [
     "C10",
     "x_vars[2,A]"
    ],
    [
     "C11",
     "x_vars[2,B]"
    ],
    [
     "C12",
     "x_vars[2,C]"
    ],
    [
     "C13",
     "x_vars[2,D]"
    ],
    [
     "C14",
     "x_vars[2,E]"
    ],
    [
     "C15",
     "x_vars[2,F]"
    ],
    [
     "C16",
     "x_vars[2,G]"
    ],
    [
     "C17",
     "x_vars[2,H]"
    ],
    [
     "C18",
     "x_vars[2,I]"
    ],
    [
     "C19",
     "x_vars[2,J]"
    ],
    [
     "C20",
     "x_vars[3,A]"
    ],
    [
     "C21",
     "x_vars[3,B]"
    ],
    [
     "C22",
     "x_vars[3,C]"
    ],
    [
     "C23",
     "x_vars[3,D]"
    ],
    [
     "C24",
     "x_vars[3,E]"
    ],
    [
     "C25",
     "x_vars[3,F]"
    ],
    [
     "C26",
     "x_vars[3,G]"
    ],
    [
     "C27",
     "x_vars[3,H]"
    ],
    [
     "C28",
     "x_vars[3,I]"
    ],
    [
     "C29",
     "x_vars[3,J]"
    ],
    [
     "C30",
     "x_vars[4,A]"
    ],
    [
     "C31",
     "x_vars[4,B]"
    ],
    [
     "C32",
     "x_vars[4,C]"
    ],
    [
     "C33",
     "x_vars[4,D]"
    ],
    [
     "C34",
     "x_vars[4,E]"
    ],
    [
     "C35",
     "x_vars[4,F]"
    ],
    [
     "C36",
     "x_vars[4,G]"
    ],
    [
     "C37",
     "x_vars[4,H]"
    ],
    [
     "C38",
     "x_vars[4,I]"
    ],
    [
     "C39",
     "x_vars[4,J]"
    ],
    [
     "C40",
     "x_vars[5,A]"
    ],
    [
     "C41",
     "x_vars[5,B]"
    ],
    [
     "C42",
     "x_vars[5,C]"
    ],
    [
     "C43",
     "x_vars[5,D]"
    ],
    [
     "C44",
     "x_vars[5,E]"
    ],
    [
     "C45",
     "x_vars[5,F]"
    ],
    [
     "C46",
     "x_vars[5,G]"
    ],
    [
     "C47",
     "x_vars[5,H]"
    ],
    [
     "C48",
     "x_vars[5,I]"
    ],
    [
     "C49",
     "x_vars[5,J]"
    ],
    [
     "C50",
     "x_vars[6,A]"
    ],
    [
     "C51",
     "x_vars[6,B]"
    ],
    [
     "C52",
     "x_vars[6,C]"
    ],
    [
     "C53",
     "x_vars[6,D]"
    ],
    [
     "C54",
     "x_vars[6,E]"
    ],
    [
     "C55",
     "x_vars[6,F]"
    ],
    [
     "C56",
     "x_vars[6,G]"
    ],
    [
     "C57",
     "x_vars[6,H]"
    ],
    [
     "C58",
     "x_vars[6,I]"
    ],
    [
     "C59",
     "x_vars[6,J]"
    ],
    [
     "C60",
     "x_vars[7,A]"
    ],
    [
     "C61",
     "x_vars[7,B]"
    ],
    [
     "C62",
     "x_vars[7,C]"
    ],
    [
     "C63",
     "x_vars[7,D]"
    ],
    [
     "C64",
     "x_vars[7,E]"
    ],
    [
     "C65",
     "x_vars[7,F]"
    ],
    [
     "C66",
     "x_vars[7,G]"
    ],
    [
     "C67",
     "x_vars[7,H]"
    ],
    [
     "C68",
     "x_vars[7,I]"
    ],
    [
     "C69",
     "x_vars[7,J]"
    ],
    [
     "C70",
     "x_vars[8,A]"
    ],
    [
     "C71",
     "x_vars[8,B]"
    ],
    [
     "C72",
     "x_vars[8,C]"
    ],
    [
     "C73",
     "x_vars[8,D]"
    ],
    [
     "C74",
     "x_vars[8,E]"
    ],
    [
     "C75",
     "x_vars[8,F]"
    ],
    [
     "C76",
     "x_vars[8,G]"
    ],
    [
     "C77",
     "x_vars[8,H]"
    ],
    [
     "C78",
     "x_vars[8,I]"
    ],
    [
     "C79",
     "x_vars[8,J]"
    ],
    [
     "C80",
     "x_vars[9,A]"
    ],
    [
     "C81",
     "x_vars[9,B]"
    ],
    [
     "C82",
     "x_vars[9,C]"
    ],
    [
     "C83",
     "x_vars[9,D]"
    ],
    [
     "C84",
     "x_vars[9,E]"
    ],
    [
     "C85",
     "x_vars[9,F]"
    ],
    [
     "C86",
     "x_vars[9,G]"
    ],
    [
     "C87",
     "x_vars[9,H]"
    ],
    [
     "C88",
     "x_vars[9,I]"
    ],
    [
     "C89",
     "x_vars[9,J]"
    ],
    [
     "C90",
     "x_vars[10,A]"
    ],
    [
     "C91",
     "x_vars[10,B]"
    ],
    [
     "C92",
     "x_vars[10,C]"
    ],
    [
     "C93",
     "x_vars[10,D]"
    ],
    [
     "C94",
     "x_vars[10,E]"
    ],
    [
     "C95",
     "x_vars[10,F]"
    ],
    [
     "C96",
     "x_vars[10,G]"
    ],
    [
     "C97",
     "x_vars[10,H]"
    ],
    [
     "C98",
     "x_vars[10,I]"
    ],
    [
     "C99",
     "x_vars[10,J]"
    ],
    [
     "C100",
     "x_vars[11,A]"
    ],
    [
     "C101",
     "x_vars[11,B]"
    ],
    [
     "C102",
     "x_vars[11,C]"
    ],
    [
     "C103",
     "x_vars[11,D]"
    ],
    [
     "C104",
     "x_vars[11,E]"
    ],
    [
     "C105",
     "x_vars[11,F]"
    ],
    [
     "C106",
     "x_vars[11,G]"
    ],
    [
     "C107",
     "x_vars[11,H]"
    ],
    [
     "C108",
     "x_vars[11,I]"
    ],
    [
     "C109",
     "x_vars[11,J]"
    ],
    [
     "C110",
     "x_vars[12,A]"
    ],
    [
     "C111",
     "x_vars[12,B]"
    ],
    [
     "C112",
     "x_vars[12,C]"
    ],
    [
     "C113",
     "x_vars[12,D]"
    ],
    [
     "C114",
     "x_vars[12,E]"
    ],
    [
     "C115",
     "x_vars[12,F]"
    ],
    [
     "C116",
     "x_vars[12,G]"
    ],
    [
     "C117",
     "x_vars[12,H]"
    ],
    [
     "C118",
     "x_vars[12,I]"
    ],
    [
     "C119",
     "x_vars[12,J]"
    ],
    [
     "C120",
     "y_vars[1]"
    ],
    [
     "C121",
     "y_vars[2]"
    ],
    [
     "C122",
     "y_vars[3]"
    ],
    [
     "C123",
     "y_vars[4]"
    ],
    [
     "C124",
     "y_vars[5]"
    ],
    [
     "C125",
     "y_vars[6]"
    ],
    [
     "C126",
     "y_vars[7]"
    ],
    [
     "C127",
     "y_vars[8]"
    ],
    [
     "C128",
     "y_vars[9]"
    ],
    [
     "C129",
     "y_vars[10]"
    ],
    [
     "C130",
     "y_vars[11]"
    ],
    [
     "C131",
     "y_vars[12]"
    ]
   ],
   "primary_mapping": {
    "x[9,A]": "x_vars[9,A]",
    "x[8,A]": "x_vars[8,A]",
    "x[3,F]": "x_vars[3,F]",
    "x[10,H]": "x_vars[10,H]",
    "x[3,A]": "x_vars[3,A]",
    "x[9,G]": "x_vars[9,G]",
    "x[5,J]": "x_vars[5,J]",
    "x[12,A]": "x_vars[12,A]",
    "x[5,D]": "x_vars[5,D]",
    "x[8,H]": "x_vars[8,H]",
    "x[7,G]": "x_vars[7,G]",
    "x[3,H]": "x_vars[3,H]",
    "x[12,E]": "x_vars[12,E]",
    "x[10,D]": "x_vars[10,D]",
    "x[9,B]": "x_vars[9,B]",
    "x[11,B]": "x_vars[11,B]",
    "x[11,C]": "x_vars[11,C]",
    "x[12,J]": "x_vars[12,J]",
    "x[1,I]": "x_vars[1,I]",
    "x[11,G]": "x_vars[11,G]",
    "x[9,J]": "x_vars[9,J]",
    "x[5,A]": "x_vars[5,A]",
    "x[12,B]": "x_vars[12,B]",
    "x[3,D]": "x_vars[3,D]",
    "x[4,H]": "x_vars[4,H]",
    "x[11,I]": "x_vars[11,I]",
    "x[6,E]": "x_vars[6,E]",
    "x[11,E]": "x_vars[11,E]",
    "x[7,B]": "x_vars[7,B]",
    "x[4,E]": "x_vars[4,E]",
    "x[1,D]": "x_vars[1,D]",
    "x[9,I]": "x_vars[9,I]",
    "x[10,C]": "x_vars[10,C]",
    "x[8,B]": "x_vars[8,B]",
    "x[1,A]": "x_vars[1,A]",
    "x[3,J]": "x_vars[3,J]",
    "x[2,B]": "x_vars[2,B]",
    "x[6,D]": "x_vars[6,D]",
    "x[4,I]": "x_vars[4,I]",
    "x[5,E]": "x_vars[5,E]",
    "x[6,I]": "x_vars[6,I]",
    "x[2,A]": "x_vars[2,A]",
    "x[10,I]": "x_vars[10,I]",
    "x[10,F]": "x_vars[10,F]",
    "x[5,B]": "x_vars[5,B]",
    "x[7,I]": "x_vars[7,I]",
    "x[1,G]": "x_vars[1,G]",
    "x[6,C]": "x_vars[6,C]",
    "x[9,F]": "x_vars[9,F]",
    "x[2,E]": "x_vars[2,E]",
    "x[2,D]": "x_vars[2,D]",
    "x[9,D]": "x_vars[9,D]",
    "x[6,F]": "x_vars[6,F]",
    "x[5,G]": "x_vars[5,G]",
    "x[10,G]": "x_vars[10,G]",
    "x[7,A]": "x_vars[7,A]",
    "x[8,F]": "x_vars[8,F]",
    "x[6,B]": "x_vars[6,B]",
    "x[12,F]": "x_vars[12,F]",
    "x[2,C]": "x_vars[2,C]",
    "x[3,E]": "x_vars[3,E]",
    "x[8,D]": "x_vars[8,D]",
    "x[7,D]": "x_vars[7,D]",
    "x[1,J]": "x_vars[1,J]",
    "x[10,J]": "x_vars[10,J]",
    "x[4,J]": "x_vars[4,J]",
    "x[1,B]": "x_vars[1,B]",
    "x[10,B]": "x_vars[10,B]",
    "x[10,A]": "x_vars[10,A]",
    "x[8,G]": "x_vars[8,G]",
    "x[6,A]": "x_vars[6,A]",
    "x[2,H]": "x_vars[2,H]",
    "x[2,J]": "x_vars[2,J]",
    "x[12,G]": "x_vars[12,G]",
    "x[3,B]": "x_vars[3,B]",
    "x[8,I]": "x_vars[8,I]",
    "x[2,I]": "x_vars[2,I]",
    "x[7,H]": "x_vars[7,H]",
    "x[5,H]": "x_vars[5,H]",
    "x[11,A]": "x_vars[11,A]",
    "x[1,E]": "x_vars[1,E]",
    "x[7,F]": "x_vars[7,F]",
    "x[12,D]": "x_vars[12,D]",
    "x[2,F]": "x_vars[2,F]",
    "x[4,D]": "x_vars[4,D]",
    "x[8,C]": "x_vars[8,C]",
    "x[2,G]": "x_vars[2,G]",
    "x[12,C]": "x_vars[12,C]",
    "x[11,F]": "x_vars[11,F]",
    "x[12,H]": "x_vars[12,H]",
    "x[8,J]": "x_vars[8,J]",
    "x[4,G]": "x_vars[4,G]",
    "x[4,A]": "x_vars[4,A]",
    "x[9,H]": "x_vars[9,H]",
    "x[6,G]": "x_vars[6,G]",
    "x[1,H]": "x_vars[1,H]",
    "x[9,C]": "x_vars[9,C]",
    "x[7,E]": "x_vars[7,E]",
    "x[10,E]": "x_vars[10,E]",
    "x[4,F]": "x_vars[4,F]",
    "x[11,D]": "x_vars[11,D]",
    "x[12,I]": "x_vars[12,I]",
    "x[5,I]": "x_vars[5,I]",
    "x[5,C]": "x_vars[5,C]",
    "x[3,C]": "x_vars[3,C]",
    "x[3,I]": "x_vars[3,I]",
    "x[1,F]": "x_vars[1,F]",
    "x[6,J]": "x_vars[6,J]",
    "x[11,H]": "x_vars[11,H]",
    "x[11,J]": "x_vars[11,J]",
    "x[1,C]": "x_vars[1,C]",
    "x[4,B]": "x_vars[4,B]",
    "x[4,C]": "x_vars[4,C]",
    "x[6,H]": "x_vars[6,H]",
    "x[7,C]": "x_vars[7,C]",
    "x[5,F]": "x_vars[5,F]",
    "x[8,E]": "x_vars[8,E]",
    "x[9,E]": "x_vars[9,E]",
    "x[3,G]": "x_vars[3,G]",
    "x[7,J]": "x_vars[7,J]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-084": {
   "source_code_sha256": "e222cf8f78d36e22b296434f9194aa18e39dc7ecb1c682bca0bf0683e044ba4f",
   "original_solution_sha256": "f476b57efa4909a18430a2b6560c6abbdfecc3ed8bc7bc73385121143a3798a3",
   "bindings": [
    [
     "C0",
     "x_vars[1,1]"
    ],
    [
     "C1",
     "x_vars[1,2]"
    ],
    [
     "C2",
     "x_vars[1,3]"
    ],
    [
     "C3",
     "x_vars[2,1]"
    ],
    [
     "C4",
     "x_vars[2,2]"
    ],
    [
     "C5",
     "x_vars[2,3]"
    ],
    [
     "C6",
     "x_vars[3,1]"
    ],
    [
     "C7",
     "x_vars[3,2]"
    ],
    [
     "C8",
     "x_vars[3,3]"
    ],
    [
     "C9",
     "x_vars[4,1]"
    ],
    [
     "C10",
     "x_vars[4,2]"
    ],
    [
     "C11",
     "x_vars[4,3]"
    ],
    [
     "C12",
     "x_vars[5,1]"
    ],
    [
     "C13",
     "x_vars[5,2]"
    ],
    [
     "C14",
     "x_vars[5,3]"
    ],
    [
     "C15",
     "x_vars[6,1]"
    ],
    [
     "C16",
     "x_vars[6,2]"
    ],
    [
     "C17",
     "x_vars[6,3]"
    ],
    [
     "C18",
     "x_vars[7,1]"
    ],
    [
     "C19",
     "x_vars[7,2]"
    ],
    [
     "C20",
     "x_vars[7,3]"
    ],
    [
     "C21",
     "x_vars[8,1]"
    ],
    [
     "C22",
     "x_vars[8,2]"
    ],
    [
     "C23",
     "x_vars[8,3]"
    ],
    [
     "C24",
     "x_vars[9,1]"
    ],
    [
     "C25",
     "x_vars[9,2]"
    ],
    [
     "C26",
     "x_vars[9,3]"
    ],
    [
     "C27",
     "x_vars[10,1]"
    ],
    [
     "C28",
     "x_vars[10,2]"
    ],
    [
     "C29",
     "x_vars[10,3]"
    ],
    [
     "C30",
     "x_vars[11,1]"
    ],
    [
     "C31",
     "x_vars[11,2]"
    ],
    [
     "C32",
     "x_vars[11,3]"
    ],
    [
     "C33",
     "x_vars[12,1]"
    ],
    [
     "C34",
     "x_vars[12,2]"
    ],
    [
     "C35",
     "x_vars[12,3]"
    ],
    [
     "C36",
     "x_vars[13,1]"
    ],
    [
     "C37",
     "x_vars[13,2]"
    ],
    [
     "C38",
     "x_vars[13,3]"
    ],
    [
     "C39",
     "x_vars[14,1]"
    ],
    [
     "C40",
     "x_vars[14,2]"
    ],
    [
     "C41",
     "x_vars[14,3]"
    ],
    [
     "C42",
     "x_vars[15,1]"
    ],
    [
     "C43",
     "x_vars[15,2]"
    ],
    [
     "C44",
     "x_vars[15,3]"
    ],
    [
     "C45",
     "x_vars[16,1]"
    ],
    [
     "C46",
     "x_vars[16,2]"
    ],
    [
     "C47",
     "x_vars[16,3]"
    ],
    [
     "C48",
     "x_vars[17,1]"
    ],
    [
     "C49",
     "x_vars[17,2]"
    ],
    [
     "C50",
     "x_vars[17,3]"
    ],
    [
     "C51",
     "x_vars[18,1]"
    ],
    [
     "C52",
     "x_vars[18,2]"
    ],
    [
     "C53",
     "x_vars[18,3]"
    ],
    [
     "C54",
     "x_vars[19,1]"
    ],
    [
     "C55",
     "x_vars[19,2]"
    ],
    [
     "C56",
     "x_vars[19,3]"
    ],
    [
     "C57",
     "x_vars[20,1]"
    ],
    [
     "C58",
     "x_vars[20,2]"
    ],
    [
     "C59",
     "x_vars[20,3]"
    ],
    [
     "C60",
     "x_vars[21,1]"
    ],
    [
     "C61",
     "x_vars[21,2]"
    ],
    [
     "C62",
     "x_vars[21,3]"
    ],
    [
     "C63",
     "x_vars[22,1]"
    ],
    [
     "C64",
     "x_vars[22,2]"
    ],
    [
     "C65",
     "x_vars[22,3]"
    ],
    [
     "C66",
     "x_vars[23,1]"
    ],
    [
     "C67",
     "x_vars[23,2]"
    ],
    [
     "C68",
     "x_vars[23,3]"
    ],
    [
     "C69",
     "x_vars[24,1]"
    ],
    [
     "C70",
     "x_vars[24,2]"
    ],
    [
     "C71",
     "x_vars[24,3]"
    ],
    [
     "C72",
     "x_vars[25,1]"
    ],
    [
     "C73",
     "x_vars[25,2]"
    ],
    [
     "C74",
     "x_vars[25,3]"
    ],
    [
     "C75",
     "x_vars[26,1]"
    ],
    [
     "C76",
     "x_vars[26,2]"
    ],
    [
     "C77",
     "x_vars[26,3]"
    ],
    [
     "C78",
     "x_vars[27,1]"
    ],
    [
     "C79",
     "x_vars[27,2]"
    ],
    [
     "C80",
     "x_vars[27,3]"
    ],
    [
     "C81",
     "x_vars[28,1]"
    ],
    [
     "C82",
     "x_vars[28,2]"
    ],
    [
     "C83",
     "x_vars[28,3]"
    ],
    [
     "C84",
     "x_vars[29,1]"
    ],
    [
     "C85",
     "x_vars[29,2]"
    ],
    [
     "C86",
     "x_vars[29,3]"
    ],
    [
     "C87",
     "x_vars[30,1]"
    ],
    [
     "C88",
     "x_vars[30,2]"
    ],
    [
     "C89",
     "x_vars[30,3]"
    ],
    [
     "C90",
     "x_vars[31,1]"
    ],
    [
     "C91",
     "x_vars[31,2]"
    ],
    [
     "C92",
     "x_vars[31,3]"
    ],
    [
     "C93",
     "x_vars[32,1]"
    ],
    [
     "C94",
     "x_vars[32,2]"
    ],
    [
     "C95",
     "x_vars[32,3]"
    ],
    [
     "C96",
     "x_vars[33,1]"
    ],
    [
     "C97",
     "x_vars[33,2]"
    ],
    [
     "C98",
     "x_vars[33,3]"
    ],
    [
     "C99",
     "x_vars[34,1]"
    ],
    [
     "C100",
     "x_vars[34,2]"
    ],
    [
     "C101",
     "x_vars[34,3]"
    ],
    [
     "C102",
     "x_vars[35,1]"
    ],
    [
     "C103",
     "x_vars[35,2]"
    ],
    [
     "C104",
     "x_vars[35,3]"
    ],
    [
     "C105",
     "x_vars[36,1]"
    ],
    [
     "C106",
     "x_vars[36,2]"
    ],
    [
     "C107",
     "x_vars[36,3]"
    ],
    [
     "C108",
     "x_vars[37,1]"
    ],
    [
     "C109",
     "x_vars[37,2]"
    ],
    [
     "C110",
     "x_vars[37,3]"
    ],
    [
     "C111",
     "x_vars[38,1]"
    ],
    [
     "C112",
     "x_vars[38,2]"
    ],
    [
     "C113",
     "x_vars[38,3]"
    ],
    [
     "C114",
     "x_vars[39,1]"
    ],
    [
     "C115",
     "x_vars[39,2]"
    ],
    [
     "C116",
     "x_vars[39,3]"
    ],
    [
     "C117",
     "x_vars[40,1]"
    ],
    [
     "C118",
     "x_vars[40,2]"
    ],
    [
     "C119",
     "x_vars[40,3]"
    ],
    [
     "Cmax",
     "Cmax"
    ]
   ],
   "primary_mapping": {
    "x[1,1]": "x_vars[1,1]",
    "x[1,2]": "x_vars[2,1]",
    "x[1,3]": "x_vars[3,1]",
    "x[1,4]": "x_vars[4,1]",
    "x[1,5]": "x_vars[5,1]",
    "x[1,6]": "x_vars[6,1]",
    "x[1,7]": "x_vars[7,1]",
    "x[1,8]": "x_vars[8,1]",
    "x[1,9]": "x_vars[9,1]",
    "x[1,10]": "x_vars[10,1]",
    "x[1,11]": "x_vars[11,1]",
    "x[1,12]": "x_vars[12,1]",
    "x[1,13]": "x_vars[13,1]",
    "x[1,14]": "x_vars[14,1]",
    "x[1,15]": "x_vars[15,1]",
    "x[1,16]": "x_vars[16,1]",
    "x[1,17]": "x_vars[17,1]",
    "x[1,18]": "x_vars[18,1]",
    "x[1,19]": "x_vars[19,1]",
    "x[1,20]": "x_vars[20,1]",
    "x[1,21]": "x_vars[21,1]",
    "x[1,22]": "x_vars[22,1]",
    "x[1,23]": "x_vars[23,1]",
    "x[1,24]": "x_vars[24,1]",
    "x[1,25]": "x_vars[25,1]",
    "x[1,26]": "x_vars[26,1]",
    "x[1,27]": "x_vars[27,1]",
    "x[1,28]": "x_vars[28,1]",
    "x[1,29]": "x_vars[29,1]",
    "x[1,30]": "x_vars[30,1]",
    "x[1,31]": "x_vars[31,1]",
    "x[1,32]": "x_vars[32,1]",
    "x[1,33]": "x_vars[33,1]",
    "x[1,34]": "x_vars[34,1]",
    "x[1,35]": "x_vars[35,1]",
    "x[1,36]": "x_vars[36,1]",
    "x[1,37]": "x_vars[37,1]",
    "x[1,38]": "x_vars[38,1]",
    "x[1,39]": "x_vars[39,1]",
    "x[1,40]": "x_vars[40,1]",
    "x[2,1]": "x_vars[1,2]",
    "x[2,2]": "x_vars[2,2]",
    "x[2,3]": "x_vars[3,2]",
    "x[2,4]": "x_vars[4,2]",
    "x[2,5]": "x_vars[5,2]",
    "x[2,6]": "x_vars[6,2]",
    "x[2,7]": "x_vars[7,2]",
    "x[2,8]": "x_vars[8,2]",
    "x[2,9]": "x_vars[9,2]",
    "x[2,10]": "x_vars[10,2]",
    "x[2,11]": "x_vars[11,2]",
    "x[2,12]": "x_vars[12,2]",
    "x[2,13]": "x_vars[13,2]",
    "x[2,14]": "x_vars[14,2]",
    "x[2,15]": "x_vars[15,2]",
    "x[2,16]": "x_vars[16,2]",
    "x[2,17]": "x_vars[17,2]",
    "x[2,18]": "x_vars[18,2]",
    "x[2,19]": "x_vars[19,2]",
    "x[2,20]": "x_vars[20,2]",
    "x[2,21]": "x_vars[21,2]",
    "x[2,22]": "x_vars[22,2]",
    "x[2,23]": "x_vars[23,2]",
    "x[2,24]": "x_vars[24,2]",
    "x[2,25]": "x_vars[25,2]",
    "x[2,26]": "x_vars[26,2]",
    "x[2,27]": "x_vars[27,2]",
    "x[2,28]": "x_vars[28,2]",
    "x[2,29]": "x_vars[29,2]",
    "x[2,30]": "x_vars[30,2]",
    "x[2,31]": "x_vars[31,2]",
    "x[2,32]": "x_vars[32,2]",
    "x[2,33]": "x_vars[33,2]",
    "x[2,34]": "x_vars[34,2]",
    "x[2,35]": "x_vars[35,2]",
    "x[2,36]": "x_vars[36,2]",
    "x[2,37]": "x_vars[37,2]",
    "x[2,38]": "x_vars[38,2]",
    "x[2,39]": "x_vars[39,2]",
    "x[2,40]": "x_vars[40,2]",
    "x[3,1]": "x_vars[1,3]",
    "x[3,2]": "x_vars[2,3]",
    "x[3,3]": "x_vars[3,3]",
    "x[3,4]": "x_vars[4,3]",
    "x[3,5]": "x_vars[5,3]",
    "x[3,6]": "x_vars[6,3]",
    "x[3,7]": "x_vars[7,3]",
    "x[3,8]": "x_vars[8,3]",
    "x[3,9]": "x_vars[9,3]",
    "x[3,10]": "x_vars[10,3]",
    "x[3,11]": "x_vars[11,3]",
    "x[3,12]": "x_vars[12,3]",
    "x[3,13]": "x_vars[13,3]",
    "x[3,14]": "x_vars[14,3]",
    "x[3,15]": "x_vars[15,3]",
    "x[3,16]": "x_vars[16,3]",
    "x[3,17]": "x_vars[17,3]",
    "x[3,18]": "x_vars[18,3]",
    "x[3,19]": "x_vars[19,3]",
    "x[3,20]": "x_vars[20,3]",
    "x[3,21]": "x_vars[21,3]",
    "x[3,22]": "x_vars[22,3]",
    "x[3,23]": "x_vars[23,3]",
    "x[3,24]": "x_vars[24,3]",
    "x[3,25]": "x_vars[25,3]",
    "x[3,26]": "x_vars[26,3]",
    "x[3,27]": "x_vars[27,3]",
    "x[3,28]": "x_vars[28,3]",
    "x[3,29]": "x_vars[29,3]",
    "x[3,30]": "x_vars[30,3]",
    "x[3,31]": "x_vars[31,3]",
    "x[3,32]": "x_vars[32,3]",
    "x[3,33]": "x_vars[33,3]",
    "x[3,34]": "x_vars[34,3]",
    "x[3,35]": "x_vars[35,3]",
    "x[3,36]": "x_vars[36,3]",
    "x[3,37]": "x_vars[37,3]",
    "x[3,38]": "x_vars[38,3]",
    "x[3,39]": "x_vars[39,3]",
    "x[3,40]": "x_vars[40,3]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-085": {
   "source_code_sha256": "f0cf088924ae1ddec71df22d55c22f49f70894b12ca27e50944c89e951215e63",
   "original_solution_sha256": "bc5a1d6c73e928c52c48021919d8502c2d6b4df133eba0d23cfbd3835a92e9c0",
   "bindings": [
    [
     "C0",
     "x_vars[1,2]"
    ],
    [
     "C1",
     "x_vars[1,3]"
    ],
    [
     "C2",
     "x_vars[1,4]"
    ],
    [
     "C3",
     "x_vars[1,5]"
    ],
    [
     "C4",
     "x_vars[1,6]"
    ],
    [
     "C5",
     "x_vars[1,7]"
    ],
    [
     "C6",
     "x_vars[1,8]"
    ],
    [
     "C7",
     "x_vars[1,9]"
    ],
    [
     "C8",
     "x_vars[1,10]"
    ],
    [
     "C9",
     "x_vars[1,11]"
    ],
    [
     "C10",
     "x_vars[1,12]"
    ],
    [
     "C11",
     "x_vars[1,13]"
    ],
    [
     "C12",
     "x_vars[1,14]"
    ],
    [
     "C13",
     "x_vars[1,15]"
    ],
    [
     "C14",
     "x_vars[2,1]"
    ],
    [
     "C15",
     "x_vars[2,3]"
    ],
    [
     "C16",
     "x_vars[2,4]"
    ],
    [
     "C17",
     "x_vars[2,5]"
    ],
    [
     "C18",
     "x_vars[2,6]"
    ],
    [
     "C19",
     "x_vars[2,7]"
    ],
    [
     "C20",
     "x_vars[2,8]"
    ],
    [
     "C21",
     "x_vars[2,9]"
    ],
    [
     "C22",
     "x_vars[2,10]"
    ],
    [
     "C23",
     "x_vars[2,11]"
    ],
    [
     "C24",
     "x_vars[2,12]"
    ],
    [
     "C25",
     "x_vars[2,13]"
    ],
    [
     "C26",
     "x_vars[2,14]"
    ],
    [
     "C27",
     "x_vars[2,15]"
    ],
    [
     "C28",
     "x_vars[3,1]"
    ],
    [
     "C29",
     "x_vars[3,2]"
    ],
    [
     "C30",
     "x_vars[3,4]"
    ],
    [
     "C31",
     "x_vars[3,5]"
    ],
    [
     "C32",
     "x_vars[3,6]"
    ],
    [
     "C33",
     "x_vars[3,7]"
    ],
    [
     "C34",
     "x_vars[3,8]"
    ],
    [
     "C35",
     "x_vars[3,9]"
    ],
    [
     "C36",
     "x_vars[3,10]"
    ],
    [
     "C37",
     "x_vars[3,11]"
    ],
    [
     "C38",
     "x_vars[3,12]"
    ],
    [
     "C39",
     "x_vars[3,13]"
    ],
    [
     "C40",
     "x_vars[3,14]"
    ],
    [
     "C41",
     "x_vars[3,15]"
    ],
    [
     "C42",
     "x_vars[4,1]"
    ],
    [
     "C43",
     "x_vars[4,2]"
    ],
    [
     "C44",
     "x_vars[4,3]"
    ],
    [
     "C45",
     "x_vars[4,5]"
    ],
    [
     "C46",
     "x_vars[4,6]"
    ],
    [
     "C47",
     "x_vars[4,7]"
    ],
    [
     "C48",
     "x_vars[4,8]"
    ],
    [
     "C49",
     "x_vars[4,9]"
    ],
    [
     "C50",
     "x_vars[4,10]"
    ],
    [
     "C51",
     "x_vars[4,11]"
    ],
    [
     "C52",
     "x_vars[4,12]"
    ],
    [
     "C53",
     "x_vars[4,13]"
    ],
    [
     "C54",
     "x_vars[4,14]"
    ],
    [
     "C55",
     "x_vars[4,15]"
    ],
    [
     "C56",
     "x_vars[5,1]"
    ],
    [
     "C57",
     "x_vars[5,2]"
    ],
    [
     "C58",
     "x_vars[5,3]"
    ],
    [
     "C59",
     "x_vars[5,4]"
    ],
    [
     "C60",
     "x_vars[5,6]"
    ],
    [
     "C61",
     "x_vars[5,7]"
    ],
    [
     "C62",
     "x_vars[5,8]"
    ],
    [
     "C63",
     "x_vars[5,9]"
    ],
    [
     "C64",
     "x_vars[5,10]"
    ],
    [
     "C65",
     "x_vars[5,11]"
    ],
    [
     "C66",
     "x_vars[5,12]"
    ],
    [
     "C67",
     "x_vars[5,13]"
    ],
    [
     "C68",
     "x_vars[5,14]"
    ],
    [
     "C69",
     "x_vars[5,15]"
    ],
    [
     "C70",
     "x_vars[6,1]"
    ],
    [
     "C71",
     "x_vars[6,2]"
    ],
    [
     "C72",
     "x_vars[6,3]"
    ],
    [
     "C73",
     "x_vars[6,4]"
    ],
    [
     "C74",
     "x_vars[6,5]"
    ],
    [
     "C75",
     "x_vars[6,7]"
    ],
    [
     "C76",
     "x_vars[6,8]"
    ],
    [
     "C77",
     "x_vars[6,9]"
    ],
    [
     "C78",
     "x_vars[6,10]"
    ],
    [
     "C79",
     "x_vars[6,11]"
    ],
    [
     "C80",
     "x_vars[6,12]"
    ],
    [
     "C81",
     "x_vars[6,13]"
    ],
    [
     "C82",
     "x_vars[6,14]"
    ],
    [
     "C83",
     "x_vars[6,15]"
    ],
    [
     "C84",
     "x_vars[7,1]"
    ],
    [
     "C85",
     "x_vars[7,2]"
    ],
    [
     "C86",
     "x_vars[7,3]"
    ],
    [
     "C87",
     "x_vars[7,4]"
    ],
    [
     "C88",
     "x_vars[7,5]"
    ],
    [
     "C89",
     "x_vars[7,6]"
    ],
    [
     "C90",
     "x_vars[7,8]"
    ],
    [
     "C91",
     "x_vars[7,9]"
    ],
    [
     "C92",
     "x_vars[7,10]"
    ],
    [
     "C93",
     "x_vars[7,11]"
    ],
    [
     "C94",
     "x_vars[7,12]"
    ],
    [
     "C95",
     "x_vars[7,13]"
    ],
    [
     "C96",
     "x_vars[7,14]"
    ],
    [
     "C97",
     "x_vars[7,15]"
    ],
    [
     "C98",
     "x_vars[8,1]"
    ],
    [
     "C99",
     "x_vars[8,2]"
    ],
    [
     "C100",
     "x_vars[8,3]"
    ],
    [
     "C101",
     "x_vars[8,4]"
    ],
    [
     "C102",
     "x_vars[8,5]"
    ],
    [
     "C103",
     "x_vars[8,6]"
    ],
    [
     "C104",
     "x_vars[8,7]"
    ],
    [
     "C105",
     "x_vars[8,9]"
    ],
    [
     "C106",
     "x_vars[8,10]"
    ],
    [
     "C107",
     "x_vars[8,11]"
    ],
    [
     "C108",
     "x_vars[8,12]"
    ],
    [
     "C109",
     "x_vars[8,13]"
    ],
    [
     "C110",
     "x_vars[8,14]"
    ],
    [
     "C111",
     "x_vars[8,15]"
    ],
    [
     "C112",
     "x_vars[9,1]"
    ],
    [
     "C113",
     "x_vars[9,2]"
    ],
    [
     "C114",
     "x_vars[9,3]"
    ],
    [
     "C115",
     "x_vars[9,4]"
    ],
    [
     "C116",
     "x_vars[9,5]"
    ],
    [
     "C117",
     "x_vars[9,6]"
    ],
    [
     "C118",
     "x_vars[9,7]"
    ],
    [
     "C119",
     "x_vars[9,8]"
    ],
    [
     "C120",
     "x_vars[9,10]"
    ],
    [
     "C121",
     "x_vars[9,11]"
    ],
    [
     "C122",
     "x_vars[9,12]"
    ],
    [
     "C123",
     "x_vars[9,13]"
    ],
    [
     "C124",
     "x_vars[9,14]"
    ],
    [
     "C125",
     "x_vars[9,15]"
    ],
    [
     "C126",
     "x_vars[10,1]"
    ],
    [
     "C127",
     "x_vars[10,2]"
    ],
    [
     "C128",
     "x_vars[10,3]"
    ],
    [
     "C129",
     "x_vars[10,4]"
    ],
    [
     "C130",
     "x_vars[10,5]"
    ],
    [
     "C131",
     "x_vars[10,6]"
    ],
    [
     "C132",
     "x_vars[10,7]"
    ],
    [
     "C133",
     "x_vars[10,8]"
    ],
    [
     "C134",
     "x_vars[10,9]"
    ],
    [
     "C135",
     "x_vars[10,11]"
    ],
    [
     "C136",
     "x_vars[10,12]"
    ],
    [
     "C137",
     "x_vars[10,13]"
    ],
    [
     "C138",
     "x_vars[10,14]"
    ],
    [
     "C139",
     "x_vars[10,15]"
    ],
    [
     "C140",
     "x_vars[11,1]"
    ],
    [
     "C141",
     "x_vars[11,2]"
    ],
    [
     "C142",
     "x_vars[11,3]"
    ],
    [
     "C143",
     "x_vars[11,4]"
    ],
    [
     "C144",
     "x_vars[11,5]"
    ],
    [
     "C145",
     "x_vars[11,6]"
    ],
    [
     "C146",
     "x_vars[11,7]"
    ],
    [
     "C147",
     "x_vars[11,8]"
    ],
    [
     "C148",
     "x_vars[11,9]"
    ],
    [
     "C149",
     "x_vars[11,10]"
    ],
    [
     "C150",
     "x_vars[11,12]"
    ],
    [
     "C151",
     "x_vars[11,13]"
    ],
    [
     "C152",
     "x_vars[11,14]"
    ],
    [
     "C153",
     "x_vars[11,15]"
    ],
    [
     "C154",
     "x_vars[12,1]"
    ],
    [
     "C155",
     "x_vars[12,2]"
    ],
    [
     "C156",
     "x_vars[12,3]"
    ],
    [
     "C157",
     "x_vars[12,4]"
    ],
    [
     "C158",
     "x_vars[12,5]"
    ],
    [
     "C159",
     "x_vars[12,6]"
    ],
    [
     "C160",
     "x_vars[12,7]"
    ],
    [
     "C161",
     "x_vars[12,8]"
    ],
    [
     "C162",
     "x_vars[12,9]"
    ],
    [
     "C163",
     "x_vars[12,10]"
    ],
    [
     "C164",
     "x_vars[12,11]"
    ],
    [
     "C165",
     "x_vars[12,13]"
    ],
    [
     "C166",
     "x_vars[12,14]"
    ],
    [
     "C167",
     "x_vars[12,15]"
    ],
    [
     "C168",
     "x_vars[13,1]"
    ],
    [
     "C169",
     "x_vars[13,2]"
    ],
    [
     "C170",
     "x_vars[13,3]"
    ],
    [
     "C171",
     "x_vars[13,4]"
    ],
    [
     "C172",
     "x_vars[13,5]"
    ],
    [
     "C173",
     "x_vars[13,6]"
    ],
    [
     "C174",
     "x_vars[13,7]"
    ],
    [
     "C175",
     "x_vars[13,8]"
    ],
    [
     "C176",
     "x_vars[13,9]"
    ],
    [
     "C177",
     "x_vars[13,10]"
    ],
    [
     "C178",
     "x_vars[13,11]"
    ],
    [
     "C179",
     "x_vars[13,12]"
    ],
    [
     "C180",
     "x_vars[13,14]"
    ],
    [
     "C181",
     "x_vars[13,15]"
    ],
    [
     "C182",
     "x_vars[14,1]"
    ],
    [
     "C183",
     "x_vars[14,2]"
    ],
    [
     "C184",
     "x_vars[14,3]"
    ],
    [
     "C185",
     "x_vars[14,4]"
    ],
    [
     "C186",
     "x_vars[14,5]"
    ],
    [
     "C187",
     "x_vars[14,6]"
    ],
    [
     "C188",
     "x_vars[14,7]"
    ],
    [
     "C189",
     "x_vars[14,8]"
    ],
    [
     "C190",
     "x_vars[14,9]"
    ],
    [
     "C191",
     "x_vars[14,10]"
    ],
    [
     "C192",
     "x_vars[14,11]"
    ],
    [
     "C193",
     "x_vars[14,12]"
    ],
    [
     "C194",
     "x_vars[14,13]"
    ],
    [
     "C195",
     "x_vars[14,15]"
    ],
    [
     "C196",
     "x_vars[15,1]"
    ],
    [
     "C197",
     "x_vars[15,2]"
    ],
    [
     "C198",
     "x_vars[15,3]"
    ],
    [
     "C199",
     "x_vars[15,4]"
    ],
    [
     "C200",
     "x_vars[15,5]"
    ],
    [
     "C201",
     "x_vars[15,6]"
    ],
    [
     "C202",
     "x_vars[15,7]"
    ],
    [
     "C203",
     "x_vars[15,8]"
    ],
    [
     "C204",
     "x_vars[15,9]"
    ],
    [
     "C205",
     "x_vars[15,10]"
    ],
    [
     "C206",
     "x_vars[15,11]"
    ],
    [
     "C207",
     "x_vars[15,12]"
    ],
    [
     "C208",
     "x_vars[15,13]"
    ],
    [
     "C209",
     "x_vars[15,14]"
    ],
    [
     "C210",
     "u_vars[2]"
    ],
    [
     "C211",
     "u_vars[3]"
    ],
    [
     "C212",
     "u_vars[4]"
    ],
    [
     "C213",
     "u_vars[5]"
    ],
    [
     "C214",
     "u_vars[6]"
    ],
    [
     "C215",
     "u_vars[7]"
    ],
    [
     "C216",
     "u_vars[8]"
    ],
    [
     "C217",
     "u_vars[9]"
    ],
    [
     "C218",
     "u_vars[10]"
    ],
    [
     "C219",
     "u_vars[11]"
    ],
    [
     "C220",
     "u_vars[12]"
    ],
    [
     "C221",
     "u_vars[13]"
    ],
    [
     "C222",
     "u_vars[14]"
    ],
    [
     "C223",
     "u_vars[15]"
    ]
   ],
   "primary_mapping": {
    "x[1,2]": "x_vars[1,2]",
    "x[1,3]": "x_vars[1,3]",
    "x[1,4]": "x_vars[1,4]",
    "x[1,5]": "x_vars[1,5]",
    "x[1,6]": "x_vars[1,6]",
    "x[1,7]": "x_vars[1,7]",
    "x[1,8]": "x_vars[1,8]",
    "x[1,9]": "x_vars[1,9]",
    "x[1,10]": "x_vars[1,10]",
    "x[1,11]": "x_vars[1,11]",
    "x[1,12]": "x_vars[1,12]",
    "x[1,13]": "x_vars[1,13]",
    "x[1,14]": "x_vars[1,14]",
    "x[1,15]": "x_vars[1,15]",
    "x[2,1]": "x_vars[2,1]",
    "x[2,3]": "x_vars[2,3]",
    "x[2,4]": "x_vars[2,4]",
    "x[2,5]": "x_vars[2,5]",
    "x[2,6]": "x_vars[2,6]",
    "x[2,7]": "x_vars[2,7]",
    "x[2,8]": "x_vars[2,8]",
    "x[2,9]": "x_vars[2,9]",
    "x[2,10]": "x_vars[2,10]",
    "x[2,11]": "x_vars[2,11]",
    "x[2,12]": "x_vars[2,12]",
    "x[2,13]": "x_vars[2,13]",
    "x[2,14]": "x_vars[2,14]",
    "x[2,15]": "x_vars[2,15]",
    "x[3,1]": "x_vars[3,1]",
    "x[3,2]": "x_vars[3,2]",
    "x[3,4]": "x_vars[3,4]",
    "x[3,5]": "x_vars[3,5]",
    "x[3,6]": "x_vars[3,6]",
    "x[3,7]": "x_vars[3,7]",
    "x[3,8]": "x_vars[3,8]",
    "x[3,9]": "x_vars[3,9]",
    "x[3,10]": "x_vars[3,10]",
    "x[3,11]": "x_vars[3,11]",
    "x[3,12]": "x_vars[3,12]",
    "x[3,13]": "x_vars[3,13]",
    "x[3,14]": "x_vars[3,14]",
    "x[3,15]": "x_vars[3,15]",
    "x[4,1]": "x_vars[4,1]",
    "x[4,2]": "x_vars[4,2]",
    "x[4,3]": "x_vars[4,3]",
    "x[4,5]": "x_vars[4,5]",
    "x[4,6]": "x_vars[4,6]",
    "x[4,7]": "x_vars[4,7]",
    "x[4,8]": "x_vars[4,8]",
    "x[4,9]": "x_vars[4,9]",
    "x[4,10]": "x_vars[4,10]",
    "x[4,11]": "x_vars[4,11]",
    "x[4,12]": "x_vars[4,12]",
    "x[4,13]": "x_vars[4,13]",
    "x[4,14]": "x_vars[4,14]",
    "x[4,15]": "x_vars[4,15]",
    "x[5,1]": "x_vars[5,1]",
    "x[5,2]": "x_vars[5,2]",
    "x[5,3]": "x_vars[5,3]",
    "x[5,4]": "x_vars[5,4]",
    "x[5,6]": "x_vars[5,6]",
    "x[5,7]": "x_vars[5,7]",
    "x[5,8]": "x_vars[5,8]",
    "x[5,9]": "x_vars[5,9]",
    "x[5,10]": "x_vars[5,10]",
    "x[5,11]": "x_vars[5,11]",
    "x[5,12]": "x_vars[5,12]",
    "x[5,13]": "x_vars[5,13]",
    "x[5,14]": "x_vars[5,14]",
    "x[5,15]": "x_vars[5,15]",
    "x[6,1]": "x_vars[6,1]",
    "x[6,2]": "x_vars[6,2]",
    "x[6,3]": "x_vars[6,3]",
    "x[6,4]": "x_vars[6,4]",
    "x[6,5]": "x_vars[6,5]",
    "x[6,7]": "x_vars[6,7]",
    "x[6,8]": "x_vars[6,8]",
    "x[6,9]": "x_vars[6,9]",
    "x[6,10]": "x_vars[6,10]",
    "x[6,11]": "x_vars[6,11]",
    "x[6,12]": "x_vars[6,12]",
    "x[6,13]": "x_vars[6,13]",
    "x[6,14]": "x_vars[6,14]",
    "x[6,15]": "x_vars[6,15]",
    "x[7,1]": "x_vars[7,1]",
    "x[7,2]": "x_vars[7,2]",
    "x[7,3]": "x_vars[7,3]",
    "x[7,4]": "x_vars[7,4]",
    "x[7,5]": "x_vars[7,5]",
    "x[7,6]": "x_vars[7,6]",
    "x[7,8]": "x_vars[7,8]",
    "x[7,9]": "x_vars[7,9]",
    "x[7,10]": "x_vars[7,10]",
    "x[7,11]": "x_vars[7,11]",
    "x[7,12]": "x_vars[7,12]",
    "x[7,13]": "x_vars[7,13]",
    "x[7,14]": "x_vars[7,14]",
    "x[7,15]": "x_vars[7,15]",
    "x[8,1]": "x_vars[8,1]",
    "x[8,2]": "x_vars[8,2]",
    "x[8,3]": "x_vars[8,3]",
    "x[8,4]": "x_vars[8,4]",
    "x[8,5]": "x_vars[8,5]",
    "x[8,6]": "x_vars[8,6]",
    "x[8,7]": "x_vars[8,7]",
    "x[8,9]": "x_vars[8,9]",
    "x[8,10]": "x_vars[8,10]",
    "x[8,11]": "x_vars[8,11]",
    "x[8,12]": "x_vars[8,12]",
    "x[8,13]": "x_vars[8,13]",
    "x[8,14]": "x_vars[8,14]",
    "x[8,15]": "x_vars[8,15]",
    "x[9,1]": "x_vars[9,1]",
    "x[9,2]": "x_vars[9,2]",
    "x[9,3]": "x_vars[9,3]",
    "x[9,4]": "x_vars[9,4]",
    "x[9,5]": "x_vars[9,5]",
    "x[9,6]": "x_vars[9,6]",
    "x[9,7]": "x_vars[9,7]",
    "x[9,8]": "x_vars[9,8]",
    "x[9,10]": "x_vars[9,10]",
    "x[9,11]": "x_vars[9,11]",
    "x[9,12]": "x_vars[9,12]",
    "x[9,13]": "x_vars[9,13]",
    "x[9,14]": "x_vars[9,14]",
    "x[9,15]": "x_vars[9,15]",
    "x[10,1]": "x_vars[10,1]",
    "x[10,2]": "x_vars[10,2]",
    "x[10,3]": "x_vars[10,3]",
    "x[10,4]": "x_vars[10,4]",
    "x[10,5]": "x_vars[10,5]",
    "x[10,6]": "x_vars[10,6]",
    "x[10,7]": "x_vars[10,7]",
    "x[10,8]": "x_vars[10,8]",
    "x[10,9]": "x_vars[10,9]",
    "x[10,11]": "x_vars[10,11]",
    "x[10,12]": "x_vars[10,12]",
    "x[10,13]": "x_vars[10,13]",
    "x[10,14]": "x_vars[10,14]",
    "x[10,15]": "x_vars[10,15]",
    "x[11,1]": "x_vars[11,1]",
    "x[11,2]": "x_vars[11,2]",
    "x[11,3]": "x_vars[11,3]",
    "x[11,4]": "x_vars[11,4]",
    "x[11,5]": "x_vars[11,5]",
    "x[11,6]": "x_vars[11,6]",
    "x[11,7]": "x_vars[11,7]",
    "x[11,8]": "x_vars[11,8]",
    "x[11,9]": "x_vars[11,9]",
    "x[11,10]": "x_vars[11,10]",
    "x[11,12]": "x_vars[11,12]",
    "x[11,13]": "x_vars[11,13]",
    "x[11,14]": "x_vars[11,14]",
    "x[11,15]": "x_vars[11,15]",
    "x[12,1]": "x_vars[12,1]",
    "x[12,2]": "x_vars[12,2]",
    "x[12,3]": "x_vars[12,3]",
    "x[12,4]": "x_vars[12,4]",
    "x[12,5]": "x_vars[12,5]",
    "x[12,6]": "x_vars[12,6]",
    "x[12,7]": "x_vars[12,7]",
    "x[12,8]": "x_vars[12,8]",
    "x[12,9]": "x_vars[12,9]",
    "x[12,10]": "x_vars[12,10]",
    "x[12,11]": "x_vars[12,11]",
    "x[12,13]": "x_vars[12,13]",
    "x[12,14]": "x_vars[12,14]",
    "x[12,15]": "x_vars[12,15]",
    "x[13,1]": "x_vars[13,1]",
    "x[13,2]": "x_vars[13,2]",
    "x[13,3]": "x_vars[13,3]",
    "x[13,4]": "x_vars[13,4]",
    "x[13,5]": "x_vars[13,5]",
    "x[13,6]": "x_vars[13,6]",
    "x[13,7]": "x_vars[13,7]",
    "x[13,8]": "x_vars[13,8]",
    "x[13,9]": "x_vars[13,9]",
    "x[13,10]": "x_vars[13,10]",
    "x[13,11]": "x_vars[13,11]",
    "x[13,12]": "x_vars[13,12]",
    "x[13,14]": "x_vars[13,14]",
    "x[13,15]": "x_vars[13,15]",
    "x[14,1]": "x_vars[14,1]",
    "x[14,2]": "x_vars[14,2]",
    "x[14,3]": "x_vars[14,3]",
    "x[14,4]": "x_vars[14,4]",
    "x[14,5]": "x_vars[14,5]",
    "x[14,6]": "x_vars[14,6]",
    "x[14,7]": "x_vars[14,7]",
    "x[14,8]": "x_vars[14,8]",
    "x[14,9]": "x_vars[14,9]",
    "x[14,10]": "x_vars[14,10]",
    "x[14,11]": "x_vars[14,11]",
    "x[14,12]": "x_vars[14,12]",
    "x[14,13]": "x_vars[14,13]",
    "x[14,15]": "x_vars[14,15]",
    "x[15,1]": "x_vars[15,1]",
    "x[15,2]": "x_vars[15,2]",
    "x[15,3]": "x_vars[15,3]",
    "x[15,4]": "x_vars[15,4]",
    "x[15,5]": "x_vars[15,5]",
    "x[15,6]": "x_vars[15,6]",
    "x[15,7]": "x_vars[15,7]",
    "x[15,8]": "x_vars[15,8]",
    "x[15,9]": "x_vars[15,9]",
    "x[15,10]": "x_vars[15,10]",
    "x[15,11]": "x_vars[15,11]",
    "x[15,12]": "x_vars[15,12]",
    "x[15,13]": "x_vars[15,13]",
    "x[15,14]": "x_vars[15,14]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-086": {
   "source_code_sha256": "db8ca734d30b8722b91a95108eb94ade2d0d290631ae95290519a7ddf6267551",
   "original_solution_sha256": "2a3170f3f626351be700f77678074a7fe23d39db2f413b09ac015f602e742eac",
   "bindings": [
    [
     "C0",
     "x_vars[I,Red]"
    ],
    [
     "C1",
     "x_vars[I,Yellow]"
    ],
    [
     "C2",
     "x_vars[I,Blue]"
    ],
    [
     "C3",
     "x_vars[II,Red]"
    ],
    [
     "C4",
     "x_vars[II,Yellow]"
    ],
    [
     "C5",
     "x_vars[II,Blue]"
    ],
    [
     "C6",
     "x_vars[III,Red]"
    ],
    [
     "C7",
     "x_vars[III,Yellow]"
    ],
    [
     "C8",
     "x_vars[III,Blue]"
    ]
   ],
   "primary_mapping": {
    "x[Red,I]": "x_vars[I,Red]",
    "x[Red,II]": "x_vars[II,Red]",
    "x[Red,III]": "x_vars[III,Red]",
    "x[Yellow,I]": "x_vars[I,Yellow]",
    "x[Yellow,II]": "x_vars[II,Yellow]",
    "x[Yellow,III]": "x_vars[III,Yellow]",
    "x[Blue,I]": "x_vars[I,Blue]",
    "x[Blue,II]": "x_vars[II,Blue]",
    "x[Blue,III]": "x_vars[III,Blue]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-087": {
   "source_code_sha256": "6b74106a1b78eb44cc121e6b32366cc756deb7a2e6a41ef3190bc63a7ef6d353",
   "original_solution_sha256": "76390fa4d0955f719484d36c10b60e3e7d6b05f3b0ceea21b2c6110921481ced",
   "bindings": [
    [
     "C0",
     "x_vars[A1]"
    ],
    [
     "C1",
     "x_vars[A2]"
    ],
    [
     "C2",
     "x_vars[A3]"
    ],
    [
     "C3",
     "x_vars[A4]"
    ],
    [
     "C4",
     "x_vars[A5]"
    ],
    [
     "C5",
     "x_vars[A6]"
    ],
    [
     "C6",
     "x_vars[A7]"
    ],
    [
     "C7",
     "x_vars[A8]"
    ],
    [
     "C8",
     "x_vars[A9]"
    ],
    [
     "C9",
     "x_vars[A10]"
    ],
    [
     "C10",
     "x_vars[A11]"
    ],
    [
     "C11",
     "x_vars[A12]"
    ],
    [
     "C12",
     "x_vars[A13]"
    ],
    [
     "C13",
     "x_vars[A14]"
    ],
    [
     "C14",
     "x_vars[A15]"
    ],
    [
     "C15",
     "x_vars[A16]"
    ],
    [
     "C16",
     "x_vars[A17]"
    ],
    [
     "C17",
     "x_vars[A18]"
    ],
    [
     "C18",
     "x_vars[A19]"
    ],
    [
     "C19",
     "x_vars[A20]"
    ],
    [
     "C20",
     "x_vars[A21]"
    ],
    [
     "C21",
     "x_vars[A22]"
    ],
    [
     "C22",
     "x_vars[A23]"
    ],
    [
     "C23",
     "x_vars[A24]"
    ],
    [
     "C24",
     "x_vars[A25]"
    ],
    [
     "C25",
     "x_vars[A26]"
    ],
    [
     "C26",
     "x_vars[A27]"
    ],
    [
     "C27",
     "x_vars[A28]"
    ],
    [
     "C28",
     "x_vars[A29]"
    ],
    [
     "C29",
     "x_vars[A30]"
    ],
    [
     "C30",
     "x_vars[A31]"
    ],
    [
     "C31",
     "x_vars[A32]"
    ],
    [
     "C32",
     "x_vars[A33]"
    ],
    [
     "C33",
     "x_vars[A34]"
    ],
    [
     "C34",
     "x_vars[A35]"
    ],
    [
     "C35",
     "x_vars[A36]"
    ],
    [
     "C36",
     "x_vars[A37]"
    ],
    [
     "C37",
     "x_vars[A38]"
    ],
    [
     "C38",
     "x_vars[A39]"
    ],
    [
     "C39",
     "x_vars[A40]"
    ],
    [
     "C40",
     "x_vars[A41]"
    ],
    [
     "C41",
     "x_vars[A42]"
    ],
    [
     "C42",
     "x_vars[A43]"
    ],
    [
     "C43",
     "x_vars[A44]"
    ],
    [
     "C44",
     "x_vars[A45]"
    ],
    [
     "C45",
     "x_vars[A46]"
    ],
    [
     "C46",
     "x_vars[A47]"
    ],
    [
     "C47",
     "x_vars[A48]"
    ],
    [
     "C48",
     "x_vars[A49]"
    ],
    [
     "C49",
     "x_vars[A50]"
    ],
    [
     "C50",
     "x_vars[A51]"
    ],
    [
     "C51",
     "x_vars[A52]"
    ],
    [
     "C52",
     "x_vars[A53]"
    ],
    [
     "C53",
     "x_vars[A54]"
    ],
    [
     "C54",
     "x_vars[A55]"
    ],
    [
     "C55",
     "x_vars[A56]"
    ],
    [
     "C56",
     "x_vars[A57]"
    ],
    [
     "C57",
     "x_vars[A58]"
    ],
    [
     "C58",
     "x_vars[A59]"
    ],
    [
     "C59",
     "x_vars[A60]"
    ],
    [
     "C60",
     "x_vars[A61]"
    ],
    [
     "C61",
     "x_vars[A62]"
    ],
    [
     "C62",
     "x_vars[A63]"
    ],
    [
     "C63",
     "x_vars[A64]"
    ],
    [
     "C64",
     "x_vars[A65]"
    ],
    [
     "C65",
     "x_vars[A66]"
    ],
    [
     "C66",
     "x_vars[A67]"
    ],
    [
     "C67",
     "x_vars[A68]"
    ],
    [
     "C68",
     "x_vars[A69]"
    ],
    [
     "C69",
     "x_vars[A70]"
    ],
    [
     "C70",
     "x_vars[A71]"
    ],
    [
     "C71",
     "x_vars[A72]"
    ],
    [
     "C72",
     "x_vars[A73]"
    ],
    [
     "C73",
     "x_vars[A74]"
    ],
    [
     "C74",
     "x_vars[A75]"
    ],
    [
     "C75",
     "x_vars[A76]"
    ],
    [
     "C76",
     "x_vars[A77]"
    ],
    [
     "C77",
     "x_vars[A78]"
    ],
    [
     "C78",
     "x_vars[A79]"
    ],
    [
     "C79",
     "x_vars[A80]"
    ],
    [
     "C80",
     "y_vars[A1]"
    ],
    [
     "C81",
     "y_vars[A2]"
    ],
    [
     "C82",
     "y_vars[A3]"
    ],
    [
     "C83",
     "y_vars[A4]"
    ],
    [
     "C84",
     "y_vars[A5]"
    ],
    [
     "C85",
     "y_vars[A6]"
    ],
    [
     "C86",
     "y_vars[A7]"
    ],
    [
     "C87",
     "y_vars[A8]"
    ],
    [
     "C88",
     "y_vars[A9]"
    ],
    [
     "C89",
     "y_vars[A10]"
    ],
    [
     "C90",
     "y_vars[A11]"
    ],
    [
     "C91",
     "y_vars[A12]"
    ],
    [
     "C92",
     "y_vars[A13]"
    ],
    [
     "C93",
     "y_vars[A14]"
    ],
    [
     "C94",
     "y_vars[A15]"
    ],
    [
     "C95",
     "y_vars[A16]"
    ],
    [
     "C96",
     "y_vars[A17]"
    ],
    [
     "C97",
     "y_vars[A18]"
    ],
    [
     "C98",
     "y_vars[A19]"
    ],
    [
     "C99",
     "y_vars[A20]"
    ],
    [
     "C100",
     "y_vars[A21]"
    ],
    [
     "C101",
     "y_vars[A22]"
    ],
    [
     "C102",
     "y_vars[A23]"
    ],
    [
     "C103",
     "y_vars[A24]"
    ],
    [
     "C104",
     "y_vars[A25]"
    ],
    [
     "C105",
     "y_vars[A26]"
    ],
    [
     "C106",
     "y_vars[A27]"
    ],
    [
     "C107",
     "y_vars[A28]"
    ],
    [
     "C108",
     "y_vars[A29]"
    ],
    [
     "C109",
     "y_vars[A30]"
    ],
    [
     "C110",
     "y_vars[A31]"
    ],
    [
     "C111",
     "y_vars[A32]"
    ],
    [
     "C112",
     "y_vars[A33]"
    ],
    [
     "C113",
     "y_vars[A34]"
    ],
    [
     "C114",
     "y_vars[A35]"
    ],
    [
     "C115",
     "y_vars[A36]"
    ],
    [
     "C116",
     "y_vars[A37]"
    ],
    [
     "C117",
     "y_vars[A38]"
    ],
    [
     "C118",
     "y_vars[A39]"
    ],
    [
     "C119",
     "y_vars[A40]"
    ],
    [
     "C120",
     "y_vars[A41]"
    ],
    [
     "C121",
     "y_vars[A42]"
    ],
    [
     "C122",
     "y_vars[A43]"
    ],
    [
     "C123",
     "y_vars[A44]"
    ],
    [
     "C124",
     "y_vars[A45]"
    ],
    [
     "C125",
     "y_vars[A46]"
    ],
    [
     "C126",
     "y_vars[A47]"
    ],
    [
     "C127",
     "y_vars[A48]"
    ],
    [
     "C128",
     "y_vars[A49]"
    ],
    [
     "C129",
     "y_vars[A50]"
    ],
    [
     "C130",
     "y_vars[A51]"
    ],
    [
     "C131",
     "y_vars[A52]"
    ],
    [
     "C132",
     "y_vars[A53]"
    ],
    [
     "C133",
     "y_vars[A54]"
    ],
    [
     "C134",
     "y_vars[A55]"
    ],
    [
     "C135",
     "y_vars[A56]"
    ],
    [
     "C136",
     "y_vars[A57]"
    ],
    [
     "C137",
     "y_vars[A58]"
    ],
    [
     "C138",
     "y_vars[A59]"
    ],
    [
     "C139",
     "y_vars[A60]"
    ],
    [
     "C140",
     "y_vars[A61]"
    ],
    [
     "C141",
     "y_vars[A62]"
    ],
    [
     "C142",
     "y_vars[A63]"
    ],
    [
     "C143",
     "y_vars[A64]"
    ],
    [
     "C144",
     "y_vars[A65]"
    ],
    [
     "C145",
     "y_vars[A66]"
    ],
    [
     "C146",
     "y_vars[A67]"
    ],
    [
     "C147",
     "y_vars[A68]"
    ],
    [
     "C148",
     "y_vars[A69]"
    ],
    [
     "C149",
     "y_vars[A70]"
    ],
    [
     "C150",
     "y_vars[A71]"
    ],
    [
     "C151",
     "y_vars[A72]"
    ],
    [
     "C152",
     "y_vars[A73]"
    ],
    [
     "C153",
     "y_vars[A74]"
    ],
    [
     "C154",
     "y_vars[A75]"
    ],
    [
     "C155",
     "y_vars[A76]"
    ],
    [
     "C156",
     "y_vars[A77]"
    ],
    [
     "C157",
     "y_vars[A78]"
    ],
    [
     "C158",
     "y_vars[A79]"
    ],
    [
     "C159",
     "y_vars[A80]"
    ]
   ],
   "primary_mapping": {
    "x[1]": "x_vars[A1]",
    "x[2]": "x_vars[A2]",
    "x[3]": "x_vars[A3]",
    "x[4]": "x_vars[A4]",
    "x[5]": "x_vars[A5]",
    "x[6]": "x_vars[A6]",
    "x[7]": "x_vars[A7]",
    "x[8]": "x_vars[A8]",
    "x[9]": "x_vars[A9]",
    "x[10]": "x_vars[A10]",
    "x[11]": "x_vars[A11]",
    "x[12]": "x_vars[A12]",
    "x[13]": "x_vars[A13]",
    "x[14]": "x_vars[A14]",
    "x[15]": "x_vars[A15]",
    "x[16]": "x_vars[A16]",
    "x[17]": "x_vars[A17]",
    "x[18]": "x_vars[A18]",
    "x[19]": "x_vars[A19]",
    "x[20]": "x_vars[A20]",
    "x[21]": "x_vars[A21]",
    "x[22]": "x_vars[A22]",
    "x[23]": "x_vars[A23]",
    "x[24]": "x_vars[A24]",
    "x[25]": "x_vars[A25]",
    "x[26]": "x_vars[A26]",
    "x[27]": "x_vars[A27]",
    "x[28]": "x_vars[A28]",
    "x[29]": "x_vars[A29]",
    "x[30]": "x_vars[A30]",
    "x[31]": "x_vars[A31]",
    "x[32]": "x_vars[A32]",
    "x[33]": "x_vars[A33]",
    "x[34]": "x_vars[A34]",
    "x[35]": "x_vars[A35]",
    "x[36]": "x_vars[A36]",
    "x[37]": "x_vars[A37]",
    "x[38]": "x_vars[A38]",
    "x[39]": "x_vars[A39]",
    "x[40]": "x_vars[A40]",
    "x[41]": "x_vars[A41]",
    "x[42]": "x_vars[A42]",
    "x[43]": "x_vars[A43]",
    "x[44]": "x_vars[A44]",
    "x[45]": "x_vars[A45]",
    "x[46]": "x_vars[A46]",
    "x[47]": "x_vars[A47]",
    "x[48]": "x_vars[A48]",
    "x[49]": "x_vars[A49]",
    "x[50]": "x_vars[A50]",
    "x[51]": "x_vars[A51]",
    "x[52]": "x_vars[A52]",
    "x[53]": "x_vars[A53]",
    "x[54]": "x_vars[A54]",
    "x[55]": "x_vars[A55]",
    "x[56]": "x_vars[A56]",
    "x[57]": "x_vars[A57]",
    "x[58]": "x_vars[A58]",
    "x[59]": "x_vars[A59]",
    "x[60]": "x_vars[A60]",
    "x[61]": "x_vars[A61]",
    "x[62]": "x_vars[A62]",
    "x[63]": "x_vars[A63]",
    "x[64]": "x_vars[A64]",
    "x[65]": "x_vars[A65]",
    "x[66]": "x_vars[A66]",
    "x[67]": "x_vars[A67]",
    "x[68]": "x_vars[A68]",
    "x[69]": "x_vars[A69]",
    "x[70]": "x_vars[A70]",
    "x[71]": "x_vars[A71]",
    "x[72]": "x_vars[A72]",
    "x[73]": "x_vars[A73]",
    "x[74]": "x_vars[A74]",
    "x[75]": "x_vars[A75]",
    "x[76]": "x_vars[A76]",
    "x[77]": "x_vars[A77]",
    "x[78]": "x_vars[A78]",
    "x[79]": "x_vars[A79]",
    "x[80]": "x_vars[A80]",
    "y[1]": "y_vars[A1]",
    "y[2]": "y_vars[A2]",
    "y[3]": "y_vars[A3]",
    "y[4]": "y_vars[A4]",
    "y[5]": "y_vars[A5]",
    "y[6]": "y_vars[A6]",
    "y[7]": "y_vars[A7]",
    "y[8]": "y_vars[A8]",
    "y[9]": "y_vars[A9]",
    "y[10]": "y_vars[A10]",
    "y[11]": "y_vars[A11]",
    "y[12]": "y_vars[A12]",
    "y[13]": "y_vars[A13]",
    "y[14]": "y_vars[A14]",
    "y[15]": "y_vars[A15]",
    "y[16]": "y_vars[A16]",
    "y[17]": "y_vars[A17]",
    "y[18]": "y_vars[A18]",
    "y[19]": "y_vars[A19]",
    "y[20]": "y_vars[A20]",
    "y[21]": "y_vars[A21]",
    "y[22]": "y_vars[A22]",
    "y[23]": "y_vars[A23]",
    "y[24]": "y_vars[A24]",
    "y[25]": "y_vars[A25]",
    "y[26]": "y_vars[A26]",
    "y[27]": "y_vars[A27]",
    "y[28]": "y_vars[A28]",
    "y[29]": "y_vars[A29]",
    "y[30]": "y_vars[A30]",
    "y[31]": "y_vars[A31]",
    "y[32]": "y_vars[A32]",
    "y[33]": "y_vars[A33]",
    "y[34]": "y_vars[A34]",
    "y[35]": "y_vars[A35]",
    "y[36]": "y_vars[A36]",
    "y[37]": "y_vars[A37]",
    "y[38]": "y_vars[A38]",
    "y[39]": "y_vars[A39]",
    "y[40]": "y_vars[A40]",
    "y[41]": "y_vars[A41]",
    "y[42]": "y_vars[A42]",
    "y[43]": "y_vars[A43]",
    "y[44]": "y_vars[A44]",
    "y[45]": "y_vars[A45]",
    "y[46]": "y_vars[A46]",
    "y[47]": "y_vars[A47]",
    "y[48]": "y_vars[A48]",
    "y[49]": "y_vars[A49]",
    "y[50]": "y_vars[A50]",
    "y[51]": "y_vars[A51]",
    "y[52]": "y_vars[A52]",
    "y[53]": "y_vars[A53]",
    "y[54]": "y_vars[A54]",
    "y[55]": "y_vars[A55]",
    "y[56]": "y_vars[A56]",
    "y[57]": "y_vars[A57]",
    "y[58]": "y_vars[A58]",
    "y[59]": "y_vars[A59]",
    "y[60]": "y_vars[A60]",
    "y[61]": "y_vars[A61]",
    "y[62]": "y_vars[A62]",
    "y[63]": "y_vars[A63]",
    "y[64]": "y_vars[A64]",
    "y[65]": "y_vars[A65]",
    "y[66]": "y_vars[A66]",
    "y[67]": "y_vars[A67]",
    "y[68]": "y_vars[A68]",
    "y[69]": "y_vars[A69]",
    "y[70]": "y_vars[A70]",
    "y[71]": "y_vars[A71]",
    "y[72]": "y_vars[A72]",
    "y[73]": "y_vars[A73]",
    "y[74]": "y_vars[A74]",
    "y[75]": "y_vars[A75]",
    "y[76]": "y_vars[A76]",
    "y[77]": "y_vars[A77]",
    "y[78]": "y_vars[A78]",
    "y[79]": "y_vars[A79]",
    "y[80]": "y_vars[A80]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-088": {
   "source_code_sha256": "381e621df74efc40db58b0ae91533cc876347a6bf66126c55f80b1befb54d7bf",
   "original_solution_sha256": "adec244846701a6ce834f3cba1af320e227572c5de8141ebabdea6a0aae7a9a6",
   "bindings": [
    [
     "C0",
     "x_vars[F1,C1]"
    ],
    [
     "C1",
     "x_vars[F1,C2]"
    ],
    [
     "C2",
     "x_vars[F1,C3]"
    ],
    [
     "C3",
     "x_vars[F1,C4]"
    ],
    [
     "C4",
     "x_vars[F1,C5]"
    ],
    [
     "C5",
     "x_vars[F1,C6]"
    ],
    [
     "C6",
     "x_vars[F1,C7]"
    ],
    [
     "C7",
     "x_vars[F1,C8]"
    ],
    [
     "C8",
     "x_vars[F1,C9]"
    ],
    [
     "C9",
     "x_vars[F1,C10]"
    ],
    [
     "C10",
     "x_vars[F1,C11]"
    ],
    [
     "C11",
     "x_vars[F1,C12]"
    ],
    [
     "C12",
     "x_vars[F1,C13]"
    ],
    [
     "C13",
     "x_vars[F1,C14]"
    ],
    [
     "C14",
     "x_vars[F1,C15]"
    ],
    [
     "C15",
     "x_vars[F2,C1]"
    ],
    [
     "C16",
     "x_vars[F2,C2]"
    ],
    [
     "C17",
     "x_vars[F2,C3]"
    ],
    [
     "C18",
     "x_vars[F2,C4]"
    ],
    [
     "C19",
     "x_vars[F2,C5]"
    ],
    [
     "C20",
     "x_vars[F2,C6]"
    ],
    [
     "C21",
     "x_vars[F2,C7]"
    ],
    [
     "C22",
     "x_vars[F2,C8]"
    ],
    [
     "C23",
     "x_vars[F2,C9]"
    ],
    [
     "C24",
     "x_vars[F2,C10]"
    ],
    [
     "C25",
     "x_vars[F2,C11]"
    ],
    [
     "C26",
     "x_vars[F2,C12]"
    ],
    [
     "C27",
     "x_vars[F2,C13]"
    ],
    [
     "C28",
     "x_vars[F2,C14]"
    ],
    [
     "C29",
     "x_vars[F2,C15]"
    ],
    [
     "C30",
     "x_vars[F3,C1]"
    ],
    [
     "C31",
     "x_vars[F3,C2]"
    ],
    [
     "C32",
     "x_vars[F3,C3]"
    ],
    [
     "C33",
     "x_vars[F3,C4]"
    ],
    [
     "C34",
     "x_vars[F3,C5]"
    ],
    [
     "C35",
     "x_vars[F3,C6]"
    ],
    [
     "C36",
     "x_vars[F3,C7]"
    ],
    [
     "C37",
     "x_vars[F3,C8]"
    ],
    [
     "C38",
     "x_vars[F3,C9]"
    ],
    [
     "C39",
     "x_vars[F3,C10]"
    ],
    [
     "C40",
     "x_vars[F3,C11]"
    ],
    [
     "C41",
     "x_vars[F3,C12]"
    ],
    [
     "C42",
     "x_vars[F3,C13]"
    ],
    [
     "C43",
     "x_vars[F3,C14]"
    ],
    [
     "C44",
     "x_vars[F3,C15]"
    ],
    [
     "C45",
     "x_vars[F4,C1]"
    ],
    [
     "C46",
     "x_vars[F4,C2]"
    ],
    [
     "C47",
     "x_vars[F4,C3]"
    ],
    [
     "C48",
     "x_vars[F4,C4]"
    ],
    [
     "C49",
     "x_vars[F4,C5]"
    ],
    [
     "C50",
     "x_vars[F4,C6]"
    ],
    [
     "C51",
     "x_vars[F4,C7]"
    ],
    [
     "C52",
     "x_vars[F4,C8]"
    ],
    [
     "C53",
     "x_vars[F4,C9]"
    ],
    [
     "C54",
     "x_vars[F4,C10]"
    ],
    [
     "C55",
     "x_vars[F4,C11]"
    ],
    [
     "C56",
     "x_vars[F4,C12]"
    ],
    [
     "C57",
     "x_vars[F4,C13]"
    ],
    [
     "C58",
     "x_vars[F4,C14]"
    ],
    [
     "C59",
     "x_vars[F4,C15]"
    ],
    [
     "C60",
     "x_vars[F5,C1]"
    ],
    [
     "C61",
     "x_vars[F5,C2]"
    ],
    [
     "C62",
     "x_vars[F5,C3]"
    ],
    [
     "C63",
     "x_vars[F5,C4]"
    ],
    [
     "C64",
     "x_vars[F5,C5]"
    ],
    [
     "C65",
     "x_vars[F5,C6]"
    ],
    [
     "C66",
     "x_vars[F5,C7]"
    ],
    [
     "C67",
     "x_vars[F5,C8]"
    ],
    [
     "C68",
     "x_vars[F5,C9]"
    ],
    [
     "C69",
     "x_vars[F5,C10]"
    ],
    [
     "C70",
     "x_vars[F5,C11]"
    ],
    [
     "C71",
     "x_vars[F5,C12]"
    ],
    [
     "C72",
     "x_vars[F5,C13]"
    ],
    [
     "C73",
     "x_vars[F5,C14]"
    ],
    [
     "C74",
     "x_vars[F5,C15]"
    ],
    [
     "C75",
     "x_vars[F6,C1]"
    ],
    [
     "C76",
     "x_vars[F6,C2]"
    ],
    [
     "C77",
     "x_vars[F6,C3]"
    ],
    [
     "C78",
     "x_vars[F6,C4]"
    ],
    [
     "C79",
     "x_vars[F6,C5]"
    ],
    [
     "C80",
     "x_vars[F6,C6]"
    ],
    [
     "C81",
     "x_vars[F6,C7]"
    ],
    [
     "C82",
     "x_vars[F6,C8]"
    ],
    [
     "C83",
     "x_vars[F6,C9]"
    ],
    [
     "C84",
     "x_vars[F6,C10]"
    ],
    [
     "C85",
     "x_vars[F6,C11]"
    ],
    [
     "C86",
     "x_vars[F6,C12]"
    ],
    [
     "C87",
     "x_vars[F6,C13]"
    ],
    [
     "C88",
     "x_vars[F6,C14]"
    ],
    [
     "C89",
     "x_vars[F6,C15]"
    ],
    [
     "C90",
     "x_vars[F7,C1]"
    ],
    [
     "C91",
     "x_vars[F7,C2]"
    ],
    [
     "C92",
     "x_vars[F7,C3]"
    ],
    [
     "C93",
     "x_vars[F7,C4]"
    ],
    [
     "C94",
     "x_vars[F7,C5]"
    ],
    [
     "C95",
     "x_vars[F7,C6]"
    ],
    [
     "C96",
     "x_vars[F7,C7]"
    ],
    [
     "C97",
     "x_vars[F7,C8]"
    ],
    [
     "C98",
     "x_vars[F7,C9]"
    ],
    [
     "C99",
     "x_vars[F7,C10]"
    ],
    [
     "C100",
     "x_vars[F7,C11]"
    ],
    [
     "C101",
     "x_vars[F7,C12]"
    ],
    [
     "C102",
     "x_vars[F7,C13]"
    ],
    [
     "C103",
     "x_vars[F7,C14]"
    ],
    [
     "C104",
     "x_vars[F7,C15]"
    ],
    [
     "C105",
     "x_vars[F8,C1]"
    ],
    [
     "C106",
     "x_vars[F8,C2]"
    ],
    [
     "C107",
     "x_vars[F8,C3]"
    ],
    [
     "C108",
     "x_vars[F8,C4]"
    ],
    [
     "C109",
     "x_vars[F8,C5]"
    ],
    [
     "C110",
     "x_vars[F8,C6]"
    ],
    [
     "C111",
     "x_vars[F8,C7]"
    ],
    [
     "C112",
     "x_vars[F8,C8]"
    ],
    [
     "C113",
     "x_vars[F8,C9]"
    ],
    [
     "C114",
     "x_vars[F8,C10]"
    ],
    [
     "C115",
     "x_vars[F8,C11]"
    ],
    [
     "C116",
     "x_vars[F8,C12]"
    ],
    [
     "C117",
     "x_vars[F8,C13]"
    ],
    [
     "C118",
     "x_vars[F8,C14]"
    ],
    [
     "C119",
     "x_vars[F8,C15]"
    ],
    [
     "C120",
     "x_vars[F9,C1]"
    ],
    [
     "C121",
     "x_vars[F9,C2]"
    ],
    [
     "C122",
     "x_vars[F9,C3]"
    ],
    [
     "C123",
     "x_vars[F9,C4]"
    ],
    [
     "C124",
     "x_vars[F9,C5]"
    ],
    [
     "C125",
     "x_vars[F9,C6]"
    ],
    [
     "C126",
     "x_vars[F9,C7]"
    ],
    [
     "C127",
     "x_vars[F9,C8]"
    ],
    [
     "C128",
     "x_vars[F9,C9]"
    ],
    [
     "C129",
     "x_vars[F9,C10]"
    ],
    [
     "C130",
     "x_vars[F9,C11]"
    ],
    [
     "C131",
     "x_vars[F9,C12]"
    ],
    [
     "C132",
     "x_vars[F9,C13]"
    ],
    [
     "C133",
     "x_vars[F9,C14]"
    ],
    [
     "C134",
     "x_vars[F9,C15]"
    ],
    [
     "C135",
     "x_vars[F10,C1]"
    ],
    [
     "C136",
     "x_vars[F10,C2]"
    ],
    [
     "C137",
     "x_vars[F10,C3]"
    ],
    [
     "C138",
     "x_vars[F10,C4]"
    ],
    [
     "C139",
     "x_vars[F10,C5]"
    ],
    [
     "C140",
     "x_vars[F10,C6]"
    ],
    [
     "C141",
     "x_vars[F10,C7]"
    ],
    [
     "C142",
     "x_vars[F10,C8]"
    ],
    [
     "C143",
     "x_vars[F10,C9]"
    ],
    [
     "C144",
     "x_vars[F10,C10]"
    ],
    [
     "C145",
     "x_vars[F10,C11]"
    ],
    [
     "C146",
     "x_vars[F10,C12]"
    ],
    [
     "C147",
     "x_vars[F10,C13]"
    ],
    [
     "C148",
     "x_vars[F10,C14]"
    ],
    [
     "C149",
     "x_vars[F10,C15]"
    ],
    [
     "C150",
     "x_vars[F11,C1]"
    ],
    [
     "C151",
     "x_vars[F11,C2]"
    ],
    [
     "C152",
     "x_vars[F11,C3]"
    ],
    [
     "C153",
     "x_vars[F11,C4]"
    ],
    [
     "C154",
     "x_vars[F11,C5]"
    ],
    [
     "C155",
     "x_vars[F11,C6]"
    ],
    [
     "C156",
     "x_vars[F11,C7]"
    ],
    [
     "C157",
     "x_vars[F11,C8]"
    ],
    [
     "C158",
     "x_vars[F11,C9]"
    ],
    [
     "C159",
     "x_vars[F11,C10]"
    ],
    [
     "C160",
     "x_vars[F11,C11]"
    ],
    [
     "C161",
     "x_vars[F11,C12]"
    ],
    [
     "C162",
     "x_vars[F11,C13]"
    ],
    [
     "C163",
     "x_vars[F11,C14]"
    ],
    [
     "C164",
     "x_vars[F11,C15]"
    ],
    [
     "C165",
     "x_vars[F12,C1]"
    ],
    [
     "C166",
     "x_vars[F12,C2]"
    ],
    [
     "C167",
     "x_vars[F12,C3]"
    ],
    [
     "C168",
     "x_vars[F12,C4]"
    ],
    [
     "C169",
     "x_vars[F12,C5]"
    ],
    [
     "C170",
     "x_vars[F12,C6]"
    ],
    [
     "C171",
     "x_vars[F12,C7]"
    ],
    [
     "C172",
     "x_vars[F12,C8]"
    ],
    [
     "C173",
     "x_vars[F12,C9]"
    ],
    [
     "C174",
     "x_vars[F12,C10]"
    ],
    [
     "C175",
     "x_vars[F12,C11]"
    ],
    [
     "C176",
     "x_vars[F12,C12]"
    ],
    [
     "C177",
     "x_vars[F12,C13]"
    ],
    [
     "C178",
     "x_vars[F12,C14]"
    ],
    [
     "C179",
     "x_vars[F12,C15]"
    ],
    [
     "C180",
     "x_vars[F13,C1]"
    ],
    [
     "C181",
     "x_vars[F13,C2]"
    ],
    [
     "C182",
     "x_vars[F13,C3]"
    ],
    [
     "C183",
     "x_vars[F13,C4]"
    ],
    [
     "C184",
     "x_vars[F13,C5]"
    ],
    [
     "C185",
     "x_vars[F13,C6]"
    ],
    [
     "C186",
     "x_vars[F13,C7]"
    ],
    [
     "C187",
     "x_vars[F13,C8]"
    ],
    [
     "C188",
     "x_vars[F13,C9]"
    ],
    [
     "C189",
     "x_vars[F13,C10]"
    ],
    [
     "C190",
     "x_vars[F13,C11]"
    ],
    [
     "C191",
     "x_vars[F13,C12]"
    ],
    [
     "C192",
     "x_vars[F13,C13]"
    ],
    [
     "C193",
     "x_vars[F13,C14]"
    ],
    [
     "C194",
     "x_vars[F13,C15]"
    ],
    [
     "C195",
     "x_vars[F14,C1]"
    ],
    [
     "C196",
     "x_vars[F14,C2]"
    ],
    [
     "C197",
     "x_vars[F14,C3]"
    ],
    [
     "C198",
     "x_vars[F14,C4]"
    ],
    [
     "C199",
     "x_vars[F14,C5]"
    ],
    [
     "C200",
     "x_vars[F14,C6]"
    ],
    [
     "C201",
     "x_vars[F14,C7]"
    ],
    [
     "C202",
     "x_vars[F14,C8]"
    ],
    [
     "C203",
     "x_vars[F14,C9]"
    ],
    [
     "C204",
     "x_vars[F14,C10]"
    ],
    [
     "C205",
     "x_vars[F14,C11]"
    ],
    [
     "C206",
     "x_vars[F14,C12]"
    ],
    [
     "C207",
     "x_vars[F14,C13]"
    ],
    [
     "C208",
     "x_vars[F14,C14]"
    ],
    [
     "C209",
     "x_vars[F14,C15]"
    ],
    [
     "C210",
     "x_vars[F15,C1]"
    ],
    [
     "C211",
     "x_vars[F15,C2]"
    ],
    [
     "C212",
     "x_vars[F15,C3]"
    ],
    [
     "C213",
     "x_vars[F15,C4]"
    ],
    [
     "C214",
     "x_vars[F15,C5]"
    ],
    [
     "C215",
     "x_vars[F15,C6]"
    ],
    [
     "C216",
     "x_vars[F15,C7]"
    ],
    [
     "C217",
     "x_vars[F15,C8]"
    ],
    [
     "C218",
     "x_vars[F15,C9]"
    ],
    [
     "C219",
     "x_vars[F15,C10]"
    ],
    [
     "C220",
     "x_vars[F15,C11]"
    ],
    [
     "C221",
     "x_vars[F15,C12]"
    ],
    [
     "C222",
     "x_vars[F15,C13]"
    ],
    [
     "C223",
     "x_vars[F15,C14]"
    ],
    [
     "C224",
     "x_vars[F15,C15]"
    ],
    [
     "C225",
     "y_vars[F1]"
    ],
    [
     "C226",
     "y_vars[F2]"
    ],
    [
     "C227",
     "y_vars[F3]"
    ],
    [
     "C228",
     "y_vars[F4]"
    ],
    [
     "C229",
     "y_vars[F5]"
    ],
    [
     "C230",
     "y_vars[F6]"
    ],
    [
     "C231",
     "y_vars[F7]"
    ],
    [
     "C232",
     "y_vars[F8]"
    ],
    [
     "C233",
     "y_vars[F9]"
    ],
    [
     "C234",
     "y_vars[F10]"
    ],
    [
     "C235",
     "y_vars[F11]"
    ],
    [
     "C236",
     "y_vars[F12]"
    ],
    [
     "C237",
     "y_vars[F13]"
    ],
    [
     "C238",
     "y_vars[F14]"
    ],
    [
     "C239",
     "y_vars[F15]"
    ]
   ],
   "primary_mapping": {
    "y_F1": "y_vars[F1]",
    "y_F2": "y_vars[F2]",
    "y_F3": "y_vars[F3]",
    "y_F4": "y_vars[F4]",
    "y_F5": "y_vars[F5]",
    "y_F6": "y_vars[F6]",
    "y_F7": "y_vars[F7]",
    "y_F8": "y_vars[F8]",
    "y_F9": "y_vars[F9]",
    "y_F10": "y_vars[F10]",
    "y_F11": "y_vars[F11]",
    "y_F12": "y_vars[F12]",
    "y_F13": "y_vars[F13]",
    "y_F14": "y_vars[F14]",
    "y_F15": "y_vars[F15]",
    "x_F1_C1": "x_vars[F1,C1]",
    "x_F2_C2": "x_vars[F2,C2]",
    "x_F3_C3": "x_vars[F3,C3]",
    "x_F4_C4": "x_vars[F4,C4]",
    "x_F5_C5": "x_vars[F5,C5]",
    "x_F6_C6": "x_vars[F6,C6]",
    "x_F7_C7": "x_vars[F7,C7]",
    "x_F8_C8": "x_vars[F8,C8]",
    "x_F9_C9": "x_vars[F9,C9]",
    "x_F10_C10": "x_vars[F10,C10]",
    "x_F11_C11": "x_vars[F11,C11]",
    "x_F12_C12": "x_vars[F12,C12]",
    "x_F13_C13": "x_vars[F13,C13]",
    "x_F14_C14": "x_vars[F14,C14]",
    "x_F15_C15": "x_vars[F15,C15]",
    "x_F1_C2": "x_vars[F1,C2]",
    "x_F1_C3": "x_vars[F1,C3]",
    "x_F1_C4": "x_vars[F1,C4]",
    "x_F1_C5": "x_vars[F1,C5]",
    "x_F1_C6": "x_vars[F1,C6]",
    "x_F1_C7": "x_vars[F1,C7]",
    "x_F1_C8": "x_vars[F1,C8]",
    "x_F1_C9": "x_vars[F1,C9]",
    "x_F1_C10": "x_vars[F1,C10]",
    "x_F1_C11": "x_vars[F1,C11]",
    "x_F1_C12": "x_vars[F1,C12]",
    "x_F1_C13": "x_vars[F1,C13]",
    "x_F1_C14": "x_vars[F1,C14]",
    "x_F1_C15": "x_vars[F1,C15]",
    "x_F2_C1": "x_vars[F2,C1]",
    "x_F2_C3": "x_vars[F2,C3]",
    "x_F2_C4": "x_vars[F2,C4]",
    "x_F2_C5": "x_vars[F2,C5]",
    "x_F2_C6": "x_vars[F2,C6]",
    "x_F2_C7": "x_vars[F2,C7]",
    "x_F2_C8": "x_vars[F2,C8]",
    "x_F2_C9": "x_vars[F2,C9]",
    "x_F2_C10": "x_vars[F2,C10]",
    "x_F2_C11": "x_vars[F2,C11]",
    "x_F2_C12": "x_vars[F2,C12]",
    "x_F2_C13": "x_vars[F2,C13]",
    "x_F2_C14": "x_vars[F2,C14]",
    "x_F2_C15": "x_vars[F2,C15]",
    "x_F3_C1": "x_vars[F3,C1]",
    "x_F3_C2": "x_vars[F3,C2]",
    "x_F3_C4": "x_vars[F3,C4]",
    "x_F3_C5": "x_vars[F3,C5]",
    "x_F3_C6": "x_vars[F3,C6]",
    "x_F3_C7": "x_vars[F3,C7]",
    "x_F3_C8": "x_vars[F3,C8]",
    "x_F3_C9": "x_vars[F3,C9]",
    "x_F3_C10": "x_vars[F3,C10]",
    "x_F3_C11": "x_vars[F3,C11]",
    "x_F3_C12": "x_vars[F3,C12]",
    "x_F3_C13": "x_vars[F3,C13]",
    "x_F3_C14": "x_vars[F3,C14]",
    "x_F3_C15": "x_vars[F3,C15]",
    "x_F4_C1": "x_vars[F4,C1]",
    "x_F4_C2": "x_vars[F4,C2]",
    "x_F4_C3": "x_vars[F4,C3]",
    "x_F4_C5": "x_vars[F4,C5]",
    "x_F4_C6": "x_vars[F4,C6]",
    "x_F4_C7": "x_vars[F4,C7]",
    "x_F4_C8": "x_vars[F4,C8]",
    "x_F4_C9": "x_vars[F4,C9]",
    "x_F4_C10": "x_vars[F4,C10]",
    "x_F4_C11": "x_vars[F4,C11]",
    "x_F4_C12": "x_vars[F4,C12]",
    "x_F4_C13": "x_vars[F4,C13]",
    "x_F4_C14": "x_vars[F4,C14]",
    "x_F4_C15": "x_vars[F4,C15]",
    "x_F5_C1": "x_vars[F5,C1]",
    "x_F5_C2": "x_vars[F5,C2]",
    "x_F5_C3": "x_vars[F5,C3]",
    "x_F5_C4": "x_vars[F5,C4]",
    "x_F5_C6": "x_vars[F5,C6]",
    "x_F5_C7": "x_vars[F5,C7]",
    "x_F5_C8": "x_vars[F5,C8]",
    "x_F5_C9": "x_vars[F5,C9]",
    "x_F5_C10": "x_vars[F5,C10]",
    "x_F5_C11": "x_vars[F5,C11]",
    "x_F5_C12": "x_vars[F5,C12]",
    "x_F5_C13": "x_vars[F5,C13]",
    "x_F5_C14": "x_vars[F5,C14]",
    "x_F5_C15": "x_vars[F5,C15]",
    "x_F6_C1": "x_vars[F6,C1]",
    "x_F6_C2": "x_vars[F6,C2]",
    "x_F6_C3": "x_vars[F6,C3]",
    "x_F6_C4": "x_vars[F6,C4]",
    "x_F6_C5": "x_vars[F6,C5]",
    "x_F6_C7": "x_vars[F6,C7]",
    "x_F6_C8": "x_vars[F6,C8]",
    "x_F6_C9": "x_vars[F6,C9]",
    "x_F6_C10": "x_vars[F6,C10]",
    "x_F6_C11": "x_vars[F6,C11]",
    "x_F6_C12": "x_vars[F6,C12]",
    "x_F6_C13": "x_vars[F6,C13]",
    "x_F6_C14": "x_vars[F6,C14]",
    "x_F6_C15": "x_vars[F6,C15]",
    "x_F7_C1": "x_vars[F7,C1]",
    "x_F7_C2": "x_vars[F7,C2]",
    "x_F7_C3": "x_vars[F7,C3]",
    "x_F7_C4": "x_vars[F7,C4]",
    "x_F7_C5": "x_vars[F7,C5]",
    "x_F7_C6": "x_vars[F7,C6]",
    "x_F7_C8": "x_vars[F7,C8]",
    "x_F7_C9": "x_vars[F7,C9]",
    "x_F7_C10": "x_vars[F7,C10]",
    "x_F7_C11": "x_vars[F7,C11]",
    "x_F7_C12": "x_vars[F7,C12]",
    "x_F7_C13": "x_vars[F7,C13]",
    "x_F7_C14": "x_vars[F7,C14]",
    "x_F7_C15": "x_vars[F7,C15]",
    "x_F8_C1": "x_vars[F8,C1]",
    "x_F8_C2": "x_vars[F8,C2]",
    "x_F8_C3": "x_vars[F8,C3]",
    "x_F8_C4": "x_vars[F8,C4]",
    "x_F8_C5": "x_vars[F8,C5]",
    "x_F8_C6": "x_vars[F8,C6]",
    "x_F8_C7": "x_vars[F8,C7]",
    "x_F8_C9": "x_vars[F8,C9]",
    "x_F8_C10": "x_vars[F8,C10]",
    "x_F8_C11": "x_vars[F8,C11]",
    "x_F8_C12": "x_vars[F8,C12]",
    "x_F8_C13": "x_vars[F8,C13]",
    "x_F8_C14": "x_vars[F8,C14]",
    "x_F8_C15": "x_vars[F8,C15]",
    "x_F9_C1": "x_vars[F9,C1]",
    "x_F9_C2": "x_vars[F9,C2]",
    "x_F9_C3": "x_vars[F9,C3]",
    "x_F9_C4": "x_vars[F9,C4]",
    "x_F9_C5": "x_vars[F9,C5]",
    "x_F9_C6": "x_vars[F9,C6]",
    "x_F9_C7": "x_vars[F9,C7]",
    "x_F9_C8": "x_vars[F9,C8]",
    "x_F9_C10": "x_vars[F9,C10]",
    "x_F9_C11": "x_vars[F9,C11]",
    "x_F9_C12": "x_vars[F9,C12]",
    "x_F9_C13": "x_vars[F9,C13]",
    "x_F9_C14": "x_vars[F9,C14]",
    "x_F9_C15": "x_vars[F9,C15]",
    "x_F10_C1": "x_vars[F10,C1]",
    "x_F10_C2": "x_vars[F10,C2]",
    "x_F10_C3": "x_vars[F10,C3]",
    "x_F10_C4": "x_vars[F10,C4]",
    "x_F10_C5": "x_vars[F10,C5]",
    "x_F10_C6": "x_vars[F10,C6]",
    "x_F10_C7": "x_vars[F10,C7]",
    "x_F10_C8": "x_vars[F10,C8]",
    "x_F10_C9": "x_vars[F10,C9]",
    "x_F10_C11": "x_vars[F10,C11]",
    "x_F10_C12": "x_vars[F10,C12]",
    "x_F10_C13": "x_vars[F10,C13]",
    "x_F10_C14": "x_vars[F10,C14]",
    "x_F10_C15": "x_vars[F10,C15]",
    "x_F11_C1": "x_vars[F11,C1]",
    "x_F11_C2": "x_vars[F11,C2]",
    "x_F11_C3": "x_vars[F11,C3]",
    "x_F11_C4": "x_vars[F11,C4]",
    "x_F11_C5": "x_vars[F11,C5]",
    "x_F11_C6": "x_vars[F11,C6]",
    "x_F11_C7": "x_vars[F11,C7]",
    "x_F11_C8": "x_vars[F11,C8]",
    "x_F11_C9": "x_vars[F11,C9]",
    "x_F11_C10": "x_vars[F11,C10]",
    "x_F11_C12": "x_vars[F11,C12]",
    "x_F11_C13": "x_vars[F11,C13]",
    "x_F11_C14": "x_vars[F11,C14]",
    "x_F11_C15": "x_vars[F11,C15]",
    "x_F12_C1": "x_vars[F12,C1]",
    "x_F12_C2": "x_vars[F12,C2]",
    "x_F12_C3": "x_vars[F12,C3]",
    "x_F12_C4": "x_vars[F12,C4]",
    "x_F12_C5": "x_vars[F12,C5]",
    "x_F12_C6": "x_vars[F12,C6]",
    "x_F12_C7": "x_vars[F12,C7]",
    "x_F12_C8": "x_vars[F12,C8]",
    "x_F12_C9": "x_vars[F12,C9]",
    "x_F12_C10": "x_vars[F12,C10]",
    "x_F12_C11": "x_vars[F12,C11]",
    "x_F12_C13": "x_vars[F12,C13]",
    "x_F12_C14": "x_vars[F12,C14]",
    "x_F12_C15": "x_vars[F12,C15]",
    "x_F13_C1": "x_vars[F13,C1]",
    "x_F13_C2": "x_vars[F13,C2]",
    "x_F13_C3": "x_vars[F13,C3]",
    "x_F13_C4": "x_vars[F13,C4]",
    "x_F13_C5": "x_vars[F13,C5]",
    "x_F13_C6": "x_vars[F13,C6]",
    "x_F13_C7": "x_vars[F13,C7]",
    "x_F13_C8": "x_vars[F13,C8]",
    "x_F13_C9": "x_vars[F13,C9]",
    "x_F13_C10": "x_vars[F13,C10]",
    "x_F13_C11": "x_vars[F13,C11]",
    "x_F13_C12": "x_vars[F13,C12]",
    "x_F13_C14": "x_vars[F13,C14]",
    "x_F13_C15": "x_vars[F13,C15]",
    "x_F14_C1": "x_vars[F14,C1]",
    "x_F14_C2": "x_vars[F14,C2]",
    "x_F14_C3": "x_vars[F14,C3]",
    "x_F14_C4": "x_vars[F14,C4]",
    "x_F14_C5": "x_vars[F14,C5]",
    "x_F14_C6": "x_vars[F14,C6]",
    "x_F14_C7": "x_vars[F14,C7]",
    "x_F14_C8": "x_vars[F14,C8]",
    "x_F14_C9": "x_vars[F14,C9]",
    "x_F14_C10": "x_vars[F14,C10]",
    "x_F14_C11": "x_vars[F14,C11]",
    "x_F14_C12": "x_vars[F14,C12]",
    "x_F14_C13": "x_vars[F14,C13]",
    "x_F14_C15": "x_vars[F14,C15]",
    "x_F15_C1": "x_vars[F15,C1]",
    "x_F15_C2": "x_vars[F15,C2]",
    "x_F15_C3": "x_vars[F15,C3]",
    "x_F15_C4": "x_vars[F15,C4]",
    "x_F15_C5": "x_vars[F15,C5]",
    "x_F15_C6": "x_vars[F15,C6]",
    "x_F15_C7": "x_vars[F15,C7]",
    "x_F15_C8": "x_vars[F15,C8]",
    "x_F15_C9": "x_vars[F15,C9]",
    "x_F15_C10": "x_vars[F15,C10]",
    "x_F15_C11": "x_vars[F15,C11]",
    "x_F15_C12": "x_vars[F15,C12]",
    "x_F15_C13": "x_vars[F15,C13]",
    "x_F15_C14": "x_vars[F15,C14]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-089": {
   "source_code_sha256": "6f2feb030d55e551400398f075d8a10239b2f26905f5e058eb711eed5789d1e4",
   "original_solution_sha256": "6637ba4c2c3a1adc8e06eabb9de58b44e76971b645e1e606b6bfa6f3a7488717",
   "bindings": [
    [
     "C0",
     "y_vars[SC1]"
    ],
    [
     "C1",
     "y_vars[SC2]"
    ],
    [
     "C2",
     "y_vars[SC3]"
    ],
    [
     "C3",
     "y_vars[SC4]"
    ],
    [
     "C4",
     "y_vars[SC5]"
    ],
    [
     "C5",
     "y_vars[SC6]"
    ],
    [
     "C6",
     "y_vars[SC7]"
    ],
    [
     "C7",
     "y_vars[SC8]"
    ],
    [
     "C8",
     "y_vars[SC9]"
    ],
    [
     "C9",
     "y_vars[SC10]"
    ],
    [
     "C10",
     "x_vars[SC1,C1]"
    ],
    [
     "C11",
     "x_vars[SC1,C2]"
    ],
    [
     "C12",
     "x_vars[SC1,C3]"
    ],
    [
     "C13",
     "x_vars[SC1,C4]"
    ],
    [
     "C14",
     "x_vars[SC1,C5]"
    ],
    [
     "C15",
     "x_vars[SC1,C6]"
    ],
    [
     "C16",
     "x_vars[SC1,C7]"
    ],
    [
     "C17",
     "x_vars[SC1,C8]"
    ],
    [
     "C18",
     "x_vars[SC1,C9]"
    ],
    [
     "C19",
     "x_vars[SC1,C10]"
    ],
    [
     "C20",
     "x_vars[SC1,C11]"
    ],
    [
     "C21",
     "x_vars[SC1,C12]"
    ],
    [
     "C22",
     "x_vars[SC1,C13]"
    ],
    [
     "C23",
     "x_vars[SC1,C14]"
    ],
    [
     "C24",
     "x_vars[SC1,C15]"
    ],
    [
     "C25",
     "x_vars[SC2,C1]"
    ],
    [
     "C26",
     "x_vars[SC2,C2]"
    ],
    [
     "C27",
     "x_vars[SC2,C3]"
    ],
    [
     "C28",
     "x_vars[SC2,C4]"
    ],
    [
     "C29",
     "x_vars[SC2,C5]"
    ],
    [
     "C30",
     "x_vars[SC2,C6]"
    ],
    [
     "C31",
     "x_vars[SC2,C7]"
    ],
    [
     "C32",
     "x_vars[SC2,C8]"
    ],
    [
     "C33",
     "x_vars[SC2,C9]"
    ],
    [
     "C34",
     "x_vars[SC2,C10]"
    ],
    [
     "C35",
     "x_vars[SC2,C11]"
    ],
    [
     "C36",
     "x_vars[SC2,C12]"
    ],
    [
     "C37",
     "x_vars[SC2,C13]"
    ],
    [
     "C38",
     "x_vars[SC2,C14]"
    ],
    [
     "C39",
     "x_vars[SC2,C15]"
    ],
    [
     "C40",
     "x_vars[SC3,C1]"
    ],
    [
     "C41",
     "x_vars[SC3,C2]"
    ],
    [
     "C42",
     "x_vars[SC3,C3]"
    ],
    [
     "C43",
     "x_vars[SC3,C4]"
    ],
    [
     "C44",
     "x_vars[SC3,C5]"
    ],
    [
     "C45",
     "x_vars[SC3,C6]"
    ],
    [
     "C46",
     "x_vars[SC3,C7]"
    ],
    [
     "C47",
     "x_vars[SC3,C8]"
    ],
    [
     "C48",
     "x_vars[SC3,C9]"
    ],
    [
     "C49",
     "x_vars[SC3,C10]"
    ],
    [
     "C50",
     "x_vars[SC3,C11]"
    ],
    [
     "C51",
     "x_vars[SC3,C12]"
    ],
    [
     "C52",
     "x_vars[SC3,C13]"
    ],
    [
     "C53",
     "x_vars[SC3,C14]"
    ],
    [
     "C54",
     "x_vars[SC3,C15]"
    ],
    [
     "C55",
     "x_vars[SC4,C1]"
    ],
    [
     "C56",
     "x_vars[SC4,C2]"
    ],
    [
     "C57",
     "x_vars[SC4,C3]"
    ],
    [
     "C58",
     "x_vars[SC4,C4]"
    ],
    [
     "C59",
     "x_vars[SC4,C5]"
    ],
    [
     "C60",
     "x_vars[SC4,C6]"
    ],
    [
     "C61",
     "x_vars[SC4,C7]"
    ],
    [
     "C62",
     "x_vars[SC4,C8]"
    ],
    [
     "C63",
     "x_vars[SC4,C9]"
    ],
    [
     "C64",
     "x_vars[SC4,C10]"
    ],
    [
     "C65",
     "x_vars[SC4,C11]"
    ],
    [
     "C66",
     "x_vars[SC4,C12]"
    ],
    [
     "C67",
     "x_vars[SC4,C13]"
    ],
    [
     "C68",
     "x_vars[SC4,C14]"
    ],
    [
     "C69",
     "x_vars[SC4,C15]"
    ],
    [
     "C70",
     "x_vars[SC5,C1]"
    ],
    [
     "C71",
     "x_vars[SC5,C2]"
    ],
    [
     "C72",
     "x_vars[SC5,C3]"
    ],
    [
     "C73",
     "x_vars[SC5,C4]"
    ],
    [
     "C74",
     "x_vars[SC5,C5]"
    ],
    [
     "C75",
     "x_vars[SC5,C6]"
    ],
    [
     "C76",
     "x_vars[SC5,C7]"
    ],
    [
     "C77",
     "x_vars[SC5,C8]"
    ],
    [
     "C78",
     "x_vars[SC5,C9]"
    ],
    [
     "C79",
     "x_vars[SC5,C10]"
    ],
    [
     "C80",
     "x_vars[SC5,C11]"
    ],
    [
     "C81",
     "x_vars[SC5,C12]"
    ],
    [
     "C82",
     "x_vars[SC5,C13]"
    ],
    [
     "C83",
     "x_vars[SC5,C14]"
    ],
    [
     "C84",
     "x_vars[SC5,C15]"
    ],
    [
     "C85",
     "x_vars[SC6,C1]"
    ],
    [
     "C86",
     "x_vars[SC6,C2]"
    ],
    [
     "C87",
     "x_vars[SC6,C3]"
    ],
    [
     "C88",
     "x_vars[SC6,C4]"
    ],
    [
     "C89",
     "x_vars[SC6,C5]"
    ],
    [
     "C90",
     "x_vars[SC6,C6]"
    ],
    [
     "C91",
     "x_vars[SC6,C7]"
    ],
    [
     "C92",
     "x_vars[SC6,C8]"
    ],
    [
     "C93",
     "x_vars[SC6,C9]"
    ],
    [
     "C94",
     "x_vars[SC6,C10]"
    ],
    [
     "C95",
     "x_vars[SC6,C11]"
    ],
    [
     "C96",
     "x_vars[SC6,C12]"
    ],
    [
     "C97",
     "x_vars[SC6,C13]"
    ],
    [
     "C98",
     "x_vars[SC6,C14]"
    ],
    [
     "C99",
     "x_vars[SC6,C15]"
    ],
    [
     "C100",
     "x_vars[SC7,C1]"
    ],
    [
     "C101",
     "x_vars[SC7,C2]"
    ],
    [
     "C102",
     "x_vars[SC7,C3]"
    ],
    [
     "C103",
     "x_vars[SC7,C4]"
    ],
    [
     "C104",
     "x_vars[SC7,C5]"
    ],
    [
     "C105",
     "x_vars[SC7,C6]"
    ],
    [
     "C106",
     "x_vars[SC7,C7]"
    ],
    [
     "C107",
     "x_vars[SC7,C8]"
    ],
    [
     "C108",
     "x_vars[SC7,C9]"
    ],
    [
     "C109",
     "x_vars[SC7,C10]"
    ],
    [
     "C110",
     "x_vars[SC7,C11]"
    ],
    [
     "C111",
     "x_vars[SC7,C12]"
    ],
    [
     "C112",
     "x_vars[SC7,C13]"
    ],
    [
     "C113",
     "x_vars[SC7,C14]"
    ],
    [
     "C114",
     "x_vars[SC7,C15]"
    ],
    [
     "C115",
     "x_vars[SC8,C1]"
    ],
    [
     "C116",
     "x_vars[SC8,C2]"
    ],
    [
     "C117",
     "x_vars[SC8,C3]"
    ],
    [
     "C118",
     "x_vars[SC8,C4]"
    ],
    [
     "C119",
     "x_vars[SC8,C5]"
    ],
    [
     "C120",
     "x_vars[SC8,C6]"
    ],
    [
     "C121",
     "x_vars[SC8,C7]"
    ],
    [
     "C122",
     "x_vars[SC8,C8]"
    ],
    [
     "C123",
     "x_vars[SC8,C9]"
    ],
    [
     "C124",
     "x_vars[SC8,C10]"
    ],
    [
     "C125",
     "x_vars[SC8,C11]"
    ],
    [
     "C126",
     "x_vars[SC8,C12]"
    ],
    [
     "C127",
     "x_vars[SC8,C13]"
    ],
    [
     "C128",
     "x_vars[SC8,C14]"
    ],
    [
     "C129",
     "x_vars[SC8,C15]"
    ],
    [
     "C130",
     "x_vars[SC9,C1]"
    ],
    [
     "C131",
     "x_vars[SC9,C2]"
    ],
    [
     "C132",
     "x_vars[SC9,C3]"
    ],
    [
     "C133",
     "x_vars[SC9,C4]"
    ],
    [
     "C134",
     "x_vars[SC9,C5]"
    ],
    [
     "C135",
     "x_vars[SC9,C6]"
    ],
    [
     "C136",
     "x_vars[SC9,C7]"
    ],
    [
     "C137",
     "x_vars[SC9,C8]"
    ],
    [
     "C138",
     "x_vars[SC9,C9]"
    ],
    [
     "C139",
     "x_vars[SC9,C10]"
    ],
    [
     "C140",
     "x_vars[SC9,C11]"
    ],
    [
     "C141",
     "x_vars[SC9,C12]"
    ],
    [
     "C142",
     "x_vars[SC9,C13]"
    ],
    [
     "C143",
     "x_vars[SC9,C14]"
    ],
    [
     "C144",
     "x_vars[SC9,C15]"
    ],
    [
     "C145",
     "x_vars[SC10,C1]"
    ],
    [
     "C146",
     "x_vars[SC10,C2]"
    ],
    [
     "C147",
     "x_vars[SC10,C3]"
    ],
    [
     "C148",
     "x_vars[SC10,C4]"
    ],
    [
     "C149",
     "x_vars[SC10,C5]"
    ],
    [
     "C150",
     "x_vars[SC10,C6]"
    ],
    [
     "C151",
     "x_vars[SC10,C7]"
    ],
    [
     "C152",
     "x_vars[SC10,C8]"
    ],
    [
     "C153",
     "x_vars[SC10,C9]"
    ],
    [
     "C154",
     "x_vars[SC10,C10]"
    ],
    [
     "C155",
     "x_vars[SC10,C11]"
    ],
    [
     "C156",
     "x_vars[SC10,C12]"
    ],
    [
     "C157",
     "x_vars[SC10,C13]"
    ],
    [
     "C158",
     "x_vars[SC10,C14]"
    ],
    [
     "C159",
     "x_vars[SC10,C15]"
    ]
   ],
   "primary_mapping": {
    "y_SC1": "y_vars[SC1]",
    "y_SC2": "y_vars[SC2]",
    "y_SC3": "y_vars[SC3]",
    "y_SC4": "y_vars[SC4]",
    "y_SC5": "y_vars[SC5]",
    "y_SC6": "y_vars[SC6]",
    "y_SC7": "y_vars[SC7]",
    "y_SC8": "y_vars[SC8]",
    "y_SC9": "y_vars[SC9]",
    "y_SC10": "y_vars[SC10]",
    "x_SC1_C1": "x_vars[SC1,C1]",
    "x_SC1_C11": "x_vars[SC1,C11]",
    "x_SC1_C12": "x_vars[SC1,C12]",
    "x_SC1_C13": "x_vars[SC1,C13]",
    "x_SC1_C14": "x_vars[SC1,C14]",
    "x_SC1_C15": "x_vars[SC1,C15]",
    "x_SC2_C2": "x_vars[SC2,C2]",
    "x_SC2_C11": "x_vars[SC2,C11]",
    "x_SC2_C12": "x_vars[SC2,C12]",
    "x_SC2_C13": "x_vars[SC2,C13]",
    "x_SC2_C14": "x_vars[SC2,C14]",
    "x_SC2_C15": "x_vars[SC2,C15]",
    "x_SC3_C3": "x_vars[SC3,C3]",
    "x_SC3_C11": "x_vars[SC3,C11]",
    "x_SC3_C12": "x_vars[SC3,C12]",
    "x_SC3_C13": "x_vars[SC3,C13]",
    "x_SC3_C14": "x_vars[SC3,C14]",
    "x_SC3_C15": "x_vars[SC3,C15]",
    "x_SC4_C4": "x_vars[SC4,C4]",
    "x_SC4_C11": "x_vars[SC4,C11]",
    "x_SC4_C12": "x_vars[SC4,C12]",
    "x_SC4_C13": "x_vars[SC4,C13]",
    "x_SC4_C14": "x_vars[SC4,C14]",
    "x_SC4_C15": "x_vars[SC4,C15]",
    "x_SC5_C5": "x_vars[SC5,C5]",
    "x_SC5_C11": "x_vars[SC5,C11]",
    "x_SC5_C12": "x_vars[SC5,C12]",
    "x_SC5_C13": "x_vars[SC5,C13]",
    "x_SC5_C14": "x_vars[SC5,C14]",
    "x_SC5_C15": "x_vars[SC5,C15]",
    "x_SC6_C6": "x_vars[SC6,C6]",
    "x_SC6_C11": "x_vars[SC6,C11]",
    "x_SC6_C12": "x_vars[SC6,C12]",
    "x_SC6_C13": "x_vars[SC6,C13]",
    "x_SC6_C14": "x_vars[SC6,C14]",
    "x_SC6_C15": "x_vars[SC6,C15]",
    "x_SC7_C7": "x_vars[SC7,C7]",
    "x_SC7_C11": "x_vars[SC7,C11]",
    "x_SC7_C12": "x_vars[SC7,C12]",
    "x_SC7_C13": "x_vars[SC7,C13]",
    "x_SC7_C14": "x_vars[SC7,C14]",
    "x_SC7_C15": "x_vars[SC7,C15]",
    "x_SC8_C8": "x_vars[SC8,C8]",
    "x_SC8_C11": "x_vars[SC8,C11]",
    "x_SC8_C12": "x_vars[SC8,C12]",
    "x_SC8_C13": "x_vars[SC8,C13]",
    "x_SC8_C14": "x_vars[SC8,C14]",
    "x_SC8_C15": "x_vars[SC8,C15]",
    "x_SC9_C9": "x_vars[SC9,C9]",
    "x_SC9_C11": "x_vars[SC9,C11]",
    "x_SC9_C12": "x_vars[SC9,C12]",
    "x_SC9_C13": "x_vars[SC9,C13]",
    "x_SC9_C14": "x_vars[SC9,C14]",
    "x_SC9_C15": "x_vars[SC9,C15]",
    "x_SC10_C10": "x_vars[SC10,C10]",
    "x_SC10_C11": "x_vars[SC10,C11]",
    "x_SC10_C12": "x_vars[SC10,C12]",
    "x_SC10_C13": "x_vars[SC10,C13]",
    "x_SC10_C14": "x_vars[SC10,C14]",
    "x_SC10_C15": "x_vars[SC10,C15]",
    "x_SC1_C2": "x_vars[SC1,C2]",
    "x_SC1_C3": "x_vars[SC1,C3]",
    "x_SC1_C4": "x_vars[SC1,C4]",
    "x_SC1_C5": "x_vars[SC1,C5]",
    "x_SC1_C6": "x_vars[SC1,C6]",
    "x_SC1_C7": "x_vars[SC1,C7]",
    "x_SC1_C8": "x_vars[SC1,C8]",
    "x_SC1_C9": "x_vars[SC1,C9]",
    "x_SC1_C10": "x_vars[SC1,C10]",
    "x_SC2_C1": "x_vars[SC2,C1]",
    "x_SC2_C3": "x_vars[SC2,C3]",
    "x_SC2_C4": "x_vars[SC2,C4]",
    "x_SC2_C5": "x_vars[SC2,C5]",
    "x_SC2_C6": "x_vars[SC2,C6]",
    "x_SC2_C7": "x_vars[SC2,C7]",
    "x_SC2_C8": "x_vars[SC2,C8]",
    "x_SC2_C9": "x_vars[SC2,C9]",
    "x_SC2_C10": "x_vars[SC2,C10]",
    "x_SC3_C1": "x_vars[SC3,C1]",
    "x_SC3_C2": "x_vars[SC3,C2]",
    "x_SC3_C4": "x_vars[SC3,C4]",
    "x_SC3_C5": "x_vars[SC3,C5]",
    "x_SC3_C6": "x_vars[SC3,C6]",
    "x_SC3_C7": "x_vars[SC3,C7]",
    "x_SC3_C8": "x_vars[SC3,C8]",
    "x_SC3_C9": "x_vars[SC3,C9]",
    "x_SC3_C10": "x_vars[SC3,C10]",
    "x_SC4_C1": "x_vars[SC4,C1]",
    "x_SC4_C2": "x_vars[SC4,C2]",
    "x_SC4_C3": "x_vars[SC4,C3]",
    "x_SC4_C5": "x_vars[SC4,C5]",
    "x_SC4_C6": "x_vars[SC4,C6]",
    "x_SC4_C7": "x_vars[SC4,C7]",
    "x_SC4_C8": "x_vars[SC4,C8]",
    "x_SC4_C9": "x_vars[SC4,C9]",
    "x_SC4_C10": "x_vars[SC4,C10]",
    "x_SC5_C1": "x_vars[SC5,C1]",
    "x_SC5_C2": "x_vars[SC5,C2]",
    "x_SC5_C3": "x_vars[SC5,C3]",
    "x_SC5_C4": "x_vars[SC5,C4]",
    "x_SC5_C6": "x_vars[SC5,C6]",
    "x_SC5_C7": "x_vars[SC5,C7]",
    "x_SC5_C8": "x_vars[SC5,C8]",
    "x_SC5_C9": "x_vars[SC5,C9]",
    "x_SC5_C10": "x_vars[SC5,C10]",
    "x_SC6_C1": "x_vars[SC6,C1]",
    "x_SC6_C2": "x_vars[SC6,C2]",
    "x_SC6_C3": "x_vars[SC6,C3]",
    "x_SC6_C4": "x_vars[SC6,C4]",
    "x_SC6_C5": "x_vars[SC6,C5]",
    "x_SC6_C7": "x_vars[SC6,C7]",
    "x_SC6_C8": "x_vars[SC6,C8]",
    "x_SC6_C9": "x_vars[SC6,C9]",
    "x_SC6_C10": "x_vars[SC6,C10]",
    "x_SC7_C1": "x_vars[SC7,C1]",
    "x_SC7_C2": "x_vars[SC7,C2]",
    "x_SC7_C3": "x_vars[SC7,C3]",
    "x_SC7_C4": "x_vars[SC7,C4]",
    "x_SC7_C5": "x_vars[SC7,C5]",
    "x_SC7_C6": "x_vars[SC7,C6]",
    "x_SC7_C8": "x_vars[SC7,C8]",
    "x_SC7_C9": "x_vars[SC7,C9]",
    "x_SC7_C10": "x_vars[SC7,C10]",
    "x_SC8_C1": "x_vars[SC8,C1]",
    "x_SC8_C2": "x_vars[SC8,C2]",
    "x_SC8_C3": "x_vars[SC8,C3]",
    "x_SC8_C4": "x_vars[SC8,C4]",
    "x_SC8_C5": "x_vars[SC8,C5]",
    "x_SC8_C6": "x_vars[SC8,C6]",
    "x_SC8_C7": "x_vars[SC8,C7]",
    "x_SC8_C9": "x_vars[SC8,C9]",
    "x_SC8_C10": "x_vars[SC8,C10]",
    "x_SC9_C1": "x_vars[SC9,C1]",
    "x_SC9_C2": "x_vars[SC9,C2]",
    "x_SC9_C3": "x_vars[SC9,C3]",
    "x_SC9_C4": "x_vars[SC9,C4]",
    "x_SC9_C5": "x_vars[SC9,C5]",
    "x_SC9_C6": "x_vars[SC9,C6]",
    "x_SC9_C7": "x_vars[SC9,C7]",
    "x_SC9_C8": "x_vars[SC9,C8]",
    "x_SC9_C10": "x_vars[SC9,C10]",
    "x_SC10_C1": "x_vars[SC10,C1]",
    "x_SC10_C2": "x_vars[SC10,C2]",
    "x_SC10_C3": "x_vars[SC10,C3]",
    "x_SC10_C4": "x_vars[SC10,C4]",
    "x_SC10_C5": "x_vars[SC10,C5]",
    "x_SC10_C6": "x_vars[SC10,C6]",
    "x_SC10_C7": "x_vars[SC10,C7]",
    "x_SC10_C8": "x_vars[SC10,C8]",
    "x_SC10_C9": "x_vars[SC10,C9]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-090": {
   "source_code_sha256": "45f7f3f34feea0ad655f53462943df6608902bd7776734485879849a279dc87c",
   "original_solution_sha256": "6e0684ae6efc4314940b81919f714be15af9c55276287e7cc77d5d7a5f28bf85",
   "bindings": [
    [
     "C0",
     "x_vars[P1]"
    ],
    [
     "C1",
     "x_vars[P2]"
    ],
    [
     "C2",
     "x_vars[P3]"
    ],
    [
     "C3",
     "x_vars[P4]"
    ],
    [
     "C4",
     "x_vars[P5]"
    ],
    [
     "C5",
     "x_vars[P6]"
    ],
    [
     "C6",
     "x_vars[P7]"
    ],
    [
     "C7",
     "x_vars[P8]"
    ],
    [
     "C8",
     "x_vars[P9]"
    ],
    [
     "C9",
     "x_vars[P10]"
    ],
    [
     "C10",
     "x_vars[P11]"
    ],
    [
     "C11",
     "x_vars[P12]"
    ],
    [
     "C12",
     "x_vars[P13]"
    ],
    [
     "C13",
     "x_vars[P14]"
    ],
    [
     "C14",
     "x_vars[P15]"
    ],
    [
     "C15",
     "x_vars[P16]"
    ],
    [
     "C16",
     "x_vars[P17]"
    ],
    [
     "C17",
     "x_vars[P18]"
    ],
    [
     "C18",
     "x_vars[P19]"
    ],
    [
     "C19",
     "x_vars[P20]"
    ],
    [
     "C20",
     "x_vars[P21]"
    ],
    [
     "C21",
     "x_vars[P22]"
    ],
    [
     "C22",
     "x_vars[P23]"
    ],
    [
     "C23",
     "x_vars[P24]"
    ],
    [
     "C24",
     "x_vars[P25]"
    ],
    [
     "C25",
     "x_vars[P26]"
    ],
    [
     "C26",
     "x_vars[P27]"
    ],
    [
     "C27",
     "x_vars[P28]"
    ],
    [
     "C28",
     "x_vars[P29]"
    ],
    [
     "C29",
     "x_vars[P30]"
    ],
    [
     "C30",
     "x_vars[P31]"
    ],
    [
     "C31",
     "x_vars[P32]"
    ],
    [
     "C32",
     "x_vars[P33]"
    ],
    [
     "C33",
     "x_vars[P34]"
    ],
    [
     "C34",
     "x_vars[P35]"
    ],
    [
     "C35",
     "x_vars[P36]"
    ],
    [
     "C36",
     "x_vars[P37]"
    ],
    [
     "C37",
     "x_vars[P38]"
    ],
    [
     "C38",
     "x_vars[P39]"
    ],
    [
     "C39",
     "x_vars[P40]"
    ],
    [
     "C40",
     "x_vars[P41]"
    ],
    [
     "C41",
     "x_vars[P42]"
    ],
    [
     "C42",
     "x_vars[P43]"
    ],
    [
     "C43",
     "x_vars[P44]"
    ],
    [
     "C44",
     "x_vars[P45]"
    ],
    [
     "C45",
     "x_vars[P46]"
    ],
    [
     "C46",
     "x_vars[P47]"
    ],
    [
     "C47",
     "x_vars[P48]"
    ],
    [
     "C48",
     "x_vars[P49]"
    ],
    [
     "C49",
     "x_vars[P50]"
    ],
    [
     "C50",
     "x_vars[P51]"
    ],
    [
     "C51",
     "x_vars[P52]"
    ],
    [
     "C52",
     "x_vars[P53]"
    ],
    [
     "C53",
     "x_vars[P54]"
    ],
    [
     "C54",
     "x_vars[P55]"
    ],
    [
     "C55",
     "x_vars[P56]"
    ],
    [
     "C56",
     "x_vars[P57]"
    ],
    [
     "C57",
     "x_vars[P58]"
    ],
    [
     "C58",
     "x_vars[P59]"
    ],
    [
     "C59",
     "x_vars[P60]"
    ],
    [
     "C60",
     "x_vars[P61]"
    ],
    [
     "C61",
     "x_vars[P62]"
    ],
    [
     "C62",
     "x_vars[P63]"
    ],
    [
     "C63",
     "x_vars[P64]"
    ],
    [
     "C64",
     "x_vars[P65]"
    ],
    [
     "C65",
     "x_vars[P66]"
    ],
    [
     "C66",
     "x_vars[P67]"
    ],
    [
     "C67",
     "x_vars[P68]"
    ],
    [
     "C68",
     "x_vars[P69]"
    ],
    [
     "C69",
     "x_vars[P70]"
    ],
    [
     "C70",
     "x_vars[P71]"
    ],
    [
     "C71",
     "x_vars[P72]"
    ],
    [
     "C72",
     "x_vars[P73]"
    ],
    [
     "C73",
     "x_vars[P74]"
    ],
    [
     "C74",
     "x_vars[P75]"
    ],
    [
     "C75",
     "x_vars[P76]"
    ],
    [
     "C76",
     "x_vars[P77]"
    ],
    [
     "C77",
     "x_vars[P78]"
    ],
    [
     "C78",
     "x_vars[P79]"
    ],
    [
     "C79",
     "x_vars[P80]"
    ],
    [
     "C80",
     "x_vars[P81]"
    ],
    [
     "C81",
     "x_vars[P82]"
    ],
    [
     "C82",
     "x_vars[P83]"
    ],
    [
     "C83",
     "x_vars[P84]"
    ],
    [
     "C84",
     "x_vars[P85]"
    ],
    [
     "C85",
     "x_vars[P86]"
    ],
    [
     "C86",
     "x_vars[P87]"
    ],
    [
     "C87",
     "x_vars[P88]"
    ],
    [
     "C88",
     "x_vars[P89]"
    ],
    [
     "C89",
     "x_vars[P90]"
    ],
    [
     "C90",
     "x_vars[P91]"
    ],
    [
     "C91",
     "x_vars[P92]"
    ],
    [
     "C92",
     "x_vars[P93]"
    ],
    [
     "C93",
     "x_vars[P94]"
    ],
    [
     "C94",
     "x_vars[P95]"
    ],
    [
     "C95",
     "x_vars[P96]"
    ],
    [
     "C96",
     "x_vars[P97]"
    ],
    [
     "C97",
     "x_vars[P98]"
    ],
    [
     "C98",
     "x_vars[P99]"
    ],
    [
     "C99",
     "x_vars[P100]"
    ]
   ],
   "primary_mapping": {
    "x[1]": "x_vars[P1]",
    "x[2]": "x_vars[P2]",
    "x[3]": "x_vars[P3]",
    "x[4]": "x_vars[P4]",
    "x[5]": "x_vars[P5]",
    "x[6]": "x_vars[P6]",
    "x[7]": "x_vars[P7]",
    "x[8]": "x_vars[P8]",
    "x[9]": "x_vars[P9]",
    "x[10]": "x_vars[P10]",
    "x[11]": "x_vars[P11]",
    "x[12]": "x_vars[P12]",
    "x[13]": "x_vars[P13]",
    "x[14]": "x_vars[P14]",
    "x[15]": "x_vars[P15]",
    "x[16]": "x_vars[P16]",
    "x[17]": "x_vars[P17]",
    "x[18]": "x_vars[P18]",
    "x[19]": "x_vars[P19]",
    "x[20]": "x_vars[P20]",
    "x[21]": "x_vars[P21]",
    "x[22]": "x_vars[P22]",
    "x[23]": "x_vars[P23]",
    "x[24]": "x_vars[P24]",
    "x[25]": "x_vars[P25]",
    "x[26]": "x_vars[P26]",
    "x[27]": "x_vars[P27]",
    "x[28]": "x_vars[P28]",
    "x[29]": "x_vars[P29]",
    "x[30]": "x_vars[P30]",
    "x[31]": "x_vars[P31]",
    "x[32]": "x_vars[P32]",
    "x[33]": "x_vars[P33]",
    "x[34]": "x_vars[P34]",
    "x[35]": "x_vars[P35]",
    "x[36]": "x_vars[P36]",
    "x[37]": "x_vars[P37]",
    "x[38]": "x_vars[P38]",
    "x[39]": "x_vars[P39]",
    "x[40]": "x_vars[P40]",
    "x[41]": "x_vars[P41]",
    "x[42]": "x_vars[P42]",
    "x[43]": "x_vars[P43]",
    "x[44]": "x_vars[P44]",
    "x[45]": "x_vars[P45]",
    "x[46]": "x_vars[P46]",
    "x[47]": "x_vars[P47]",
    "x[48]": "x_vars[P48]",
    "x[49]": "x_vars[P49]",
    "x[50]": "x_vars[P50]",
    "x[51]": "x_vars[P51]",
    "x[52]": "x_vars[P52]",
    "x[53]": "x_vars[P53]",
    "x[54]": "x_vars[P54]",
    "x[55]": "x_vars[P55]",
    "x[56]": "x_vars[P56]",
    "x[57]": "x_vars[P57]",
    "x[58]": "x_vars[P58]",
    "x[59]": "x_vars[P59]",
    "x[60]": "x_vars[P60]",
    "x[61]": "x_vars[P61]",
    "x[62]": "x_vars[P62]",
    "x[63]": "x_vars[P63]",
    "x[64]": "x_vars[P64]",
    "x[65]": "x_vars[P65]",
    "x[66]": "x_vars[P66]",
    "x[67]": "x_vars[P67]",
    "x[68]": "x_vars[P68]",
    "x[69]": "x_vars[P69]",
    "x[70]": "x_vars[P70]",
    "x[71]": "x_vars[P71]",
    "x[72]": "x_vars[P72]",
    "x[73]": "x_vars[P73]",
    "x[74]": "x_vars[P74]",
    "x[75]": "x_vars[P75]",
    "x[76]": "x_vars[P76]",
    "x[77]": "x_vars[P77]",
    "x[78]": "x_vars[P78]",
    "x[79]": "x_vars[P79]",
    "x[80]": "x_vars[P80]",
    "x[81]": "x_vars[P81]",
    "x[82]": "x_vars[P82]",
    "x[83]": "x_vars[P83]",
    "x[84]": "x_vars[P84]",
    "x[85]": "x_vars[P85]",
    "x[86]": "x_vars[P86]",
    "x[87]": "x_vars[P87]",
    "x[88]": "x_vars[P88]",
    "x[89]": "x_vars[P89]",
    "x[90]": "x_vars[P90]",
    "x[91]": "x_vars[P91]",
    "x[92]": "x_vars[P92]",
    "x[93]": "x_vars[P93]",
    "x[94]": "x_vars[P94]",
    "x[95]": "x_vars[P95]",
    "x[96]": "x_vars[P96]",
    "x[97]": "x_vars[P97]",
    "x[98]": "x_vars[P98]",
    "x[99]": "x_vars[P99]",
    "x[100]": "x_vars[P100]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-091": {
   "source_code_sha256": "57755fc370def47d5f7a51e13cf9392ccfcb087a1ab0e3b4e210be3f4069b294",
   "original_solution_sha256": "847ffb2527b4faca7883c5666cd04a909f8149f4045b7f83785cc75113ee682c",
   "bindings": [
    [
     "C0",
     "x_vars[C22]"
    ],
    [
     "C1",
     "x_vars[C23]"
    ],
    [
     "C2",
     "x_vars[C24]"
    ],
    [
     "C3",
     "x_vars[C25]"
    ],
    [
     "C4",
     "x_vars[C26]"
    ],
    [
     "C5",
     "x_vars[C27]"
    ],
    [
     "C6",
     "x_vars[C28]"
    ]
   ],
   "primary_mapping": {
    "x[1]": "x_vars[C22]",
    "x[2]": "x_vars[C23]",
    "x[3]": "x_vars[C24]",
    "x[4]": "x_vars[C25]",
    "x[5]": "x_vars[C26]",
    "x[6]": "x_vars[C27]",
    "x[7]": "x_vars[C28]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-092": {
   "source_code_sha256": "a55a9c8ee0f64a9c58a8e7a1bd5d1758f02223595f14b94e07106d471f3a1d64",
   "original_solution_sha256": "1d91fe4a3eed6a60d8e40db31bec568e4838ce3a256ac7a9846c95f51c78a49e",
   "bindings": [
    [
     "C0",
     "t_vars[S1,D1]"
    ],
    [
     "C1",
     "t_vars[S1,D2]"
    ],
    [
     "C2",
     "t_vars[S1,D3]"
    ],
    [
     "C3",
     "t_vars[S1,D4]"
    ],
    [
     "C4",
     "t_vars[S1,D5]"
    ],
    [
     "C5",
     "t_vars[S1,D6]"
    ],
    [
     "C6",
     "t_vars[S1,D7]"
    ],
    [
     "C7",
     "t_vars[S1,D8]"
    ],
    [
     "C8",
     "t_vars[S1,D9]"
    ],
    [
     "C9",
     "t_vars[S1,D10]"
    ],
    [
     "C10",
     "t_vars[S1,D11]"
    ],
    [
     "C11",
     "t_vars[S1,D12]"
    ],
    [
     "C12",
     "t_vars[S1,D13]"
    ],
    [
     "C13",
     "t_vars[S1,D14]"
    ],
    [
     "C14",
     "t_vars[S1,D15]"
    ],
    [
     "C15",
     "t_vars[S1,D16]"
    ],
    [
     "C16",
     "t_vars[S1,D17]"
    ],
    [
     "C17",
     "t_vars[S1,D18]"
    ],
    [
     "C18",
     "t_vars[S1,D19]"
    ],
    [
     "C19",
     "t_vars[S1,D20]"
    ],
    [
     "C20",
     "t_vars[S2,D1]"
    ],
    [
     "C21",
     "t_vars[S2,D2]"
    ],
    [
     "C22",
     "t_vars[S2,D3]"
    ],
    [
     "C23",
     "t_vars[S2,D4]"
    ],
    [
     "C24",
     "t_vars[S2,D5]"
    ],
    [
     "C25",
     "t_vars[S2,D6]"
    ],
    [
     "C26",
     "t_vars[S2,D7]"
    ],
    [
     "C27",
     "t_vars[S2,D8]"
    ],
    [
     "C28",
     "t_vars[S2,D9]"
    ],
    [
     "C29",
     "t_vars[S2,D10]"
    ],
    [
     "C30",
     "t_vars[S2,D11]"
    ],
    [
     "C31",
     "t_vars[S2,D12]"
    ],
    [
     "C32",
     "t_vars[S2,D13]"
    ],
    [
     "C33",
     "t_vars[S2,D14]"
    ],
    [
     "C34",
     "t_vars[S2,D15]"
    ],
    [
     "C35",
     "t_vars[S2,D16]"
    ],
    [
     "C36",
     "t_vars[S2,D17]"
    ],
    [
     "C37",
     "t_vars[S2,D18]"
    ],
    [
     "C38",
     "t_vars[S2,D19]"
    ],
    [
     "C39",
     "t_vars[S2,D20]"
    ],
    [
     "C40",
     "t_vars[S3,D1]"
    ],
    [
     "C41",
     "t_vars[S3,D2]"
    ],
    [
     "C42",
     "t_vars[S3,D3]"
    ],
    [
     "C43",
     "t_vars[S3,D4]"
    ],
    [
     "C44",
     "t_vars[S3,D5]"
    ],
    [
     "C45",
     "t_vars[S3,D6]"
    ],
    [
     "C46",
     "t_vars[S3,D7]"
    ],
    [
     "C47",
     "t_vars[S3,D8]"
    ],
    [
     "C48",
     "t_vars[S3,D9]"
    ],
    [
     "C49",
     "t_vars[S3,D10]"
    ],
    [
     "C50",
     "t_vars[S3,D11]"
    ],
    [
     "C51",
     "t_vars[S3,D12]"
    ],
    [
     "C52",
     "t_vars[S3,D13]"
    ],
    [
     "C53",
     "t_vars[S3,D14]"
    ],
    [
     "C54",
     "t_vars[S3,D15]"
    ],
    [
     "C55",
     "t_vars[S3,D16]"
    ],
    [
     "C56",
     "t_vars[S3,D17]"
    ],
    [
     "C57",
     "t_vars[S3,D18]"
    ],
    [
     "C58",
     "t_vars[S3,D19]"
    ],
    [
     "C59",
     "t_vars[S3,D20]"
    ],
    [
     "C60",
     "t_vars[S4,D1]"
    ],
    [
     "C61",
     "t_vars[S4,D2]"
    ],
    [
     "C62",
     "t_vars[S4,D3]"
    ],
    [
     "C63",
     "t_vars[S4,D4]"
    ],
    [
     "C64",
     "t_vars[S4,D5]"
    ],
    [
     "C65",
     "t_vars[S4,D6]"
    ],
    [
     "C66",
     "t_vars[S4,D7]"
    ],
    [
     "C67",
     "t_vars[S4,D8]"
    ],
    [
     "C68",
     "t_vars[S4,D9]"
    ],
    [
     "C69",
     "t_vars[S4,D10]"
    ],
    [
     "C70",
     "t_vars[S4,D11]"
    ],
    [
     "C71",
     "t_vars[S4,D12]"
    ],
    [
     "C72",
     "t_vars[S4,D13]"
    ],
    [
     "C73",
     "t_vars[S4,D14]"
    ],
    [
     "C74",
     "t_vars[S4,D15]"
    ],
    [
     "C75",
     "t_vars[S4,D16]"
    ],
    [
     "C76",
     "t_vars[S4,D17]"
    ],
    [
     "C77",
     "t_vars[S4,D18]"
    ],
    [
     "C78",
     "t_vars[S4,D19]"
    ],
    [
     "C79",
     "t_vars[S4,D20]"
    ],
    [
     "C80",
     "t_vars[S5,D1]"
    ],
    [
     "C81",
     "t_vars[S5,D2]"
    ],
    [
     "C82",
     "t_vars[S5,D3]"
    ],
    [
     "C83",
     "t_vars[S5,D4]"
    ],
    [
     "C84",
     "t_vars[S5,D5]"
    ],
    [
     "C85",
     "t_vars[S5,D6]"
    ],
    [
     "C86",
     "t_vars[S5,D7]"
    ],
    [
     "C87",
     "t_vars[S5,D8]"
    ],
    [
     "C88",
     "t_vars[S5,D9]"
    ],
    [
     "C89",
     "t_vars[S5,D10]"
    ],
    [
     "C90",
     "t_vars[S5,D11]"
    ],
    [
     "C91",
     "t_vars[S5,D12]"
    ],
    [
     "C92",
     "t_vars[S5,D13]"
    ],
    [
     "C93",
     "t_vars[S5,D14]"
    ],
    [
     "C94",
     "t_vars[S5,D15]"
    ],
    [
     "C95",
     "t_vars[S5,D16]"
    ],
    [
     "C96",
     "t_vars[S5,D17]"
    ],
    [
     "C97",
     "t_vars[S5,D18]"
    ],
    [
     "C98",
     "t_vars[S5,D19]"
    ],
    [
     "C99",
     "t_vars[S5,D20]"
    ],
    [
     "C100",
     "t_vars[S6,D1]"
    ],
    [
     "C101",
     "t_vars[S6,D2]"
    ],
    [
     "C102",
     "t_vars[S6,D3]"
    ],
    [
     "C103",
     "t_vars[S6,D4]"
    ],
    [
     "C104",
     "t_vars[S6,D5]"
    ],
    [
     "C105",
     "t_vars[S6,D6]"
    ],
    [
     "C106",
     "t_vars[S6,D7]"
    ],
    [
     "C107",
     "t_vars[S6,D8]"
    ],
    [
     "C108",
     "t_vars[S6,D9]"
    ],
    [
     "C109",
     "t_vars[S6,D10]"
    ],
    [
     "C110",
     "t_vars[S6,D11]"
    ],
    [
     "C111",
     "t_vars[S6,D12]"
    ],
    [
     "C112",
     "t_vars[S6,D13]"
    ],
    [
     "C113",
     "t_vars[S6,D14]"
    ],
    [
     "C114",
     "t_vars[S6,D15]"
    ],
    [
     "C115",
     "t_vars[S6,D16]"
    ],
    [
     "C116",
     "t_vars[S6,D17]"
    ],
    [
     "C117",
     "t_vars[S6,D18]"
    ],
    [
     "C118",
     "t_vars[S6,D19]"
    ],
    [
     "C119",
     "t_vars[S6,D20]"
    ],
    [
     "C120",
     "t_vars[S7,D1]"
    ],
    [
     "C121",
     "t_vars[S7,D2]"
    ],
    [
     "C122",
     "t_vars[S7,D3]"
    ],
    [
     "C123",
     "t_vars[S7,D4]"
    ],
    [
     "C124",
     "t_vars[S7,D5]"
    ],
    [
     "C125",
     "t_vars[S7,D6]"
    ],
    [
     "C126",
     "t_vars[S7,D7]"
    ],
    [
     "C127",
     "t_vars[S7,D8]"
    ],
    [
     "C128",
     "t_vars[S7,D9]"
    ],
    [
     "C129",
     "t_vars[S7,D10]"
    ],
    [
     "C130",
     "t_vars[S7,D11]"
    ],
    [
     "C131",
     "t_vars[S7,D12]"
    ],
    [
     "C132",
     "t_vars[S7,D13]"
    ],
    [
     "C133",
     "t_vars[S7,D14]"
    ],
    [
     "C134",
     "t_vars[S7,D15]"
    ],
    [
     "C135",
     "t_vars[S7,D16]"
    ],
    [
     "C136",
     "t_vars[S7,D17]"
    ],
    [
     "C137",
     "t_vars[S7,D18]"
    ],
    [
     "C138",
     "t_vars[S7,D19]"
    ],
    [
     "C139",
     "t_vars[S7,D20]"
    ],
    [
     "C140",
     "t_vars[S8,D1]"
    ],
    [
     "C141",
     "t_vars[S8,D2]"
    ],
    [
     "C142",
     "t_vars[S8,D3]"
    ],
    [
     "C143",
     "t_vars[S8,D4]"
    ],
    [
     "C144",
     "t_vars[S8,D5]"
    ],
    [
     "C145",
     "t_vars[S8,D6]"
    ],
    [
     "C146",
     "t_vars[S8,D7]"
    ],
    [
     "C147",
     "t_vars[S8,D8]"
    ],
    [
     "C148",
     "t_vars[S8,D9]"
    ],
    [
     "C149",
     "t_vars[S8,D10]"
    ],
    [
     "C150",
     "t_vars[S8,D11]"
    ],
    [
     "C151",
     "t_vars[S8,D12]"
    ],
    [
     "C152",
     "t_vars[S8,D13]"
    ],
    [
     "C153",
     "t_vars[S8,D14]"
    ],
    [
     "C154",
     "t_vars[S8,D15]"
    ],
    [
     "C155",
     "t_vars[S8,D16]"
    ],
    [
     "C156",
     "t_vars[S8,D17]"
    ],
    [
     "C157",
     "t_vars[S8,D18]"
    ],
    [
     "C158",
     "t_vars[S8,D19]"
    ],
    [
     "C159",
     "t_vars[S8,D20]"
    ],
    [
     "C160",
     "t_vars[S9,D1]"
    ],
    [
     "C161",
     "t_vars[S9,D2]"
    ],
    [
     "C162",
     "t_vars[S9,D3]"
    ],
    [
     "C163",
     "t_vars[S9,D4]"
    ],
    [
     "C164",
     "t_vars[S9,D5]"
    ],
    [
     "C165",
     "t_vars[S9,D6]"
    ],
    [
     "C166",
     "t_vars[S9,D7]"
    ],
    [
     "C167",
     "t_vars[S9,D8]"
    ],
    [
     "C168",
     "t_vars[S9,D9]"
    ],
    [
     "C169",
     "t_vars[S9,D10]"
    ],
    [
     "C170",
     "t_vars[S9,D11]"
    ],
    [
     "C171",
     "t_vars[S9,D12]"
    ],
    [
     "C172",
     "t_vars[S9,D13]"
    ],
    [
     "C173",
     "t_vars[S9,D14]"
    ],
    [
     "C174",
     "t_vars[S9,D15]"
    ],
    [
     "C175",
     "t_vars[S9,D16]"
    ],
    [
     "C176",
     "t_vars[S9,D17]"
    ],
    [
     "C177",
     "t_vars[S9,D18]"
    ],
    [
     "C178",
     "t_vars[S9,D19]"
    ],
    [
     "C179",
     "t_vars[S9,D20]"
    ],
    [
     "C180",
     "t_vars[S10,D1]"
    ],
    [
     "C181",
     "t_vars[S10,D2]"
    ],
    [
     "C182",
     "t_vars[S10,D3]"
    ],
    [
     "C183",
     "t_vars[S10,D4]"
    ],
    [
     "C184",
     "t_vars[S10,D5]"
    ],
    [
     "C185",
     "t_vars[S10,D6]"
    ],
    [
     "C186",
     "t_vars[S10,D7]"
    ],
    [
     "C187",
     "t_vars[S10,D8]"
    ],
    [
     "C188",
     "t_vars[S10,D9]"
    ],
    [
     "C189",
     "t_vars[S10,D10]"
    ],
    [
     "C190",
     "t_vars[S10,D11]"
    ],
    [
     "C191",
     "t_vars[S10,D12]"
    ],
    [
     "C192",
     "t_vars[S10,D13]"
    ],
    [
     "C193",
     "t_vars[S10,D14]"
    ],
    [
     "C194",
     "t_vars[S10,D15]"
    ],
    [
     "C195",
     "t_vars[S10,D16]"
    ],
    [
     "C196",
     "t_vars[S10,D17]"
    ],
    [
     "C197",
     "t_vars[S10,D18]"
    ],
    [
     "C198",
     "t_vars[S10,D19]"
    ],
    [
     "C199",
     "t_vars[S10,D20]"
    ],
    [
     "C200",
     "x_vars[S1,D1]"
    ],
    [
     "C201",
     "x_vars[S1,D2]"
    ],
    [
     "C202",
     "x_vars[S1,D3]"
    ],
    [
     "C203",
     "x_vars[S1,D4]"
    ],
    [
     "C204",
     "x_vars[S1,D5]"
    ],
    [
     "C205",
     "x_vars[S1,D6]"
    ],
    [
     "C206",
     "x_vars[S1,D7]"
    ],
    [
     "C207",
     "x_vars[S1,D8]"
    ],
    [
     "C208",
     "x_vars[S1,D9]"
    ],
    [
     "C209",
     "x_vars[S1,D10]"
    ],
    [
     "C210",
     "x_vars[S1,D11]"
    ],
    [
     "C211",
     "x_vars[S1,D12]"
    ],
    [
     "C212",
     "x_vars[S1,D13]"
    ],
    [
     "C213",
     "x_vars[S1,D14]"
    ],
    [
     "C214",
     "x_vars[S1,D15]"
    ],
    [
     "C215",
     "x_vars[S1,D16]"
    ],
    [
     "C216",
     "x_vars[S1,D17]"
    ],
    [
     "C217",
     "x_vars[S1,D18]"
    ],
    [
     "C218",
     "x_vars[S1,D19]"
    ],
    [
     "C219",
     "x_vars[S1,D20]"
    ],
    [
     "C220",
     "x_vars[S2,D1]"
    ],
    [
     "C221",
     "x_vars[S2,D2]"
    ],
    [
     "C222",
     "x_vars[S2,D3]"
    ],
    [
     "C223",
     "x_vars[S2,D4]"
    ],
    [
     "C224",
     "x_vars[S2,D5]"
    ],
    [
     "C225",
     "x_vars[S2,D6]"
    ],
    [
     "C226",
     "x_vars[S2,D7]"
    ],
    [
     "C227",
     "x_vars[S2,D8]"
    ],
    [
     "C228",
     "x_vars[S2,D9]"
    ],
    [
     "C229",
     "x_vars[S2,D10]"
    ],
    [
     "C230",
     "x_vars[S2,D11]"
    ],
    [
     "C231",
     "x_vars[S2,D12]"
    ],
    [
     "C232",
     "x_vars[S2,D13]"
    ],
    [
     "C233",
     "x_vars[S2,D14]"
    ],
    [
     "C234",
     "x_vars[S2,D15]"
    ],
    [
     "C235",
     "x_vars[S2,D16]"
    ],
    [
     "C236",
     "x_vars[S2,D17]"
    ],
    [
     "C237",
     "x_vars[S2,D18]"
    ],
    [
     "C238",
     "x_vars[S2,D19]"
    ],
    [
     "C239",
     "x_vars[S2,D20]"
    ],
    [
     "C240",
     "x_vars[S3,D1]"
    ],
    [
     "C241",
     "x_vars[S3,D2]"
    ],
    [
     "C242",
     "x_vars[S3,D3]"
    ],
    [
     "C243",
     "x_vars[S3,D4]"
    ],
    [
     "C244",
     "x_vars[S3,D5]"
    ],
    [
     "C245",
     "x_vars[S3,D6]"
    ],
    [
     "C246",
     "x_vars[S3,D7]"
    ],
    [
     "C247",
     "x_vars[S3,D8]"
    ],
    [
     "C248",
     "x_vars[S3,D9]"
    ],
    [
     "C249",
     "x_vars[S3,D10]"
    ],
    [
     "C250",
     "x_vars[S3,D11]"
    ],
    [
     "C251",
     "x_vars[S3,D12]"
    ],
    [
     "C252",
     "x_vars[S3,D13]"
    ],
    [
     "C253",
     "x_vars[S3,D14]"
    ],
    [
     "C254",
     "x_vars[S3,D15]"
    ],
    [
     "C255",
     "x_vars[S3,D16]"
    ],
    [
     "C256",
     "x_vars[S3,D17]"
    ],
    [
     "C257",
     "x_vars[S3,D18]"
    ],
    [
     "C258",
     "x_vars[S3,D19]"
    ],
    [
     "C259",
     "x_vars[S3,D20]"
    ],
    [
     "C260",
     "x_vars[S4,D1]"
    ],
    [
     "C261",
     "x_vars[S4,D2]"
    ],
    [
     "C262",
     "x_vars[S4,D3]"
    ],
    [
     "C263",
     "x_vars[S4,D4]"
    ],
    [
     "C264",
     "x_vars[S4,D5]"
    ],
    [
     "C265",
     "x_vars[S4,D6]"
    ],
    [
     "C266",
     "x_vars[S4,D7]"
    ],
    [
     "C267",
     "x_vars[S4,D8]"
    ],
    [
     "C268",
     "x_vars[S4,D9]"
    ],
    [
     "C269",
     "x_vars[S4,D10]"
    ],
    [
     "C270",
     "x_vars[S4,D11]"
    ],
    [
     "C271",
     "x_vars[S4,D12]"
    ],
    [
     "C272",
     "x_vars[S4,D13]"
    ],
    [
     "C273",
     "x_vars[S4,D14]"
    ],
    [
     "C274",
     "x_vars[S4,D15]"
    ],
    [
     "C275",
     "x_vars[S4,D16]"
    ],
    [
     "C276",
     "x_vars[S4,D17]"
    ],
    [
     "C277",
     "x_vars[S4,D18]"
    ],
    [
     "C278",
     "x_vars[S4,D19]"
    ],
    [
     "C279",
     "x_vars[S4,D20]"
    ],
    [
     "C280",
     "x_vars[S5,D1]"
    ],
    [
     "C281",
     "x_vars[S5,D2]"
    ],
    [
     "C282",
     "x_vars[S5,D3]"
    ],
    [
     "C283",
     "x_vars[S5,D4]"
    ],
    [
     "C284",
     "x_vars[S5,D5]"
    ],
    [
     "C285",
     "x_vars[S5,D6]"
    ],
    [
     "C286",
     "x_vars[S5,D7]"
    ],
    [
     "C287",
     "x_vars[S5,D8]"
    ],
    [
     "C288",
     "x_vars[S5,D9]"
    ],
    [
     "C289",
     "x_vars[S5,D10]"
    ],
    [
     "C290",
     "x_vars[S5,D11]"
    ],
    [
     "C291",
     "x_vars[S5,D12]"
    ],
    [
     "C292",
     "x_vars[S5,D13]"
    ],
    [
     "C293",
     "x_vars[S5,D14]"
    ],
    [
     "C294",
     "x_vars[S5,D15]"
    ],
    [
     "C295",
     "x_vars[S5,D16]"
    ],
    [
     "C296",
     "x_vars[S5,D17]"
    ],
    [
     "C297",
     "x_vars[S5,D18]"
    ],
    [
     "C298",
     "x_vars[S5,D19]"
    ],
    [
     "C299",
     "x_vars[S5,D20]"
    ],
    [
     "C300",
     "x_vars[S6,D1]"
    ],
    [
     "C301",
     "x_vars[S6,D2]"
    ],
    [
     "C302",
     "x_vars[S6,D3]"
    ],
    [
     "C303",
     "x_vars[S6,D4]"
    ],
    [
     "C304",
     "x_vars[S6,D5]"
    ],
    [
     "C305",
     "x_vars[S6,D6]"
    ],
    [
     "C306",
     "x_vars[S6,D7]"
    ],
    [
     "C307",
     "x_vars[S6,D8]"
    ],
    [
     "C308",
     "x_vars[S6,D9]"
    ],
    [
     "C309",
     "x_vars[S6,D10]"
    ],
    [
     "C310",
     "x_vars[S6,D11]"
    ],
    [
     "C311",
     "x_vars[S6,D12]"
    ],
    [
     "C312",
     "x_vars[S6,D13]"
    ],
    [
     "C313",
     "x_vars[S6,D14]"
    ],
    [
     "C314",
     "x_vars[S6,D15]"
    ],
    [
     "C315",
     "x_vars[S6,D16]"
    ],
    [
     "C316",
     "x_vars[S6,D17]"
    ],
    [
     "C317",
     "x_vars[S6,D18]"
    ],
    [
     "C318",
     "x_vars[S6,D19]"
    ],
    [
     "C319",
     "x_vars[S6,D20]"
    ],
    [
     "C320",
     "x_vars[S7,D1]"
    ],
    [
     "C321",
     "x_vars[S7,D2]"
    ],
    [
     "C322",
     "x_vars[S7,D3]"
    ],
    [
     "C323",
     "x_vars[S7,D4]"
    ],
    [
     "C324",
     "x_vars[S7,D5]"
    ],
    [
     "C325",
     "x_vars[S7,D6]"
    ],
    [
     "C326",
     "x_vars[S7,D7]"
    ],
    [
     "C327",
     "x_vars[S7,D8]"
    ],
    [
     "C328",
     "x_vars[S7,D9]"
    ],
    [
     "C329",
     "x_vars[S7,D10]"
    ],
    [
     "C330",
     "x_vars[S7,D11]"
    ],
    [
     "C331",
     "x_vars[S7,D12]"
    ],
    [
     "C332",
     "x_vars[S7,D13]"
    ],
    [
     "C333",
     "x_vars[S7,D14]"
    ],
    [
     "C334",
     "x_vars[S7,D15]"
    ],
    [
     "C335",
     "x_vars[S7,D16]"
    ],
    [
     "C336",
     "x_vars[S7,D17]"
    ],
    [
     "C337",
     "x_vars[S7,D18]"
    ],
    [
     "C338",
     "x_vars[S7,D19]"
    ],
    [
     "C339",
     "x_vars[S7,D20]"
    ],
    [
     "C340",
     "x_vars[S8,D1]"
    ],
    [
     "C341",
     "x_vars[S8,D2]"
    ],
    [
     "C342",
     "x_vars[S8,D3]"
    ],
    [
     "C343",
     "x_vars[S8,D4]"
    ],
    [
     "C344",
     "x_vars[S8,D5]"
    ],
    [
     "C345",
     "x_vars[S8,D6]"
    ],
    [
     "C346",
     "x_vars[S8,D7]"
    ],
    [
     "C347",
     "x_vars[S8,D8]"
    ],
    [
     "C348",
     "x_vars[S8,D9]"
    ],
    [
     "C349",
     "x_vars[S8,D10]"
    ],
    [
     "C350",
     "x_vars[S8,D11]"
    ],
    [
     "C351",
     "x_vars[S8,D12]"
    ],
    [
     "C352",
     "x_vars[S8,D13]"
    ],
    [
     "C353",
     "x_vars[S8,D14]"
    ],
    [
     "C354",
     "x_vars[S8,D15]"
    ],
    [
     "C355",
     "x_vars[S8,D16]"
    ],
    [
     "C356",
     "x_vars[S8,D17]"
    ],
    [
     "C357",
     "x_vars[S8,D18]"
    ],
    [
     "C358",
     "x_vars[S8,D19]"
    ],
    [
     "C359",
     "x_vars[S8,D20]"
    ],
    [
     "C360",
     "x_vars[S9,D1]"
    ],
    [
     "C361",
     "x_vars[S9,D2]"
    ],
    [
     "C362",
     "x_vars[S9,D3]"
    ],
    [
     "C363",
     "x_vars[S9,D4]"
    ],
    [
     "C364",
     "x_vars[S9,D5]"
    ],
    [
     "C365",
     "x_vars[S9,D6]"
    ],
    [
     "C366",
     "x_vars[S9,D7]"
    ],
    [
     "C367",
     "x_vars[S9,D8]"
    ],
    [
     "C368",
     "x_vars[S9,D9]"
    ],
    [
     "C369",
     "x_vars[S9,D10]"
    ],
    [
     "C370",
     "x_vars[S9,D11]"
    ],
    [
     "C371",
     "x_vars[S9,D12]"
    ],
    [
     "C372",
     "x_vars[S9,D13]"
    ],
    [
     "C373",
     "x_vars[S9,D14]"
    ],
    [
     "C374",
     "x_vars[S9,D15]"
    ],
    [
     "C375",
     "x_vars[S9,D16]"
    ],
    [
     "C376",
     "x_vars[S9,D17]"
    ],
    [
     "C377",
     "x_vars[S9,D18]"
    ],
    [
     "C378",
     "x_vars[S9,D19]"
    ],
    [
     "C379",
     "x_vars[S9,D20]"
    ],
    [
     "C380",
     "x_vars[S10,D1]"
    ],
    [
     "C381",
     "x_vars[S10,D2]"
    ],
    [
     "C382",
     "x_vars[S10,D3]"
    ],
    [
     "C383",
     "x_vars[S10,D4]"
    ],
    [
     "C384",
     "x_vars[S10,D5]"
    ],
    [
     "C385",
     "x_vars[S10,D6]"
    ],
    [
     "C386",
     "x_vars[S10,D7]"
    ],
    [
     "C387",
     "x_vars[S10,D8]"
    ],
    [
     "C388",
     "x_vars[S10,D9]"
    ],
    [
     "C389",
     "x_vars[S10,D10]"
    ],
    [
     "C390",
     "x_vars[S10,D11]"
    ],
    [
     "C391",
     "x_vars[S10,D12]"
    ],
    [
     "C392",
     "x_vars[S10,D13]"
    ],
    [
     "C393",
     "x_vars[S10,D14]"
    ],
    [
     "C394",
     "x_vars[S10,D15]"
    ],
    [
     "C395",
     "x_vars[S10,D16]"
    ],
    [
     "C396",
     "x_vars[S10,D17]"
    ],
    [
     "C397",
     "x_vars[S10,D18]"
    ],
    [
     "C398",
     "x_vars[S10,D19]"
    ],
    [
     "C399",
     "x_vars[S10,D20]"
    ]
   ],
   "primary_mapping": {
    "x[1,1]": "t_vars[S1,D1]",
    "x[1,2]": "t_vars[S1,D2]",
    "x[1,3]": "t_vars[S1,D3]",
    "x[1,4]": "t_vars[S1,D4]",
    "x[1,5]": "t_vars[S1,D5]",
    "x[1,6]": "t_vars[S1,D6]",
    "x[1,7]": "t_vars[S1,D7]",
    "x[1,8]": "t_vars[S1,D8]",
    "x[1,9]": "t_vars[S1,D9]",
    "x[1,10]": "t_vars[S1,D10]",
    "x[1,11]": "t_vars[S1,D11]",
    "x[1,12]": "t_vars[S1,D12]",
    "x[1,13]": "t_vars[S1,D13]",
    "x[1,14]": "t_vars[S1,D14]",
    "x[1,15]": "t_vars[S1,D15]",
    "x[1,16]": "t_vars[S1,D16]",
    "x[1,17]": "t_vars[S1,D17]",
    "x[1,18]": "t_vars[S1,D18]",
    "x[1,19]": "t_vars[S1,D19]",
    "x[1,20]": "t_vars[S1,D20]",
    "x[2,1]": "t_vars[S2,D1]",
    "x[2,2]": "t_vars[S2,D2]",
    "x[2,3]": "t_vars[S2,D3]",
    "x[2,4]": "t_vars[S2,D4]",
    "x[2,5]": "t_vars[S2,D5]",
    "x[2,6]": "t_vars[S2,D6]",
    "x[2,7]": "t_vars[S2,D7]",
    "x[2,8]": "t_vars[S2,D8]",
    "x[2,9]": "t_vars[S2,D9]",
    "x[2,10]": "t_vars[S2,D10]",
    "x[2,11]": "t_vars[S2,D11]",
    "x[2,12]": "t_vars[S2,D12]",
    "x[2,13]": "t_vars[S2,D13]",
    "x[2,14]": "t_vars[S2,D14]",
    "x[2,15]": "t_vars[S2,D15]",
    "x[2,16]": "t_vars[S2,D16]",
    "x[2,17]": "t_vars[S2,D17]",
    "x[2,18]": "t_vars[S2,D18]",
    "x[2,19]": "t_vars[S2,D19]",
    "x[2,20]": "t_vars[S2,D20]",
    "x[3,1]": "t_vars[S3,D1]",
    "x[3,2]": "t_vars[S3,D2]",
    "x[3,3]": "t_vars[S3,D3]",
    "x[3,4]": "t_vars[S3,D4]",
    "x[3,5]": "t_vars[S3,D5]",
    "x[3,6]": "t_vars[S3,D6]",
    "x[3,7]": "t_vars[S3,D7]",
    "x[3,8]": "t_vars[S3,D8]",
    "x[3,9]": "t_vars[S3,D9]",
    "x[3,10]": "t_vars[S3,D10]",
    "x[3,11]": "t_vars[S3,D11]",
    "x[3,12]": "t_vars[S3,D12]",
    "x[3,13]": "t_vars[S3,D13]",
    "x[3,14]": "t_vars[S3,D14]",
    "x[3,15]": "t_vars[S3,D15]",
    "x[3,16]": "t_vars[S3,D16]",
    "x[3,17]": "t_vars[S3,D17]",
    "x[3,18]": "t_vars[S3,D18]",
    "x[3,19]": "t_vars[S3,D19]",
    "x[3,20]": "t_vars[S3,D20]",
    "x[4,1]": "t_vars[S4,D1]",
    "x[4,2]": "t_vars[S4,D2]",
    "x[4,3]": "t_vars[S4,D3]",
    "x[4,4]": "t_vars[S4,D4]",
    "x[4,5]": "t_vars[S4,D5]",
    "x[4,6]": "t_vars[S4,D6]",
    "x[4,7]": "t_vars[S4,D7]",
    "x[4,8]": "t_vars[S4,D8]",
    "x[4,9]": "t_vars[S4,D9]",
    "x[4,10]": "t_vars[S4,D10]",
    "x[4,11]": "t_vars[S4,D11]",
    "x[4,12]": "t_vars[S4,D12]",
    "x[4,13]": "t_vars[S4,D13]",
    "x[4,14]": "t_vars[S4,D14]",
    "x[4,15]": "t_vars[S4,D15]",
    "x[4,16]": "t_vars[S4,D16]",
    "x[4,17]": "t_vars[S4,D17]",
    "x[4,18]": "t_vars[S4,D18]",
    "x[4,19]": "t_vars[S4,D19]",
    "x[4,20]": "t_vars[S4,D20]",
    "x[5,1]": "t_vars[S5,D1]",
    "x[5,2]": "t_vars[S5,D2]",
    "x[5,3]": "t_vars[S5,D3]",
    "x[5,4]": "t_vars[S5,D4]",
    "x[5,5]": "t_vars[S5,D5]",
    "x[5,6]": "t_vars[S5,D6]",
    "x[5,7]": "t_vars[S5,D7]",
    "x[5,8]": "t_vars[S5,D8]",
    "x[5,9]": "t_vars[S5,D9]",
    "x[5,10]": "t_vars[S5,D10]",
    "x[5,11]": "t_vars[S5,D11]",
    "x[5,12]": "t_vars[S5,D12]",
    "x[5,13]": "t_vars[S5,D13]",
    "x[5,14]": "t_vars[S5,D14]",
    "x[5,15]": "t_vars[S5,D15]",
    "x[5,16]": "t_vars[S5,D16]",
    "x[5,17]": "t_vars[S5,D17]",
    "x[5,18]": "t_vars[S5,D18]",
    "x[5,19]": "t_vars[S5,D19]",
    "x[5,20]": "t_vars[S5,D20]",
    "x[6,1]": "t_vars[S6,D1]",
    "x[6,2]": "t_vars[S6,D2]",
    "x[6,3]": "t_vars[S6,D3]",
    "x[6,4]": "t_vars[S6,D4]",
    "x[6,5]": "t_vars[S6,D5]",
    "x[6,6]": "t_vars[S6,D6]",
    "x[6,7]": "t_vars[S6,D7]",
    "x[6,8]": "t_vars[S6,D8]",
    "x[6,9]": "t_vars[S6,D9]",
    "x[6,10]": "t_vars[S6,D10]",
    "x[6,11]": "t_vars[S6,D11]",
    "x[6,12]": "t_vars[S6,D12]",
    "x[6,13]": "t_vars[S6,D13]",
    "x[6,14]": "t_vars[S6,D14]",
    "x[6,15]": "t_vars[S6,D15]",
    "x[6,16]": "t_vars[S6,D16]",
    "x[6,17]": "t_vars[S6,D17]",
    "x[6,18]": "t_vars[S6,D18]",
    "x[6,19]": "t_vars[S6,D19]",
    "x[6,20]": "t_vars[S6,D20]",
    "x[7,1]": "t_vars[S7,D1]",
    "x[7,2]": "t_vars[S7,D2]",
    "x[7,3]": "t_vars[S7,D3]",
    "x[7,4]": "t_vars[S7,D4]",
    "x[7,5]": "t_vars[S7,D5]",
    "x[7,6]": "t_vars[S7,D6]",
    "x[7,7]": "t_vars[S7,D7]",
    "x[7,8]": "t_vars[S7,D8]",
    "x[7,9]": "t_vars[S7,D9]",
    "x[7,10]": "t_vars[S7,D10]",
    "x[7,11]": "t_vars[S7,D11]",
    "x[7,12]": "t_vars[S7,D12]",
    "x[7,13]": "t_vars[S7,D13]",
    "x[7,14]": "t_vars[S7,D14]",
    "x[7,15]": "t_vars[S7,D15]",
    "x[7,16]": "t_vars[S7,D16]",
    "x[7,17]": "t_vars[S7,D17]",
    "x[7,18]": "t_vars[S7,D18]",
    "x[7,19]": "t_vars[S7,D19]",
    "x[7,20]": "t_vars[S7,D20]",
    "x[8,1]": "t_vars[S8,D1]",
    "x[8,2]": "t_vars[S8,D2]",
    "x[8,3]": "t_vars[S8,D3]",
    "x[8,4]": "t_vars[S8,D4]",
    "x[8,5]": "t_vars[S8,D5]",
    "x[8,6]": "t_vars[S8,D6]",
    "x[8,7]": "t_vars[S8,D7]",
    "x[8,8]": "t_vars[S8,D8]",
    "x[8,9]": "t_vars[S8,D9]",
    "x[8,10]": "t_vars[S8,D10]",
    "x[8,11]": "t_vars[S8,D11]",
    "x[8,12]": "t_vars[S8,D12]",
    "x[8,13]": "t_vars[S8,D13]",
    "x[8,14]": "t_vars[S8,D14]",
    "x[8,15]": "t_vars[S8,D15]",
    "x[8,16]": "t_vars[S8,D16]",
    "x[8,17]": "t_vars[S8,D17]",
    "x[8,18]": "t_vars[S8,D18]",
    "x[8,19]": "t_vars[S8,D19]",
    "x[8,20]": "t_vars[S8,D20]",
    "x[9,1]": "t_vars[S9,D1]",
    "x[9,2]": "t_vars[S9,D2]",
    "x[9,3]": "t_vars[S9,D3]",
    "x[9,4]": "t_vars[S9,D4]",
    "x[9,5]": "t_vars[S9,D5]",
    "x[9,6]": "t_vars[S9,D6]",
    "x[9,7]": "t_vars[S9,D7]",
    "x[9,8]": "t_vars[S9,D8]",
    "x[9,9]": "t_vars[S9,D9]",
    "x[9,10]": "t_vars[S9,D10]",
    "x[9,11]": "t_vars[S9,D11]",
    "x[9,12]": "t_vars[S9,D12]",
    "x[9,13]": "t_vars[S9,D13]",
    "x[9,14]": "t_vars[S9,D14]",
    "x[9,15]": "t_vars[S9,D15]",
    "x[9,16]": "t_vars[S9,D16]",
    "x[9,17]": "t_vars[S9,D17]",
    "x[9,18]": "t_vars[S9,D18]",
    "x[9,19]": "t_vars[S9,D19]",
    "x[9,20]": "t_vars[S9,D20]",
    "x[10,1]": "t_vars[S10,D1]",
    "x[10,2]": "t_vars[S10,D2]",
    "x[10,3]": "t_vars[S10,D3]",
    "x[10,4]": "t_vars[S10,D4]",
    "x[10,5]": "t_vars[S10,D5]",
    "x[10,6]": "t_vars[S10,D6]",
    "x[10,7]": "t_vars[S10,D7]",
    "x[10,8]": "t_vars[S10,D8]",
    "x[10,9]": "t_vars[S10,D9]",
    "x[10,10]": "t_vars[S10,D10]",
    "x[10,11]": "t_vars[S10,D11]",
    "x[10,12]": "t_vars[S10,D12]",
    "x[10,13]": "t_vars[S10,D13]",
    "x[10,14]": "t_vars[S10,D14]",
    "x[10,15]": "t_vars[S10,D15]",
    "x[10,16]": "t_vars[S10,D16]",
    "x[10,17]": "t_vars[S10,D17]",
    "x[10,18]": "t_vars[S10,D18]",
    "x[10,19]": "t_vars[S10,D19]",
    "x[10,20]": "t_vars[S10,D20]",
    "y[1,1]": "x_vars[S1,D1]",
    "y[1,2]": "x_vars[S1,D2]",
    "y[1,3]": "x_vars[S1,D3]",
    "y[1,4]": "x_vars[S1,D4]",
    "y[1,5]": "x_vars[S1,D5]",
    "y[1,6]": "x_vars[S1,D6]",
    "y[1,7]": "x_vars[S1,D7]",
    "y[1,8]": "x_vars[S1,D8]",
    "y[1,9]": "x_vars[S1,D9]",
    "y[1,10]": "x_vars[S1,D10]",
    "y[1,11]": "x_vars[S1,D11]",
    "y[1,12]": "x_vars[S1,D12]",
    "y[1,13]": "x_vars[S1,D13]",
    "y[1,14]": "x_vars[S1,D14]",
    "y[1,15]": "x_vars[S1,D15]",
    "y[1,16]": "x_vars[S1,D16]",
    "y[1,17]": "x_vars[S1,D17]",
    "y[1,18]": "x_vars[S1,D18]",
    "y[1,19]": "x_vars[S1,D19]",
    "y[1,20]": "x_vars[S1,D20]",
    "y[2,1]": "x_vars[S2,D1]",
    "y[2,2]": "x_vars[S2,D2]",
    "y[2,3]": "x_vars[S2,D3]",
    "y[2,4]": "x_vars[S2,D4]",
    "y[2,5]": "x_vars[S2,D5]",
    "y[2,6]": "x_vars[S2,D6]",
    "y[2,7]": "x_vars[S2,D7]",
    "y[2,8]": "x_vars[S2,D8]",
    "y[2,9]": "x_vars[S2,D9]",
    "y[2,10]": "x_vars[S2,D10]",
    "y[2,11]": "x_vars[S2,D11]",
    "y[2,12]": "x_vars[S2,D12]",
    "y[2,13]": "x_vars[S2,D13]",
    "y[2,14]": "x_vars[S2,D14]",
    "y[2,15]": "x_vars[S2,D15]",
    "y[2,16]": "x_vars[S2,D16]",
    "y[2,17]": "x_vars[S2,D17]",
    "y[2,18]": "x_vars[S2,D18]",
    "y[2,19]": "x_vars[S2,D19]",
    "y[2,20]": "x_vars[S2,D20]",
    "y[3,1]": "x_vars[S3,D1]",
    "y[3,2]": "x_vars[S3,D2]",
    "y[3,3]": "x_vars[S3,D3]",
    "y[3,4]": "x_vars[S3,D4]",
    "y[3,5]": "x_vars[S3,D5]",
    "y[3,6]": "x_vars[S3,D6]",
    "y[3,7]": "x_vars[S3,D7]",
    "y[3,8]": "x_vars[S3,D8]",
    "y[3,9]": "x_vars[S3,D9]",
    "y[3,10]": "x_vars[S3,D10]",
    "y[3,11]": "x_vars[S3,D11]",
    "y[3,12]": "x_vars[S3,D12]",
    "y[3,13]": "x_vars[S3,D13]",
    "y[3,14]": "x_vars[S3,D14]",
    "y[3,15]": "x_vars[S3,D15]",
    "y[3,16]": "x_vars[S3,D16]",
    "y[3,17]": "x_vars[S3,D17]",
    "y[3,18]": "x_vars[S3,D18]",
    "y[3,19]": "x_vars[S3,D19]",
    "y[3,20]": "x_vars[S3,D20]",
    "y[4,1]": "x_vars[S4,D1]",
    "y[4,2]": "x_vars[S4,D2]",
    "y[4,3]": "x_vars[S4,D3]",
    "y[4,4]": "x_vars[S4,D4]",
    "y[4,5]": "x_vars[S4,D5]",
    "y[4,6]": "x_vars[S4,D6]",
    "y[4,7]": "x_vars[S4,D7]",
    "y[4,8]": "x_vars[S4,D8]",
    "y[4,9]": "x_vars[S4,D9]",
    "y[4,10]": "x_vars[S4,D10]",
    "y[4,11]": "x_vars[S4,D11]",
    "y[4,12]": "x_vars[S4,D12]",
    "y[4,13]": "x_vars[S4,D13]",
    "y[4,14]": "x_vars[S4,D14]",
    "y[4,15]": "x_vars[S4,D15]",
    "y[4,16]": "x_vars[S4,D16]",
    "y[4,17]": "x_vars[S4,D17]",
    "y[4,18]": "x_vars[S4,D18]",
    "y[4,19]": "x_vars[S4,D19]",
    "y[4,20]": "x_vars[S4,D20]",
    "y[5,1]": "x_vars[S5,D1]",
    "y[5,2]": "x_vars[S5,D2]",
    "y[5,3]": "x_vars[S5,D3]",
    "y[5,4]": "x_vars[S5,D4]",
    "y[5,5]": "x_vars[S5,D5]",
    "y[5,6]": "x_vars[S5,D6]",
    "y[5,7]": "x_vars[S5,D7]",
    "y[5,8]": "x_vars[S5,D8]",
    "y[5,9]": "x_vars[S5,D9]",
    "y[5,10]": "x_vars[S5,D10]",
    "y[5,11]": "x_vars[S5,D11]",
    "y[5,12]": "x_vars[S5,D12]",
    "y[5,13]": "x_vars[S5,D13]",
    "y[5,14]": "x_vars[S5,D14]",
    "y[5,15]": "x_vars[S5,D15]",
    "y[5,16]": "x_vars[S5,D16]",
    "y[5,17]": "x_vars[S5,D17]",
    "y[5,18]": "x_vars[S5,D18]",
    "y[5,19]": "x_vars[S5,D19]",
    "y[5,20]": "x_vars[S5,D20]",
    "y[6,1]": "x_vars[S6,D1]",
    "y[6,2]": "x_vars[S6,D2]",
    "y[6,3]": "x_vars[S6,D3]",
    "y[6,4]": "x_vars[S6,D4]",
    "y[6,5]": "x_vars[S6,D5]",
    "y[6,6]": "x_vars[S6,D6]",
    "y[6,7]": "x_vars[S6,D7]",
    "y[6,8]": "x_vars[S6,D8]",
    "y[6,9]": "x_vars[S6,D9]",
    "y[6,10]": "x_vars[S6,D10]",
    "y[6,11]": "x_vars[S6,D11]",
    "y[6,12]": "x_vars[S6,D12]",
    "y[6,13]": "x_vars[S6,D13]",
    "y[6,14]": "x_vars[S6,D14]",
    "y[6,15]": "x_vars[S6,D15]",
    "y[6,16]": "x_vars[S6,D16]",
    "y[6,17]": "x_vars[S6,D17]",
    "y[6,18]": "x_vars[S6,D18]",
    "y[6,19]": "x_vars[S6,D19]",
    "y[6,20]": "x_vars[S6,D20]",
    "y[7,1]": "x_vars[S7,D1]",
    "y[7,2]": "x_vars[S7,D2]",
    "y[7,3]": "x_vars[S7,D3]",
    "y[7,4]": "x_vars[S7,D4]",
    "y[7,5]": "x_vars[S7,D5]",
    "y[7,6]": "x_vars[S7,D6]",
    "y[7,7]": "x_vars[S7,D7]",
    "y[7,8]": "x_vars[S7,D8]",
    "y[7,9]": "x_vars[S7,D9]",
    "y[7,10]": "x_vars[S7,D10]",
    "y[7,11]": "x_vars[S7,D11]",
    "y[7,12]": "x_vars[S7,D12]",
    "y[7,13]": "x_vars[S7,D13]",
    "y[7,14]": "x_vars[S7,D14]",
    "y[7,15]": "x_vars[S7,D15]",
    "y[7,16]": "x_vars[S7,D16]",
    "y[7,17]": "x_vars[S7,D17]",
    "y[7,18]": "x_vars[S7,D18]",
    "y[7,19]": "x_vars[S7,D19]",
    "y[7,20]": "x_vars[S7,D20]",
    "y[8,1]": "x_vars[S8,D1]",
    "y[8,2]": "x_vars[S8,D2]",
    "y[8,3]": "x_vars[S8,D3]",
    "y[8,4]": "x_vars[S8,D4]",
    "y[8,5]": "x_vars[S8,D5]",
    "y[8,6]": "x_vars[S8,D6]",
    "y[8,7]": "x_vars[S8,D7]",
    "y[8,8]": "x_vars[S8,D8]",
    "y[8,9]": "x_vars[S8,D9]",
    "y[8,10]": "x_vars[S8,D10]",
    "y[8,11]": "x_vars[S8,D11]",
    "y[8,12]": "x_vars[S8,D12]",
    "y[8,13]": "x_vars[S8,D13]",
    "y[8,14]": "x_vars[S8,D14]",
    "y[8,15]": "x_vars[S8,D15]",
    "y[8,16]": "x_vars[S8,D16]",
    "y[8,17]": "x_vars[S8,D17]",
    "y[8,18]": "x_vars[S8,D18]",
    "y[8,19]": "x_vars[S8,D19]",
    "y[8,20]": "x_vars[S8,D20]",
    "y[9,1]": "x_vars[S9,D1]",
    "y[9,2]": "x_vars[S9,D2]",
    "y[9,3]": "x_vars[S9,D3]",
    "y[9,4]": "x_vars[S9,D4]",
    "y[9,5]": "x_vars[S9,D5]",
    "y[9,6]": "x_vars[S9,D6]",
    "y[9,7]": "x_vars[S9,D7]",
    "y[9,8]": "x_vars[S9,D8]",
    "y[9,9]": "x_vars[S9,D9]",
    "y[9,10]": "x_vars[S9,D10]",
    "y[9,11]": "x_vars[S9,D11]",
    "y[9,12]": "x_vars[S9,D12]",
    "y[9,13]": "x_vars[S9,D13]",
    "y[9,14]": "x_vars[S9,D14]",
    "y[9,15]": "x_vars[S9,D15]",
    "y[9,16]": "x_vars[S9,D16]",
    "y[9,17]": "x_vars[S9,D17]",
    "y[9,18]": "x_vars[S9,D18]",
    "y[9,19]": "x_vars[S9,D19]",
    "y[9,20]": "x_vars[S9,D20]",
    "y[10,1]": "x_vars[S10,D1]",
    "y[10,2]": "x_vars[S10,D2]",
    "y[10,3]": "x_vars[S10,D3]",
    "y[10,4]": "x_vars[S10,D4]",
    "y[10,5]": "x_vars[S10,D5]",
    "y[10,6]": "x_vars[S10,D6]",
    "y[10,7]": "x_vars[S10,D7]",
    "y[10,8]": "x_vars[S10,D8]",
    "y[10,9]": "x_vars[S10,D9]",
    "y[10,10]": "x_vars[S10,D10]",
    "y[10,11]": "x_vars[S10,D11]",
    "y[10,12]": "x_vars[S10,D12]",
    "y[10,13]": "x_vars[S10,D13]",
    "y[10,14]": "x_vars[S10,D14]",
    "y[10,15]": "x_vars[S10,D15]",
    "y[10,16]": "x_vars[S10,D16]",
    "y[10,17]": "x_vars[S10,D17]",
    "y[10,18]": "x_vars[S10,D18]",
    "y[10,19]": "x_vars[S10,D19]",
    "y[10,20]": "x_vars[S10,D20]"
   },
   "mapping_basis": "frozen_automatic_name_axis_projection"
  },
  "OR-093": {
   "source_code_sha256": "baec83dcc0910f59253f5f0c8bcf34f59ebf51a0f506c223ee1a0a2ae91567d7",
   "original_solution_sha256": "818a5456a4f6112000373feaccd8f83c1a2d753c7bf0c41e145030a3fcb182a8",
   "bindings": [
    [
     "C0",
     "x_vars[M1,A]"
    ],
    [
     "C1",
     "x_vars[M1,B]"
    ],
    [
     "C2",
     "x_vars[M1,C]"
    ],
    [
     "C3",
     "x_vars[M1,D]"
    ],
    [
     "C4",
     "x_vars[M1,E]"
    ],
    [
     "C5",
     "x_vars[M1,F]"
    ],
    [
     "C6",
     "x_vars[M1,G]"
    ],
    [
     "C7",
     "x_vars[M1,H]"
    ],
    [
     "C8",
     "x_vars[M1,I]"
    ],
    [
     "C9",
     "x_vars[M1,J]"
    ],
    [
     "C10",
     "x_vars[M1,K]"
    ],
    [
     "C11",
     "x_vars[M1,L]"
    ],
    [
     "C12",
     "x_vars[M2,A]"
    ],
    [
     "C13",
     "x_vars[M2,B]"
    ],
    [
     "C14",
     "x_vars[M2,C]"
    ],
    [
     "C15",
     "x_vars[M2,D]"
    ],
    [
     "C16",
     "x_vars[M2,E]"
    ],
    [
     "C17",
     "x_vars[M2,F]"
    ],
    [
     "C18",
     "x_vars[M2,G]"
    ],
    [
     "C19",
     "x_vars[M2,H]"
    ],
    [
     "C20",
     "x_vars[M2,I]"
    ],
    [
     "C21",
     "x_vars[M2,J]"
    ],
    [
     "C22",
     "x_vars[M2,K]"
    ],
    [
     "C23",
     "x_vars[M2,L]"
    ],
    [
     "C24",
     "x_vars[M3,A]"
    ],
    [
     "C25",
     "x_vars[M3,B]"
    ],
    [
     "C26",
     "x_vars[M3,C]"
    ],
    [
     "C27",
     "x_vars[M3,D]"
    ],
    [
     "C28",
     "x_vars[M3,E]"
    ],
    [
     "C29",
     "x_vars[M3,F]"
    ],
    [
     "C30",
     "x_vars[M3,G]"
    ],
    [
     "C31",
     "x_vars[M3,H]"
    ],
    [
     "C32",
     "x_vars[M3,I]"
    ],
    [
     "C33",
     "x_vars[M3,J]"
    ],
    [
     "C34",
     "x_vars[M3,K]"
    ],
    [
     "C35",
     "x_vars[M3,L]"
    ],
    [
     "C36",
     "x_vars[M4,A]"
    ],
    [
     "C37",
     "x_vars[M4,B]"
    ],
    [
     "C38",
     "x_vars[M4,C]"
    ],
    [
     "C39",
     "x_vars[M4,D]"
    ],
    [
     "C40",
     "x_vars[M4,E]"
    ],
    [
     "C41",
     "x_vars[M4,F]"
    ],
    [
     "C42",
     "x_vars[M4,G]"
    ],
    [
     "C43",
     "x_vars[M4,H]"
    ],
    [
     "C44",
     "x_vars[M4,I]"
    ],
    [
     "C45",
     "x_vars[M4,J]"
    ],
    [
     "C46",
     "x_vars[M4,K]"
    ],
    [
     "C47",
     "x_vars[M4,L]"
    ],
    [
     "C48",
     "x_vars[M5,A]"
    ],
    [
     "C49",
     "x_vars[M5,B]"
    ],
    [
     "C50",
     "x_vars[M5,C]"
    ],
    [
     "C51",
     "x_vars[M5,D]"
    ],
    [
     "C52",
     "x_vars[M5,E]"
    ],
    [
     "C53",
     "x_vars[M5,F]"
    ],
    [
     "C54",
     "x_vars[M5,G]"
    ],
    [
     "C55",
     "x_vars[M5,H]"
    ],
    [
     "C56",
     "x_vars[M5,I]"
    ],
    [
     "C57",
     "x_vars[M5,J]"
    ],
    [
     "C58",
     "x_vars[M5,K]"
    ],
    [
     "C59",
     "x_vars[M5,L]"
    ],
    [
     "C60",
     "x_vars[M6,A]"
    ],
    [
     "C61",
     "x_vars[M6,B]"
    ],
    [
     "C62",
     "x_vars[M6,C]"
    ],
    [
     "C63",
     "x_vars[M6,D]"
    ],
    [
     "C64",
     "x_vars[M6,E]"
    ],
    [
     "C65",
     "x_vars[M6,F]"
    ],
    [
     "C66",
     "x_vars[M6,G]"
    ],
    [
     "C67",
     "x_vars[M6,H]"
    ],
    [
     "C68",
     "x_vars[M6,I]"
    ],
    [
     "C69",
     "x_vars[M6,J]"
    ],
    [
     "C70",
     "x_vars[M6,K]"
    ],
    [
     "C71",
     "x_vars[M6,L]"
    ],
    [
     "C72",
     "x_vars[M7,A]"
    ],
    [
     "C73",
     "x_vars[M7,B]"
    ],
    [
     "C74",
     "x_vars[M7,C]"
    ],
    [
     "C75",
     "x_vars[M7,D]"
    ],
    [
     "C76",
     "x_vars[M7,E]"
    ],
    [
     "C77",
     "x_vars[M7,F]"
    ],
    [
     "C78",
     "x_vars[M7,G]"
    ],
    [
     "C79",
     "x_vars[M7,H]"
    ],
    [
     "C80",
     "x_vars[M7,I]"
    ],
    [
     "C81",
     "x_vars[M7,J]"
    ],
    [
     "C82",
     "x_vars[M7,K]"
    ],
    [
     "C83",
     "x_vars[M7,L]"
    ],
    [
     "C84",
     "x_vars[M8,A]"
    ],
    [
     "C85",
     "x_vars[M8,B]"
    ],
    [
     "C86",
     "x_vars[M8,C]"
    ],
    [
     "C87",
     "x_vars[M8,D]"
    ],
    [
     "C88",
     "x_vars[M8,E]"
    ],
    [
     "C89",
     "x_vars[M8,F]"
    ],
    [
     "C90",
     "x_vars[M8,G]"
    ],
    [
     "C91",
     "x_vars[M8,H]"
    ],
    [
     "C92",
     "x_vars[M8,I]"
    ],
    [
     "C93",
     "x_vars[M8,J]"
    ],
    [
     "C94",
     "x_vars[M8,K]"
    ],
    [
     "C95",
     "x_vars[M8,L]"
    ],
    [
     "C96",
     "x_vars[M9,A]"
    ],
    [
     "C97",
     "x_vars[M9,B]"
    ],
    [
     "C98",
     "x_vars[M9,C]"
    ],
    [
     "C99",
     "x_vars[M9,D]"
    ],
    [
     "C100",
     "x_vars[M9,E]"
    ],
    [
     "C101",
     "x_vars[M9,F]"
    ],
    [
     "C102",
     "x_vars[M9,G]"
    ],
    [
     "C103",
     "x_vars[M9,H]"
    ],
    [
     "C104",
     "x_vars[M9,I]"
    ],
    [
     "C105",
     "x_vars[M9,J]"
    ],
    [
     "C106",
     "x_vars[M9,K]"
    ],
    [
     "C107",
     "x_vars[M9,L]"
    ],
    [
     "C108",
     "x_vars[M10,A]"
    ],
    [
     "C109",
     "x_vars[M10,B]"
    ],
    [
     "C110",
     "x_vars[M10,C]"
    ],
    [
     "C111",
     "x_vars[M10,D]"
    ],
    [
     "C112",
     "x_vars[M10,E]"
    ],
    [
     "C113",
     "x_vars[M10,F]"
    ],
    [
     "C114",
     "x_vars[M10,G]"
    ],
    [
     "C115",
     "x_vars[M10,H]"
    ],
    [
     "C116",
     "x_vars[M10,I]"
    ],
    [
     "C117",
     "x_vars[M10,J]"
    ],
    [
     "C118",
     "x_vars[M10,K]"
    ],
    [
     "C119",
     "x_vars[M10,L]"
    ],
    [
     "C120",
     "x_vars[M11,A]"
    ],
    [
     "C121",
     "x_vars[M11,B]"
    ],
    [
     "C122",
     "x_vars[M11,C]"
    ],
    [
     "C123",
     "x_vars[M11,D]"
    ],
    [
     "C124",
     "x_vars[M11,E]"
    ],
    [
     "C125",
     "x_vars[M11,F]"
    ],
    [
     "C126",
     "x_vars[M11,G]"
    ],
    [
     "C127",
     "x_vars[M11,H]"
    ],
    [
     "C128",
     "x_vars[M11,I]"
    ],
    [
     "C129",
     "x_vars[M11,J]"
    ],
    [
     "C130",
     "x_vars[M11,K]"
    ],
    [
     "C131",
     "x_vars[M11,L]"
    ],
    [
     "C132",
     "x_vars[M12,A]"
    ],
    [
     "C133",
     "x_vars[M12,B]"
    ],
    [
     "C134",
     "x_vars[M12,C]"
    ],
    [
     "C135",
     "x_vars[M12,D]"
    ],
    [
     "C136",
     "x_vars[M12,E]"
    ],
    [
     "C137",
     "x_vars[M12,F]"
    ],
    [
     "C138",
     "x_vars[M12,G]"
    ],
    [
     "C139",
     "x_vars[M12,H]"
    ],
    [
     "C140",
     "x_vars[M12,I]"
    ],
    [
     "C141",
     "x_vars[M12,J]"
    ],
    [
     "C142",
     "x_vars[M12,K]"
    ],
    [
     "C143",
     "x_vars[M12,L]"
    ]
   ],
   "primary_mapping": {
    "x[1,A]": "x_vars[M1,A]",
    "x[1,B]": "x_vars[M1,B]",
    "x[1,C]": "x_vars[M1,C]",
    "x[1,D]": "x_vars[M1,D]",
    "x[1,E]": "x_vars[M1,E]",
    "x[1,F]": "x_vars[M1,F]",
    "x[1,G]": "x_vars[M1,G]",
    "x[1,H]": "x_vars[M1,H]",
    "x[1,I]": "x_vars[M1,I]",
    "x[1,J]": "x_vars[M1,J]",
    "x[1,K]": "x_vars[M1,K]",
    "x[1,L]": "x_vars[M1,L]",
    "x[2,A]": "x_vars[M2,A]",
    "x[2,B]": "x_vars[M2,B]",
    "x[2,C]": "x_vars[M2,C]",
    "x[2,D]": "x_vars[M2,D]",
    "x[2,E]": "x_vars[M2,E]",
    "x[2,F]": "x_vars[M2,F]",
    "x[2,G]": "x_vars[M2,G]",
    "x[2,H]": "x_vars[M2,H]",
    "x[2,I]": "x_vars[M2,I]",
    "x[2,J]": "x_vars[M2,J]",
    "x[2,K]": "x_vars[M2,K]",
    "x[2,L]": "x_vars[M2,L]",
    "x[3,A]": "x_vars[M3,A]",
    "x[3,B]": "x_vars[M3,B]",
    "x[3,C]": "x_vars[M3,C]",
    "x[3,D]": "x_vars[M3,D]",
    "x[3,E]": "x_vars[M3,E]",
    "x[3,F]": "x_vars[M3,F]",
    "x[3,G]": "x_vars[M3,G]",
    "x[3,H]": "x_vars[M3,H]",
    "x[3,I]": "x_vars[M3,I]",
    "x[3,J]": "x_vars[M3,J]",
    "x[3,K]": "x_vars[M3,K]",
    "x[3,L]": "x_vars[M3,L]",
    "x[4,A]": "x_vars[M4,A]",
    "x[4,B]": "x_vars[M4,B]",
    "x[4,C]": "x_vars[M4,C]",
    "x[4,D]": "x_vars[M4,D]",
    "x[4,E]": "x_vars[M4,E]",
    "x[4,F]": "x_vars[M4,F]",
    "x[4,G]": "x_vars[M4,G]",
    "x[4,H]": "x_vars[M4,H]",
    "x[4,I]": "x_vars[M4,I]",
    "x[4,J]": "x_vars[M4,J]",
    "x[4,K]": "x_vars[M4,K]",
    "x[4,L]": "x_vars[M4,L]",
    "x[5,A]": "x_vars[M5,A]",
    "x[5,B]": "x_vars[M5,B]",
    "x[5,C]": "x_vars[M5,C]",
    "x[5,D]": "x_vars[M5,D]",
    "x[5,E]": "x_vars[M5,E]",
    "x[5,F]": "x_vars[M5,F]",
    "x[5,G]": "x_vars[M5,G]",
    "x[5,H]": "x_vars[M5,H]",
    "x[5,I]": "x_vars[M5,I]",
    "x[5,J]": "x_vars[M5,J]",
    "x[5,K]": "x_vars[M5,K]",
    "x[5,L]": "x_vars[M5,L]",
    "x[6,A]": "x_vars[M6,A]",
    "x[6,B]": "x_vars[M6,B]",
    "x[6,C]": "x_vars[M6,C]",
    "x[6,D]": "x_vars[M6,D]",
    "x[6,E]": "x_vars[M6,E]",
    "x[6,F]": "x_vars[M6,F]",
    "x[6,G]": "x_vars[M6,G]",
    "x[6,H]": "x_vars[M6,H]",
    "x[6,I]": "x_vars[M6,I]",
    "x[6,J]": "x_vars[M6,J]",
    "x[6,K]": "x_vars[M6,K]",
    "x[6,L]": "x_vars[M6,L]",
    "x[7,A]": "x_vars[M7,A]",
    "x[7,B]": "x_vars[M7,B]",
    "x[7,C]": "x_vars[M7,C]",
    "x[7,D]": "x_vars[M7,D]",
    "x[7,E]": "x_vars[M7,E]",
    "x[7,F]": "x_vars[M7,F]",
    "x[7,G]": "x_vars[M7,G]",
    "x[7,H]": "x_vars[M7,H]",
    "x[7,I]": "x_vars[M7,I]",
    "x[7,J]": "x_vars[M7,J]",
    "x[7,K]": "x_vars[M7,K]",
    "x[7,L]": "x_vars[M7,L]",
    "x[8,A]": "x_vars[M8,A]",
    "x[8,B]": "x_vars[M8,B]",
    "x[8,C]": "x_vars[M8,C]",
    "x[8,D]": "x_vars[M8,D]",
    "x[8,E]": "x_vars[M8,E]",
    "x[8,F]": "x_vars[M8,F]",
    "x[8,G]": "x_vars[M8,G]",
    "x[8,H]": "x_vars[M8,H]",
    "x[8,I]": "x_vars[M8,I]",
    "x[8,J]": "x_vars[M8,J]",
    "x[8,K]": "x_vars[M8,K]",
    "x[8,L]": "x_vars[M8,L]",
    "x[9,A]": "x_vars[M9,A]",
    "x[9,B]": "x_vars[M9,B]",
    "x[9,C]": "x_vars[M9,C]",
    "x[9,D]": "x_vars[M9,D]",
    "x[9,E]": "x_vars[M9,E]",
    "x[9,F]": "x_vars[M9,F]",
    "x[9,G]": "x_vars[M9,G]",
    "x[9,H]": "x_vars[M9,H]",
    "x[9,I]": "x_vars[M9,I]",
    "x[9,J]": "x_vars[M9,J]",
    "x[9,K]": "x_vars[M9,K]",
    "x[9,L]": "x_vars[M9,L]",
    "x[10,A]": "x_vars[M10,A]",
    "x[10,B]": "x_vars[M10,B]",
    "x[10,C]": "x_vars[M10,C]",
    "x[10,D]": "x_vars[M10,D]",
    "x[10,E]": "x_vars[M10,E]",
    "x[10,F]": "x_vars[M10,F]",
    "x[10,G]": "x_vars[M10,G]",
    "x[10,H]": "x_vars[M10,H]",
    "x[10,I]": "x_vars[M10,I]",
    "x[10,J]": "x_vars[M10,J]",
    "x[10,K]": "x_vars[M10,K]",
    "x[10,L]": "x_vars[M10,L]",
    "x[11,A]": "x_vars[M11,A]",
    "x[11,B]": "x_vars[M11,B]",
    "x[11,C]": "x_vars[M11,C]",
    "x[11,D]": "x_vars[M11,D]",
    "x[11,E]": "x_vars[M11,E]",
    "x[11,F]": "x_vars[M11,F]",
    "x[11,G]": "x_vars[M11,G]",
    "x[11,H]": "x_vars[M11,H]",
    "x[11,I]": "x_vars[M11,I]",
    "x[11,J]": "x_vars[M11,J]",
    "x[11,K]": "x_vars[M11,K]",
    "x[11,L]": "x_vars[M11,L]",
    "x[12,A]": "x_vars[M12,A]",
    "x[12,B]": "x_vars[M12,B]",
    "x[12,C]": "x_vars[M12,C]",
    "x[12,D]": "x_vars[M12,D]",
    "x[12,E]": "x_vars[M12,E]",
    "x[12,F]": "x_vars[M12,F]",
    "x[12,G]": "x_vars[M12,G]",
    "x[12,H]": "x_vars[M12,H]",
    "x[12,I]": "x_vars[M12,I]",
    "x[12,J]": "x_vars[M12,J]",
    "x[12,K]": "x_vars[M12,K]",
    "x[12,L]": "x_vars[M12,L]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-094": {
   "source_code_sha256": "2b394d68a66894ccd02b71f587af89f2fff19c93d4c5bb50898c66eb56f202f4",
   "original_solution_sha256": "a0db7118366407996ddcf2e8fb576d74dd67f464ce69946a284b502f95a67ec5",
   "bindings": [
    [
     "C0",
     "x_vars[HiFi1]"
    ],
    [
     "C1",
     "x_vars[HiFi2]"
    ],
    [
     "C2",
     "x_vars[HiFi3]"
    ],
    [
     "C3",
     "x_vars[HiFi4]"
    ],
    [
     "C4",
     "x_vars[HiFi5]"
    ],
    [
     "C5",
     "x_vars[HiFi6]"
    ],
    [
     "C6",
     "x_vars[HiFi7]"
    ],
    [
     "C7",
     "x_vars[HiFi8]"
    ],
    [
     "C8",
     "x_vars[HiFi9]"
    ],
    [
     "C9",
     "x_vars[HiFi10]"
    ],
    [
     "C10",
     "x_vars[HiFi11]"
    ],
    [
     "C11",
     "x_vars[HiFi12]"
    ],
    [
     "C12",
     "x_vars[HiFi13]"
    ],
    [
     "C13",
     "x_vars[HiFi14]"
    ],
    [
     "C14",
     "x_vars[HiFi15]"
    ],
    [
     "C15",
     "x_vars[HiFi16]"
    ],
    [
     "C16",
     "x_vars[HiFi17]"
    ],
    [
     "C17",
     "x_vars[HiFi18]"
    ],
    [
     "C18",
     "x_vars[HiFi19]"
    ],
    [
     "C19",
     "x_vars[HiFi20]"
    ],
    [
     "C20",
     "x_vars[HiFi21]"
    ],
    [
     "C21",
     "x_vars[HiFi22]"
    ],
    [
     "C22",
     "x_vars[HiFi23]"
    ],
    [
     "C23",
     "x_vars[HiFi24]"
    ],
    [
     "C24",
     "x_vars[HiFi25]"
    ],
    [
     "C25",
     "x_vars[HiFi26]"
    ],
    [
     "C26",
     "x_vars[HiFi27]"
    ],
    [
     "C27",
     "x_vars[HiFi28]"
    ],
    [
     "C28",
     "x_vars[HiFi29]"
    ],
    [
     "C29",
     "x_vars[HiFi30]"
    ],
    [
     "C30",
     "x_vars[HiFi31]"
    ],
    [
     "C31",
     "x_vars[HiFi32]"
    ],
    [
     "C32",
     "x_vars[HiFi33]"
    ],
    [
     "C33",
     "x_vars[HiFi34]"
    ],
    [
     "C34",
     "x_vars[HiFi35]"
    ],
    [
     "C35",
     "x_vars[HiFi36]"
    ],
    [
     "C36",
     "x_vars[HiFi37]"
    ],
    [
     "C37",
     "x_vars[HiFi38]"
    ],
    [
     "C38",
     "x_vars[HiFi39]"
    ],
    [
     "C39",
     "x_vars[HiFi40]"
    ],
    [
     "C40",
     "x_vars[HiFi41]"
    ],
    [
     "C41",
     "x_vars[HiFi42]"
    ],
    [
     "C42",
     "x_vars[HiFi43]"
    ],
    [
     "C43",
     "x_vars[HiFi44]"
    ],
    [
     "C44",
     "x_vars[HiFi45]"
    ],
    [
     "C45",
     "x_vars[HiFi46]"
    ],
    [
     "C46",
     "x_vars[HiFi47]"
    ],
    [
     "C47",
     "x_vars[HiFi48]"
    ],
    [
     "C48",
     "x_vars[HiFi49]"
    ],
    [
     "C49",
     "x_vars[HiFi50]"
    ],
    [
     "C50",
     "x_vars[HiFi51]"
    ],
    [
     "C51",
     "x_vars[HiFi52]"
    ],
    [
     "C52",
     "x_vars[HiFi53]"
    ],
    [
     "C53",
     "x_vars[HiFi54]"
    ],
    [
     "C54",
     "x_vars[HiFi55]"
    ],
    [
     "C55",
     "x_vars[HiFi56]"
    ],
    [
     "C56",
     "x_vars[HiFi57]"
    ],
    [
     "C57",
     "x_vars[HiFi58]"
    ],
    [
     "C58",
     "x_vars[HiFi59]"
    ],
    [
     "C59",
     "x_vars[HiFi60]"
    ],
    [
     "C60",
     "x_vars[HiFi61]"
    ],
    [
     "C61",
     "x_vars[HiFi62]"
    ],
    [
     "C62",
     "x_vars[HiFi63]"
    ],
    [
     "C63",
     "x_vars[HiFi64]"
    ],
    [
     "C64",
     "x_vars[HiFi65]"
    ],
    [
     "C65",
     "x_vars[HiFi66]"
    ],
    [
     "C66",
     "x_vars[HiFi67]"
    ],
    [
     "C67",
     "x_vars[HiFi68]"
    ],
    [
     "C68",
     "x_vars[HiFi69]"
    ],
    [
     "C69",
     "x_vars[HiFi70]"
    ],
    [
     "C70",
     "x_vars[HiFi71]"
    ],
    [
     "C71",
     "x_vars[HiFi72]"
    ],
    [
     "C72",
     "x_vars[HiFi73]"
    ],
    [
     "C73",
     "x_vars[HiFi74]"
    ],
    [
     "C74",
     "x_vars[HiFi75]"
    ],
    [
     "C75",
     "x_vars[HiFi76]"
    ],
    [
     "C76",
     "x_vars[HiFi77]"
    ],
    [
     "C77",
     "x_vars[HiFi78]"
    ],
    [
     "C78",
     "x_vars[HiFi79]"
    ],
    [
     "C79",
     "x_vars[HiFi80]"
    ],
    [
     "C80",
     "x_vars[HiFi81]"
    ],
    [
     "C81",
     "x_vars[HiFi82]"
    ],
    [
     "C82",
     "x_vars[HiFi83]"
    ],
    [
     "C83",
     "x_vars[HiFi84]"
    ],
    [
     "C84",
     "x_vars[HiFi85]"
    ],
    [
     "C85",
     "x_vars[HiFi86]"
    ],
    [
     "C86",
     "x_vars[HiFi87]"
    ],
    [
     "C87",
     "x_vars[HiFi88]"
    ],
    [
     "C88",
     "x_vars[HiFi89]"
    ],
    [
     "C89",
     "x_vars[HiFi90]"
    ],
    [
     "C90",
     "x_vars[HiFi91]"
    ],
    [
     "C91",
     "x_vars[HiFi92]"
    ],
    [
     "C92",
     "x_vars[HiFi93]"
    ],
    [
     "C93",
     "x_vars[HiFi94]"
    ],
    [
     "C94",
     "x_vars[HiFi95]"
    ],
    [
     "C95",
     "x_vars[HiFi96]"
    ],
    [
     "C96",
     "x_vars[HiFi97]"
    ],
    [
     "C97",
     "x_vars[HiFi98]"
    ],
    [
     "C98",
     "x_vars[HiFi99]"
    ],
    [
     "C99",
     "x_vars[HiFi100]"
    ],
    [
     "C100",
     "x_vars[HiFi101]"
    ],
    [
     "C101",
     "y_vars[1]"
    ],
    [
     "C102",
     "y_vars[2]"
    ],
    [
     "C103",
     "y_vars[3]"
    ]
   ],
   "primary_mapping": {
    "x[0]": "x_vars[HiFi1]",
    "x[1]": "x_vars[HiFi2]",
    "x[2]": "x_vars[HiFi3]",
    "x[3]": "x_vars[HiFi4]",
    "x[4]": "x_vars[HiFi5]",
    "x[5]": "x_vars[HiFi6]",
    "x[6]": "x_vars[HiFi7]",
    "x[7]": "x_vars[HiFi8]",
    "x[8]": "x_vars[HiFi9]",
    "x[9]": "x_vars[HiFi10]",
    "x[10]": "x_vars[HiFi11]",
    "x[11]": "x_vars[HiFi12]",
    "x[12]": "x_vars[HiFi13]",
    "x[13]": "x_vars[HiFi14]",
    "x[14]": "x_vars[HiFi15]",
    "x[15]": "x_vars[HiFi16]",
    "x[16]": "x_vars[HiFi17]",
    "x[17]": "x_vars[HiFi18]",
    "x[18]": "x_vars[HiFi19]",
    "x[19]": "x_vars[HiFi20]",
    "x[20]": "x_vars[HiFi21]",
    "x[21]": "x_vars[HiFi22]",
    "x[22]": "x_vars[HiFi23]",
    "x[23]": "x_vars[HiFi24]",
    "x[24]": "x_vars[HiFi25]",
    "x[25]": "x_vars[HiFi26]",
    "x[26]": "x_vars[HiFi27]",
    "x[27]": "x_vars[HiFi28]",
    "x[28]": "x_vars[HiFi29]",
    "x[29]": "x_vars[HiFi30]",
    "x[30]": "x_vars[HiFi31]",
    "x[31]": "x_vars[HiFi32]",
    "x[32]": "x_vars[HiFi33]",
    "x[33]": "x_vars[HiFi34]",
    "x[34]": "x_vars[HiFi35]",
    "x[35]": "x_vars[HiFi36]",
    "x[36]": "x_vars[HiFi37]",
    "x[37]": "x_vars[HiFi38]",
    "x[38]": "x_vars[HiFi39]",
    "x[39]": "x_vars[HiFi40]",
    "x[40]": "x_vars[HiFi41]",
    "x[41]": "x_vars[HiFi42]",
    "x[42]": "x_vars[HiFi43]",
    "x[43]": "x_vars[HiFi44]",
    "x[44]": "x_vars[HiFi45]",
    "x[45]": "x_vars[HiFi46]",
    "x[46]": "x_vars[HiFi47]",
    "x[47]": "x_vars[HiFi48]",
    "x[48]": "x_vars[HiFi49]",
    "x[49]": "x_vars[HiFi50]",
    "x[50]": "x_vars[HiFi51]",
    "x[51]": "x_vars[HiFi52]",
    "x[52]": "x_vars[HiFi53]",
    "x[53]": "x_vars[HiFi54]",
    "x[54]": "x_vars[HiFi55]",
    "x[55]": "x_vars[HiFi56]",
    "x[56]": "x_vars[HiFi57]",
    "x[57]": "x_vars[HiFi58]",
    "x[58]": "x_vars[HiFi59]",
    "x[59]": "x_vars[HiFi60]",
    "x[60]": "x_vars[HiFi61]",
    "x[61]": "x_vars[HiFi62]",
    "x[62]": "x_vars[HiFi63]",
    "x[63]": "x_vars[HiFi64]",
    "x[64]": "x_vars[HiFi65]",
    "x[65]": "x_vars[HiFi66]",
    "x[66]": "x_vars[HiFi67]",
    "x[67]": "x_vars[HiFi68]",
    "x[68]": "x_vars[HiFi69]",
    "x[69]": "x_vars[HiFi70]",
    "x[70]": "x_vars[HiFi71]",
    "x[71]": "x_vars[HiFi72]",
    "x[72]": "x_vars[HiFi73]",
    "x[73]": "x_vars[HiFi74]",
    "x[74]": "x_vars[HiFi75]",
    "x[75]": "x_vars[HiFi76]",
    "x[76]": "x_vars[HiFi77]",
    "x[77]": "x_vars[HiFi78]",
    "x[78]": "x_vars[HiFi79]",
    "x[79]": "x_vars[HiFi80]",
    "x[80]": "x_vars[HiFi81]",
    "x[81]": "x_vars[HiFi82]",
    "x[82]": "x_vars[HiFi83]",
    "x[83]": "x_vars[HiFi84]",
    "x[84]": "x_vars[HiFi85]",
    "x[85]": "x_vars[HiFi86]",
    "x[86]": "x_vars[HiFi87]",
    "x[87]": "x_vars[HiFi88]",
    "x[88]": "x_vars[HiFi89]",
    "x[89]": "x_vars[HiFi90]",
    "x[90]": "x_vars[HiFi91]",
    "x[91]": "x_vars[HiFi92]",
    "x[92]": "x_vars[HiFi93]",
    "x[93]": "x_vars[HiFi94]",
    "x[94]": "x_vars[HiFi95]",
    "x[95]": "x_vars[HiFi96]",
    "x[96]": "x_vars[HiFi97]",
    "x[97]": "x_vars[HiFi98]",
    "x[98]": "x_vars[HiFi99]",
    "x[99]": "x_vars[HiFi100]",
    "x[100]": "x_vars[HiFi101]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-095": {
   "source_code_sha256": "74ad121cdfed4e37f43c7edb0db7378ba82462a2e413a04e3a15f692beedbd71",
   "original_solution_sha256": "d3643c67fb9bdbdb71147fdae974adff6f7fe9862f15ec3f6ed600fb55d27605",
   "bindings": [
    [
     "C0",
     "x_vars[Widget1]"
    ],
    [
     "C1",
     "x_vars[Widget2]"
    ],
    [
     "C2",
     "x_vars[Widget3]"
    ],
    [
     "C3",
     "x_vars[Widget4]"
    ],
    [
     "C4",
     "x_vars[Widget5]"
    ],
    [
     "C5",
     "x_vars[Widget6]"
    ],
    [
     "C6",
     "x_vars[Widget7]"
    ],
    [
     "C7",
     "x_vars[Widget8]"
    ],
    [
     "C8",
     "x_vars[Widget9]"
    ],
    [
     "C9",
     "x_vars[Widget10]"
    ],
    [
     "C10",
     "x_vars[Widget11]"
    ],
    [
     "C11",
     "x_vars[Widget12]"
    ],
    [
     "C12",
     "x_vars[Widget13]"
    ],
    [
     "C13",
     "x_vars[Widget14]"
    ],
    [
     "C14",
     "x_vars[Widget15]"
    ],
    [
     "C15",
     "x_vars[Widget16]"
    ],
    [
     "C16",
     "x_vars[Widget17]"
    ],
    [
     "C17",
     "x_vars[Widget18]"
    ],
    [
     "C18",
     "x_vars[Widget19]"
    ],
    [
     "C19",
     "x_vars[Widget20]"
    ],
    [
     "C20",
     "x_vars[Widget21]"
    ],
    [
     "C21",
     "x_vars[Widget22]"
    ],
    [
     "C22",
     "x_vars[Widget23]"
    ],
    [
     "C23",
     "x_vars[Widget24]"
    ],
    [
     "C24",
     "x_vars[Widget25]"
    ],
    [
     "C25",
     "x_vars[Widget26]"
    ],
    [
     "C26",
     "x_vars[Widget27]"
    ],
    [
     "C27",
     "x_vars[Widget28]"
    ],
    [
     "C28",
     "x_vars[Widget29]"
    ],
    [
     "C29",
     "x_vars[Widget30]"
    ],
    [
     "C30",
     "x_vars[Widget31]"
    ],
    [
     "C31",
     "x_vars[Widget32]"
    ],
    [
     "C32",
     "x_vars[Widget33]"
    ],
    [
     "C33",
     "x_vars[Widget34]"
    ],
    [
     "C34",
     "x_vars[Widget35]"
    ],
    [
     "C35",
     "x_vars[Widget36]"
    ],
    [
     "C36",
     "x_vars[Widget37]"
    ],
    [
     "C37",
     "x_vars[Widget38]"
    ],
    [
     "C38",
     "x_vars[Widget39]"
    ],
    [
     "C39",
     "x_vars[Widget40]"
    ],
    [
     "C40",
     "x_vars[Widget41]"
    ],
    [
     "C41",
     "x_vars[Widget42]"
    ],
    [
     "C42",
     "x_vars[Widget43]"
    ],
    [
     "C43",
     "x_vars[Widget44]"
    ],
    [
     "C44",
     "x_vars[Widget45]"
    ],
    [
     "C45",
     "x_vars[Widget46]"
    ],
    [
     "C46",
     "x_vars[Widget47]"
    ],
    [
     "C47",
     "x_vars[Widget48]"
    ],
    [
     "C48",
     "x_vars[Widget49]"
    ],
    [
     "C49",
     "x_vars[Widget50]"
    ],
    [
     "C50",
     "x_vars[Widget51]"
    ],
    [
     "C51",
     "x_vars[Widget52]"
    ],
    [
     "C52",
     "x_vars[Widget53]"
    ],
    [
     "C53",
     "x_vars[Widget54]"
    ],
    [
     "C54",
     "x_vars[Widget55]"
    ],
    [
     "C55",
     "x_vars[Widget56]"
    ],
    [
     "C56",
     "x_vars[Widget57]"
    ],
    [
     "C57",
     "x_vars[Widget58]"
    ],
    [
     "C58",
     "x_vars[Widget59]"
    ],
    [
     "C59",
     "x_vars[Widget60]"
    ],
    [
     "C60",
     "x_vars[Widget61]"
    ],
    [
     "C61",
     "x_vars[Widget62]"
    ],
    [
     "C62",
     "x_vars[Widget63]"
    ],
    [
     "C63",
     "x_vars[Widget64]"
    ],
    [
     "C64",
     "x_vars[Widget65]"
    ],
    [
     "C65",
     "x_vars[Widget66]"
    ],
    [
     "C66",
     "x_vars[Widget67]"
    ],
    [
     "C67",
     "x_vars[Widget68]"
    ],
    [
     "C68",
     "x_vars[Widget69]"
    ],
    [
     "C69",
     "x_vars[Widget70]"
    ],
    [
     "C70",
     "x_vars[Widget71]"
    ],
    [
     "C71",
     "x_vars[Widget72]"
    ],
    [
     "C72",
     "x_vars[Widget73]"
    ],
    [
     "C73",
     "x_vars[Widget74]"
    ],
    [
     "C74",
     "x_vars[Widget75]"
    ],
    [
     "C75",
     "x_vars[Widget76]"
    ],
    [
     "C76",
     "x_vars[Widget77]"
    ],
    [
     "C77",
     "x_vars[Widget78]"
    ],
    [
     "C78",
     "x_vars[Widget79]"
    ],
    [
     "C79",
     "x_vars[Widget80]"
    ],
    [
     "C80",
     "x_vars[Widget81]"
    ],
    [
     "C81",
     "x_vars[Widget82]"
    ],
    [
     "C82",
     "x_vars[Widget83]"
    ],
    [
     "C83",
     "x_vars[Widget84]"
    ],
    [
     "C84",
     "x_vars[Widget85]"
    ],
    [
     "C85",
     "x_vars[Widget86]"
    ],
    [
     "C86",
     "x_vars[Widget87]"
    ],
    [
     "C87",
     "x_vars[Widget88]"
    ],
    [
     "C88",
     "x_vars[Widget89]"
    ],
    [
     "C89",
     "x_vars[Widget90]"
    ],
    [
     "C90",
     "x_vars[Widget91]"
    ],
    [
     "C91",
     "x_vars[Widget92]"
    ],
    [
     "C92",
     "x_vars[Widget93]"
    ],
    [
     "C93",
     "x_vars[Widget94]"
    ],
    [
     "C94",
     "x_vars[Widget95]"
    ],
    [
     "C95",
     "x_vars[Widget96]"
    ],
    [
     "C96",
     "x_vars[Widget97]"
    ],
    [
     "C97",
     "x_vars[Widget98]"
    ],
    [
     "C98",
     "x_vars[Widget99]"
    ],
    [
     "C99",
     "x_vars[Widget100]"
    ],
    [
     "C100",
     "x_vars[Widget101]"
    ],
    [
     "C101",
     "x_vars[Widget102]"
    ],
    [
     "C102",
     "x_vars[Widget103]"
    ],
    [
     "C103",
     "x_vars[Widget104]"
    ],
    [
     "C104",
     "x_vars[Widget105]"
    ],
    [
     "C105",
     "x_vars[Widget106]"
    ],
    [
     "C106",
     "x_vars[Widget107]"
    ],
    [
     "C107",
     "x_vars[Widget108]"
    ],
    [
     "C108",
     "x_vars[Widget109]"
    ],
    [
     "C109",
     "x_vars[Widget110]"
    ],
    [
     "C110",
     "x_vars[Widget111]"
    ],
    [
     "C111",
     "x_vars[Widget112]"
    ],
    [
     "C112",
     "x_vars[Widget113]"
    ],
    [
     "C113",
     "x_vars[Widget114]"
    ],
    [
     "C114",
     "x_vars[Widget115]"
    ],
    [
     "C115",
     "x_vars[Widget116]"
    ],
    [
     "C116",
     "x_vars[Widget117]"
    ],
    [
     "C117",
     "x_vars[Widget118]"
    ],
    [
     "C118",
     "x_vars[Widget119]"
    ],
    [
     "C119",
     "x_vars[Widget120]"
    ],
    [
     "C120",
     "x_vars[Widget121]"
    ],
    [
     "C121",
     "x_vars[Widget122]"
    ],
    [
     "C122",
     "x_vars[Widget123]"
    ],
    [
     "C123",
     "x_vars[Widget124]"
    ],
    [
     "C124",
     "x_vars[Widget125]"
    ],
    [
     "C125",
     "x_vars[Widget126]"
    ],
    [
     "C126",
     "x_vars[Widget127]"
    ],
    [
     "C127",
     "x_vars[Widget128]"
    ],
    [
     "C128",
     "x_vars[Widget129]"
    ],
    [
     "C129",
     "x_vars[Widget130]"
    ],
    [
     "C130",
     "x_vars[Widget131]"
    ],
    [
     "C131",
     "x_vars[Widget132]"
    ],
    [
     "C132",
     "x_vars[Widget133]"
    ],
    [
     "C133",
     "x_vars[Widget134]"
    ],
    [
     "C134",
     "x_vars[Widget135]"
    ],
    [
     "C135",
     "x_vars[Widget136]"
    ],
    [
     "C136",
     "x_vars[Widget137]"
    ],
    [
     "C137",
     "x_vars[Widget138]"
    ],
    [
     "C138",
     "x_vars[Widget139]"
    ],
    [
     "C139",
     "x_vars[Widget140]"
    ],
    [
     "C140",
     "x_vars[Widget141]"
    ],
    [
     "s",
     "s"
    ],
    [
     "d",
     "d"
    ]
   ],
   "primary_mapping": {
    "x[1]": "x_vars[Widget1]",
    "x[2]": "x_vars[Widget2]",
    "x[3]": "x_vars[Widget3]",
    "x[4]": "x_vars[Widget4]",
    "x[5]": "x_vars[Widget5]",
    "x[6]": "x_vars[Widget6]",
    "x[7]": "x_vars[Widget7]",
    "x[8]": "x_vars[Widget8]",
    "x[9]": "x_vars[Widget9]",
    "x[10]": "x_vars[Widget10]",
    "x[11]": "x_vars[Widget11]",
    "x[12]": "x_vars[Widget12]",
    "x[13]": "x_vars[Widget13]",
    "x[14]": "x_vars[Widget14]",
    "x[15]": "x_vars[Widget15]",
    "x[16]": "x_vars[Widget16]",
    "x[17]": "x_vars[Widget17]",
    "x[18]": "x_vars[Widget18]",
    "x[19]": "x_vars[Widget19]",
    "x[20]": "x_vars[Widget20]",
    "x[21]": "x_vars[Widget21]",
    "x[22]": "x_vars[Widget22]",
    "x[23]": "x_vars[Widget23]",
    "x[24]": "x_vars[Widget24]",
    "x[25]": "x_vars[Widget25]",
    "x[26]": "x_vars[Widget26]",
    "x[27]": "x_vars[Widget27]",
    "x[28]": "x_vars[Widget28]",
    "x[29]": "x_vars[Widget29]",
    "x[30]": "x_vars[Widget30]",
    "x[31]": "x_vars[Widget31]",
    "x[32]": "x_vars[Widget32]",
    "x[33]": "x_vars[Widget33]",
    "x[34]": "x_vars[Widget34]",
    "x[35]": "x_vars[Widget35]",
    "x[36]": "x_vars[Widget36]",
    "x[37]": "x_vars[Widget37]",
    "x[38]": "x_vars[Widget38]",
    "x[39]": "x_vars[Widget39]",
    "x[40]": "x_vars[Widget40]",
    "x[41]": "x_vars[Widget41]",
    "x[42]": "x_vars[Widget42]",
    "x[43]": "x_vars[Widget43]",
    "x[44]": "x_vars[Widget44]",
    "x[45]": "x_vars[Widget45]",
    "x[46]": "x_vars[Widget46]",
    "x[47]": "x_vars[Widget47]",
    "x[48]": "x_vars[Widget48]",
    "x[49]": "x_vars[Widget49]",
    "x[50]": "x_vars[Widget50]",
    "x[51]": "x_vars[Widget51]",
    "x[52]": "x_vars[Widget52]",
    "x[53]": "x_vars[Widget53]",
    "x[54]": "x_vars[Widget54]",
    "x[55]": "x_vars[Widget55]",
    "x[56]": "x_vars[Widget56]",
    "x[57]": "x_vars[Widget57]",
    "x[58]": "x_vars[Widget58]",
    "x[59]": "x_vars[Widget59]",
    "x[60]": "x_vars[Widget60]",
    "x[61]": "x_vars[Widget61]",
    "x[62]": "x_vars[Widget62]",
    "x[63]": "x_vars[Widget63]",
    "x[64]": "x_vars[Widget64]",
    "x[65]": "x_vars[Widget65]",
    "x[66]": "x_vars[Widget66]",
    "x[67]": "x_vars[Widget67]",
    "x[68]": "x_vars[Widget68]",
    "x[69]": "x_vars[Widget69]",
    "x[70]": "x_vars[Widget70]",
    "x[71]": "x_vars[Widget71]",
    "x[72]": "x_vars[Widget72]",
    "x[73]": "x_vars[Widget73]",
    "x[74]": "x_vars[Widget74]",
    "x[75]": "x_vars[Widget75]",
    "x[76]": "x_vars[Widget76]",
    "x[77]": "x_vars[Widget77]",
    "x[78]": "x_vars[Widget78]",
    "x[79]": "x_vars[Widget79]",
    "x[80]": "x_vars[Widget80]",
    "x[81]": "x_vars[Widget81]",
    "x[82]": "x_vars[Widget82]",
    "x[83]": "x_vars[Widget83]",
    "x[84]": "x_vars[Widget84]",
    "x[85]": "x_vars[Widget85]",
    "x[86]": "x_vars[Widget86]",
    "x[87]": "x_vars[Widget87]",
    "x[88]": "x_vars[Widget88]",
    "x[89]": "x_vars[Widget89]",
    "x[90]": "x_vars[Widget90]",
    "x[91]": "x_vars[Widget91]",
    "x[92]": "x_vars[Widget92]",
    "x[93]": "x_vars[Widget93]",
    "x[94]": "x_vars[Widget94]",
    "x[95]": "x_vars[Widget95]",
    "x[96]": "x_vars[Widget96]",
    "x[97]": "x_vars[Widget97]",
    "x[98]": "x_vars[Widget98]",
    "x[99]": "x_vars[Widget99]",
    "x[100]": "x_vars[Widget100]",
    "x[101]": "x_vars[Widget101]",
    "x[102]": "x_vars[Widget102]",
    "x[103]": "x_vars[Widget103]",
    "x[104]": "x_vars[Widget104]",
    "x[105]": "x_vars[Widget105]",
    "x[106]": "x_vars[Widget106]",
    "x[107]": "x_vars[Widget107]",
    "x[108]": "x_vars[Widget108]",
    "x[109]": "x_vars[Widget109]",
    "x[110]": "x_vars[Widget110]",
    "x[111]": "x_vars[Widget111]",
    "x[112]": "x_vars[Widget112]",
    "x[113]": "x_vars[Widget113]",
    "x[114]": "x_vars[Widget114]",
    "x[115]": "x_vars[Widget115]",
    "x[116]": "x_vars[Widget116]",
    "x[117]": "x_vars[Widget117]",
    "x[118]": "x_vars[Widget118]",
    "x[119]": "x_vars[Widget119]",
    "x[120]": "x_vars[Widget120]",
    "x[121]": "x_vars[Widget121]",
    "x[122]": "x_vars[Widget122]",
    "x[123]": "x_vars[Widget123]",
    "x[124]": "x_vars[Widget124]",
    "x[125]": "x_vars[Widget125]",
    "x[126]": "x_vars[Widget126]",
    "x[127]": "x_vars[Widget127]",
    "x[128]": "x_vars[Widget128]",
    "x[129]": "x_vars[Widget129]",
    "x[130]": "x_vars[Widget130]",
    "x[131]": "x_vars[Widget131]",
    "x[132]": "x_vars[Widget132]",
    "x[133]": "x_vars[Widget133]",
    "x[134]": "x_vars[Widget134]",
    "x[135]": "x_vars[Widget135]",
    "x[136]": "x_vars[Widget136]",
    "x[137]": "x_vars[Widget137]",
    "x[138]": "x_vars[Widget138]",
    "x[139]": "x_vars[Widget139]",
    "x[140]": "x_vars[Widget140]",
    "x[141]": "x_vars[Widget141]",
    "s": "s"
   },
   "mapping_basis": "frozen_automatic_name_axis_projection"
  },
  "OR-096": {
   "source_code_sha256": "fd1532b5d560ea11bcb2140e7b416ef0f9819d6a4b9fb55c40476f1bbfdac53d",
   "original_solution_sha256": "071e24647a7f09e4c073d64fb8dcc91978637bc82adf1fa0bda03fb051f444ac",
   "bindings": [
    [
     "C0",
     "x_vars[I,N01,White]"
    ],
    [
     "C1",
     "x_vars[I,N01,NonWhite]"
    ],
    [
     "C2",
     "x_vars[I,N02,White]"
    ],
    [
     "C3",
     "x_vars[I,N02,NonWhite]"
    ],
    [
     "C4",
     "x_vars[I,N03,White]"
    ],
    [
     "C5",
     "x_vars[I,N03,NonWhite]"
    ],
    [
     "C6",
     "x_vars[I,N04,White]"
    ],
    [
     "C7",
     "x_vars[I,N04,NonWhite]"
    ],
    [
     "C8",
     "x_vars[I,N05,White]"
    ],
    [
     "C9",
     "x_vars[I,N05,NonWhite]"
    ],
    [
     "C10",
     "x_vars[I,N06,White]"
    ],
    [
     "C11",
     "x_vars[I,N06,NonWhite]"
    ],
    [
     "C12",
     "x_vars[I,N07,White]"
    ],
    [
     "C13",
     "x_vars[I,N07,NonWhite]"
    ],
    [
     "C14",
     "x_vars[I,N08,White]"
    ],
    [
     "C15",
     "x_vars[I,N08,NonWhite]"
    ],
    [
     "C16",
     "x_vars[I,N09,White]"
    ],
    [
     "C17",
     "x_vars[I,N09,NonWhite]"
    ],
    [
     "C18",
     "x_vars[I,N10,White]"
    ],
    [
     "C19",
     "x_vars[I,N10,NonWhite]"
    ],
    [
     "C20",
     "x_vars[I,N11,White]"
    ],
    [
     "C21",
     "x_vars[I,N11,NonWhite]"
    ],
    [
     "C22",
     "x_vars[I,N12,White]"
    ],
    [
     "C23",
     "x_vars[I,N12,NonWhite]"
    ],
    [
     "C24",
     "x_vars[I,N13,White]"
    ],
    [
     "C25",
     "x_vars[I,N13,NonWhite]"
    ],
    [
     "C26",
     "x_vars[I,N14,White]"
    ],
    [
     "C27",
     "x_vars[I,N14,NonWhite]"
    ],
    [
     "C28",
     "x_vars[I,N15,White]"
    ],
    [
     "C29",
     "x_vars[I,N15,NonWhite]"
    ],
    [
     "C30",
     "x_vars[I,N16,White]"
    ],
    [
     "C31",
     "x_vars[I,N16,NonWhite]"
    ],
    [
     "C32",
     "x_vars[I,N17,White]"
    ],
    [
     "C33",
     "x_vars[I,N17,NonWhite]"
    ],
    [
     "C34",
     "x_vars[I,N18,White]"
    ],
    [
     "C35",
     "x_vars[I,N18,NonWhite]"
    ],
    [
     "C36",
     "x_vars[I,N19,White]"
    ],
    [
     "C37",
     "x_vars[I,N19,NonWhite]"
    ],
    [
     "C38",
     "x_vars[I,N20,White]"
    ],
    [
     "C39",
     "x_vars[I,N20,NonWhite]"
    ],
    [
     "C40",
     "x_vars[I,N21,White]"
    ],
    [
     "C41",
     "x_vars[I,N21,NonWhite]"
    ],
    [
     "C42",
     "x_vars[I,N22,White]"
    ],
    [
     "C43",
     "x_vars[I,N22,NonWhite]"
    ],
    [
     "C44",
     "x_vars[I,N23,White]"
    ],
    [
     "C45",
     "x_vars[I,N23,NonWhite]"
    ],
    [
     "C46",
     "x_vars[I,N24,White]"
    ],
    [
     "C47",
     "x_vars[I,N24,NonWhite]"
    ],
    [
     "C48",
     "x_vars[I,N25,White]"
    ],
    [
     "C49",
     "x_vars[I,N25,NonWhite]"
    ],
    [
     "C50",
     "x_vars[I,N26,White]"
    ],
    [
     "C51",
     "x_vars[I,N26,NonWhite]"
    ],
    [
     "C52",
     "x_vars[I,N27,White]"
    ],
    [
     "C53",
     "x_vars[I,N27,NonWhite]"
    ],
    [
     "C54",
     "x_vars[I,N28,White]"
    ],
    [
     "C55",
     "x_vars[I,N28,NonWhite]"
    ],
    [
     "C56",
     "x_vars[I,N29,White]"
    ],
    [
     "C57",
     "x_vars[I,N29,NonWhite]"
    ],
    [
     "C58",
     "x_vars[I,N30,White]"
    ],
    [
     "C59",
     "x_vars[I,N30,NonWhite]"
    ],
    [
     "C60",
     "x_vars[I,N31,White]"
    ],
    [
     "C61",
     "x_vars[I,N31,NonWhite]"
    ],
    [
     "C62",
     "x_vars[II,N01,White]"
    ],
    [
     "C63",
     "x_vars[II,N01,NonWhite]"
    ],
    [
     "C64",
     "x_vars[II,N02,White]"
    ],
    [
     "C65",
     "x_vars[II,N02,NonWhite]"
    ],
    [
     "C66",
     "x_vars[II,N03,White]"
    ],
    [
     "C67",
     "x_vars[II,N03,NonWhite]"
    ],
    [
     "C68",
     "x_vars[II,N04,White]"
    ],
    [
     "C69",
     "x_vars[II,N04,NonWhite]"
    ],
    [
     "C70",
     "x_vars[II,N05,White]"
    ],
    [
     "C71",
     "x_vars[II,N05,NonWhite]"
    ],
    [
     "C72",
     "x_vars[II,N06,White]"
    ],
    [
     "C73",
     "x_vars[II,N06,NonWhite]"
    ],
    [
     "C74",
     "x_vars[II,N07,White]"
    ],
    [
     "C75",
     "x_vars[II,N07,NonWhite]"
    ],
    [
     "C76",
     "x_vars[II,N08,White]"
    ],
    [
     "C77",
     "x_vars[II,N08,NonWhite]"
    ],
    [
     "C78",
     "x_vars[II,N09,White]"
    ],
    [
     "C79",
     "x_vars[II,N09,NonWhite]"
    ],
    [
     "C80",
     "x_vars[II,N10,White]"
    ],
    [
     "C81",
     "x_vars[II,N10,NonWhite]"
    ],
    [
     "C82",
     "x_vars[II,N11,White]"
    ],
    [
     "C83",
     "x_vars[II,N11,NonWhite]"
    ],
    [
     "C84",
     "x_vars[II,N12,White]"
    ],
    [
     "C85",
     "x_vars[II,N12,NonWhite]"
    ],
    [
     "C86",
     "x_vars[II,N13,White]"
    ],
    [
     "C87",
     "x_vars[II,N13,NonWhite]"
    ],
    [
     "C88",
     "x_vars[II,N14,White]"
    ],
    [
     "C89",
     "x_vars[II,N14,NonWhite]"
    ],
    [
     "C90",
     "x_vars[II,N15,White]"
    ],
    [
     "C91",
     "x_vars[II,N15,NonWhite]"
    ],
    [
     "C92",
     "x_vars[II,N16,White]"
    ],
    [
     "C93",
     "x_vars[II,N16,NonWhite]"
    ],
    [
     "C94",
     "x_vars[II,N17,White]"
    ],
    [
     "C95",
     "x_vars[II,N17,NonWhite]"
    ],
    [
     "C96",
     "x_vars[II,N18,White]"
    ],
    [
     "C97",
     "x_vars[II,N18,NonWhite]"
    ],
    [
     "C98",
     "x_vars[II,N19,White]"
    ],
    [
     "C99",
     "x_vars[II,N19,NonWhite]"
    ],
    [
     "C100",
     "x_vars[II,N20,White]"
    ],
    [
     "C101",
     "x_vars[II,N20,NonWhite]"
    ],
    [
     "C102",
     "x_vars[II,N21,White]"
    ],
    [
     "C103",
     "x_vars[II,N21,NonWhite]"
    ],
    [
     "C104",
     "x_vars[II,N22,White]"
    ],
    [
     "C105",
     "x_vars[II,N22,NonWhite]"
    ],
    [
     "C106",
     "x_vars[II,N23,White]"
    ],
    [
     "C107",
     "x_vars[II,N23,NonWhite]"
    ],
    [
     "C108",
     "x_vars[II,N24,White]"
    ],
    [
     "C109",
     "x_vars[II,N24,NonWhite]"
    ],
    [
     "C110",
     "x_vars[II,N25,White]"
    ],
    [
     "C111",
     "x_vars[II,N25,NonWhite]"
    ],
    [
     "C112",
     "x_vars[II,N26,White]"
    ],
    [
     "C113",
     "x_vars[II,N26,NonWhite]"
    ],
    [
     "C114",
     "x_vars[II,N27,White]"
    ],
    [
     "C115",
     "x_vars[II,N27,NonWhite]"
    ],
    [
     "C116",
     "x_vars[II,N28,White]"
    ],
    [
     "C117",
     "x_vars[II,N28,NonWhite]"
    ],
    [
     "C118",
     "x_vars[II,N29,White]"
    ],
    [
     "C119",
     "x_vars[II,N29,NonWhite]"
    ],
    [
     "C120",
     "x_vars[II,N30,White]"
    ],
    [
     "C121",
     "x_vars[II,N30,NonWhite]"
    ],
    [
     "C122",
     "x_vars[II,N31,White]"
    ],
    [
     "C123",
     "x_vars[II,N31,NonWhite]"
    ]
   ],
   "primary_mapping": {
    "x_w_I_N01": "x_vars[I,N01,White]",
    "x_n_I_N01": "x_vars[I,N01,NonWhite]",
    "x_w_I_N02": "x_vars[I,N02,White]",
    "x_n_I_N02": "x_vars[I,N02,NonWhite]",
    "x_w_I_N03": "x_vars[I,N03,White]",
    "x_n_I_N03": "x_vars[I,N03,NonWhite]",
    "x_w_I_N04": "x_vars[I,N04,White]",
    "x_n_I_N04": "x_vars[I,N04,NonWhite]",
    "x_w_I_N05": "x_vars[I,N05,White]",
    "x_n_I_N05": "x_vars[I,N05,NonWhite]",
    "x_w_I_N06": "x_vars[I,N06,White]",
    "x_n_I_N06": "x_vars[I,N06,NonWhite]",
    "x_w_I_N07": "x_vars[I,N07,White]",
    "x_n_I_N07": "x_vars[I,N07,NonWhite]",
    "x_w_I_N08": "x_vars[I,N08,White]",
    "x_n_I_N08": "x_vars[I,N08,NonWhite]",
    "x_w_I_N09": "x_vars[I,N09,White]",
    "x_n_I_N09": "x_vars[I,N09,NonWhite]",
    "x_w_I_N10": "x_vars[I,N10,White]",
    "x_n_I_N10": "x_vars[I,N10,NonWhite]",
    "x_w_I_N11": "x_vars[I,N11,White]",
    "x_n_I_N11": "x_vars[I,N11,NonWhite]",
    "x_w_I_N12": "x_vars[I,N12,White]",
    "x_n_I_N12": "x_vars[I,N12,NonWhite]",
    "x_w_I_N13": "x_vars[I,N13,White]",
    "x_n_I_N13": "x_vars[I,N13,NonWhite]",
    "x_w_I_N14": "x_vars[I,N14,White]",
    "x_n_I_N14": "x_vars[I,N14,NonWhite]",
    "x_w_I_N15": "x_vars[I,N15,White]",
    "x_n_I_N15": "x_vars[I,N15,NonWhite]",
    "x_w_I_N16": "x_vars[I,N16,White]",
    "x_n_I_N16": "x_vars[I,N16,NonWhite]",
    "x_w_I_N17": "x_vars[I,N17,White]",
    "x_n_I_N17": "x_vars[I,N17,NonWhite]",
    "x_w_I_N18": "x_vars[I,N18,White]",
    "x_n_I_N18": "x_vars[I,N18,NonWhite]",
    "x_w_I_N19": "x_vars[I,N19,White]",
    "x_n_I_N19": "x_vars[I,N19,NonWhite]",
    "x_w_I_N20": "x_vars[I,N20,White]",
    "x_n_I_N20": "x_vars[I,N20,NonWhite]",
    "x_w_I_N21": "x_vars[I,N21,White]",
    "x_n_I_N21": "x_vars[I,N21,NonWhite]",
    "x_w_I_N22": "x_vars[I,N22,White]",
    "x_n_I_N22": "x_vars[I,N22,NonWhite]",
    "x_w_I_N23": "x_vars[I,N23,White]",
    "x_n_I_N23": "x_vars[I,N23,NonWhite]",
    "x_w_I_N24": "x_vars[I,N24,White]",
    "x_n_I_N24": "x_vars[I,N24,NonWhite]",
    "x_w_I_N25": "x_vars[I,N25,White]",
    "x_n_I_N25": "x_vars[I,N25,NonWhite]",
    "x_w_I_N26": "x_vars[I,N26,White]",
    "x_n_I_N26": "x_vars[I,N26,NonWhite]",
    "x_w_I_N27": "x_vars[I,N27,White]",
    "x_n_I_N27": "x_vars[I,N27,NonWhite]",
    "x_w_I_N28": "x_vars[I,N28,White]",
    "x_n_I_N28": "x_vars[I,N28,NonWhite]",
    "x_w_I_N29": "x_vars[I,N29,White]",
    "x_n_I_N29": "x_vars[I,N29,NonWhite]",
    "x_w_I_N30": "x_vars[I,N30,White]",
    "x_n_I_N30": "x_vars[I,N30,NonWhite]",
    "x_w_I_N31": "x_vars[I,N31,White]",
    "x_n_I_N31": "x_vars[I,N31,NonWhite]",
    "x_w_II_N01": "x_vars[II,N01,White]",
    "x_n_II_N01": "x_vars[II,N01,NonWhite]",
    "x_w_II_N02": "x_vars[II,N02,White]",
    "x_n_II_N02": "x_vars[II,N02,NonWhite]",
    "x_w_II_N03": "x_vars[II,N03,White]",
    "x_n_II_N03": "x_vars[II,N03,NonWhite]",
    "x_w_II_N04": "x_vars[II,N04,White]",
    "x_n_II_N04": "x_vars[II,N04,NonWhite]",
    "x_w_II_N05": "x_vars[II,N05,White]",
    "x_n_II_N05": "x_vars[II,N05,NonWhite]",
    "x_w_II_N06": "x_vars[II,N06,White]",
    "x_n_II_N06": "x_vars[II,N06,NonWhite]",
    "x_w_II_N07": "x_vars[II,N07,White]",
    "x_n_II_N07": "x_vars[II,N07,NonWhite]",
    "x_w_II_N08": "x_vars[II,N08,White]",
    "x_n_II_N08": "x_vars[II,N08,NonWhite]",
    "x_w_II_N09": "x_vars[II,N09,White]",
    "x_n_II_N09": "x_vars[II,N09,NonWhite]",
    "x_w_II_N10": "x_vars[II,N10,White]",
    "x_n_II_N10": "x_vars[II,N10,NonWhite]",
    "x_w_II_N11": "x_vars[II,N11,White]",
    "x_n_II_N11": "x_vars[II,N11,NonWhite]",
    "x_w_II_N12": "x_vars[II,N12,White]",
    "x_n_II_N12": "x_vars[II,N12,NonWhite]",
    "x_w_II_N13": "x_vars[II,N13,White]",
    "x_n_II_N13": "x_vars[II,N13,NonWhite]",
    "x_w_II_N14": "x_vars[II,N14,White]",
    "x_n_II_N14": "x_vars[II,N14,NonWhite]",
    "x_w_II_N15": "x_vars[II,N15,White]",
    "x_n_II_N15": "x_vars[II,N15,NonWhite]",
    "x_w_II_N16": "x_vars[II,N16,White]",
    "x_n_II_N16": "x_vars[II,N16,NonWhite]",
    "x_w_II_N17": "x_vars[II,N17,White]",
    "x_n_II_N17": "x_vars[II,N17,NonWhite]",
    "x_w_II_N18": "x_vars[II,N18,White]",
    "x_n_II_N18": "x_vars[II,N18,NonWhite]",
    "x_w_II_N19": "x_vars[II,N19,White]",
    "x_n_II_N19": "x_vars[II,N19,NonWhite]",
    "x_w_II_N20": "x_vars[II,N20,White]",
    "x_n_II_N20": "x_vars[II,N20,NonWhite]",
    "x_w_II_N21": "x_vars[II,N21,White]",
    "x_n_II_N21": "x_vars[II,N21,NonWhite]",
    "x_w_II_N22": "x_vars[II,N22,White]",
    "x_n_II_N22": "x_vars[II,N22,NonWhite]",
    "x_w_II_N23": "x_vars[II,N23,White]",
    "x_n_II_N23": "x_vars[II,N23,NonWhite]",
    "x_w_II_N24": "x_vars[II,N24,White]",
    "x_n_II_N24": "x_vars[II,N24,NonWhite]",
    "x_w_II_N25": "x_vars[II,N25,White]",
    "x_n_II_N25": "x_vars[II,N25,NonWhite]",
    "x_w_II_N26": "x_vars[II,N26,White]",
    "x_n_II_N26": "x_vars[II,N26,NonWhite]",
    "x_w_II_N27": "x_vars[II,N27,White]",
    "x_n_II_N27": "x_vars[II,N27,NonWhite]",
    "x_w_II_N28": "x_vars[II,N28,White]",
    "x_n_II_N28": "x_vars[II,N28,NonWhite]",
    "x_w_II_N29": "x_vars[II,N29,White]",
    "x_n_II_N29": "x_vars[II,N29,NonWhite]",
    "x_w_II_N30": "x_vars[II,N30,White]",
    "x_n_II_N30": "x_vars[II,N30,NonWhite]",
    "x_w_II_N31": "x_vars[II,N31,White]",
    "x_n_II_N31": "x_vars[II,N31,NonWhite]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-097": {
   "source_code_sha256": "a7321fdc44ae8d836544aaba0f07195337023c9ccf67b36d7d3089e28a1fe966",
   "original_solution_sha256": "57685888572686b98e1ac16e74ac60d1bd481d5d715e39ce7c16ced5a9ed2e87",
   "bindings": [
    [
     "C0",
     "x_vars[Opt_1]"
    ],
    [
     "C1",
     "x_vars[Opt_2]"
    ],
    [
     "C2",
     "x_vars[Opt_3]"
    ],
    [
     "C3",
     "x_vars[Opt_4]"
    ],
    [
     "C4",
     "x_vars[Opt_5]"
    ],
    [
     "C5",
     "x_vars[Opt_6]"
    ],
    [
     "C6",
     "x_vars[Opt_7]"
    ],
    [
     "C7",
     "x_vars[Opt_8]"
    ],
    [
     "C8",
     "x_vars[Opt_9]"
    ],
    [
     "C9",
     "x_vars[Opt_10]"
    ],
    [
     "C10",
     "x_vars[Opt_11]"
    ],
    [
     "C11",
     "x_vars[Opt_12]"
    ],
    [
     "C12",
     "x_vars[Opt_13]"
    ],
    [
     "C13",
     "x_vars[Opt_14]"
    ],
    [
     "C14",
     "x_vars[Opt_15]"
    ],
    [
     "C15",
     "x_vars[Opt_16]"
    ],
    [
     "C16",
     "x_vars[Opt_17]"
    ],
    [
     "C17",
     "x_vars[Opt_18]"
    ],
    [
     "C18",
     "x_vars[Opt_19]"
    ],
    [
     "C19",
     "x_vars[Opt_20]"
    ],
    [
     "C20",
     "x_vars[Opt_21]"
    ],
    [
     "C21",
     "x_vars[Opt_22]"
    ],
    [
     "C22",
     "x_vars[Opt_23]"
    ],
    [
     "C23",
     "x_vars[Opt_24]"
    ],
    [
     "C24",
     "x_vars[Opt_25]"
    ],
    [
     "C25",
     "x_vars[Opt_26]"
    ],
    [
     "C26",
     "x_vars[Opt_27]"
    ],
    [
     "C27",
     "x_vars[Opt_28]"
    ],
    [
     "C28",
     "x_vars[Opt_29]"
    ],
    [
     "C29",
     "x_vars[Opt_30]"
    ],
    [
     "C30",
     "x_vars[Opt_31]"
    ],
    [
     "C31",
     "x_vars[Opt_32]"
    ],
    [
     "C32",
     "x_vars[Opt_33]"
    ],
    [
     "C33",
     "x_vars[Opt_34]"
    ],
    [
     "C34",
     "x_vars[Opt_35]"
    ],
    [
     "C35",
     "x_vars[Opt_36]"
    ],
    [
     "C36",
     "x_vars[Opt_37]"
    ],
    [
     "C37",
     "x_vars[Opt_38]"
    ],
    [
     "C38",
     "x_vars[Opt_39]"
    ],
    [
     "C39",
     "x_vars[Opt_40]"
    ],
    [
     "C40",
     "x_vars[Opt_41]"
    ],
    [
     "C41",
     "x_vars[Opt_42]"
    ],
    [
     "C42",
     "x_vars[Opt_43]"
    ],
    [
     "C43",
     "x_vars[Opt_44]"
    ],
    [
     "C44",
     "x_vars[Opt_45]"
    ],
    [
     "C45",
     "x_vars[Opt_46]"
    ],
    [
     "C46",
     "x_vars[Opt_47]"
    ],
    [
     "C47",
     "x_vars[Opt_48]"
    ],
    [
     "C48",
     "x_vars[Opt_49]"
    ],
    [
     "C49",
     "x_vars[Opt_50]"
    ],
    [
     "C50",
     "x_vars[Opt_51]"
    ],
    [
     "C51",
     "x_vars[Opt_52]"
    ],
    [
     "C52",
     "x_vars[Opt_53]"
    ],
    [
     "C53",
     "x_vars[Opt_54]"
    ],
    [
     "C54",
     "x_vars[Opt_55]"
    ],
    [
     "C55",
     "x_vars[Opt_56]"
    ],
    [
     "C56",
     "x_vars[Opt_57]"
    ],
    [
     "C57",
     "x_vars[Opt_58]"
    ],
    [
     "C58",
     "x_vars[Opt_59]"
    ],
    [
     "C59",
     "x_vars[Opt_60]"
    ],
    [
     "C60",
     "x_vars[Opt_61]"
    ],
    [
     "C61",
     "x_vars[Opt_62]"
    ],
    [
     "C62",
     "x_vars[Opt_63]"
    ],
    [
     "C63",
     "x_vars[Opt_64]"
    ],
    [
     "C64",
     "x_vars[Opt_65]"
    ],
    [
     "C65",
     "x_vars[Opt_66]"
    ],
    [
     "C66",
     "x_vars[Opt_67]"
    ],
    [
     "C67",
     "x_vars[Opt_68]"
    ],
    [
     "C68",
     "x_vars[Opt_69]"
    ],
    [
     "C69",
     "x_vars[Opt_70]"
    ],
    [
     "C70",
     "x_vars[Opt_71]"
    ],
    [
     "C71",
     "x_vars[Opt_72]"
    ],
    [
     "C72",
     "x_vars[Opt_73]"
    ],
    [
     "C73",
     "x_vars[Opt_74]"
    ],
    [
     "C74",
     "x_vars[Opt_75]"
    ],
    [
     "C75",
     "x_vars[Opt_76]"
    ],
    [
     "C76",
     "x_vars[Opt_77]"
    ],
    [
     "C77",
     "x_vars[Opt_78]"
    ],
    [
     "C78",
     "x_vars[Opt_79]"
    ],
    [
     "C79",
     "x_vars[Opt_80]"
    ],
    [
     "C80",
     "x_vars[Opt_81]"
    ],
    [
     "C81",
     "x_vars[Opt_82]"
    ],
    [
     "C82",
     "x_vars[Opt_83]"
    ],
    [
     "C83",
     "x_vars[Opt_84]"
    ],
    [
     "C84",
     "x_vars[Opt_85]"
    ],
    [
     "C85",
     "x_vars[Opt_86]"
    ],
    [
     "C86",
     "x_vars[Opt_87]"
    ],
    [
     "C87",
     "x_vars[Opt_88]"
    ],
    [
     "C88",
     "x_vars[Opt_89]"
    ],
    [
     "C89",
     "x_vars[Opt_90]"
    ],
    [
     "C90",
     "x_vars[Opt_91]"
    ],
    [
     "C91",
     "x_vars[Opt_92]"
    ],
    [
     "C92",
     "x_vars[Opt_93]"
    ],
    [
     "C93",
     "x_vars[Opt_94]"
    ],
    [
     "C94",
     "x_vars[Opt_95]"
    ],
    [
     "C95",
     "x_vars[Opt_96]"
    ],
    [
     "C96",
     "x_vars[Opt_97]"
    ],
    [
     "C97",
     "x_vars[Opt_98]"
    ],
    [
     "C98",
     "x_vars[Opt_99]"
    ],
    [
     "C99",
     "x_vars[Opt_100]"
    ],
    [
     "C100",
     "x_vars[Opt_101]"
    ],
    [
     "C101",
     "x_vars[Opt_102]"
    ],
    [
     "C102",
     "x_vars[Opt_103]"
    ],
    [
     "C103",
     "x_vars[Opt_104]"
    ],
    [
     "C104",
     "x_vars[Opt_105]"
    ],
    [
     "C105",
     "x_vars[Opt_106]"
    ],
    [
     "C106",
     "x_vars[Opt_107]"
    ],
    [
     "C107",
     "x_vars[Opt_108]"
    ],
    [
     "C108",
     "x_vars[Opt_109]"
    ],
    [
     "C109",
     "x_vars[Opt_110]"
    ],
    [
     "C110",
     "x_vars[Opt_111]"
    ],
    [
     "C111",
     "x_vars[Opt_112]"
    ],
    [
     "C112",
     "x_vars[Opt_113]"
    ],
    [
     "C113",
     "x_vars[Opt_114]"
    ],
    [
     "C114",
     "x_vars[Opt_115]"
    ],
    [
     "C115",
     "x_vars[Opt_116]"
    ],
    [
     "C116",
     "x_vars[Opt_117]"
    ],
    [
     "C117",
     "x_vars[Opt_118]"
    ],
    [
     "C118",
     "x_vars[Opt_119]"
    ],
    [
     "C119",
     "x_vars[Opt_120]"
    ],
    [
     "C120",
     "y_vars[Opt_1]"
    ],
    [
     "C121",
     "y_vars[Opt_2]"
    ],
    [
     "C122",
     "y_vars[Opt_3]"
    ],
    [
     "C123",
     "y_vars[Opt_4]"
    ],
    [
     "C124",
     "y_vars[Opt_5]"
    ],
    [
     "C125",
     "y_vars[Opt_6]"
    ],
    [
     "C126",
     "y_vars[Opt_7]"
    ],
    [
     "C127",
     "y_vars[Opt_8]"
    ],
    [
     "C128",
     "y_vars[Opt_9]"
    ],
    [
     "C129",
     "y_vars[Opt_10]"
    ],
    [
     "C130",
     "y_vars[Opt_11]"
    ],
    [
     "C131",
     "y_vars[Opt_12]"
    ],
    [
     "C132",
     "y_vars[Opt_13]"
    ],
    [
     "C133",
     "y_vars[Opt_14]"
    ],
    [
     "C134",
     "y_vars[Opt_15]"
    ],
    [
     "C135",
     "y_vars[Opt_16]"
    ],
    [
     "C136",
     "y_vars[Opt_17]"
    ],
    [
     "C137",
     "y_vars[Opt_18]"
    ],
    [
     "C138",
     "y_vars[Opt_19]"
    ],
    [
     "C139",
     "y_vars[Opt_20]"
    ],
    [
     "C140",
     "y_vars[Opt_21]"
    ],
    [
     "C141",
     "y_vars[Opt_22]"
    ],
    [
     "C142",
     "y_vars[Opt_23]"
    ],
    [
     "C143",
     "y_vars[Opt_24]"
    ],
    [
     "C144",
     "y_vars[Opt_25]"
    ],
    [
     "C145",
     "y_vars[Opt_26]"
    ],
    [
     "C146",
     "y_vars[Opt_27]"
    ],
    [
     "C147",
     "y_vars[Opt_28]"
    ],
    [
     "C148",
     "y_vars[Opt_29]"
    ],
    [
     "C149",
     "y_vars[Opt_30]"
    ],
    [
     "C150",
     "y_vars[Opt_31]"
    ],
    [
     "C151",
     "y_vars[Opt_32]"
    ],
    [
     "C152",
     "y_vars[Opt_33]"
    ],
    [
     "C153",
     "y_vars[Opt_34]"
    ],
    [
     "C154",
     "y_vars[Opt_35]"
    ],
    [
     "C155",
     "y_vars[Opt_36]"
    ],
    [
     "C156",
     "y_vars[Opt_37]"
    ],
    [
     "C157",
     "y_vars[Opt_38]"
    ],
    [
     "C158",
     "y_vars[Opt_39]"
    ],
    [
     "C159",
     "y_vars[Opt_40]"
    ],
    [
     "C160",
     "y_vars[Opt_41]"
    ],
    [
     "C161",
     "y_vars[Opt_42]"
    ],
    [
     "C162",
     "y_vars[Opt_43]"
    ],
    [
     "C163",
     "y_vars[Opt_44]"
    ],
    [
     "C164",
     "y_vars[Opt_45]"
    ],
    [
     "C165",
     "y_vars[Opt_46]"
    ],
    [
     "C166",
     "y_vars[Opt_47]"
    ],
    [
     "C167",
     "y_vars[Opt_48]"
    ],
    [
     "C168",
     "y_vars[Opt_49]"
    ],
    [
     "C169",
     "y_vars[Opt_50]"
    ],
    [
     "C170",
     "y_vars[Opt_51]"
    ],
    [
     "C171",
     "y_vars[Opt_52]"
    ],
    [
     "C172",
     "y_vars[Opt_53]"
    ],
    [
     "C173",
     "y_vars[Opt_54]"
    ],
    [
     "C174",
     "y_vars[Opt_55]"
    ],
    [
     "C175",
     "y_vars[Opt_56]"
    ],
    [
     "C176",
     "y_vars[Opt_57]"
    ],
    [
     "C177",
     "y_vars[Opt_58]"
    ],
    [
     "C178",
     "y_vars[Opt_59]"
    ],
    [
     "C179",
     "y_vars[Opt_60]"
    ],
    [
     "C180",
     "y_vars[Opt_61]"
    ],
    [
     "C181",
     "y_vars[Opt_62]"
    ],
    [
     "C182",
     "y_vars[Opt_63]"
    ],
    [
     "C183",
     "y_vars[Opt_64]"
    ],
    [
     "C184",
     "y_vars[Opt_65]"
    ],
    [
     "C185",
     "y_vars[Opt_66]"
    ],
    [
     "C186",
     "y_vars[Opt_67]"
    ],
    [
     "C187",
     "y_vars[Opt_68]"
    ],
    [
     "C188",
     "y_vars[Opt_69]"
    ],
    [
     "C189",
     "y_vars[Opt_70]"
    ],
    [
     "C190",
     "y_vars[Opt_71]"
    ],
    [
     "C191",
     "y_vars[Opt_72]"
    ],
    [
     "C192",
     "y_vars[Opt_73]"
    ],
    [
     "C193",
     "y_vars[Opt_74]"
    ],
    [
     "C194",
     "y_vars[Opt_75]"
    ],
    [
     "C195",
     "y_vars[Opt_76]"
    ],
    [
     "C196",
     "y_vars[Opt_77]"
    ],
    [
     "C197",
     "y_vars[Opt_78]"
    ],
    [
     "C198",
     "y_vars[Opt_79]"
    ],
    [
     "C199",
     "y_vars[Opt_80]"
    ],
    [
     "C200",
     "y_vars[Opt_81]"
    ],
    [
     "C201",
     "y_vars[Opt_82]"
    ],
    [
     "C202",
     "y_vars[Opt_83]"
    ],
    [
     "C203",
     "y_vars[Opt_84]"
    ],
    [
     "C204",
     "y_vars[Opt_85]"
    ],
    [
     "C205",
     "y_vars[Opt_86]"
    ],
    [
     "C206",
     "y_vars[Opt_87]"
    ],
    [
     "C207",
     "y_vars[Opt_88]"
    ],
    [
     "C208",
     "y_vars[Opt_89]"
    ],
    [
     "C209",
     "y_vars[Opt_90]"
    ],
    [
     "C210",
     "y_vars[Opt_91]"
    ],
    [
     "C211",
     "y_vars[Opt_92]"
    ],
    [
     "C212",
     "y_vars[Opt_93]"
    ],
    [
     "C213",
     "y_vars[Opt_94]"
    ],
    [
     "C214",
     "y_vars[Opt_95]"
    ],
    [
     "C215",
     "y_vars[Opt_96]"
    ],
    [
     "C216",
     "y_vars[Opt_97]"
    ],
    [
     "C217",
     "y_vars[Opt_98]"
    ],
    [
     "C218",
     "y_vars[Opt_99]"
    ],
    [
     "C219",
     "y_vars[Opt_100]"
    ],
    [
     "C220",
     "y_vars[Opt_101]"
    ],
    [
     "C221",
     "y_vars[Opt_102]"
    ],
    [
     "C222",
     "y_vars[Opt_103]"
    ],
    [
     "C223",
     "y_vars[Opt_104]"
    ],
    [
     "C224",
     "y_vars[Opt_105]"
    ],
    [
     "C225",
     "y_vars[Opt_106]"
    ],
    [
     "C226",
     "y_vars[Opt_107]"
    ],
    [
     "C227",
     "y_vars[Opt_108]"
    ],
    [
     "C228",
     "y_vars[Opt_109]"
    ],
    [
     "C229",
     "y_vars[Opt_110]"
    ],
    [
     "C230",
     "y_vars[Opt_111]"
    ],
    [
     "C231",
     "y_vars[Opt_112]"
    ],
    [
     "C232",
     "y_vars[Opt_113]"
    ],
    [
     "C233",
     "y_vars[Opt_114]"
    ],
    [
     "C234",
     "y_vars[Opt_115]"
    ],
    [
     "C235",
     "y_vars[Opt_116]"
    ],
    [
     "C236",
     "y_vars[Opt_117]"
    ],
    [
     "C237",
     "y_vars[Opt_118]"
    ],
    [
     "C238",
     "y_vars[Opt_119]"
    ],
    [
     "C239",
     "y_vars[Opt_120]"
    ]
   ],
   "primary_mapping": {
    "x[1]": "x_vars[Opt_1]",
    "x[2]": "x_vars[Opt_2]",
    "x[3]": "x_vars[Opt_3]",
    "x[4]": "x_vars[Opt_4]",
    "x[5]": "x_vars[Opt_5]",
    "x[6]": "x_vars[Opt_6]",
    "x[7]": "x_vars[Opt_7]",
    "x[8]": "x_vars[Opt_8]",
    "x[9]": "x_vars[Opt_9]",
    "x[10]": "x_vars[Opt_10]",
    "x[11]": "x_vars[Opt_11]",
    "x[12]": "x_vars[Opt_12]",
    "x[13]": "x_vars[Opt_13]",
    "x[14]": "x_vars[Opt_14]",
    "x[15]": "x_vars[Opt_15]",
    "x[16]": "x_vars[Opt_16]",
    "x[17]": "x_vars[Opt_17]",
    "x[18]": "x_vars[Opt_18]",
    "x[19]": "x_vars[Opt_19]",
    "x[20]": "x_vars[Opt_20]",
    "x[21]": "x_vars[Opt_21]",
    "x[22]": "x_vars[Opt_22]",
    "x[23]": "x_vars[Opt_23]",
    "x[24]": "x_vars[Opt_24]",
    "x[25]": "x_vars[Opt_25]",
    "x[26]": "x_vars[Opt_26]",
    "x[27]": "x_vars[Opt_27]",
    "x[28]": "x_vars[Opt_28]",
    "x[29]": "x_vars[Opt_29]",
    "x[30]": "x_vars[Opt_30]",
    "x[31]": "x_vars[Opt_31]",
    "x[32]": "x_vars[Opt_32]",
    "x[33]": "x_vars[Opt_33]",
    "x[34]": "x_vars[Opt_34]",
    "x[35]": "x_vars[Opt_35]",
    "x[36]": "x_vars[Opt_36]",
    "x[37]": "x_vars[Opt_37]",
    "x[38]": "x_vars[Opt_38]",
    "x[39]": "x_vars[Opt_39]",
    "x[40]": "x_vars[Opt_40]",
    "x[41]": "x_vars[Opt_41]",
    "x[42]": "x_vars[Opt_42]",
    "x[43]": "x_vars[Opt_43]",
    "x[44]": "x_vars[Opt_44]",
    "x[45]": "x_vars[Opt_45]",
    "x[46]": "x_vars[Opt_46]",
    "x[47]": "x_vars[Opt_47]",
    "x[48]": "x_vars[Opt_48]",
    "x[49]": "x_vars[Opt_49]",
    "x[50]": "x_vars[Opt_50]",
    "x[51]": "x_vars[Opt_51]",
    "x[52]": "x_vars[Opt_52]",
    "x[53]": "x_vars[Opt_53]",
    "x[54]": "x_vars[Opt_54]",
    "x[55]": "x_vars[Opt_55]",
    "x[56]": "x_vars[Opt_56]",
    "x[57]": "x_vars[Opt_57]",
    "x[58]": "x_vars[Opt_58]",
    "x[59]": "x_vars[Opt_59]",
    "x[60]": "x_vars[Opt_60]",
    "x[61]": "x_vars[Opt_61]",
    "x[62]": "x_vars[Opt_62]",
    "x[63]": "x_vars[Opt_63]",
    "x[64]": "x_vars[Opt_64]",
    "x[65]": "x_vars[Opt_65]",
    "x[66]": "x_vars[Opt_66]",
    "x[67]": "x_vars[Opt_67]",
    "x[68]": "x_vars[Opt_68]",
    "x[69]": "x_vars[Opt_69]",
    "x[70]": "x_vars[Opt_70]",
    "x[71]": "x_vars[Opt_71]",
    "x[72]": "x_vars[Opt_72]",
    "x[73]": "x_vars[Opt_73]",
    "x[74]": "x_vars[Opt_74]",
    "x[75]": "x_vars[Opt_75]",
    "x[76]": "x_vars[Opt_76]",
    "x[77]": "x_vars[Opt_77]",
    "x[78]": "x_vars[Opt_78]",
    "x[79]": "x_vars[Opt_79]",
    "x[80]": "x_vars[Opt_80]",
    "x[81]": "x_vars[Opt_81]",
    "x[82]": "x_vars[Opt_82]",
    "x[83]": "x_vars[Opt_83]",
    "x[84]": "x_vars[Opt_84]",
    "x[85]": "x_vars[Opt_85]",
    "x[86]": "x_vars[Opt_86]",
    "x[87]": "x_vars[Opt_87]",
    "x[88]": "x_vars[Opt_88]",
    "x[89]": "x_vars[Opt_89]",
    "x[90]": "x_vars[Opt_90]",
    "x[91]": "x_vars[Opt_91]",
    "x[92]": "x_vars[Opt_92]",
    "x[93]": "x_vars[Opt_93]",
    "x[94]": "x_vars[Opt_94]",
    "x[95]": "x_vars[Opt_95]",
    "x[96]": "x_vars[Opt_96]",
    "x[97]": "x_vars[Opt_97]",
    "x[98]": "x_vars[Opt_98]",
    "x[99]": "x_vars[Opt_99]",
    "x[100]": "x_vars[Opt_100]",
    "x[101]": "x_vars[Opt_101]",
    "x[102]": "x_vars[Opt_102]",
    "x[103]": "x_vars[Opt_103]",
    "x[104]": "x_vars[Opt_104]",
    "x[105]": "x_vars[Opt_105]",
    "x[106]": "x_vars[Opt_106]",
    "x[107]": "x_vars[Opt_107]",
    "x[108]": "x_vars[Opt_108]",
    "x[109]": "x_vars[Opt_109]",
    "x[110]": "x_vars[Opt_110]",
    "x[111]": "x_vars[Opt_111]",
    "x[112]": "x_vars[Opt_112]",
    "x[113]": "x_vars[Opt_113]",
    "x[114]": "x_vars[Opt_114]",
    "x[115]": "x_vars[Opt_115]",
    "x[116]": "x_vars[Opt_116]",
    "x[117]": "x_vars[Opt_117]",
    "x[118]": "x_vars[Opt_118]",
    "x[119]": "x_vars[Opt_119]",
    "x[120]": "x_vars[Opt_120]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-098": {
   "source_code_sha256": "38dc975431c131502c6c386d93d8944c799f6fd8287cb277ca46493aa0d1ca74",
   "original_solution_sha256": "3b69dc05c5ad0bd769f2ed239169ab698f91804aa1f3f977856c1325a000834f",
   "bindings": [
    [
     "C0",
     "w_vars[Carpenter]"
    ],
    [
     "C1",
     "w_vars[Electrician]"
    ],
    [
     "C2",
     "w_vars[Painter]"
    ],
    [
     "C3",
     "w_vars[Worker_004]"
    ],
    [
     "C4",
     "w_vars[Worker_005]"
    ],
    [
     "C5",
     "w_vars[Worker_006]"
    ],
    [
     "C6",
     "w_vars[Worker_007]"
    ],
    [
     "C7",
     "w_vars[Worker_008]"
    ],
    [
     "C8",
     "w_vars[Worker_009]"
    ],
    [
     "C9",
     "w_vars[Worker_010]"
    ],
    [
     "C10",
     "w_vars[Worker_011]"
    ],
    [
     "C11",
     "w_vars[Worker_012]"
    ],
    [
     "C12",
     "w_vars[Worker_013]"
    ],
    [
     "C13",
     "w_vars[Worker_014]"
    ],
    [
     "C14",
     "w_vars[Worker_015]"
    ],
    [
     "C15",
     "w_vars[Worker_016]"
    ],
    [
     "C16",
     "w_vars[Worker_017]"
    ],
    [
     "C17",
     "w_vars[Worker_018]"
    ],
    [
     "C18",
     "w_vars[Worker_019]"
    ],
    [
     "C19",
     "w_vars[Worker_020]"
    ],
    [
     "C20",
     "w_vars[Worker_021]"
    ],
    [
     "C21",
     "w_vars[Worker_022]"
    ],
    [
     "C22",
     "w_vars[Worker_023]"
    ],
    [
     "C23",
     "w_vars[Worker_024]"
    ],
    [
     "C24",
     "w_vars[Worker_025]"
    ],
    [
     "C25",
     "w_vars[Worker_026]"
    ],
    [
     "C26",
     "w_vars[Worker_027]"
    ],
    [
     "C27",
     "w_vars[Worker_028]"
    ],
    [
     "C28",
     "w_vars[Worker_029]"
    ],
    [
     "C29",
     "w_vars[Worker_030]"
    ],
    [
     "C30",
     "w_vars[Worker_031]"
    ],
    [
     "C31",
     "w_vars[Worker_032]"
    ],
    [
     "C32",
     "w_vars[Worker_033]"
    ],
    [
     "C33",
     "w_vars[Worker_034]"
    ],
    [
     "C34",
     "w_vars[Worker_035]"
    ],
    [
     "C35",
     "w_vars[Worker_036]"
    ],
    [
     "C36",
     "w_vars[Worker_037]"
    ],
    [
     "C37",
     "w_vars[Worker_038]"
    ],
    [
     "C38",
     "w_vars[Worker_039]"
    ],
    [
     "C39",
     "w_vars[Worker_040]"
    ],
    [
     "C40",
     "w_vars[Worker_041]"
    ],
    [
     "C41",
     "w_vars[Worker_042]"
    ],
    [
     "C42",
     "w_vars[Worker_043]"
    ],
    [
     "C43",
     "w_vars[Worker_044]"
    ],
    [
     "C44",
     "w_vars[Worker_045]"
    ],
    [
     "C45",
     "w_vars[Worker_046]"
    ],
    [
     "C46",
     "w_vars[Worker_047]"
    ],
    [
     "C47",
     "w_vars[Worker_048]"
    ],
    [
     "C48",
     "w_vars[Worker_049]"
    ],
    [
     "C49",
     "w_vars[Worker_050]"
    ],
    [
     "C50",
     "w_vars[Worker_051]"
    ],
    [
     "C51",
     "w_vars[Worker_052]"
    ],
    [
     "C52",
     "w_vars[Worker_053]"
    ],
    [
     "C53",
     "w_vars[Worker_054]"
    ],
    [
     "C54",
     "w_vars[Worker_055]"
    ],
    [
     "C55",
     "w_vars[Worker_056]"
    ],
    [
     "C56",
     "w_vars[Worker_057]"
    ],
    [
     "C57",
     "w_vars[Worker_058]"
    ],
    [
     "C58",
     "w_vars[Worker_059]"
    ],
    [
     "C59",
     "w_vars[Worker_060]"
    ],
    [
     "C60",
     "w_vars[Worker_061]"
    ],
    [
     "C61",
     "w_vars[Worker_062]"
    ],
    [
     "C62",
     "w_vars[Worker_063]"
    ],
    [
     "C63",
     "w_vars[Worker_064]"
    ],
    [
     "C64",
     "w_vars[Worker_065]"
    ],
    [
     "C65",
     "w_vars[Worker_066]"
    ],
    [
     "C66",
     "w_vars[Worker_067]"
    ],
    [
     "C67",
     "w_vars[Worker_068]"
    ],
    [
     "C68",
     "w_vars[Worker_069]"
    ],
    [
     "C69",
     "w_vars[Worker_070]"
    ],
    [
     "C70",
     "w_vars[Worker_071]"
    ],
    [
     "C71",
     "w_vars[Worker_072]"
    ],
    [
     "C72",
     "w_vars[Worker_073]"
    ],
    [
     "C73",
     "w_vars[Worker_074]"
    ],
    [
     "C74",
     "w_vars[Worker_075]"
    ],
    [
     "C75",
     "w_vars[Worker_076]"
    ],
    [
     "C76",
     "w_vars[Worker_077]"
    ],
    [
     "C77",
     "w_vars[Worker_078]"
    ],
    [
     "C78",
     "w_vars[Worker_079]"
    ],
    [
     "C79",
     "w_vars[Worker_080]"
    ],
    [
     "C80",
     "w_vars[Worker_081]"
    ],
    [
     "C81",
     "w_vars[Worker_082]"
    ],
    [
     "C82",
     "w_vars[Worker_083]"
    ],
    [
     "C83",
     "w_vars[Worker_084]"
    ],
    [
     "C84",
     "w_vars[Worker_085]"
    ],
    [
     "C85",
     "w_vars[Worker_086]"
    ],
    [
     "C86",
     "w_vars[Worker_087]"
    ],
    [
     "C87",
     "w_vars[Worker_088]"
    ],
    [
     "C88",
     "w_vars[Worker_089]"
    ],
    [
     "C89",
     "w_vars[Worker_090]"
    ],
    [
     "C90",
     "w_vars[Worker_091]"
    ],
    [
     "C91",
     "w_vars[Worker_092]"
    ],
    [
     "C92",
     "w_vars[Worker_093]"
    ],
    [
     "C93",
     "w_vars[Worker_094]"
    ],
    [
     "C94",
     "w_vars[Worker_095]"
    ],
    [
     "C95",
     "w_vars[Worker_096]"
    ],
    [
     "C96",
     "w_vars[Worker_097]"
    ],
    [
     "C97",
     "w_vars[Worker_098]"
    ],
    [
     "C98",
     "w_vars[Worker_099]"
    ],
    [
     "C99",
     "w_vars[Worker_100]"
    ],
    [
     "C100",
     "w_vars[Worker_101]"
    ],
    [
     "C101",
     "w_vars[Worker_102]"
    ],
    [
     "C102",
     "w_vars[Worker_103]"
    ],
    [
     "C103",
     "w_vars[Worker_104]"
    ],
    [
     "C104",
     "w_vars[Worker_105]"
    ],
    [
     "C105",
     "w_vars[Worker_106]"
    ],
    [
     "C106",
     "w_vars[Worker_107]"
    ],
    [
     "C107",
     "w_vars[Worker_108]"
    ],
    [
     "C108",
     "w_vars[Worker_109]"
    ],
    [
     "C109",
     "w_vars[Worker_110]"
    ],
    [
     "C110",
     "w_vars[Worker_111]"
    ],
    [
     "C111",
     "w_vars[Worker_112]"
    ],
    [
     "C112",
     "w_vars[Worker_113]"
    ],
    [
     "C113",
     "w_vars[Worker_114]"
    ],
    [
     "C114",
     "w_vars[Worker_115]"
    ],
    [
     "C115",
     "w_vars[Worker_116]"
    ],
    [
     "C116",
     "w_vars[Worker_117]"
    ],
    [
     "C117",
     "w_vars[Worker_118]"
    ],
    [
     "C118",
     "w_vars[Worker_119]"
    ],
    [
     "C119",
     "w_vars[Worker_120]"
    ],
    [
     "C120",
     "w_vars[Worker_121]"
    ],
    [
     "C121",
     "w_vars[Worker_122]"
    ],
    [
     "C122",
     "w_vars[Worker_123]"
    ],
    [
     "C123",
     "w_vars[Worker_124]"
    ],
    [
     "C124",
     "w_vars[Worker_125]"
    ],
    [
     "C125",
     "w_vars[Worker_126]"
    ],
    [
     "C126",
     "w_vars[Worker_127]"
    ],
    [
     "C127",
     "w_vars[Worker_128]"
    ],
    [
     "C128",
     "w_vars[Worker_129]"
    ],
    [
     "C129",
     "w_vars[Worker_130]"
    ],
    [
     "C130",
     "w_vars[Worker_131]"
    ],
    [
     "C131",
     "w_vars[Worker_132]"
    ],
    [
     "C132",
     "w_vars[Worker_133]"
    ],
    [
     "C133",
     "w_vars[Worker_134]"
    ],
    [
     "C134",
     "w_vars[Worker_135]"
    ],
    [
     "C135",
     "w_vars[Worker_136]"
    ],
    [
     "C136",
     "w_vars[Worker_137]"
    ],
    [
     "C137",
     "w_vars[Worker_138]"
    ],
    [
     "C138",
     "w_vars[Worker_139]"
    ],
    [
     "C139",
     "w_vars[Worker_140]"
    ],
    [
     "C140",
     "w_vars[Worker_141]"
    ],
    [
     "C141",
     "w_vars[Worker_142]"
    ],
    [
     "C142",
     "w_vars[Worker_143]"
    ],
    [
     "C143",
     "w_vars[Worker_144]"
    ],
    [
     "C144",
     "w_vars[Worker_145]"
    ],
    [
     "C145",
     "w_vars[Worker_146]"
    ],
    [
     "C146",
     "w_vars[Worker_147]"
    ],
    [
     "C147",
     "w_vars[Worker_148]"
    ],
    [
     "C148",
     "w_vars[Worker_149]"
    ],
    [
     "C149",
     "w_vars[Worker_150]"
    ]
   ],
   "primary_mapping": {
    "wage[0]": "w_vars[Carpenter]",
    "wage[1]": "w_vars[Electrician]",
    "wage[2]": "w_vars[Painter]",
    "wage[3]": "w_vars[Worker_004]",
    "wage[4]": "w_vars[Worker_005]",
    "wage[5]": "w_vars[Worker_006]",
    "wage[6]": "w_vars[Worker_007]",
    "wage[7]": "w_vars[Worker_008]",
    "wage[8]": "w_vars[Worker_009]",
    "wage[9]": "w_vars[Worker_010]",
    "wage[10]": "w_vars[Worker_011]",
    "wage[11]": "w_vars[Worker_012]",
    "wage[12]": "w_vars[Worker_013]",
    "wage[13]": "w_vars[Worker_014]",
    "wage[14]": "w_vars[Worker_015]",
    "wage[15]": "w_vars[Worker_016]",
    "wage[16]": "w_vars[Worker_017]",
    "wage[17]": "w_vars[Worker_018]",
    "wage[18]": "w_vars[Worker_019]",
    "wage[19]": "w_vars[Worker_020]",
    "wage[20]": "w_vars[Worker_021]",
    "wage[21]": "w_vars[Worker_022]",
    "wage[22]": "w_vars[Worker_023]",
    "wage[23]": "w_vars[Worker_024]",
    "wage[24]": "w_vars[Worker_025]",
    "wage[25]": "w_vars[Worker_026]",
    "wage[26]": "w_vars[Worker_027]",
    "wage[27]": "w_vars[Worker_028]",
    "wage[28]": "w_vars[Worker_029]",
    "wage[29]": "w_vars[Worker_030]",
    "wage[30]": "w_vars[Worker_031]",
    "wage[31]": "w_vars[Worker_032]",
    "wage[32]": "w_vars[Worker_033]",
    "wage[33]": "w_vars[Worker_034]",
    "wage[34]": "w_vars[Worker_035]",
    "wage[35]": "w_vars[Worker_036]",
    "wage[36]": "w_vars[Worker_037]",
    "wage[37]": "w_vars[Worker_038]",
    "wage[38]": "w_vars[Worker_039]",
    "wage[39]": "w_vars[Worker_040]",
    "wage[40]": "w_vars[Worker_041]",
    "wage[41]": "w_vars[Worker_042]",
    "wage[42]": "w_vars[Worker_043]",
    "wage[43]": "w_vars[Worker_044]",
    "wage[44]": "w_vars[Worker_045]",
    "wage[45]": "w_vars[Worker_046]",
    "wage[46]": "w_vars[Worker_047]",
    "wage[47]": "w_vars[Worker_048]",
    "wage[48]": "w_vars[Worker_049]",
    "wage[49]": "w_vars[Worker_050]",
    "wage[50]": "w_vars[Worker_051]",
    "wage[51]": "w_vars[Worker_052]",
    "wage[52]": "w_vars[Worker_053]",
    "wage[53]": "w_vars[Worker_054]",
    "wage[54]": "w_vars[Worker_055]",
    "wage[55]": "w_vars[Worker_056]",
    "wage[56]": "w_vars[Worker_057]",
    "wage[57]": "w_vars[Worker_058]",
    "wage[58]": "w_vars[Worker_059]",
    "wage[59]": "w_vars[Worker_060]",
    "wage[60]": "w_vars[Worker_061]",
    "wage[61]": "w_vars[Worker_062]",
    "wage[62]": "w_vars[Worker_063]",
    "wage[63]": "w_vars[Worker_064]",
    "wage[64]": "w_vars[Worker_065]",
    "wage[65]": "w_vars[Worker_066]",
    "wage[66]": "w_vars[Worker_067]",
    "wage[67]": "w_vars[Worker_068]",
    "wage[68]": "w_vars[Worker_069]",
    "wage[69]": "w_vars[Worker_070]",
    "wage[70]": "w_vars[Worker_071]",
    "wage[71]": "w_vars[Worker_072]",
    "wage[72]": "w_vars[Worker_073]",
    "wage[73]": "w_vars[Worker_074]",
    "wage[74]": "w_vars[Worker_075]",
    "wage[75]": "w_vars[Worker_076]",
    "wage[76]": "w_vars[Worker_077]",
    "wage[77]": "w_vars[Worker_078]",
    "wage[78]": "w_vars[Worker_079]",
    "wage[79]": "w_vars[Worker_080]",
    "wage[80]": "w_vars[Worker_081]",
    "wage[81]": "w_vars[Worker_082]",
    "wage[82]": "w_vars[Worker_083]",
    "wage[83]": "w_vars[Worker_084]",
    "wage[84]": "w_vars[Worker_085]",
    "wage[85]": "w_vars[Worker_086]",
    "wage[86]": "w_vars[Worker_087]",
    "wage[87]": "w_vars[Worker_088]",
    "wage[88]": "w_vars[Worker_089]",
    "wage[89]": "w_vars[Worker_090]",
    "wage[90]": "w_vars[Worker_091]",
    "wage[91]": "w_vars[Worker_092]",
    "wage[92]": "w_vars[Worker_093]",
    "wage[93]": "w_vars[Worker_094]",
    "wage[94]": "w_vars[Worker_095]",
    "wage[95]": "w_vars[Worker_096]",
    "wage[96]": "w_vars[Worker_097]",
    "wage[97]": "w_vars[Worker_098]",
    "wage[98]": "w_vars[Worker_099]",
    "wage[99]": "w_vars[Worker_100]",
    "wage[100]": "w_vars[Worker_101]",
    "wage[101]": "w_vars[Worker_102]",
    "wage[102]": "w_vars[Worker_103]",
    "wage[103]": "w_vars[Worker_104]",
    "wage[104]": "w_vars[Worker_105]",
    "wage[105]": "w_vars[Worker_106]",
    "wage[106]": "w_vars[Worker_107]",
    "wage[107]": "w_vars[Worker_108]",
    "wage[108]": "w_vars[Worker_109]",
    "wage[109]": "w_vars[Worker_110]",
    "wage[110]": "w_vars[Worker_111]",
    "wage[111]": "w_vars[Worker_112]",
    "wage[112]": "w_vars[Worker_113]",
    "wage[113]": "w_vars[Worker_114]",
    "wage[114]": "w_vars[Worker_115]",
    "wage[115]": "w_vars[Worker_116]",
    "wage[116]": "w_vars[Worker_117]",
    "wage[117]": "w_vars[Worker_118]",
    "wage[118]": "w_vars[Worker_119]",
    "wage[119]": "w_vars[Worker_120]",
    "wage[120]": "w_vars[Worker_121]",
    "wage[121]": "w_vars[Worker_122]",
    "wage[122]": "w_vars[Worker_123]",
    "wage[123]": "w_vars[Worker_124]",
    "wage[124]": "w_vars[Worker_125]",
    "wage[125]": "w_vars[Worker_126]",
    "wage[126]": "w_vars[Worker_127]",
    "wage[127]": "w_vars[Worker_128]",
    "wage[128]": "w_vars[Worker_129]",
    "wage[129]": "w_vars[Worker_130]",
    "wage[130]": "w_vars[Worker_131]",
    "wage[131]": "w_vars[Worker_132]",
    "wage[132]": "w_vars[Worker_133]",
    "wage[133]": "w_vars[Worker_134]",
    "wage[134]": "w_vars[Worker_135]",
    "wage[135]": "w_vars[Worker_136]",
    "wage[136]": "w_vars[Worker_137]",
    "wage[137]": "w_vars[Worker_138]",
    "wage[138]": "w_vars[Worker_139]",
    "wage[139]": "w_vars[Worker_140]",
    "wage[140]": "w_vars[Worker_141]",
    "wage[141]": "w_vars[Worker_142]",
    "wage[142]": "w_vars[Worker_143]",
    "wage[143]": "w_vars[Worker_144]",
    "wage[144]": "w_vars[Worker_145]",
    "wage[145]": "w_vars[Worker_146]",
    "wage[146]": "w_vars[Worker_147]",
    "wage[147]": "w_vars[Worker_148]",
    "wage[148]": "w_vars[Worker_149]",
    "wage[149]": "w_vars[Worker_150]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-099": {
   "source_code_sha256": "8e98727c740b28ee29d1701cd5688c7dddd97e4bb437717bcd71e57524381d25",
   "original_solution_sha256": "205ceda6c3138e5950628f40c255a74f6de2d6d5d0cf6111a9b1b71d2c86e21a",
   "bindings": [
    [
     "C0",
     "x_vars[1,1]"
    ],
    [
     "C1",
     "x_vars[1,2]"
    ],
    [
     "C2",
     "x_vars[1,3]"
    ],
    [
     "C3",
     "x_vars[1,4]"
    ],
    [
     "C4",
     "x_vars[1,5]"
    ],
    [
     "C5",
     "x_vars[1,6]"
    ],
    [
     "C6",
     "x_vars[1,7]"
    ],
    [
     "C7",
     "x_vars[1,8]"
    ],
    [
     "C8",
     "x_vars[1,9]"
    ],
    [
     "C9",
     "x_vars[1,10]"
    ],
    [
     "C10",
     "x_vars[1,11]"
    ],
    [
     "C11",
     "x_vars[2,1]"
    ],
    [
     "C12",
     "x_vars[2,2]"
    ],
    [
     "C13",
     "x_vars[2,3]"
    ],
    [
     "C14",
     "x_vars[2,4]"
    ],
    [
     "C15",
     "x_vars[2,5]"
    ],
    [
     "C16",
     "x_vars[2,6]"
    ],
    [
     "C17",
     "x_vars[2,7]"
    ],
    [
     "C18",
     "x_vars[2,8]"
    ],
    [
     "C19",
     "x_vars[2,9]"
    ],
    [
     "C20",
     "x_vars[2,10]"
    ],
    [
     "C21",
     "x_vars[2,11]"
    ],
    [
     "C22",
     "x_vars[3,1]"
    ],
    [
     "C23",
     "x_vars[3,2]"
    ],
    [
     "C24",
     "x_vars[3,3]"
    ],
    [
     "C25",
     "x_vars[3,4]"
    ],
    [
     "C26",
     "x_vars[3,5]"
    ],
    [
     "C27",
     "x_vars[3,6]"
    ],
    [
     "C28",
     "x_vars[3,7]"
    ],
    [
     "C29",
     "x_vars[3,8]"
    ],
    [
     "C30",
     "x_vars[3,9]"
    ],
    [
     "C31",
     "x_vars[3,10]"
    ],
    [
     "C32",
     "x_vars[3,11]"
    ],
    [
     "C33",
     "x_vars[4,1]"
    ],
    [
     "C34",
     "x_vars[4,2]"
    ],
    [
     "C35",
     "x_vars[4,3]"
    ],
    [
     "C36",
     "x_vars[4,4]"
    ],
    [
     "C37",
     "x_vars[4,5]"
    ],
    [
     "C38",
     "x_vars[4,6]"
    ],
    [
     "C39",
     "x_vars[4,7]"
    ],
    [
     "C40",
     "x_vars[4,8]"
    ],
    [
     "C41",
     "x_vars[4,9]"
    ],
    [
     "C42",
     "x_vars[4,10]"
    ],
    [
     "C43",
     "x_vars[4,11]"
    ],
    [
     "C44",
     "x_vars[5,1]"
    ],
    [
     "C45",
     "x_vars[5,2]"
    ],
    [
     "C46",
     "x_vars[5,3]"
    ],
    [
     "C47",
     "x_vars[5,4]"
    ],
    [
     "C48",
     "x_vars[5,5]"
    ],
    [
     "C49",
     "x_vars[5,6]"
    ],
    [
     "C50",
     "x_vars[5,7]"
    ],
    [
     "C51",
     "x_vars[5,8]"
    ],
    [
     "C52",
     "x_vars[5,9]"
    ],
    [
     "C53",
     "x_vars[5,10]"
    ],
    [
     "C54",
     "x_vars[5,11]"
    ],
    [
     "C55",
     "x_vars[6,1]"
    ],
    [
     "C56",
     "x_vars[6,2]"
    ],
    [
     "C57",
     "x_vars[6,3]"
    ],
    [
     "C58",
     "x_vars[6,4]"
    ],
    [
     "C59",
     "x_vars[6,5]"
    ],
    [
     "C60",
     "x_vars[6,6]"
    ],
    [
     "C61",
     "x_vars[6,7]"
    ],
    [
     "C62",
     "x_vars[6,8]"
    ],
    [
     "C63",
     "x_vars[6,9]"
    ],
    [
     "C64",
     "x_vars[6,10]"
    ],
    [
     "C65",
     "x_vars[6,11]"
    ],
    [
     "C66",
     "x_vars[7,1]"
    ],
    [
     "C67",
     "x_vars[7,2]"
    ],
    [
     "C68",
     "x_vars[7,3]"
    ],
    [
     "C69",
     "x_vars[7,4]"
    ],
    [
     "C70",
     "x_vars[7,5]"
    ],
    [
     "C71",
     "x_vars[7,6]"
    ],
    [
     "C72",
     "x_vars[7,7]"
    ],
    [
     "C73",
     "x_vars[7,8]"
    ],
    [
     "C74",
     "x_vars[7,9]"
    ],
    [
     "C75",
     "x_vars[7,10]"
    ],
    [
     "C76",
     "x_vars[7,11]"
    ],
    [
     "C77",
     "x_vars[8,1]"
    ],
    [
     "C78",
     "x_vars[8,2]"
    ],
    [
     "C79",
     "x_vars[8,3]"
    ],
    [
     "C80",
     "x_vars[8,4]"
    ],
    [
     "C81",
     "x_vars[8,5]"
    ],
    [
     "C82",
     "x_vars[8,6]"
    ],
    [
     "C83",
     "x_vars[8,7]"
    ],
    [
     "C84",
     "x_vars[8,8]"
    ],
    [
     "C85",
     "x_vars[8,9]"
    ],
    [
     "C86",
     "x_vars[8,10]"
    ],
    [
     "C87",
     "x_vars[8,11]"
    ],
    [
     "C88",
     "x_vars[9,1]"
    ],
    [
     "C89",
     "x_vars[9,2]"
    ],
    [
     "C90",
     "x_vars[9,3]"
    ],
    [
     "C91",
     "x_vars[9,4]"
    ],
    [
     "C92",
     "x_vars[9,5]"
    ],
    [
     "C93",
     "x_vars[9,6]"
    ],
    [
     "C94",
     "x_vars[9,7]"
    ],
    [
     "C95",
     "x_vars[9,8]"
    ],
    [
     "C96",
     "x_vars[9,9]"
    ],
    [
     "C97",
     "x_vars[9,10]"
    ],
    [
     "C98",
     "x_vars[9,11]"
    ],
    [
     "C99",
     "x_vars[10,1]"
    ],
    [
     "C100",
     "x_vars[10,2]"
    ],
    [
     "C101",
     "x_vars[10,3]"
    ],
    [
     "C102",
     "x_vars[10,4]"
    ],
    [
     "C103",
     "x_vars[10,5]"
    ],
    [
     "C104",
     "x_vars[10,6]"
    ],
    [
     "C105",
     "x_vars[10,7]"
    ],
    [
     "C106",
     "x_vars[10,8]"
    ],
    [
     "C107",
     "x_vars[10,9]"
    ],
    [
     "C108",
     "x_vars[10,10]"
    ],
    [
     "C109",
     "x_vars[10,11]"
    ],
    [
     "C110",
     "x_vars[11,1]"
    ],
    [
     "C111",
     "x_vars[11,2]"
    ],
    [
     "C112",
     "x_vars[11,3]"
    ],
    [
     "C113",
     "x_vars[11,4]"
    ],
    [
     "C114",
     "x_vars[11,5]"
    ],
    [
     "C115",
     "x_vars[11,6]"
    ],
    [
     "C116",
     "x_vars[11,7]"
    ],
    [
     "C117",
     "x_vars[11,8]"
    ],
    [
     "C118",
     "x_vars[11,9]"
    ],
    [
     "C119",
     "x_vars[11,10]"
    ],
    [
     "C120",
     "x_vars[11,11]"
    ],
    [
     "C121",
     "y_vars[1]"
    ],
    [
     "C122",
     "y_vars[2]"
    ],
    [
     "C123",
     "y_vars[3]"
    ],
    [
     "C124",
     "y_vars[4]"
    ],
    [
     "C125",
     "y_vars[5]"
    ],
    [
     "C126",
     "y_vars[6]"
    ],
    [
     "C127",
     "y_vars[7]"
    ],
    [
     "C128",
     "y_vars[8]"
    ],
    [
     "C129",
     "y_vars[9]"
    ],
    [
     "C130",
     "y_vars[10]"
    ],
    [
     "C131",
     "y_vars[11]"
    ]
   ],
   "primary_mapping": {
    "x_W1_S1": "x_vars[1,1]",
    "x_W1_S2": "x_vars[1,2]",
    "x_W1_S3": "x_vars[1,3]",
    "x_W1_S4": "x_vars[1,4]",
    "x_W1_S5": "x_vars[1,5]",
    "x_W1_S6": "x_vars[1,6]",
    "x_W1_S7": "x_vars[1,7]",
    "x_W1_S8": "x_vars[1,8]",
    "x_W1_S9": "x_vars[1,9]",
    "x_W1_S10": "x_vars[1,10]",
    "x_W1_S11": "x_vars[1,11]",
    "x_W2_S1": "x_vars[2,1]",
    "x_W2_S2": "x_vars[2,2]",
    "x_W2_S3": "x_vars[2,3]",
    "x_W2_S4": "x_vars[2,4]",
    "x_W2_S5": "x_vars[2,5]",
    "x_W2_S6": "x_vars[2,6]",
    "x_W2_S7": "x_vars[2,7]",
    "x_W2_S8": "x_vars[2,8]",
    "x_W2_S9": "x_vars[2,9]",
    "x_W2_S10": "x_vars[2,10]",
    "x_W2_S11": "x_vars[2,11]",
    "x_W3_S1": "x_vars[3,1]",
    "x_W3_S2": "x_vars[3,2]",
    "x_W3_S3": "x_vars[3,3]",
    "x_W3_S4": "x_vars[3,4]",
    "x_W3_S5": "x_vars[3,5]",
    "x_W3_S6": "x_vars[3,6]",
    "x_W3_S7": "x_vars[3,7]",
    "x_W3_S8": "x_vars[3,8]",
    "x_W3_S9": "x_vars[3,9]",
    "x_W3_S10": "x_vars[3,10]",
    "x_W3_S11": "x_vars[3,11]",
    "x_W4_S1": "x_vars[4,1]",
    "x_W4_S2": "x_vars[4,2]",
    "x_W4_S3": "x_vars[4,3]",
    "x_W4_S4": "x_vars[4,4]",
    "x_W4_S5": "x_vars[4,5]",
    "x_W4_S6": "x_vars[4,6]",
    "x_W4_S7": "x_vars[4,7]",
    "x_W4_S8": "x_vars[4,8]",
    "x_W4_S9": "x_vars[4,9]",
    "x_W4_S10": "x_vars[4,10]",
    "x_W4_S11": "x_vars[4,11]",
    "x_W5_S1": "x_vars[5,1]",
    "x_W5_S2": "x_vars[5,2]",
    "x_W5_S3": "x_vars[5,3]",
    "x_W5_S4": "x_vars[5,4]",
    "x_W5_S5": "x_vars[5,5]",
    "x_W5_S6": "x_vars[5,6]",
    "x_W5_S7": "x_vars[5,7]",
    "x_W5_S8": "x_vars[5,8]",
    "x_W5_S9": "x_vars[5,9]",
    "x_W5_S10": "x_vars[5,10]",
    "x_W5_S11": "x_vars[5,11]",
    "x_W6_S1": "x_vars[6,1]",
    "x_W6_S2": "x_vars[6,2]",
    "x_W6_S3": "x_vars[6,3]",
    "x_W6_S4": "x_vars[6,4]",
    "x_W6_S5": "x_vars[6,5]",
    "x_W6_S6": "x_vars[6,6]",
    "x_W6_S7": "x_vars[6,7]",
    "x_W6_S8": "x_vars[6,8]",
    "x_W6_S9": "x_vars[6,9]",
    "x_W6_S10": "x_vars[6,10]",
    "x_W6_S11": "x_vars[6,11]",
    "x_W7_S1": "x_vars[7,1]",
    "x_W7_S2": "x_vars[7,2]",
    "x_W7_S3": "x_vars[7,3]",
    "x_W7_S4": "x_vars[7,4]",
    "x_W7_S5": "x_vars[7,5]",
    "x_W7_S6": "x_vars[7,6]",
    "x_W7_S7": "x_vars[7,7]",
    "x_W7_S8": "x_vars[7,8]",
    "x_W7_S9": "x_vars[7,9]",
    "x_W7_S10": "x_vars[7,10]",
    "x_W7_S11": "x_vars[7,11]",
    "x_W8_S1": "x_vars[8,1]",
    "x_W8_S2": "x_vars[8,2]",
    "x_W8_S3": "x_vars[8,3]",
    "x_W8_S4": "x_vars[8,4]",
    "x_W8_S5": "x_vars[8,5]",
    "x_W8_S6": "x_vars[8,6]",
    "x_W8_S7": "x_vars[8,7]",
    "x_W8_S8": "x_vars[8,8]",
    "x_W8_S9": "x_vars[8,9]",
    "x_W8_S10": "x_vars[8,10]",
    "x_W8_S11": "x_vars[8,11]",
    "x_W9_S1": "x_vars[9,1]",
    "x_W9_S2": "x_vars[9,2]",
    "x_W9_S3": "x_vars[9,3]",
    "x_W9_S4": "x_vars[9,4]",
    "x_W9_S5": "x_vars[9,5]",
    "x_W9_S6": "x_vars[9,6]",
    "x_W9_S7": "x_vars[9,7]",
    "x_W9_S8": "x_vars[9,8]",
    "x_W9_S9": "x_vars[9,9]",
    "x_W9_S10": "x_vars[9,10]",
    "x_W9_S11": "x_vars[9,11]",
    "x_W10_S1": "x_vars[10,1]",
    "x_W10_S2": "x_vars[10,2]",
    "x_W10_S3": "x_vars[10,3]",
    "x_W10_S4": "x_vars[10,4]",
    "x_W10_S5": "x_vars[10,5]",
    "x_W10_S6": "x_vars[10,6]",
    "x_W10_S7": "x_vars[10,7]",
    "x_W10_S8": "x_vars[10,8]",
    "x_W10_S9": "x_vars[10,9]",
    "x_W10_S10": "x_vars[10,10]",
    "x_W10_S11": "x_vars[10,11]",
    "x_W11_S1": "x_vars[11,1]",
    "x_W11_S2": "x_vars[11,2]",
    "x_W11_S3": "x_vars[11,3]",
    "x_W11_S4": "x_vars[11,4]",
    "x_W11_S5": "x_vars[11,5]",
    "x_W11_S6": "x_vars[11,6]",
    "x_W11_S7": "x_vars[11,7]",
    "x_W11_S8": "x_vars[11,8]",
    "x_W11_S9": "x_vars[11,9]",
    "x_W11_S10": "x_vars[11,10]",
    "x_W11_S11": "x_vars[11,11]",
    "y_W1": "y_vars[1]",
    "y_W2": "y_vars[2]",
    "y_W3": "y_vars[3]",
    "y_W4": "y_vars[4]",
    "y_W5": "y_vars[5]",
    "y_W6": "y_vars[6]",
    "y_W7": "y_vars[7]",
    "y_W8": "y_vars[8]",
    "y_W9": "y_vars[9]",
    "y_W10": "y_vars[10]",
    "y_W11": "y_vars[11]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-100": {
   "source_code_sha256": "6960dab533a19cd61b6e3f987675533c6cd44fd34af96ad6fc018a3f861dda02",
   "original_solution_sha256": "9c4b9195636cb1aa0c8e47c6ba42f178080a4684d879c2ac6d9f373e40a74b27",
   "bindings": [
    [
     "C0",
     "x_vars[C1]"
    ],
    [
     "C1",
     "x_vars[C2]"
    ],
    [
     "C2",
     "x_vars[C3]"
    ],
    [
     "C3",
     "x_vars[C4]"
    ],
    [
     "C4",
     "x_vars[C5]"
    ],
    [
     "C5",
     "x_vars[C6]"
    ],
    [
     "C6",
     "x_vars[C7]"
    ],
    [
     "C7",
     "x_vars[C8]"
    ],
    [
     "C8",
     "x_vars[C9]"
    ],
    [
     "C9",
     "x_vars[C10]"
    ],
    [
     "C10",
     "x_vars[C11]"
    ],
    [
     "C11",
     "x_vars[C12]"
    ],
    [
     "C12",
     "x_vars[C13]"
    ],
    [
     "C13",
     "x_vars[C14]"
    ],
    [
     "C14",
     "x_vars[C15]"
    ],
    [
     "C15",
     "x_vars[C16]"
    ],
    [
     "C16",
     "x_vars[C17]"
    ],
    [
     "C17",
     "x_vars[C18]"
    ],
    [
     "C18",
     "x_vars[C19]"
    ],
    [
     "C19",
     "x_vars[C20]"
    ],
    [
     "C20",
     "x_vars[C21]"
    ],
    [
     "C21",
     "x_vars[C22]"
    ],
    [
     "C22",
     "x_vars[C23]"
    ],
    [
     "C23",
     "x_vars[C24]"
    ],
    [
     "C24",
     "x_vars[C25]"
    ],
    [
     "C25",
     "x_vars[C26]"
    ],
    [
     "C26",
     "x_vars[C27]"
    ],
    [
     "C27",
     "x_vars[C28]"
    ],
    [
     "C28",
     "x_vars[C29]"
    ],
    [
     "C29",
     "x_vars[C30]"
    ],
    [
     "C30",
     "x_vars[C31]"
    ],
    [
     "C31",
     "x_vars[C32]"
    ],
    [
     "C32",
     "x_vars[C33]"
    ],
    [
     "C33",
     "x_vars[C34]"
    ],
    [
     "C34",
     "x_vars[C35]"
    ],
    [
     "C35",
     "x_vars[C36]"
    ],
    [
     "C36",
     "x_vars[C37]"
    ],
    [
     "C37",
     "x_vars[C38]"
    ],
    [
     "C38",
     "x_vars[C39]"
    ],
    [
     "C39",
     "x_vars[C40]"
    ],
    [
     "C40",
     "x_vars[C41]"
    ],
    [
     "C41",
     "x_vars[C42]"
    ],
    [
     "C42",
     "x_vars[C43]"
    ],
    [
     "C43",
     "x_vars[C44]"
    ],
    [
     "C44",
     "x_vars[C45]"
    ],
    [
     "C45",
     "x_vars[C46]"
    ],
    [
     "C46",
     "x_vars[C47]"
    ],
    [
     "C47",
     "x_vars[C48]"
    ],
    [
     "C48",
     "x_vars[C49]"
    ],
    [
     "C49",
     "x_vars[C50]"
    ],
    [
     "C50",
     "x_vars[C51]"
    ],
    [
     "C51",
     "x_vars[C52]"
    ],
    [
     "C52",
     "x_vars[C53]"
    ],
    [
     "C53",
     "x_vars[C54]"
    ],
    [
     "C54",
     "x_vars[C55]"
    ],
    [
     "C55",
     "x_vars[C56]"
    ],
    [
     "C56",
     "x_vars[C57]"
    ],
    [
     "C57",
     "x_vars[C58]"
    ],
    [
     "C58",
     "x_vars[C59]"
    ],
    [
     "C59",
     "x_vars[C60]"
    ],
    [
     "C60",
     "x_vars[C61]"
    ],
    [
     "C61",
     "x_vars[C62]"
    ],
    [
     "C62",
     "x_vars[C63]"
    ],
    [
     "C63",
     "x_vars[C64]"
    ],
    [
     "C64",
     "x_vars[C65]"
    ],
    [
     "C65",
     "x_vars[C66]"
    ],
    [
     "C66",
     "x_vars[C67]"
    ],
    [
     "C67",
     "x_vars[C68]"
    ],
    [
     "C68",
     "x_vars[C69]"
    ],
    [
     "C69",
     "x_vars[C70]"
    ],
    [
     "C70",
     "x_vars[C71]"
    ],
    [
     "C71",
     "x_vars[C72]"
    ],
    [
     "C72",
     "x_vars[C73]"
    ],
    [
     "C73",
     "x_vars[C74]"
    ],
    [
     "C74",
     "x_vars[C75]"
    ],
    [
     "C75",
     "x_vars[C76]"
    ],
    [
     "C76",
     "x_vars[C77]"
    ],
    [
     "C77",
     "x_vars[C78]"
    ],
    [
     "C78",
     "x_vars[C79]"
    ],
    [
     "C79",
     "x_vars[C80]"
    ],
    [
     "C80",
     "x_vars[C81]"
    ],
    [
     "C81",
     "x_vars[C82]"
    ],
    [
     "C82",
     "x_vars[C83]"
    ],
    [
     "C83",
     "x_vars[C84]"
    ],
    [
     "C84",
     "x_vars[C85]"
    ],
    [
     "C85",
     "x_vars[C86]"
    ],
    [
     "C86",
     "x_vars[C87]"
    ],
    [
     "C87",
     "x_vars[C88]"
    ],
    [
     "C88",
     "x_vars[C89]"
    ],
    [
     "C89",
     "x_vars[C90]"
    ],
    [
     "C90",
     "x_vars[C91]"
    ],
    [
     "C91",
     "x_vars[C92]"
    ],
    [
     "C92",
     "x_vars[C93]"
    ],
    [
     "C93",
     "x_vars[C94]"
    ],
    [
     "C94",
     "x_vars[C95]"
    ],
    [
     "C95",
     "x_vars[C96]"
    ],
    [
     "C96",
     "x_vars[C97]"
    ],
    [
     "C97",
     "x_vars[C98]"
    ],
    [
     "C98",
     "x_vars[C99]"
    ],
    [
     "C99",
     "x_vars[C100]"
    ],
    [
     "C100",
     "x_vars[C101]"
    ],
    [
     "C101",
     "x_vars[C102]"
    ],
    [
     "C102",
     "x_vars[C103]"
    ],
    [
     "C103",
     "x_vars[C104]"
    ],
    [
     "C104",
     "x_vars[C105]"
    ],
    [
     "C105",
     "x_vars[C106]"
    ],
    [
     "C106",
     "x_vars[C107]"
    ],
    [
     "C107",
     "x_vars[C108]"
    ],
    [
     "C108",
     "x_vars[C109]"
    ],
    [
     "C109",
     "x_vars[C110]"
    ],
    [
     "C110",
     "x_vars[C111]"
    ]
   ],
   "primary_mapping": {
    "x[1]": "x_vars[C1]",
    "x[2]": "x_vars[C2]",
    "x[3]": "x_vars[C3]",
    "x[4]": "x_vars[C4]",
    "x[5]": "x_vars[C5]",
    "x[6]": "x_vars[C6]",
    "x[7]": "x_vars[C7]",
    "x[8]": "x_vars[C8]",
    "x[9]": "x_vars[C9]",
    "x[10]": "x_vars[C10]",
    "x[11]": "x_vars[C11]",
    "x[12]": "x_vars[C12]",
    "x[13]": "x_vars[C13]",
    "x[14]": "x_vars[C14]",
    "x[15]": "x_vars[C15]",
    "x[16]": "x_vars[C16]",
    "x[17]": "x_vars[C17]",
    "x[18]": "x_vars[C18]",
    "x[19]": "x_vars[C19]",
    "x[20]": "x_vars[C20]",
    "x[21]": "x_vars[C21]",
    "x[22]": "x_vars[C22]",
    "x[23]": "x_vars[C23]",
    "x[24]": "x_vars[C24]",
    "x[25]": "x_vars[C25]",
    "x[26]": "x_vars[C26]",
    "x[27]": "x_vars[C27]",
    "x[28]": "x_vars[C28]",
    "x[29]": "x_vars[C29]",
    "x[30]": "x_vars[C30]",
    "x[31]": "x_vars[C31]",
    "x[32]": "x_vars[C32]",
    "x[33]": "x_vars[C33]",
    "x[34]": "x_vars[C34]",
    "x[35]": "x_vars[C35]",
    "x[36]": "x_vars[C36]",
    "x[37]": "x_vars[C37]",
    "x[38]": "x_vars[C38]",
    "x[39]": "x_vars[C39]",
    "x[40]": "x_vars[C40]",
    "x[41]": "x_vars[C41]",
    "x[42]": "x_vars[C42]",
    "x[43]": "x_vars[C43]",
    "x[44]": "x_vars[C44]",
    "x[45]": "x_vars[C45]",
    "x[46]": "x_vars[C46]",
    "x[47]": "x_vars[C47]",
    "x[48]": "x_vars[C48]",
    "x[49]": "x_vars[C49]",
    "x[50]": "x_vars[C50]",
    "x[51]": "x_vars[C51]",
    "x[52]": "x_vars[C52]",
    "x[53]": "x_vars[C53]",
    "x[54]": "x_vars[C54]",
    "x[55]": "x_vars[C55]",
    "x[56]": "x_vars[C56]",
    "x[57]": "x_vars[C57]",
    "x[58]": "x_vars[C58]",
    "x[59]": "x_vars[C59]",
    "x[60]": "x_vars[C60]",
    "x[61]": "x_vars[C61]",
    "x[62]": "x_vars[C62]",
    "x[63]": "x_vars[C63]",
    "x[64]": "x_vars[C64]",
    "x[65]": "x_vars[C65]",
    "x[66]": "x_vars[C66]",
    "x[67]": "x_vars[C67]",
    "x[68]": "x_vars[C68]",
    "x[69]": "x_vars[C69]",
    "x[70]": "x_vars[C70]",
    "x[71]": "x_vars[C71]",
    "x[72]": "x_vars[C72]",
    "x[73]": "x_vars[C73]",
    "x[74]": "x_vars[C74]",
    "x[75]": "x_vars[C75]",
    "x[76]": "x_vars[C76]",
    "x[77]": "x_vars[C77]",
    "x[78]": "x_vars[C78]",
    "x[79]": "x_vars[C79]",
    "x[80]": "x_vars[C80]",
    "x[81]": "x_vars[C81]",
    "x[82]": "x_vars[C82]",
    "x[83]": "x_vars[C83]",
    "x[84]": "x_vars[C84]",
    "x[85]": "x_vars[C85]",
    "x[86]": "x_vars[C86]",
    "x[87]": "x_vars[C87]",
    "x[88]": "x_vars[C88]",
    "x[89]": "x_vars[C89]",
    "x[90]": "x_vars[C90]",
    "x[91]": "x_vars[C91]",
    "x[92]": "x_vars[C92]",
    "x[93]": "x_vars[C93]",
    "x[94]": "x_vars[C94]",
    "x[95]": "x_vars[C95]",
    "x[96]": "x_vars[C96]",
    "x[97]": "x_vars[C97]",
    "x[98]": "x_vars[C98]",
    "x[99]": "x_vars[C99]",
    "x[100]": "x_vars[C100]",
    "x[101]": "x_vars[C101]",
    "x[102]": "x_vars[C102]",
    "x[103]": "x_vars[C103]",
    "x[104]": "x_vars[C104]",
    "x[105]": "x_vars[C105]",
    "x[106]": "x_vars[C106]",
    "x[107]": "x_vars[C107]",
    "x[108]": "x_vars[C108]",
    "x[109]": "x_vars[C109]",
    "x[110]": "x_vars[C110]",
    "x[111]": "x_vars[C111]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  },
  "OR-101": {
   "source_code_sha256": "ade618355868ed653c3afaaaacb0b9ad9b04528b1a80cb46654de1776f2f2314",
   "original_solution_sha256": "5df9e97dfcc3bb4dd11f7409f24f6d74efca385c2031c324aa4040608aad7a02",
   "bindings": [
    [
     "C0",
     "x_vars[P1]"
    ],
    [
     "C1",
     "x_vars[P2]"
    ],
    [
     "C2",
     "x_vars[P3]"
    ],
    [
     "C3",
     "x_vars[P4]"
    ],
    [
     "C4",
     "x_vars[P5]"
    ],
    [
     "C5",
     "x_vars[P6]"
    ],
    [
     "C6",
     "x_vars[P7]"
    ],
    [
     "C7",
     "x_vars[P8]"
    ],
    [
     "C8",
     "x_vars[P9]"
    ],
    [
     "C9",
     "x_vars[P10]"
    ],
    [
     "C10",
     "x_vars[P11]"
    ],
    [
     "C11",
     "x_vars[P12]"
    ],
    [
     "C12",
     "x_vars[P13]"
    ],
    [
     "C13",
     "x_vars[P14]"
    ],
    [
     "C14",
     "x_vars[P15]"
    ],
    [
     "C15",
     "x_vars[P16]"
    ],
    [
     "C16",
     "x_vars[P17]"
    ],
    [
     "C17",
     "x_vars[P18]"
    ],
    [
     "C18",
     "x_vars[P19]"
    ],
    [
     "C19",
     "x_vars[P20]"
    ],
    [
     "C20",
     "x_vars[P21]"
    ],
    [
     "C21",
     "x_vars[P22]"
    ],
    [
     "C22",
     "x_vars[P23]"
    ],
    [
     "C23",
     "x_vars[P24]"
    ],
    [
     "C24",
     "x_vars[P25]"
    ],
    [
     "C25",
     "x_vars[P26]"
    ],
    [
     "C26",
     "x_vars[P27]"
    ],
    [
     "C27",
     "x_vars[P28]"
    ],
    [
     "C28",
     "x_vars[P29]"
    ],
    [
     "C29",
     "x_vars[P30]"
    ],
    [
     "C30",
     "x_vars[P31]"
    ],
    [
     "C31",
     "x_vars[P32]"
    ],
    [
     "C32",
     "x_vars[P33]"
    ],
    [
     "C33",
     "x_vars[P34]"
    ],
    [
     "C34",
     "x_vars[P35]"
    ],
    [
     "C35",
     "x_vars[P36]"
    ],
    [
     "C36",
     "x_vars[P37]"
    ],
    [
     "C37",
     "x_vars[P38]"
    ],
    [
     "C38",
     "x_vars[P39]"
    ],
    [
     "C39",
     "x_vars[P40]"
    ],
    [
     "C40",
     "x_vars[P41]"
    ],
    [
     "C41",
     "x_vars[P42]"
    ],
    [
     "C42",
     "x_vars[P43]"
    ],
    [
     "C43",
     "x_vars[P44]"
    ],
    [
     "C44",
     "x_vars[P45]"
    ],
    [
     "C45",
     "x_vars[P46]"
    ],
    [
     "C46",
     "x_vars[P47]"
    ],
    [
     "C47",
     "x_vars[P48]"
    ],
    [
     "C48",
     "x_vars[P49]"
    ],
    [
     "C49",
     "x_vars[P50]"
    ],
    [
     "C50",
     "x_vars[P51]"
    ],
    [
     "C51",
     "x_vars[P52]"
    ],
    [
     "C52",
     "x_vars[P53]"
    ],
    [
     "C53",
     "x_vars[P54]"
    ],
    [
     "C54",
     "x_vars[P55]"
    ],
    [
     "C55",
     "x_vars[P56]"
    ],
    [
     "C56",
     "x_vars[P57]"
    ],
    [
     "C57",
     "x_vars[P58]"
    ],
    [
     "C58",
     "x_vars[P59]"
    ],
    [
     "C59",
     "x_vars[P60]"
    ],
    [
     "C60",
     "x_vars[P61]"
    ],
    [
     "C61",
     "x_vars[P62]"
    ],
    [
     "C62",
     "x_vars[P63]"
    ],
    [
     "C63",
     "x_vars[P64]"
    ],
    [
     "C64",
     "x_vars[P65]"
    ],
    [
     "C65",
     "x_vars[P66]"
    ],
    [
     "C66",
     "x_vars[P67]"
    ],
    [
     "C67",
     "x_vars[P68]"
    ],
    [
     "C68",
     "x_vars[P69]"
    ],
    [
     "C69",
     "x_vars[P70]"
    ],
    [
     "C70",
     "x_vars[P71]"
    ],
    [
     "C71",
     "x_vars[P72]"
    ],
    [
     "C72",
     "x_vars[P73]"
    ],
    [
     "C73",
     "x_vars[P74]"
    ],
    [
     "C74",
     "x_vars[P75]"
    ],
    [
     "C75",
     "x_vars[P76]"
    ],
    [
     "C76",
     "x_vars[P77]"
    ],
    [
     "C77",
     "x_vars[P78]"
    ],
    [
     "C78",
     "x_vars[P79]"
    ],
    [
     "C79",
     "x_vars[P80]"
    ],
    [
     "C80",
     "x_vars[P81]"
    ],
    [
     "C81",
     "x_vars[P82]"
    ],
    [
     "C82",
     "x_vars[P83]"
    ],
    [
     "C83",
     "x_vars[P84]"
    ],
    [
     "C84",
     "x_vars[P85]"
    ],
    [
     "C85",
     "x_vars[P86]"
    ],
    [
     "C86",
     "x_vars[P87]"
    ],
    [
     "C87",
     "x_vars[P88]"
    ],
    [
     "C88",
     "x_vars[P89]"
    ],
    [
     "C89",
     "x_vars[P90]"
    ],
    [
     "C90",
     "x_vars[P91]"
    ],
    [
     "C91",
     "x_vars[P92]"
    ],
    [
     "C92",
     "x_vars[P93]"
    ],
    [
     "C93",
     "x_vars[P94]"
    ],
    [
     "C94",
     "x_vars[P95]"
    ],
    [
     "C95",
     "x_vars[P96]"
    ],
    [
     "C96",
     "x_vars[P97]"
    ],
    [
     "C97",
     "x_vars[P98]"
    ],
    [
     "C98",
     "x_vars[P99]"
    ],
    [
     "C99",
     "x_vars[P100]"
    ],
    [
     "C100",
     "x_vars[P101]"
    ],
    [
     "C101",
     "x_vars[P102]"
    ],
    [
     "C102",
     "x_vars[P103]"
    ],
    [
     "C103",
     "x_vars[P104]"
    ],
    [
     "C104",
     "x_vars[P105]"
    ],
    [
     "C105",
     "x_vars[P106]"
    ],
    [
     "C106",
     "x_vars[P107]"
    ],
    [
     "C107",
     "x_vars[P108]"
    ],
    [
     "C108",
     "x_vars[P109]"
    ],
    [
     "C109",
     "x_vars[P110]"
    ],
    [
     "C110",
     "x_vars[P111]"
    ]
   ],
   "primary_mapping": {
    "x[P1]": "x_vars[P1]",
    "x[P2]": "x_vars[P2]",
    "x[P3]": "x_vars[P3]",
    "x[P4]": "x_vars[P4]",
    "x[P5]": "x_vars[P5]",
    "x[P6]": "x_vars[P6]",
    "x[P7]": "x_vars[P7]",
    "x[P8]": "x_vars[P8]",
    "x[P9]": "x_vars[P9]",
    "x[P10]": "x_vars[P10]",
    "x[P11]": "x_vars[P11]",
    "x[P12]": "x_vars[P12]",
    "x[P13]": "x_vars[P13]",
    "x[P14]": "x_vars[P14]",
    "x[P15]": "x_vars[P15]",
    "x[P16]": "x_vars[P16]",
    "x[P17]": "x_vars[P17]",
    "x[P18]": "x_vars[P18]",
    "x[P19]": "x_vars[P19]",
    "x[P20]": "x_vars[P20]",
    "x[P21]": "x_vars[P21]",
    "x[P22]": "x_vars[P22]",
    "x[P23]": "x_vars[P23]",
    "x[P24]": "x_vars[P24]",
    "x[P25]": "x_vars[P25]",
    "x[P26]": "x_vars[P26]",
    "x[P27]": "x_vars[P27]",
    "x[P28]": "x_vars[P28]",
    "x[P29]": "x_vars[P29]",
    "x[P30]": "x_vars[P30]",
    "x[P31]": "x_vars[P31]",
    "x[P32]": "x_vars[P32]",
    "x[P33]": "x_vars[P33]",
    "x[P34]": "x_vars[P34]",
    "x[P35]": "x_vars[P35]",
    "x[P36]": "x_vars[P36]",
    "x[P37]": "x_vars[P37]",
    "x[P38]": "x_vars[P38]",
    "x[P39]": "x_vars[P39]",
    "x[P40]": "x_vars[P40]",
    "x[P41]": "x_vars[P41]",
    "x[P42]": "x_vars[P42]",
    "x[P43]": "x_vars[P43]",
    "x[P44]": "x_vars[P44]",
    "x[P45]": "x_vars[P45]",
    "x[P46]": "x_vars[P46]",
    "x[P47]": "x_vars[P47]",
    "x[P48]": "x_vars[P48]",
    "x[P49]": "x_vars[P49]",
    "x[P50]": "x_vars[P50]",
    "x[P51]": "x_vars[P51]",
    "x[P52]": "x_vars[P52]",
    "x[P53]": "x_vars[P53]",
    "x[P54]": "x_vars[P54]",
    "x[P55]": "x_vars[P55]",
    "x[P56]": "x_vars[P56]",
    "x[P57]": "x_vars[P57]",
    "x[P58]": "x_vars[P58]",
    "x[P59]": "x_vars[P59]",
    "x[P60]": "x_vars[P60]",
    "x[P61]": "x_vars[P61]",
    "x[P62]": "x_vars[P62]",
    "x[P63]": "x_vars[P63]",
    "x[P64]": "x_vars[P64]",
    "x[P65]": "x_vars[P65]",
    "x[P66]": "x_vars[P66]",
    "x[P67]": "x_vars[P67]",
    "x[P68]": "x_vars[P68]",
    "x[P69]": "x_vars[P69]",
    "x[P70]": "x_vars[P70]",
    "x[P71]": "x_vars[P71]",
    "x[P72]": "x_vars[P72]",
    "x[P73]": "x_vars[P73]",
    "x[P74]": "x_vars[P74]",
    "x[P75]": "x_vars[P75]",
    "x[P76]": "x_vars[P76]",
    "x[P77]": "x_vars[P77]",
    "x[P78]": "x_vars[P78]",
    "x[P79]": "x_vars[P79]",
    "x[P80]": "x_vars[P80]",
    "x[P81]": "x_vars[P81]",
    "x[P82]": "x_vars[P82]",
    "x[P83]": "x_vars[P83]",
    "x[P84]": "x_vars[P84]",
    "x[P85]": "x_vars[P85]",
    "x[P86]": "x_vars[P86]",
    "x[P87]": "x_vars[P87]",
    "x[P88]": "x_vars[P88]",
    "x[P89]": "x_vars[P89]",
    "x[P90]": "x_vars[P90]",
    "x[P91]": "x_vars[P91]",
    "x[P92]": "x_vars[P92]",
    "x[P93]": "x_vars[P93]",
    "x[P94]": "x_vars[P94]",
    "x[P95]": "x_vars[P95]",
    "x[P96]": "x_vars[P96]",
    "x[P97]": "x_vars[P97]",
    "x[P98]": "x_vars[P98]",
    "x[P99]": "x_vars[P99]",
    "x[P100]": "x_vars[P100]",
    "x[P101]": "x_vars[P101]",
    "x[P102]": "x_vars[P102]",
    "x[P103]": "x_vars[P103]",
    "x[P104]": "x_vars[P104]",
    "x[P105]": "x_vars[P105]",
    "x[P106]": "x_vars[P106]",
    "x[P107]": "x_vars[P107]",
    "x[P108]": "x_vars[P108]",
    "x[P109]": "x_vars[P109]",
    "x[P110]": "x_vars[P110]",
    "x[P111]": "x_vars[P111]"
   },
   "mapping_basis": "revalidated_entity_keys_container_aliases"
  }
 }
}
""")


EMBEDDED_INPUT_SHA256 = json.loads(r"""
{
 "COHORTS": "835f28568b8687221c3fd3c1d0213ab49bc5f76a00e89443cc7d2e4e2bf9efcb",
 "DATA_SHA256": "4b938580df768f05fbfe37d2d320e819728d5b7fb610b392cb2911033b118668",
 "REFERENCE_CERTIFICATES": "ae20c417b98d56b5634b00fb7a72034dab48c2ba4faffe203066e11e3f6862d5",
 "IDENTITY_ARTIFACTS": "c7b7318b3841cd8323fb6a96373ee541258ce826911005c794515edd0eb612c6"
}
""")


for name, expected in EMBEDDED_INPUT_SHA256.items():
    value = globals()[name]
    actual = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert actual == expected, f'Embedded input annotations changed: {name}'
assert set(METHODS) <= set(COHORTS)
assert all(set(IDENTITY_ARTIFACTS[method]) == EXPECTED_IDS for method in METHODS)
display(pd.DataFrame.from_dict(COHORTS, orient='index')[['relative_directory', 'source_commit']])




from dataclasses import dataclass, field
from pathlib import Path
import math
import re
import tempfile
from typing import Any, Callable

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


MARKDOWN_MODEL_IDS = {73, 80, 83, 84, 85, 87, 90, 91, 92, 93, 95, 97, 100}





_LP_AUXILIARY_VARIABLES: dict[int, set[str]] = {
    82: {"U[1]", "U[2]", "U[3]"},
}

_SECTION_RE = re.compile(
    r"(?im)^\s*(Bounds|Binaries|Binary|Generals|General|Integers|Integer|End)\s*$"
)
_INTEGER_SECTIONS = {"generals", "general", "integers", "integer"}
_BINARY_SECTIONS = {"binaries", "binary"}
_NUMBER_RE = re.compile(
    r"Optimal\s+objective\s*[:=]?\s*"
    r"([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)",
    re.I,
)


@dataclass
class GroundTruthBundle:
    benchmark_index: int
    model: gp.Model
    source_kind: str
    published_objective: float | None
    optimum: float | None = None
    solver_status: int | None = None
    auxiliary_variables: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_published_objective(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return None if math.isnan(number) else number
    text = str(value).strip()
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    if match:
        return float(match.group(1).replace(",", ""))
    cleaned = re.sub(r"[,\s$¥€£]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _segment_declared_names(payload: str, candidates: list[str]) -> tuple[list[str], str]:
    compact = re.sub(r"\s+", "", payload)
    names = sorted(set(candidates), key=lambda value: (-len(value), value))
    found: list[str] = []
    unknown: list[str] = []
    cursor = 0
    while cursor < len(compact):
        match = next((name for name in names if compact.startswith(name, cursor)), None)
        if match is not None:
            found.append(match)
            cursor += len(match)
        else:
            unknown.append(compact[cursor])
            cursor += 1
    return found, "".join(unknown)


def repair_lp_variable_types(model: gp.Model, text: str) -> dict[str, Any]:


    matches = list(_SECTION_RE.finditer(text))
    first_type_offset = min(
        (
            match.start()
            for match in matches
            if match.group(1).strip().casefold()
            in _INTEGER_SECTIONS | _BINARY_SECTIONS
        ),
        default=len(text),
    )
    body = text[:first_type_offset]
    variables = list(model.getVars())
    candidates = [var.VarName for var in variables if var.VarName in body]
    assignments: dict[str, str] = {}
    unknown_fragments: list[str] = []

    for index, match in enumerate(matches):
        section = match.group(1).strip().casefold()
        if section not in _INTEGER_SECTIONS | _BINARY_SECTIONS:
            continue
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        names, unknown = _segment_declared_names(text[match.end() : stop], candidates)
        if unknown:


            unknown_fragments.append(unknown[:500])
        target_type = GRB.INTEGER if section in _INTEGER_SECTIONS else GRB.BINARY
        for name in names:
            assignments[name] = target_type

    for name, variable_type in assignments.items():
        variable = model.getVarByName(name)
        if variable is not None:
            variable.VType = variable_type

    ghosts = [
        variable
        for variable in variables
        if variable.VarName not in body
        and model.getCol(variable).size() == 0
        and abs(variable.Obj) == 0.0
    ]
    ghost_names = [variable.VarName for variable in ghosts]
    if ghosts:
        model.remove(ghosts)
    model.update()
    return {
        "repaired_variable_count": len(assignments),
        "repaired_variables": sorted(assignments),
        "removed_ghost_count": len(ghost_names),
        "removed_ghost_variables": ghost_names,
        "unparsed_type_fragments": unknown_fragments,
    }


def _quiet_model(name: str) -> gp.Model:
    model = gp.Model(name)
    model.Params.OutputFlag = 0
    return model


def _csv(repo_root: Path, relative: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(repo_root / relative, **kwargs)


def _build_73(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv"
    data = _csv(repo_root, relative).set_index("Equipment / Cost")
    equipment_products = {
        "A1": ["I", "II"],
        "A2": ["I", "II", "III"],
        "B1": ["I", "II"],
        "B2": ["I", "III"],
        "B3": ["I"],
    }
    column = {"I": "Product I", "II": "Product II", "III": "Product III"}
    price = {p: float(data.loc["Unit Price (yuan/unit)", column[p]]) for p in column}
    raw = {p: float(data.loc["Raw Material Cost (yuan/unit)", column[p]]) for p in column}
    model = _quiet_model("gt_073_mixture")
    x = {
        (equipment, product): model.addVar(lb=0.0, name=f"x[{equipment},{product}]")
        for equipment, products in equipment_products.items()
        for product in products
    }
    production = {
        "I": x["A1", "I"] + x["A2", "I"],
        "II": x["A1", "II"] + x["A2", "II"],
        "III": x["A2", "III"],
    }
    revenue_minus_raw = gp.quicksum((price[p] - raw[p]) * production[p] for p in column)
    equipment_cost = gp.quicksum(
        float(data.loc[equipment, "Equipment Cost at Full Load (yuan)"])
        / float(data.loc[equipment, "Available Equipment Operating Time"])
        * gp.quicksum(
            float(data.loc[equipment, column[product]]) * x[equipment, product]
            for product in products
        )
        for equipment, products in equipment_products.items()
    )
    model.setObjective(revenue_minus_raw - equipment_cost, GRB.MAXIMIZE)
    for equipment, products in equipment_products.items():
        model.addConstr(
            gp.quicksum(
                float(data.loc[equipment, column[product]]) * x[equipment, product]
                for product in products
            )
            <= float(data.loc[equipment, "Available Equipment Operating Time"]),
            name=f"capacity[{equipment}]",
        )
    model.addConstr(
        production["I"] == x["B1", "I"] + x["B2", "I"] + x["B3", "I"],
        name="flow[I]",
    )
    model.addConstr(production["II"] == x["B1", "II"], name="flow[II]")
    model.addConstr(production["III"] == x["B2", "III"], name="flow[III]")
    return model, set()


def _build_80(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture6/parameters.csv"
    data = _csv(repo_root, relative)
    trucks = [int(value) for value in data["truck_id"]]
    periods = [1, 2, 3, 4]
    q = dict(zip(trucks, data["Q"].astype(float)))
    startup = dict(zip(trucks, data["S"].astype(float)))
    cost = dict(zip(trucks, data["C"].astype(float)))
    demand = {t: float(data[f"d{t}"].iloc[0]) for t in periods}
    model = _quiet_model("gt_080_truck_schedule")
    w = model.addVars(trucks, periods, lb=0.0, name="w")
    y = model.addVars(trucks, periods, vtype=GRB.BINARY, name="y")
    u = model.addVars(trucks, [2, 3, 4], vtype=GRB.BINARY, name="u")
    model.setObjective(
        gp.quicksum(cost[i] * w[i, t] for i in trucks for t in periods)
        + gp.quicksum(startup[i] * y[i, 1] for i in trucks)
        + gp.quicksum(startup[i] * u[i, t] for i in trucks for t in [2, 3, 4]),
        GRB.MINIMIZE,
    )
    for t in periods:
        model.addConstr(gp.quicksum(w[i, t] for i in trucks) >= demand[t], name=f"demand[{t}]")
        model.addConstr(
            gp.quicksum(w[i, t] for i in trucks)
            <= 0.9 * gp.quicksum(q[i] * y[i, t] for i in trucks),
            name=f"reserve[{t}]",
        )
    for i in trucks:
        for t in periods:
            model.addConstr(w[i, t] <= q[i] * y[i, t], name=f"link[{i},{t}]")
        for t in [2, 3, 4]:
            model.addConstr(u[i, t] >= y[i, t] - y[i, t - 1], name=f"startup_lb[{i},{t}]")
            model.addConstr(u[i, t] <= y[i, t], name=f"startup_on[{i},{t}]")
            model.addConstr(u[i, t] <= 1 - y[i, t - 1], name=f"startup_off[{i},{t}]")
            model.addConstr(w[i, t] - w[i, t - 1] <= 300, name=f"ramp_up[{i},{t}]")
            model.addConstr(w[i, t - 1] - w[i, t] <= 300, name=f"ramp_down[{i},{t}]")
        model.addConstr(y[i, 1] <= y[i, 2], name=f"min_runtime_initial[{i}]")
        for t in [2, 3]:
            model.addConstr(y[i, t] + y[i, t + 1] >= 2 * u[i, t], name=f"min_runtime[{i},{t}]")
            model.addConstr(
                y[i, t - 1] - y[i, t] <= 1 - y[i, t + 1],
                name=f"cooldown[{i},{t}]",
            )
        model.addConstr(u[i, 4] == 0, name=f"no_start_4[{i}]")
    return model, {f"u[{i},{t}]" for i in trucks for t in [2, 3, 4]}


def _build_83(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv"
    frame = _csv(repo_root, relative)
    task_columns = list(frame.columns[1:])
    numeric = frame[task_columns].apply(pd.to_numeric, errors="coerce")
    keep = numeric.notna().all(axis=1)
    hours = numeric.loc[keep].to_numpy(dtype=float)
    workers = list(range(1, hours.shape[0] + 1))
    model = _quiet_model("gt_083_assignment")
    x = model.addVars(workers, task_columns, vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(hours[i - 1, j] * x[i, task] for i in workers for j, task in enumerate(task_columns)),
        GRB.MINIMIZE,
    )
    for task in task_columns:
        model.addConstr(gp.quicksum(x[i, task] for i in workers) == 1, name=f"task[{task}]")
    for i in workers:
        model.addConstr(gp.quicksum(x[i, task] for task in task_columns) <= 1, name=f"worker[{i}]")
    model.addConstr(gp.quicksum(x.values()) == 10, name="ten_assignments")
    return model, set()


def _build_84(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/Others_example/Others6/18.csv"
    frame = _csv(repo_root, relative)
    instructions = frame.iloc[0, 1:].astype(float).to_numpy()
    processors = [1, 2, 3]
    speeds = {1: 1.33, 2: 2.0, 3: 2.66}
    tasks = list(range(1, len(instructions) + 1))
    model = _quiet_model("gt_084_parallel_machine")
    x = model.addVars(processors, tasks, vtype=GRB.BINARY, name="x")
    cmax = model.addVar(lb=0.0, name="Cmax")
    model.setObjective(cmax, GRB.MINIMIZE)
    for task in tasks:
        model.addConstr(gp.quicksum(x[m, task] for m in processors) == 1, name=f"assign[{task}]")
    for machine in processors:
        model.addConstr(
            gp.quicksum(instructions[task - 1] / speeds[machine] * x[machine, task] for task in tasks)
            <= cmax,
            name=f"load[{machine}]",
        )
    return model, {"Cmax"}


def _build_85(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/Others_example/Others7/20.csv"
    raw = _csv(repo_root, relative, index_col=0).apply(pd.to_numeric, errors="coerce").to_numpy()
    n = raw.shape[0]
    distance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = raw[i, j] if not np.isnan(raw[i, j]) else raw[j, i]
            distance[i, j] = float(value)
    cities = list(range(1, n + 1))
    model = _quiet_model("gt_085_tsp")
    arcs = [(i, j) for i in cities for j in cities if i != j]
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    order = model.addVars(cities[1:], lb=2.0, ub=float(n), name="u")
    model.setObjective(
        gp.quicksum(distance[i - 1, j - 1] * x[i, j] for i, j in arcs),
        GRB.MINIMIZE,
    )
    for i in cities:
        model.addConstr(gp.quicksum(x[i, j] for j in cities if j != i) == 1, name=f"out[{i}]")
    for j in cities:
        model.addConstr(gp.quicksum(x[i, j] for i in cities if i != j) == 1, name=f"in[{j}]")
    for i in cities[1:]:
        for j in cities[1:]:
            if i != j:
                model.addConstr(order[i] - order[j] + n * x[i, j] <= n - 1, name=f"mtz[{i},{j}]")
    return model, {f"u[{city}]" for city in cities[1:]}


def _build_87(repo_root: Path) -> tuple[gp.Model, set[str]]:
    base = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture9"
    attributes = _csv(repo_root, f"{base}/36-1.csv", index_col=0)
    activation = _csv(repo_root, f"{base}/36-2.csv", index_col=0)
    batches = _csv(repo_root, f"{base}/36-3.csv", index_col=0)
    products = list(range(1, 81))
    demand = attributes.loc["Maximum Demand (100 kg units)"].astype(float).to_numpy()
    selling = attributes.loc["Selling Price ($/100 kg)"].astype(float).to_numpy()
    production = attributes.loc["Production Cost ($/100 kg)"].astype(float).to_numpy()
    quota = attributes.loc["Production Quota (max per day)"].astype(float).to_numpy()
    fixed = activation.iloc[0].astype(float).to_numpy()
    minimum = batches.iloc[0].astype(float).to_numpy()
    model = _quiet_model("gt_087_fixed_charge_production")
    x = model.addVars(products, lb=0.0, vtype=GRB.INTEGER, name="x")
    y = model.addVars(products, vtype=GRB.BINARY, name="y")
    model.setObjective(
        gp.quicksum((selling[i - 1] - production[i - 1]) * x[i] - fixed[i - 1] * y[i] for i in products),
        GRB.MAXIMIZE,
    )
    for i in products:
        model.addConstr(x[i] <= demand[i - 1], name=f"demand[{i}]")
        model.addConstr(x[i] <= 22.0 * quota[i - 1] * y[i], name=f"activation_ub[{i}]")
        model.addConstr(x[i] >= minimum[i - 1] * y[i], name=f"activation_lb[{i}]")
    return model, set()


def _build_90(repo_root: Path) -> tuple[gp.Model, set[str]]:
    base = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10"
    products_frame = _csv(repo_root, f"{base}/factory_products_100.csv")
    capacities_frame = _csv(repo_root, f"{base}/resources_capacities.csv")
    products = list(range(1, len(products_frame) + 1))
    batch = products_frame["batch_size_units"].astype(float).to_numpy()
    profit = products_frame["profit_per_unit"].astype(float).to_numpy()
    demand = products_frame["upper_demand_units"].astype(float).to_numpy()
    capacities = dict(zip(capacities_frame["resource"], capacities_frame["capacity"].astype(float)))
    model = _quiet_model("gt_090_batch_production")
    x = model.addVars(products, lb=0.0, vtype=GRB.INTEGER, name="x")
    model.setObjective(gp.quicksum(batch[i - 1] * profit[i - 1] * x[i] for i in products), GRB.MAXIMIZE)
    for resource, column in [("R1", "r1_per_unit"), ("R2", "r2_per_unit"), ("R3", "r3_per_unit")]:
        usage = products_frame[column].astype(float).to_numpy()
        model.addConstr(
            gp.quicksum(batch[i - 1] * usage[i - 1] * x[i] for i in products) <= capacities[resource],
            name=f"resource[{resource}]",
        )
    for i in products:
        model.addConstr(batch[i - 1] * x[i] <= demand[i - 1], name=f"demand[{i}]")
    return model, set()


def _build_91(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv"
    frame = _csv(repo_root, relative)
    frame = frame.loc[frame["discipline"].astype(str).str.casefold() == "operations research"].reset_index(drop=True)
    courses = list(range(1, len(frame) + 1))
    model = _quiet_model("gt_091_course_selection")
    x = model.addVars(courses, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(float(frame.loc[i - 1, "interest_points"]) * x[i] for i in courses), GRB.MAXIMIZE)
    model.addConstr(
        gp.quicksum(float(frame.loc[i - 1, "credits"]) * x[i] for i in courses) <= 20,
        name="credit_limit",
    )
    return model, set()


def _build_92(repo_root: Path) -> tuple[gp.Model, set[str]]:
    base = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12"
    sources = _csv(repo_root, f"{base}/expanded_sources.csv")
    destinations = _csv(repo_root, f"{base}/expanded_destinations.csv")
    costs = _csv(repo_root, f"{base}/expanded_cost_matrix.csv", index_col=0)
    origins = list(range(1, len(sources) + 1))
    sinks = list(range(1, len(destinations) + 1))
    supply = sources["supply_units"].astype(float).to_numpy()
    demand = destinations["demand_units"].astype(float).to_numpy()
    cost = costs.astype(float).to_numpy()
    model = _quiet_model("gt_092_partial_truck_transport")
    cargo = model.addVars(origins, sinks, lb=0.0, name="y")
    trucks = model.addVars(origins, sinks, lb=0.0, vtype=GRB.INTEGER, name="x")
    model.setObjective(gp.quicksum(cost[i - 1, j - 1] * cargo[i, j] for i in origins for j in sinks), GRB.MINIMIZE)
    for j in sinks:
        model.addConstr(gp.quicksum(cargo[i, j] for i in origins) == demand[j - 1], name=f"demand[{j}]")
    for i in origins:
        model.addConstr(gp.quicksum(cargo[i, j] for j in sinks) <= supply[i - 1], name=f"supply[{i}]")
    for i in origins:
        for j in sinks:
            model.addConstr(cargo[i, j] <= 10.0 * trucks[i, j], name=f"truck_capacity[{i},{j}]")
    return model, set()


def _build_93(repo_root: Path) -> tuple[gp.Model, set[str]]:
    relative = "Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv"
    frame = _csv(repo_root, relative, index_col=0)
    machines = list(range(1, len(frame) + 1))
    tasks = list(frame.columns)
    costs = frame.astype(float).to_numpy()
    model = _quiet_model("gt_093_assignment")
    x = model.addVars(machines, tasks, vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(costs[i - 1, j] * x[i, task] for i in machines for j, task in enumerate(tasks)),
        GRB.MINIMIZE,
    )
    for i in machines:
        model.addConstr(gp.quicksum(x[i, task] for task in tasks) == 1, name=f"machine[{i}]")
    for task in tasks:
        model.addConstr(gp.quicksum(x[i, task] for i in machines) == 1, name=f"task[{task}]")
    return model, set()


def _build_95(repo_root: Path) -> tuple[gp.Model, set[str]]:
    base = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14"
    products_frame = _csv(repo_root, f"{base}/product_resources.csv")
    limits_frame = _csv(repo_root, f"{base}/resource_limits.csv")
    limits = dict(zip(limits_frame["Resource"], limits_frame["MonthlyLimit"].astype(float)))
    products = list(range(1, len(products_frame) + 1))
    model = _quiet_model("gt_095_byproduct_production")
    x = model.addVars(products, lb=0.0, name="x")
    sold = model.addVar(lb=0.0, name="s")
    profit = products_frame["Profit"].astype(float).to_numpy()


    model.setObjective(
        gp.quicksum(profit[i - 1] * x[i] for i in products) - 1000.0 * x[3] + 500.0 * sold,
        GRB.MAXIMIZE,
    )
    for column, key in [("LaborHours", "LaborHours"), ("MaterialA", "MaterialA"), ("MaterialB", "MaterialB")]:
        coefficients = products_frame[column].astype(float).to_numpy()
        model.addConstr(
            gp.quicksum(coefficients[i - 1] * x[i] for i in products) <= limits[key],
            name=f"resource[{key}]",
        )
    model.addConstr(sold <= 1500.0, name="sales_cap")
    model.addConstr(sold <= 5.0 * x[3], name="byproduct_balance")
    return model, set()


def _build_97(repo_root: Path) -> tuple[gp.Model, set[str]]:
    base = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16"
    options = _csv(repo_root, f"{base}/OptionCharacteristics.csv")
    references = _csv(repo_root, f"{base}/Option_AssetReferenceMatrix.csv", index_col=0)
    indices = list(range(1, len(options) + 1))
    multiplicity = references.astype(float).sum(axis=1).to_numpy()
    model = _quiet_model("gt_097_option_hedging")
    x = {
        i: model.addVar(
            lb=float(options.loc[i - 1, "MaxShort"]),
            ub=float(options.loc[i - 1, "MaxLong"]),
            vtype=GRB.INTEGER,
            name=f"x[{i}]",
        )
        for i in indices
    }
    z = model.addVars(indices, lb=0.0, name="z")
    model.setObjective(gp.quicksum(float(options.loc[i - 1, "Cost"]) * z[i] for i in indices), GRB.MINIMIZE)
    for i in indices:
        model.addConstr(z[i] >= x[i], name=f"abs_pos[{i}]")
        model.addConstr(z[i] >= -x[i], name=f"abs_neg[{i}]")
    for greek, initial, tolerance in [("Delta", 0.25, 0.06), ("Gamma", 0.08, 0.05), ("Vega", 0.17, 0.07)]:
        exposure = gp.quicksum(
            float(options.loc[i - 1, greek]) * multiplicity[i - 1] * x[i]
            for i in indices
        )
        model.addConstr(exposure <= tolerance - initial, name=f"{greek}_upper")
        model.addConstr(exposure >= -tolerance - initial, name=f"{greek}_lower")
    return model, {f"z[{i}]" for i in indices}


def _build_100(repo_root: Path) -> tuple[gp.Model, set[str]]:
    base = "Test_Dataset/Large-scale-or/Others_example/Others8"
    times = _csv(repo_root, f"{base}/processing_time_unit.csv", index_col=0)
    prices = _csv(repo_root, f"{base}/unit_price.csv", index_col=0)
    hours = _csv(repo_root, f"{base}/total_working_hours.csv")
    components = list(range(1, times.shape[1] + 1))
    price = prices.iloc[:, 0].astype(float).to_numpy()
    capacities = dict(zip(hours["workshop"], hours["total_hours"].astype(float)))
    model = _quiet_model("gt_100_flexible_flow_shop")
    x = model.addVars(components, lb=0.0, name="x")
    model.setObjective(gp.quicksum(price[j - 1] * x[j] for j in components), GRB.MAXIMIZE)
    for workshop, row in times.iterrows():
        coefficients = row.astype(float).to_numpy()
        model.addConstr(
            gp.quicksum(coefficients[j - 1] * x[j] for j in components) <= capacities[workshop],
            name=f"workshop[{workshop}]",
        )
    return model, set()


_BUILDERS: dict[int, Callable[[Path], tuple[gp.Model, set[str]]]] = {
    73: _build_73,
    80: _build_80,
    83: _build_83,
    84: _build_84,
    85: _build_85,
    87: _build_87,
    90: _build_90,
    91: _build_91,
    92: _build_92,
    93: _build_93,
    95: _build_95,
    97: _build_97,
    100: _build_100,
}


def _read_lp_model(text: str) -> tuple[gp.Model, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
        handle.flush()
        model = gp.read(handle.name)
    model.Params.OutputFlag = 0
    metadata = repair_lp_variable_types(model, text)
    return model, metadata


def load_ground_truth(
    repo_root: str | Path,
    row: pd.Series,
    benchmark_index: int,
    *,
    solve: bool = True,
    time_limit_s: float = 300.0,
) -> GroundTruthBundle:


    root = Path(repo_root).resolve()
    published = parse_published_objective(row.get("Label-objective"))
    if benchmark_index in _BUILDERS:
        model, auxiliary = _BUILDERS[benchmark_index](root)
        source_kind = "deterministic_markdown_reconstruction"
        metadata: dict[str, Any] = {"constructor_id": benchmark_index}
    else:
        model, metadata = _read_lp_model(str(row["Label-model"]))
        auxiliary = set(_LP_AUXILIARY_VARIABLES.get(benchmark_index, set()))
        missing_auxiliary = auxiliary.difference(
            variable.VarName for variable in model.getVars()
        )
        if missing_auxiliary:
            raise ValueError(
                f"declared LP auxiliaries missing for row {benchmark_index}: "
                f"{sorted(missing_auxiliary)}"
            )
        if auxiliary:
            metadata["declared_lp_auxiliary_variables"] = sorted(auxiliary)
        source_kind = "repaired_label_lp"

    model.update()
    bundle = GroundTruthBundle(
        benchmark_index=benchmark_index,
        model=model,
        source_kind=source_kind,
        published_objective=published,
        auxiliary_variables=auxiliary,
        metadata=metadata,
    )
    if solve:
        model.Params.TimeLimit = float(time_limit_s)
        model.optimize()
        bundle.solver_status = int(model.Status)
        bundle.optimum = float(model.ObjVal) if model.SolCount > 0 else None
        bundle.metadata.update(
            num_variables=int(model.NumVars),
            num_constraints=int(model.NumConstrs),
            num_integer=int(sum(v.VType in {GRB.INTEGER, GRB.BINARY, GRB.SEMIINT} for v in model.getVars())),
            model_sense="min" if model.ModelSense == GRB.MINIMIZE else "max",
            label_relative_error=(
                abs(bundle.optimum - published) / max(1.0, abs(published))
                if bundle.optimum is not None and published is not None
                else None
            ),
        )
    return bundle


def _derive_auxiliary_values(
    bundle: GroundTruthBundle,
    primary_values: dict[str, float],
    auxiliary_names: set[str],
) -> tuple[dict[str, float], dict[str, Any]]:












    if not auxiliary_names:
        return {}, {"auxiliary_completion": "not_needed"}
    completion = bundle.model.copy()
    completion.Params.OutputFlag = 0
    completion.Params.TimeLimit = 30.0
    completion.update()
    for name, value in primary_values.items():
        variable = completion.getVarByName(name)
        if variable is None:
            continue
        variable.VType = GRB.CONTINUOUS
        variable.LB = min(variable.LB, value)
        variable.UB = max(variable.UB, value)
        completion.addConstr(variable == value, name=f"__transfer_fix__{name}")
    completion.optimize()
    relaxed = False
    if completion.SolCount == 0:
        relaxed = True
        completion.feasRelaxS(0, False, False, True)
        completion.optimize()
    values: dict[str, float] = {}
    if completion.SolCount > 0:
        for name in auxiliary_names:
            variable = completion.getVarByName(name)
            if variable is not None:
                values[name] = float(variable.X)
    details = {
        "auxiliary_completion": "least_row_relaxation" if relaxed else "ground_truth_objective_completion",
        "auxiliary_completion_status": int(completion.Status),
        "auxiliary_completion_count": len(values),
    }
    completion.dispose()
    return values, details


def binary_fraction(value):
    value = float(value)
    assert math.isfinite(value)
    return Fraction.from_float(value)


def residual(lhs, rhs, sense):
    if sense == "<":
        return max(Fraction(0), lhs - rhs)
    if sense == ">":
        return max(Fraction(0), rhs - lhs)
    assert sense == "="
    return abs(lhs - rhs)


def check_model(model, values):
    model.update()
    assert not any((model.NumQNZs, model.NumQConstrs, model.NumGenConstrs, model.NumSOS))
    variables = {v.VarName: v for v in model.getVars()}
    assert set(values) == set(variables)
    x = {name: binary_fraction(value) for name, value in values.items()}
    checks = []
    for row in model.getConstrs():
        expr = model.getRow(row)
        lhs = sum((binary_fraction(expr.getCoeff(k)) * x[expr.getVar(k).VarName]
                   for k in range(expr.size())), Fraction(0))
        rhs = binary_fraction(row.RHS)
        checks.append(dict(kind="constraint", name=row.ConstrName, sense=row.Sense,
                           lhs=lhs, rhs=rhs, violation=residual(lhs, rhs, row.Sense),
                           historical_tolerance=binary_fraction(1e-6 + 1e-6 * max(1, abs(float(rhs))))))
    for name, var in variables.items():
        assert var.VType in {"C", "I", "B"}
        for side, bound in (("lower", float(var.LB)), ("upper", float(var.UB))):
            if abs(bound) >= gp.GRB.INFINITY / 2:
                continue
            b = binary_fraction(bound)
            violation = max(Fraction(0), b - x[name] if side == "lower" else x[name] - b)
            checks.append(dict(kind="bound", name=name, side=side, value=x[name], bound=b,
                               violation=violation,
                               historical_tolerance=binary_fraction(1e-6 + 1e-6 * max(1, abs(bound)))))
        if var.VType in {"I", "B"}:
            checks.append(dict(kind="integrality", name=name, value=x[name],
                               violation=abs(x[name] - round(x[name])),
                               historical_tolerance=binary_fraction(1e-5)))
        if var.VType == "B":
            checks.append(dict(kind="binary_domain", name=name, value=x[name],
                               violation=max(Fraction(0), -x[name], x[name] - 1),
                               historical_tolerance=binary_fraction(2e-6)))
    objective = binary_fraction(model.ObjCon) + sum(
        (binary_fraction(v.Obj) * x[name] for name, v in variables.items()), Fraction(0))
    return objective, checks


REPLAY_WORKER_SOURCE = 'from __future__ import annotations\nimport ast\nimport contextlib\nimport io\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport sys\nimport traceback\nimport gurobipy as gp\nimport numpy as np\nimport pandas as pd\n\n_BLOCKED_IMPORT_ROOTS = {\n    "cffi",\n    "ctypes",\n    "http",\n    "openai",\n    "requests",\n    "shutil",\n    "socket",\n    "subprocess",\n    "urllib",\n}\n\n\n_BLOCKED_CALLS = {\n    "compile",\n    "eval",\n    "exec",\n    "__import__",\n    "os.remove",\n    "os.unlink",\n    "os.rmdir",\n    "os.removedirs",\n    "os.rename",\n    "os.replace",\n    "os.system",\n    "os.popen",\n    "pathlib.Path.unlink",\n    "pathlib.Path.rmdir",\n    "pathlib.Path.rename",\n    "pathlib.Path.replace",\n    "pathlib.Path.write_text",\n    "pathlib.Path.write_bytes",\n}\n\n\n_FORBIDDEN_READ_FRAGMENTS = {\n    "large-scale-or-101.csv",\n    "results/benchmark_test/",\n    "results/large-scale-or-results-",\n    "/runs/",\n}\n\n\ndef _call_name(node: ast.AST) -> str:\n    parts: list[str] = []\n    while isinstance(node, ast.Attribute):\n        parts.append(node.attr)\n        node = node.value\n    if isinstance(node, ast.Name):\n        parts.append(node.id)\n    return ".".join(reversed(parts))\n\n\ndef scan_code(code: str) -> list[str]:\n    try:\n        tree = ast.parse(code)\n    except SyntaxError as exc:\n        return [f"SyntaxError: {exc}"]\n    issues: list[str] = []\n    for node in ast.walk(tree):\n        if isinstance(node, ast.Constant) and isinstance(node.value, str):\n            normalized = node.value.replace("\\\\", "/").casefold()\n            for fragment in _FORBIDDEN_READ_FRAGMENTS:\n                if fragment in normalized:\n                    issues.append(f"blocked evaluation-artifact path: {fragment}")\n        if isinstance(node, ast.Import):\n            for alias in node.names:\n                if alias.name.split(".")[0] in _BLOCKED_IMPORT_ROOTS:\n                    issues.append(f"blocked import: {alias.name}")\n        elif isinstance(node, ast.ImportFrom):\n            root = (node.module or "").split(".")[0]\n            if root in _BLOCKED_IMPORT_ROOTS:\n                issues.append(f"blocked import: {node.module}")\n        elif isinstance(node, ast.Call):\n            name = _call_name(node.func)\n            if name in _BLOCKED_CALLS or name.startswith(("subprocess.", "shutil.")):\n                issues.append(f"blocked call: {name}")\n            if name == "open":\n                mode: str | None = None\n                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):\n                    mode = str(node.args[1].value)\n                for keyword in node.keywords:\n                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):\n                        mode = str(keyword.value.value)\n                if mode and any(flag in mode for flag in "wax+"):\n                    issues.append(f"blocked write-mode open: {mode}")\n    return sorted(set(issues))\n\n\ndef flatten(value, keys=(), depth=0):\n    if depth > 5:\n        return []\n    if isinstance(value, gp.Var):\n        return [(value, keys)]\n    result = []\n    if isinstance(value, dict):\n        for key, item in value.items():\n            result.extend(flatten(item, keys + (tuple(key) if isinstance(key, tuple) else (key,)), depth + 1))\n    elif isinstance(value, (list, tuple)):\n        for position, item in enumerate(value):\n            result.extend(flatten(item, keys + (position,), depth + 1))\n    return result\n\n\ndef replay(code, workdir):\n    issues = scan_code(code)\n    if issues:\n        return dict(ok=False, error=\'unsafe_source_rejected\', scan_issues=issues)\n    tree = ast.parse(code)\n    constructors = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)\n        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)\n        and node.value.func.attr in {\'addVars\', \'addVar\'}\n        for target in node.targets if isinstance(target, ast.Name)}\n    models, snapshots = [], []\n    optimize = gp.Model.optimize\n    write = gp.Model.write\n\n    def observe_and_solve(model, *args, **kwargs):\n        assert not models, \'Only one optimization per saved program is allowed\'\n        models.append(model)\n        model.update()\n        frame = sys._getframe().f_back\n        while frame is not None:\n            if frame.f_code.co_filename == \'<fixed-code-replay>\':\n                for symbol, value in list(frame.f_locals.items()):\n                    if symbol.startswith(\'_\') or symbol in {\'m\', \'model\', \'CSVQA_DATA\'}:\n                        continue\n                    items = flatten(value)\n                    if items:\n                        snapshots.append((frame.f_code.co_name, symbol, items))\n            frame = frame.f_back\n        model.Params.Seed = 0\n        model.Params.Threads = 1\n        model.Params.TimeLimit = 180\n        model.Params.MIPGap = 1e-6\n        optimize(model, *args, **kwargs)\n\n    def prohibit_write(*args, **kwargs):\n        raise PermissionError(\'Generated model writes are disabled\')\n\n    gp.Model.optimize, gp.Model.write = observe_and_solve, prohibit_write\n    namespace = {\'__name__\': \'__main__\'}\n    output = io.StringIO()\n    os.chdir(workdir)\n    try:\n        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):\n            exec(compile(code, \'<fixed-code-replay>\', \'exec\'), namespace)\n        assert len(models) == 1\n        model = models[0]\n        if model.SolCount <= 0:\n            return dict(ok=False, error=\'generated_model_has_no_incumbent\',\n                solver_status=int(model.Status), real_generated_optimizations=len(models),\n                code_output_tail=output.getvalue()[-1200:])\n        assert model.Status == gp.GRB.OPTIMAL, (\'not_optimal\', model.Status)\n        assert not any((model.NumQNZs, model.NumQConstrs, model.NumGenConstrs, model.NumSOS))\n        bindings = {}\n        for scope, symbol, items in snapshots:\n            family = re.sub(r\'[^A-Za-z0-9_]\', \'_\', symbol).strip(\'_\')\n            for variable, keys in items:\n                labels = [str(key) for key in keys]\n                semantic = family + \'[\' + \',\'.join(labels) + \']\' if labels else family\n                identity = dict(scope=scope, family=family, keys=labels, semantic_name=semantic)\n                rank = (0 if len(items) > 1 else 1, 0 if symbol in constructors else 1,\n                        0 if labels else 1, -len(items), symbol, scope)\n                bindings.setdefault(variable.VarName, []).append((rank, identity))\n        identities, variables = [], []\n        for variable in model.getVars():\n            raw = variable.VarName\n            options = sorted(bindings.get(raw, []), key=lambda pair: pair[0])\n            binding = options[0][1] if options else None\n            explicit = not re.fullmatch(r\'C\\d+\', raw)\n            assert explicit or binding is not None\n            semantic = raw if explicit else binding[\'semantic_name\']\n            identities.append(dict(original_name=raw, semantic_name=semantic, binding=binding,\n                                   preserved_explicit_name=bool(explicit)))\n            variables.append(dict(name=semantic, original_name=raw, value=float(variable.X),\n                vtype=variable.VType, lb=float(variable.LB), ub=float(variable.UB),\n                objective_coefficient=float(variable.Obj)))\n        assert len({record[\'semantic_name\'] for record in identities}) == len(identities)\n        constraints = []\n        for constraint in model.getConstrs():\n            expression = model.getRow(constraint)\n            constraints.append(dict(name=constraint.ConstrName, sense=constraint.Sense,\n                rhs=float(constraint.RHS), terms=[[expression.getVar(i).VarName, float(expression.getCoeff(i))]\n                                               for i in range(expression.size())]))\n        return dict(ok=True, objective=float(model.ObjVal), objective_constant=float(model.ObjCon),\n            variables=variables, variable_identity=identities, generated_constraints=constraints,\n            num_variables=model.NumVars, num_constraints=model.NumConstrs, num_nonzeros=model.NumNZs,\n            solver_status=int(model.Status), runtime_seconds=float(model.Runtime),\n            mip_gap=float(model.MIPGap) if model.IsMIP else 0.0,\n            objective_bound=float(model.ObjBound), model_solution_count=int(model.SolCount),\n            real_generated_optimizations=len(models), metadata_consults_ground_truth=False,\n            semantic_identity_complete=True, llm_calls=0, code_output_tail=output.getvalue()[-1200:])\n    except BaseException as exc:\n        return dict(ok=False, error=f\'{type(exc).__name__}: {exc}\',\n            traceback_tail=traceback.format_exc()[-2000:],\n            real_generated_optimizations=len(models), code_output_tail=output.getvalue()[-1200:])\n    finally:\n        gp.Model.optimize, gp.Model.write = optimize, write\n\n\ndef main():\n    code, output, workdir = map(Path, sys.argv[1:])\n    for key in list(os.environ):\n        if any(part in key.upper() for part in [\'OPENAI\', \'OPENROUTER\', \'DEEPSEEK\', \'API_KEY\', \'TOKEN\']):\n            os.environ.pop(key, None)\n    gp.setParam(\'OutputFlag\', 0)\n    gp.setParam(\'Threads\', 1)\n    def audit(event, args):\n        if event in {\'socket.connect\', \'socket.getaddrinfo\', \'subprocess.Popen\', \'os.system\'}:\n            raise PermissionError(\'Network/process calls prohibited\')\n        if event == \'open\' and len(args) >= 3:\n            frame = sys._getframe()\n            while frame is not None:\n                if frame.f_code.co_filename == \'<fixed-code-replay>\':\n                    mode, flags = args[1:3]\n                    if (isinstance(mode, str) and any(c in mode for c in \'wax+\')) or (\n                        isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):\n                        raise PermissionError(\'Generated file writes prohibited\')\n                    break\n                frame = frame.f_back\n    sys.addaudithook(audit)\n    result = replay(code.read_text(), workdir)\n    result[\'llm_calls\'] = 0\n    output.write_text(json.dumps(result, ensure_ascii=False) + \'\\n\')\n\n\nif __name__ == "__main__":\n    main()\n'


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path is outside the selected repository: {relative}')
    return path


def read_records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 101 or {r['problem_id'] for r in records} != EXPECTED_IDS:
        raise ValueError('The source cohort must contain exactly the 101 unique benchmark IDs.')
    return sorted(records, key=lambda r: r['problem_id'])


def verify_inputs():
    for relative, expected in DATA_SHA256.items():
        path = safe_path(PROJECT_ROOT, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f'Missing or changed benchmark input: {relative}')
    source = {}
    for method in METHODS:
        directory = safe_path(PROJECT_ROOT, COHORTS[method]['relative_directory'])
        if not (directory / 'results.csv').is_file():
            raise FileNotFoundError(f'Required public cohort is missing: {COHORTS[method]["relative_directory"]}')
        if file_sha256(directory / 'results.csv') != COHORTS[method]['results_sha256']:
            raise ValueError(f'The source results.csv changed for {method}; use the documented source commit.')
        records = read_records(directory / 'results.csv')
        for row in records:
            fixture = IDENTITY_ARTIFACTS[method][row['problem_id']]
            for field in ['solve_code', 'solution']:
                path = safe_path(directory, row[field + '_path'])
                if file_sha256(path) != row[field + '_sha256']:
                    raise ValueError(f'Artifact hash mismatch: {method}/{row["problem_id"]}/{field}')
            if row['solve_code_sha256'] != fixture['source_code_sha256']:
                raise ValueError('Runtime identity metadata belongs to a different generated program.')
            if row['solution_sha256'] != fixture['original_solution_sha256']:
                raise ValueError('The original saved source solution has changed.')
        source[method] = records
    return source


def map_primary_values(bundle, solution, fixture, identities=None):

    pairs = solution.get('variables') or []
    raw = dict(pairs)
    if len(raw) != len(pairs) or not all(math.isfinite(float(v)) for v in raw.values()):
        raise ValueError('Duplicate variable names or non-finite incumbent values.')
    bindings = fixture['bindings'] if identities is None else identities
    original_to_semantic = dict(bindings)
    if set(original_to_semantic) != set(raw) or len(original_to_semantic) != len(bindings):
        raise ValueError('Saved-variable identities do not match the incumbent.')
    semantic_to_original = {semantic: original for original, semantic in bindings}
    if len(semantic_to_original) != len(bindings):
        raise ValueError('Variable identity is ambiguous.')
    mapping = fixture.get('primary_mapping')
    primary = {v.VarName for v in bundle.model.getVars()} - bundle.auxiliary_variables
    if mapping is None or set(mapping) != primary:
        return None
    if len(set(mapping.values())) != len(mapping) or not set(mapping.values()) <= set(semantic_to_original):
        return None
    values = {name: float(raw[semantic_to_original[semantic]]) for name, semantic in mapping.items()}
    assert all(values[name] == float(raw[semantic_to_original[semantic]]) for name, semantic in mapping.items())
    return values


def decimal_crosscheck(model, values, reference, exact_objective, checks):

    d = lambda value: Decimal.from_float(float(value))
    with localcontext() as context:
        context.prec = 120
        x = {name: d(value) for name, value in values.items()}
        independent = []
        for row in model.getConstrs():
            expression = model.getRow(row)
            lhs = sum((d(expression.getCoeff(i)) * x[expression.getVar(i).VarName]
                       for i in range(expression.size())), Decimal(0))
            rhs = d(row.RHS)
            residual = abs(lhs - rhs) if row.Sense == '=' else max(Decimal(0), lhs - rhs if row.Sense == '<' else rhs - lhs)
            independent.append(('constraint', residual))
        for var in model.getVars():
            value = x[var.VarName]
            for side, bound in [('lower', float(var.LB)), ('upper', float(var.UB))]:
                if abs(bound) < gp.GRB.INFINITY / 2:
                    independent.append(('bound', max(Decimal(0), d(bound) - value if side == 'lower' else value - d(bound))))
            if var.VType in {'I', 'B'}:
                independent.append(('integrality', abs(value - value.to_integral_value())))
            if var.VType == 'B':
                independent.append(('binary_domain', max(Decimal(0), -value, value - 1)))
        assert len(independent) == len(checks)
        for (kind, value), exact in zip(independent, checks):
            assert kind == exact['kind']
            for tolerance in TOLERANCES.values():
                threshold = exact['historical_tolerance'] if tolerance is None else tolerance
                decimal_threshold = Decimal(threshold.numerator) / Decimal(threshold.denominator)
                assert (value > decimal_threshold) == (exact['violation'] > threshold)
        objective = d(model.ObjCon) + sum((d(v.Obj) * x[v.VarName] for v in model.getVars()), Decimal(0))
        assert abs(float(objective) - float(exact_objective)) <= max(1e-9, abs(float(exact_objective)) * 1e-14)
        gap = 100 * abs(objective - d(reference)) / max(abs(d(reference)), Decimal(1))
        fraction_gap = 100 * abs(exact_objective - binary_fraction(reference)) / max(abs(binary_fraction(reference)), Fraction(1))
        assert abs(float(gap) - float(fraction_gap)) < 1e-10
        return {'linear_rows': model.NumConstrs, 'checks': len(checks), 'gap_percent': float(gap)}


def normalized_data_paths(row):
    result = []
    for address in row['dataset_address'].replace('\\n', '\n').splitlines():
        address = address.strip()
        if not address:
            continue
        if address.count('Test_Dataset/') != 1:
            raise ValueError('A dataset address must point into Test_Dataset/.')
        relative = 'Test_Dataset/' + address.split('Test_Dataset/', 1)[1]
        result.append((address, relative, safe_path(PROJECT_ROOT, relative)))
    return result


def replay_fixed_program(directory, row):

    if sys.version_info < (3, 12):
        raise RuntimeError('Fixed-code replay requires Python 3.12 or newer.')
    original = safe_path(directory, row['solve_code_path']).read_text(encoding='utf-8')
    rewrites = []
    with tempfile.TemporaryDirectory(prefix='lean_transfer_') as temporary:
        temporary = Path(temporary)
        inputs = temporary / 'inputs'
        inputs.mkdir()
        source = original
        for address, relative, path in normalized_data_paths(row):
            shutil.copyfile(path, inputs / path.name)
            for old in [address, relative]:
                if old in source:
                    source = source.replace(old, str(inputs / path.name))
                    rewrites.append(old)
            for old in [str(Path(address).parent), str(Path(relative).parent)]:
                if old in source:
                    source = source.replace(old, str(inputs))
                    rewrites.append(old)
        program, worker, captured = temporary / 'solve.py', temporary / 'worker.py', temporary / 'capture.json'
        program.write_text(source, encoding='utf-8')
        worker.write_text(REPLAY_WORKER_SOURCE, encoding='utf-8')
        environment = os.environ.copy()
        for key in list(environment):
            if any(part in key.upper() for part in ['OPENAI', 'OPENROUTER', 'DEEPSEEK', 'API_KEY', 'TOKEN']):
                environment.pop(key, None)
        environment.update(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        environment.pop('LEAN_CSV_READ_ADAPTER', None)
        try:
            process = subprocess.run([sys.executable, str(worker), str(program), str(captured), str(inputs)],
                env=environment, capture_output=True, text=True, timeout=210)
            if not captured.exists():
                raise RuntimeError(f'Replay worker failed with exit status {process.returncode}.')
            capture = json.loads(captured.read_text())
        except subprocess.TimeoutExpired:
            return {'status': 'ERROR', 'objective': None, 'variables': [], 'error': 'replay_wall_timeout_210s'}, None
    if not capture['ok']:
        return {'status': 'ERROR', 'objective': None, 'variables': [], 'error': capture['error']}, None
    saved = {'status': 'OPTIMAL', 'objective': capture['objective'],
             'variables': [[v['original_name'], v['value']] for v in capture['variables']]}
    identities = [[v['original_name'], v['semantic_name']] for v in capture['variable_identity']]
    return saved, identities


def evaluate_cohorts(source_records, benchmark):
    rows, violations, audit = [], [], []
    gp.setParam('OutputFlag', 0)
    gp.setParam('Threads', 1)
    gp.setParam('Seed', 0)
    for method in METHODS:
        directory = safe_path(PROJECT_ROOT, COHORTS[method]['relative_directory'])
        for source in source_records[method]:
            pid, index = source['problem_id'], int(source['problem_id'][-3:])
            fixture = IDENTITY_ARTIFACTS[method][pid]
            if source['query'] != benchmark.iloc[index - 1]['Query']:
                raise ValueError(f'Query mismatch for {method}/{pid}')
            published = parse_published_objective(benchmark.iloc[index - 1]['Label-objective'])
            if not math.isclose(float(source['label_objective']), published, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'Reference objective mismatch for {method}/{pid}')
            if method in REPLAY_METHODS:
                solution, live_bindings = replay_fixed_program(directory, source)
            elif COHORTS[method]['incumbent_protocol'] == 'published_saved_solution':
                solution = json.loads(safe_path(directory, source['solution_path']).read_text())
                live_bindings = None
            else:
                solution = fixture['frozen_replay_solution']
                live_bindings = None
            solved = bool(solution.get('status') == 'OPTIMAL' and solution.get('variables'))
            protocol = 'fresh_fixed_code_replay' if method in REPLAY_METHODS else COHORTS[method]['incumbent_protocol']
            basic = {'method': method, 'problem_id': pid, 'protocol': protocol, 'solved': solved,
                'predicted_label': source['predicted_label'], 'assigned_route': source['assigned_route'],
                'source_code_sha256': source['solve_code_sha256'],
                'original_solution_sha256': source['solution_sha256']}
            if not solved:
                failure = solution.get('error') or solution.get('status') or 'no_valid_incumbent'
                for profile in TOLERANCES:
                    rows.append({**basic, 'profile': profile, 'mapping_success': False, 'evaluable': False,
                        'transfer_feasible': False, 'feasible_gap_le_1': False,
                        **{kind + '_violation_count': None for kind in CHECK_KINDS},
                        'gap_percent': None, 'failure_reason': failure})
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = load_ground_truth(PROJECT_ROOT, benchmark.iloc[index - 1], index, solve=False)
            primary = map_primary_values(bundle, solution, fixture, live_bindings)
            if primary is None:
                for profile in TOLERANCES:
                    rows.append({**basic, 'profile': profile, 'mapping_success': False, 'evaluable': False,
                        'transfer_feasible': False, 'feasible_gap_le_1': False,
                        **{kind + '_violation_count': None for kind in CHECK_KINDS},
                        'gap_percent': None, 'failure_reason': fixture.get('mapping_failure', 'incomplete_or_ambiguous_mapping')})
                bundle.model.dispose()
                continue
            auxiliary, completion = _derive_auxiliary_values(bundle, primary, bundle.auxiliary_variables)
            if set(auxiliary) != bundle.auxiliary_variables:
                raise RuntimeError(f'Incomplete reference auxiliary completion: {method}/{pid}')
            transferred = {**primary, **auxiliary}
            assert all(transferred[name] == value for name, value in primary.items())
            objective, checks = check_model(bundle.model, transferred)
            certificate = REFERENCE_CERTIFICATES[pid]
            assert certificate['status'] == 2 and certificate['solution_count'] > 0
            optimum = certificate['historical_reference_objective']
            if optimum is None:
                optimum = certificate['objective']
            gap = float(100 * abs(objective - binary_fraction(optimum)) / max(abs(binary_fraction(optimum)), Fraction(1)))
            verified = decimal_crosscheck(bundle.model, transferred, optimum, objective, checks)
            for profile, tolerance in TOLERANCES.items():
                failed = [check for check in checks if check['violation'] > (check['historical_tolerance'] if tolerance is None else tolerance)]
                feasible = not failed
                rows.append({**basic, 'profile': profile, 'mapping_success': True, 'evaluable': True,
                    'transfer_feasible': feasible, 'feasible_gap_le_1': feasible and gap <= 1,
                    **{kind + '_violation_count': sum(c['kind'] == kind for c in failed) for kind in CHECK_KINDS},
                    'transferred_objective': float(objective), 'ground_truth_optimum': optimum,
                    'gap_percent': gap if feasible else None, 'failure_reason': ''})
                violations.extend({'method': method, 'problem_id': pid, 'profile': profile,
                    'kind': c['kind'], 'name': c['name'], 'violation': float(c['violation']),
                    'exact_encoded_residual': str(c['violation'])} for c in failed)
            audit.append({'method': method, 'problem_id': pid, 'protocol': protocol,
                'primary_values_unchanged': True, 'auxiliary_completion': completion,
                'absolute_gap_percent': gap, 'linear_rows_checked': verified['linear_rows'],
                'total_checks': verified['checks'], 'independent_decimal_crosscheck_passed': True,
                'mapping': fixture['primary_mapping'], 'transferred_values': transferred})
            bundle.model.dispose()
        print(f'{method}: 101 instances evaluated.')
    return rows, violations, audit


def summarize(rows):
    summary = []
    for method in METHODS:
        for profile in TOLERANCES:
            selected = [r for r in rows if r['method'] == method and r['profile'] == profile]
            assert len(selected) == 101 and {r['problem_id'] for r in selected} == EXPECTED_IDS
            feasible = [r for r in selected if r['transfer_feasible']]
            evaluated = [r for r in selected if r['evaluable']]
            summary.append({'method': method, 'profile': profile, 'denominator': len(selected),
                'solved': sum(r['solved'] for r in selected), 'mapped': len(evaluated),
                'transfer_feasible': len(feasible), 'feasible_gap_le_1': sum(r['feasible_gap_le_1'] for r in selected),
                **{kind + '_violation_count': sum(r[kind + '_violation_count'] for r in evaluated) for kind in CHECK_KINDS},
                'mean_feasible_gap_percent': statistics.mean(r['gap_percent'] for r in feasible) if feasible else None,
                'median_feasible_gap_percent': statistics.median(r['gap_percent'] for r in feasible) if feasible else None,
                'max_feasible_gap_percent': max((r['gap_percent'] for r in feasible), default=None)})
    return summary


def save_results(rows, violations, audit, summary):
    if OUTPUT_DIR.resolve() == PROJECT_ROOT.resolve():
        raise ValueError('Use a dedicated output directory, not the repository root.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    def write_table(name, records):
        records = list(records)
        if not records:
            return
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (OUTPUT_DIR / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    write_table('solution_transfer_results.csv', (r for r in rows if r['profile'] == 'diagnostic_1e-12'))
    write_table('tolerance_sensitivity.csv', rows)
    write_table('violations.csv', violations)
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (OUTPUT_DIR / 'transfer_audit.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in audit))
    report = {'source_commits': {name: COHORTS[name]['source_commit'] for name in METHODS},
        'reference_data_sha256_verified': len(DATA_SHA256), 'llm_calls': 0,
        'fresh_generated_code_execution_attempts': 101 * len(REPLAY_METHODS),
        'saved_vector_reproduction': not bool(REPLAY_METHODS),
        'primary_decisions_never_repaired': True, 'mapping_uses_no_optimum_or_feasibility_feedback': True,
        'decimal_crosscheck_passed_for_every_evaluated_instance': True,
        'missing_violation_counts_are_NA': True,
        'embedded_input_sha256': EMBEDDED_INPUT_SHA256,
        'absolute_gap_formula': '100*abs(z_transfer-z_reference)/max(abs(z_reference),1)',
        'primary_profile': 'diagnostic_1e-12',
        'arithmetic': 'Exact Fraction evaluation of binary64 model and incumbent; independent Decimal120 crosscheck',
        'versions': {'python': sys.version.split()[0], 'gurobi': '.'.join(map(str, gp.gurobi.version())),
                     'numpy': np.__version__, 'pandas': pd.__version__}}
    (OUTPUT_DIR / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


assert residual(Fraction(3), Fraction(5), '<') == 0
assert residual(Fraction(7), Fraction(5), '<') == 2
assert residual(Fraction(3), Fraction(5), '>') == 2
assert residual(Fraction(3), Fraction(5), '=') == 2

for directory in ['Test_Dataset', *(COHORTS[m]['relative_directory'] for m in METHODS)]:
    protected = (PROJECT_ROOT / directory).resolve()
    assert not OUTPUT_DIR.resolve().is_relative_to(protected)
    assert not protected.is_relative_to(OUTPUT_DIR.resolve())

source_records = verify_inputs()
benchmark = pd.read_csv(PROJECT_ROOT / 'Test_Dataset/Large-scale-or/Large-scale-or-101.csv')
assert len(benchmark) == 101
print('Verified benchmark + referenced CSV inputs:', len(DATA_SHA256))
print('Verified source cohort rows:', {method: len(records) for method, records in source_records.items()})


with contextlib.redirect_stdout(io.StringIO()):
    rows, violations, audit = evaluate_cohorts(source_records, benchmark)
summary = summarize(rows)

assert len(rows) == len({(r['method'], r['profile'], r['problem_id']) for r in rows}) == 101 * len(METHODS) * len(TOLERANCES)
assert all(r['solved'] or not r['mapping_success'] for r in rows)
assert all(not r['feasible_gap_le_1'] or r['transfer_feasible'] for r in rows)
assert all(not r['transfer_feasible'] or r['mapping_success'] for r in rows)
assert all(r['primary_values_unchanged'] and r['independent_decimal_crosscheck_passed'] for r in audit)
assert all(all(r[k + '_violation_count'] is None for k in CHECK_KINDS) for r in rows if not r['evaluable'])
verify_inputs()
save_results(rows, violations, audit, summary)
print('Transfer recomputed and cross-checked for all selected cohorts; LLM calls: 0.')
print('Results written to:', OUTPUT_DIR.relative_to(PROJECT_ROOT) if OUTPUT_DIR.is_relative_to(PROJECT_ROOT) else OUTPUT_DIR.name)


def rate(count):
    return f'{count}/101 ({100 * count / 101:.2f}%)'

main_table = pd.DataFrame([{
    'Method': r['method'],
    'Solved': rate(r['solved']),
    'Transfer feasible': rate(r['transfer_feasible']),
    'Feasible and gap ≤ 1%': rate(r['feasible_gap_le_1']),
    'Violated constraints': r['constraint_violation_count'],
} for r in summary if r['profile'] == 'diagnostic_1e-12'])
display(main_table.set_index('Method'))


sensitivity = pd.DataFrame(summary)[[
    'method', 'profile', 'solved', 'mapped', 'transfer_feasible', 'feasible_gap_le_1',
    'constraint_violation_count', 'bound_violation_count', 'integrality_violation_count',
    'binary_domain_violation_count', 'mean_feasible_gap_percent', 'max_feasible_gap_percent']]
display(sensitivity.set_index(['method', 'profile']))

unavailable = [r for r in rows if r['profile'] == 'diagnostic_1e-12' and not r['evaluable']]
if unavailable:
    display(pd.DataFrame(unavailable)[['method', 'problem_id', 'solved', 'failure_reason']].set_index(['method', 'problem_id']))

