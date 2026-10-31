#!/usr/bin/env python3
"""
build_ls_ablations.py
=====================
Generate the two Air-NRM *large-scale* ablation notebooks from the large-scale
full-pipeline notebook, so the three arms cannot drift apart.

    python build_ls_ablations.py            # writes both notebooks
    python build_ls_ablations.py --check    # regenerate in memory and diff

Why generated rather than hand-edited
-------------------------------------
An ablation is only readable as an ablation if exactly one thing differs from
the full arm. The small-scale pair (Ablation_Study_Air_NRM_RAG_Only.ipynb,
Ablation_Study_Air_NRM_Few-shot_Only.ipynb) were hand-copied from the
small-scale main notebook and have since diverged in ways that have nothing to
do with the ablation. Deriving these from
Test_Dataset/Air_NRM/large_scale/LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb
means the retriever settings, the batch loop, the fragmented-output extractor
and the scoring cell are the same objects in all three arms by construction.

What each arm removes
---------------------
Abl-RAGOnly       the retrieved SBLP exemplar. Tools (CSVQA / CName) and the
                  whole retrieval stack stay. Answers the question "does the
                  worked example carry the method, or is retrieval enough?"

Abl-FewShotOnly   custom retrieval. The static exemplar stays; complete raw files
                  are attached through the Responses API input_file interface.
                  The exemplar's parameter block is precomputed (see
                  EXEMPLAR_DOC below) rather than retrieved at run time, or the
                  arm would still be using the retriever it is meant to remove.

Both keep the classifier, the batch loop, the run logger, and the output-format
paragraph of the prompt. Both notebooks run the CA and NP/Flow large-scale
query sets; the same component is removed in both problem types. The
output-format paragraph is a harness guard against truncated answers at 421
variables, not part of the method, so removing it in one arm only would
confound the comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LS_DIR = HERE / "Test_Dataset" / "Air_NRM" / "large_scale"
SRC = LS_DIR / "LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb"
OUT_RAG = LS_DIR / "Ablation_Study_Air_NRM_RAG_Only_LargeScale.ipynb"
OUT_FS = LS_DIR / "Ablation_Study_Air_NRM_Few-shot_Only_LargeScale.ipynb"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def src_of(cell) -> str:
    return "".join(cell["source"])


def set_src(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": text.splitlines(keepends=True)}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def find_cell(nb: dict, needle: str, start: int = 0) -> int:
    for i in range(start, len(nb["cells"])):
        if needle in src_of(nb["cells"][i]):
            return i
    raise SystemExit(f"build_ls_ablations: no cell contains {needle!r} -- the "
                     f"source notebook changed shape; update this script "
                     f"rather than editing the generated notebooks")


def load_src() -> dict:
    if not SRC.exists():
        raise SystemExit(f"source notebook not found: {SRC}")
    return json.loads(SRC.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the exemplar, precomputed
# ---------------------------------------------------------------------------
# csv_qa_tool_CA() reaches avg_price and avg_pax through FAISS, but it asserts
# both against flight.csv / od_demand.csv before returning, so a plain dataframe
# lookup reproduces its output exactly. Everything else it emits (v1/v2 cells,
# the capacity coefficients, the per-flight capacity) was already pure pandas.
#
# Regenerate with --refresh-exemplar if the large_scale CSVs or
# RAG_Example_SBLP_CA_LargeScale.csv ever change; --check will tell you.
EXEMPLAR_PROMPT = (
    "Based on all flight ticket choices in 'flight.csv' and 'od_demand.csv' with "
    "attraction values in v1 and shadow attraction value ratios in v2, develop the "
    "SBLP(sales-based linear programming) formulation for among flights "
    "(OD = ('865', '678') AND Departure Time='01:30'), "
    "(OD = ('915', '865') AND Departure Time='10:35'),  that maximize the total "
    "revenue of flight ticket sales. The SBLP should include decision variables, "
    "objective function, balance constraints, scale constraints, nonnegative "
    "constraints and selection constraints. Each flight's capacity is 28% of its "
    "weekly economy capacity 'Y Seats/Week' in 'flight_capacity.csv', matched on "
    "(Oneway_OD, Departure Time) — the seven points of sale this dataset covers "
    "account for 28% of the economy cabin. Please consider that each Eco_flexi "
    "ticket consumes 2 units of flight capacity, each Eco_standard ticket consumes "
    "1.6 units, each Eco_value ticket consumes 1.3 units and each Eco_lite ticket "
    "consumes 1 unit of capacity"
)

EXEMPLAR_MATCHES = (
    "Retrieve avg_price, avg_pax, value_list, ratio_list, value_0_list, "
    "ratio_0_list, and flight_capacity for the following flights:\n"
    "(OD = ('865', '678') AND Departure Time='01:30'), "
    "(OD = ('915', '865') AND Departure Time='10:35').\n"
    "Each flight's capacity is 28% of its weekly economy capacity 'Y Seats/Week' "
    "in 'flight_capacity.csv', matched on (Oneway_OD, Departure Time) — the "
    "seven points of sale this dataset covers account for 28% of the economy "
    "cabin. Please consider that each Eco_flexi ticket consumes 2 units of flight "
    "capacity, each Eco_standard ticket consumes 1.6 units, each Eco_value ticket "
    "consumes 1.3 units and each Eco_lite ticket consumes 1 unit of capacity"
)

EXEMPLAR_DOC = (
    "avg_price={'(865678,01:30,f)': 898.85, '(865678,01:30,s)': 646.2, "
    "'(865678,01:30,v)': 494.22, '(865678,01:30,l)': 374.15, "
    "'(915865,10:35,f)': 995.75, '(915865,10:35,s)': 556.38, "
    "'(915865,10:35,v)': 411.27, '(915865,10:35,l)': 312.32}\n"
    "value_list={'(865678,01:30,f)': 0.028394, '(865678,01:30,s)': 0.05569, "
    "'(865678,01:30,v)': 0.045195, '(865678,01:30,l)': 0.042998, "
    "'(915865,10:35,f)': 0.018204, '(915865,10:35,s)': 0.021255, "
    "'(915865,10:35,v)': 0.017254, '(915865,10:35,l)': 0.030832}\n"
    "ratio_list={'(865678,01:30,f)': 0.678261, '(865678,01:30,s)': 0.664991, "
    "'(865678,01:30,v)': 0.630688, '(865678,01:30,l)': 0.561085, "
    "'(915865,10:35,f)': 0.960781, '(915865,10:35,s)': 0.955126, "
    "'(915865,10:35,v)': 0.948269, '(915865,10:35,l)': 0.936171}\n"
    "value_0_list={'865678': 0.383933, '915865': 0.638646}\n"
    "ratio_0_list={'865678': 1.783683, '915865': 1.025854}\n"
    "avg_pax={'865678': 2279.18, '915865': 2294.45}\n"
    "capacity_consum={'f': 2.0, 's': 1.6, 'v': 1.3, 'l': 1.0}\n"
    "flight_capacity={'(865678,01:30)': 589.96, '(915865,10:35)': 360.64}\n"
)


def recompute_exemplar() -> tuple[str, str, str]:
    """Rebuild EXEMPLAR_* from the CSVs, without touching the retriever."""
    import re
    from datetime import datetime, time as time_class
    import pandas as pd

    ls = LS_DIR
    v1 = pd.read_csv(ls / "v1.csv")
    v2 = pd.read_csv(ls / "v2.csv")
    demand = pd.read_csv(ls / "od_demand.csv")
    flight_df = pd.read_csv(ls / "flight.csv")
    capacity_df = pd.read_csv(ls / "flight_capacity.csv")

    products = []
    for column in v1.columns:
        if "*" in column:
            name = column.split("*")[0]
            if name not in products:
                products.append(name)

    flight_re = re.compile(
        r"\(OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*AND\s+"
        r"Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)")
    consume_re = re.compile(r"each\s+(Eco_\w+)\s+ticket\s+consumes\s+([\d.]+)\s+unit")
    haircut_re = re.compile(r"capacity is (\d+(?:\.\d+)?)% of its weekly economy capacity")

    def short(p):
        return p.split("_")[-1][0]

    def window(t):
        intervals = {"12pm~6pm": (time_class(12, 0), time_class(18, 0)),
                     "6pm~10pm": (time_class(18, 0), time_class(22, 0)),
                     "10pm~8am": (time_class(22, 0), time_class(8, 0)),
                     "8am~12pm": (time_class(8, 0), time_class(12, 0))}
        for name, (s, e) in intervals.items():
            if s < e:
                if s <= t < e:
                    return name
            elif t >= s or t < e:
                return name
        return "Unknown"

    prompt = pd.read_csv(HERE / "Large_Scale_Or_Files" /
                         "RAG_Example_SBLP_CA_LargeScale.csv")["prompt"].iloc[0]
    prompt = prompt.replace("prompt:", "").strip()

    flights = flight_re.findall(prompt)
    matches = ("Retrieve avg_price, avg_pax, value_list, ratio_list, value_0_list, "
               "ratio_0_list, and flight_capacity for the following flights:\n"
               + ", ".join(f"(OD = ('{o}', '{d}') AND Departure Time='{t}')"
                           for o, d, t in flights) + ".")
    tail = re.search(r"Each flight's capacity is.*", prompt, re.S)
    if tail:
        matches += "\n" + tail.group().strip()

    stated = dict(consume_re.findall(prompt))
    capacity_consum = {short(p): float(stated[p]) for p in products}
    hit = haircut_re.search(prompt)
    fraction = float(hit.group(1)) / 100.0 if hit else 1.0

    cap = capacity_df.set_index(["Oneway_OD", "Departure Time"])["Y Seats/Week"]
    price = flight_df.set_index(["Oneway_OD", "Departure Time", "Oneway_Product"])["Avg Price"]
    pax = demand.set_index("Oneway_OD")["Avg Pax"]

    avg_price, value_list, ratio_list = {}, {}, {}
    value_0_list, ratio_0_list, avg_pax, flight_capacity = {}, {}, {}, {}
    for origin, destination, time in flights:
        od = str((origin, destination))
        code_o = f"{origin}{destination}"
        interval = window(datetime.strptime(time, "%H:%M").time())
        for product in products:
            key = f"{product}*({interval})"
            cell = f"({code_o},{time},{short(product)})"
            avg_price[cell] = float(price[(od, time, product)])
            value_list[cell] = float(v1[v1["OD Pairs"] == od][key].values[0])
            ratio_list[cell] = float(v2[v2["OD Pairs"] == od][key].values[0])
        value_0_list[code_o] = float(v1[v1["OD Pairs"] == od]["no_purchase"].values[0])
        ratio_0_list[code_o] = float(v2[v2["OD Pairs"] == od]["no_purchase"].values[0])
        flight_capacity[f"({code_o},{time})"] = round(float(cap[(od, time)]) * fraction, 2)
        avg_pax[code_o] = float(pax[od])

    doc = "\n".join([
        f"avg_price={avg_price}", f"value_list={value_list}",
        f"ratio_list={ratio_list}", f"value_0_list={value_0_list}",
        f"ratio_0_list={ratio_0_list}", f"avg_pax={avg_pax}",
        f"capacity_consum={capacity_consum}", f"flight_capacity={flight_capacity}",
    ]) + "\n"
    return prompt, matches, doc


# ---------------------------------------------------------------------------
# pieces lifted out of the full-arm notebook, so the arms cannot drift
# ---------------------------------------------------------------------------

def extract_shared_prompt_pieces(nb: dict) -> tuple[str, str]:
    """(final-answer template, output-format paragraph) from the full arm.

    Both are shared by all three arms:

      * the Final Answer template is the SBLP formulation plus the gurobipy
        skeleton. It is what the exemplar *teaches*, so Abl-FewShotOnly has to
        show exactly the same one -- otherwise the two ablations would differ
        by more than the component under test.
      * the output-format paragraph ("one python code block, no ellipses") is a
        guard against truncated answers at 421 variables. It is a property of
        the harness, not of the method, so every arm keeps it.

    Read out of the source rather than duplicated here: an edit to the full
    arm's prompt then reaches the ablations on the next build instead of
    silently making them incomparable.
    """
    cell = src_of(nb["cells"][find_cell(nb, "def CA_Agent(query):")])

    start = cell.index("    Final Answer:\n") + len("    Final Answer:\n")
    end = cell.index("\n    '''", start)
    template = cell[start:end]

    note = "    Note: Please retrieve all neccessary information from the CSV file"
    n0 = cell.index(note)
    n1 = cell.index('\n    """', n0)
    output_format = cell[n0:n1]

    for name, text in (("final-answer template", template),
                       ("output-format paragraph", output_format)):
        if '"""' in text:
            raise SystemExit(f"{name} contains a triple quote; the generator "
                             f"emits it inside one")
        if text.rstrip("\n").endswith("\\"):
            raise SystemExit(f"{name} ends in a backslash")
    return template, output_format


# ---------------------------------------------------------------------------
# arm 1: Abl-RAGOnly  (retrieval kept, exemplar removed)
# ---------------------------------------------------------------------------

RAG_ONLY_CA_AGENT = '''

def CA_Agent(query):
    """ABLATION Abl-RAGOnly -- the same agent, with no worked example.

    The full arm retrieves an SBLP exemplar from
    Large_Scale_Or_Files/RAG_Example_SBLP_CA_LargeScale.csv, runs csv_qa_tool_CA
    over it, and pastes the resulting question/thought/action/answer trace into
    the system prompt. That whole block is gone here. Everything else is
    untouched: same two tools, same retrieval stack behind CSVQA, same model,
    same output-format instructions, same agent settings.

    So the arm still sees the data -- CSVQA returns exactly what it returns for
    the full arm -- it just has to decide on its own what an SBLP looks like.
    """
    tools = [Tool(name="CSVQA", func=csv_qa_tool_CA, description="Retrieve flight data."),
             Tool(name="CName", func=retrieve_key_information, description="Retrieve flight information.")]

    llm = lx.build_llm(CFG, "data_agent")
    # Plain string, not an f-string: nothing is interpolated any more now that
    # the exemplar is gone, and an f-string here would eat literal braces if
    # the output-format paragraph ever grew one.
    prefix = """You are an assistant that generates a SBLP mathematical model and the corresponding Gurobi Python code based on the user's description and provided CSV data.  When you need to retrieve information from the CSV file, please use the CSVQA tool. When you need to retrieve flight information, please use the CName tool.

{OUTPUT_FORMAT}
    """

    suffix = """

        Begin!

        User Description: {input}
        {agent_scratchpad}"""


    agent2 = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    **lx.agent_kwargs(CFG, prefix, suffix),
    )

    return agent2

def conversational_chat(query):
    agent2 = CA_Agent(query)
    result = agent2.invoke({"input": query})
    output = result['output']
    return output

def get_answer(query):
    agent2 = CA_Agent(query)
    result = agent2.invoke({"input": query})
    output_model = result['output']
    return output_model
def ProcessCA(query):
    CA_model = get_answer(query)
    return CA_model
'''


def build_rag_only(nb: dict, output_format: str) -> dict:
    out = copy.deepcopy(nb)

    # header
    i = find_cell(out, "large-scale fork")
    set_src(out["cells"][i], GENERATED_BANNER.format(
        arm="Abl-RAGOnly", removed="the retrieved SBLP exemplar")
        + src_of(out["cells"][i]))

    # method
    i = find_cell(out, "lx.load_config(")
    set_src(out["cells"][i],
            src_of(out["cells"][i]).replace("method='LEAN-LLM-OPT'",
                                            "method='Abl-RAGOnly'"))

    # CA_Agent: drop the exemplar
    i = find_cell(out, "def CA_Agent(query):")
    text = src_of(out["cells"][i])
    head = text[:text.index("def CA_Agent(query):")].rstrip("\n")
    body = RAG_ONLY_CA_AGENT.replace("{OUTPUT_FORMAT}", output_format)
    set_src(out["cells"][i], head + "\n" + body)

    # Replace the obsolete small-scale Flow implementation with the maintained
    # large-scale NP workflow in RAG-only mode: exact current-query retrieval,
    # no worked exemplar.
    i = find_cell(out, "def csv_qa_tool_flow(query: str):")
    set_src(out["cells"][i], '''# LARGE-SCALE NP/FLOW, RAG-only ablation.
from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import install_np_workflow

_NP_WORKFLOW = install_np_workflow(globals(), ablation="rag_only")
globals().update(_NP_WORKFLOW)
''')

    # run cell: separate output filenames
    i = find_cell(out, "example_answer_CA_largescale_GPT4.1.csv")
    text = src_of(out["cells"][i])
    for old, new in (("example_answer_CA_largescale_GPT4.1.csv",
                      "RAGOnly_answer_CA_largescale_GPT4.1.csv"),
                     ("example_code_CA_test_GPT4.1_New.csv",
                      "RAGOnly_code_CA_largescale_GPT4.1.csv"),
                     ("OBJ_CA_4.1.csv", "OBJ_CA_4.1_RAGOnly.csv"),
                     ("final_answer_CA_test_GPT4.1_New.csv",
                      "final_answer_CA_largescale_GPT4.1_RAGOnly.csv")):
        text = text.replace(old, new)
    set_src(out["cells"][i], text)

    # Activate the maintained large-scale NP query set immediately after CA,
    # and discard the two stale small-scale placeholder cells.
    np_heading = find_cell(out, "### Run NP Flow")
    set_src(out["cells"][np_heading + 1], RAG_NP_RUN)
    close = find_cell(out, "agg = LOG.close()", np_heading + 2)
    del out["cells"][np_heading + 2:close]

    out["cells"].insert(0, markdown(
        "# Ablation (large scale): RAG only\n"
        "\n"
        "`Abl-RAGOnly` on `Air-NRM-CA-LS` and `Air-NRM-NP-LS`. Generated from "
        "`LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb` by `build_ls_ablations.py`;\n"
        "the CA and NP agents retain current-query data retrieval but no longer "
        "build or show a retrieved SBLP exemplar."))
    return out


GENERATED_BANNER = """# GENERATED FILE -- do not edit by hand.
#
# Built by build_ls_ablations.py from
#   Test_Dataset/Air_NRM/large_scale/LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb
# Arm: {arm} on Air-NRM-CA-LS / Air-NRM-NP-LS. What this arm removes: {removed}.
# Edit the generator, re-run it, and commit both.

"""


# ---------------------------------------------------------------------------
# arm 2: Abl-FewShotOnly  (exemplar kept, retrieval removed)
# ---------------------------------------------------------------------------

FS_IMPORTS = '''from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import initialize_agent, AgentType, Tool
from langchain_core.documents import Document
from typing import List, Dict
from openai import OpenAI
import openai
from datetime import datetime
import re
from datetime import time as time_class
'''

FS_UPLOAD = '''# Files are attached through Responses input_file, never as bare ID strings.
from airnrm_api_files import ApiFileInput
API_FILES = ApiFileInput(CFG, role="data_agent")
'''

FS_AGENT = '''def Fewshot_Only_CA(query: str) -> str:
    return API_FILES.answer(query, PREFIX_CA)
'''

FS_NP_INSTALL = '''# LARGE-SCALE NP/FLOW, few-shot-only ablation.
# The fixed NP exemplar remains; current-query numeric retrieval is replaced by
# the same API file attachments used by the CA half of this notebook.
from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import install_np_workflow

_NP_WORKFLOW = install_np_workflow(globals(), ablation="fewshot_only")
globals().update(_NP_WORKFLOW)
'''

FS_PROCESS_INPUT = '''def Process_Input(query):
  """Route both large-scale CA and NP queries through few-shot-only agents.

  The classifier is not part of the ablation, so it is the classifier of
  LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb, unchanged. Both branches keep a
  fixed worked example and receive API file attachments rather than retrieved values.
  """
  Type, output_model = None, None
  with lx.stage("classification"):
      category_original = Problemtype(query)
  print(f"Problem type classification finished, it belongs to {category_original}.")
  if "Sales-Based Linear Programming" in    category_original:
    print("Processing AirNRM queries")
    if "flow conservation constraints" in query or "flow conservation constraint" in query:
        print('----------Flow Constraints----------')
        print("Recommend Optimal Flights With Flow Conservation Constraints")
        with lx.stage("modeling"):
            output_model = ProcessPolicyFlow(query)
        Type = "Policy_Flow"

    else:
        print('----------CA----------')
        print("Only Develop Mathematic Formulations. No Recommendation for Flights.")
        with lx.stage("modeling"):
            output_model = Fewshot_Only_CA(query)
        Type = "CA"
  if output_model is None:
      raise ValueError(
          f"unroutable problem type: {category_original!r}")
  return Type, output_model
'''

FS_RUN = '''# LARGE-SCALE, Abl-FewShotOnly: 15 instances, 112-421 variables each.
# run_notebook.py overrides this via --data / --rows.
df = pd.read_csv('Test_Dataset/Air_NRM/large_scale/query_CA.csv')
result_df = Batch_Process_Queries(df)
result_df.to_csv(LOG.dir / "FewshotOnly_answer_CA_largescale_GPT4.1.csv", index=False)

Code = []
for i in range(len(result_df['Output'])):
    text = result_df['Output'][i]
    code = _extract_gurobipy_code(text)
    Code.append(code if code else "No code found")

code_df = pd.DataFrame(Code, columns=['Code'])
code_df.to_csv(LOG.dir / "FewshotOnly_code_CA_largescale_GPT4.1.csv", index=False)

obj = gain_obj(code_df)
obj_df = pd.DataFrame({'Optimal Value': obj})
obj_df.to_csv(LOG.dir / "OBJ_CA_4.1_FewshotOnly.csv", index=False)

combined_df = pd.concat([result_df, code_df, obj_df], axis=1)
combined_df.to_csv(LOG.dir / "final_answer_CA_largescale_GPT4.1_FewshotOnly.csv", index=False)
'''

RAG_NP_RUN = '''# LARGE-SCALE NP/FLOW, Abl-RAGOnly: 15 instances.
# run_notebook.py overrides this via --data / --rows.
df = pd.read_csv('Test_Dataset/Air_NRM/large_scale/query_NP_Flow.csv')
result_df = Batch_Process_Queries(df)
result_df.to_csv(LOG.dir / "RAGOnly_answer_NP_Flow_largescale_GPT4.1.csv", index=False)

Code = []
for text in result_df['Output']:
    extracted = _extract_gurobipy_code(text)
    Code.append(extracted if extracted else "No code found")

code_df = pd.DataFrame(Code, columns=['Code'])
code_df.to_csv(LOG.dir / "RAGOnly_code_NP_Flow_largescale_GPT4.1.csv", index=False)

obj = gain_obj(code_df)
obj_df = pd.DataFrame({'Optimal Value': obj})
obj_df.to_csv(LOG.dir / "OBJ_NP_4.1_RAGOnly.csv", index=False)

combined_df = pd.concat([result_df, code_df, obj_df], axis=1)
combined_df.to_csv(LOG.dir / "final_answer_NP_Flow_largescale_GPT4.1_RAGOnly.csv", index=False)
'''

FS_NP_RUN = '''# LARGE-SCALE NP/FLOW, Abl-FewShotOnly: 15 instances.
# run_notebook.py overrides this via --data / --rows.
df = pd.read_csv('Test_Dataset/Air_NRM/large_scale/query_NP_Flow.csv')
result_df = Batch_Process_Queries(df)
result_df.to_csv(LOG.dir / "FewshotOnly_answer_NP_Flow_largescale_GPT4.1.csv", index=False)

Code = []
for text in result_df['Output']:
    extracted = _extract_gurobipy_code(text)
    Code.append(extracted if extracted else "No code found")

code_df = pd.DataFrame(Code, columns=['Code'])
code_df.to_csv(LOG.dir / "FewshotOnly_code_NP_Flow_largescale_GPT4.1.csv", index=False)

obj = gain_obj(code_df)
obj_df = pd.DataFrame({'Optimal Value': obj})
obj_df.to_csv(LOG.dir / "OBJ_NP_4.1_FewshotOnly.csv", index=False)

combined_df = pd.concat([result_df, code_df, obj_df], axis=1)
combined_df.to_csv(LOG.dir / "final_answer_NP_Flow_largescale_GPT4.1_FewshotOnly.csv", index=False)
'''


def build_fewshot_only(nb: dict, template: str, output_format: str) -> dict:
    """Assemble the few-shot-only arm out of the full arm's parts."""
    chdir_cell = src_of(nb["cells"][find_cell(nb, "large-scale fork")])
    config_cell = src_of(nb["cells"][find_cell(nb, "lx.load_config(")])
    classify_cell = src_of(nb["cells"][find_cell(nb, "def Classification_Agent(")])
    preprocess_cell = src_of(nb["cells"][find_cell(nb, "def LoadFiles(")])
    batch_cell = src_of(nb["cells"][find_cell(nb, "def Batch_Process_Queries(")])
    solve_cell = src_of(nb["cells"][find_cell(nb, "def gain_obj(")])
    close_cell = "API_FILES.close()\n" + src_of(nb["cells"][find_cell(nb, "agg = LOG.close()")])

    # config: same cell, different arm, and no retrieval to configure. The
    # embeddings object is still built -- the classifier retrieves over RefData,
    # and the classifier is not what is being ablated.
    config_cell = config_cell.replace("method='LEAN-LLM-OPT'",
                                      "method='Abl-FewShotOnly'")
    # the imports at the top of that cell are the full arm's; swap in ours
    marker = "# ---------------------------------------------------------------------------\n"
    config_cell = FS_IMPORTS + config_cell[config_cell.index(marker):]

    # Preserve the worked formulation and code, without fictitious file-ID tools.
    trace = "\nQuestion: " + EXEMPLAR_PROMPT + "\nFinal Answer:\n"
    output_format = output_format.replace(
        "exactly as the CSVQA tool returned them", "as determined from the attached CSV files")
    fewshot = trace + template.replace("{example_data_description}", EXEMPLAR_DOC)

    # One mechanical escape of every literal brace. langchain's PromptTemplate
    # reads brace-name-brace as a variable and refuses to start the agent
    # otherwise; the small-scale arm does the same thing, in two stages.
    escaped = fewshot.replace("{", "{{").replace("}", "}}")

    if "{" in output_format or "}" in output_format:
        raise SystemExit("the output-format paragraph grew a brace; it is "
                         "pasted into a PromptTemplate unescaped in every arm")

    prompt_cell = (
        "# The exemplar. Precomputed, not retrieved -- that is the point of this\n"
        "# arm. The parameter block below is byte-for-byte what csv_qa_tool_CA\n"
        "# returns for the two exemplar flights in the full arm; it is checked\n"
        "# against the CSVs by build_ls_ablations.py --check.\n"
        "EXEMPLAR_DOC = " + repr(EXEMPLAR_DOC) + "\n\n"
        "FEWSHOT_CA = " + repr(fewshot) + "\n\n"
        "ESCAPED_FEWSHOT_CA = FEWSHOT_CA.replace('{', '{{').replace('}', '}}')\n\n"
        "PREFIX_CA = (\n"
        '    "Generate the SBLP mathematical model and Gurobi code from the current '
        'question and the attached raw CSV files. Read the files through the API; '
        'derive the parameters yourself. No custom data tools are available.\\n\\n"\n'
        '    "Please refer to the following example and generate the answer in the '
        'same format:\\n\\n"\n'
        "    + FEWSHOT_CA\n"
        '    + "\\n\\n"\n'
        "    + " + repr(output_format) + "\n"
        ")\n\n"
        'SUFFIX = """\n\n'
        "Begin!\n\n"
        "User Description: {input}\n"
        "{agent_scratchpad}\n"
        '"""\n'
    )

    cells = [
        markdown("# Ablation (large scale): few-shot only\n"
                 "\n"
                 "`Abl-FewShotOnly` on `Air-NRM-CA-LS` and `Air-NRM-NP-LS`. "
                 "Generated by "
                 "`build_ls_ablations.py` from the full-arm notebook\n"
                 "`LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb`: same classifier, "
                 "same batch loop, same scoring,\n"
                 "same Final Answer template in the exemplar. Raw CSV text is attached "
                 "through Responses input_file; no custom data retrieval is used."),
        code(GENERATED_BANNER.format(
            arm="Abl-FewShotOnly",
            removed="custom retrieval (raw files supplied through API attachments)") + chdir_cell),
        code(config_cell),
        markdown("## Problem classification\n"
                 "\n"
                 "Unchanged from the full arm -- the classifier is not the "
                 "component under test."),
        code(classify_cell),
        markdown("## Few-shot only\n"
                 "\n"
                 "Attach complete raw CSV files through the API and keep the worked example."),
        code(preprocess_cell),
        code(FS_UPLOAD),
        code(prompt_cell),
        code(FS_AGENT),
        code(FS_NP_INSTALL),
        markdown("## Combined function"),
        code(FS_PROCESS_INPUT),
        markdown("## Run the test"),
        code(batch_cell),
        code(solve_cell),
        markdown("### Run CA"),
        code(FS_RUN),
        markdown("### Run NP Flow"),
        code(FS_NP_RUN),
        code(close_cell),
    ]

    out = copy.deepcopy(nb)
    out["cells"] = cells
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory, verify the exemplar against the "
                         "CSVs, and report whether the files on disk match")
    args = ap.parse_args()

    nb = load_src()
    template, output_format = extract_shared_prompt_pieces(nb)

    if args.check:
        prompt, matches, doc = recompute_exemplar()
        for name, want, got in (("EXEMPLAR_PROMPT", EXEMPLAR_PROMPT, prompt),
                                ("EXEMPLAR_MATCHES", EXEMPLAR_MATCHES, matches),
                                ("EXEMPLAR_DOC", EXEMPLAR_DOC, doc)):
            print(f"{name:18s} {'OK' if want == got else 'STALE -- rebuild'}")
            if want != got:
                print("  on disk :", repr(want[:120]))
                print("  from CSV:", repr(got[:120]))

    built = {OUT_RAG: build_rag_only(nb, output_format),
             OUT_FS: build_fewshot_only(nb, template, output_format)}

    for path, doc_nb in built.items():
        text = json.dumps(doc_nb, indent=1, ensure_ascii=False) + "\n"
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            print(f"{path.name:55s} {'up to date' if same else 'DIFFERS'}")
            continue
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(HERE)}  ({len(doc_nb['cells'])} cells)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
