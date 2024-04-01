#!/usr/bin/env python
# coding: utf-8

# ## Setup
import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random

# packages used by helper functions
import uuid


# packages for prompting definitions
import sys
sys.path.append("..")
import json

# Setting up Ollama as LLM
from langchain_community.llms import Ollama
import ollama
from ollama import Client
client = Client(host='http://localhost:11434')


# ## Prompt definitions (included in function)

#################################
# Definition of used LLM
#################################
##########################################################################
def graphPrompt(input: str, metadata={}, model="mixtral:latest"):
    if model == None:
        model = "mixtral:latest"
    
    chunk_id = metadata.get('chunk_id', None)

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = ("You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include person (agent), location, organization, date, duration, \n"
            "\tcondition, concept, object, entity  etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them like the follwing. NEVER change the value of the chunk_ID as defined in this prompt: \n"
        "[\n"
        "   {\n"
        '       "chunk_id": "CHUNK_ID_GOES_HERE",\n'
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n' 
        "   }, {...}\n"
        "]"
    )
    SYS_PROMPT = SYS_PROMPT.replace('CHUNK_ID_GOES_HERE', chunk_id)

    USER_PROMPT = f"context: ```{input}``` \n\n output: "

    response = client.generate(model="mixtral:latest", system=SYS_PROMPT, prompt=USER_PROMPT)

    aux1 = response['response']
    # Find the index of the first open bracket '['
    start_index = aux1.find('[')
    # Slice the string from start_index to extract the JSON part and fix an unexpected problem with insertes escapes (WHY ?)
    json_string = aux1[start_index:]
    json_string = json_string.replace('\\\\\_', '_')
    json_string = json_string.replace('\\\\_', '_')
    json_string = json_string.replace('\\\_', '_')
    json_string = json_string.replace('\\_', '_')
    json_string = json_string.replace('\_', '_')
    json_string.lstrip() # eliminate eventual leading blank spaces
#####################################################
    print("json-string:\n" + json_string)
#####################################################         
    try:
        result = json.loads(json_string)
        result = [dict(item) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")

    return result


# ## Functions

def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
    return graph_dataframe

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


# ## Variables

## Input data directory
##########################################################
input_file_name = "Saxony_Eastern_Expansion_EP_96.txt"
##########################################################
data_dir = "HotG_Data/"+input_file_name
inputdirectory = Path(f"./{data_dir}")

## This is where the output csv files will be written
outputdirectory = Path(f"./data_output")

output_graph_file_name = f"graph_{input_file_name[:-4]}.csv"
output_graph_file_with_path = outputdirectory/output_graph_file_name

output_chunks_file_name = f"chunks_{input_file_name[:-4]}.csv"
output_chunks_file_with_path = outputdirectory/output_chunks_file_name

output_context_prox_file_name = f"graph_contex_prox_{input_file_name[:-4]}.csv"
output_context_prox_file_with_path = outputdirectory/output_context_prox_file_name

print(output_graph_file_with_path)
print(output_chunks_file_with_path)
print(output_context_prox_file_with_path)


# ## Load Documents

#loader = TextLoader("./HotG_Data/Hanse.txt")
loader = TextLoader(inputdirectory)
Document = loader.load()
# clean unnecessary line breaks
Document[0].page_content = Document[0].page_content.replace("\n", " ")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

pages = splitter.split_documents(Document)
print("Number of chunks = ", len(pages))
print(pages[5].page_content)


# ## Create a dataframe of all the chunks

df = documents2Dataframe(pages)
print(df.shape)
df.head()


# ## Extract Concepts

## To regenerate the graph with LLM, set this to True
##################
regenerate = False  # toggle to True if the time-consuming (re-)generation of the knowlege extraction is required
##################
if regenerate:
#########################################################    
    concepts_list = df2Graph(df, model='mixtral:latest')
#########################################################
    dfg1 = graph2Df(concepts_list)
    
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    
    dfg1.to_csv(output_graph_file_with_path, sep=";", index=False)
    df.to_csv(output_chunks_file_with_path, sep=";", index=False)
else:
    dfg1 = pd.read_csv(output_graph_file_with_path, sep=";")

dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4 
## Increasing the weight of the relation to 4. 
## We will assign the weight of 1 when later the contextual proximity will be calculated.  
print(dfg1.shape)
dfg1.head()


# ## Calculating contextual proximity

dfg2 = contextual_proximity(dfg1)
dfg2.to_csv(output_context_prox_file_with_path, sep=";", index=False)
dfg2.tail()


# ### Merge both the dataframes

dfg = pd.concat([dfg1, dfg2], axis=0)
dfg = (
    dfg.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)
dfg


# ## Calculate the NetworkX Graph

nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
nodes.shape

import networkx as nx
G = nx.Graph()

## Add nodes to the graph
for node in nodes:
    G.add_node(
        str(node)
    )

## Add edges to the graph
for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],
        weight=row['count']/4
    )


# ### Calculate communities for coloring the nodes

communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of Communities = ", len(communities))
print(communities)


# ### Create a dataframe for community colors

import seaborn as sns
palette = "hls"

## Now add these colors to communities and make another dataframe
def colors2Community(communities) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

colors = colors2Community(communities)
colors


# ### Add colors to the graph

for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]

from pyvis.network import Network

#graph_output_directory = "./docs/index.html"

net = Network(
    notebook=True,
    # bgcolor="#1a1a1a",
    cdn_resources="remote",
    height="800px",
    width="100%",
    select_menu=True,
    # font_color="#cccccc",
    filter_menu=False,
)

net.from_nx(G)
# net.repulsion(node_distance=150, spring_length=400)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
# net.barnes_hut(gravity=-18100, central_gravity=5.05, spring_length=380)

# net.show(graph_output_directory)
net.show_buttons(filter_=['physics'])
net.show("knowledge_graph.html")







# DETAILED STEPS OF TERM PROXIMITY CALCULATION (same as function, only step by step to better understand the process)
## Melt the dataframe into a list of nodes
dfg_long = pd.melt(
    dfg1, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
)
dfg_long.tail(5)
dfg_long.drop(columns=["variable"], inplace=True)
# Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
dfg_long.tail(5)
dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
dfg_wide.head()
# drop self loops
self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
dfgraph2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
dfgraph2.head()
## Group and count edges.
dfgraph2 = (
    dfgraph2.groupby(["node_1", "node_2"])
    .agg({"chunk_id": [",".join, "count"]})
    .reset_index()
)
dfgraph2.head()
dfgraph2.columns = ["node_1", "node_2", "chunk_id", "count"]
dfgraph2.replace("", np.nan, inplace=True)
dfgraph2.dropna(subset=["node_1", "node_2"], inplace=True)
# Drop edges with 1 count
dfgraph2 = dfg2[dfg2["count"] != 1]
dfgraph2["edge"] = "contextual proximity"
dfgraph2.head()
dfgraph2.info()



