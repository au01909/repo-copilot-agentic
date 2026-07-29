"""Repository knowledge graph: builds a module-level import graph from chunk
metadata (already extracted during chunking) and renders it as Mermaid, which
the PRD's "Architecture Understanding" / "Mermaid Architecture Generation"
features ask for.
"""
import os
from typing import Dict, List

import networkx as nx

from .chunking import Chunk


def build_import_graph(chunks: List[Chunk]) -> nx.DiGraph:
    """Nodes are files; edges point from a file to the local modules it imports.
    Only imports that resolve to another file in the same repo are kept — external
    library imports (os, requests, numpy...) would just clutter the graph."""
    graph = nx.DiGraph()
    files = {c.file for c in chunks}
    module_to_file = {}
    for f in files:
        graph.add_node(f)
        module_name = os.path.splitext(f)[0].replace("/", ".")
        module_to_file[module_name] = f
        module_to_file[os.path.basename(f).split(".")[0]] = f

    for c in chunks:
        for imp in c.imports:
            # try exact match, then the top-level package name
            target = module_to_file.get(imp) or module_to_file.get(imp.split(".")[0])
            if target and target != c.file:
                graph.add_edge(c.file, target)

    return graph


def to_mermaid(graph: nx.DiGraph, max_nodes: int = 60) -> str:
    """Render as a Mermaid graph TD diagram. Truncates to the most-connected
    nodes if the repo is large, since a 500-node Mermaid diagram is unreadable."""
    if graph.number_of_nodes() > max_nodes:
        degrees = dict(graph.degree())
        keep = set(sorted(degrees, key=degrees.get, reverse=True)[:max_nodes])
        graph = graph.subgraph(keep)

    def sanitize(name: str) -> str:
        return name.replace("/", "_").replace(".", "_").replace("-", "_")

    lines = ["graph TD"]
    labels = {n: sanitize(n) for n in graph.nodes()}
    for n in graph.nodes():
        lines.append(f'    {labels[n]}["{n}"]')
    for u, v in graph.edges():
        lines.append(f"    {labels[u]} --> {labels[v]}")
    return "\n".join(lines)


def graph_stats(graph: nx.DiGraph) -> Dict:
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "most_depended_on": [], "most_dependencies": []}
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    top_in = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    top_out = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "most_depended_on": [{"file": f, "in_degree": d} for f, d in top_in if d > 0],
        "most_dependencies": [{"file": f, "out_degree": d} for f, d in top_out if d > 0],
    }
