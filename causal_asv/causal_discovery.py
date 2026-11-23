"""PC algorithm for constraint-based causal discovery."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from itertools import combinations, chain
from typing import List, Tuple, Dict, Set, Optional
import networkx as nx
from tqdm import tqdm


class PCAlgorithm:
    """PC (Peter-Clark) algorithm for causal discovery."""

    def __init__(self, significance_level: float = 0.05,
                 max_conditioning_set_size: Optional[int] = None):
        self.significance_level = significance_level
        self.max_conditioning_set_size = max_conditioning_set_size
        self.graph = None
        self.sep_sets = {}  # Separation sets for removed edges

    def fit(self, data: pd.DataFrame, verbose: bool = True) -> nx.DiGraph:
        """Discover causal structure from observational data."""
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.var_names = list(data.columns)
        self.max_conditioning_set_size = (self.max_conditioning_set_size or
                                         min(self.n_vars - 2, 10))

        if verbose:
            print(f"Discovering causal structure on {self.n_samples} samples "
                  f"with {self.n_vars} variables...")

        # Step 1: Start with complete undirected graph
        graph = self._create_complete_graph()

        # Step 2: Remove edges based on conditional independence tests
        graph, sep_sets = self._remove_edges(graph, verbose)
        self.sep_sets = sep_sets

        # Step 3: Orient edges (v-structures and propagation rules)
        dag = self._orient_edges(graph, verbose)

        self.graph = dag
        return dag

    def _create_complete_graph(self) -> nx.Graph:
        """Create complete undirected graph over all variables."""
        graph = nx.Graph()
        graph.add_nodes_from(self.var_names)
        graph.add_edges_from(combinations(self.var_names, 2))
        return graph

    def _remove_edges(self, graph: nx.Graph, verbose: bool) -> Tuple[nx.Graph, Dict]:
        """Remove edges by testing conditional independencies."""
        sep_sets = {}

        for depth in range(self.max_conditioning_set_size + 1):
            if verbose:
                print(f"  Testing conditional independence at depth {depth}...")

            edges_to_remove = []

            # Test each edge
            for u, v in list(graph.edges()):
                if depth == 0:
                    # Test marginal independence
                    independent, p_value = self._test_independence(u, v, set())
                    if independent:
                        sep_sets[(u, v)] = set()
                        edges_to_remove.append((u, v))
                else:
                    # Test conditional independence given all sets of size depth
                    independent = False
                    neighbors_u = set(graph.neighbors(u)) - {v}
                    neighbors_v = set(graph.neighbors(v)) - {u}
                    neighbors = neighbors_u | neighbors_v

                    if len(neighbors) < depth:
                        continue

                    for cond_set in combinations(neighbors, depth):
                        independent, p_value = self._test_independence(
                            u, v, set(cond_set)
                        )
                        if independent:
                            sep_sets[(u, v)] = set(cond_set)
                            edges_to_remove.append((u, v))
                            break

            # Remove independent edges
            graph.remove_edges_from(edges_to_remove)

            if verbose:
                print(f"    Removed {len(edges_to_remove)} edges. "
                      f"Remaining: {graph.number_of_edges()} edges")

        return graph, sep_sets

    def _test_independence(self, var1: str, var2: str,
                          cond_set: Set[str]) -> Tuple[bool, float]:
        """Test conditional independence using partial correlation."""
        # Select relevant columns
        cols = [var1, var2] + list(cond_set)
        data_subset = self.data[cols].values

        if len(cond_set) == 0:
            # Test marginal independence
            corr, p_value = self._correlation_test(var1, var2, data_subset)
        else:
            # Test partial correlation (conditional independence)
            corr, p_value = self._partial_correlation_test(
                var1, var2, list(cond_set), data_subset
            )

        independent = p_value > self.significance_level
        return independent, p_value

    def _correlation_test(self, var1: str, var2: str,
                         data: np.ndarray) -> Tuple[float, float]:
        """Test marginal independence using correlation."""
        from scipy.stats import pearsonr

        corr, p_value = pearsonr(data[:, 0], data[:, 1])
        return corr, p_value

    def _partial_correlation_test(self, var1: str, var2: str,
                                  cond_vars: List[str],
                                  data: np.ndarray) -> Tuple[float, float]:
        """Test partial correlation (conditional independence).
        
                Uses ols residuals from regressing var1 and var2 on cond_vars.
        """
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr

        n_cond = len(cond_vars)

        # Regress var1 on cond_vars, get residuals
        X_cond = data[:, 2:2+n_cond]
        y1 = data[:, 0]
        y2 = data[:, 1]

        reg1 = LinearRegression().fit(X_cond, y1)
        reg2 = LinearRegression().fit(X_cond, y2)

        resid1 = y1 - reg1.predict(X_cond)
        resid2 = y2 - reg2.predict(X_cond)

        # Test correlation between residuals
        corr, p_value = pearsonr(resid1, resid2)
        return corr, p_value

    def _orient_edges(self, graph: nx.Graph, verbose: bool) -> nx.DiGraph:
        """Orient edges in the graph using:
        1. V-structure detection (colliders)
        2. Orientation propagation rules
        3. Ensure no cycles are created
        """
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes())
        dag.add_edges_from(graph.edges())

        if verbose:
            print("  Orienting edges...")

        # Step 1: Detect v-structures (colliders)
        n_v_structures = 0
        for triple in self._get_unshielded_triples(graph):
            u, v, w = triple
            if (u, w) not in dag.edges() and (w, u) not in dag.edges():
                # Check if v is in separation set of u and w
                if (u, w) in self.sep_sets:
                    if v not in self.sep_sets[(u, w)]:
                        # v is a collider
                        if (u, v) in dag.edges():
                            dag.remove_edge(u, v)
                        if (w, v) in dag.edges():
                            dag.remove_edge(w, v)
                        dag.add_edge(u, v)
                        dag.add_edge(w, v)
                        n_v_structures += 1

        if verbose:
            print(f"    Detected {n_v_structures} v-structures")

        # Step 2: Orient additional edges (carefully to avoid cycles)
        changed = True
        iteration = 0
        while changed and iteration < 10:
            changed = False
            iteration += 1

            for u in dag.nodes():
                for v in list(dag.neighbors(u)):  # Convert to list to avoid modification during iteration
                    # Check if edge is already directed
                    if dag.has_edge(u, v) and not dag.has_edge(v, u):
                        continue  # Already directed

                    if dag.has_edge(u, v) or dag.has_edge(v, u):
                        # Undirected edge - try to orient it
                        # If there's w -> u, then u -> v
                        for w in list(dag.predecessors(u)):
                            if w != v and not dag.has_edge(w, v):
                                # Remove bidirectional edge if exists
                                if dag.has_edge(v, u):
                                    dag.remove_edge(v, u)
                                if dag.has_edge(u, v):
                                    dag.remove_edge(u, v)
                                # Add unidirectional edge
                                dag.add_edge(u, v)
                                changed = True
                                break

                        if not dag.has_edge(v, u):  # If not already oriented
                            # If there's v -> w and u - v, then u -> v
                            for w in list(dag.successors(v)):
                                if w != u:
                                    # Remove bidirectional edge if exists
                                    if dag.has_edge(v, u):
                                        dag.remove_edge(v, u)
                                    if dag.has_edge(u, v):
                                        dag.remove_edge(u, v)
                                    # Add unidirectional edge
                                    dag.add_edge(u, v)
                                    changed = True
                                    break

        # Step 3: Convert remaining undirected edges and ensure acyclicity
        undirected_edges = []
        for u in dag.nodes():
            for v in list(dag.neighbors(u)):
                if dag.has_edge(v, u):  # Check if bidirectional
                    undirected_edges.append((u, v))

        # Orient remaining edges (use vertex ordering to avoid cycles)
        for u, v in undirected_edges:
            # Remove bidirectional edge
            dag.remove_edge(v, u) if dag.has_edge(v, u) else None
            dag.remove_edge(u, v) if dag.has_edge(u, v) else None
            # Orient according to some consistent ordering (e.g., alphabetical)
            if u < v:
                dag.add_edge(u, v)
            else:
                dag.add_edge(v, u)

        dag = self._ensure_acyclic(dag, verbose)

        return dag

    def _ensure_acyclic(self, dag: nx.DiGraph, verbose: bool) -> nx.DiGraph:
        """Ensure the graph is acyclic by breaking cycles if they exist."""
        # Use a safer method to detect cycles - don't use topological_sort
        # as it can fail during iteration
        cycles = list(nx.simple_cycles(dag))

        if len(cycles) == 0:
            # Already acyclic
            return dag

        # Has cycles, need to break them
        if verbose:
            print(f"    Breaking {len(cycles)} cycles to ensure DAG...")

        # Break cycles iteratively until graph becomes acyclic
        max_iterations = 50
        iteration = 0

        while iteration < max_iterations:
            # Find all cycles
            cycles = list(nx.simple_cycles(dag))

            if len(cycles) == 0:
                # No more cycles!
                if verbose and iteration > 0:
                    print(f"    Successfully broke all cycles after {iteration} iterations")
                break

            # For each cycle, remove one edge
            edges_to_remove = set()
            for cycle in cycles:
                if len(cycle) < 2:
                    continue
                # Remove the first edge of the cycle
                edges_to_remove.add((cycle[0], cycle[1]))

            # Remove edges to break cycles
            for u, v in edges_to_remove:
                if dag.has_edge(u, v):
                    dag.remove_edge(u, v)
                    if verbose:
                        print(f"      Removed edge {u} -> {v}")

            iteration += 1

        if iteration >= max_iterations:
            if verbose:
                print(f"    Warning: Max iterations reached. Graph may still have cycles.")

        return dag

    def _get_unshielded_triples(self, graph: nx.Graph) -> List[Tuple]:
        """Get unshielded triples: u-v-w where u and w are not connected."""
        triples = []
        for v in graph.nodes():
            neighbors = list(graph.neighbors(v))
            for u, w in combinations(neighbors, 2):
                if not graph.has_edge(u, w):
                    triples.append((u, v, w))
        return triples

    def get_causal_order(self) -> List[str]:
        """Extract causal ordering from the discovered graph.
        
                Returns
                -------
                List[str]
                    Variables ordered by causal precedence
        """
        if self.graph is None:
            raise ValueError("Must call fit() before get_causal_order()")

        # Use topological sort to get causal order
        try:
            return list(nx.topological_sort(self.graph))
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            # Graph has cycles or other issues, use approximate ordering
            try:
                # Try to get weak topological ordering
                return list(nx.lexicographical_topological_sort(self.graph))
            except (nx.NetworkXError, nx.NetworkXUnfeasible):
                # Fall back to simple heuristic
                in_degrees = dict(self.graph.in_degree())
                return sorted(self.var_names, key=lambda x: in_degrees.get(x, 0))

