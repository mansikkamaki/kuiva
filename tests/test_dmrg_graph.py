"""Tier-0 tests for the tensor-network tree topology.

NetworkGraph is the single owner of topology, so what is tested is its contract: tree
validation refuses everything that is not a tree, the Euler-tour sweep schedule visits every
directed bond exactly once with the center walking (never jumping), and the subtree content
hash depends on exactly the orbital content behind the bond — the property environment cache
invalidation relies on.
"""
import numpy as np
import pytest

from kuiva.dmrg.graph import NetworkGraph

#: A 9-node tree with a branch and a long arm — small but not a path and not a star.
#:        0 - 1 - 2 - 3
#:            |       |
#:            4       7 - 8
#:            |
#:            5 - 6
TREE_EDGES = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (3, 7), (7, 8)]


def tree():
    return NetworkGraph(9, TREE_EDGES)


# --- validation ---------------------------------------------------------------------------

def test_non_trees_are_refused():
    with pytest.raises(ValueError, match="edges"):
        NetworkGraph(3, [(0, 1)])                        # too few edges
    with pytest.raises(ValueError, match="edges"):
        NetworkGraph(3, [(0, 1), (1, 2), (2, 0)])        # cycle: too many edges
    with pytest.raises(ValueError, match="connected"):
        NetworkGraph(4, [(1, 2), (2, 3), (3, 1)])        # right count, disconnected + cycle
    with pytest.raises(ValueError, match="self-loop"):
        NetworkGraph(2, [(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        NetworkGraph(3, [(0, 1), (1, 0), (1, 2)])
    with pytest.raises(ValueError, match="range"):
        NetworkGraph(2, [(0, 5)])


def test_contents_default_and_disjointness():
    g = tree()
    assert g.contents == tuple((i,) for i in range(9))
    NetworkGraph(2, [(0, 1)], contents=[(3, 4), ()])     # empty content is allowed
    with pytest.raises(ValueError, match="disjoint"):
        NetworkGraph(2, [(0, 1)], contents=[(0,), (0,)])


def test_single_node_graph():
    g = NetworkGraph(1, [])
    assert g.leaves() == (0,)
    assert g.sweep_schedule(0) == []
    parent, preorder = g.parents(0)
    assert parent[0] == -1 and list(preorder) == [0]


# --- structure queries --------------------------------------------------------------------

def test_neighbors_degrees_leaves():
    g = tree()
    assert g.neighbors(1) == (0, 2, 4)
    assert g.degree(1) == 3
    assert g.leaves() == (0, 6, 8)
    assert len(g.edges) == 8
    assert len(g.directed_bonds()) == 16


def test_rooted_orientation():
    g = tree()
    for center in range(9):
        parent, preorder = g.parents(center)
        assert parent[center] == -1
        assert preorder[0] == center
        seen = set()
        for u in preorder:
            if int(u) != center:
                assert int(parent[u]) in seen            # every node after its parent
            seen.add(int(u))
        for u in range(9):                               # parents lead to the center
            steps, w = 0, u
            while w != center:
                w = int(parent[w])
                steps += 1
                assert steps <= 9
    # deterministic: children visited ascending
    _, preorder = g.parents(1)
    assert list(preorder[:2]) == [1, 0]


# --- sweep schedule -----------------------------------------------------------------------

def test_path_schedule_is_the_conventional_sweep():
    g = NetworkGraph.path(4)
    assert g.sweep_schedule(0) == [(0, 1), (1, 2), (2, 3), (3, 2), (2, 1), (1, 0)]


@pytest.mark.parametrize("center", [0, 1, 3, 5])
def test_schedule_is_a_closed_euler_tour(center):
    g = tree()
    schedule = g.sweep_schedule(center)
    assert sorted(schedule) == sorted(g.directed_bonds())     # each direction exactly once
    assert schedule[0][0] == center and schedule[-1][1] == center
    for (a, b), (c, d) in zip(schedule, schedule[1:]):
        assert b == c                                    # the center walks, never jumps


def test_schedule_is_deterministic():
    g = tree()
    assert g.sweep_schedule(2) == g.sweep_schedule(2)


# --- subtrees and hashing -----------------------------------------------------------------

def test_subtrees_partition_the_tree():
    g = tree()
    for u, v in g.edges:
        side_u = set(g.subtree_nodes(u, v))
        side_v = set(g.subtree_nodes(v, u))
        assert side_u | side_v == set(range(9))
        assert not (side_u & side_v)
        assert u in side_u and v in side_v
    with pytest.raises(ValueError, match="not an edge"):
        g.subtree_nodes(0, 2)


def test_subtree_contents_follow_the_nodes():
    g = NetworkGraph(3, [(0, 1), (1, 2)], contents=[(10, 11), (), (12,)])
    assert g.subtree_contents(0, 1) == (10, 11)
    assert g.subtree_contents(1, 2) == (10, 11)          # the empty node adds nothing
    assert g.subtree_contents(2, 1) == (12,)


def test_subtree_hash_depends_only_on_the_content_set():
    # two different topologies whose bond hides the same orbital set -> the same key
    a = NetworkGraph.path(3)                                     # chain 0 - 1 - 2
    b = NetworkGraph(3, [(0, 2), (2, 1)])                        # chain 0 - 2 - 1
    assert a.subtree_hash(0, 1) == b.subtree_hash(0, 2)          # both hide {0}
    c = NetworkGraph(3, [(1, 0), (0, 2)])                        # chain 1 - 0 - 2
    assert a.subtree_hash(2, 1) == c.subtree_hash(2, 0)          # both hide {2}
    # content follows the node id, not its position: b's bond (1, 2) hides {1}, not {2}
    assert a.subtree_hash(2, 1) != b.subtree_hash(1, 2)
    # the two directions of one bond are complementary sets, never equal here
    assert a.subtree_hash(0, 1) != a.subtree_hash(1, 0)
    # moving an orbital to another node changes exactly the keys whose subtree changed
    c = NetworkGraph(3, [(0, 1), (1, 2)], contents=[(0,), (1, 2), ()])
    d = NetworkGraph(3, [(0, 1), (1, 2)], contents=[(0,), (1,), (2,)])
    assert c.subtree_hash(0, 1) == d.subtree_hash(0, 1)          # {0} behind it in both
    assert c.subtree_contents(1, 2) == (0, 1, 2)                 # orbital 2 sits at node 1
    assert d.subtree_contents(1, 2) == (0, 1)                    # ... but at node 2 here
    assert c.subtree_hash(1, 2) != d.subtree_hash(1, 2)          # so this key changed
    assert c.subtree_hash(2, 1) != d.subtree_hash(2, 1)          # ({} vs {2}) changed too


def test_graph_equality_and_hash():
    assert tree() == tree()
    assert hash(tree()) == hash(tree())
    other = NetworkGraph(9, TREE_EDGES, contents=[(i + 100,) for i in range(9)])
    assert tree() != other
