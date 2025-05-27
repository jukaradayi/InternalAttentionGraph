import os

import ipdb
import pickle
import argparse
import itertools
import numpy as np
import networkx as nx

from habanero import Crossref
from collections import Counter, defaultdict
from urllib.error import HTTPError
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import (
    homogeneity_score,
    completeness_score,
    contingency_matrix,
)

# call Crossref
cr = Crossref()


def read_input_csv(
    csv_path="InternalAttention_25092024_FieldsandDef.csv", separator=","
):
    """Read input CSV.
    Expected Header is
    ID Abstract Authors Title Year DOI Journal Field_1 Field_2 Field_3

    and expected field separator is ','.
    Parameter
    ---------
    csv_path: str
        path to the CSV input
    separator: char
        field separator used in CSV file

    Return
    ------
    annot: list
        list of articles stored as tuple (doc_id, DOI, label1, label2, label3)

    doc2lab: dict
        keys are doc_id, values are concatenated labels

    doi2node: dict
        keys are DOI, values are doc_id

    annot_dict: dict
        keys are doc_id, values are (DOI, label1, label2, label3)

    """

    annot = []  # store document info as tuple
    annot_dict = {}
    doc2lab = {}  # doc_id: label
    # doc2doi = {}
    doi2node = {}
    # all_labs = set()

    # parse CSV
    with open(csv_path, "r") as fin:
        for line_idx, line in enumerate(fin):

            if line_idx == 0:
                continue  # skip header

            (doc_id, DOI, label1, label2, label3, definition) = line.strip().split(
                separator
            )  # TODO get columns from header ?

            # concatenate labels
            if label2 == label1:
                # remove duplicates - there should be only 3 cases
                label2 = ""
            labels = (label1, label2, label3, definition)  # TODO list of tuples?
            # labels = [label1]

            # concatenate labels
            # node_label = "_".join(sorted(list({lab for lab in labels if len(lab) > 0})))
            node_label = labels
            # node_label = [lab for lab in labels if len(lab) > 0 ]

            doc2lab[int(doc_id)] = node_label
            # doc2doi[int(doc_id)] = DOI
            doi2node[DOI] = int(doc_id)

            # TODO not necessary..?
            annot.append((int(doc_id), DOI, label1, label2, label3))
            annot_dict[int(doc_id)] = (DOI, label1, label2, label3)

    return annot, doc2lab, doi2node, annot_dict


def dump_graph(gx, name):
    """dump graph as pickle"""
    with open(name, "wb") as fout:
        pickle.dump(gx, fout)


def load_graph(pickled_graph):
    """load pickled graph"""
    with open(pickled_graph, "rb") as fin:
        gx = pickle.load(fin)
    return gx


def create_direct_citation_graph(
    annot, doc2lab, doi2node, dump, output, ref2articles
):  # TODO option pour switch + noter dans fichier sortie
    """Create direct citation graph from list of DOI
    Parameter
    ---------
    annot: list
        list of articles stored as tuple (doc_id, DOI, label1, label2, label3)

    doc2lab: dict
        keys are doc_id, values are concatenated labels

    doi2node: dict
        keys are DOI, values are doc_id

    dump: bool
        enable to write graph object as a pickle

    output: str
        path to the output folder

    ref2articles: bool
        enable to write mapping of references to the articles citing them

    Return
    ------
    gx: networkx.graph
        Direct citation graph object
    """

    # Create graph object and add nodes with their label
    gx = nx.Graph()
    gx.add_nodes_from(doc2lab)

    # for each reference , store ids of articles
    ref2article = defaultdict(list)

    # store number of fails using crossRef
    fails = []
    succ = []
    failref = []
    failbis = []

    # Build direct citation graph
    # for each doi, check if it's accessible through crossref,
    # if it is check if references are accessible
    doi_set = {DOI for doc_id, DOI, _, _, _ in annot}
    # for doi in doi_list:
    for doc_id, DOI, _, _, _ in annot:

        ## add edge when one DOI is citing another
        try:
            work = cr.works(ids=DOI)
            references = work["message"]["reference"]
            for ref in references:
                if ("DOI" in ref) and ref["DOI"] in doi_set:
                    ref2article[ref["DOI"]].append(doc_id)

                    # graph.addEdge(doi2node[doi], doi2node[ref['DOI']])
                    gx.add_edge(doi2node[DOI], doi2node[ref["DOI"]], weight=1)
                else:
                    ## referenced DOI is not in the input DOI list
                    if "DOI" not in ref:
                        failref.append((DOI, ref))
                    elif ref["DOI"] not in doi_set:
                        ref2article[ref["DOI"]].append(doc_id)
                        failbis.append((DOI, ref["DOI"]))
            succ.append(DOI)

        except HTTPError:
            ## either doi is not on crossref,
            # or there are no references in the dict.
            fails.append(DOI)

    # write mapping of references to article
    if ref2articles:
        with open(os.path.join(output, "ref2articles.csv"), "w") as fout:
            for ref, article_ids in ref2article.items():
                for article in article_ids:
                    fout.write(f"{ref},{article}\n")

    if dump:
        dump_graph(gx, os.path.join(output, "direct_citation.pickle"))
    return gx


def create_common_citation_graph(annot, doc2lab, dump, output, ref2articles, norm):
    """Create common citation graph from list of DOI
    Parameter
    ---------
    annot: list
        list of articles stored as tuple (doc_id, DOI, label1, label2, label3)

    doc2lab: dict
        keys are doc_id, values are concatenated labels

    dump: bool
        enable to write graph object as a pickle

    output: str
        path to the output folder

    ref2articles: bool
        enable to write mapping of references to the articles citing them

    norm: str
        choose the normalisation to use to compute the weights.
        Choices are:
            - 'association' : Association strength
            - 'cosine' : cosine similarity
            - 'inclusion': inclusion
            - 'jaccard': Jaccard measure
            - 'no_norm': no normalisation, use size of union

    Return
    ------
    gx: networkx.graph
        Common citation graph object
    """
    # Create graph object and add nodes with their label
    gx = nx.Graph()
    gx.add_nodes_from(doc2lab)

    # store number of fails using crossRef # TODO remove ?
    fails = []
    succ = []

    # store distribution of number of common ref
    common_ref_distrib = []

    # for each reference , store ids of articles
    ref2article = defaultdict(list)

    # Build graph by adding an edge
    # when two graph have at least one ref in common.
    # Edge of link is number of ref in common
    doi2ref = {}
    for doc_id, DOI, _, _, _ in annot:

        try:
            # get article on crossref and its list of references
            work = cr.works(ids=DOI)
            references = work["message"]["reference"]
            doi2ref[doc_id] = {ref["DOI"] for ref in references if "DOI" in ref}
            for ref in references:
                if "DOI" in ref:
                    ref2article[ref["DOI"]].append(doc_id)
            # doi accessible through crossref + crossref api gives reference
            succ.append(DOI)

        except HTTPError:
            # either doi is not on crossref,
            # or there are no references in the dict.
            fails.append(DOI)

    # create graph
    for doi_u in doi2ref:
        for doi_v in doi2ref:
            if doi_u == doi_v:
                continue
            common_ref = doi2ref[doi_u].intersection(doi2ref[doi_v])
            n_refs_u = len(doi2ref[doi_u])  # size of u's biblio
            n_refs_v = len(doi2ref[doi_v])  # size of v's biblio

            # TODO add other measures for weights
            # TODO : add in output file name of measures used
            # list measures :
            # association strengh (6)
            weight_meas = {
                "association": len(common_ref) / (n_refs_u * n_refs_v),
                "cosine": len(common_ref) / np.sqrt(n_refs_u * n_refs_v),
                "inclusion": len(common_ref) / np.min([n_refs_u, n_refs_v]),
                "jaccard": len(common_ref) / (n_refs_u + n_refs_v - len(common_ref)),
                "no_norm": len(common_ref),
            }
            # assoc_str_w = len(common_ref) / (n_refs_u * n_refs_v)

            ## cosine (7)
            # cosine = len(common_ref) / np.sqrt(n_refs_u * n_refs_v)
            ## inclusion (8)
            ## jaccard (9)
            assert (n_refs_u + n_refs_v - len(common_ref)) == len(
                doi2ref[doi_u].union(doi2ref[doi_v])
            ), "bug in biblio union size "
            # jaccard = len(common_ref) / (n_refs_u + n_refs_v - len(common_ref))

            if len(common_ref) > 0:
                common_ref_distrib.append(len(common_ref))

                # gx.add_edge(doi_u, doi_v, weight=len(common_ref))
                gx.add_edge(doi_u, doi_v, weight=weight_meas[norm])

    # write mapping of references to article
    if ref2articles:
        with open(os.path.join(output, "ref2articles.csv"), "w") as fout:
            for ref, article_ids in ref2article.items():
                for article in article_ids:
                    fout.write(f"{ref},{article}\n")

    # write the distribution of the number of common ref
    with open(os.path.join(output, "common_ref_distrib.csv"), "w") as fout:
        print(common_ref_distrib)
        fout.write(",".join([str(cr) for cr in common_ref_distrib]))

    if dump:
        dump_graph(gx, os.path.join(output, "common_citation.pickle"))

    return gx


def write_graph(gx, output, graph_name):
    """write graph as csv readable by Gephi
    Header is 'Source,Target,Weight' , Source and Target are both nodes,
    Weight is the weight of the link. This header should be recognized by Gephi.
    """
    # write graph as a csv
    # nx.write_edgelist(gx, 'doiGraph_common.csv', data=True)
    with open(os.path.join(output, graph_name), "w") as fout:
        fout.write("Source,Target,Weight\n")
        for u, v in gx.edges:
            w = gx.get_edge_data(u, v)["weight"]
            fout.write(f"{u},{v},{w}\n")


def compute_community(
    gx,
    resolutions,
    write_contingency,
    verbose,
    output,
    doc2lab,
    annot,
    cov_thresh,
    clus_thresh,
    direct_citation,
    use_def,
    n_comm,
    norm,
):
    """Compute Greedy Modularity Communities
    Parameter
    ---------
    gx: networkx.graph
        Graph object

    resolutions: list
        list of floats used as resolution parameter for greedy community detection.

    write_contingency: bool
        if enabled, will write contingency matrix in output folder

    verbose: bool
        if enabled, be more verbose

    output: str
        path to output folder

    doc2lab: dict
        keys are doc_id, values are concatenated labels

    annot: list
        list of articles stored as tuple (doc_id, DOI, label1, label2, label3)

    cov_thresh: float
        After community detection, keep enough partitions to cover cov_thresh % of articles

    clus_thresh : int
        After community detection, keep clus_thresh partitions

    direct_citation: bool
        Enable if using the direct citation graph

    use_def: bool
        Enable if using the definitions for homogeneity and completeness

    n_comm: int
        Number of community used to create subgraphs, to study interactions between
        pairs of those communities.

    norm: str
        choose the normalisation to use to compute the weights.
        Choices are:
            - 'association' : Association strength
            - 'cosine' : cosine similarity
            - 'inclusion': inclusion
            - 'jaccard': Jaccard measure
            - 'no_norm': no normalisation, use size of union

    Return
    ------
    comm: list
        list of frozenset of nodes, each frozenset is a community
    """

    # Get size of graph, to compute coverage of clustering
    N = gx.number_of_nodes()

    print("run Greedy Modularity community detection")
    for res in resolutions:

        comm = nx.community.greedy_modularity_communities(
            gx, resolution=res, weight="weight"
        )

        if clus_thresh:
            N_clus = clus_thresh
            if clus_thresh >= len(comm):
                print(
                    f"Warning: cluster threshold {clus_thresh}"
                    "is higher than the actual"
                    "number of clusters for resolution {res:.1f}"
                )

            cov = sum([len(C) for C in comm[:clus_thresh]]) / N
            print(
                f"{cov:.3f}% of articles covered"
                f" by {clus_thresh} with resolution {res:.1f}"
            )

        elif cov_thresh:
            N_clus = 0
            cov = sum([len(C) for C in comm[:N_clus]]) / N
            while cov < cov_thresh or N_clus >= len(comm):
                N_clus += 1
                cov = sum([len(C) for C in comm[:N_clus]]) / N
            print(
                f"{N_clus} clusters needed to cover {cov:.3f}%"
                f" of articles with resolution {res:.1f}"
            )
        else:
            N_clus = len(comm)
            cov = 1.0
        covered_nodes = [u for C in comm[:N_clus] for u in C]

        # covered_nodes = [u for C in comm[:N_clus] for u in C]
        comm_label, doc2uniqLab = majority_class_per_cluster(comm, doc2lab)

        # With 1 label per document, compute homogeneity and completeness
        (
            part2label,
            y_pred,
            y_true,
            label_max,
            y_pred_covered,
            y_true_covered,
        ) = print_community_homogeneity(gx, comm, doc2lab, covered_nodes, use_def)

        homogeneity = homogeneity_score(y_true, y_pred)
        completeness = completeness_score(y_true, y_pred)

        contingency = contingency_matrix(y_true, y_pred)
        if cov_thresh is None and clus_thresh is None:
            contingency_covered = contingency
        else:
            contingency_covered = contingency_matrix(y_true_covered, y_pred_covered)

        if verbose:
            print(
                f"homogeneity score : {homogeneity:.3f}, completenesse score : {completeness:.3f}"
            )

        # ari = adjusted_rand_score(y_true, y_pred)

        # compute metrics
        metrics = compute_metrics(gx, comm, res)

        if write_contingency:
            header_true = np.unique(y_true, return_inverse=True)
            header_pred = np.unique(y_pred, return_inverse=True)
            with open(
                os.path.join(
                    output,
                    f"contingency_res_{res:.1f}_hom_{homogeneity:.3f}"
                    f"_comp{completeness:.3f}.csv",
                ),
                "w",
            ) as fout:
                fout.write("x," + ",".join([str(v) for v in header_pred[0]]) + "\n")
                for row_idx, row in enumerate(contingency):
                    fout.write(
                        header_true[0][row_idx]
                        + ","
                        + ",".join([str(v) for v in row])
                        + "\n"
                    )

            header_true = np.unique(y_true_covered, return_inverse=True)
            header_pred = np.unique(y_pred_covered, return_inverse=True)
            with open(
                os.path.join(
                    output,
                    f"contingency_covered_res_{res:.1f}_"
                    f"hom_{homogeneity:.3f}_comp{completeness:.3f}"
                    f"_{N_clus}_{cov}.csv",
                ),
                "w",
                encoding="utf-8",
            ) as fout:
                fout.write("x," + ",".join([str(v) for v in header_pred[0]]) + "\n")
                for row_idx, row in enumerate(contingency_covered):
                    fout.write(
                        header_true[0][row_idx]
                        + ","
                        + ",".join([str(v) for v in row])
                        + "\n"
                    )

        # export community list
        if direct_citation:
            filename = f"directCitation_communities_res_{res:.1f}_{N_clus}clusters_{cov:.3f}coverage_{metrics['modularity']:.3f}modularity.csv"
        else:
            filename = f"commonCitation_communities_{norm}_res_{res:.1f}_{N_clus}clusters_{cov:.3f}coverage_{metrics['modularity']:.3f}modularity.csv"
        metrics = get_subgraph_communities_pair(gx, comm, metrics, n_comm)
        write_communities(
            comm,
            annot,
            comm_label,
            metrics,
            N_clus,
            os.path.join(
                output,
                filename,
                # f"communities_res_{res:.1f}_{N_clus}clusters_{cov:.3f}coverage.csv",
            ),
            n_comm,
        )

    return comm


def compute_metrics(gx, comm, res):
    """Compute the density, the local clustering and the betweenness.
    The density and betweenness are computed for the complete graph,
    and on the subgraph induced by each community.

    Parameter
    ---------
    gx: networkx.graph
        the complete graph

    comm: list
        A list of frozenset of nodes, each is a community

    Return
    ------
    metrics: dict
        A dictionnary with a dict of all the metrics for each node
    """
    # initialize output dictionnaries
    metrics = {}

    # compute metrics on complete graph
    density = nx.density(gx)
    clustering = nx.clustering(gx)
    betweenness = nx.betweenness_centrality(gx, weight=None)
    mod = nx.community.modularity(gx, comm, resolution=1, weight="weight")
    metrics["modularity"] = mod

    # copy graph and update weights as 1 - association
    gx_dist = gx.copy()
    for e in gx_dist.edges():
        e_as = gx_dist[e[0]][e[1]]["weight"]
        gx_dist[e[0]][e[1]]["weight"] = 1 - e_as

    as_betweenness = nx.betweenness_centrality(gx_dist, weight="weight")

    # compute subgraph induced by communities
    for part in comm:
        comm_gx = nx.induced_subgraph(gx, part)
        _comm_density = nx.density(comm_gx)
        betweenness_comm = nx.betweenness_centrality(comm_gx, weight=None)

        comm_gx_dist = comm_gx.copy()
        for e in comm_gx_dist.edges():
            e_as = comm_gx_dist[e[0]][e[1]]["weight"]
            comm_gx_dist[e[0]][e[1]]["weight"] = 1 - e_as
        as_comm_betw = nx.betweenness_centrality(comm_gx, weight="weight")

        # comm_metrics[part_id] = {"density": _comm_density}
        for u in part:
            metrics[u] = {
                "density_community": _comm_density,
                "density_global": density,
                "local_clustering": clustering[u],
                "degree": gx.degree(u),
                "degree_in_community": comm_gx.degree(u),
                "centrality_in_community": betweenness_comm[u],
                "centrality_in_community_with_weights": as_comm_betw[u],
                "centrality": betweenness[u],
                "centrality_with_weights": as_betweenness[u],
            }

    return metrics


def majority_class_per_cluster(comm, doc2lab):
    """For each cluster get the label that is the most reprensented,
    and compute a simili-purity measure for the cluster.
    For each document in the cluster, if it has multiple labels, assign only
    the most common one in the cluster.
    Parameter
    ---------
    comm: list
        A list of frozenset of nodes, each is a community

    doc2lab: dict
        keys are doc_id, values are concatenated labels

    Return
    ------
    comm_label: dict
        keys are the community ids (int), values are the majority label

    doc2uniqLab: dict
        keys are doc_id, values are the majority label in the community
    """

    comm_label = {}
    doc2uniqLab = {}
    purities = []

    # loop through communities, for each community "flatten" the labels
    # and count the one occuring the most
    for comm_idx, c in enumerate(comm):
        comm_labels = [label for doc in c for label in doc2lab[doc] if label != ""]
        # comcom_labels = [doc2lab[doc] for doc in c]

        # count occurences each element
        count_labels = Counter(comm_labels)
        winner_label = count_labels.most_common(1)[0]

        # assign this label to community, and for each doc in this community,
        # assign this label if this label was one of the three labels decided.
        comm_label[comm_idx] = winner_label + (len(c),)
        purities.append(winner_label[1] / len(c))

        for doc in c:
            label_count = [
                count_labels[doc2lab[doc][0]],
                count_labels[doc2lab[doc][1]],
                count_labels[doc2lab[doc][2]],
            ]
            # label_importance = [count_labels[lab] for lab in labels if lab != ""]
            final_lab = doc2lab[doc][np.argmax(label_count)]
            # ipdb.set_trace()
            doc2uniqLab[doc] = final_lab

        # print(comcom_labels)
        print(
            f"cluster {comm_idx} has purity {winner_label[1]/len(c):.3f}, with size {len(c)}"
        )
        # TODO do we do this ? What about other nodes.. ?
        # for node in c:
        #    if winner_label in doc2lab[node]:
        #        doc2lab[node] = winner_label
    return comm_label, doc2uniqLab


def get_subgraph_communities_pair(gx, comm, metrics, n_comm):
    """pick the n_comm biggest communities in the graph, for all pairs of those communities,
    get the subgraph induced by the nodes in the community pair, and compute the
    betweenness centrality for this subgraph.
    Parameter
    ---------
    gx: networkx.graph
        Graph object

    comm: list
        A list of frozenset of nodes, each is a community

    metrics: dict
        A dictionnary with a dict of all the metrics for each node

    n_comm: int
        Number of community used to create subgraphs, to study interactions between
        pairs of those communities.

    Return
    ------
    metrics: dict
        A dictionnary with a dict of all the metrics for each node
    """
    if n_comm == 0:
        return metrics

    # get 3 biggest communities
    pairs_idx = list(itertools.combinations(range(n_comm), 2))
    for i0, i1 in pairs_idx:
        # get subgraph induced by pair of communities
        comm0, comm1 = comm[i0], comm[i1]
        comm_pair = comm0.union(comm1)
        sub_gx = gx.subgraph(comm_pair)
        sub_gx_dist = sub_gx.copy()
        for e in sub_gx_dist.edges():
            e_as = sub_gx_dist[e[0]][e[1]]["weight"]
            sub_gx_dist[e[0]][e[1]]["weight"] = 1 - e_as

        sub_as_betw = nx.betweenness_centrality(sub_gx_dist, weight="weight")

        # compute metrics on subgraph
        sub_betweenness = nx.betweenness_centrality(sub_gx, weight=None)
        sub_as_betw = nx.betweenness_centrality(sub_gx_dist, weight="weight")
        for u in comm_pair:
            metrics[u][f"comm_{i0}_{i1}_centrality"] = sub_betweenness[u]
            metrics[u][f"comm_{i0}_{i1}_centrality_weight"] = sub_as_betw[u]

    return metrics


def print_community_homogeneity(gx, comm, doc2lab, covered_nodes, use_def):
    """get manual and automatic labels
    Parameter
    ---------
    gx: networkx.graph
        the complete graph

    comm: list
        A list of frozenset of nodes, each is a community

    doc2lab: dict
        keys are doc_id, values are concatenated labels

    covered_nodes: list
        list of the ids of covered articles in selected partitions

    use_def: bool
        enable to use the definitions

    Return
    ------
    part2label: dict
        keys are the partition ID (int), value are the majority label

    y_pred: list
        list indexed by the nodes, each element is the partition id corresponding to
        the indexed node

    y_true: list
        list indexed by the nodes, each element is the label, or the definition if use_def is
        enabled

    communities: list
        list indexed by the communities, each element is the majority label in the comunity

    y_pred_covered: list
        same as y_pred but only for nodes in covered_nodes

    y_true_covered: list
        same as y_true but only for nodes in covered_nodes
    """
    part2label = {}
    node2part = {}
    set_covered_nodes = set(covered_nodes)
    communities = []
    y_pred = []
    y_true = []
    y_pred_covered = []
    y_true_covered = []

    for part_id, part in enumerate(comm):
        part2label[part_id] = [doc2lab[u] for u in part]
        label_count = Counter(part2label[part_id])
        label_max = max(label_count, key=label_count.get)
        communities.append(label_max)
        for u in part:
            node2part[u] = part_id

    for u in gx.nodes():
        # labels = doc2lab[u]
        if use_def:
            label = doc2lab[u][3]
        else:
            label = "_".join(
                sorted(
                    list(
                        {
                            lab
                            for lab in [doc2lab[u][0], doc2lab[u][1], doc2lab[u][2]]
                            if len(lab) > 0
                        }
                    )
                )
            )

        y_pred.append(node2part[u])
        y_true.append(label)
        if u in set_covered_nodes:
            y_pred_covered.append(node2part[u])
            y_true_covered.append(label)

    return (
        part2label,
        y_pred,
        y_true,
        communities,
        y_pred_covered,
        y_true_covered,
    )


def write_communities(comm, annot, comm_label, metrics, N_clus, name, n_comm):
    """Write communities in csv file.
    Header is:
        file_ID, DOI, community, label
    """
    is_covered = True
    with open(name, "w", encoding="utf-8") as fout:
        pairs_idx = list(itertools.combinations(range(n_comm), 2))
        sub_comm_centr_header = ""
        for pair in pairs_idx:
            sub_comm_centr_header += f"comm_{pair[0]}_{pair[1]}_centrality,comm_{pair[0]}_{pair[1]}_centrality_with_weights,"

        header = (
            "ID,DOI,community,Label,community_Label,"
            "is_covered,community_density,graph_density,local_clustering,"
            "degree,degree_in_community,centrality,centrality_in_community,weighted_centrality,weighted_centrality_in_community,"
        )
        header += sub_comm_centr_header[:-1]  # don't include last ","
        header += "\n"
        # fout.write(
        #    "ID,DOI,community,Label,community_Label,"
        #    "is_covered,community_density,graph_density,local_clustering,"
        #    "degree,degree_in_community,centrality,centrality_in_community\n"
        # )
        fout.write(header)
        for comm_id, community in enumerate(comm):

            if comm_id >= N_clus:
                is_covered = False

            for doc_id in community:
                DOI, lab1, lab2, lab3 = annot[doc_id]
                node_label = "_".join(
                    sorted(list({lab for lab in [lab1, lab2, lab3] if len(lab) > 0}))
                )  # concatenate labels
                metrics_out = (
                    f"{doc_id},{DOI},comm_{comm_id},{node_label},"
                    f"{comm_label[comm_id][0]},{is_covered},"
                    f"{metrics[doc_id]['density_community']},"
                    f"{metrics[doc_id]['density_global']},"
                    f"{metrics[doc_id]['local_clustering']},"
                    f"{metrics[doc_id]['degree']},"
                    f"{metrics[doc_id]['degree_in_community']},"
                    f"{metrics[doc_id]['centrality']},"
                    f"{metrics[doc_id]['centrality_in_community']},"
                    f"{metrics[doc_id]['centrality_with_weights']},"
                    f"{metrics[doc_id]['centrality_in_community_with_weights']},"
                )
                sub_comm_centr = ""
                for pair in pairs_idx:
                    if f"comm_{pair[0]}_{pair[1]}_centrality" not in metrics[doc_id]:
                        metrics[doc_id][
                            f"comm_{pair[0]}_{pair[1]}_centrality"
                        ] = -1  # TODO NaN ? else ?
                    try:
                        _centr = metrics[doc_id][f"comm_{pair[0]}_{pair[1]}_centrality"]
                        _wcentr = metrics[doc_id][
                            f"comm_{pair[0]}_{pair[1]}_centrality_weight"
                        ]
                        sub_comm_centr += f"{_centr},{_wcentr}"

                        # sub_comm_centr += (
                        #    f"{metrics[doc_id]['comm_{pair[0]}_{pair[1]}_centrality']},"
                        # )
                    except:
                        ipdb.set_trace()
                        #               metrics_out += sub_comm_centr[:-1]
                metrics_out += sub_comm_centr[:-1]
                metrics_out += "\n"
                fout.write(metrics_out)


def main():
    """Main function"""
    #  parse arguments
    parser = argparse.ArgumentParser(description="k edge swap")

    # input output arguments
    parser.add_argument(
        "-f",
        "--dataset",
        type=str,
        help="path to the input CSV",
    )

    parser.add_argument(
        "-F",
        "--separator",
        type=str,
        default=",",
        help="field separator used in csv. Default to ,",
    )

    parser.add_argument(
        "--ref2articles",
        action="store_true",
        help="Enable to write mapping of references to the articles referencing them",
    )

    parser.add_argument(
        "-dc",
        "--direct_citation",
        action="store_true",
        help="Specify -dc to use direct citation graph."
        " If not specified, it uses the common citation graph.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="path to the output folder. The folder will be created if it does not exists.",
    )

    parser.add_argument(
        "--write_graph",
        action="store_true",
        help="write the graph in a format readable by gephi",
    )

    # options to dump/load networkx graph
    pickle_group = parser.add_mutually_exclusive_group()

    pickle_group.add_argument(
        "--dump",
        action="store_true",
        help="write graph as pickle object to avoid running through crossref for future runs.",
    )

    pickle_group.add_argument(
        "--load", default=None, help="load pickled graph object generated using --dump."
    )

    # options to put a threshold on the coverage
    threshold_group = parser.add_mutually_exclusive_group()

    threshold_group.add_argument(
        "-tc",
        "--threshold_coverage",
        type=float,
        default=None,
        help="Set a threshold on the percentage of articles"
        "covered, to select the number of clusters. Mutually exclusive with -tn.",
    )

    threshold_group.add_argument(
        "-tn",
        "--threshold_cluster",
        type=int,
        default=None,
        help="Set a threshold on the number of clusters. Mutually exclusive with -tc",
    )

    # option to use the definitions as community labels
    parser.add_argument(
        "--use_def",
        action="store_true",
        help="use the definitions as community labels, to"
        " compute homogeneity and completeness metrics",
    )

    # parser.add_argument('-g', '--girvanNewman', action="store_true",
    #    help='Use Girvan Newman algorithm to find partitions.')

    parser.add_argument(
        "-c,", "--contingency", action="store_true", help="export contingency matrix"
    )

    # Community detection resolution steps
    parser.add_argument(
        "-rm",
        "--resolutionMin",
        default=0.5,
        type=float,
        help="min resolution for greedy modularity community detection",
    )

    parser.add_argument(
        "-rM",
        "--resolutionMax",
        default=1.5,
        type=float,
        help="max resolution for greedy modularity community detection",
    )

    parser.add_argument(
        "-rS",
        "--resolutionStep",
        default=0.1,
        type=float,
        help="resolution step for greedy modularity community detection",
    )

    parser.add_argument(
        "-nc",
        "--ncommunities",
        default=0,
        type=int,
        help="number of biggest communities to use when studying"
        "subgraphs of pairs of communities",
    )

    parser.add_argument(
        "-w",
        "--weights",
        choices=["association", "cosine", "inclusion", "jaccard", "no_norm"],
        type=str,
        help="choose the type of normalisation to use to compute the weights",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase verbosity"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # read input dataset and create common citation graph
    annot, doc2lab, doi2node, annot_dict = read_input_csv(
        csv_path=args.dataset, separator=args.separator
    )

    # don't generate graph if --load is used
    if args.load is None:

        if args.direct_citation:
            graph_name = "directCitationGraph.csv"
            gx = create_direct_citation_graph(
                annot, doc2lab, doi2node, args.dump, args.output, args.ref2articles
            )
            # create_common_citation_graph(annot, doc2lab)
        else:
            graph_name = "commonCitationGraph.csv"
            gx = create_common_citation_graph(
                annot, doc2lab, args.dump, args.output, args.ref2articles, args.weights
            )

        # when requested, write graph
        if args.write_graph:
            write_graph(gx, args.output, graph_name)
    else:
        gx = load_graph(args.load)

    # run community detection
    if args.resolutionMin == args.resolutionMax:
        resolutions = [args.resolutionMin]
    else:
        resolutions = np.arange(
            args.resolutionMin, args.resolutionMax, args.resolutionStep
        )

    if args.verbose:
        print(f"Running community detection for resolutions: {resolutions}")
    if not args.weights:
        raise RuntimeError(
            "no weight defined. Please choose a normalisation using -w with one the "
            "choices available"
        )
    compute_community(
        gx,
        resolutions,
        args.contingency,
        args.verbose,
        args.output,
        doc2lab,
        annot_dict,
        args.threshold_coverage,
        args.threshold_cluster,
        args.direct_citation,
        args.use_def,
        args.ncommunities,
        args.weights,
    )
    # else:
    #    k = 80 # can change k to change "resolution"
    #    compute_girvanNewman(gx, k, args.contingency, args.verbose,
    #                         args.output, doc2lab, annot_dict)


if __name__ == "__main__":
    main()
