import os

import ipdb
import yaml
import pickle
import shutil
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


def parse_config(config_path):
    """Read input configuration for parameters

    Parameter
    ---------
    config_path: str
        path to the configuration file, in yaml format

    Return
    ------
    config: dict
        dictionnary with the parameter names as key, and param values as value.
    """
    with open(config_path, "r") as fin:
        config = yaml.safe_load(fin)

    ## check config values
    # check weight
    if config["graph"]["weight"] not in [
        "association",
        "cosine",
        "inclusion",
        "jaccard",
        "no_norm",
    ]:
        raise RuntimeError(
            "no weight defined. Please choose a normalisation using -w with one the "
            "choices available"
        )

    # check booleans
    if not isinstance(config["graph"]["ref2articles"], bool):
        raise RuntimeError(
            f"ref2articles argument should be True "
            "or False, not {config['graph']['ref2articles']}"
        )
    if not isinstance(config["graph"]["write_graph"], bool):
        raise RuntimeError(
            f"write_graph argument should be True "
            "or False, not {config['graph']['write_graph']}"
        )
    if not isinstance(config["graph"]["dump"], bool):
        raise RuntimeError(
            f"dump argument should be True " "or False, not {config['graph']['dump']}"
        )
    if not isinstance(config["verbose"], bool):
        raise RuntimeError(
            f"verbose argument should be True " "or False, not {config['verbose']}"
        )

    # create output folder if it doesn't exist
    if not os.path.isdir(config["input_output"]["output"]):
        os.mkdir(config["input_output"]["output"])
    # check resolution
    if not (
        isinstance(config["communities"]["greedy"]["resolutionMax"], float)
        and isinstance(config["communities"]["greedy"]["resolutionMin"], float)
    ):
        raise RuntimeError(f"resolutionMin and Max should be floats")

    if (
        config["communities"]["greedy"]["resolutionMax"]
        < config["communities"]["greedy"]["resolutionMin"]
    ):
        raise RuntimeError(f"resolutionMin should be less or equal to resolutionMax")

    # copy config file in output folder
    shutil.copy(config_path, config["input_output"]["output"])

    return config


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

    def parse_header(header, separator):
        """Parse header to find position of the following columns:
        ID,  DOI, Final_Field1, Final_Field2, Final_Field3, Final_definition
        """
        col_names = header.strip().split(separator)
        id_pos = col_names.index("ID")
        doi_pos = col_names.index("DOI")
        field1_pos = col_names.index("Final_Field1")
        field2_pos = col_names.index("Final_Field2")
        field3_pos = col_names.index("Final_Field3")
        def_pos = col_names.index("Final_definition")
        return (id_pos, doi_pos, field1_pos, field2_pos, field3_pos, def_pos)

    doc2lab = {}  # doc_id: label
    doi2node = {}
    doc_list = []
    # parse CSV
    with open(csv_path, "r") as fin:
        for line_idx, line in enumerate(fin):

            # parse header and get position of each column
            if line_idx == 0:
                (id_pos, doi_pos, field1_pos, field2_pos, field3_pos, def_pos) = (
                    parse_header(line, separator)
                )

                continue

            # parse line and get each value
            col_values = line.strip().split(separator)
            doc_id = col_values[id_pos]
            DOI = col_values[doi_pos]
            # label1 = col_values[field1_pos] # labels are unused
            # label2 = col_values[field2_pos]
            # label3 = col_values[field3_pos]
            definition = col_values[def_pos]

            # keep definition used for article
            node_label = definition

            doc2lab[int(doc_id)] = {"definition": node_label, "DOI": DOI}
            doi2node[DOI] = int(doc_id)

            doc_list.append((int(doc_id), DOI))  # , label1, label2, label3))

    return doc2lab, doi2node, doc_list


def dump_graph(gx, name):
    """dump graph as pickle"""
    with open(name, "wb") as fout:
        pickle.dump(gx, fout)


def load_graph(pickled_graph):
    """load pickled graph"""
    with open(pickled_graph, "rb") as fin:
        gx = pickle.load(fin)
    return gx


def create_bibliographic_coupling_graph(
    doc_list, doc2lab, dump, output, ref2articles, norm
):
    """Create common citation graph from list of DOI
    Parameter
    ---------
    annot: list
        list of articles stored as tuple (doc_id, DOI, label1, label2, label3)
    #TODO annot plus necessaire
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

    # store number of fails using crossRef for debugging purposes
    fails = []
    succ = []

    # store distribution of number of common ref
    common_ref_distrib = {}

    # for each reference , store ids of articles
    ref2article = defaultdict(list)

    # Build graph by adding an edge
    # when two graph have at least one ref in common.
    # Edge of link is number of ref in common
    doi2ref = {}
    doc2attr = {}
    for doc_id, DOI in doc_list:

        try:
            # get article on crossref and its list of references
            work = cr.works(ids=DOI)
            references = work["message"]["reference"]
            if "published-print" in work["message"]:
                _date = work["message"]["published-print"]["date-parts"][0]
            elif "published-online" in work["message"]:
                _date = work["message"]["published-online"]["date-parts"][0]

            if len(_date) == 2:
                date = f"{_date[1]}/{_date[0]}"
            else:
                date = f"{_date[0]}"
            doc2attr[doc_id] = {"date": date, "doi": DOI}

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

    nx.set_node_attributes(gx, doc2attr)
    # create graph
    edge_attr = {}
    for doi_u in doi2ref:
        for doi_v in doi2ref:
            if doi_u == doi_v:
                continue
            common_ref = doi2ref[doi_u].intersection(doi2ref[doi_v])
            n_refs_u = len(doi2ref[doi_u])  # size of u's biblio
            n_refs_v = len(doi2ref[doi_v])  # size of v's biblio

            # list measures :
            weight_meas = {
                "association": len(common_ref) / (n_refs_u * n_refs_v),
                "cosine": len(common_ref) / np.sqrt(n_refs_u * n_refs_v),
                "inclusion": len(common_ref) / np.min([n_refs_u, n_refs_v]),
                "jaccard": len(common_ref) / (n_refs_u + n_refs_v - len(common_ref)),
                "no_norm": len(common_ref),
            }
            dist_meas = 1 - weight_meas[norm]
            assert (n_refs_u + n_refs_v - len(common_ref)) == len(
                doi2ref[doi_u].union(doi2ref[doi_v])
            ), "bug in biblio union size "
            # jaccard = len(common_ref) / (n_refs_u + n_refs_v - len(common_ref))

            if len(common_ref) > 0:
                common_ref_distrib[(doi_u, doi_v)] = len(common_ref)

                # gx.add_edge(doi_u, doi_v, weight=len(common_ref))
                gx.add_edge(doi_u, doi_v, weight=weight_meas[norm])
                edge_attr[(doi_u, doi_v)] = {"distance": dist_meas}

    # set edges distance attribute
    nx.set_edge_attributes(gx, edge_attr)

    # write mapping of references to article
    if ref2articles:
        with open(os.path.join(output, "ref2articles.csv"), "w") as fout:
            for ref, article_ids in ref2article.items():
                for article in article_ids:
                    fout.write(f"{ref},{article}\n")

    # write the distribution of the number of common ref
    with open(os.path.join(output, "common_ref_distrib.csv"), "w") as fout:
        fout.write("ID_u,ID_v,DOI_u,DOI_v,number_common_ref\n")
        for e in gx.edges():
            _, DOI_u = doc2attr[e[0]]
            _, DOI_v = doc2attr[e[1]]

            # DOI_u, _, _, _ = annot_dict[e[0]]
            # DOI_v, _, _, _ = annot_dict[e[1]]
            # w = gx[e[0]][e[1]]["weight"]
            w = common_ref_distrib[(e[0], e[1])]
            fout.write(f"{e[0]},{e[1]},{DOI_u},{DOI_v},{w}\n")

    if dump:
        dump_graph(gx, os.path.join(output, "common_citation.pickle"))

    return gx


def compute_communities(
    gx,
    resolutions,
    verbose,
    output,
    doc2lab,
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

    verbose: bool
        if enabled, be more verbose

    output: str
        path to output folder

    doc2lab: dict
        keys are doc_id, values are concatenated labels

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

        # compute metrics
        metrics, graph_metrics = compute_metrics(gx, comm, res)

        # export community list
        filename = f"bibliographicCoupling_norm_{norm}_res_{res:.1f}.csv"
        filename_graph = f"bibliographicCoupling__graphMetrics_res_{res:.1f}.csv"

        # write all metrics in output file
        write_communities(
            comm,
            metrics,
            graph_metrics,
            os.path.join(
                output,
                filename,
                # f"communities_res_{res:.1f}_{N_clus}clusters_{cov:.3f}coverage.csv",
            ),
            os.path.join(
                output,
                filename_graph,
            ),
            n_comm,
            doc2lab,
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
    graph_metrics = {}
    # compute metrics on complete graph
    density = nx.density(gx)
    clustering = nx.clustering(gx)
    mod = nx.community.modularity(gx, comm, resolution=1, weight="weight")
    centrality = nx.betweenness_centrality(gx, weight="distance")

    # add global metrics
    graph_metrics["global"] = {"density": density, "modularity": mod}

    # compute subgraph induced by communities
    for comm_idx, part in enumerate(comm):
        comm_gx = nx.induced_subgraph(gx, part)
        _comm_density = nx.density(comm_gx)
        centrality_comm = nx.betweenness_centrality(comm_gx, weight="distance")
        graph_metrics[f"{comm_idx}"] = {"density": _comm_density, "modularity": "NA"}

        for u in part:
            metrics[u] = {
                "local_clustering": clustering[u],
                "degree": gx.degree(u),
                "degree_in_community": comm_gx.degree(u),
                "centrality_in_community": centrality_comm[u],
                "centrality": centrality[u],
            }

    return metrics, graph_metrics


def get_subgraph_communities_pair(gx, comm, metrics, n_comm, output):
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

    # get n_comm biggest communities
    pairs_idx = list(itertools.combinations(range(n_comm), 2))
    boundaries = {}
    for i0, i1 in pairs_idx:
        comm0, comm1 = comm[i0], comm[i1]

        comm_pair = comm0.union(comm1)

        sub_gx = gx.subgraph(comm_pair)
        write_graph(sub_gx, output, f"subgraph_comm{i0}_comm{i1}.csv")

    return


def write_communities(
    comm, metrics, graph_metrics, filename, filename_graph, n_comm, doc2lab
):
    """Write communities in csv file.
    Header is:
        file_ID, DOI, community, label
    """
    with open(filename_graph, "w", encoding="utf-8") as fout:
        # write header global graph metrics
        fout.write("graph,density,modularity\n")
        fout.write(
            f"global,{graph_metrics['global']['density']},{graph_metrics['global']['modularity']}\n"
        )
        for comm_idx in range(len(comm)):
            fout.write(
                f"comm_{comm_idx},{graph_metrics[str(comm_idx)]['density']},{graph_metrics[str(comm_idx)]['modularity']}\n"
            )

    with open(filename, "w", encoding="utf-8") as fout:
        pairs_idx = list(itertools.combinations(range(n_comm), 2))

        header = (
            "ID,DOI,community,definition,"
            "community_density,graph_density,local_clustering,"
            "degree,degree_in_community,centrality,centrality_in_community,weighted_centrality,weighted_centrality_in_community,"
        )
        header = header[:-1]
        header += "\n"
        # fout.write(
        #    "ID,DOI,community,Label,community_Label,"
        #    "is_covered,community_density,graph_density,local_clustering,"
        #    "degree,degree_in_community,centrality,centrality_in_community\n"
        # )
        fout.write(header)
        for comm_id, community in enumerate(comm):

            for doc_id in community:
                DOI, definition = doc2lab[doc_id]["DOI"], doc2lab[doc_id]["definition"]
                metrics_out = (
                    f"{doc_id},{DOI},comm_{comm_id},{definition},"
                    f"{metrics[doc_id]['local_clustering']},"
                    f"{metrics[doc_id]['degree']},"
                    f"{metrics[doc_id]['degree_in_community']},"
                    f"{metrics[doc_id]['centrality']},"
                    f"{metrics[doc_id]['centrality_in_community']},"
                )
                metrics_out = metrics_out[:-1]
                metrics_out += "\n"
                fout.write(metrics_out)


def main():
    parser = argparse.ArgumentParser(description="Study communities")

    # input output arguments
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to the configuration file in yaml format",
    )
    args = parser.parse_args()
    config = parse_config(args.config)

    # declare variables for input parameters called multiple times
    output = config["input_output"]["output"]
    # if not os.path.isdir(output):
    #    os.mkdir(output)

    # read input dataset and create common citation graph
    doc2lab, doi2node, doc_list = read_input_csv(
        csv_path=config["input_output"]["dataset"],
        separator=config["input_output"]["separator"],
    )

    # don't generate graph if --load is used
    if config["graph"]["load"] is None or config["graph"]["load"] == "":
        graph_name = "commonCitationGraph.csv"
        gx = create_bibliographic_coupling_graph(
            doc_list,
            doc2lab,
            config["graph"]["dump"],
            output,
            config["graph"]["ref2articles"],
            config["graph"]["weight"],
        )

        # when requested, write graph
        if config["graph"]["write_graph"]:
            write_graph(gx, output, graph_name)
    else:
        gx = load_graph(config["graph"]["load"])

    # run community detection
    if (
        config["communities"]["greedy"]["resolutionMin"]
        == config["communities"]["greedy"]["resolutionMax"]
    ):
        resolutions = [config["communities"]["greedy"]["resolutionMin"]]
    else:
        resolutions = np.arange(
            config["communities"]["greedy"]["resolutionMin"],
            config["communities"]["greedy"]["resolutionMax"],
            config["communities"]["greedy"]["resolutionStep"],
        )

    if config["verbose"]:
        print(f"Running community detection for resolutions: {resolutions}")

    compute_communities(
        gx,
        resolutions,
        config["verbose"],
        output,
        doc2lab,
        config["communities"]["ncommunities"],
        config["graph"]["weight"],
    )


if __name__ == "__main__":
    main()
