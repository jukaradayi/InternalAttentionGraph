import os
import ipdb
import argparse
import itertools
import numpy as np
import networkx as nx

from habanero import Crossref
from collections import defaultdict, Counter
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_score, completeness_score, contingency_matrix

# call Crossref
cr = Crossref()

def read_input_csv(csv_path='Internal_Att_review_for_Julien28062024.csv'):
    """ Read input CSV. 
    Expected Header is 
    ID Abstract Authors Title Year DOI Journal Field_1 Field_2 Field_3

    and expected field separator is '§'.
    Output is a list of tuples.
    """

    annot = list() # store document info as tuple
    annot_dict = dict()
    doc2lab = dict() # doc_id: label
    doc2doi = dict() 
    all_labs = set()

    # parse CSV
    with open(csv_path, 'r') as fin:
        for line_idx, line in enumerate(fin):

            if line_idx == 0:
                continue # skip header

            doc_id, _, _, _, _, DOI,_, label1, label2, label3 = line.strip().split('§')

            # concatenate labels
            labels = [label1, label2, label3]
            #labels = [label1]
            node_label = '_'.join(sorted(list(set([lab for lab in labels if len(lab) > 0 ])))) # concatenate labels
            doc2lab[int(doc_id)] = node_label
            doc2doi[int(doc_id)] = DOI

            # TODO not necessary..?
            annot.append((int(doc_id), DOI, label1, label2, label3))
            annot_dict[int(doc_id)] = (DOI, label1, label2, label3)

    return annot, doc2lab, doc2doi, annot_dict


def create_graph(annot, doc2lab, doc2doi):
    """ Create common citation graph from list of DOI"""
    
    # Create graph object and add nodes with their label
    gx = nx.Graph()
    gx.add_nodes_from(doc2lab)

    # store number of fails using crossRef
    fails = []
    succ = []
    failref = []
    failbis = []

    ### Build direct citation graph
    ## for each doi, check if it's accessible through crossref, if it is check if references are accessible
    #for doi in doi_list:
    #    ## add edge when one DOI is citing another
    #    try:
    #        work = cr.works(ids = doi)
    #        references = work['message']['reference']
    #        for ref in references :
    #            if ("DOI" in ref) and ref['DOI'] in doi_list:
    #                print('adding edge to graph')
    #                #graph.addEdge(doi2node[doi], doi2node[ref['DOI']])
    #                gx.add_edge(doi2node[doi], doi2node[ref['DOI']])
    #            else:
    #                ## referenced DOI is not in the input DOI list
    #                if 'DOI' not in ref:
    #                    failref.append((doi,ref))
    #                elif ref['DOI'] not in doi_list:
    #                    failbis.append((doi,ref['DOI']))
    #        succ.append(doi) # doi accessible through crossref + crossref api gives reference : happy
    #
    #    except:
    #        ## either doi is not on crossref, or there are no references in the dict.
    #        fails.append(doi)
    
    ### Build common citation graph 
    # Build graph by adding an edge when two graph have at least one ref in common.
    # Edge of link is number of ref in common
    doi2ref = dict()
    for doc_id, DOI, _, _, _ in annot:
    
        try:
            # get article on crossref and its list of references
            work = cr.works(ids = DOI)
            references = work['message']['reference']
            doi2ref[doc_id] = set([ref['DOI'] for ref in references if "DOI" in ref])
            succ.append(doi) # doi accessible through crossref + crossref api gives reference : happy
    
        except:
            ## either doi is not on crossref, or there are no references in the dict.
            fails.append(DOI)
    
    # create graph
    for doi_u in doi2ref:
        for doi_v in doi2ref:
            if doi_u == doi_v:
                continue
            common_ref = doi2ref[doi_u].intersection(doi2ref[doi_v])
            if len(common_ref) > 0:
                gx.add_edge(doi_u, doi_v, weight=len(common_ref))
    return gx


def write_graph(gx, output):
    # write graph as a csv
    #nx.write_edgelist(gx, 'doiGraph_common.csv', data=True)
    with open(os.path.join(output, 'commonCitationGraph.csv'), 'w') as fout:
        fout.write('Source,Target,Weight\n')
        for u,v in gx.edges:
            w = gx.get_edge_data(u,v)['weight']
            fout.write(f'{u},{v},{w}\n')

def compute_community(gx, resolutions, write_contingency, verbose, output, doc2lab, annot):
    """ Compute Greedy Modularity Communities"""

    print('run Greedy Modularity community detection')
    for res in resolutions:

        comm = nx.community.greedy_modularity_communities(gx, resolution=res)
        ipdb.set_trace()
        # compare community detection with manual annotation
        part2label, y_pred, y_true, label_max = print_community_homogeneity(gx, comm, doc2lab)
        homogeneity = homogeneity_score(y_true, y_pred)
        completeness = completeness_score(y_true, y_pred)
        contingency = contingency_matrix(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred) 
        
        # write outputs to command line and to file
        print(f"resolution: {res:.1f}, homogeneity: {homogeneity}, completeness: {completeness}, adjusted rand idnex:{ari}")

        if write_contingency:
            header_true = np.unique(y_true, return_inverse=True)
            header_pred = np.unique(y_pred, return_inverse=True)
            with open(os.path.join(output, f'contingency_res_{res:.1f}_hom_{homogeneity:.3f}_comp{completeness:.3f}.csv'), 'w') as fout:
                fout.write('x,' + ','.join([str(v) for v in header_pred[0]]) + '\n')
                for row_idx, row in enumerate(contingency):
                    fout.write(header_true[0][row_idx] + ',' + ','.join([str(v) for v in row]) + '\n')

        # export community list
        write_communities(comm, annot, os.path.join(output,f'communities_res_{res:.1f}_hom_{homogeneity:.3f}_comp{completeness:.3f}.csv'))

    return comm

def compute_girvanNewman(gx, k, write_contingency, verbose, output, doc2lab, annot):
    """ Compute community partition using Girvan Newman algorithm"""

    print('Running Girvan Newman Community detection')
    comm = nx.community.girvan_newman(gx) #, resolution=resolution)
    all_comm = list()

    # run Girvan Newman community detection with k steps
    comp = nx.community.girvan_newman(gx)
    k_idx = 0
    for communities in itertools.islice(comp, k):

        current_comm = list(sorted(c) for c in communities)
        all_comm.append(current_comm)

        part2label, y_pred, y_true, label_max = print_community_homogeneity(gx, current_comm, doc2lab)
        homogeneity = homogeneity_score(y_true, y_pred)
        completeness = completeness_score(y_true, y_pred)
        contingency = contingency_matrix(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred) 

        print(f"resolution: {k_idx:.1f}, homogeneity: {homogeneity:.3f}, completeness: {completeness:.3f}, adjusted rand idnex:{ari}")

        if write_contingency:
            header_true = np.unique(y_true, return_inverse=True)
            header_pred = np.unique(y_pred, return_inverse=True)
            with open(os.path.join(output, f'contingency_{k_idx:.1f}_hom_{homogeneity:.3f}_comp{completeness:.3f}.csv'), 'w') as fout:
                fout.write('x,' + ','.join([str(v) for v in header_pred[0]]) + '\n')
                for row_idx, row in enumerate(contingency):
                    fout.write(header_true[0][row_idx] + ',' + ','.join([str(v) for v in row]) + '\n')

        # export community list
        write_communities(current_comm, annot, os.path.join(output,f'communities_res_{k_idx:.1f}_hom_{homogeneity:.3f}_comp{completeness:.3f}.csv'))

        k_idx += 1
    return all_comm

def print_community_homogeneity(gx, comm, doc2lab):
    """ get manual and automatic labels"""
    part2label = dict()
    node2part = dict()
    communities = []
    y_pred = []
    y_true = []

    for part_id, part in enumerate(comm):
        part2label[part_id] = [doc2lab[u] for u in part]
        label_count = Counter(part2label[part_id])
        label_max = max(label_count, key=label_count.get)
        communities.append(label_max)
        for u in part:
            node2part[u] = part_id

    for u in gx.nodes():
        y_pred.append(node2part[u])
        y_true.append(doc2lab[u])

    return part2label, y_pred, y_true, communities

def write_communities(comm, annot, name):
    with open(name, 'w') as fout:
        fout.write(f'ID,DOI,community,Label\n')
        for comm_id, community in enumerate(comm):
            for doc_id in community:
                DOI, lab1, lab2, lab3 = annot[doc_id]
                node_label = '_'.join(sorted(list(set([lab for lab in [lab1, lab2, lab3] if len(lab) > 0 ])))) # concatenate labels

                fout.write(f'{doc_id},{DOI},comm_{comm_id},{node_label}\n')

def main():
    #  parse arguments
    parser = argparse.ArgumentParser(description='k edge swap')

    # input output arguments
    parser.add_argument('-f', '--dataset', type=str, 
            default='Internal_Att_review_for_Julien28062024.csv',
            help='path to the input CSV')

    parser.add_argument('-o', '--output', type=str, default=None, 
            help='path to the output folder. The folder will be created if it does not exists.')

    parser.add_argument('--write_graph', action="store_true",
            help="write the graph in a format readable by gephi")

    parser.add_argument('-g', '--girvanNewman', action="store_true",
        help='Use Girvan Newman algorithm to find partitions.')

    parser.add_argument('-c,', '--contingency', action="store_true", 
        help="export contingency matrix")

    parser.add_argument('-rm', '--resolutionMin', default=0.5, type=float,
        help="min resolution for greedy modularity community detection")
  
    parser.add_argument('-rM', '--resolutionMax', default=1.5, type=float,
        help="max resolution for greedy modularity community detection")

    parser.add_argument('-rS', '--resolutionStep', default=0.1, type=float,
        help="resolution step for greedy modularity community detection")

    parser.add_argument('-v', '--verbose', action="store_true",
        help="increase verbosity")

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # read input dataset and create common citation graph
    annot, doc2lab, doc2doi, annot_dict = read_input_csv(csv_path=args.dataset)
    gx = create_graph(annot, doc2lab, doc2doi)
    
    # when requested, write graph
    if args.write_graph:
        write_graph(gx, args.output)

    # run community detection
    resolutions = np.arange(args.resolutionMin, args.resolutionMax, args.resolutionStep) 

    if not args.girvanNewman:
        compute_community(gx, resolutions, args.contingency, args.verbose, args.output, doc2lab, annot_dict)
    else:
        k = 80 # can change k to change "resolution" 
        compute_girvanNewman(gx, k, args.contingency, args.verbose, args.output, doc2lab, annot_dict)


if __name__ == "__main__" :
    main()

