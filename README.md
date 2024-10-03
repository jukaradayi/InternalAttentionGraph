Graph Internal
==============

I Requirements
--------------

* pip install numpy networkx habanero scikit-learn

II Usage
--------

* Get list of arguments :
    ```python commonCitationsCommunities.py -h```

* The script doesn't write the graph by default, call with --write_graph to write it in a format readable by Gephi.

* You can write the graph and load it again using the --dump and --load commands. While loading, you still need the .csv file.

* Use -c or --contingency to write the contingency matrix for all resolution steps.

* To compute community detection at a single resolution, set -rm and -rM to that resolution, e.g.: 
    ```python  commonCitationsCommunities.py --output modularity --contingency -rm 1.1 -rM 1.1```

* By default, the script runs a Greedy Community detection algorithm. The resolutions used go from min resolution 0.5 to max resolution 1.5 with a resolution step of 0.1. Call -rm -rM or -rS to change either of those values;

* Usage example:
    ```python  commonCitationsCommunities.py --output modularity --contingency  --write_graph -rm 0.8 -rM 1.2 -rS 0.1```

III results
