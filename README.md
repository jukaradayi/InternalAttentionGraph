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

* Use -c or --contingency to write the contingency matrix for all resolution steps.

* By default, the script runs a Greedy Community detection algorithm. The resolutions used go from min resolution 0.5 to max resolution 1.5 with a resolution step of 0.1. Call -rm -rM or -rS to change either of those values;

* Call with --girvanNewman to run with a Girvan Newman community detection. By default it runs it with k=80, edit the code directly to change that value.

* Usage example:
    ```python  commonCitationsCommunities.py --output modularity --contingency  --write_graph -rm 0.7 -rM 1.7 -rS 0.2```

III results
