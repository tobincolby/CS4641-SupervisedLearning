To run the analysis, first enter the project directory. Make sure that numpy,
scipy, and scikit-learn are installed. Then run:

1. "python split.py"        (to split data into train/test)
2. "python cv.py"           (to run 5-fold cross validation and generate optimal models)
3. "python visualize.py"    (to produce cv graphs)
4. "python analyze.py"      (to create/graph learning curves from optimal models)

This will produce all the classifiers, data, and graphics that were used for the
analysis of the algorithms.
