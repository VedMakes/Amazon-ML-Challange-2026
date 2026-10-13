"""Settings of the submitted run. Step 2 and step 3 must use the same values: the cross-encoder scores the
development samples and the test chunks that the matcher uses. Change a value here, then run the steps again."""

SEED = 20260926          # component-disjoint splits and all samples
DENSE_TRAIN_MOD = 10     # S1 records with eid % 10 == 0 train the bi-encoder and the cross-encoder (never used by the matcher)
K = 20                   # dense candidates for each S1 and each target source (40 candidates for each S1)
R = 5                    # length of the reverse list (for each target: its best S1 records)
CE_TOPK = 20             # the cross-encoder scores the best 20 of the 40 candidates of each S1
N_TRAIN = 60_000         # S1 records that train the matcher
N_VAL = 6_000            # S1 records of inner validation (model and rule selection)
N_OUTER = 6_000          # S1 records of the outer holdout (one final check)
TEST_QCHUNK = 50_000     # S1 records for each test chunk (37 chunks)
MAX_PAIRS = 500_000      # candidate pairs for each feature job
