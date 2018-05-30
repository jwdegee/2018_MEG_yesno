[0;31mSignature:[0m [0mmne[0m[0;34m.[0m[0mstats[0m[0;34m.[0m[0mpermutation_cluster_test[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mthreshold[0m[0;34m=[0m[0mNone[0m[0;34m,[0m [0mn_permutations[0m[0;34m=[0m[0;36m1024[0m[0;34m,[0m [0mtail[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0mstat_fun[0m[0;34m=[0m[0mNone[0m[0;34m,[0m [0mconnectivity[0m[0;34m=[0m[0mNone[0m[0;34m,[0m [0mn_jobs[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mseed[0m[0;34m=[0m[0mNone[0m[0;34m,[0m [0mmax_step[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mexclude[0m[0;34m=[0m[0mNone[0m[0;34m,[0m [0mstep_down_p[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m [0mt_power[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mout_type[0m[0;34m=[0m[0;34m'mask'[0m[0;34m,[0m [0mcheck_disjoint[0m[0;34m=[0m[0mFalse[0m[0;34m,[0m [0mbuffer_size[0m[0;34m=[0m[0;36m1000[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0mNone[0m[0;34m)[0m[0;34m[0m[0m
[0;31mDocstring:[0m
Cluster-level statistical permutation test.

For a list of nd-arrays of data, e.g. 2d for time series or 3d for
time-frequency power values, calculate some statistics corrected for
multiple comparisons using permutations and cluster level correction.
Each element of the list X contains the data for one group of
observations. Randomized data are generated with random partitions
of the data.

Parameters
----------
X : list
    List of nd-arrays containing the data. Each element of X contains
    the samples for one group. First dimension of each element is the
    number of samples/observations in this group. The other dimensions
    are for the size of the observations. For example if X = [X1, X2]
    with X1.shape = (20, 50, 4) and X2.shape = (17, 50, 4) one has
    2 groups with respectively 20 and 17 observations in each.
    Each data point is of shape (50, 4).
threshold : float | dict | None
    If threshold is None, it will choose an F-threshold equivalent to
    p < 0.05 for the given number of observations (only valid when
    using an F-statistic). If a dict is used, then threshold-free
    cluster enhancement (TFCE) will be used, and it must have keys
    ``'start'`` and ``'step'`` to specify the integration parameters,
    see the :ref:`TFCE example <tfce_example>`.
n_permutations : int
    The number of permutations to compute.
tail : -1 or 0 or 1 (default = 0)
    If tail is 0, the statistic is thresholded on both sides of
    the distribution.
    If tail is 1, the statistic is thresholded above threshold.
    If tail is -1, the statistic is thresholded below threshold, and
    the values in ``threshold`` must correspondingly be negative.
stat_fun : callable | None
    Function called to calculate statistics, must accept 1d-arrays as
    arguments (default None uses :func:`mne.stats.f_oneway`).
connectivity : sparse matrix.
    Defines connectivity between features. The matrix is assumed to
    be symmetric and only the upper triangular half is used.
    Default is None, i.e, a regular lattice connectivity.
n_jobs : int
    Number of permutations to run in parallel (requires joblib package).
seed : int | instance of RandomState | None
    Seed the random number generator for results reproducibility.
max_step : int
    When connectivity is a n_vertices x n_vertices matrix, specify the
    maximum number of steps between vertices along the second dimension
    (typically time) to be considered connected. This is not used for full
    or None connectivity matrices.
exclude : boolean array or None
    Mask to apply to the data to exclude certain points from clustering
    (e.g., medial wall vertices). Should be the same shape as X. If None,
    no points are excluded.
step_down_p : float
    To perform a step-down-in-jumps test, pass a p-value for clusters to
    exclude from each successive iteration. Default is zero, perform no
    step-down test (since no clusters will be smaller than this value).
    Setting this to a reasonable value, e.g. 0.05, can increase sensitivity
    but costs computation time.
t_power : float
    Power to raise the statistical values (usually F-values) by before
    summing (sign will be retained). Note that t_power == 0 will give a
    count of nodes in each cluster, t_power == 1 will weight each node by
    its statistical score.
out_type : str
    For arrays with connectivity, this sets the output format for clusters.
    If 'mask', it will pass back a list of boolean mask arrays.
    If 'indices', it will pass back a list of lists, where each list is the
    set of vertices in a given cluster. Note that the latter may use far
    less memory for large datasets.
check_disjoint : bool
    If True, the connectivity matrix (or list) will be examined to
    determine of it can be separated into disjoint sets. In some cases
    (usually with connectivity as a list and many "time" points), this
    can lead to faster clustering, but results should be identical.
buffer_size: int or None
    The statistics will be computed for blocks of variables of size
    "buffer_size" at a time. This is option significantly reduces the
    memory requirements when n_jobs > 1 and memory sharing between
    processes is enabled (see set_cache_dir()), as X will be shared
    between processes and each process only needs to allocate space
    for a small block of variables.
verbose : bool, str, int, or None
    If not None, override default verbose level (see :func:`mne.verbose`
    and :ref:`Logging documentation <tut_logging>` for more).

Returns
-------
F_obs : array, shape (n_tests,)
    Statistic (F by default) observed for all variables.
clusters : list
    List type defined by out_type above.
cluster_pv : array
    P-value for each cluster
H0 : array, shape (n_permutations,)
    Max cluster level stats observed under permutation.

References
----------
.. [1] Maris/Oostenveld (2007), "Nonparametric statistical testing of
   EEG- and MEG-data" Journal of Neuroscience Methods,
   Vol. 164, No. 1., pp. 177-190. doi:10.1016/j.jneumeth.2007.03.024.
[0;31mFile:[0m      ~/repos/mne-python/mne/stats/cluster_level.py
[0;31mType:[0m      function
