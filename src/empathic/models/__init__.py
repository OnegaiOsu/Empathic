"""Model package exports."""

from .classical import (  # noqa: F401
    ClassicalModel,
    default_classical_models,
    make_baseline,
    make_logistic_regression,
    make_random_forest,
    make_xgboost,
)
from .conformer import (  # noqa: F401
    Conformer, ConformerConfig,
    TinyTCN, TinyTCNConfig,
    BiLSTM, BiLSTMConfig,
    CNN1D, CNN1DConfig,
    count_parameters,
)
from .advanced import (  # noqa: F401
    DANNConformer, DANNConformerConfig,
    TSTCCEncoder, TSTCCConfig,
    MLPProbe,
    GradientReversalLayer, grad_reverse,
    tstcc_weak_aug, tstcc_strong_aug,
    nt_xent_loss, temporal_contrast_loss,
)
from .multistream import (  # noqa: F401
    MultiStream, MultiStreamConfig, EMOWORK_GROUPS,
)
