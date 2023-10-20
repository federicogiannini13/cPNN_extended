import river
from river import tree
from evaluation.learner_config import LearnerConfig
from models.clstm import cLSTMLinear
from models.cpnn import cPNN
from models.mcrnn import mcRNN
from models.temporally_augmented_classifier import TemporallyAugmentedClassifier


NUM_OLD_LABELS = 0
SEQ_LEN = 0
NUM_FEATURES = 0
BATCH_SIZE = 0
ITERATIONS = 0
eval_cl = None
eval_preq = None


def initialize(num_old_labels_, seq_len_, num_features_, batch_size_, iterations_):
    global NUM_OLD_LABELS, SEQ_LEN, NUM_FEATURES, BATCH_SIZE, ITERATIONS, eval_cl, eval_preq
    NUM_OLD_LABELS = num_old_labels_
    SEQ_LEN = seq_len_
    NUM_FEATURES = num_features_
    BATCH_SIZE = batch_size_
    ITERATIONS = iterations_


def initialize_callback(eval_cl_, eval_preq_):
    global eval_preq, eval_cl
    eval_cl = eval_cl_
    eval_preq = eval_preq_


def create_hat():
    return tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=100,
        split_confidence=1e-5,
        leaf_prediction="nb",
        nb_threshold=10,
    )


def create_hat_ta():
    return TemporallyAugmentedClassifier(
        base_learner=create_hat(),
        num_old_labels=NUM_OLD_LABELS,
    )


def create_arf():
    return river.ensemble.AdaptiveRandomForestClassifier(leaf_prediction="nb")


def create_arf_ta():
    return TemporallyAugmentedClassifier(
        base_learner=create_arf(),
        num_old_labels=NUM_OLD_LABELS,
    )


def create_cpnn():
    return cPNN(
        column_class=cLSTMLinear,
        device="cpu",
        seq_len=SEQ_LEN,
        train_verbose=False,
        input_size=NUM_FEATURES,
        output_size=2,
        hidden_size=50,
        batch_size=BATCH_SIZE,
    )


def create_mclstm():
    return mcRNN(
        column_class=cLSTMLinear,
        input_size=NUM_FEATURES,
        output_size=2,
        hidden_size=50,
        batch_size=BATCH_SIZE,
        device="cpu",
        seq_len=SEQ_LEN,
        train_verbose=False,
    )


anytime_learners_sml = [
    LearnerConfig(
        name="ARF",
        model=create_arf,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    LearnerConfig(
        name="ARF_TA",
        model=create_arf_ta,
        numeric=False,
        batch_learner=False,
        drift=False,
        cpnn=False,
    ),
    # LearnerConfig(
    #     name="HAT",
    #     model=create_hat,
    #     numeric=False,
    #     batch_learner=False,
    #     drift=False,
    #     cpnn=False,
    # ),
    # LearnerConfig(
    #     name="HAT_TA",
    #     model=create_hat_ta,
    #     numeric=False,
    #     batch_learner=False,
    #     drift=False,
    #     cpnn=False,
    # ),
]

batch_learners_cpnn = [
    LearnerConfig(
        name="cPNN",
        model=create_cpnn,
        numeric=True,
        batch_learner=True,
        drift=True,
        cpnn=True,
    ),
    LearnerConfig(
        name="cLSTM",
        model=create_cpnn,
        numeric=True,
        batch_learner=True,
        drift=False,
        cpnn=True,
    ),
    LearnerConfig(
        name="mcLSTM",
        model=create_mclstm,
        numeric=True,
        batch_learner=True,
        drift=False,
        cpnn=True,
    ),
]
