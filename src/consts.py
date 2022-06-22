FLOAT32 = 'float32'
MLFLOW_LOG = 'log_to_mlflow'
TENSORBOARD_LOGDIR = 'tensorfboard_log'

#TODO: extract all strings from experiments to consts!

# LOGGING PARAMS CONSTS


POLICY_ARCH = 'policy_arch'
POLICY_ACT_F = 'policy_act_f'
POLICY_OUT_ACT_F = 'policy_out_act_f'

VALUE_ARCH = 'value_arch'
VALUE_ACT_F = 'value_act_f'
VALUE_OUT_ACT_F = 'value_out_act_f'

ENTROPY_VAL = 'entropy_val'
ADVANTAGE_F = 'advantage'

POLICY_OPT = 'policy_opt'
VALUE_OPT = 'value_opt'

W_INITIALIZER = 'w_initializer'
LEARNING_RATE = 'lr'
LEARNING_RATE_POLICY = 'lr_policy'
LEARNING_RATE_VALUE = 'lr_value'

CLIPPING_EPS = 'clipping_eps'
N_AGENTS = 'n_agents'
N_STEPS = 'n_steps'

EP_MAX_STEPS = 'ep_max_steps'

BATCH_SIZE = 'batch_size'
NEPISODES = 'nepisodes'
PRINT_INTERVAL = 'print_interval'
LAMBDA = 'lambda_'
GAMMA = 'gamma'

CRITIC_ARCH = 'critic_arch'
CRITIC_A_SZ = 'critic_a_size'
CRITIC_S_SZ = 'critis_s_size'
CRITIC_OPT = 'critic_opt'
WARMUP_BATCHES = 'warmup_batches'

EPSILON = 'epsilon'

Q_NET_ARCH = 'q_net_arch'
HIDDEN_ACT_F = 'hidden_activation_function'
OUTOUT_ACT_F = 'output_activation_function'
BATCH_NORM = 'batch_norm'

# Experiment consts
MODEL_COL = 'model'
SCORE_COL = 'score'
PAIRWISE_RESULT_NAME = 'pariwise_test_result'
OVERALL_RESULT_NAME = 'overall_result'
HIST_NAME = 'scores_hist'
BOXPLOT_NAME = 'scores_boxplot'
SCORES_NAME = 'scores'
