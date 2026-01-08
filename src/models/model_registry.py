from .numerical.random_forest_clf import create_random_forest_clf
from .numerical.hist_grad_boosting_clf import create_grad_boosting_clf
from .numerical.lstm import create_lstm
from .numerical.elastic_net import create_elastic_net
from .numerical.rnn import create_rnn
from .textual.text_attention_net import create_text_attention_net

MODEL_REGISTRY = {
    "RandomForestClassifier": create_random_forest_clf,
    "ElasticNet": create_elastic_net,
    "GradientBoostingClassifier": create_grad_boosting_clf,
    "LSTM": create_lstm,
    "RNN": create_rnn,
    "TextAttentionNet": create_text_attention_net,
}
