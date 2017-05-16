import tensorflow as tf
import argparse
import logging
import sys
import os
import numpy as np
import pandas as pd

from deep_learning.models.BaseModelTF import BaseModelTF
from deep_learning.recurrent_nn.LSTMCell import LSTMCell
from deep_learning.embeddings.GloveEmbeddings import GloveEmbeddings
from deep_learning.corpus_reader.Tokens import Tokens
from corpus_reader import CorpusReader

__author__ = "roopal_garg"

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s : %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


class QuestionPairVectorDistAngRNNModelTF(BaseModelTF):
    def __init__(self, v, d, m, model_name, save_dir, list_classes, optimizer=tf.train.GradientDescentOptimizer,
                 lr=0.001, max_to_keep=2, clip_norm=5.0, input_dim=[None, None], add_summary_emb=True,
                 size_dense_layer=256, activation=tf.nn.tanh
                 ):
        """
        this model, compares two sentences by concatenating the final hidden states of their representation by running
        them separately through a LSTM. the concatenated representation is them passed through a dense layer with 
        sigmoid non-linearity and then though an output layer with softmax to predict the probabilities.
        :param v: 
        :param d: 
        :param m: 
        :param model_name: 
        :param save_dir: 
        :param list_classes: 
        :param optimizer: 
        :param lr: 
        :param max_to_keep: 
        :param clip_norm: 
        :param input_dim: 
        :param add_summary_emb: 
        """

        super(QuestionPairVectorDistAngRNNModelTF, self).__init__(
            v=v, d=d, model_name=model_name, save_dir=save_dir, list_classes=list_classes, input_dim=input_dim, lr=lr,
            clip_norm=clip_norm, optimizer=optimizer, save_word_emb=True
        )

        self.M = m
        self.K = len(list_classes)
        self.f = activation

        with tf.name_scope("sentence_lstm"):
            self.LayerLSTM_1 = LSTMCell(dim=self.D, hidden_layer=self.M)

        with tf.name_scope("output_layer"):
            self.W_dense_1 = tf.Variable(
                tf.random_uniform([2 * self.M, size_dense_layer], -0.001, 0.001), dtype=tf.float32, name="W_dense_1"
            )
            self.b_dense_1 = tf.Variable(tf.zeros(shape=[size_dense_layer]), dtype=tf.float32, name="b_dense_1")

            self.W_dense_2 = tf.Variable(
                tf.random_uniform([size_dense_layer, size_dense_layer], -0.001, 0.001), dtype=tf.float32,
                name="W_dense_2"
            )
            self.b_dense_2 = tf.Variable(tf.zeros(shape=[size_dense_layer]), dtype=tf.float32, name="b_dense_2")

            self.W_op = tf.Variable(
                tf.random_uniform([size_dense_layer, self.K], -0.001, 0.001), dtype=tf.float32, name="W_op"
            )
            self.b_op = tf.Variable(tf.zeros(shape=[self.K]), dtype=tf.float32, name="b_op")

        self.build_graph()

        self.train_writer = self.add_summary_file_writer()
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.train_writer.add_graph(graph=self.tf_session.graph, global_step=1)

    def loop_layer_1(self, tensor_array_h_last, embeddings, idx_sent):
        elems = tf.gather(embeddings, idx_sent)

        """
        for all words in a sentence
        """
        hidden_cell_states = tf.scan(
            fn=self.LayerLSTM_1.recurrence, elems=elems, initializer=self.LayerLSTM_1.initial_hidden_cell_states,
            name="layer_1_scan"
        )
        h_t, c_t = tf.unstack(hidden_cell_states, axis=1)

        h_t_last = tf.reshape(h_t[-1, :], [1, self.M])

        tensor_array_h_last = tensor_array_h_last.write(idx_sent, h_t_last)
        idx_sent = tf.add(idx_sent, 1)

        return tensor_array_h_last, embeddings, idx_sent

    def build_graph(self):
        """

        :return: 
        """

        """
        the self.input_seq contains 2 items: sentence_1 and sentence_2
        they are padded per the max len between the two
        """
        input_embeddings = tf.nn.embedding_lookup(self.We, self.input_seq)

        tensor_array_sentence_h_last = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False,
            name="tensor_array_sentences"
        )

        """
        for both the sentences
        """

        def loop_layer_1_cond(tensor_array_sentence_h_last, input_embeddings, idx_sent): return tf.less(
            idx_sent, self.current_batch_size
        )

        sentences_h_last, _, _ = tf.while_loop(
            loop_layer_1_cond, self.loop_layer_1, (tensor_array_sentence_h_last, input_embeddings, 0),
            name="loop_sentences"
        )

        sentences_h_last = sentences_h_last.concat()

        """
        compute the angle and the distance between the two vectors
        """

        q1_vec, q2_vec = tf.split(sentences_h_last, num_or_size_splits=2)
        q1_vec = tf.reshape(q1_vec, [self.LayerLSTM_1.M])
        q2_vec = tf.reshape(q2_vec, [self.LayerLSTM_1.M])

        q_vec_angle = tf.multiply(q1_vec, q2_vec, name="vec_angle")
        q_vec_angle = tf.reshape(q_vec_angle, [1, self.LayerLSTM_1.M])
        q_vec_sq_dist = tf.squared_difference(q1_vec, q2_vec, name="vec_sq_dist")
        q_vec_sq_dist = tf.reshape(q_vec_sq_dist, [1, self.LayerLSTM_1.M])

        q_vec_concat = tf.concat([q_vec_angle, q_vec_sq_dist], axis=1, name="q_vec_concat")

        h_concat_drop_op = tf.nn.dropout(q_vec_concat, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("output_layer"):
            dense_layer_op_1 = self.f(
                tf.nn.xw_plus_b(h_concat_drop_op, self.W_dense_1, self.b_dense_1), name="dense_layer_activation_1"
            )
            dense_layer_drp_op_1 = tf.nn.dropout(dense_layer_op_1, keep_prob=self.dropout_keep_prob)

            dense_layer_op_2 = self.f(
                tf.nn.xw_plus_b(dense_layer_drp_op_1, self.W_dense_2, self.b_dense_2), name="dense_layer_activation_2"
            )
            dense_layer_drp_op_2 = tf.nn.dropout(dense_layer_op_2, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("predict"):
            self.logits = tf.nn.xw_plus_b(dense_layer_drp_op_2, self.W_op, self.b_op, name="logits")

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.targets, name="xentropy"
            )

            self.prediction = tf.arg_max(self.logits, dimension=1, name="prediction")
            self.logit_softmax = tf.nn.softmax(self.logits, name="logit_softmax")

            # target_one_hot = tf.one_hot(indices=self.targets, depth=self.K, dtype=tf.float32, axis=-1, name="one_hot")
            # self.loss = tf.losses.log_loss(labels=target_one_hot, predictions=self.logit_softmax)

        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name="global_step", trainable=False)

            trainables = tf.trainable_variables()
            QuestionPairVectorDistAngRNNModelTF.print_trainables(trainables)

            grads = tf.gradients(self.loss, trainables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
            grad_var_pairs = zip(grads, trainables)

            opt = self.optimizer(self.learning_rate)
            self.train_op = opt.apply_gradients(grad_var_pairs, global_step=global_step)


if __name__ == "__main__":

    """
    PYTHONPATH=/home/ubuntu/ds-tws-backend/:/home/ubuntu/quora_question_pairs/ python code/QuestionPairVectorDistAngRNNModelTF.py --mode train --data_dir /home/ubuntu/datasets/quora_question_pairs --emb_path /home/ubuntu/embeddings/glove.6B/glove.6B.300d.txt --model_name question_pair_vector_angle_LSTM_M256_DRP_rmsprop_relu --dropout_keep_prob 0.25 --save_path_pred kaggle_4 --max_to_keep 3 --size_dev_set 15000 --test_every 150000
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--emb_path", default=GloveEmbeddings.GLOVE_DIR, type=str, help="embedding path"
    )
    parser.add_argument(
        "--emb_size", default=300, type=int, help="embedding size"
    )
    parser.add_argument(
        "--data_dir", default="/Users/roopal/workspace/kaggle/quora_question_pairs/data",
        type=str, help="amazon reviews pickle path"
    )
    parser.add_argument(
        "--model_name", default="question_pair_vector_angle_LSTM_M256_DRP_rmsprop_relu", type=str,
        help="name of model"
    )
    parser.add_argument(
        "--M", default=256, type=int, help="m parameter"
    )
    parser.add_argument(
        "--mode", default="train", type=str, help="train or test"
    )
    parser.add_argument(
        "--save_dir", default="train_log", type=str, help="path to save dir"
    )
    parser.add_argument(
        "--dropout_keep_prob", default=1.0, type=float, help="dropout keep probability"
    )
    parser.add_argument(
        "--size_test_set", default=None, type=int, help="size of test set"
    )
    parser.add_argument(
        "--num_epox", default=30, type=int, help="num of epochs"
    )
    parser.add_argument(
        "--log_every", default=None, type=int, help="num of steps to log after"
    )
    parser.add_argument(
        "--test_every", default=10000, type=int, help="num of steps to test after"
    )
    parser.add_argument(
        "--size_dev_set", default=7000, type=int, help="dev set size"
    )
    parser.add_argument(
        "--save_path_pred", default="kaggle", type=str, help="save file for predictions"
    )
    parser.add_argument(
        "--size_dense_layer", default=256, type=int, help="dim of dense layer"
    )
    parser.add_argument(
        "--max_to_keep", default=2, type=int, help="max models to keep"
    )

    args = parser.parse_args()

    dim = args.emb_size

    """
    the data_dir contains 2 pickle files with the train and test data
    """
    data_path_train = os.path.join(args.data_dir, "train.csv.pkl")
    data_path_test = os.path.join(args.data_dir, "test.csv.pkl")

    model_name = args.model_name
    emb_path = args.emb_path
    save_dir = args.save_dir

    logging.info("loading word emb")
    word2idx, embedding_matrix = GloveEmbeddings.get_embeddings_with_custom_tokens(path=emb_path, embedding_dim=dim)
    vocab_size = len(word2idx)
    logging.info("word emb loaded: {}".format(vocab_size))

    logging.info("loading dataset")
    X_train, Y_train, X_dev, Y_dev, X_test, _ = CorpusReader.get_question_pair_data(data_path_train, data_path_test)

    """
    trim the test set is desired
    """
    if args.size_test_set:
        X_test = X_test[:args.size_test_set]

    assert (len(X_train) == len(Y_train)), "Train data and label size mismatch"
    logging.info("train size: {}, test size: {}, dev size: {}".format(len(X_train), len(X_test), len(X_dev)))
    logging.info("loaded dataset")

    list_classes = ["0", "1"]
    model = QuestionPairVectorDistAngRNNModelTF(
        v=vocab_size, d=dim, m=args.M, model_name=model_name, save_dir=save_dir, list_classes=list_classes,
        optimizer=tf.train.AdamOptimizer, lr=0.0001, max_to_keep=args.max_to_keep, clip_norm=5.0,
        input_dim=[None, None], add_summary_emb=True, size_dense_layer=args.size_dense_layer, activation=tf.nn.relu
    )

    if args.mode == "train":
        logging.info("beginning training")
        model.fit(
            embedding_matrix, X_train, Y_train, X_dev, Y_dev, epochs=args.num_epox, reg=1.0, print_log=True,
            keep_prob=args.dropout_keep_prob, log_every=args.log_every, net_epoch_step_idx=None, save_every=None,
            tf_log=False, test_every=args.test_every, progress_bar=True, size_dev_set=args.size_dev_set,
            save_on_higher_acc=False, save_on_lower_loss=True
        )
    elif args.mode == "test":
        logging.info("beginning testing")
        """
        setting insert_eos True to take care of empty sentences in the data set
        """
        model.insert_eos = True
        model.eos_idx = word2idx[Tokens.EOS]
        Y_test = list(np.zeros(len(X_test), dtype=int))

        accuracy_test, loss_test, y_pred, list_softmax_py_x = model.test(
            X_test, Y_test, embedding_matrix, progress_bar=True
        )
        logging.info(
            "Accuracy {accuracy_test} Loss {loss_test}".format(accuracy_test=accuracy_test, loss_test=loss_test)
        )

        list_softmax_py_x = [(item[0][0], item[0][1]) for item in list_softmax_py_x]
        logging.debug(list_softmax_py_x)
        df = pd.DataFrame.from_records(list_softmax_py_x, columns=["no_duplicate", "is_duplicate"])
        df = df[["is_duplicate"]]
        df.to_csv(args.save_path_pred, index_label="test_id")
    else:
        sys.exit("invalid mode. use test or train")

    logging.info("done...")
