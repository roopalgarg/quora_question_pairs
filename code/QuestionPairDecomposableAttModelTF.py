import logging
import os
import sys
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf

from deep_learning.models.BaseModelTF import BaseModelTF
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


class QuestionPairDecomposableAttModelTF(BaseModelTF):
    def __init__(self, v, d, m, model_name, save_dir, list_classes, optimizer=tf.train.GradientDescentOptimizer,
                 lr=0.001, max_to_keep=2, clip_norm=5.0, input_dim=[None, None], add_summary_emb=True,
                 activation=tf.nn.relu
                 ):
        """
        
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
        :param activation: 
        """

        super(QuestionPairDecomposableAttModelTF, self).__init__(
            v=v, d=d, model_name=model_name, save_dir=save_dir, list_classes=list_classes, input_dim=input_dim, lr=lr,
            clip_norm=clip_norm, optimizer=optimizer, save_word_emb=True
        )

        self.M = m
        self.K = len(list_classes)
        self.f = activation

        with tf.name_scope("attend_layer"):
            self.Wf = tf.Variable(
                tf.random_uniform([self.D, self.D], -0.001, 0.001), dtype=tf.float32, name="Wf"
            )
            self.bf = tf.Variable(tf.zeros(shape=[self.D]), dtype=tf.float32, name="bf")

        with tf.name_scope("compare_layer"):
            self.Wg = tf.Variable(
                tf.random_uniform([2*self.D, self.D], -0.001, 0.001), dtype=tf.float32, name="Wg"
            )
            self.bg = tf.Variable(tf.zeros(shape=[self.D]), dtype=tf.float32, name="bg")

        with tf.name_scope("aggregate_layer"):
            self.Wh = tf.Variable(
                tf.random_uniform([2*self.D, self.K], -0.001, 0.001), dtype=tf.float32, name="Wh"
            )
            self.bh = tf.Variable(tf.zeros(shape=[self.K]), dtype=tf.float32, name="bh")

        self.q1_rep = None
        self.q2_rep = None
        self.e_ij = None

        self.build_graph()

        self.train_writer = self.add_summary_file_writer()
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.train_writer.add_graph(graph=self.tf_session.graph, global_step=1)

    def loop_e_ij(self, tensor_array_e_ij, q1_rep, idx):
        e_ij = tf.matmul(q1_rep, self.q2_rep, transpose_b=True)

        tensor_array_e_ij = tensor_array_e_ij.write(idx, e_ij)
        idx = tf.add(idx, 1)

        return tensor_array_e_ij, q1_rep, idx

    def build_graph(self):
        """
        
        :return: 
        """

        """
         the self.input_seq contains 2 items: sentence_1 and sentence_2
         they are padded per the max len between the two
         """
        input_embeddings = tf.nn.embedding_lookup(self.We, self.input_seq)
        q1_emb, q2_emb = tf.split(input_embeddings, num_or_size_splits=2)
        q1_emb = tf.squeeze(q1_emb)
        q2_emb = tf.squeeze(q2_emb)

        max_len = tf.reduce_max(self.input_lengths, name="max_len")
        q1_len, q2_len = tf.split(self.input_lengths, num_or_size_splits=2, name="q_len")

        q1_len = tf.squeeze(q1_len)
        q2_len = tf.squeeze(q2_len)

        begin_idx_q1 = max_len - q1_len
        begin_idx_q2 = max_len - q2_len

        q1_emb = tf.slice(q1_emb, begin=[begin_idx_q1, 0], size=[-1, -1], name="q1_emb")
        q2_emb = tf.slice(q2_emb, begin=[begin_idx_q2, 0], size=[-1, -1], name="q2_emb")

        self.q1_rep = self.f(tf.nn.xw_plus_b(q1_emb, self.Wf, self.bf, name="q1_rep"))
        self.q2_rep = self.f(tf.nn.xw_plus_b(q2_emb, self.Wf, self.bf, name="q2_rep"))

        with tf.name_scope("attend_layer"):
            """
            1.1 calculate the un-normalized attention weights e_ij
            """
            # TODO: optimize which to multiple and which to recur by

            tensor_array_e_ij = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False,
                name="e_ij_weights"
            )

            def loop_cond(tensor_array_e_ij, q1_rep, idx): return tf.less(idx, q1_len)
            tensor_array_e_ij, _, _ = tf.while_loop(
                cond=loop_cond, body=self.loop_e_ij, loop_vars=(tensor_array_e_ij, self.q1_rep, 0), name="loop_e_ij"
            )

            """
            e_ij : [q1_len, q2_len]
            """
            self.e_ij = tensor_array_e_ij.concat()

            """
            1.2 normalize the e_ij weights
            
            betas_op: softmax with dim=-1 => normalize across each row
            alphas_op: softmax with dim=0 => normalize across each column
            """

            betas_sftmx = tf.nn.softmax(self.e_ij, dim=-1, name="betas_sftmx")
            alphas_sftmx = tf.nn.softmax(self.e_ij, dim=0, name="alphas_sftmx")

            betas_op = betas_sftmx * q2_emb
            alphas_op = alphas_sftmx * q1_emb

            """
            beta_i: len(beta_i) = q1_len
            alpha_j: len(alpha_j) = q2_len
            """

            beta_i = tf.reduce_sum(betas_op, axis=1, name="beta_i")
            alpha_j = tf.reduce_sum(alphas_op, axis=0, name="alpha_j")

        with tf.name_scope("compare_layer"):

            q1_compare = tf.concat([q1_emb, beta_i], axis=1, name="q1_compare")
            q2_compare = tf.concat([q2_emb, alpha_j], axis=1, name="q2_compare")

            v1_i = self.f(tf.nn.xw_plus_b(q1_compare, self.Wg, self.bg, name="v1_i"))
            v2_j = self.f(tf.nn.xw_plus_b(q2_compare, self.Wg, self.bg, name="v2_j"))

        with tf.name_scope("aggregate_layer"):

            v1 = tf.reduce_sum(v1_i, axis=0, keep_dims=True)
            v2 = tf.reduce_sum(v2_j, axis=0, keep_dims=True)

            v = tf.concat([v1, v2], axis=1, name="v")

            y_hat = tf.nn.xw_plus_b(v, self.Wh, self.bh, name="logits")

        with tf.name_scope("predict"):
            self.logits = y_hat

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.targets, name="xentropy"
            )

            self.prediction = tf.arg_max(self.logits, dimension=1, name="prediction")
            self.logit_softmax = tf.nn.softmax(self.logits, name="logit_softmax")

        with tf.name_scope("train_op"):
            global_step = tf.Variable(0, name="global_step", trainable=False)

            trainables = tf.trainable_variables()
            QuestionPairDecomposableAttModelTF.print_trainables(trainables)

            grads = tf.gradients(self.loss, trainables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
            grad_var_pairs = zip(grads, trainables)

            opt = self.optimizer(self.learning_rate)
            self.train_op = opt.apply_gradients(grad_var_pairs, global_step=global_step)


if __name__ == "__main__":

    """
    PYTHONPATH=/home/ubuntu/ds-tws-backend/:/home/ubuntu/quora_question_pairs/ python code/QuestionPairDecomposableAttModelTF.py --mode train --data_dir /home/ubuntu/datasets/quora_question_pairs --emb_path /home/ubuntu/embeddings/glove.6B/glove.6B.300d.txt --model_name question_pair_datt --dropout_keep_prob 0.4 --save_path_pred kaggle_8 --max_to_keep 3 --size_dev_set 15000 --test_every 100000
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
        type=str, help="quora question pairs pickle path"
    )
    parser.add_argument(
        "--model_name", default="question_pair_datt", type=str,
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
    model = QuestionPairDecomposableAttModelTF(
        v=vocab_size, d=dim, m=args.M, model_name=model_name, save_dir=save_dir, list_classes=list_classes,
        optimizer=tf.train.RMSPropOptimizer, lr=0.0001, max_to_keep=args.max_to_keep, clip_norm=5.0,
        input_dim=[None, None], add_summary_emb=True, activation=tf.nn.relu
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
