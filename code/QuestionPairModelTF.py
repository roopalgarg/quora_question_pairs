import tensorflow as tf


from deep_learning.models.BaseModelTF import BaseModelTF
from deep_learning.recurrent_nn.LSTMCell import LSTMCell


class QuestionPairModelTF(BaseModelTF):

    def __init__(self, v, d, m, model_name, save_dir, list_classes, optimizer=tf.train.GradientDescentOptimizer,
                 lr=0.001, max_to_keep=2, clip_norm=5.0, input_dim=[None, None], add_summary_emb=True
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
        """

        super(QuestionPairModelTF, self).__init__(
            v=v, d=d, model_name=model_name, save_dir=save_dir, list_classes=list_classes, max_to_keep=max_to_keep,
            input_dim=input_dim, lr=lr, clip_norm=clip_norm, optimizer=optimizer, save_word_emb=True
        )

        self.M = m
        self.K = len(list_classes)

        with tf.name_scope("sentence_lstm"):
            self.LayerLSTM_1 = LSTMCell(dim=self.D, hidden_layer=self.M)

        with tf.name_scope("output_layer"):
            self.W_1 = tf.Variable(tf.random_uniform([self.M, self.M], -0.001, 0.001), dtype=tf.float32, name="W_1")
            self.b_1 = tf.Variable(tf.zeros(shape=[self.M]), dtype=tf.float32, name="b_1")
            self.W_2 = tf.Variable(tf.random_uniform([self.M, self.K], -0.001, 0.001), dtype=tf.float32, name="W_2")
            self.b_2 = tf.Variable(tf.zeros(shape=[self.K]), dtype=tf.float32, name="b_2")

        self.build_graph()

        self.train_writer = self.add_summary_file_writer()
        self.saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.train_writer.add_graph(graph=self.tf_session.graph, global_step=1)

    def build_graph(self):
        input_embeddings = tf.nn.embedding_lookup(self.We, self.input_seq)

        tensor_array_sentence_h_last = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False,
            name="tensor_array_sentences"
        )

        """
        for all sentences in a paragraph
        """
        def loop_layer_1_cond(tensor_array_sentence_h_last, input_embeddings, idx_sent): return tf.less(
            idx_sent, self.current_batch_size
        )
        sentences_h_last, _, _ = tf.while_loop(
            loop_layer_1_cond, self.LayerLSTM_1, (tensor_array_sentence_h_last, input_embeddings, 0),
            name="loop_sentences"
        )

        sentences_h_last = sentences_h_last.concat()
        h_concat = tf.reshape(
            sentences_h_last, [1, 2*self.LayerLSTM_1.M], name="sentences_h_last"
        )






