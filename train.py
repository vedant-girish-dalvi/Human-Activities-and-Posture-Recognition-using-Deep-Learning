import gin
import tensorflow as tf
from tensorflow import keras
import logging
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_test, ds_info, run_paths, total_steps, log_interval, ckpt_interval):

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Loss objective
        self.loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = keras.optimizers.Adam()

        # Metrics
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = keras.metrics.Mean(name='test_loss')
        self.test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.t_loss = keras.metrics.Mean(name='t_loss')
        self.t_accuracy = keras.metrics.SparseCategoricalAccuracy(name='t_accuracy')

        # Summary Writer
        
        self.train_summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_train"])
        self.test_summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_eval"])
        self.image_summary_writer = tf.summary.create_file_writer(self.run_paths["path_summary_image"])

        # Checkpoint Manager
        
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(
            1), optimizer=self.optimizer, net=self.model)
        self.manager = tf.train.CheckpointManager(
            self.ckpt, self.run_paths["path_ckpts_train"], max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def test1_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.t_loss(t_loss)
        self.t_accuracy(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1

            self.train_step(images, labels)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_state()
                self.test_accuracy.reset_state()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # Write summary to tensorboard
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)

                # Reset train metrics
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()
                logging.info("saved checkpoint for step {} ".format(int(step)))

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint

                logging.info("Evaluating")
                for test_images, test_labels in self.ds_test:
                    self.test1_step(test_images, test_labels)

                template = 'Test1 Loss: {}, Test1 Accuracy: {}'
                logging.info(template.format(
                                             self.t_loss.result(),
                                             self.t_accuracy.result() * 100))
                
                self.manager.save()
                logging.info("saved checkpoint for final step {} ".format(int(step)))
                return self.test_accuracy.result().numpy()
