import gin
import logging
import tensorflow as tf
from absl import app, flags
from input_pipeline.dataset import load
from train import Trainer
from evaluation.eval import evaluate

from utils import utils_params, utils_misc
from models.rnn_model import rnn_model
from keras.utils import plot_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

FLAGS = flags.FLAGS
flags.DEFINE_string('mode', default='train',
                    help="Choose from ['train', 'evaluate']")
flags.DEFINE_string('model_name', default='gru',
                    help="Choose from ['lstm', 'gru']")

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    if FLAGS.model_name == 'gru':
        gin.parse_config_files_and_bindings(['/home/RUS_CIP/st179677/project/HAPT/configs/config_gru.gin'], [])
        checkpoint_path = '/home/RUS_CIP/st179677/project/HAPT/checkpoints/gru/ckpts'
    else:
        gin.parse_config_files_and_bindings(['/home/RUS_CIP/st179677/project/HAPT/configs/config_lstm.gin'], [])
        checkpoint_path = '/home/RUS_CIP/st179677/project/HAPT/checkpoints/lstm/ckpts'

    utils_params.save_config(run_paths['path_gin'], gin.operative_config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load(True, "hapt", "/home/RUS_CIP/st179677/project/", 250, 125)

    # model
    model = rnn_model(ds_info=ds_info)
    model.summary()
    plot_model(model, to_file="model_fig.png", show_shapes=True)

    if FLAGS.mode == 'train':
        trainer = Trainer(model, ds_train, ds_val, ds_test, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
       
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
           optimizer=tf.keras.optimizers.Adam(), net=model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
        model.trainable = False
        evaluate(model, checkpoint, ds_test, checkpoint_path, run_paths, 12)


if __name__ == "__main__":
    app.run(main)
