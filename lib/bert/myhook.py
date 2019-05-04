import tensorflow as tf
from tensorflow.python import debug as tf_debug
class evalute_hook(tf.train.SessionRunHook):
    def __init__(self,handle,feed_handle,run_op,evl_step):
        self.handle = handle
        self.feed_handle = feed_handle
        self.run_op = run_op
        self.evl_step = evl_step
        self.step = 0
    def after_create_session(self, session, coord):
        super().after_create_session(session, coord)
        if self.step % self.evl_step ==0:
            self.evl_loss = session.run(fetches=self.run_op,feed_dict={self.handle:self.feed_handle})
            tf.logging.info("step:{}********evl_loss:{}".format(self.step,self.evl_loss))
        self.step += 1


class train_hook(tf.train.SessionRunHook):
    def __init__(self,handle,feed_handle):
        self.handle = handle
        self.feed_handle = feed_handle

    def before_run(self, run_context):
        super().before_run(run_context)
        return tf.train.SessionRunArgs(fetches=None,feed_dict={self.handle:self.feed_handle})

class tensor_filter(tf.train.SessionRunHook):
    def before_run(self, run_context):
        super().before_run(run_context)
        run_context.session.add_tensor_filter("nan_and_inf",tf_debug.has_inf_or_nan)

