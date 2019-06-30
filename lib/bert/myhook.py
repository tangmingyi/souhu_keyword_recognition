import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
import json

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

class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class run_time_summary_hook(tf.train.SessionRunHook):

    def __init__(self, summary):
        self.step = 0
        self.summary = summary

    def before_run(self, run_context):
        if self.step == 0:
            graph = run_context.session.graph
            self.writer = tf.summary.FileWriter("test_summary", graph)
        if self.step % 100 == 99:
            self.add_summary = True
            self.step += 1
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            return tf.train.SessionRunArgs(fetches=self.summary, options=self.run_options)
        self.step += 1
        self.add_summary = False

    def after_run(self, run_context, run_values):
        if self.add_summary:
            self.writer.add_run_metadata(run_metadata=run_values.run_metadata, tag="step%s" % self.step)
            self.writer.add_summary(run_values.results, self.step)


class timeline_hook(tf.train.SessionRunHook):
    def __init__(self, with_one_timeline=False):
        self.step = 0
        self.with_one_timeline = with_one_timeline
        self.multimeline = TimeLiner()

    def before_run(self, run_context):
        self.step += 1
        if self.step % 100 == 99:
            self.towriter = True
            return tf.train.SessionRunArgs(fetches=None, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
        self.towriter = False

    def after_run(self, run_context, run_values):
        if self.towriter:
            fetched_timeline = timeline.Timeline(run_values.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            if self.with_one_timeline:
                self.multimeline.update_timeline(chrome_trace)
            else:
                with open("test_summary/timeline_step_%s.json" % self.step, 'w') as f:
                    f.write(chrome_trace)

    def end(self, session):
        if self.with_one_timeline:
            self.multimeline.save("test_summary/timeline_step_%s.json" % self.step)



