import numpy as np
from tensorboard_logger import configure, log_value
import ipdb

DEF_WINDOW = 100

class EpisodeReporter(object):
	def __init__(self, fname):
		self.stats = []
		self.stats_dict = {}
		self.ids_to_log = []
		configure(fname, flush_secs=5)


	def log_env(self, id):
		self.ids_to_log.append(id)

	def add_stat(self, env_id, steps, reward, total_steps):
		self.stats.append([env_id,steps,reward])
		if not env_id in self.stats_dict.keys():
			self.stats_dict[env_id] = []
		self.stats_dict[env_id].append((steps, reward))
		self.total_steps = total_steps

	def __len__(self):
		return len(self.stats)


	def report_stats(self):
		_, steps, rewards = zip(*self.stats)
		print('Total episodes so far: %d' % len(steps))
		print('Total steps so far: %d' % sum(steps))
		print('Total rewards so far: %f' % sum(rewards))
		totals = sorted([(k,len(val), *zip(*val)) for k, val in self.stats_dict.items()],key=lambda x: -x[1])

		print('Data for the 10 most common envs:')
		for vals in totals[:10]:
			s = vals[2][-DEF_WINDOW:]
			print('Env %d appeared %d times, with moving (100) mean/std %f/%f:' % (vals[0], vals[1], np.mean(s), np.std(s)))
			print(s)
			print('\n\n')

		for id in self.ids_to_log:
			stats = self.stats_dict[id][-DEF_WINDOW:]
			steps, rewards = zip(*stats)
			log_value('env {} reward'.format(id), np.mean(steps), self.total_steps)
