import numpy as np
from tensorboard_logger import configure, log_value
import ipdb

class EpisodeReporter(object):
	def __init__(self):
		self.stats = []
		self.stats_dict = {}

	def add_stat(self, env_id, steps, reward):
		self.stats.append([env_id,steps,reward])
		if not env_id in self.stats_dict.keys():
			self.stats_dict[env_id] = []
		self.stats_dict[env_id].append((steps, reward))

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
			print('Env %d appeared %d times, with mean/std %f/%f:' % (vals[0], vals[1], np.mean(vals[2]), np.std(vals[2])))
			print(vals[2])
			print('\n\n')