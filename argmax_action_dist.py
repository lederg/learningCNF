from ray.rllib.models.torch.torch_action_dist import TorchCategorical

class TorchCategoricalArgmax(TorchCategorical):
	def sample(self):
		return self.dist.logits.argmax().unsqueeze(0)

