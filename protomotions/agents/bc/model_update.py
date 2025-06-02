def forward(self, batch_dict):
    """
    Defines the forward pass of the network

    :param observation: observation(s) to query the policy
    :return:
        action: sampled action(s) from the policy
    """
    return dist = self.model._actor(batch_dict)

def update(self, batch_dict):
    """
    Update the policy using expert demonstrations
    """
    # Get policy's action distribution
    dist = self.model._actor(batch_dict)
    logstd = self.model._actor.logstd
    std = torch.exp(logstd)
    neglogp = self.model.neglogp(batch_dict["actions"], dist.mean, std, logstd)
    
    loss = neglogp.mean()    
    
    self.optimizer.zero_grad()
    self.fabric.backward(loss)
    self.optimizer.step()
    
    return {"loss": loss.detach().item()}