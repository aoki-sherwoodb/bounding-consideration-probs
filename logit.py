import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

class Logit(nn.Module):

    def __init__(self, num_items):
        """
        Initialize a logit model for inference
        @param num_items: total number of items
        """
        super().__init__()
        self.num_items = num_items
        self.utilities = nn.Parameter(torch.zeros(self.num_items))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        @param choice_sets: the choice sets, represented as binary item presence vectors
        @return: log(choice probabilities) over every choice set
        """

        utilities = self.utilities * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)
    
    def fit(self, data, choices, epochs=500, learning_rate=0.05, l2_lambda=1e-6):
        """
        Fit a logit model to data using the given optimizer.
        @param data: input into the model (either single tensor or tuple of tensors)
        @param choices: indices of chosen items
        @param epochs: max number of optimization epochs
        @param learning_rate: step size hyperparameter for Rprop
        @param l2_lambda: regularization hyperparameter
        """
        torch.set_num_threads(1)

        optimizer = torch.optim.Rprop(self.parameters(), lr=learning_rate)

        if type(data) == torch.Tensor:
            data = (data,)

        for epoch in range(epochs):
            self.train()
            losses = []

            optimizer.zero_grad()

            loss = nnf.nll_loss(self(*data), choices)

            # Add L2 regularization
            l2_reg = torch.tensor(0.)
            for param in self.parameters():
                l2_reg += torch.pow(param, 2).sum()
            loss += l2_lambda * l2_reg

            loss.backward()

            # Check if loss gradient is small (i.e., we're at an optimum); if so, break
            with torch.no_grad():
                gradient = torch.stack([(item.grad ** 2).sum() for item in self.parameters()]).sum()

                if gradient.item() < 10 ** -8:
                    break

            optimizer.step()

        # Compute final loss with no regularization
        loss = nnf.nll_loss(self(*data), choices)
        loss.backward()
        with torch.no_grad():
            gradient = torch.stack([(item.grad ** 2).sum() for item in self.parameters()]).sum()

        print('Done. Final gradient:', gradient.item(), 'Final NLL:', loss.item() * len(choices))


        return loss.item() * len(choices)