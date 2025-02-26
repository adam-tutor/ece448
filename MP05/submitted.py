import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    neural_object = nn.Sequential(
        nn.Linear(in_features = 2, out_features = 3),
        nn.Sigmoid(),
        nn.Linear(in_features = 3, out_features = 5)
    )
    return neural_object
    
    #raise NotImplementedError("You need to write this part!")


def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    l_ll_ll_lL = nn.CrossEntropyLoss() #haha loss
    return l_ll_ll_lL
    #raise NotImplementedError("You need to write this part!")


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super(NeuralNet, self).__init__()
        ################# Your Code Starts Here #################
        self.network = nn.Sequential(
            nn.Linear(2883, 330),
            nn.ReLU(),
            nn.Linear(330, 5)
        )
        
        #raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        y = self.network(x)
        #raise NotImplementedError("You need to write this part!")
        return y
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer
    model = NeuralNet()
    weight_decay = 0.001
    loss_fn = create_loss_function()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01, weight_decay = weight_decay)
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

    for epoch in range(epochs):
        model.train()
        for x, y in train_dataloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            regularization = 0.0
            for parameter in model.parameters():
                regularization += torch.norm(parameter, 2)
            
            loss += weight_decay * regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_pred = model(x)
            test_loss = loss_fn(test_pred, y)
        if(epoch % 10 == 0):
            epoch_count.append(epoch)
            train_loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}")

    return model
