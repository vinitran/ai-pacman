import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        self.w = nn.Parameter(1, dimensions)
    def get_weights(self):
        return self.w
    def run(self, x):
        return nn.DotProduct(x, self.w)
    def get_prediction(self, x):
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1
    def train(self, dataset):
        mistake = True
        while mistake:
            still_mistake = False
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    still_mistake = True
            if still_mistake:
                mistake = True
            else:
                mistake = False
class RegressionModel(object):
    def __init__(self):
        self.learning_rate = .05
        self.w1 = nn.Parameter(1, 512)
        self.b1 = nn.Parameter(1, 512)
        self.w2 = nn.Parameter(512, 1)
        self.b2 = nn.Parameter(1, 1)
        self.params = [self.w1, self.b1, self.w2, self.b2]
    def run(self, x):
        input = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1),
                                         self.b1))
        output = nn.AddBias(nn.Linear(input, self.w2),
                                  self.b2)
        return output
    def get_loss(self, x, y):
        return nn.SquareLoss(self.run(x), y)
    def train(self, dataset):
        batch_size = 200
        loss = float('inf')
        while loss >= .01:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                i=0
                while i < len(self.params):
                    self.params[i].update(grads[i], -self.learning_rate)
                    i += 1
class DigitClassificationModel(object):
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
