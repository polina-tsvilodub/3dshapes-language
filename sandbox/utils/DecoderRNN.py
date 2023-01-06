import torch.nn as nn
import torch

# speaker language module
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, visual_embed_size, batch_size=1, num_layers=1):
        """
        Initialize the language module consisting of a one-layer LSTM and 
        trainable embeddings. The image embeddings (both target and distractor!)
        are used as additional context at every step of the training 
        (prepended to each word embedding).
        
        Args:
        -----
            embed_size: int
                Dimensionality of trainable embeddings.
            hidden_size: int
                Hidden/ cell state dimensionality of the LSTM.
            vocab_size: int
                Length of vocabulary.
            visual_embed_size: int
                Dimensionality of each image embedding to be appended at each time step as additional context.
            batch_size: int
                Batch size.
            num_layers: int
                Number of LSTM layers.
        """
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size= embed_size
        self.vocabulary_size = vocab_size
        self.visual_embed_size = visual_embed_size
        # embedding layer
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size) 
        # layer projecting ResNet features of a single image to desired size
        self.project = nn.Linear(2048, self.visual_embed_size)
        # LSTM takes as input the word embedding with prepended embeddings of the two images at each time step
        # note that the batch dimension comes first
        self.lstm = nn.LSTM(self.embed_size + 2*self.visual_embed_size, self.hidden_size , self.num_layers, batch_first=True)
        # transforming last lstm hidden state to scores over vocabulary
        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
        
        self.batch_size = batch_size
        # initial hidden state of the lstm
        self.hidden = self.init_hidden(self.batch_size)

        # initialization of the layers
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, batch_size):
            
        """ 
        At the start of training, we need to initialize a hidden state;
        Defines a hidden state with all zeroes
        The axes are (num_layers, batch_size, hidden_size)
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions, prev_hidden):
        """
        Perform forward step through the LSTM.
        
        Args:
        -----
            features: torch.tensor(batch_size, 2, embed_size)
                Embeddings of images, target and distractor concatenated in this order.
            captions: torch.tensor(batch_size, caption_length)
                Lists of indices representing tokens of each caption.
            prev_hidden: (torch.tensor(num_layers, batch_size, hidden_size), torch.tensor(num_layers, batch_size, hidden_size))
                Tuple containing previous hidden and cell states of the LSTM.
        Returns:
        ------
            outputs: torch.tensor(batch_size, caption_length, embedding_dim)
                Scores over vocabulary for each token in each caption.
            hidden_state: (torch.tensor(num_layers, batch_size, hidden_size), torch.tensor(num_layers, batch_size, hidden_size))
                Tuple containing new hidden and cell state of the LSTM.
        """
        
        # features of shape (batch_size, 2, 2048)
        image_emb = self.project(features) # image_emb should have shape (batch_size, 2, 512)
        # concatenate target and distractor embeddings
        img_features = torch.cat((image_emb[:, 0, :], image_emb[:, 1, :]), dim=-1).unsqueeze(1) 
        embeddings = self.embed(captions)
        # repeat image features such that they can be prepended to each token
        img_features_reps = img_features.repeat(1, embeddings.shape[1], 1)
        # PREpend the feature embedding as additional context as first token, assume there is no END token        
        embeddings = torch.cat((img_features_reps, embeddings), dim=-1) 
        out, hidden_state = self.lstm(embeddings, prev_hidden)
        # project LSTM predictions on to vocab
        outputs = self.linear(out) # prediction shape is (batch_size, max_sequence_length, vocab_size)
        # print("outputs shape in forward ", outputs.shape)
        return outputs, hidden_state

    def log_prob_helper(self, logits, values):
        """
        Helper function for scoring the sampled token,
        because it is not implemented for MPS yet.
        Just duplicates source code from PyTorch.
        """
        values = values.long().unsqueeze(-1)
        values, log_pmf = torch.broadcast_tensors(values, logits)
        values = values[..., :1]
        return log_pmf.gather(-1, values).squeeze(-1)

    def sample(self, inputs, max_sequence_length):
        """
        Function for sampling a caption during functional (reference game) training.
        Implements greedy sampling. Sampling stops when END token is sampled or when max_sequence_length is reached.
        Also returns the log probabilities of the action (the sampled caption) for REINFORCE.
        
        Args:
        ----
            inputs: torch.tensor(1, 1, embed_size)
                pre-processed image tensor.
            max_sequence_length: int
                Max length of sequence which the nodel should generate. 
        Returns:
        ------
            output: list
                predicted sentence (list of tensor ids). 
            log_probs: torch.Tensor
                log probabilities of the generated tokens (up to and including first END token)
            raw_outputs: torch.Tensor
                Raw logits for each prediction timestep.
            entropies: torch.Tesnor
                Entropies at each generation timestep.
        """
        
        # placeholders for output
        output = []
        raw_outputs = [] # for structural loss computation
        log_probs = []
        entropies = []
        batch_size = inputs.shape[0] 
        softmax = nn.Softmax(dim=-1)
        init_hiddens = self.init_hidden(batch_size)
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        #### start sampling ####
        for i in range(max_sequence_length):
            if i == 0:
                cat_samples = torch.tensor([0]).repeat(batch_size, 1) 
                hidden_state = init_hiddens
            
            cat_samples = cat_samples.to(device)
            inputs = inputs.to(device)

            out, hidden_state = self.forward(inputs, cat_samples, hidden_state)
            
            # get and save probabilities and save raw outputs
            raw_outputs.extend(out)
            probs = softmax(out)
            if self.training:
                # try sampling from a categorical
                cat_dist = torch.distributions.categorical.Categorical(probs)
                cat_samples = cat_dist.sample()
                entropy = cat_dist.entropy()
                
                # log_p = cat_dist.log_prob(cat_samples)
                log_p = self.log_prob_helper(torch.log(1-probs), cat_samples)
            else: 
                # if in eval mode, take argmax
                max_probs, cat_samples = torch.max(probs, dim = -1)
                log_p = torch.log(max_probs)
                entropy = -log_p * max_probs
                
                top5_probs, top5_inds = torch.topk(probs, 5, dim=-1)
                
            entropies.append(entropy)
            output.append(cat_samples)
            # cat_samples = torch.cat((cat_samples, cat_samples), dim=-1)
            # print("Cat samples ", cat_samples)
            log_probs.append(log_p)
            
            
        output = torch.stack(output, dim=-1).squeeze(1)
        # stack
        log_probs = torch.stack(log_probs, dim=1).squeeze(-1)
        entropies = torch.stack(entropies, dim=1).squeeze(-1)
        
        ####
        # get effective log prob and entropy values - the ones up to (including) END (word2idx = 1)  
        # mask positions after END - both entropy and log P should be 0 at those positions
        end_mask = output.size(-1) - (torch.eq(output, 1).to(torch.int64).cumsum(dim=1) > 0).sum(dim=-1)
        # include the END token
        end_inds = end_mask.add_(1).clamp_(max=output.size(-1)) # shape: (batch_size,)
        for pos, i in enumerate(end_inds):  
            # zero out log Ps and entropies
            log_probs[pos, i:] = 0
            entropies[pos, i:] = 0
        ####
    
        raw_outputs = torch.stack(raw_outputs, dim=1).view(batch_size, -1, self.vocabulary_size)
        return output, log_probs, raw_outputs, entropies