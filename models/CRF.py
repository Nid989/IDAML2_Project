import torch
import torch.nn as nn

class LC_CRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF)

    Args:
        nb_labels (int): number of labels in tagset, including special symbols (below).
        bos_tag_id (int): integer representing the beginning of sentence symbol in
            your tagset.
        eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
        pad_tag_id (int, optional): integer representing the pad symbol in your tagset.
            If None, the model will treat the PAD as a normal tag. Otherwise, the model
            will apply constraints for PAD transitions.
    """
    def __init__(self, nb_labels, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(LC_CRF, self).__init__()

        self.nb_labels = nb_labels

        self.DEVICE = device

        self.start_transitions = nn.Parameter(torch.empty(self.nb_labels))
        self.end_transitions = nn.Parameter(torch.empty(self.nb_labels))
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask=None, reduction: str='sum'):
        """
        Compute the negative log-likelihood.
        """
        return -self.log_likelihood(emissions, tags, mask=mask, reduction=reduction)

    def log_likelihood(self, emissions, tags, mask=None, reduction: str='sum'):
        """
        Compute the probability of sequence of tags given a sequence of emission 
        scores.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label. 
                Shape (batch_size, seq_len, nb_labels)
            tags (torch.LongTensor): Sequence of labels.
                Shape (batch_size, seq_len)
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len)
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            torch.Tensor: the (summed) log-likelihoods of each sequence in the batch.
                Shape of (1,)
        """   
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        llh = scores - partition

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.sum()

    def _compute_scores(self, emissions, tags, mask):
        """
        Compute the scores for a given batch of emissions with their tags.

        Args:
            emissions (torch.Tensor): 
                Shape (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): 
                Shape (batch_size, seq_len)
            mask (Torch.FloatTensor): 
                Shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Scores for each batch
                Shape (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).to(self.DEVICE)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add the transition from BOS to the first tags for each batch
        t_scores = self.start_transitions[first_tags]

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nblabels]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(1, seq_length):

            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster due to gpu,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add the transition from the end tag to the EOS tag for each batch
        scores += self.end_transitions[last_tags]

        return scores
        
    def _compute_log_partition(self, emissions, mask):
        """
        Compute the partition function in log-space using the forward-algorithm.

        Args:
            emissions (torch.Tensor): 
                Shape (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): 
                Shape (batch_size, seq_len)

        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            alpha_t = []

            for tag in range(nb_labels):

                # get the emission for the current tag
                e_scores = emissions[:, i, tag]

                # broadcast emission to all labels
                # since it will be the same for all previous tags
                # (bs, nb_labels)
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag
                t_scores = self.transitions[:, tag]

                # broadcast the transition scores to all batches
                # (bs, nb_labels)
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                # since alphas are in log space (see logsumexp below),
                # we add them instead of multiplying
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # add the scores for the final transition
        last_transition = self.end_transitions
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return torch.logsumexp(end_scores, dim=1)

    def decode(self, emissions, mask=None):
        """
        Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels) 
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
            If None, all positions are considered valid.
                Shape (batch_size, seq_len)
        
        Returns:
            torch.Tensor: the Vertibi score for each sequence in the batch
                Shape (batch_size,)
            list of lists: the best vertibi sequence of labels for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float).to(self.DEVICE)

        scores, sequences = self._vertibi_decode(emissions, mask)
        
        if all([True if len(sequence) == emissions.shape[1] else False for sequence in sequences]):
            return torch.tensor(scores), torch.tensor(sequences)
        
        return scores, sequences

    def _vertibi_decode(self, emissions, mask):
        """
        Compute the vertibi algorithm to find the most probable sequence of labels
        given a sequence of emissions.

        Args:  
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (torch.Tensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: the vertibi score for each sequence in the batch
                Shape (batch_size,)
            list of lists: the best vertibi sequence of labels for each batch.
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.start_transitions + emissions[:, 0]

        backpointers = []
        
        for i in range(1, seq_length):
            alpha_t = []
            backpointers_t = []

            for tag in range(nb_labels):

                # get the emissions for the current tag and broadcast to all labels
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag and boradcast to all batches
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # so far is exactly like the forward algorithm
                # but now, instead of calculating the logsumexp
                # we will find the highest score and the tag associated with it
                max_score, max_score_tag = torch.max(scores, dim=-1)

                # add the max score for the current tag
                alpha_t.append(max_score)

                # add the max_score for our list of backpointers
                backpointers_t.append(max_score_tag)

            # create a torch matrix from alpha_t
            # (batch_size, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            # append the new backpointers
            backpointers.append(backpointers_t)

        # add the scores for the final transition
        last_transition = self.end_transitions
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """
        Auxiliary function to find the best path sequence for a specific sample.

        Args:
            sample_id (int): sample index in the range [0, batch_size)

            best_tag (int): tag which maximizes the final score

            backpointers (list of lists of tensors): list of pointers with
                Shape (seq_len_i-1, nb_labels, batch_size);
                where seq_len_i represents the length of the ith sample in the batch

        Returns:
            list of ints: a list of tag indexes representing the bast path
        """
        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path