import torch
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import jellyfish

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        """
        Parameters
        ----------
        input_size: int
            Number of features. The last dimension of input tensor
        hidden_size: int
            Number of hidden units in LSTM.
        num_layers: int
            Number of recurrent layers in LSTM.
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        
    def forward(self, prefix):
        """
        Parameters
        ----------
        prefix: tensor
            shape: (batch_size, prefix_leng, num_features)

        Returns
        -------
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)
        """
        outputs, (hidden, cell) = self.lstm(prefix)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """
        Parameters
        ----------
        input_size: int
            Number of features. The last dimension of input tensor.
        hidden_size: int
            Number of hidden units in LSTM.
        output_size: int
            Number of features for output.
        num_layers: int
            Number of recurrent layers in LSTM.
        """        
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size , 
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, suffix_vector, hidden, cell):
        """
        Parameters
        ----------
        suffix_vector: tensor
            shape: (batch_size, num_features)
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)

        Returns
        -------
        prediction: tensor
            shape: (batch_size, output_size)
        hidden: tensor
            shape: (num_layers, batch_size, hidden_size)
        cell: tensor
            shape: (num_layers, batch_size, hidden_size)
        """
        # nn.LSTM requires input to be of shape (batch_size, seq_len, num_features) 
        suffix_vector = suffix_vector.unsqueeze(1) 
        # suffix_vector shape: (batch_size, 1, num_features) 
        
        outputs, (hidden, cell) = self.lstm(suffix_vector, (hidden, cell)) 
        # outputs shape: (batch_size, 1, hidden_size)
        
        prediction = self.fc(outputs) 
        # prediction shape: (batch_size, 1, output_size)
        
        # in Seq2Seq model, prediction will be stored using predictions[:, t, :] = prediction, which requires prediction to be 2D tensor
        prediction = prediction.squeeze(1) 
        # prediction shape: (batch_size, output_size)

        return prediction, hidden, cell

class Seq2Seq_one_input(nn.Module): 
    """
    One encoder, two decoders

    """            
    def __init__(self, num_act, encoder, act_decoder, time_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        encoder: object
            An instance of Class Encoder used to encode log/trace prefix.
        act_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_one_input, self).__init__()
        self.num_act = num_act
        self.encoder = encoder
        self.act_decoder = act_decoder
        self.time_decoder = time_decoder

    def forward(self, prefix_tensor, trace_act_suffix_tensor, trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_features [log prefix: num_act + 1; trace prefix - num_act + 2] )
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        teacher_force_ratio: float
            The probability that ground truth suffix will be used for prediction generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = prefix_tensor.shape[0]

        hidden, cell = self.encoder(prefix_tensor)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)

        # -- For activity decoder --
        
        # one-hot encode the ground truth trace_act_suffix_tensor to use as input for teacher forcing
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0 # Ensure padding is represented by all 0s
        # trace_act_suffix_tensor shape: (batch_size, suffix_len, num_act)

        # initialize predictions tensor
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        # grab the activity label (one-hot encoded) of last event in prefix as the first input for activity decoder
        x_act = prefix_tensor[:, -1, :self.num_act]
        # x_act shape: (batch_size, num_act)

        # generate prediction step by step
        for t in range(act_suffix_len):
            # use previous hidden, cell from encoder as context for first state in decoder
            act_prediction, hidden, cell = self.act_decoder(x_act, hidden, cell)
            # act_prediction shape: (batch_size, num_act)

            # store prediction
            act_predictions[:, t, :] = act_prediction

            # get the best actibity label (index) the Decoder predicts
            best_guess = act_prediction.argmax(1)
            # best_guess shape: (batch_size)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            # best_guess shape: (batch_size, num_act)
            best_guess[:, 0] = 0 # Ensure padding is represented by all 0s

            # teacher forcing
            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float()
            # x_act shape: (batch_size, num_act)

        # -- For time decoder --

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)
        # trace_time_suffix_tensor shape: (batch_size, suffix_length, 1)

        # initialize predictions tensor
        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        # grab the timestamp of last event in prefix as the first input for timestamp decoder
        # for one-input model taking trace prefix, grab trace_ts_pre (time since previous event in case), same as target
        # for one-input model taking log prefix, grab log_ts_pre (time since previous event in log), different from target
        x_time = prefix_tensor[:, -1, -1]
        # x_time: batch_size
        x_time = x_time.unsqueeze(-1)
        # x_time shape: (batch_size, 1)

        # generate prediction step by step
        for t in range(time_suffix_len):
            # use previous hidden, cell from encoder as context for first state in decoder
            time_prediction, hidden, cell = self.time_decoder(x_time, hidden, cell)

            # store prediction
            time_predictions[:, t, :] = time_prediction

            # teacher forcing
            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction
            # x_time shape: (batch_size, 1)

        return act_predictions, time_predictions

class Seq2Seq_cat(nn.Module): 
    """
    Two encoders, two decoders

    """           
    def __init__(self, num_act, log_encoder, trace_encoder, act_cat_decoder, time_cat_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        log_encoder: object
            An instance of Class Encoder used to encode log prefix.
        trace_encoder: object
            An instance of Class Encoder used to encode trace prefix.
        act_cat_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_cat_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_cat, self).__init__()
        self.num_act = num_act
        self.log_encoder = log_encoder
        self.trace_encoder = trace_encoder
        self.act_decoder = act_cat_decoder
        self.time_decoder = time_cat_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1)
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2] )
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        teacher_force_ratio: float
            The probability that ground truth suffix will be used for prediction generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = log_prefix_tensor.shape[0]

        # --- The part below is different from Seq2Seq_one_input class --
        log_hidden, log_cell = self.log_encoder(log_prefix_tensor)
        trace_hidden, trace_cell = self.trace_encoder(trace_prefix_tensor)

        hidden = torch.cat((log_hidden, trace_hidden), -1)
        cell = torch.cat((log_cell, trace_cell), -1)
        # --- The part above is different from Seq2Seq_one_input class --

        # for activity decoder
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0
        
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]

        for t in range(act_suffix_len):

            act_prediction, hidden, cell = self.act_decoder(x_act, hidden, cell)
            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float()

        # for time decoder 

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1]
        x_time = x_time.unsqueeze(-1)

        for t in range(time_suffix_len):

            time_prediction, hidden, cell = self.time_decoder(x_time, hidden, cell)
            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

        return act_predictions, time_predictions

class Seq2Seq_add(nn.Module): 
    """
    Two encoders, two decoders

    """         
    def __init__(self, num_act, log_encoder, trace_encoder, act_decoder, time_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        log_encoder: object
            An instance of Class Encoder used to encode log prefix.
        trace_encoder: object
            An instance of Class Encoder used to encode trace prefix.
        act_cat_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_cat_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_add, self).__init__()
        self.num_act = num_act
        self.log_encoder = log_encoder
        self.trace_encoder = trace_encoder
        self.act_decoder = act_decoder
        self.time_decoder = time_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1)
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2] )
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        teacher_force_ratio: float
            The probability that ground truth suffix will be used for prediction generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """

        batch_size = log_prefix_tensor.shape[0]
        
        # --- The part below is different from Seq2Seq_one_input class --
        log_hidden, log_cell = self.log_encoder(log_prefix_tensor)
        trace_hidden, trace_cell = self.trace_encoder(trace_prefix_tensor)

        hidden = torch.add(log_hidden, trace_hidden)
        cell = torch.add(log_cell, trace_cell)
        # --- The part above is different from Seq2Seq_one_input class --

        # for activity decoder
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0
        
        act_suffix_len = trace_act_suffix_tensor.shape[1]
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]

        for t in range(act_suffix_len):

            act_prediction, hidden, cell = self.act_decoder(x_act, hidden, cell)
            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float()

        # for time decoder 

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1]
        x_time = x_time.unsqueeze(-1)

        for t in range(time_suffix_len):

            time_prediction, hidden, cell = self.time_decoder(x_time, hidden, cell)

            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

        return act_predictions, time_predictions
    
class Seq2Seq_mul(nn.Module): 
    """
    Two encoders, two decoders

    """     
    def __init__(self, num_act, log_encoder, trace_encoder, act_decoder, time_decoder):
        """
        Parameters
        ----------
        num_act: int
            Number of activity labels (including padding, SOC, EOC, unknown label)
        log_encoder: object
            An instance of Class Encoder used to encode log prefix.
        trace_encoder: object
            An instance of Class Encoder used to encode trace prefix.
        act_cat_dncoder: object
            An instance of Class Decoder used to generate activity label suffix prediction.
        time_cat_dncoder: object
            An instance of Class Decoder used to generate timestamp suffix prediction.
        """
        super(Seq2Seq_mul, self).__init__()
        self.num_act = num_act
        self.log_encoder = log_encoder
        self.trace_encoder = trace_encoder
        self.act_decoder = act_decoder
        self.time_decoder = time_decoder

    def forward(self, log_prefix_tensor, trace_prefix_tensor, trace_act_suffix_tensor, trace_time_suffix_tensor, teacher_force_ratio):
        """
        Parameters
        ----------
        log_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 1)
        trace_prefix_tensor: tensor
            shape: (batch_size, prefix_len, num_act + 2] )
        trace_act_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        trace_time_suffix_tensor: tensor
            shape: (batch_size, suffix_len)
        teacher_force_ratio: float
            The probability that ground truth suffix will be used for prediction generation  
        
        Returns
        -------
        act_predictions: tensor
            shape: (batch_size, suffix_len, num_act)
        time_predictions: tensor
            shape: (batch_size, suffix_len, 1)
        """
        
        batch_size = log_prefix_tensor.shape[0]
        
        # --- The part below is different from Seq2Seq_one_input class --
        log_hidden, log_cell = self.log_encoder(log_prefix_tensor)
        trace_hidden, trace_cell = self.trace_encoder(trace_prefix_tensor)

        hidden = torch.mul(log_hidden, trace_hidden)
        cell = torch.mul(log_cell, trace_cell)
        # --- The part above is different from Seq2Seq_one_input class --

        # for activity decoder
        
        trace_act_suffix_tensor = F.one_hot(trace_act_suffix_tensor, num_classes=self.num_act)
        trace_act_suffix_tensor[:, :, 0] = 0
        
        act_suffix_len = trace_act_suffix_tensor.shape[1]  
        act_predictions = torch.zeros(batch_size, act_suffix_len, self.num_act)

        x_act = trace_prefix_tensor[:, -1, :self.num_act]

        for t in range(act_suffix_len):

            act_prediction, hidden, cell = self.act_decoder(x_act, hidden, cell)
            act_predictions[:, t, :] = act_prediction

            best_guess = act_prediction.argmax(1)
            best_guess = F.one_hot(best_guess, num_classes=self.num_act)
            best_guess[:, 0] = 0

            x_act = trace_act_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else best_guess
            x_act = x_act.float()

        # for time decoder 

        trace_time_suffix_tensor = trace_time_suffix_tensor.unsqueeze(-1)

        time_suffix_len = trace_time_suffix_tensor.shape[1]
        time_predictions = torch.zeros(batch_size, time_suffix_len, 1)

        x_time = trace_prefix_tensor[:, -1, -1]
        x_time = x_time.unsqueeze(-1)

        for t in range(time_suffix_len):

            time_prediction, hidden, cell = self.time_decoder(x_time, hidden, cell)

            time_predictions[:, t, :] = time_prediction

            x_time = trace_time_suffix_tensor[:, t, :] if random.random() < teacher_force_ratio else time_prediction

        return act_predictions, time_predictions

def normalized_DL_distance(predictions,
                           target):
    """
    Normalized Damerau-Levenshtein Distance ranging between 0 and 1.

    Parameters:
    -----------
    predictions: tensor (one-hot encoded)
        shape: (batch_size, suffix_len, num_act)
    target: tensor
        shape: (batch_size, suffix_len)

    Returns
    -------
    loss: float
        Normalized Damerau-Levenshtein Distance averaged among all sequence pairs
    
    """
    # apply argmax to predictions
    predictions = predictions.argmax(2)
    # predictions shape: (batch_size, suffix_len)

    # get batch_size
    batch_size = predictions.shape[0]
    total_loss = 0.0

    for i in range(batch_size):
        # convert each sequence to a list
        pred_seq = predictions[i].tolist()
        target_seq = target[i].tolist()

        # initialize strings for each sequence
        pred_str = ""
        target_str = ""

        for p in pred_seq:
            pred_str += str(p)
            # If there appears a EOC token (index 3), stop producing the string
            if p == 3:
                break
        
        for t in target_seq:
            target_str += str(t)
            # If there appears a EOC token (index 3), stop producing the string
            if t == 3:
                break
        
        # calculate unnormalized Damerau-Levenshtein distance
        distance = jellyfish.damerau_levenshtein_distance(pred_str, target_str)

         # normalize by the max length of two strings
        max_len = max(len(pred_str), len(target_str))
        normalized_distance = distance / max_len if max_len > 0 else 0

        total_loss += normalized_distance
    
    # average the losses by batch size
    loss = total_loss / batch_size

    return loss

