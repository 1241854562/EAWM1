import torch.nn as nn
import torch
class WeightConditionedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # weight: [out_features, in_features]
        W = self.weight
  
        row_norms = W.norm(p=2, dim=1, keepdim=True)  # [out_features, 1]
   
        scale = 1.0 / (row_norms + self.eps)
        W_scaled = W * scale    
        return F.linear(x, W_scaled, self.bias)

class FrequencyDomainFeatureExtractor(nn.Module):
    def __init__(self, n_fft=16, hop_length=8, win_length=None):
        super(FrequencyDomainFeatureExtractor, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft

        self.freq_enhancement = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)), 
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, channels)
        batch_size, seq_len, channels = x.shape
        freq_features = []

        for i in range(channels):
  
            channel_data = x[:, :, i]  # (batch_size, seq_len)

     
            stft = torch.stft(
                channel_data.reshape(-1, seq_len),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True
            )

          magnitude = torch.abs(stft)  # (batch_size, freq_bins, time_frames)

     
            magnitude = magnitude.unsqueeze(1)  # (batch_size, 1, freq_bins, time_frames)

     
            enhanced_freq = self.freq_enhancement(magnitude)  # (batch_size, 64)
            freq_features.append(enhanced_freq)


        freq_features = torch.stack(freq_features, dim=2)  # (batch_size, 64, channels)

        return freq_features
class MultiScaleXLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, scale=3, batch_first=True, bidirectional=False, num_mem_tokens=8, dropout=0.1):
        super(MultiScaleXLSTM, self).__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.xlstms = nn.ModuleList([
            XLSTM(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  num_heads2=num_heads2,
                  batch_first=batch_first,
                  bidirectional=bidirectional,
                  num_mem_tokens=num_mem_tokens)
            for _ in range(scale)
        ])
        self.attention_layer = AttentionLayer(d_model1)
    def forward(self, x):
        B, L, D = x.size()
        scale = self.scale
        x_scaled = {}
        for i in range(1, scale + 1):
            start_idx = L - (L // i)
            x_scaled[f'x{i}'] = x[:, start_idx:, :]
        upsampled_outputs = []
        for i in range(1, scale + 1):
            xi = x_scaled[f'x{i}']
            if xi.shape[1] < L:
                xi = xi.transpose(1, 2)
                xi = nn.functional.interpolate(xi, size=L, mode='linear', align_corners=False)
                xi = xi.transpose(1, 2)
            elif xi.shape[1] > L:
                xi = xi[:, :L, :]
            upsampled_outputs.append(xi)
        outputs = []
        for i in range(scale):
            xi = upsampled_outputs[i - 1]
            xlstm = self.xlstms[i]
            xi = xlstm(xi)
            outputs.append(xi)
        merged_output = torch.stack(outputs, dim=1)
        output = self.attention_layer(merged_output)
        return output
class XLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads2):
        super(XLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads2)
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size*2),
                               nn.GELU(),
                               nn.Linear(hidden_size*2, hidden_size))

        self.layer_norm_attn = nn.LayerNorm(hidden_size)
        self.layer_norm_out = nn.LayerNorm(hidden_size)
    def forward(self, x, hx, mem_tokens):
        h_prev, c_prev = hx
        combined = torch.cat([x, h_prev], dim=1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(self.output_gate(combined))
        h_t = o_t * torch.tanh(c_t)
        h_t = h_t.unsqueeze(0)
        if mem_tokens is not None:
            # mem_tokens = repeat(mem_tokens, 'm d -> m b d', b=h_t.size(1))
            # combined_tokens = torch.cat([mem_tokens, h_t], dim=0)
            attn_output, _ = self.attention(h_t, h_t, h_t)
            attn_output = self.attention_layer(attn_output)
        else:
            attn_output, _ = self.attention(h_t, h_t, h_t)
            attn_output = self.attention_layer(attn_output)

        attn_output = attn_output.squeeze(0)
        attn_output = self.layer_norm_attn(attn_output)

        h_t = attn_output + h_prev

        h_t = self.layer_norm_out(h_t)
        return h_t, c_t
class XLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads2, batch_first=True, bidirectional=False, num_mem_tokens=6):
        super(XLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads2 = num_heads2
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_mem_tokens = num_mem_tokens
        self.dropout = nn.Dropout(dropout)
        self.forward_layers = nn.ModuleList([
            XLSTMCell(input_size if i == 0 else hidden_size, hidden_size, num_heads2=num_heads2) for i in
            range(num_layers)
        ])
        if self.bidirectional:
            self.backward_layers = nn.ModuleList([
                XLSTMCell(input_size if i == 0 else hidden_size, hidden_size, num_heads2=num_heads2) for i in
                range(num_layers)
            ])

        if num_mem_tokens > 0:
            self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, hidden_size) * 0.01)
        else:
            self.mem_tokens = None

        self.forward_residual_projs = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])

        if self.bidirectional:
            self.backward_residual_projs = nn.ModuleList([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
            ])
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        if self.batch_first:
            x = x.permute(1, 0, 2)

        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            for layer_idx, layer in enumerate(self.forward_layers):
                residual = input_t
                h[layer_idx], c[layer_idx] = layer(input_t, (h[layer_idx], c[layer_idx]), mem_tokens=self.mem_tokens)
                residual = self.forward_residual_projs[layer_idx](residual)
                input_t = self.dropout(h[layer_idx] + residual)
            outputs.append(input_t)
        if self.bidirectional:
            h_back = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            c_back = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
            backward_outputs = []
            for t in reversed(range(seq_len)):
                input_t = x[t]
                for layer_idx, layer in enumerate(self.backward_layers):
                    residual_back = input_t
                    h_back[layer_idx], c_back[layer_idx] = layer(input_t, (h_back[layer_idx], c_back[layer_idx]), mem_tokens=self.mem_tokens)
                    residual_back = self.backward_residual_projs[layer_idx](residual_back)
                    input_t = self.dropout(h_back[layer_idx] + residual_back)
                backward_outputs.append(input_t)
            backward_outputs.reverse()
            outputs = [torch.cat([f, b], dim=1) for f, b in zip(outputs, backward_outputs)]
            outputs = [self.layer_norm(out) for out in outputs]
        outputs = torch.stack(outputs, dim=0)
        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)
        return outputs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class EncoderLayer(nn.Module):
    def __init__(self, d_model1,num_heads, d_ff,d_ff1, dropout,d_state,d_model,num_heads2):
        super(EncoderLayer, self).__init__()
        self.mamba_forward_list = nn.ModuleList([
            Mamba(d_model1, d_state=d_state, d_conv=2, expand=1) for _ in range(d_model)
        ])
        self.cross_attn1 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(d_model1, num_heads2)
        self.norm1 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model*2),
                                  nn.GELU(),
                                  nn.Linear(d_model*2, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model1, d_ff1),
                                  nn.GELU(),
                                  nn.Linear(d_ff1, d_model1))
        self.norm2 = nn.LayerNorm(d_model1)
        self.dense = nn.Linear(8,d_model1)
        self.dropout = nn.Dropout(dropout)
        self.channel_weights = nn.Parameter(torch.ones(d_model), requires_grad=True)
    def forward(self, x,x123):
        all_outputs = []
        for i in range(x.size(1)):
            channel_x = x[:, i, :].unsqueeze(1)
            forward_out = self.mamba_forward_list[i](channel_x)
            weighted_out = self.channel_weights[i] * forward_out
            all_outputs.append(weighted_out)
        all_outputs = torch.stack(all_outputs, dim=1)  #
        all_outputs = all_outputs.view(all_outputs.size(0), -1, all_outputs.size(-1))
        all_outputs = (all_outputs + x).permute(2,0,1)
        x123 = self.dense(x123)
        attn_output,_ = self.cross_attn1(all_outputs, all_outputs,all_outputs)
        attn_output = self.norm1(all_outputs + self.dropout(attn_output))
        all_outputs = attn_output + self.dropout(self.MLP1(attn_output))
        x = (self.norm1(all_outputs)).permute(2,1,0)
        x_att,_ = self.cross_attn2(x,x,x)
        x_att = self.norm2(x + self.dropout(x_att))
        x_att = x_att + self.dropout(self.MLP2(x_att))
        x = (self.norm2(x_att)).permute(1,0,2)
        return x
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, scale=4):
        super(Encoder, self).__init__()
        self.scale = scale
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.attention_layer = AttentionLayer(d_model1)
    def forward(self, x, x123):
        B, L, D = x.shape
        scale = self.scale
        x_scaled = {}
        for i in range(1, scale + 1):
            start_idx = L - (L // i)
            x_scaled[f'x{i}'] = x[:, start_idx:, :]
        upsampled_outputs = []
        for i in range(1, scale + 1):
            xi = x_scaled[f'x{i}']
            if xi.shape[1] < L:
                xi = xi.transpose(1, 2)
                xi = nn.functional.interpolate(xi, size=L, mode='linear', align_corners=False)
                xi = xi.transpose(1, 2)
            elif xi.shape[1] > L:
                xi = xi[:, :L, :]
            upsampled_outputs.append(xi)
        outputs = []
        for i in range(1, scale + 1):
            xi = upsampled_outputs[i - 1]
            for layer in self.encoder_layers:
                xi = layer(xi, x123)
            outputs.append(xi)
        merged_output = torch.stack(outputs, dim=1)
        output = self.attention_layer(merged_output)
        return output

class AttentionLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out

class BidirectionalMambaModel(nn.Module):
    def __init__(self, d_model, d_state, d_ff,d_ff1, output_dim, dropout, num_heads, d_model1, num_heads2,d_model2):
        super(BidirectionalMambaModel, self).__init__()
        self.freq_extractor = FrequencyDomainFeatureExtractor(n_fft=16, hop_length=8)
        self.layernorm = nn.LayerNorm(4)
        self.layernorm1 = nn.LayerNorm(8)
        self.dense1 = nn.Linear(16,d_model)
        self.dense3 = nn.Linear(12, 24)
        self.dense5 =WeightConditionedLinear(d_model1, d_ff1)
        self.dense10 = nn.Linear(d_model1, d_ff1)
        self.selfdense12 = nn.Linear(18,8)
        self.selfdense321 = nn.Linear(18,  4)
        self.freq_fusion = nn.Linear(64 * 12, 4)
        self.cross_attention1 = nn.MultiheadAttention(embed_dim=d_ff, num_heads=num_heads, dropout=dropout)
        self.x_lstm1 = MultiScaleXLSTM(
            input_size=d_model1,
            hidden_size=d_model1,
            num_layers=1,
            num_heads=num_heads,
            scale=2,
            batch_first=True,
            bidirectional=False,
            num_mem_tokens=num_mem_tokens,
            dropout=dropout
        )
        self.dropout = nn.Dropout(0.2)
        self.projector = nn.Linear(d_ff1, output_dim, bias=True)
        self.attention_layer = AttentionLayer(d_ff1)
        self.encoder_embedding = nn.Linear(32, d_model1)
        self.positional_encoding = PositionalEncoding(d_model1, max_seq_length)
        encoder_layer = EncoderLayer(
            d_model1=d_model1,
            num_heads=num_heads,
            d_ff=d_ff,
            d_ff1=d_ff1,
            dropout=dropout,
            d_state=d_state,
            d_model=d_model,
            num_heads2=num_heads2
        )
        self.encoder = Encoder(
            encoder_layer=encoder_layer,
            num_layers=2,
            scale=3
        )
        self.MLP1 = nn.Sequential(nn.Linear(d_ff1, d_ff1*2),
                              nn.GELU(),
                              nn.Linear(d_ff1*2, d_ff1))
        self.norm1  = nn.LayerNorm(d_ff1)

    def forward(self, x,x1,x2,x3,x11,x21,x31):
        x123 = torch.cat((x1,x2,x3),dim=2)
        x123 = self.selfdense12(x123)
        x123 = self.layernorm1(x123)
        x123 = nn.ReLU()(x123)
        x321 = torch.cat((x11, x21,x31), dim=2)
        x321 = self.selfdense321(x321)
        x321 = nn.ReLU()(x321)
        x321 = self.layernorm(x321)
        x321 = torch.cat((x321, x321, x321,x321,x321,x321,x321,x321,x321,x321,x321,x321), dim=1)
    
        # 扩展频域特征以匹配原始维度
      
        x = x.permute(0, 2, 1)
        x = torch.cat((x, freq_enhanced_expanded), dim=2)
        x = self.dense1(x)
        x = self.dense3(x.permute(0, 2, 1))
        x = torch.cat((x, x123), dim=2)
        x_embedded = self.encoder_embedding(x)
        x_embedded = x_embedded + self.dropout(self.positional_encoding(x_embedded))
        enc_output = self.encoder(x_embedded, x123)
        all_outputs = enc_output + x_embedded
        res = self.dense5(all_outputs)
        lstm_out = self.x_lstm1(all_outputs)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.dense10(lstm_out)
        lstm_out = lstm_out + res
        all_outputs = lstm_out + self.dropout(self.MLP1(lstm_out))
        attn_output2 = self.norm1(all_outputs)
        pooled_output = self.attention_layer(attn_output2)
        output = self.projector(pooled_output)
        return output
