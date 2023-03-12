import torch
from torch import nn
from torch.nn.init import xavier_uniform_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AddSpatialInfo(nn.Module):

  def _create_coord(self, img_feat):
    batch_size, _, h, w = img_feat.size()
    coord_map = img_feat.new_zeros(2, h, w)
    for i in range(h):
      for j in range(w):
        coord_map[0][i][j] = (j * 2.0 / w) - 1
        coord_map[1][i][j] = (i * 2.0 / h) - 1
    sequence = [coord_map] * batch_size
    coord_map_in_batch = torch.stack(sequence)
    return coord_map_in_batch

  def forward(self, img_feat):
    coord_map = self._create_coord(img_feat)
    img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
    return img_feat_aug


class Dual_SAM(nn.Module):

  def __init__(self, input_dim, dim):
    super().__init__()
    self.input_dim = input_dim
    self.dim = dim

    self.embed = nn.Sequential(
      nn.Conv2d(self.input_dim, self.dim, kernel_size=1, padding=0),
      nn.GroupNorm(32, self.dim),
      nn.Dropout(0.5),
      nn.ReLU()
    )

    self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)

  def forward(self, input_1, input_2):
    batch_size, t, K1, H, W = input_1.size()
    input_diff = input_2 - input_1

    input_before = torch.cat([input_1, input_diff], 2)
    input_after = torch.cat([input_2, input_diff], 2)

    input_before = input_before.view(batch_size * t, 2 * K1, H, W)
    input_after = input_after.view(batch_size * t, 2 * K1, H, W)

    embed_before = self.embed(input_before)
    embed_after = self.embed(input_after)

    att_weight_before = torch.sigmoid(self.att(embed_before))
    att_weight_after = torch.sigmoid(self.att(embed_after))

    att_1_expand = att_weight_before.expand(batch_size * t, K1, H, W)
    attended_1 = (input_1.view(batch_size * t, K1, H, W) * att_1_expand).sum(2).sum(2)  # (batch, dim)

    att_2_expand = att_weight_after.expand(batch_size * t, K1, H, W)
    attended_2 = (input_2.view(batch_size * t, K1, H, W) * att_2_expand).sum(2).sum(2)  # (batch, dim)

    attended_1 = attended_1.view(batch_size, t, -1)
    attended_2 = attended_2.view(batch_size, t, -1)

    input_attended = attended_2 - attended_1

    output = torch.cat((attended_1, input_attended, attended_2), dim=-1)

    return output


class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """
    def __init__(self, dropout, d_model=512, n_head = 4):
      """
      :param dropout: dropout rate
      :param d_model: dimension of hidden state
      :param n_head: number of heads in multi head attention
      """
      super(CrossTransformer, self).__init__()
      self.attention = nn.MultiheadAttention(d_model, n_head, dropout = dropout)

      self.norm1 = nn.LayerNorm(d_model)
      self.norm2 = nn.LayerNorm(d_model)

      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)

      self.activation = nn.ReLU()

      self.linear1 = nn.Linear(d_model, d_model * 4)
      self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2):
      attn_output, attn_weight = self.attention(self.norm1(input1), self.norm1(input2), self.norm1(input2))
      output = input1 + self.dropout1(attn_output)
      ff_output = self.linear2(self.dropout2(self.activation(self.linear1(self.norm2(output)))))
      output = output + ff_output
      return output


class Cross_SAM(nn.Module):
    """
    Cross_SAM
    """
    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head = 4, n_layers = 2):
      """
      :param feature_dim: dimension of input features
      :param dropout: dropout rate
      :param d_model: dimension of hidden state
      :param n_head: number of heads in multi head attention
      :param n_layer: number of layers of transformer layer
      """
      super(Cross_SAM, self).__init__()
      self.d_model = d_model
      self.n_layers = n_layers

      self.w_embedding = nn.Embedding(w, int(d_model/2))
      self.h_embedding = nn.Embedding(h, int(d_model/2))

      self.projection = nn.Conv2d(feature_dim, d_model, kernel_size = 1)
      self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

      self._reset_parameters()

    def _reset_parameters(self):
      """Initiate parameters in the transformer model."""
      for p in self.parameters():
        if p.dim() > 1:
          xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):

      batch_size, t, K1, H, W = img_feat1.size()

      img_feat1 = img_feat1.view(batch_size * t, K1, H, W)
      img_feat2 = img_feat2.view(batch_size * t, K1, H, W)

      # img_feat1 (batch_size, feature_dim, h, w)
      batch = img_feat1.size(0)
      feature_dim = img_feat1.size(1)
      w, h = img_feat1.size(2), img_feat1.size(3)

      img_feat1 = self.projection(img_feat1)# + position_embedding # (batch_size, d_model, h, w)
      img_feat2 = self.projection(img_feat2)# + position_embedding # (batch_size, d_model, h, w)

      pos_w = torch.arange(w,device=device).to(device)
      pos_h = torch.arange(h,device=device).to(device)
      embed_w = self.w_embedding(pos_w)
      embed_h = self.h_embedding(pos_h)
      position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                     embed_h.unsqueeze(1).repeat(1, w, 1)],
                                     dim = -1)
      #(h, w, d_model)
      position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1) #(batch, d_model, h, w)

      img_feat1 = img_feat1 + position_embedding # (batch_size, d_model, h, w)
      img_feat2 = img_feat2 + position_embedding # (batch_size, d_model, h, w)

      output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)
      output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1) # (h*w, batch_size, d_model)

      for l in self.transformer:
        output1, output2 = l(output1, output2), l(output2, output1)

      output1 = output1.permute(1, 2, 0).view(batch,512,8,8) #(batch_size, d_model, h*w)
      output2 = output2.permute(1, 2, 0).view(batch,512,8,8) #(batch_size, d_model, h*w)

      # output1 = F.adaptive_max_pool2d(output1, (1, 1))
      # output2 = F.adaptive_max_pool2d(output2, (1, 1))

      output1 = output1.sum(2).sum(2)
      output2 = output2.sum(2).sum(2)

      output1 = output1.view(batch_size, t, -1)
      output2 = output2.view(batch_size, t, -1)

      output = torch.cat([output1, output2-output1, output2], dim=2)

      return output


class Proposed_SAM_NoDual(nn.Module):
  """
  Proposed_SAM_NoDual
  """

  def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=2):
    """
    :param feature_dim: dimension of input features
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    :param n_layer: number of layers of transformer layer
    """
    super(Proposed_SAM_NoDual, self).__init__()
    self.d_model = d_model
    self.n_layers = n_layers

    self.w_embedding = nn.Embedding(w, int(d_model / 2))
    self.h_embedding = nn.Embedding(h, int(d_model / 2))

    self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
    self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)

  def forward(self, img_feat1, img_feat2):

    batch_size, t, K1, H, W = img_feat1.size()

    img_feat1 = img_feat1.view(batch_size * t, K1, H, W)
    img_feat2 = img_feat2.view(batch_size * t, K1, H, W)

    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    w, h = img_feat1.size(2), img_feat1.size(3)

    img_feat1 = self.projection(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = self.projection(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)

    pos_w = torch.arange(w, device=device).to(device)
    pos_h = torch.arange(h, device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)],
                                   dim=-1)
    # (h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                 1)  # (batch, d_model, h, w)

    img_feat1 = img_feat1 + position_embedding  # (batch_size, d_model, h, w)
    img_feat2 = img_feat2 + position_embedding  # (batch_size, d_model, h, w)

    output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
    output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)

    for l in self.transformer:
      diff = output2 - output1
      output1 = l(output1, diff)
      output2 = l(output2, diff)

    output1 = output1.permute(1, 2, 0).view(batch, self.d_model, H, W)
    output2 = output2.permute(1, 2, 0).view(batch, self.d_model, H, W)

    output1 = output1.sum(2).sum(2)  # (batch, dim)
    output2 = output2.sum(2).sum(2)  # (batch, dim)

    attended_1 = output1.view(batch_size, t, -1)
    attended_2 = output2.view(batch_size, t, -1)

    input_attended = attended_2 - attended_1

    output = torch.cat((attended_1, input_attended, attended_2), dim=-1)

    return output


class Proposed_SAM(nn.Module):
  """
  Proposed_SAM
  """

  def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=2):
    """
    :param feature_dim: dimension of input features
    :param dropout: dropout rate
    :param d_model: dimension of hidden state
    :param n_head: number of heads in multi head attention
    :param n_layer: number of layers of transformer layer
    """
    super(Proposed_SAM, self).__init__()
    self.d_model = d_model
    self.n_layers = n_layers

    self.w_embedding = nn.Embedding(w, int(d_model / 2))
    self.h_embedding = nn.Embedding(h, int(d_model / 2))

    self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
    self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

    self.embed = nn.Sequential(
      nn.Conv2d(self.d_model * 3, 64, kernel_size=1, padding=0),
      nn.GroupNorm(32, 64),
      nn.Dropout(0.5),
      nn.ReLU()
    )

    self.att = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""
    for p in self.parameters():
      if p.dim() > 1:
        xavier_uniform_(p)

  def forward(self, img_feat1, img_feat2):

    batch_size, t, K1, H, W = img_feat1.size()

    img_feat1 = img_feat1.view(batch_size * t, K1, H, W)
    img_feat2 = img_feat2.view(batch_size * t, K1, H, W)

    # img_feat1 (batch_size, feature_dim, h, w)
    batch = img_feat1.size(0)
    feature_dim = img_feat1.size(1)
    w, h = img_feat1.size(2), img_feat1.size(3)

    img_feat1 = self.projection(img_feat1)  # + position_embedding # (batch_size, d_model, h, w)
    img_feat2 = self.projection(img_feat2)  # + position_embedding # (batch_size, d_model, h, w)

    pos_w = torch.arange(w, device=device).to(device)
    pos_h = torch.arange(h, device=device).to(device)
    embed_w = self.w_embedding(pos_w)
    embed_h = self.h_embedding(pos_h)
    position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                    embed_h.unsqueeze(1).repeat(1, w, 1)],
                                   dim=-1)
    # (h, w, d_model)
    position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                 1)  # (batch, d_model, h, w)

    img_feat1 = img_feat1 + position_embedding  # (batch_size, d_model, h, w)
    img_feat2 = img_feat2 + position_embedding  # (batch_size, d_model, h, w)

    output1 = img_feat1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
    output2 = img_feat2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)

    output1_bef = output1
    output2_bef = output2
    input_diff = output2 - output1
    for l in self.transformer:
      diff = output2 - output1
      output1 = l(output1, diff)
      output2 = l(output2, diff)

    output1_bef = output1_bef.permute(1, 2, 0).view(batch, self.d_model, H, W)
    output2_bef = output2_bef.permute(1, 2, 0).view(batch, self.d_model, H, W)
    output1 = output1.permute(1, 2, 0).view(batch, self.d_model, H, W)
    output2 = output2.permute(1, 2, 0).view(batch, self.d_model, H, W)
    input_diff = input_diff.permute(1, 2, 0).view(batch, self.d_model, H, W)

    input_before = torch.cat([output1_bef, input_diff, output1], 1)
    input_after = torch.cat([output2_bef, input_diff, output2], 1)

    embed_before = self.embed(input_before)
    embed_after = self.embed(input_after)
    att_weight_before = torch.sigmoid(self.att(embed_before))
    att_weight_after = torch.sigmoid(self.att(embed_after))

    att_1_expand = att_weight_before.expand_as(output1_bef)
    attended_1 = (output1_bef * att_1_expand).sum(2).sum(2)  # (batch, dim)
    att_2_expand = att_weight_after.expand_as(output2_bef)
    attended_2 = (output2_bef * att_2_expand).sum(2).sum(2)  # (batch, dim)

    attended_1 = attended_1.view(batch_size, t, -1)
    attended_2 = attended_2.view(batch_size, t, -1)

    input_attended = attended_2 - attended_1

    output = torch.cat((attended_1, input_attended, attended_2), dim=-1)

    return output

