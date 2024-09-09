import torch
import torch.nn as nn

class BlockDiagonalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks=8):
        super(BlockDiagonalGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks

        # Input weights
        self.w_ir = nn.Linear(input_size, 3 * hidden_size)
        
        # Recurrent weights (block-diagonal)
        self.w_hr = nn.ModuleList([
            nn.Linear(self.block_size, 3 * self.block_size, bias=False)
            for _ in range(num_blocks)
        ])

        # Biases
        self.b_ir = nn.Parameter(torch.zeros(3 * hidden_size))
        self.b_hr = nn.Parameter(torch.zeros(3 * hidden_size))

    def forward(self, x, h):
        # x: (batch, input_size)
        # h: (batch, hidden_size)
        
        batch_size = x.size(0)

        # Compute input contribution
        w_x = self.w_ir(x) + self.b_ir
        
        # Split input contribution into reset, update, and new gates
        w_x_r, w_x_z, w_x_n = w_x.chunk(3, dim=1)

        # Initialize output hidden state
        h_new = torch.zeros_like(h)

        # Process each block
        for i in range(self.num_blocks):
            # Extract the block from the hidden state
            h_block = h[:, i*self.block_size:(i+1)*self.block_size]
            
            # Compute recurrent contribution for this block
            w_h = self.w_hr[i](h_block)
            w_h = w_h + self.b_hr[i*3*self.block_size:(i+1)*3*self.block_size]
            
            # Split recurrent contribution into reset, update, and new gates
            w_h_r, w_h_z, w_h_n = w_h.chunk(3, dim=1)
            
            # Compute gate values
            r = torch.sigmoid(w_x_r[:, i*self.block_size:(i+1)*self.block_size] + w_h_r)
            z = torch.sigmoid(w_x_z[:, i*self.block_size:(i+1)*self.block_size] + w_h_z)
            n = torch.tanh(w_x_n[:, i*self.block_size:(i+1)*self.block_size] + r * w_h_n)
            
            # Compute new hidden state for this block
            h_new[:, i*self.block_size:(i+1)*self.block_size] = (
                (1 - z) * h_block + z * n
            )

        return h_new