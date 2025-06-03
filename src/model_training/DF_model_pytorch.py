import torch
import torch.nn as nn
import torch.nn.functional as F

class DF_PyTorch(nn.Module):
    def __init__(self, input_channels=1, emb_size=64):
        super(DF_PyTorch, self).__init__()
        
        filter_num = [None, 32, 64, 128, 256]
        kernel_size = 8  # Kernel size is consistent (8)
        conv_stride_size = 1 # Stride for conv is consistent (1)
        pool_stride_size = 4 # Stride for pooling is consistent (4)
        pool_size = 8        # Pool size is consistent (8)

        # Block 1
        self.block1_conv1 = nn.Conv1d(input_channels, filter_num[1], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(filter_num[1], filter_num[1], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride_size, padding=pool_size//2 -1) # Adjusted padding for MaxPool1d 'same' equivalent
        self.block1_dropout = nn.Dropout(0.1)

        # Block 2
        self.block2_conv1 = nn.Conv1d(filter_num[1], filter_num[2], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(filter_num[2], filter_num[2], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride_size, padding=pool_size//2 -1)
        self.block2_dropout = nn.Dropout(0.1)

        # Block 3
        self.block3_conv1 = nn.Conv1d(filter_num[2], filter_num[3], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(filter_num[3], filter_num[3], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride_size, padding=pool_size//2 -1)
        self.block3_dropout = nn.Dropout(0.1)

        # Block 4
        self.block4_conv1 = nn.Conv1d(filter_num[3], filter_num[4], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(filter_num[4], filter_num[4], kernel_size=kernel_size, stride=conv_stride_size, padding='same')
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride_size, padding=pool_size//2-1)
        
        # Classifier
        # The input size to the linear layer depends on the output shape of the last pooling layer.
        # Keras Flatten and Dense: (None, sequence_length_after_pooling, num_filters_last_conv) -> (None, sequence_length_after_pooling * num_filters_last_conv)
        # We need to calculate the flattened_size. Assuming input sequence length is 5000.
        # After 4 pooling layers with stride 4: 5000 / (4^4) = 5000 / 256 = 19.53. Keras 'same' padding might result in ceil(5000/4) for each pool.
        # Let's calculate it more carefully based on PyTorch's MaxPool1d behavior or by running a dummy input.
        # For now, let's make it adaptable or calculate it. A common way is to do a forward pass with a dummy input.
        # For simplicity, let's assume a calculated size. If input is (N, C_in, L_in) = (N, 1, 5000)
        # Pool1: L_out = floor((L_in + 2*padding - kernel_size)/stride + 1) = floor((5000 + 2*(8//2-1) - 8)/4 + 1) = floor((5000+6-8)/4+1) = floor(4998/4+1) = 1249+1 = 1250
        # Pool2: 1250 -> floor((1250+6-8)/4+1) = floor(1248/4+1) = 312+1 = 313
        # Pool3: 313 -> floor((313+6-8)/4+1) = floor(311/4+1) = 77+1 = 78
        # Pool4: 78 -> floor((78+6-8)/4+1) = floor(76/4+1) = 19+1 = 20
        # So, flattened_size = filter_num[4] * 20 = 256 * 20 = 5120
        self.flattened_size = filter_num[4] * 20 # This needs to be precise.
        self.fc = nn.Linear(self.flattened_size, emb_size)

    def forward(self, x):
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        # Block 4
        x = self.block4_conv1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_relu2(x)
        x = self.block4_pool(x)
        
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.fc(x)
        return x

    def _calculate_flattened_size(self, input_shape_l=5000):
        # Helper to dynamically calculate flattened size if needed
        # input_shape should be (Channels, Length), e.g., (1, 5000)
        # This is a more robust way to get the flattened size
        
        # Determine the device from one of the model's parameters
        # This ensures dummy_input is on the same device as the model
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 1, input_shape_l, device=device) # Batch_size=1, Channels=1, move to model's device
        
        # Pass through convolutional and pooling layers
        x = self.block1_pool(self.block1_elu2(self.block1_conv2(self.block1_elu1(self.block1_conv1(dummy_input)))))
        x = self.block2_pool(self.block2_relu2(self.block2_conv2(self.block2_relu1(self.block2_conv1(x)))))
        x = self.block3_pool(self.block3_relu2(self.block3_conv2(self.block3_relu1(self.block3_conv1(x)))))
        x = self.block4_pool(self.block4_relu2(self.block4_conv2(self.block4_relu1(self.block4_conv1(x)))))
        
        return x.numel() # Returns the total number of elements after conv/pool operations

# Example usage (and to verify flattened_size):
if __name__ == '__main__':
    model_test = DF_PyTorch(emb_size=64)
    # Update flattened_size dynamically
    model_test.flattened_size = model_test._calculate_flattened_size()
    model_test.fc = nn.Linear(model_test.flattened_size, 64) # Re-initialize fc layer
    print(f"Calculated flattened size: {model_test.flattened_size}")

    dummy_data = torch.randn(128, 1, 5000) # (batch_size, channels, length)
    output = model_test(dummy_data)
    print(f"Output shape: {output.shape}") # Expected: (128, 64)