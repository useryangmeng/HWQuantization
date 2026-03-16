import torch
from logging import getLogger

logger = getLogger(__name__)


def dequant_dim0(qweight, bits, maxq, infeatures, outfeatures):
    """
    dequant the quantized tensor to fp tensor using PyTorch
    B is of shape (M/(32//bits), N) int32
    C is of shape (M, N) float16
    """
    bits_per_feature = 32 // bits
    output = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=torch.float16)
    
    # Iterate through each block
    for i in range(0, infeatures, bits_per_feature):
        # Calculate the number of features in this block
        block_size = min(bits_per_feature, infeatures - i)
        
        # Get the qweight block
        qweight_block = qweight[i // bits_per_feature].unsqueeze(0)
        
        # For each bit position in the block
        for j in range(block_size):
            # Calculate the shift amount
            shift = j * bits
            
            # Extract the bits and mask them
            dequantized = (qweight_block >> shift) & maxq
            
            # Store in output
            output[i + j] = dequantized.squeeze(0).float().half()
    
    return output


def dequant_dim1(qweight, bits, maxq, infeatures, outfeatures):
    """
    dequant the quantized tensor to fp tensor using PyTorch
    B is of shape (M, N/(32//bits)) int32
    C is of shape (M, N) float16
    """
    bits_per_feature = 32 // bits
    output = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=torch.float16)
    
    # Iterate through each row
    for i in range(infeatures):
        # Iterate through each block in the row
        for j in range(0, outfeatures, bits_per_feature):
            # Calculate the number of features in this block
            block_size = min(bits_per_feature, outfeatures - j)
            
            # Get the qweight block
            qweight_block = qweight[i, j // bits_per_feature].unsqueeze(0)
            
            # For each bit position in the block
            for k in range(block_size):
                # Calculate the shift amount
                shift = k * bits
                
                # Extract the bits and mask them
                dequantized = (qweight_block >> shift) & maxq
                
                # Store in output
                output[i, j + k] = dequantized.float().half()
    
    return output









