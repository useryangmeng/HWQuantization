import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.recon_loss import get_recon_loss


def multi_layer_distillation_loss(
    fp_blocks,
    quant_blocks,
    x,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    loss_type="mse",
    hidden_weight=1.0,
    attn_weight=0.5,
    return_hidden_states=False,
    return_attentions=False
):
    """
    Compute multi-layer distillation loss between full-precision (teacher) and quantized (student) blocks.
    
    Args:
        fp_blocks: List of full-precision Transformer blocks
        quant_blocks: List of quantized Transformer blocks
        x: Input tensor (hidden states)
        attention_mask: Attention mask tensor
        position_ids: Position IDs tensor
        past_key_values: Past key values for prefix cache
        loss_type: Type of loss to use for hidden states ("mse", "clamp_mse", etc.)
        hidden_weight: Weight for hidden states loss
        attn_weight: Weight for attention KL divergence loss
        return_hidden_states: Whether to return hidden states
        return_attentions: Whether to return attention maps
    
    Returns:
        loss: Total distillation loss
        hidden_states (optional): Dict with fp and quant hidden states
        attentions (optional): Dict with fp and quant attention maps
    """
    assert len(fp_blocks) == len(quant_blocks), "fp_blocks and quant_blocks must have the same length"
    num_layers = len(fp_blocks)
    
    loss_func = get_recon_loss(loss_type)
    total_loss = 0.0
    
    fp_hidden = x
    quant_hidden = x
    
    fp_hidden_states = []
    quant_hidden_states = []
    fp_attentions = []
    quant_attentions = []
    
    for i, (fp_block, quant_block) in enumerate(zip(fp_blocks, quant_blocks)):
        # Teacher forward pass with torch.no_grad()
        with torch.no_grad():
            fp_output = fp_block(
                fp_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values else None,
                output_attentions=True
            )
            fp_hidden = fp_output[0]
            fp_attn = fp_output[1] if len(fp_output) > 1 else None
            fp_hidden_states.append(fp_hidden)
            fp_attentions.append(fp_attn)
        
        # Student forward pass
        quant_output = quant_block(
            quant_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[i] if past_key_values else None,
            output_attentions=True
        )
        quant_hidden = quant_output[0]
        quant_attn = quant_output[1] if len(quant_output) > 1 else None
        quant_hidden_states.append(quant_hidden)
        quant_attentions.append(quant_attn)
        
        # Compute Hidden States MSE Loss
        loss_hidden = loss_func(quant_hidden, fp_hidden)
        layer_loss = hidden_weight * loss_hidden
        
        # Compute Attention KL Divergence Loss (if available) with numerical stability
        if fp_attn is not None and quant_attn is not None and attn_weight > 0:
            # Numerical stability: clamp to avoid 0/inf
            quant_attn_clamped = torch.clamp(quant_attn, min=-100, max=100)
            fp_attn_clamped = torch.clamp(fp_attn, min=-100, max=100)
            
            log_quant_attn = F.log_softmax(quant_attn_clamped, dim=-1)
            soft_fp_attn = F.softmax(fp_attn_clamped, dim=-1)
            
            # Additional clamping for log_quant_attn to avoid -inf
            log_quant_attn = torch.clamp(log_quant_attn, min=-100)
            
            loss_attn = F.kl_div(
                log_quant_attn,
                soft_fp_attn,
                reduction='batchmean'
            )
            
            # Check if loss_attn is finite
            if torch.isfinite(loss_attn):
                layer_loss += attn_weight * loss_attn
        
        total_loss += layer_loss
    
    # Normalize loss by number of layers to prevent gradient explosion
    if num_layers > 0:
        total_loss = total_loss / num_layers
    
    return_dict = {"loss": total_loss}
    if return_hidden_states:
        return_dict["hidden_states"] = {
            "fp_hidden_states": fp_hidden_states,
            "quant_hidden_states": quant_hidden_states
        }
    if return_attentions:
        return_dict["attentions"] = {
            "fp_attentions": fp_attentions,
            "quant_attentions": quant_attentions
        }
    
    if len(return_dict) == 1:
        return total_loss
    return return_dict


def get_multi_layer_blocks(layers, start_idx, num_layers=2):
    """
    Get a continuous range of Transformer blocks.
    
    Args:
        layers: List of all model layers
        start_idx: Starting index of the block range
        num_layers: Number of consecutive layers to include (2-4)
    
    Returns:
        blocks: List of consecutive layers
    """
    num_layers = max(2, min(4, num_layers))
    end_idx = min(start_idx + num_layers, len(layers))
    return layers[start_idx:end_idx]
