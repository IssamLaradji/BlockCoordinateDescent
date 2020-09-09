import numpy as np


def get_partition(A, b, loss, block_size, p_rule):
  n_params = int(loss.n_params)
  L = loss.lipschitz

  # ASSERTIONS
  if p_rule != "Ada" and p_rule != "VB":
    # Assert fixed block methods have block size that divides the coordinates
    # equally
    
    assert (n_params % block_size) == 0 

  if p_rule == "Ada" or p_rule == "VB":
      return None

  elif p_rule == "PCyclic" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)

  elif p_rule == "PCyclicSoftmax" or p_rule =="Order":
      """Fixed partition cyclic rule"""
      block_indices = np.arange(n_params)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)

  elif p_rule =="Hsort":
      # Group by lipschitz values
      Hb = loss.Hb_func(None, A, b)
      scores = np.sum(np.abs(Hb), 1)
      block_indices = np.argsort(scores)
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)

  elif p_rule =="Havg":
      Hb = loss.Hb_func(None, A, b)
      scores = np.sum(np.abs(Hb), 1)
      indices = np.argsort(scores)
      n_blocks = int(n_params / block_size)
      
      block_indices = np.ones(n_params, int) * -1

      # Alternate between adding large and small lipschitz values
      for i in range(int(n_params / 2)):
        block_indices[2*i] = indices[i]
        block_indices[2*i + 1] = indices[n_params - 1 - i]

      np.testing.assert_equal(np.unique(block_indices), np.arange(n_params)) 
      assert -1 not in block_indices

      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)

  elif p_rule == "LipGroup" or p_rule =="Sort":
      # Group by lipschitz values
      block_indices = np.argsort(L)
      
      n_blocks = int(n_params / block_size)
      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)

  elif p_rule == "AvgLip" or p_rule =="Avg":
      # Group by average lipschitz
      indices = np.argsort(L)
      n_blocks = int(n_params / block_size)
      
      block_indices = np.ones(n_params, int) * -1

      # Alternate between adding large and small lipschitz values
      for i in range(int(n_params / 2)):
        block_indices[2*i] = indices[i]
        block_indices[2*i + 1] = indices[n_params - 1 - i]

      np.testing.assert_equal(np.unique(block_indices), np.arange(n_params)) 
      assert -1 not in block_indices

      fixed_blocks = get_fixed_blocks(block_indices, n_blocks, block_size)
  else:
      raise ValueError("Partition rule %s does not exist" % p_rule)

  # Assert all blocks have been chosen
  np.testing.assert_equal(np.unique(fixed_blocks), np.arange(n_params)) 

  return fixed_blocks

def get_fixed_blocks(block_indices, n_blocks, block_size):
    fixed_blocks = np.zeros((n_blocks, block_size), int)
    for i in range(n_blocks):
        fixed_blocks[i] = block_indices[i*block_size: i*block_size + block_size]

    return fixed_blocks