__shared__ union {
  typename Load::TempStorage load;
  typename LoadFloat::TempStorage loadf;
  typename BlockReduce::TempStorage reduce;
} temp_storage;

for (unsigned int i = base_idx; i < n_full; i += gridDim.x*BLOCK_SIZE)
{
  valid_items = n - i >= (BLOCK_SIZE) ? (BLOCK_SIZE) : n - i;

  __syncthreads();
  Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items, 0.0f);
  __syncthreads();
  LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);
  __syncthreads();
  LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items, 0.0f);

  # pragma unroll NUM_VALS
  for(unsigned int j = 0; j < NUM_VALS; j++)
    g_vals[j] = gnorm_scale*((float)g_vals[j]);

  # pragma unroll NUM_VALS
  for(unsigned int j = 0; j < NUM_VALS; j++)
  {
    switch(OPTIMIZER)
    {
      case ADAM:
        s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
        s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
        s1_vals[j] *= correction1;
        s2_vals[j] *= correction2;
        s1_vals[j] = s1_vals[j]/(sqrtf(s2_vals[j])+eps); // update
        s1_vals[j] *= s1_vals[j]; // update l2 norm (update*update)
        break;
      case ADAMA:
        s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
        s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j]))); // squared version
        s1_vals[j] *= correction1;
        s2_vals[j] *= correction2;
        s1_vals[j] = s1_vals[j]/(sqrtf(s2_vals[j])+eps); // update
        s1_vals[j] = sqrtf(s1_vals[j]); // update l2 norm (square root of update)
        break;
    }
  }

  __syncthreads();
  StoreT(storet).Store(&(p[i]), s1_vals, valid_items);
  __syncthreads();
  StoreT(storet).Store(&(unorm[i]), s2_vals, valid_items);
  __syncthreads();

  for(unsigned int j = 0; j < valid_items; j++)
  {
    p[i+j] -= lr*(s1_vals[j] + weight_decay*p[i+j]);
  }
}
__syncthreads();
Load(temp_storage.load).Load(&(g[i]), g_vals, valid_items, 0.0f);
__syncthreads();
LoadFloat(temp_storage.loadf).Load(&(state1[i]), s1_vals, valid_items, 0.0f);
__syncthreads();
LoadFloat(temp_storage.loadf).Load(&(state2[i]), s2_vals, valid_items, 0.0f);

# pragma unroll NUM_VALS
for(unsigned int j = 0; j < NUM_VALS; j++)
  g_vals[j] = gnorm_scale*((float)g_vals[j]);

# pragma unroll NUM_VALS
for(unsigned int j = 0; j < NUM_VALS; j++)
{
  switch(OPTIMIZER)
  {
    case ADAM:
      s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
      s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j])));
      s1_vals[j] *= correction1;
      s2_vals[j] *= correction2;
      s1_vals[j] = s1_vals[j]/(sqrtf(s2_vals[j])+eps); // update
      s1_vals[j] *= s1_vals[j]; // update l2 norm (update*update)
      break;
    case ADAMA:
      s1_vals[j] = s1_vals[j]*beta1 + ((1.0f -beta1)*((float)g_vals[j]));
      s2_vals[j] = s2_vals[j]*beta2 + ((1.0f -beta2)*(((float)g_vals[j])*((float)g_vals[j]))); // squared version
      s1_vals[j] *= correction1;
      s2_vals[j] *= correction2;
      s1_vals[j] = s1_vals[j]/(sqrtf(s2_vals[j])+eps); // update
      s1_vals[j] = sqrtf(s1_vals[j]); // update l2 norm (square root of update)
      break;
  }
}

__syncthreads();
StoreT(storet).Store(&(p[i]), s1_vals, valid_items);
__syncthreads();
StoreT(storet).Store(&(unorm[i]), s2_vals, valid_items);
__syncthreads();

for(unsigned int j = 0; j < valid_items; j++)
{
  p[i+j] -= lr*(s1_vals[j] + weight_decay*p[i+j]);
}

# pragma unroll NUM_VALS-1
for(unsigned int j = 1; j < NUM_VALS; j++)
  s1_vals[0] += s1_vals[j];

__syncthreads();
s1_vals[0] = BlockReduce(temp_storage.reduce).Sum(s1_vals[0]);

if(threadIdx.x == 0)
  atomicAdd(&unorm[0], s1_vals[0]);

__syncwarp();
}