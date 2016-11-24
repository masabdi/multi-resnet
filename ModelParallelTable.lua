--[[
   This file implements data parallelism for Torch modules.
   The same model is replicated on multiple GPUs. The input is split, typically
   into smaller mini-batches. Each replicated model handles only its portion of the input.
   The weight updates for each replica are summed together on the first replica
   in accGradParameters.
   By default, this module uses only one thread and relies on asynchronous kernel launches.
   To use multiple threads, call ModelParallelTable:threads(initFunc).
   For best performance, install NCCL:
    https://github.com/NVIDIA/nccl
    https://github.com/ngimel/nccl.torch
]]--
local ModelParallelTable, parent = torch.class('nn.ModelParallelTable', 'nn.Container')

local Impls = {}
local BasicImpl = torch.class('nn.ModelParallelTable.Basic', Impls)
local ThreadsImpl = torch.class('nn.ModelParallelTable.Threads', Impls)
local unpack = unpack and unpack or table.unpack -- lua52 compatibility

-- NCCL does not work when CUDA_LAUNCH_BLOCKING is set
local cudaLaunchBlocking = os.getenv('CUDA_LAUNCH_BLOCKING') == '1'

-- extracts the value at idx from each entry in tbl
local function pluck(tbl, idx)
   local r = {}
   for n, val in ipairs(tbl) do
      r[n] = val[idx]
   end
   return r
end

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
local function waitForDevice(dst, src)
   local stream = cutorch.getStream()
   if stream ~= 0 then
      cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
   end
end

function ModelParallelTable:__init(nGpus, nResFunc ,flattenParams, usenccl)
   parent.__init(self)
   if nGpus~= 2 then
      error "Only two GPU supported!"
   end


   self.nGpus = nGpus
   self.nResFunc = nResFunc
   self.modules = {}
   self.gpuAssignments = {}  -- Which gpuid each module sits on
   self.inputGpu = {}  -- inputs for each gpu
   self.gradOutputGpu = {} -- gradOutputs for each gpu
   self.outputGpu = {} -- outputs for each gpu
   self.gradInputGpu = {} -- gradInput for each gpu
   self.flattenedParams = nil -- flattened parameters for each gpu
   self.flattenParams = flattenParams or false
   self.usenccl = false
   self.needsSync = false
   self.impl = Impls.Basic(self)
   if usenccl then
      assert(self.flattenParams, 'cannot use nccl without flattenParams')
      self.usenccl = pcall(require, 'nccl')
      if not self.usenccl then
         print("warning: could not load nccl, falling back to default communication")
      end
   end
end

function ModelParallelTable:add(module, gpus)
   if type(gpus) == 'number' then
      if #self.modules == 0 then
         table.insert(self.modules, module)
      end
      table.insert(self.gpuAssignments, gpus)
      return self
   end

   assert(torch.type(gpus) == 'table' and #gpus >= 1, 'table of GPU IDs required')
   assert(#self.modules == 0, 'add should only be called once with a table of GPU assignments')
   self.modules[1] = module
   self.gpuAssignments = gpus
   return self
end

function ModelParallelTable:threads(initFunc)
   require 'threads'
   self.impl:close()
   self.impl = Impls.Threads(self, initFunc)
   return self
end

function ModelParallelTable:__tostring()
   return 'ModelParallelTable: ' .. #self.gpuAssignments .. ' x ' .. tostring(self.modules[1])
end

function ModelParallelTable:get(index)
   return self.modules[index]
end

-- this flattens parameters, so that syncParameters and accGradParameters can be much more efficient
function ModelParallelTable:flattenParameters()
   self.flattenedParams = self.impl:exec(function(module)
      local p, dp = module:parameters()
      local flattened = true
      for i=2,#p do
         if p[i]:storage() ~= p[1]:storage()
            or dp[i]:storage() ~= dp[1]:storage() then
            flattened = false
            break
         end
      end
      if flattened then
         local pp = torch.CudaTensor(p[1]:storage(), p[1]:storageOffset(),
                    p[#p]:storageOffset()+p[#p]:numel()-p[1]:storageOffset())
         local dpp = torch.CudaTensor(dp[1]:storage(), dp[1]:storageOffset(),
                     dp[#dp]:storageOffset()+dp[#dp]:numel()
                      - dp[1]:storageOffset())
         return {pp, dpp}
      else
         return { module:getParameters() }
      end
   end)
   self.flattenParams = true
end

function ModelParallelTable:getParameters()
   self:flattenParameters()
   return table.unpack(self.flattenedParams[1])
end

local function hasFlattenedParameters(self)
   if not self.flattenedParams then
      return false
   end
   for _, param in ipairs(self.modules[1]:parameters()) do
      if param:storage() ~= self.flattenedParams[1][1]:storage() then
         return false
      end
   end
   return true
end

function ModelParallelTable:training()
   self.impl:exec(function(module)
      module:training()
   end)
   parent.training(self)
end

function ModelParallelTable:evaluate()
   self.impl:exec(function(module)
      module:evaluate()
   end)
   parent.evaluate(self)
end

function ModelParallelTable:clearState()
   self.impl:exec(function(module)
      module:clearState()
   end)
   return parent.clearState(self)
end

local function _hasData(input)
   if torch.isTensor(input) then
      return input:numel() ~= 0
   else
      assert(type(input) == 'table')
      for i = 1, #input do
         if _hasData(input[i]) then
            return true
         end
      end
      return false
   end
end

function ModelParallelTable:updateOutput(input)
   if self.flattenParams and not hasFlattenedParameters(self) then
      self:flattenParameters()
   end
   if self.needsSync then
      self:syncParameters()
   end

   local prevGpuid = cutorch.getDevice()

   -- distribute the input to GPUs
   self:_distribute(self.inputGpu, input)

   -- update output for each module
   local k = self.nResFunc / 2
   local inputGpu = self.inputGpu
   self.outputGpu = self.impl:exec(function(m, i)
      if _hasData(inputGpu[i]) then
         local output =  m:get(k*i):updateOutput(inputGpu[i])
         for j=1,k-1 do
             output:add(m:get(k*i-j):updateOutput(inputGpu[i]))
         end
         if i == 1 then -- identity is computed on the first GPU
             output:add(m:get(2*k+1):updateOutput(inputGpu[i]))
         end
         return output
      else
         return inputGpu[i]
      end
   end)

   cutorch.setDevice(prevGpuid)
   -- concatenate the outputs to the base GPU
   self.output = self:_add(self.output, self.outputGpu)


   cutorch.setDevice(prevGpuid)

   return self.output
end


function ModelParallelTable:_distribute(dst, src)
   for i = 1, #self.gpuAssignments do
      cutorch.setDevice(self.gpuAssignments[i])
      dst[i] = torch.type(dst[i]) == 'torch.CudaTensor' and dst[i] or torch.CudaTensor()
      dst[i]:resize(src:size()):copy(src)
      waitForDevice(dst[i]:getDevice(), src:getDevice())
   end
end

function ModelParallelTable:_add(dst, src)
   dst = torch.type(dst) == 'torch.CudaTensor' and dst or torch.CudaTensor()
   dst:resize(src[2]:size()):copy(src[2])
   waitForDevice(dst:getDevice(), src[2]:getDevice())
   dst:add(src[1])
   return dst
end

function ModelParallelTable:moduleParameters()
   -- Returns a table containing the parameters for each replica
   if self.flattenedParams then
      local res = {}
      for i, params in ipairs(self.flattenedParams) do
         res[i] = { {params[1]}, {params[2]} }
      end
      return res
   end
   return self.impl:exec(function(m)
      return { m:parameters() }
   end)
end

function ModelParallelTable:__backward(method, input, gradOutput, scale)
   local prevGpuid = cutorch.getDevice()
   local inputGpu, gradOutputGpu = self.inputGpu, self.gradOutputGpu

   if method == 'backward' or method == 'updateGradInput' then
      -- distribute the gradOutput to GPUs
      self:_distribute(self.gradOutputGpu, gradOutput)

      local k = self.nResFunc / 2
      self.gradInputGpu = self.impl:exec(function(m, i)
         if torch.isTensor(inputGpu[i]) and inputGpu[i]:numel() == 0 then
            return torch.CudaTensor()
         else
            local output =  m:get(k*i):backward(inputGpu[i], gradOutputGpu[i])
            for j=1,k-1 do
               output:add(m:get(k*i-j):backward(inputGpu[i], gradOutputGpu[i]))
            end
            if i == 1 then -- identity is computed on the first GPU
                   output:add(m:get(2*k+1):backward(inputGpu[i], gradOutputGpu[i]))
            end
            return output
         end
      end)


      if self.gradInput then
         cutorch.setDevice(prevGpuid)
         -- concatenate the gradInput to the base GPU
         self.gradInput = self:_add(self.gradInput, self.gradInputGpu)
      end
   end

   if method == 'accGradParameters' then
      self.impl:exec(function(m, i)
         if not _hasData(inputGpu[i]) then
            return inputGpu[i]
         else
            return m:accGradParameters(inputGpu[i], gradOutputGpu[i], scale)
         end
      end)
   end

   if method == 'backward' or method == 'accGradParameters' then
      local params = self:moduleParameters()
      -- Accumulate the gradients onto the base GPU
      if self.flattenedParams and self.usenccl and not cudaLaunchBlocking then
         if #self.gpuAssignments > 1 then
            nccl.reduce(pluck(self.flattenedParams, 2), nil, true, 1)
         end
      else
         self:_reduce(pluck(params, 2))
      end
      -- Zero out gradients on the other GPUs
      for i = 2, #self.gpuAssignments do
         cutorch.setDevice(self.gpuAssignments[i])
         for _, gradParam in ipairs(params[i][2]) do
            gradParam:zero()
         end
      end
      self.needsSync = true
   end

   cutorch.setDevice(prevGpuid)
   return self.gradInput
end

function ModelParallelTable:backward(input, gradOutput, scale)
   return self:__backward('backward', input, gradOutput, scale)
end

function ModelParallelTable:updateGradInput(input, gradOutput)
   return self:__backward('updateGradInput', input, gradOutput)
end

function ModelParallelTable:accGradParameters(input, gradOutput, scale)
   self:__backward('accGradParameters', input, gradOutput, scale)
end

function ModelParallelTable:syncParameters()
   local prevGpuid = cutorch.getDevice()
   if self.flattenedParams and self.usenccl and not cudaLaunchBlocking then
      if #self.gpuAssignments > 1 then
         nccl.bcast(pluck(self.flattenedParams, 1), true, 1)
      end
   else
      self:_broadcast(pluck(self:moduleParameters(), 1))
   end
   self.needsSync = false
   cutorch.setDevice(prevGpuid)
end

function ModelParallelTable:accUpdateGradParameters(input, gradOutput, lr)
   error("accUpdateGradParameters not supported for ModelParallelTable.")
end

function ModelParallelTable:zeroGradParameters()
   local prevGpuid = cutorch.getDevice()
   if self.flattenedParams then
      for i, parameters in ipairs(self.flattenedParams) do
         cutorch.setDevice(self.gpuAssignments[i])
         parameters[2]:zero()
      end
   else
      self.impl:exec(function(m)
         m:zeroGradParameters()
      end)
   end
   cutorch.setDevice(prevGpuid)
end

function ModelParallelTable:updateParameters(learningRate)
   local prevGpuid = cutorch.getDevice()
   cutorch.setDevice(self.gpuAssignments[1])
   self.modules[1]:updateParameters(learningRate)
   self:syncParameters()
   cutorch.setDevice(prevGpuid)
end

function ModelParallelTable:parameters()
   return self.modules[1]:parameters()
end

function ModelParallelTable:share(mlp,...)
   error("Share not supported for ModelParallelTable")
end

function ModelParallelTable:clone(...)
   assert(select('#',...) == 0, "Sharing not supported for ModelParallelTable")
   return parent.clone(self)
end

function ModelParallelTable:reset(stdv)
   local prevGpuid = cutorch.getDevice()
   cutorch.setDevice(self.gpuAssignments[1])
   self.modules[1]:reset(stdv)
   self:syncParameters()
   cutorch.setDevice(prevGpuid)
end

function ModelParallelTable:type(typeStr)
   assert(typeStr == 'torch.CudaTensor', 'ModelParallelTable supports only torch.CudaTensor type')
   for i, m in ipairs(self.modules) do
      m:type(typeStr)
   end
   return self
end

-- Backward compatibility purposes
ModelParallelTable.__version = 3

-- ModelParallelTable.deserializeNGPUs controls how many GPUs to deserialize
-- upon, otherwise will deserialize to as many GPUs as serialized and error
-- out if it doesn;t have enough available
function ModelParallelTable:__read(file, version)
   if version < 2 then
      local var = file:readObject()
      for k, v in pairs(var) do
         self[k] = v
      end
      self.impl = self.impl or Impls.Basic(self)
      return
   end

   -- Pre-read gpuAssignments and either use them of ignore them depending on
   -- whether ModelParallelTable.deserializeNGPUs is set.
   local gpuAssignments = file:readObject()
   if ModelParallelTable.deserializeNGPUs then
      gpuAssignments = {}
      for i = 1, ModelParallelTable.deserializeNGPUs do gpuAssignments[i] = i end
      if ModelParallelTable.deserializeNGPUs > cutorch.getDeviceCount() then
         error('Deserialization requested on too many GPUs: ' ..
                  ModelParallelTable.deserializeNGPUs .. ' vs ' ..
                  cutorch.getDeviceCount() .. ' available')
      end
   end

   -- If ModelParallelTable.deserializeNGPUs, deserialization overrides
   -- gpu assignments anyway. If not, we need as many GPUs as the max,
   -- there may be holes.
   local nGPUs = math.max(unpack(gpuAssignments))
   if nGPUs > cutorch.getDeviceCount() then
      error('Model was serialized on ' ..
               math.max(unpack(gpuAssignments)) ..
               ' nGPUs, but you are running on ' .. cutorch.getDeviceCount() ..
               ' please set ModelParallelTable.deserializeNGPUs to ignore ' ..
               ' serialized tower-GPU assignments')
   end

   local prevGpuid = cutorch.getDevice()
   cutorch.setDevice(gpuAssignments[1])
   -- Deserialize from table
   local var = file:readObject()
   for k, v in pairs(var) do
      self[k] = v
   end
   cutorch.setDevice(prevGpuid)

   if self.usenccl then
      self.usenccl = pcall(require, 'nccl')
   end
   if not self.impl then
      self.impl = Impls.Basic(self)
   end

   -- use previously deserialize / recomputed gpuAssignments
   self.gpuAssignments = gpuAssignments
   assert(#self.modules == 1)

   local flattenedParams = self.flattenedParams
   if flattenedParams then
      self.flattenedParams = self.impl:exec(function(m, i)
         if i == 1 then
            return flattenedParams[1]
         else
            return { m:getParameters() }
         end
      end)
   end
end

function ModelParallelTable:__write(file)
   -- Prewrite the current assignments, we may need them to
   -- deserialize the first tower
   file:writeObject(self.gpuAssignments)
   -- Convert to table
   local t = {}
   for k, v in pairs(self) do
      -- Only keep the flattenedParams from the first module
      if k  == 'flattenedParams' then
         t[k] = {v[1]}
      elseif k == 'inputGpu' or k == 'outputGpu' or k == 'gradInputGpu' or k == 'gradOutputGpu' then
         t[k] = {}
      elseif k == 'buffer' then
         t[k] = nil
      else
         t[k] = v
      end
   end
   file:writeObject(t)
   -- Force synchronization, this keeps you honest
   self:syncParameters()
end

function ModelParallelTable:_reflattenReplicaParameters()
   local flattenedParams = self.flattenedParams
   if flattenedParams then
      self.flattenedParams = self.impl:exec(function(m, i)
         if i == 1 then
            return flattenedParams[1]
         else
            return { m:getParameters() }
         end
      end)
   end
end

function ModelParallelTable:apply(callback)
   parent.apply(self, callback)
   self.impl:applyChanges()
   self:_reflattenReplicaParameters()
end

local function sliceRange(nElem, idx, splits)
   local eltsPerMod = nElem / splits
   local rangeStart = math.ceil((idx - 1) * eltsPerMod) + 1
   if idx == splits then
      return rangeStart, nElem - rangeStart + 1
   else
      return rangeStart, math.ceil(idx * eltsPerMod) - rangeStart + 1
   end
end

local function sumSizes(tensors, dim)
   local size
   for i=1,#tensors do
      if tensors[i]:numel() > 0 then
         if size then
            size[dim] = size[dim] + tensors[i]:size(dim)
         else
            size = tensors[i]:size()
         end
      end
   end
   return size
end

-- Copies the parameters from the first replica to all other replicas
function ModelParallelTable:_broadcast(params)
   for moduleIdx = 2, #params do
      for paramIdx = 1, #params[moduleIdx] do
         params[moduleIdx][paramIdx]:copy(params[1][paramIdx])
      end
      waitForDevice(self.gpuAssignments[moduleIdx], self.gpuAssignments[1])
   end
end

-- Sums all the gradParams on to the first replica
function ModelParallelTable:_reduce(gradParams)
   local dstGpuid = self.gpuAssignments[1]
   cutorch.setDevice(dstGpuid)

   self.buffer = self.buffer or torch.CudaTensor()
   for moduleIdx = 2, #gradParams do
      for paramIdx = 1, #gradParams[moduleIdx] do
         local dst = gradParams[1][paramIdx]
         local src = gradParams[moduleIdx][paramIdx]

         -- Synchronize before and after copy to ensure that it doesn't overlap
         -- with this add or previous adds
         waitForDevice(self.gpuAssignments[moduleIdx], dstGpuid)
         self.buffer:resizeAs(src):copy(src)
         waitForDevice(dstGpuid, self.gpuAssignments[moduleIdx])

         dst:add(self.buffer)
      end
   end
end


-- Single-thread dispatch
function BasicImpl:__init(dpt)
   self.dpt = dpt
end

-- Re-copies the first replica onto all the other GPUs, if already setup
function BasicImpl:applyChanges()
   if self.modules then
      local prevGpuid = cutorch.getDevice()
      self.modules = { self.dpt.modules[1] }
      collectgarbage()
      for i=2,#self.dpt.gpuAssignments do
         cutorch.setDevice(self.dpt.gpuAssignments[i])
         table.insert(self.modules, self.dpt.modules[1]:clone())
      end
      cutorch.setDevice(prevGpuid)
   end
end

-- Copies the first replica onto all the other GPUs, if necessary
function BasicImpl:setup()
   if not self.modules then
      self.modules = {}
      self:applyChanges()
   end
end

-- Applies a function to each replica, combining the results into a table
function BasicImpl:exec(closure)
   local prevGpuid = cutorch.getDevice()
   self:setup()
   local res = {}
   for i, gpu in ipairs(self.dpt.gpuAssignments) do
      cutorch.setDevice(gpu)
      res[i] = closure(self.modules[i], i)
   end
   cutorch.setDevice(prevGpuid)
   return res
end

function BasicImpl:__write(file)
   local t = {}
   for k, v in pairs(self) do
      if k ~= 'modules' then
         t[k] = v
      end
   end
   file:writeObject(t)
end

function BasicImpl:close()
   self.modules = nil
end

-- Multi-threaded dispatch
function ThreadsImpl:__init(dpt, initFunc)
   self.dpt = dpt
   self.initFunc = initFunc
end

function ThreadsImpl:applyChanges()
   if self.__threads then
      local module = self.dpt.modules[1]
      for i, gpu in ipairs(self.dpt.gpuAssignments) do
         self.__threads:addjob(i, function()
            cutorch.setDevice(gpu)
            if i == 1 then
               _G.module = module
            else
               _G.module = nil
               collectgarbage()
               _G.module = module:clone()
            end
         end)
      end
      self.__threads:synchronize()
   end
end

function ThreadsImpl:setup()
   if not self.__threads then
      local threads = require 'threads'
      threads.Threads.serialization('threads.sharedserialize')
      self.__threads = threads.Threads(
         #self.dpt.gpuAssignments,
         function() require 'cunn' end,
         self.initFunc)
      self.__threads:specific(true)
      self:applyChanges()
   end
end

function ThreadsImpl:exec(closure)
   self:setup()
   local res = {}
   for i=1,#self.dpt.gpuAssignments do
      self.__threads:addjob(i,
         function()
            return closure(_G.module, i)
         end,
         function (_res_)
            res[i] = _res_
         end)
   end
   self.__threads:synchronize()
   return res
end

function ThreadsImpl:close()
   self.__threads:terminate()
   self.__threads = nil
end

function ThreadsImpl:__write(file)
   local t = {}
   for k, v in pairs(self) do
      if k ~= '__threads' then
         t[k] = v
      end
   end
   file:writeObject(t)
end
