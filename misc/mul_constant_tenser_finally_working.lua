--refered link:  https://github.com/casperkaae/nn/blob/master/MulConstant.lua
require 'nn'
local mul_constant_tenser, parent = torch.class('nn.mul_constant_tenser', 'nn.Module')

function mul_constant_tenser:__init(constant_scalar,batch_size)
  parent.__init(self)
  self.batch_size = batch_size 
  --assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar -- default initilize with 1, then modify later to new constatant
  self.gradInput={}
  self.gradInput2=torch.Tensor()
  self.gradInput1=torch.Tensor()
end

function mul_constant_tenser:reset_constant(new_constant)
   self.constant_value = new_constant
end

function mul_constant_tenser:updateOutput(input)
    local expert          = input[1]
    local gating_constant = input[2] -- this is caonstant value

    self.constant_scalar=gating_constant-- to update the constant value.

    self.output:resizeAs(expert)
    self.output:copy(expert)
    --print('self.batch_size',self.batch_size)

    for i=1,self.batch_size do
      --print('self.output',self.constant_scalar[i])
         self.output[i]:mul(self.constant_scalar[i])
    end
        
--print('self.output',self.output)
   -- self.output:mul(self.constant_scalar)

  return self.output
end 


--self.gradInput  and self.output these are the key word automatically defined
-- if you want to chanage like this self.gradInput1 , then need to define in initilization function
function mul_constant_tenser:updateGradInput(input, gradOutput)
     local expert          = input[1]
    local gating_constant = input[2] -- this is caonstant value

    assert(self.constant_scalar == gating_constant, 'gating_constant is not updated!')

    self.gradInput1:resizeAs(gradOutput):copy(gradOutput)
    for i=1,self.batch_size do
        self.gradInput1[i]:mul(self.constant_scalar[i])
    end
    print('self.gradInput1',self.gradInput1)

    self.gradInput2:resizeAs(input[2]):zero()
    for i=1,self.batch_size do
        self.gradInput2[i]=expert[i]:dot(gradOutput[i])--:mul(self.constant_scalar[i])
    end

    print('self.gradInput2',self.gradInput2)
   table.insert(self.gradInput, self.gradInput1)
  table.insert(self.gradInput, self.gradInput2)
-- self.gradInput={self.gradInput1,self.gradInput2}

    print('self.gradInput',self.gradInput)
      return self.gradInput
end




--ref link: https://github.com/soumith/recpool/blob/master/MulConstant.lua
-- local MulConstant, parent = torch.class('nn.MulConstant', 'nn.Module')

-- function MulConstant:__init(input_size, constant_value)
--    parent.__init(self)

--    --self.gradInput1:resize(input_size)
--    --self.output:resize(input_size) 
   
--    self.constant_value = constant_value
-- end

-- function MulConstant:reset_constant(new_constant)
--    self.constant_value = new_constant
-- end

-- function MulConstant:updateOutput(input)
--    self.output:resizeAs(input)
--    self.output:copy(input):mul(self.constant_value)
--    return self.output
-- end



--if you do twice dackward the gradInput1 is decreasing
-- function MulConstant:updateGradInput(input, gradOutput)
--    self.gradInput1:resizeAs(gradOutput)
--    self.gradInput1:copy(gradOutput):mul(self.constant_value)
--    return self.gradInput1
-- end