--refered link:  https://github.com/casperkaae/nn/blob/master/MulConstant.lua
require 'nn'
local mul_constant, parent = torch.class('nn.mul_constant', 'nn.Module')

function mul_constant:__init(constant_scalar)
  parent.__init(self)
  --assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar -- default initilize with 1, then modify later to new constatant
end

function mul_constant:reset_constant(new_constant)
   self.constant_value = new_constant
end

function mul_constant:updateOutput(input)
    local expert          = input[1]
    local gating_constant = input[2] -- this is caonstant value

    self.constant_scalar=gating_constant-- to update the constant value.

    self.output:resizeAs(expert)
    self.output:copy(expert)
    self.output:mul(self.constant_scalar)

  return self.output
end 

function mul_constant:updateGradInput(input, gradOutput)
    local expert          = input[1]
    local gating_constant = input[2] -- this is caonstant value

    assert(self.constant_scalar == gating_constant, 'gating_constant is not updated!')

    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
    self.gradInput:mul(self.constant_scalar)
  return self.gradInput
end




--ref link: https://github.com/soumith/recpool/blob/master/MulConstant.lua
-- local MulConstant, parent = torch.class('nn.MulConstant', 'nn.Module')

-- function MulConstant:__init(input_size, constant_value)
--    parent.__init(self)

--    --self.gradInput:resize(input_size)
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



--if you do twice dackward the gradInput is decreasing
-- function MulConstant:updateGradInput(input, gradOutput)
--    self.gradInput:resizeAs(gradOutput)
--    self.gradInput:copy(gradOutput):mul(self.constant_value)
--    return self.gradInput
-- end