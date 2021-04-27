--Ref: https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb
--Variational Gaussian Dropout is not Bayesian: https://arxiv.org/pdf/1711.02989.pdf


local GaussianDropout, Parent = torch.class('nn.GaussianDropout', 'nn.Module')

function GaussianDropout:__init(p,stochasticInference)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true  -- version 2 scales output during training instead of evaluation
   self.stochastic_inference = stochasticInference or false
   self.noise = torch.Tensor()
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end

end

function GaussianDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.p > 0 then
      if self.train or self.stochastic_inference then
         self.noise:resizeAs(input)
         self.noise:normal(1, (self.p/(1-self.p)))
         if self.train then  --optional-- version 2 scales output during training instead of evaluation
            self.noise:div(1-self.p)
         end
         self.output:cmul(self.noise)
      else
         self.output:mul(1-self.p)
      end
   end
   return self.output
end

function GaussianDropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.train then
      if self.p > 0 then
         self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
      end
   else
      if not self.train and self.p > 0 then
         self.gradInput:mul(1-self.p)
      end
   end
   return self.gradInput
end

function GaussianDropout:setp(p)
   self.p = p
end

function GaussianDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function GaussianDropout:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end
