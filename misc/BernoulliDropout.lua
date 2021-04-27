local BernoulliDropout, Parent = torch.class('nn.BernoulliDropout', 'nn.Module')

function BernoulliDropout:__init(p,stochasticInference)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true  -- version 2 scales output during training instead of evaluation
   self.stochastic_inference = stochasticInference or false
   self.noise = torch.Tensor()
   if self.p >= 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p < 1')
   end

end

function BernoulliDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.p > 0 then
      if self.train or self.stochastic_inference then
         self.noise:resizeAs(input)
         self.noise:bernoulli(1-self.p)
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

function BernoulliDropout:updateGradInput(input, gradOutput)
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

function BernoulliDropout:setp(p)
   self.p = p
end

function BernoulliDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function BernoulliDropout:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end
