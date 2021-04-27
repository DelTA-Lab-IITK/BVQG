-- Based on JoinTable module
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'cephes'
local THNN = require 'nn.THNN'

local KLDiv, parent = torch.class('nn.KLDiv',  'nn.Criterion')

function KLDiv:__init(alpha0_q, alpha0_p)
    parent.__init(self)
	self.alpha0_q = alpha0_q
	self.alpha0_p = alpha0_p
	self.gradInput = {}

end 

function lgamma(input)
	local temp = (cephes.lgam(input:float())):type(input:type())
	return temp:resizeAs(input)
end

function digamma(input)
	local temp = (cephes.digamma(input:float())):type(input:type())
	return temp:resizeAs(input)
end

function polygamma(input)
	local temp = (cephes.polygamma(1, input:float())):type(input:type())
	return temp:resizeAs(input)
end

function KLDiv:updateOutput(input_q, input_p)
	self.output = lgamma(self.alpha0_q) - lgamma(self.alpha0_p) - torch.mean(torch.sum(lgamma(input_q), 2)) + torch.mean(torch.sum(lgamma(input_p), 2)) + (self.alpha0_p - self.alpha0_q)*digamma(self.alpha0_q) + torch.mean(torch.sum(torch.cmul(digamma(input_q), torch.csub(input_q, input_p)), 2))
	return self.output
end

function KLDiv:updateGradInput(input_q, input_p)
    self.gradInput[1] = self.gradInput[1] or input_q.new()
    self.gradInput[1]:resizeAs(input_q)
	self.gradInput[2] = self.gradInput[2] or input_p.new()
    self.gradInput[2]:resizeAs(input_p)
	
	self.gradInput[1] = torch.cmul(polygamma(input_q), torch.csub(input_q, input_p))
	self.gradInput[2] = digamma(input_p) - digamma(input_q) 
    return self.gradInput
end




