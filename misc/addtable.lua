local addtable, parent = torch.class('nn.addtable', 'nn.Module')

function addtable:__init(ip)
   parent.__init(self)
   self.inplace = ip
   self.gradInput = {}
end

function addtable:updateOutput(input)
   if self.inplace then
      self.output:set(input[1])
   else
      self.output:resizeAs(input[1]):copy(input[1])
      print('self.output:resizeAs(input[1])',self.output:resizeAs(input[1]))
      print('resizeAs(input[1])',self.output:resizeAs(input[1]):copy(input[1]))
   end
   for i=2,#input do
      self.output:add(input[i])
   end
   return self.output
end

function addtable:updateGradInput(input, gradOutput)
   print('#input',#input)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      if self.inplace then
         self.gradInput[i]:set(gradOutput)
      else
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)
      end
   end
   print('#self.gradInput',#self.gradInput)
   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
