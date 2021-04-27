------------------------------------------------------------------------------
-- this is dataloader for sequential example reading
---------------------------------------------------------------------------------------
require 'hdf5'
cjson = require 'cjson'
utils = require 'misc/utils'

local dataloader = {}

function dataloader:initialize(opt)
    print('Reading ' .. opt.input_json)
    local file = io.open(opt.input_json, 'r')
    local text = file:read()
    file:close()
    local params = cjson.decode(text)
    for k,v in pairs(params) do self[k] = v end
    self['vocab_size'] = 0 for i,w in pairs(self['ix_to_word']) do self['vocab_size'] = self['vocab_size'] + 1 end

------------------------------------------------------------------------------------------------------
-- this is for getting image information

         if opt['input_img_train_h5'] ~= nil then
                print('Reading DataLoader loading h5 image file:' .. opt['input_img_train_h5'])
                self.h5_img_file_train = hdf5.open(opt['input_img_train_h5'], 'r')
        end
        if opt['input_img_test_h5'] ~= nil then
                print('Reading  DataLoader loading h5 image file: ' .. opt['input_img_test_h5'])
                self.h5_img_file_test  = hdf5.open(opt['input_img_test_h5'], 'r')
        end

        if opt['input_place_img_train_h5'] ~= nil then
                print('Reading DataLoader loading place_ h5 image file:' .. opt['input_place_img_train_h5'])
                self.h5_place_img_file_train = hdf5.open(opt['input_place_img_train_h5'], 'r')
        end
        if opt['input_place_img_test_h5'] ~= nil then
                print('Reading  DataLoader loading place_ h5 image file: ' .. opt['input_place_img_test_h5'])
                self.h5_place_img_file_test  = hdf5.open(opt['input_place_img_test_h5'], 'r')
        end

----------------------------------------------------------------------------------------------------------
        -- this is getting question information
        print ('DataLoader loading h5 question file: ',opt.input_ques_h5)
        local qa_data = hdf5.open(opt.input_ques_h5, 'r')
        print('qa_data',qa_data)
        print('********************************')

       -- if split == 'train' then
       -- split is not required bcz here , u have chanaged variale name as like ques_train from ques and ques_test from ques which implecite indicate split
                -- image
                self['im_list_train']   = qa_data:read('/img_pos_train'):all()  --or self['im_list_train'] =self.im_list_train
                -- question
                self['ques_train']      = qa_data:read('/ques_train'):all()
                self['ques_len_train']  = qa_data:read('ques_length_train'):all()
                --self['ques_train']      = utils.right_align(self['ques_train'], self['ques_len_train'])-- you will get  bad argument #1 to 'unpack' (table expected, got nil)--unpack(self.state[t-1])}
                self['ques_id_train']   = qa_data:read('/question_id_train'):all()
                -- caption
                self['cap_train']       = qa_data:read('/cap_train'):all()
                self['cap_len_train']   = qa_data:read('cap_length_train'):all()
                self['cap_train']       = utils.right_align(self['cap_train'], self['cap_len_train'])

                 -- Tag-Noun
                self['noun_train']       = qa_data:read('/noun_train'):all()
                self['noun_len_train']   = qa_data:read('noun_length_train'):all()
                self['noun_train']       = utils.right_align(self['noun_train'], self['noun_len_train'])


                --  Tag-Verb
                self['verb_train']       = qa_data:read('/verb_train'):all()
                self['verb_len_train']   = qa_data:read('verb_length_train'):all()
                self['verb_train']       = utils.right_align(self['verb_train'], self['verb_len_train'])


                --  Tag-Wh word
                self['whword_train']       = qa_data:read('/whword_train'):all()
                self['whword_len_train']   = qa_data:read('whword_length_train'):all()
                self['whword_train']       = utils.right_align(self['whword_train'], self['whword_len_train'])

                -- answer
               -- self['ans_train']       = qa_data:read('/answers'):all()
                self['train_id']  = 1
                self.seq_length = self.ques_train:size(2)

                -- to print complete size of each split
                print('self[ques_train]:size(1)',self['ques_train']:size(1))

        --elseif split == 'test' then
        -- split is not required bcz here , u have chanaged variale name as like ques_train from ques and ques_test from ques which implecite indicate split

                -- image
                self['im_list_test']   = qa_data:read('/img_pos_test'):all()
                -- question
                self['ques_test']      = qa_data:read('/ques_test'):all()
                self['ques_len_test']  = qa_data:read('ques_length_test'):all()
                --self['ques_test']      = utils.right_align(self['ques_test'], self['ques_len_test'])
                self['ques_id_test']   = qa_data:read('/question_id_test'):all()
                -- caption
                self['cap_test']       = qa_data:read('/cap_test'):all()
                self['cap_len_test']   = qa_data:read('cap_length_test'):all()
                self['cap_test']       = utils.right_align(self['cap_test'], self['cap_len_test'])

                -- Tag-Noun
                self['noun_test']       = qa_data:read('/noun_test'):all()
                self['noun_len_test']   = qa_data:read('noun_length_test'):all()
                self['noun_test']       = utils.right_align(self['noun_test'], self['noun_len_test'])



                --  Tag-Verb
                self['verb_test']       = qa_data:read('/verb_test'):all()
                self['verb_len_test']   = qa_data:read('verb_length_test'):all()
                self['verb_test']       = utils.right_align(self['verb_test'], self['verb_len_test'])


                --  Tag-Wh word
                self['whword_test']       = qa_data:read('/whword_test'):all()
                self['whword_len_test']   = qa_data:read('whword_length_test'):all()
                self['whword_test']       = utils.right_align(self['whword_test'], self['whword_len_test'])

                -- answer
                --self['ans_test']       = qa_data:read('/answers_test'):all()
                self['test_id']   = 1
                -- to print complete size of each split
                print('self[ques_test]:size(1)',self['ques_test']:size(1))



        --end
        qa_data:close()
end

function dataloader:next_batch(opt)
    local start_id = self['train_id'] -- start id , and it  it wiil be remember for next batch
    if start_id + opt.batch_size - 1 <= self['ques_train']:size(1) then
        end_id = start_id + opt.batch_size - 1
    else
        self['train_id'] =1  --reset train id to 1
        start_id = self['train_id']
        end_id = start_id + opt.batch_size - 1
        print('end of epoch')
    end


    local iminds = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  question index
    local qinds = torch.LongTensor(end_id - start_id + 1):fill(0) -- to keep track of  question index
    local im    = torch.LongTensor(opt.batch_size, 4096):fill(0)   --14, 14, 512):fill(0)  --chanaged for fc7 -- for store img batch of size 14x14x512 changed to 4096
    --added badri
    local im_place    = torch.LongTensor(opt.batch_size, 4096):fill(0)   --14, 14, 512):fill(0)  --chanaged for fc7 -- for store img batch of size 14x14x512 changed to 4096

    for i = 1, end_id - start_id + 1 do
        qinds[i] = start_id + i - 1               -- this  is for sequential
        iminds[i] = self['im_list_train'][qinds[i]] -- extract image id from image list
        im[i] =  self.h5_img_file_train:read('/images_train'):partial({iminds[i],iminds[i]},{1,4096}) --{1,14},{1,14},{1,512}) --chanaged for fc7
        --added badri
        im_place[i] =  self.h5_place_img_file_train:read('/images_train'):partial({iminds[i],iminds[i]},{1,4096}) --{1,14},{1,14},{1,512}) --chanaged for fc7
    end

    --local im = self['fv_im']:index(1, iminds)
    local ques    = self['ques_train']:index(1, qinds)
    --local labels  = self['ans_train']:index(1, qinds)
    local cap     = self['cap_train']:index(1, qinds)
    local ques_id = self['ques_id_train']:index(1, qinds)
    local noun     = self['noun_train']:index(1, qinds)
    local verb     = self['verb_train']:index(1, qinds)
    local whword     = self['whword_train']:index(1, qinds)

    if opt.gpuid >= 0 then
            im     = im:cuda()
            ques   = ques:cuda()
            --labels = labels:cuda()
            cap    = cap:cuda()
            im_place     = im_place:cuda()
            noun    = noun:cuda()
            verb    = verb:cuda()
            whword    = whword:cuda()
    end

        self['train_id'] = self['train_id'] + end_id - start_id + 1   -- self['test_id']=  self.test_id both have same meaning
    return {im, ques,cap,ques_id,im_place,noun,verb,whword}
end

function dataloader:next_batch_eval(opt)
    local start_id = self['test_id']
    local end_id = math.min(start_id + opt.batch_size - 1, self['ques_test']:size(1))  --here it do sequential basic because it will check complete data set

    local iminds = torch.LongTensor(end_id - start_id + 1):fill(0)
    local qinds = torch.LongTensor(end_id - start_id + 1):fill(0)
    local im    = torch.LongTensor(end_id - start_id + 1, 4096):fill(0)   --14, 14, 512):fill(0)--chanaged for fc7
    local im_place    = torch.LongTensor(end_id - start_id + 1, 4096):fill(0)   --14, 14, 512):fill(0)--chanaged for fc7

    for i = 1, end_id - start_id + 1 do
        qinds[i] = start_id + i - 1
        iminds[i] = self['im_list_test'][qinds[i]]
        im[i] = self.h5_img_file_test:read('/images_test'):partial({iminds[i],iminds[i]},{1,4096}) --{1,14},{1,14},{1,512}) --chanaged for fc7.
        im_place[i] = self.h5_place_img_file_test:read('/images_test'):partial({iminds[i],iminds[i]},{1,4096}) --{1,14},{1,14},{1,512}) --chanaged for fc7
    end

        -- print('Ques_cap_id BEFORE',self['ques_id_test'])
        --local im = self['fv_im']:index(1, iminds)
        local ques    = self['ques_test']:index(1, qinds)
        local ques_id = self['ques_id_test']:index(1, qinds)
        -- local labels  = self['ans_test']:index(1, qinds)
        local cap     = self['cap_test']:index(1, qinds)
        local noun     = self['noun_test']:index(1, qinds)
        local verb     = self['verb_test']:index(1, qinds)
        local whword     = self['whword_test']:index(1, qinds)

        if opt.gpuid >= 0 then
            im     = im:cuda()
            ques   = ques:cuda()
            --labels =labels:cuda()
            cap    = cap:cuda()
            im_place  = im_place:cuda()
            noun    = noun:cuda()
            verb    = verb:cuda()
            whword    = whword:cuda()
        end
    -- print('Ques_cap_id AFTER',ques_id)
    self['test_id'] = self['test_id'] + end_id - start_id + 1   -- self['test_id']=  self.test_id both have same meaning

    return {im, ques, ques_id,cap,im_place,noun,verb,whword}
end
function dataloader:getVocab(opt)
     return self.ix_to_word
end

function dataloader:getVocabSize()
    return self['vocab_size'] -- or self.vocab_size
end

function dataloader:resetIterator(split)
        if split ==1 then
                self['train_id'] = 1
        end
        if split ==2  then
                self['test_id']=1
        end
end


function dataloader:getDataNum(split)
        if split ==1 then
               return self['ques_train']:size(1)
        end
        if split ==2  then
             return  self['ques_test']:size(1)
        end
end

function dataloader:getSeqLength()
  return self.seq_length
end


return dataloader
