------------------------------------------------------------------------------------
--  Torch Implementation of Stack attention based Networks for Visual  Question generation
--  momentum= 0.9, learning_rate=4e-4, batch_size=100, lr_decay= no
--  dim_embed=512,dim_hidden= 512,dim_image= 4096.
--  th train -gpuid 1


------------------------------------------------------------------------------------

require 'nn'
require 'torch'
require 'rnn'
require 'optim' --this is for only log only not for update parameter
require 'misc.LanguageModel'
require 'misc.optim_updates'
require 'misc.BernoulliDropout'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local Multimodal=require 'misc.multimodal'
require 'xlua'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')


-- Data input settings
cmd:option('-input_img_train_h5','img_train_fc7.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','img_test_fc7.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','coco_data_prepro_pos_tv.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','coco_data_prepro_pos_tv.json','path to the json file containing additional info and vocab')

cmd:option('-input_place_img_train_h5','../../../vqg_context/coco_place_joint_mul_fc7_l36_tv/data/img_train_fc7_place_l36.h5','path to the h5file containing the image feature')
cmd:option('-input_place_img_test_h5','../../../vqg_context/coco_place_joint_mul_fc7_l36_tv/data/img_test_fc7_place_l36.h5','path to the h5file containing the image feature')

-- starting point
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

-- Model settings
cmd:option('-batch_size',200,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-att_size',512,'size of sttention vector which refer to k in paper')
cmd:option('-emb_size',512,'the size after embeeding from onehot')
cmd:option('-rnn_layers',1,'number of the rnn layer')

-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.0008,'learning rate')--0.0001,--0.0002,--0.005
cmd:option('-learning_rate_decay_start', 5, 'at what epoch to start decaying learning rate? (-1 = dont)')--learning_rate_decay_start', 100,
cmd:option('-learning_rate_decay_every', 5, 'every how many epoch thereafter to drop LR by half?')---learning_rate_decay_every', 1500,
cmd:option('-momentum',0.9,'momentum')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')--optim_alpha',0.99
cmd:option('-optim_beta',0.999,'beta used for adam')--optim_beta',0.995
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1250)
cmd:option('-drop_prob_lm', 0.5, 'strength of drop_prob_lm in the Language Model RNN')


-- Evaluation/Checkpointing
cmd:text('===>Save/Load Options')
cmd:option('-save',               'Results', 'save directory')
cmd:option('-checkpoint_dir', 'Results/checkpoints', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-val_images_use', 31200, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-losses_log_every', 200, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 1234, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')


cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
print(opt)
opt = cmd:parse(arg)
f = assert(io.open("Scores_of_all_seed.txt", "w"))

-------------------------------------------------------------------------------
-- Seed fix in torch and cutorch for randomization
-------------------------------------------------------------------------------

-- cutorch.manualSeed(opt.seed)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.backend == 'cudnn' then
                require 'cudnn'
        end
        cutorch.manualSeed(opt.seed)
        torch.manualSeed(opt.seed)
        -- cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end





---------------------------------------------------------------------
--Step 4: create directory and log file
------------------------------------------------------------------
------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save) -- to create result folder  save folder
cmd:log(opt.save .. '/Log_cmdline.txt', opt)  --save log file in save folder
--os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)  -- to copy network to the save file path

-- to save model parameter
os.execute('mkdir -p ' .. opt.checkpoint_dir)

-- to save log
local err_log_filename = paths.concat(opt.save,'ErrorProgress')
local err_log = optim.Logger(err_log_filename)

-- to save log
local lang_stats_filename = paths.concat(opt.save,'language_statstics')
local lang_stats_log = optim.Logger(lang_stats_filename)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
-- dataloader
--local dataloader = dofile('misc/dataloader.lua')
local dataloader = dofile('misc/dataloader.lua')
dataloader:initialize(opt)
collectgarbage()
------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
local loaded_checkpoint
local lmOpt
-- intialize language model
if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  loaded_checkpoint = torch.load(opt.start_from)
--   protos = loaded_checkpoint.protos

--   local lm_modules = protos.lm:getModulesList()
--   for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end

  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually

else
        -- create protos from scratch
        print('Building the model from scratch...')
        -- intialize language model
        lmOpt = {}
        lmOpt.vocab_size = dataloader:getVocabSize()
        lmOpt.input_encoding_size = opt.input_encoding_size
        lmOpt.rnn_size = opt.rnn_size
        lmOpt.num_layers = 1
        lmOpt.drop_prob_lm = opt.drop_prob_lm
        lmOpt.seq_length = dataloader:getSeqLength()
        lmOpt.batch_size = opt.batch_size
        lmOpt.emb_size= opt.input_encoding_size
        lmOpt.hidden_size = opt.input_encoding_size
        lmOpt.att_size = opt.att_size
        lmOpt.num_layers = opt.rnn_layers

end

lmOpt = {}
        lmOpt.vocab_size = dataloader:getVocabSize()
        lmOpt.input_encoding_size = opt.input_encoding_size
        lmOpt.rnn_size = opt.rnn_size
        lmOpt.num_layers = 1
        lmOpt.drop_prob_lm = opt.drop_prob_lm
        lmOpt.seq_length = dataloader:getSeqLength()
        lmOpt.batch_size = opt.batch_size
        lmOpt.emb_size= opt.input_encoding_size
        lmOpt.hidden_size = opt.input_encoding_size
        lmOpt.att_size = opt.att_size
        lmOpt.num_layers = opt.rnn_layers
-- Design Model From scratch
---------------------------------------------------------------------------------------------
----------------------------------------------------------------
-- Note On Dropout
--nn.Dropout(0.5,nil,nil,true) is different from nn.Dropout(0.5),
--training time both are same, but  evalution time both are different

--nn.Dropout(0.5,nil,nil,true) is similar to nn.BernoulliDropout(0.5,true)
--where nn.BernoulliDropout(0.5,true) is clear and simpler version of nn.Dropout(0.5,nil,nil,true)
-- nn.BernoulliDropout(0.5,true) is different form  nn.GaussianDropout(0.5,true)
---------------------------------------------------------------------------------
-- Encoding Part

        --Caption feature embedding
        --protos.emb = nn.emb_net(lmOpt) -- because problem in sharing network
        protos.emb = nn.Sequential()
                :add(nn.LookupTableMaskZero(lmOpt.vocab_size, lmOpt.input_encoding_size))
                :add(nn.BernoulliDropout(0.5,true))
                :add(nn.SplitTable(1, 2))
                :add(nn.Sequencer(nn.FastLSTM(lmOpt.input_encoding_size, lmOpt.rnn_size):maskZero(1)))
                :add(nn.Sequencer(nn.FastLSTM(lmOpt.rnn_size, lmOpt.rnn_size):maskZero(1)))
                :add(nn.SelectTable(-1))

        -- Place feature Image feature embedding
        protos.place = nn.Sequential()
                :add(nn.Linear(4096,opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.BernoulliDropout(0.5,true))
        -- Tag feature embedding
        protos.tag = nn.Sequential()
              :add(nn.LookupTableMaskZero(lmOpt.vocab_size, lmOpt.input_encoding_size))
              :add(nn.BernoulliDropout(0.5,true))
        -- Tag Joint feature embedding
        protos.tag_net = nn.Sequential() -- can be used joint table, add table, also like joint learnable paramater
              :add(Multimodal.AcatBcatC(opt.input_encoding_size,opt.input_encoding_size,opt.input_encoding_size,opt.input_encoding_size,0.5))

        -- Image feature embedding
        protos.cnn = nn.Sequential()
                :add(nn.Linear(4096,opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.BernoulliDropout(0.5,true))
        ---------------------------------------------------------------------------------------------
        -- jointtion feature embedding
        protos.joint1 = nn.Sequential()
                :add(Multimodal.AmulB(opt.input_encoding_size,opt.input_encoding_size,opt.input_encoding_size,0.5))
                :add(nn.BatchNormalization(opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.BernoulliDropout(0.5,true))                :add(nn.Linear(opt.input_encoding_size, opt.input_encoding_size))

        -- jointtion feature embedding
        protos.joint2 = nn.Sequential()
                :add(Multimodal.AaddB(opt.input_encoding_size,opt.input_encoding_size,opt.input_encoding_size,0.5))
                :add(nn.BatchNormalization(opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.BernoulliDropout(0.5,true))
                :add(nn.Linear(opt.input_encoding_size, opt.input_encoding_size))

        -- jointtion feature embedding
        protos.joint3 = nn.Sequential()
                :add(Multimodal.AcatB(opt.input_encoding_size,opt.input_encoding_size,opt.input_encoding_size,0.5))
                :add(nn.BatchNormalization(opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.BernoulliDropout(0.5,true))
                :add(nn.Linear(opt.input_encoding_size, opt.input_encoding_size))
        ---------------------------------------------------------------------------------------------

   ---------------------------------------------------------------------------
		--moe3 part
        -- Place feature Image feature embedding
        protos.gating_net = nn.Sequential()
                :add(nn.Linear(4096,3))
                :add(nn.SoftMax())

              -- Place feature Image feature embedding
        default_constant=torch.Tensor(opt.batch_size, 1):fill(7)
        protos.moe = nn.Sequential()
                :add(Multimodal.moe3(default_constant,opt.batch_size))
---------------------------------------------------------------------------------------------
-- Decoding Part

        -- Question feature embedding
        protos.lm = nn.LanguageModel(lmOpt)

        -- criterion for the language model
        protos.crit = nn.LanguageModelCriterion()

--print('model',protos)
print('seq_length',lmOpt.seq_length)
---------------------------------------------------------------------------------------
print('ship everything to GPU...')
-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
local pparams, grad_pparams = protos.place:getParameters()
local eparams, grad_eparams = protos.emb:getParameters()
local cparams, grad_cparams = protos.cnn:getParameters()
local jparams1, grad_jparams1 = protos.joint1:getParameters()
local jparams2, grad_jparams2 = protos.joint2:getParameters()
local jparams3, grad_jparams3 = protos.joint3:getParameters()
local mparams1, grad_mparams1 = protos.gating_net:getParameters()
local mparams2, grad_mparams2 = protos.moe:getParameters()
local lparams, grad_lparams = protos.lm:getParameters()
local tparams, grad_tparams = protos.tag:getParameters()
local tnparams, grad_tnparams = protos.tag_net:getParameters()



eparams:uniform(-0.1, 0.1)
cparams:uniform(-0.1, 0.1)
jparams1:uniform(-0.1, 0.1)
jparams2:uniform(-0.1, 0.1)
jparams3:uniform(-0.1, 0.1)
lparams:uniform(-0.1, 0.1)
pparams:uniform(-0.1, 0.1)
mparams1:uniform(-0.1, 0.1)
mparams2:uniform(-0.1, 0.1)
tparams:uniform(-0.1, 0.1)
tnparams:uniform(-0.1, 0.1)


if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  eparams:copy(loaded_checkpoint.eparams)
  cparams:copy(loaded_checkpoint.cparams)
  jparams1:copy(loaded_checkpoint.jparams1)
  jparams2:copy(loaded_checkpoint.jparams2)
  jparams3:copy(loaded_checkpoint.jparams3)
  lparams:copy(loaded_checkpoint.lparams)
  pparams:copy(loaded_checkpoint.pparams)
  mparams1:copy(loaded_checkpoint.mparams1)
  mparams2:copy(loaded_checkpoint.mparams2)
  tparams:copy(loaded_checkpoint.tparams)
  tnparams:copy(loaded_checkpoint.tnparams)


end

print('total number of parameters in Question embedding net: ', eparams:nElement())
assert(eparams:nElement() == grad_eparams:nElement())

print('total number of parameters in Image  embedding net: ', cparams:nElement())
assert(cparams:nElement() == grad_cparams:nElement())


print('total number of parameters in joint1 embedding net: ', jparams1:nElement())
assert(jparams1:nElement() == grad_jparams1:nElement())

print('total number of parameters in joint2 embedding net: ', jparams2:nElement())
assert(jparams2:nElement() == grad_jparams2:nElement())

print('total number of parameters in joint3 embedding net: ', jparams3:nElement())
assert(jparams3:nElement() == grad_jparams3:nElement())

print('total number of parameters of language Generating model ', lparams:nElement())
assert(lparams:nElement() == grad_lparams:nElement())


print('total number of parameters of place_net model ', pparams:nElement())
assert(pparams:nElement() == grad_pparams:nElement())


print('total number of parameters in gating_net embedding net: ', mparams1:nElement())
assert(mparams1:nElement() == grad_mparams1:nElement())

print('total number of parameters in moe embedding net: ', mparams2:nElement())
assert(mparams2:nElement() == grad_mparams2:nElement())


print('total number of parameters in Question embedding net: ', tparams:nElement())
assert(tparams:nElement() == grad_tparams:nElement())

print('total number of parameters in Image  embedding net: ', tnparams:nElement())
assert(tnparams:nElement() == grad_tnparams:nElement())

collectgarbage()
---------------------------------------------------------------
-- Clone net  only doing clone
---------------------------------------------------------------

CreateTriplet = function(Net)
  prl = nn.ParallelTable()
  convNetPos = Net:clone('weight', 'bias', 'gradWeight', 'gradBias')
  convNetNeg = Net:clone('weight', 'bias', 'gradWeight', 'gradBias')

  -- Parallel container
  prl:add(Net)
  prl:add(convNetPos)
  prl:add(convNetNeg)  -- give all three same input to get same output three times
  print('Cloneing Image embedding network:');
  print(prl)
  return prl
end


local img_cnn_feat_clone3 =  CreateTriplet(protos.cnn)
---------------------------------------------------------------
-- Clone net  only doing clone
---------------------------------------------------------------
local question_tag_emb_net_clone3 =  nn.MapTable():add(protos.tag)
if opt.gpuid >= 0 then
        question_tag_emb_net_clone3:cuda()
end
print '==>question_tag_emb_net_clone3 Network'
print(question_tag_emb_net_clone3)
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
        protos.emb:evaluate()
        protos.cnn:evaluate()
        protos.joint1:evaluate()
        protos.joint2:evaluate()
        protos.joint3:evaluate()
        protos.lm:evaluate()
        protos.place:evaluate()
        protos.gating_net:evaluate()
        protos.moe:evaluate()
        protos.tag:evaluate()
        protos.tag_net:evaluate()

	dataloader:resetIterator(2)-- 2 for test and 1 for train

        local verbose = utils.getopt(evalopt, 'verbose', false) -- to enable the prints statement  entry.image_id, entry.caption
        local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

        local n = 0
        local loss_sum = 0
        local loss_evals = 0
        local right_sum = 0

        total_num = dataloader:getDataNum(2) -- 2 for test and 1 for train-- this will provide total number of example in the image

        local predictions = {}
        local vocab = dataloader:getVocab()

  while true do
        --local data = loader:getBatch{batch_size = opt.batch_size, split = split}
        local batch = dataloader:next_batch_eval(opt)
        --print('Ques_cap_id In eval batch[3]',batch[3])
        local data = {}
        data.images=batch[1]-- check this in dataloader return sequence
        data.questions=batch[2]
        data.caption=batch[4]
        data.ques_id=batch[3]
        data.image_place=batch[5]
        data.nountag=batch[6]
        data.verbtag=batch[7]
        data.questiontag=batch[8]
     -------------------------------------------------------------------------------------
	n = n + data.images:size(1)
	xlua.progress(n, total_num)

      --------------------------------------------------------------------------------------
        local decode_question= data.questions:t()-- bcz in langauage models checks assert(seq:size(1) == self.seq_length) os it should be 26 x 200
        --print('after transpose data.questions',data.questions:size()) --26x200
        -- bcz this language model needs dimension of size 26x200

        --print('data.caption',data.caption:size())--[torch.DoubleTensor of size 200x1x512]

         -------------------------------------------------------------------------------------------------------------------
        local question_tag1= data.questiontag:select(2,1)
        local question_tag2= data.questiontag:select(2,2)
        local question_tag3= data.questiontag:select(2,3)
        --print("question_tag",question_tag:size())
        -------------------------------------------------------------------------------------------------------------------

        --Forward the question word feature through word embedding
        local question_tag_feat_clone3 =question_tag_emb_net_clone3:forward({question_tag1,question_tag2,question_tag3});
        --print('word_feature',word_feature:size())--[torch.DoubleTensor of size 200x1x512]

        local tag_feat=protos.tag_net:forward({question_tag_feat_clone3[1],question_tag_feat_clone3[2],question_tag_feat_clone3[3]});
        --local noun_tag_feat=nn.CAddTable(2):forward({{noun_tag_feat_clone3[1],noun_tag_feat_clone3[2]}});
        --local noun_tag_feat=nn.JoinTable(2):forward({{noun_tag_feat_clone3[1],noun_tag_feat_clone3[2]}});


        -------------------------------------------------------------------------------------------------------------------
          --Forward the question word feature through word embedding
          local ques_feat =protos.emb:forward(data.caption)
          --print('ques_feat',ques_feat:size())--[torch.DoubleTensor of size 200x1x512]
          --print('ques_feat',ques_feat:max(),ques_feat:min())

          --Forward place  feature through word embedding
          local place_feat =protos.place:forward(data.image_place)
          --print('word_feature',word_feature:size())--[torch.DoubleTensor of size 200x1x512]


          -- forward the ConvNet on images (most work happens here)
          local img_feat_clone3=img_cnn_feat_clone3:forward({data.images,data.images,data.images})
          --print('img_feat',img_feat:size())--200x512


          --joint1 on Image embedding and caption features
          local joint_feat1 = protos.joint1:forward({img_feat_clone3[1],ques_feat})


          --joint1 on Image embedding and tag features
          local joint_feat2 = protos.joint2:forward({img_feat_clone3[2],tag_feat})


          --joint1 on Image embedding and place features
          local joint_feat3 = protos.joint3:forward({img_feat_clone3[3],place_feat})
----------------------------------------------------------------------------------------------
          --joint1 on Image embedding and Question features
          local gating_feat = protos.gating_net:forward(data.images)


        -- this is peform sum of o1*g1+o2*g2
          local moe_feat= protos.moe:forward({joint_feat1,joint_feat2,joint_feat3,gating_feat})

          --print('moe_feat',moe_feat:size())
-------------------------------------------------------------------------------------------------------
          -- forward the language model
          local logprobs = protos.lm:forward({moe_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats


          -- forward the language model criterion
          local loss = protos.crit:forward(logprobs, decode_question)

        -------------------------------------------------------------------------------------------------------------------

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        -- forward the model to also get generated samples for each image
        local seq = protos.lm:sample(moe_feat)
        local sents = net_utils.decode_sequence(vocab, seq)
        for k=1,#sents do
                local entry = {image_id = data.ques_id[k], question = sents[k]} -- change here
                -- print('questions to be written to the val_predictions', sents[k])
                table.insert(predictions, entry) -- to save all the alements
                -------------------------------------------------------------------------
                -- for print log
                if verbose then
                        print(string.format('image %s: %s', entry.image_id, entry.question))
                end
                ------------------------------------------------------------------------
        end
        -- print('length of sents ', #sents) -------checking
        if n >= total_num then break end -- this is for complete val example , it should not be more than val total sample. otherwise , repetation example will save in json which will cause error in blue score evalution
        if n >= opt.val_images_use then break end -- we've used enough images

  end
        ------------------------------------------------------------------------
        -- for blue,cider score
        local lang_stats
        if opt.language_eval == 1 then
                lang_stats = net_utils.language_eval(predictions, opt.id)
                local score_statistics = {epoch = epoch, statistics = lang_stats}
                print('Current language statistics',score_statistics)
        end
         ------------------------------------------------------------------------
         -- write a (thin) json report-- for save image id and question print in json format
        local question_filename = string.format('%s/question_checkpoint_epoch%d', opt.checkpoint_dir, epoch)
        utils.write_json(question_filename .. '.json', predictions) -- for save image id and question print in json format
        print('wrote json checkpoint to ' .. question_filename .. '.json')

------------------------------------------------------------------------
  return loss_sum/loss_evals, predictions, lang_stats

end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
        protos.emb:training()
        protos.cnn:training()
        protos.joint1:training()
        protos.joint2:training()
        protos.joint3:training()
        protos.gating_net:training()
        protos.moe:training()
        protos.lm:training()
        protos.place:training()
        protos.tag:training()
        protos.tag_net:training()

----------------------------------------------------------------------------
-- Forward pass
-----------------------------------------------------------------------------
-- get batch of data
	--local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
	local batch = dataloader:next_batch(opt)
        local data = {}
        data.images=batch[1]
        data.questions=batch[2]
        data.caption=batch[3]
        data.ques_id  = batch[4]
        data.image_place=batch[5]
        data.nountag=batch[6]
        data.verbtag=batch[7]
        data.questiontag=batch[8]

        -------------------------------------------------------------------------------------------------------------------
        local decode_question= data.questions:t()-- bcz in langauage models checks assert(seq:size(1) == self.seq_length) os it should be 26 x 200
        --print('after transpose data.questions',data.questions:size()) --26x200
        -- bcz this language model needs dimension of size 26x200

        --print('data.caption',data.caption:size())--[torch.DoubleTensor of size 200x1x512]

         -------------------------------------------------------------------------------------------------------------------
        local question_tag1= data.questiontag:select(2,1)
        local question_tag2= data.questiontag:select(2,2)
        local question_tag3= data.questiontag:select(2,3)
        --print("question_tag",question_tag:size())
        -------------------------------------------------------------------------------------------------------------------

        --Forward the question word feature through word embedding
        local question_tag_feat_clone3 =question_tag_emb_net_clone3:forward({question_tag1,question_tag2,question_tag3});
        --print('word_feature',word_feature:size())--[torch.DoubleTensor of size 200x1x512]

        local tag_feat=protos.tag_net:forward({question_tag_feat_clone3[1],question_tag_feat_clone3[2],question_tag_feat_clone3[3]});
        --local noun_tag_feat=nn.CAddTable(2):forward({{noun_tag_feat_clone3[1],noun_tag_feat_clone3[2]}});
        --local noun_tag_feat=nn.JoinTable(2):forward({{noun_tag_feat_clone3[1],noun_tag_feat_clone3[2]}});


        -------------------------------------------------------------------------------------------------------------------
          --Forward the question word feature through word embedding
          local ques_feat =protos.emb:forward(data.caption)
          --print('ques_feat',ques_feat:size())--[torch.DoubleTensor of size 200x1x512]
          --print('ques_feat',ques_feat:max(),ques_feat:min())

          --Forward place  feature through word embedding
          local place_feat =protos.place:forward(data.image_place)
          --print('word_feature',word_feature:size())--[torch.DoubleTensor of size 200x1x512]


          -- forward the ConvNet on images (most work happens here)
          local img_feat_clone3=img_cnn_feat_clone3:forward({data.images,data.images,data.images})
          --print('img_feat',img_feat:size())--200x512


          --joint1 on Image embedding and caption features
          local joint_feat1 = protos.joint1:forward({img_feat_clone3[1],ques_feat})


          --joint1 on Image embedding and tag features
          local joint_feat2 = protos.joint2:forward({img_feat_clone3[2],tag_feat})


          --joint1 on Image embedding and place features
          local joint_feat3 = protos.joint3:forward({img_feat_clone3[3],place_feat})
----------------------------------------------------------------------------------------------
          --joint1 on Image embedding and Question features
          local gating_feat = protos.gating_net:forward(data.images)


        -- this is peform sum of o1*g1+o2*g2
          local moe_feat= protos.moe:forward({joint_feat1,joint_feat2,joint_feat3,gating_feat})

          --print('moe_feat',moe_feat:size())
-------------------------------------------------------------------------------------------------------
          -- forward the language model
          local logprobs = protos.lm:forward({moe_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats


          -- forward the language model criterion
          local loss = protos.crit:forward(logprobs, decode_question)

        -------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Backward pass
-----------------------------------------------------------------------------
        grad_eparams:zero()
        grad_cparams:zero()
        grad_jparams1:zero()
        grad_jparams2:zero()
        grad_jparams3:zero()
        grad_lparams:zero()
        grad_pparams:zero()
        grad_mparams1:zero()
        grad_mparams2:zero()
        grad_tparams:zero()
        grad_tnparams:zero()

        -- backprop criterion
        local dlogprobs = protos.crit:backward(logprobs, decode_question)

        -- backprop language model
        local d_lm_feats, ddummy = unpack(protos.lm:backward({moe_feat, decode_question}, dlogprobs))
----------------------------------------------------------------------------------------------
        --print('d_lm_feats',d_lm_feats:size())

        -- this is peform sum of o1*g1+o2*g2
          local d_joint_feat1,d_joint_feat2,d_joint_feat3,d_gating_feat = unpack(protos.moe:backward({joint_feat1,joint_feat2,joint_feat3,gating_feat},d_lm_feats))
--           print('d_gating_feat',d_gating_feat[1]:size())
--           print('d_joint_feat1',d_joint_feat1:size())
--           print('d_joint_feat2',d_joint_feat2:size())
--           if opt.gpuid >= 0 then
--           d_gating_feat     = d_gating_feat:cuda()
--           end
-- print('d_gating_feat',d_gating_feat:size())
          --joint1 on Image embedding and Question features
          local gating_feat = protos.gating_net:backward(data.images,d_gating_feat)

-------------------------------------------------------------------------------------------------------

        -- backprop joint1 multi model: image and caption
        local d_img_feat_clone3_1, d_ques_feat = unpack(protos.joint1:backward({img_feat_clone3[1],ques_feat}, d_joint_feat1))

        -- backprop joint1 multi model: image and tag
        local d_img_feat_clone3_2, d_tag_feat = unpack(protos.joint2:backward({img_feat_clone3[2],tag_feat}, d_joint_feat2))

         -- backprop joint1 multi model: image and place
        local d_img_feat_clone3_3, d_place_feat = unpack(protos.joint3:backward({img_feat_clone3[3],place_feat}, d_joint_feat3))

-------------------------------------------------------------------------------------------------------

        -- backprop the CNN, but only if we are finetuning
        local dummy_img_feats = img_cnn_feat_clone3:backward({data.images,data.images,data.images},{d_img_feat_clone3_1,d_img_feat_clone3_2,d_img_feat_clone3_3})

        -- backprop question embedding model
        local dummy_ques_feat= protos.emb:backward(data.caption, d_ques_feat)

        --backward place feature
        local dummy_img_place_feat =protos.place:backward(data.image_place,d_place_feat)


        -- backprop tag embedding model
        local d_question_tag_emb_feat=protos.tag_net:backward({question_tag_feat_clone3[1],question_tag_feat_clone3[2],question_tag_feat_clone3[3]},d_tag_feat);

        local dummy_question_tag1_feat,dummy_question_tag2_feat,dummy_question_tag3_feat= unpack(question_tag_emb_net_clone3:backward({question_tag1,question_tag2,question_tag3}, {d_question_tag_emb_feat[1],d_question_tag_emb_feat[2],d_question_tag_emb_feat[3]}))




  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
--Step 13:--Log Function
-------------------------------------------------------------------------------
function printlog(epoch,ErrTrain,ErrTest)
 	------------------------------------------------------------------------------
	-- log plot
	paths.mkdir(opt.save)
	err_log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
	err_log:style{['Training Error'] = '-', ['Test Error'] = '-'}
	-- err_log:plot()
	---------------------------------------------------------------------------------
	if paths.filep(opt.save..'/ErrorProgress.eps') or paths.filep(opt.save..'/accuracyProgress.eps') then
		-----------------------------------------------------------------------------------------------------------
		-- convert .eps file as .png file
		local base64im
		do
			os.execute(('convert -density 200 %s/ErrorProgress.eps %s/ErrorProgress.png'):format(opt.save,opt.save))
			os.execute(('openssl base64 -in %s/ErrorProgress.png -out %s/ErrorProgress.base64'):format(opt.save,opt.save))
			local f = io.open(opt.save..'/ErrorProgress.base64')
			if f then base64im = f:read'*all' end
		end

		-----------------------------------------------------------------------------------------------------------------------
		-- to display in .html file
		local file = io.open(opt.save..'/report.html','w')
		file:write('<h5>Training data size:  '..total_train_example ..'\n')
		file:write('<h5>Validation data size:  '..total_num ..'\n')
		file:write('<h5>batchSize:  '..opt.batch_size..'\n')
		file:write('<h5>LR:  '..opt.learning_rate..'\n')
		file:write('<h5>optimization:  '..opt.optim..'\n')
		file:write('<h5>drop_prob_lm:  '..opt.drop_prob_lm..'\n')


		file:write(([[
		<!DOCTYPE html>
		<html>
		<body>
		<title>%s - %s</title>
		<img src="data:image/png;base64,%s">
		<h4>optimState:</h4>
		<table>
		]]):format(opt.save,epoch,base64im))

	--[[	for k,v in pairs(optim_state) do
			if torch.type(v) == 'number' then
			 	file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
			end
		end --]]

		file:write'</table><pre>\n'
		file:write'</pre></body></html>'
		file:close()
	end
--[[
	if opt.visualize then
		require 'image'
		local weights = EmbeddingNet:get(1).weight:clone()
		--win = image.display(weights,5,nil,nil,nil,win)
		image.saveJPG(paths.concat(opt.save,'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
	end
--]]
	return 1
end



local function lossFun_for_val_input()
        protos.emb:training()
        protos.cnn:training()
        protos.joint1:training()
        protos.joint2:training()
        protos.joint3:training()
        protos.gating_net:training()
        protos.moe:training()
        protos.lm:training()
        protos.place:training()
        protos.tag:training()
        protos.tag_net:training()


        dataloader:resetIterator(2)
        local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

----------------------------------------------------------------------------
-- Forward pass
-----------------------------------------------------------------------------
-- get batch of data
	--local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
	local batch = dataloader:next_batch_eval(opt)
        local data = {}
        data.images=batch[1]
        data.questions=batch[2]
        data.caption=batch[4]
        data.ques_id  = batch[3]
        data.image_place=batch[5]
        data.nountag=batch[6]
        data.verbtag=batch[7]
        data.questiontag=batch[8]

        -------------------------------------------------------------------------------------------------------------------
        local decode_question= data.questions:t()-- bcz in langauage models checks assert(seq:size(1) == self.seq_length) os it should be 26 x 200
        --print('after transpose data.questions',data.questions:size()) --26x200
        -- bcz this language model needs dimension of size 26x200

        --print('data.caption',data.caption:size())--[torch.DoubleTensor of size 200x1x512]

         -------------------------------------------------------------------------------------------------------------------
        local question_tag1= data.questiontag:select(2,1)
        local question_tag2= data.questiontag:select(2,2)
        local question_tag3= data.questiontag:select(2,3)
        --print("question_tag",question_tag:size())
        -------------------------------------------------------------------------------------------------------------------

        --Forward the question word feature through word embedding
        local question_tag_feat_clone3 =question_tag_emb_net_clone3:forward({question_tag1,question_tag2,question_tag3});
        --print('word_feature',word_feature:size())--[torch.DoubleTensor of size 200x1x512]

        local tag_feat=protos.tag_net:forward({question_tag_feat_clone3[1],question_tag_feat_clone3[2],question_tag_feat_clone3[3]});
        --local noun_tag_feat=nn.CAddTable(2):forward({{noun_tag_feat_clone3[1],noun_tag_feat_clone3[2]}});
        --local noun_tag_feat=nn.JoinTable(2):forward({{noun_tag_feat_clone3[1],noun_tag_feat_clone3[2]}});


        -------------------------------------------------------------------------------------------------------------------
          --Forward the question word feature through word embedding
          local ques_feat =protos.emb:forward(data.caption)
          --print('ques_feat',ques_feat:size())--[torch.DoubleTensor of size 200x1x512]
          --print('ques_feat',ques_feat:max(),ques_feat:min())

          --Forward place  feature through word embedding
          local place_feat =protos.place:forward(data.image_place)
          --print('word_feature',word_feature:size())--[torch.DoubleTensor of size 200x1x512]


          -- forward the ConvNet on images (most work happens here)
          local img_feat_clone3=img_cnn_feat_clone3:forward({data.images,data.images,data.images})
          --print('img_feat',img_feat:size())--200x512


          --joint1 on Image embedding and caption features
          local joint_feat1 = protos.joint1:forward({img_feat_clone3[1],ques_feat})


          --joint1 on Image embedding and tag features
          local joint_feat2 = protos.joint2:forward({img_feat_clone3[2],tag_feat})


          --joint1 on Image embedding and place features
          local joint_feat3 = protos.joint3:forward({img_feat_clone3[3],place_feat})
----------------------------------------------------------------------------------------------
          --joint1 on Image embedding and Question features
          local gating_feat = protos.gating_net:forward(data.images)


        -- this is peform sum of o1*g1+o2*g2
          local moe_feat= protos.moe:forward({joint_feat1,joint_feat2,joint_feat3,gating_feat})

          --print('moe_feat',moe_feat:size())
-------------------------------------------------------------------------------------------------------
          -- forward the language model
          local logprobs = protos.lm:forward({moe_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats


          -- forward the language model criterion
          local loss = protos.crit:forward(logprobs, decode_question)

        -------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------
-- Backward pass
-----------------------------------------------------------------------------
        grad_eparams:zero()
        grad_cparams:zero()
        grad_jparams1:zero()
        grad_jparams2:zero()
        grad_jparams3:zero()
        grad_lparams:zero()
        grad_pparams:zero()
        grad_mparams1:zero()
        grad_mparams2:zero()
        grad_tparams:zero()
        grad_tnparams:zero()

        -- backprop criterion
        local dlogprobs = protos.crit:backward(logprobs, decode_question)

        -- backprop language model
        local d_lm_feats, ddummy = unpack(protos.lm:backward({moe_feat, decode_question}, dlogprobs))
----------------------------------------------------------------------------------------------
        --print('d_lm_feats',d_lm_feats:size())

        -- this is peform sum of o1*g1+o2*g2
          local d_joint_feat1,d_joint_feat2,d_joint_feat3,d_gating_feat = unpack(protos.moe:backward({joint_feat1,joint_feat2,joint_feat3,gating_feat},d_lm_feats))
--           print('d_gating_feat',d_gating_feat[1]:size())
--           print('d_joint_feat1',d_joint_feat1:size())
--           print('d_joint_feat2',d_joint_feat2:size())
--           if opt.gpuid >= 0 then
--           d_gating_feat     = d_gating_feat:cuda()
--           end
-- print('d_gating_feat',d_gating_feat:size())
          --joint1 on Image embedding and Question features
          local gating_feat = protos.gating_net:backward(data.images,d_gating_feat)

-------------------------------------------------------------------------------------------------------

        -- backprop joint1 multi model: image and caption
        local d_img_feat_clone3_1, d_ques_feat = unpack(protos.joint1:backward({img_feat_clone3[1],ques_feat}, d_joint_feat1))

        -- backprop joint1 multi model: image and tag
        local d_img_feat_clone3_2, d_tag_feat = unpack(protos.joint2:backward({img_feat_clone3[2],tag_feat}, d_joint_feat2))

         -- backprop joint1 multi model: image and place
        local d_img_feat_clone3_3, d_place_feat = unpack(protos.joint3:backward({img_feat_clone3[3],place_feat}, d_joint_feat3))

-------------------------------------------------------------------------------------------------------

        -- backprop the CNN, but only if we are finetuning
        local dummy_img_feats = img_cnn_feat_clone3:backward({data.images,data.images,data.images},{d_img_feat_clone3_1,d_img_feat_clone3_2,d_img_feat_clone3_3})

        -- backprop question embedding model
        local dummy_ques_feat= protos.emb:backward(data.caption, d_ques_feat)

        --backward place feature
        local dummy_img_place_feat =protos.place:backward(data.image_place,d_place_feat)


        -- backprop tag embedding model
        local d_question_tag_emb_feat=protos.tag_net:backward({question_tag_feat_clone3[1],question_tag_feat_clone3[2],question_tag_feat_clone3[3]},d_tag_feat);

        local dummy_question_tag1_feat,dummy_question_tag2_feat,dummy_question_tag3_feat= unpack(question_tag_emb_net_clone3:backward({question_tag1,question_tag2,question_tag3}, {d_question_tag_emb_feat[1],d_question_tag_emb_feat[2],d_question_tag_emb_feat[3]}))




  -----------------------------------------------------------------------------
  -- and lets get out!
  local losses = { total_loss = loss }
  return data.ques_id, dummy_img_feats
end




-------------------------------------------------------------------------------
--Step 12:--Training Function
-------------------------------------------------------------------------------
local e_optim_state = {}  --- to mentain state in optim
local c_optim_state = {}  --- to mentain state in optim
local j_optim_state1 = {}  --- to mentain state in optim
local j_optim_state2 = {}  --- to mentain state in optim
local j_optim_state3 = {}  --- to mentain state in optim
local l_optim_state = {}  --- to mentain state in optim
local p_optim_state = {}  --- to mentain state in optim
local m_optim_state1 = {}  --- to mentain state in optim
local m_optim_state2 = {}  --- to mentain state in optim
local t_optim_state = {}  --- to mentain state in optim
local tn_optim_state = {}  --- to mentain state in optim

local grad_clip = 0.1
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch) -- for lr decay
local learning_rate = opt.learning_rate
-- local decay_factor =0.5

total_train_example = dataloader:getDataNum(1) -- for lr decay
train_nbatch=math.ceil(total_train_example /opt.batch_size)


function Train()
	count_sum=0  -- Cannt be make local bcz it is insisde the function and other function are using this.
	local iter=1
	local ave_loss = 0  --for iter_log_print  train error
	err=0

	while iter <= train_nbatch do
		-- Training loss/gradient
		local losses = lossFun()
		err=err+ losses.total_loss
		ave_loss = ave_loss + losses.total_loss
		---------------------------------------------------------

		-- decay the learning rate
		if epoch % opt.learning_rate_decay_every ==0 then
                   learning_rate = learning_rate * decay_factor -- set the decayed rate
		end
    if epoch % 15 == 0 and iter < 10 then
                   learning_rate = learning_rate * 0.99999 *decay_factor -- set the decayed rate
    end
		---------------------------------------------------------
		if iter % opt.losses_log_every == 0 then
			ave_loss = ave_loss / opt.losses_log_every
			print(string.format('epoch:%d  iter %d: %f, %f, %f', epoch, iter, ave_loss,learning_rate, timer:time().real))
			ave_loss = 0
		end

		---------------------------------------------------------
		-- perform a parameter update
		if opt.optim == 'sgd' then
                        sgdm(eparams, grad_eparams, learning_rate, opt.momentum, e_optim_state)
                        sgdm(cparams, grad_cparams, learning_rate, opt.momentum, c_optim_state)
                        sgdm(jparams1, grad_jparams1, learning_rate, opt.momentum, j_optim_state1)
                        sgdm(lparams, grad_lparams, learning_rate, opt.momentum, l_optim_state)
                        sgdm(pparams, grad_pparams, learning_rate, opt.momentum, p_optim_state)

		elseif opt.optim == 'rmsprop' then
                        rmsprop(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
                        rmsprop(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
                        rmsprop(jparams1, grad_jparams1, learning_rate, opt.optim_alpha, opt.optim_epsilon, j_optim_state1)
                        rmsprop(jparams2, grad_jparams2, learning_rate, opt.optim_alpha, opt.optim_epsilon, j_optim_state2)
                        rmsprop(jparams3, grad_jparams3, learning_rate, opt.optim_alpha, opt.optim_epsilon, j_optim_state3)
                        rmsprop(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
                        rmsprop(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, p_optim_state)
                        rmsprop(mparams1, grad_mparams1, learning_rate, opt.optim_alpha, opt.optim_epsilon, m_optim_state1)
                        rmsprop(mparams2, grad_mparams2, learning_rate, opt.optim_alpha, opt.optim_epsilon, m_optim_state2)
                        rmsprop(tparams, grad_tparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, t_optim_state)
                        rmsprop(tnparams, grad_tnparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, tn_optim_state)
    elseif opt.optim == 'adam' then
      adam(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, e_optim_state)
      adam(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, c_optim_state)
      adam(jparams1, grad_jparams1, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, j_optim_state1)
    	adam(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, l_optim_state)
      adam(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, p_optim_state)
    elseif opt.optim == 'sgdm' then
      sgdm(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      sgdm(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      sgdm(jparams1, grad_jparams1, learning_rate, opt.optim_alpha, opt.optim_epsilon, j_optim_state1)
     sgdm(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
     sgdm(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, p_optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      sgdmom(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      sgdmom(jparams1, grad_jparams1, learning_rate, opt.optim_alpha, opt.optim_epsilon, j_optim_state1)
       sgdmom(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
       sgdmom(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, p_optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      adagrad(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      adagrad(jparams1, grad_jparams1, learning_rate, opt.optim_alpha, opt.optim_epsilon, j_optim_state1)
       adagrad(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
       adagrad(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, p_optim_state)
		else
			error('bad option opt.optim')
		end
		---------------------------------------------------------
		iter = iter + 1
                if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
                if loss0 == nil then loss0 = losses.total_loss end
                if losses.total_loss > loss0 * 20 then
                        print('loss seems to be exploding, quitting.')
                        break
                end

	end
	return err/train_nbatch
end
-------------------------------------------------------------------------------
--Step 13:--Main  Function
------------------------------------------------------------------------------
local best_score_Bleu_1
local best_score_Bleu_2
local best_score_Bleu_3
local best_score_Bleu_4
local best_score_ROUGE_L
local best_score_METEOR
local best_score_CIDEr


local best_score_max_Bleu_1
local best_score_max_Bleu_2
local best_score_max_Bleu_3
local best_score_max_Bleu_4
local best_score_max_ROUGE_L
local best_score_max_METEOR
local best_score_max_CIDEr


local best_score_min_Bleu_1
local best_score_min_Bleu_2
local best_score_min_Bleu_3
local best_score_min_Bleu_4
local best_score_min_ROUGE_L
local best_score_min_METEOR
local best_score_min_CIDEr


local best_Bleu_1_SEED_max
local best_Bleu_2_SEED_max
local best_Bleu_3_SEED_max
local best_Bleu_4_SEED_max
local best_ROUGE_L_SEED_max
local best_METEOR_SEED_max
local best_CIDEr_SEED_max



local best_Bleu_1_SEED_min
local best_Bleu_2_SEED_min
local best_Bleu_3_SEED_min
local best_Bleu_4_SEED_min
local best_ROUGE_L_SEED_min
local best_METEOR_SEED_min
local best_CIDEr_SEED_min

--local best_score_SPICE
-------------------------------------------------------------------------------
--Step 14:-- Main loop
-------------------------------------------------------------------------------
epoch = 1  -- made gloobal ,bcz inside training function, it is used
print '\n==> Starting Training\n'
-- print '\n==> Starting Training\n'
local ques_cap_id, img_fv_grad = lossFun_for_val_input()
img_fv_grad_sum = img_fv_grad[1]+img_fv_grad[2]+img_fv_grad[3]
print('gradients size', img_fv_grad_sum:size())
local gradients_file = hdf5.open('gradients_moe3_pct_bd_s1234_q_cap_id_epoch_36.h5', 'w')
c= nn.utils.recursiveType(img_fv_grad_sum, 'torch.DoubleTensor')
gradients_file:write('maps', c)
-- ques_cap_id= nn.utils.recursiveType(ques_cap_id, 'torch.DoubleTensor')
gradients_file:write('ques_cap_id', ques_cap_id)
gradients_file:close()

while epoch ~= 0 do

  print('Epoch ' .. epoch)
  --local val_loss, val_predictions, lang_stats = eval_split(2)
  local ErrTrain = Train()

     Seed_avg_Bleu_1=0
     Seed_avg_Bleu_2=0
     Seed_avg_Bleu_3=0
     Seed_avg_Bleu_4=0
     Seed_avg_ROUGE_L=0
     Seed_avg_METEOR=0
     Seed_avg_CIDEr=0

     Seed_max_Bleu_1=0
     Seed_max_Bleu_2=0
     Seed_max_Bleu_3=0
     Seed_max_Bleu_4=0
     Seed_max_ROUGE_L=0
     Seed_max_METEOR=0
     Seed_max_CIDEr=0

     Seed_min_Bleu_1=999999
     Seed_min_Bleu_2=999999
     Seed_min_Bleu_3=999999
     Seed_min_Bleu_4=999999
     Seed_min_ROUGE_L=999999
     Seed_min_METEOR=999999
     Seed_min_CIDEr=999999
     local maxBlue1_Seed
     local maxBlue2_Seed
     local maxBlue3_Seed
     local maxBlue4_Seed
     local maxROUGE_L_Seed
     local maxMETEOR_Seed
     local maxCIDEr_Seed


     local minBlue1_Seed
     local minBlue2_Seed
     local minBlue3_Seed
     local minBlue4_Seed
     local minROUGE_L_Seed
     local minMETEOR_Seed
     local minCIDEr_Seed

     local val_avg_loss=0

     local max_number_of_seed=10

     f:write("Epoch_Number ","\t" ,epoch,"\n")
      for seedIndex=1, max_number_of_seed do
                    cutorch.manualSeed(seedIndex)
                    torch.manualSeed(seedIndex)
                          print('Checkpointing. Calculating validation accuracy..')
                          local val_loss, val_predictions, lang_stats = eval_split(2)
                    print('********************************************************************')
                    print('val loss with seed ',val_loss,seedIndex)
                    print('********************************************************************')

                    if lang_stats then
                      -- use CIDEr score for deciding how well we did
                      Seed_avg_Bleu_1 = Seed_avg_Bleu_1+lang_stats['Bleu_1']
                      Seed_avg_Bleu_2 = Seed_avg_Bleu_2+lang_stats['Bleu_2']
                      Seed_avg_Bleu_3 = Seed_avg_Bleu_3+lang_stats['Bleu_3']
                      Seed_avg_Bleu_4 = Seed_avg_Bleu_4+lang_stats['Bleu_4']
                      Seed_avg_ROUGE_L = Seed_avg_ROUGE_L+lang_stats['ROUGE_L']
                      Seed_avg_METEOR = Seed_avg_METEOR+lang_stats['METEOR']
                      Seed_avg_CIDEr = Seed_avg_CIDEr+lang_stats['CIDEr']


                          f:write("seed","\t" ,seedIndex, "\t", "epoch", "\t", epoch ,"\n")
                          f:write("Bleu_1","\t" ,lang_stats['Bleu_1'],"\n")
                          f:write("Bleu_2","\t" ,lang_stats['Bleu_2'],"\n")
                          f:write("Bleu_3","\t" ,lang_stats['Bleu_3'],"\n")
                          f:write("Bleu_4","\t" ,lang_stats['Bleu_4'],"\n")
                          f:write("ROUGE_L","\t" ,lang_stats['ROUGE_L'],"\n")
                          f:write("METEOR","\t" ,lang_stats['METEOR'],"\n")
                          f:write("CIDEr","\t" ,lang_stats['CIDEr'],"\n")



                    if  lang_stats['Bleu_1'] > Seed_max_Bleu_1 then
                            Seed_max_Bleu_1 = lang_stats['Bleu_1']
                            maxBlue1_Seed=seedIndex
                    end
                    if  lang_stats['Bleu_2'] > Seed_max_Bleu_2 then
                            Seed_max_Bleu_2 = lang_stats['Bleu_2']
                            maxBlue2_Seed=seedIndex
                    end
                    if  lang_stats['Bleu_3'] > Seed_max_Bleu_3 then
                           Seed_max_Bleu_3 = lang_stats['Bleu_3']
                           maxBlue3_Seed=seedIndex
                    end
                    if  lang_stats['Bleu_4'] > Seed_max_Bleu_4 then
                            Seed_max_Bleu_4 = lang_stats['Bleu_4']
                            maxBlue4_Seed=seedIndex
                    end
                    if  lang_stats['ROUGE_L'] > Seed_max_ROUGE_L then
                            Seed_max_ROUGE_L =lang_stats['ROUGE_L']
                            maxROUGE_L_Seed=seedIndex
                    end
                    if  lang_stats['METEOR'] > Seed_max_METEOR then
                            Seed_max_METEOR = lang_stats['METEOR']
                            maxMETEOR_Seed=seedIndex
                    end
                    if  lang_stats['CIDEr'] > Seed_max_CIDEr then
                            Seed_max_CIDEr = lang_stats['CIDEr']
                            maxCIDEr_Seed=seedIndex
                    end


                    if  lang_stats['Bleu_1'] < Seed_min_Bleu_1 then
                            Seed_min_Bleu_1 = lang_stats['Bleu_1']
                            minBlue1_Seed=seedIndex
                    end
                    if  lang_stats['Bleu_2'] < Seed_min_Bleu_2 then
                            Seed_min_Bleu_2 = lang_stats['Bleu_2']
                            minBlue2_Seed=seedIndex
                    end
                    if  lang_stats['Bleu_3'] < Seed_min_Bleu_3 then
                           Seed_min_Bleu_3 = lang_stats['Bleu_3']
                           minBlue3_Seed=seedIndex
                    end
                    if  lang_stats['Bleu_4'] < Seed_min_Bleu_4 then
                            Seed_min_Bleu_4 = lang_stats['Bleu_4']
                            minBlue4_Seed=seedIndex
                    end
                    if  lang_stats['ROUGE_L'] < Seed_min_ROUGE_L then
                            Seed_min_ROUGE_L =lang_stats['ROUGE_L']
                            minROUGE_L_Seed=seedIndex
                    end
                    if  lang_stats['METEOR'] < Seed_min_METEOR then
                            Seed_min_METEOR = lang_stats['METEOR']
                            minMETEOR_Seed=seedIndex
                    end
                    if  lang_stats['CIDEr'] > Seed_min_CIDEr then
                            Seed_min_CIDEr = lang_stats['CIDEr']
                            minCIDEr_Seed=seedIndex
                    end




              else
                  -- use the (negative) validation loss as a score
                    Seed_avg_Bleu_1 = -val_loss
                    Seed_avg_Bleu_2 = -val_loss
                    Seed_avg_Bleu_3 = -val_loss
                    Seed_avg_Bleu_4 =-val_loss
                    Seed_avg_ROUGE_L = -val_loss
                    Seed_avg_METEOR = -val_loss
                    Seed_avg_CIDEr = -val_loss
                  --current_score_SPICE = -val_loss
                  end

                  val_avg_loss=val_avg_loss+val_loss
                  -----------------------------------------------------------
      end

      val_avg_loss=val_avg_loss/max_number_of_seed
    -- for print best score
    -- write the full model checkpoint as well if we did better than ever
          local current_score_Bleu_1 =Seed_avg_Bleu_1/max_number_of_seed
          local current_score_Bleu_2 =Seed_avg_Bleu_2/max_number_of_seed
          local current_score_Bleu_3 =Seed_avg_Bleu_3/max_number_of_seed
          local current_score_Bleu_4=Seed_avg_Bleu_4/max_number_of_seed
          local current_score_ROUGE_L=Seed_avg_ROUGE_L/max_number_of_seed
          local current_score_METEOR=Seed_avg_METEOR/max_number_of_seed
          local current_score_CIDEr=Seed_avg_CIDEr/max_number_of_seed
          --local current_score_SPICE





        if best_score_Bleu_1 == nil or current_score_Bleu_1 > best_score_Bleu_1 then
                best_score_Bleu_1 = current_score_Bleu_1
                best_score_max_Bleu_1 = Seed_max_Bleu_1
                best_score_min_Bleu_1 = Seed_min_Bleu_1
                best_Bleu_1_SEED_max =maxBlue1_Seed
                best_Bleu_1_SEED_min=minBlue1_Seed

        end

        if best_score_Bleu_2 == nil or current_score_Bleu_2 > best_score_Bleu_2 then
                best_score_Bleu_2 = current_score_Bleu_2
                best_score_max_Bleu_2 = Seed_max_Bleu_2
                best_score_min_Bleu_2 = Seed_min_Bleu_2
                best_Bleu_2_SEED_max =maxBlue2_Seed
                best_Bleu_2_SEED_min =minBlue2_Seed
        end

        if best_score_Bleu_3 == nil or current_score_Bleu_3 > best_score_Bleu_3 then
                best_score_Bleu_3 = current_score_Bleu_3
                best_score_max_Bleu_3 = Seed_max_Bleu_3
                best_score_min_Bleu_3 = Seed_min_Bleu_3
                best_Bleu_3_SEED_max =maxBlue3_Seed
                best_Bleu_3_SEED_min =minBlue3_Seed
        end

        if best_score_Bleu_4 == nil or current_score_Bleu_4 > best_score_Bleu_4 then
                best_score_Bleu_4 = current_score_Bleu_4
                best_score_max_Bleu_4 = Seed_max_Bleu_4
                best_score_min_Bleu_4 = Seed_min_Bleu_4
                best_Bleu_4_SEED_max =maxBlue4_Seed
                best_Bleu_4_SEED_min =minBlue4_Seed
        end

        if best_score_ROUGE_L == nil or current_score_ROUGE_L > best_score_ROUGE_L then
                best_score_ROUGE_L = current_score_ROUGE_L
                best_score_max_ROUGE_L = Seed_max_ROUGE_L
                est_score_min_ROUGE_L = Seed_min_ROUGE_L
                best_ROUGE_L_SEED_max =maxROUGE_L_Seed
                best_ROUGE_L_SEED_min =minROUGE_L_Seed
        end

        if best_score_METEOR == nil or current_score_METEOR > best_score_METEOR then
                best_score_METEOR = current_score_METEOR
                best_score_max_METEOR = Seed_max_METEOR
                best_score_min_METEOR = Seed_min_METEOR
                best_METEOR_SEED_max =maxMETEOR_Seed
                best_METEOR_SEED_min =minMETEOR_Seed
        end

        if best_score_CIDEr == nil or current_score_CIDEr > best_score_CIDEr then
                best_score_CIDEr = current_score_CIDEr
                best_score_max_CIDEr = Seed_max_CIDEr
                best_score_min_CIDEr = Seed_min_CIDEr
                best_CIDEr_SEED_max =maxCIDEr_Seed
                best_CIDEr_SEED_min =minCIDEr_Seed
        end

         --if best_score_SPICE == nil or current_score_SPICE > best_score_SPICE then
         --       best_score_SPICE = current_score_SPICE
       -- end

        print('-----------------------------------------------------------------------------------------')
         print('current_Bleu_1:', current_score_Bleu_1,'current_Bleu_2:', current_score_Bleu_2,'current_Bleu_3:', current_score_Bleu_3,'current_Bleu_4:', current_score_Bleu_4)
         print('current_ROUGE_L:', current_score_ROUGE_L, 'current_METEOR:',current_score_METEOR, 'And current_CIDEr:',current_score_CIDEr)
        print('-----------------------------------------------------------------------------------------')
         print('best_Bleu_1:', best_score_Bleu_1,'best_Bleu_2:', best_score_Bleu_2,'best_Bleu_3:', best_score_Bleu_3,'best_Bleu_4:', best_score_Bleu_4)
         print('best_ROUGE_L:', best_score_ROUGE_L, 'best_METEOR:',best_score_METEOR, 'And best_CIDEr:',best_score_CIDEr)

         print('best_score_max_Bleu_1:', best_score_max_Bleu_1,'best_score_max_Bleu_2:', best_score_max_Bleu_2,'best_score_max_Bleu_3:', best_score_max_Bleu_3,'best_score_max_Bleu_4:', best_score_max_Bleu_4)
         print('best_score_max_ROUGE_L:', best_score_max_ROUGE_L, 'best_score_max_METEOR:',best_score_max_METEOR, 'And best_score_max_CIDEr:',best_score_max_CIDEr)


         print('best_score_min_Bleu_1:', best_score_min_Bleu_1,'best_score_min_Bleu_2:', best_score_min_Bleu_2,'best_score_min_Bleu_3:', best_score_min_Bleu_3,'best_Bleu_4_SEED_max:', best_Bleu_4_SEED_max)
         print('best_score_min_ROUGE_L:', best_score_min_ROUGE_L, 'best_score_min_METEOR:',best_score_min_METEOR, 'And best_score_min_CIDEr:',best_score_min_CIDEr)

         print('best_Bleu_1_SEED_max:', best_Bleu_1_SEED_max,'best_Bleu_2_SEED_max:', best_Bleu_2_SEED_max,'best_Bleu_3_SEED_max:', best_Bleu_3_SEED_max,'best_Bleu_4_SEED_max:', best_Bleu_4_SEED_max)
         print('best_ROUGE_L_SEED_max:', best_ROUGE_L_SEED_max, 'best_METEOR_SEED_max:',best_METEOR_SEED_max, 'And best_CIDEr_SEED_max:',best_CIDEr_SEED_max)

          print('best_Bleu_1_SEED_min:', best_Bleu_1_SEED_min,'best_Bleu_2_SEED_min:', best_Bleu_2_SEED_min,'best_Bleu_3_SEED_min:', best_Bleu_3_SEED_min,'best_Bleu_4_SEED_min:', best_Bleu_4_SEED_min)
         print('best_ROUGE_L_SEED_min:', best_ROUGE_L_SEED_min, 'best_METEOR_SEED_min:',best_METEOR_SEED_min, 'And best_CIDEr_SEED_min:',best_CIDEr_SEED_min)



         print('-----------------------------------------------------------------------------------------')
         --print('Current language statistics',lang_stats)
          ----------------------------------------------------------------------------------------
        -- for print log
        lang_stats_log:add{['Bleu_1']= current_score_Bleu_1, ['Bleu_2'] = current_score_Bleu_2,['Bleu_3'] = current_score_Bleu_3,['Bleu_4'] = current_score_Bleu_4,['ROUGE_L'] = current_score_ROUGE_L,['METEOR'] = current_score_METEOR,['CIDEr'] = current_score_CIDEr}

        lang_stats_log:style{['Bleu_1']= '-', ['Bleu_2'] = '-',['Bleu_3'] = '-',['Bleu_4'] = '-',['ROUGE_L'] = '-',['METEOR'] = '-',['CIDEr'] = '-'}

        -- lang_stats_log:plot()
  -----------------------------------------------------------------------------------
                print('------------------------------------------------------------------------')
  print('Training Error:  ', ErrTrain ,'Validation Average loss: ', val_avg_loss)


  local result=printlog(epoch,ErrTrain,val_avg_loss)
        ---------------------------------------------------------------------------------------------------------------------------------------------
        local model_save_filename = string.format('%s/model_epoch%d.t7', opt.checkpoint_dir, epoch)
        --if epoch % 100==0 then --dont save on very first iteration
                torch.save(model_save_filename, {eparams=eparams, cparams=cparams,jparams1=jparams1,jparams2=jparams2,jparams3=jparams3,lparams=lparams,pparams=pparams,mparams1=mparams1,mparams2=mparams2,tparams=tparams, tnparams=tnparams, lmOpt=lmOpt})  -- vocabulary mapping is included here, so we can use the checkpoint
        --end
  print('Saving current checkpoint to:', model_save_filename)

  epoch = epoch+1
end

