from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import itertools
from ops import *
from utils import *
import pandas as pd
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
Some notes:
Folder names within train and val folders have to be single capital alphabets.
"""
class DualNet(object):
    def __init__(self, sess, image_size=256, batch_size=1,fcn_filter_dim = 64,  \
                 channels = 1, dataset_name='facades', \
                 checkpoint_dir=None, lambdax = 20.0, \
                 sample_dir=None, loss_metric = 'L1', flip = False):

        self.data_dict = {}  #To hold information about N datasets
        self.loss_dict = {}
        
        self.dataset_name = dataset_name
        self.trainFolderNames = sorted(list(os.walk("./datasets/{}/train".format(self.dataset_name)))[0][1])
        self.loss_metric = loss_metric
        


        for i,folderName in enumerate(self.trainFolderNames):
            self.data_dict[folderName] = {}
            self.data_dict[folderName]["lambda"] = lambdax
            self.data_dict[folderName]["channels"] = channels
            self.data_dict[folderName]["is_grayscale"] = (channels == 1) 
            
            ##This is to store the losses
            self.loss_dict[folderName] = {}
            self.loss_dict[folderName]["d_fake_loss"] = [100]  #Initialize to some arbitrary value
            self.loss_dict[folderName]["d_real_loss"] = [100]
            self.loss_dict[folderName]["g_loss"] = [100]

        self.allFolderCombos = [term[0] + term[1] for term in list(itertools.combinations(self.trainFolderNames, 2))]
        self.df_dim = fcn_filter_dim
        self.flip = flip      

        self.sess = sess
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.fcn_filter_dim = fcn_filter_dim
        self.logdir = "%s-img_sz_%s-fltr_dim_%d-%s" % (
                            self.dataset_name, 
                            self.image_size,
                            self.fcn_filter_dim,
                            self.loss_metric
                            )


        self.d_loss = None
        self.g_loss = None
        self.d_vars = None
        self.g_vars = None
        self.checkpoint_dir = checkpoint_dir
        self.combo_dict = {}
        for combo in self.allFolderCombos:
            self.combo_dict[combo] = {}
            #directory name for output and logs saving
            self.combo_dict[combo]["dir_name"] = str(self.dataset_name) + "-img_sz_" +\
                                                str(self.image_size) + "-fltr_dim_" +\
                                                str(self.fcn_filter_dim) +"-"+str(self.loss_metric) +\
                                                 "-lambda_" + combo + "_" + str(self.data_dict[combo[0]]["lambda"])+"_" + \
                                                 str(self.data_dict[combo[1]]["lambda"])
                                                
        self.build_model()

    def build_model(self):
    ###    define place holders
        for folderName in self.trainFolderNames:
            self.data_dict[folderName]["real_images"] = tf.placeholder(tf.float32,
                                                        [self.batch_size, self.image_size, self.image_size,
                                                         self.data_dict[folderName]["channels"]],
                                                         name="real_" + folderName)
        
        
    ###  define graphs
        count = 0
        models_done = set() #Holds the models that have already been built
        for combo in self.allFolderCombos:

            first = combo[0]
            second = combo[1]
            print("Trying to build the model for", first,second)
            print(models_done)
            print(count)
            count += 1
            lambda_A = self.data_dict[first]["lambda"]
            lambda_B = self.data_dict[second]["lambda"]
            #Save individual generator models
            if not "g_net" in self.data_dict[first].keys(): 
                print("Generator for ", first, " does not exist. Creating.")
                self.data_dict[first]["g_net"] = self.g_net(self.data_dict[first]["real_images"],
                                                            name = first, reuse = False)
            if not "g_net" in self.data_dict[second].keys(): 
                print("Generator for ", second, " does not exist. Creating.")
                self.data_dict[second]["g_net"] = self.g_net(self.data_dict[second]["real_images"], 
                                                            name = second, reuse = False)
            A2B = self.data_dict[first]["g_net"]
            B2A = self.data_dict[second]["g_net"]
            #Update the combo dictionary to store the combined images/nets
            
            tempname1 = first + '2' + second + '2' + first
            tempname2 = second + '2' + first + '2' + second
            
            print("Updating the combo dict for ", combo)
            self.combo_dict[combo][tempname1] = self.g_net(A2B,name = first, reuse = True)
            self.combo_dict[combo][tempname2] = self.g_net(B2A,name = second, reuse = True)
            A2B2A = self.combo_dict[combo][tempname1]
            B2A2B = self.combo_dict[combo][tempname2]
            real_A = self.data_dict[first]["real_images"]
            real_B = self.data_dict[second]["real_images"]
            #self.A2B2A = self.B_g_net(self.A2B, reuse = True)
            #self.B2A2B = self.A_g_net(self.B2A, reuse = True)
            if not "loss" in self.data_dict[first].keys():
                print("Defining loss for ", first)
                if self.loss_metric == 'L1':
                    self.data_dict[first]["loss"] = tf.reduce_mean(tf.abs(A2B2A - real_A))
                elif self.loss_metric == 'L2':
                    self.data_dict[first]["loss"] = tf.reduce_mean(tf.square(A2B2A - real_A))
            if not "loss" in self.data_dict[second].keys():
                print("Defining loss for ", second)
                if self.loss_metric == "L1":     
                    self.data_dict[second]["loss"] = tf.reduce_mean(tf.abs(B2A2B - real_B))
                elif self.loss_metric == 'L2':
                    self.data_dict[second]["loss"] = tf.reduce_mean(tf.square(B2A2B - real_B))
            A_loss = self.data_dict[first]["loss"]
            B_loss = self.data_dict[second]["loss"] 
            
            if not "d_logits_fake" in self.data_dict[first].keys():
                print("Model for ", first ,  " does not exist. Creating...")
                self.data_dict[first]["d_logits_fake"] = self.d_net(A2B, name = first, reuse = False)
                self.data_dict[first]["d_logits_real"] = self.d_net(real_B, name = first, reuse = True)
                Ad_logits_fake = self.data_dict[first]["d_logits_fake"]
                Ad_logits_real = self.data_dict[first]["d_logits_real"]
                self.data_dict[first]["d_loss_fake"] = celoss(Ad_logits_fake, tf.zeros_like(Ad_logits_fake))
                self.data_dict[first]["d_loss_real"] = celoss(Ad_logits_real, tf.zeros_like(Ad_logits_real))
                Ad_loss_fake = self.data_dict[first]["d_loss_fake"]
                Ad_loss_real = self.data_dict[first]["d_loss_real"]
                self.data_dict[first]["d_loss"] = Ad_loss_fake + Ad_loss_real
                self.data_dict[first]["g_loss"] = celoss(Ad_logits_fake, labels=tf.ones_like(Ad_logits_fake))+lambda_B * (B_loss)
                Ad_loss = self.data_dict[first]["d_loss"]
                Ag_loss = self.data_dict[first]["g_loss"]
            #self.Ad_logits_fake = self.A_d_net(self.A2B, reuse = False)
            #self.Ad_logits_real = self.A_d_net(self.real_B, reuse = True)
            #self.Ad_loss_real = celoss(self.Ad_logits_real, tf.ones_like(self.Ad_logits_real))
            #self.Ad_loss_fake = celoss(self.Ad_logits_fake, tf.zeros_like(self.Ad_logits_fake))
            #self.Ad_loss = self.Ad_loss_fake + self.Ad_loss_real
            #self.Ag_loss = celoss(self.Ad_logits_fake, labels=tf.ones_like(self.Ad_logits_fake))+self.lambda_B * (self.B_loss )
            if not "d_logits_fake" in self.data_dict[second].keys():
                print("Model for ", second ,  " does not exist. Creating...")
                self.data_dict[second]["d_logits_fake"] = self.d_net(B2A, name = second, reuse = False)
                self.data_dict[second]["d_logits_real"] = self.d_net(real_A, name = second, reuse = True)
                Bd_logits_fake = self.data_dict[second]["d_logits_fake"]
                Bd_logits_real = self.data_dict[second]["d_logits_real"]    
                self.data_dict[second]["d_loss_fake"] = celoss(Bd_logits_fake, tf.zeros_like(Bd_logits_fake))
                self.data_dict[second]["d_loss_real"] = celoss(Bd_logits_real, tf.zeros_like(Bd_logits_real))
                Bd_loss_fake = self.data_dict[second]["d_loss_fake"]
                Bd_loss_real = self.data_dict[second]["d_loss_real"]
                self.data_dict[second]["d_loss"] = Bd_loss_fake + Bd_loss_real
                self.data_dict[second]["g_loss"] = celoss(Bd_logits_fake, labels=tf.ones_like(Bd_logits_fake))+lambda_A * (A_loss)
                Bd_loss = self.data_dict[second]["d_loss"]
                Bg_loss = self.data_dict[second]["g_loss"]
            
            
            #self.Bd_logits_fake = self.B_d_net(self.B2A, reuse = False)
            #self.Bd_logits_real = self.B_d_net(self.real_A, reuse = True)
            #self.Bd_loss_real = celoss(self.Bd_logits_real, tf.ones_like(self.Bd_logits_real))
            #self.Bd_loss_fake = celoss(self.Bd_logits_fake, tf.zeros_like(self.Bd_logits_fake))
            #self.Bd_loss = self.Bd_loss_fake + self.Bd_loss_real
            #self.Bg_loss = celoss(self.Bd_logits_fake, tf.ones_like(self.Bd_logits_fake))+self.lambda_A * (self.A_loss)
            
            #These are the overall discriminator and generator losses
            #We need to add them only once to the overall loss
            if self.d_loss is None:
                print("Initializing overall d_loss.")
                self.d_loss = Ad_loss + Bd_loss
            else:
                if first not in models_done:
                    print("Adding d_loss term of ", first, " to the overall loss.")
                    self.d_loss += Ad_loss
                if second not in models_done:
                    print("Adding d_loss term of ", second, " to the overall loss.")
                    self.d_loss += Bd_loss

            if self.g_loss is None:
                print("Initializing overall g_vars_loss.")
                self.g_loss = Ag_loss + Bg_loss
            else:
                if first not in models_done:
                    self.g_loss += Ag_loss
                if second not in models_done:
                    self.g_loss += Bg_loss
            #self.d_loss = self.Ad_loss + self.Bd_loss
            #self.g_loss = self.Ag_loss + self.Bg_loss

            ## define trainable variables
            t_vars = tf.trainable_variables()
            self.data_dict[first]["d_vars"] = [var for var in t_vars if first + '_d_' in var.name]
            A_d_vars = self.data_dict[first]["d_vars"]
            #self.A_d_vars = [var for var in t_vars if 'A_d_' in var.name]
            self.data_dict[second]["d_vars"]     = [var for var in t_vars if second + '_d_' in var.name]
            B_d_vars = self.data_dict[second]["d_vars"]
            #self.B_d_vars = [var for var in t_vars if 'B_d_' in var.name]
            self.data_dict[first]["g_vars"] = [var for var in t_vars if first + '_g_' in var.name]
            A_g_vars = self.data_dict[first]["g_vars"]
            #self.A_g_vars = [var for var in t_vars if 'A_g_' in var.name]
            self.data_dict[second]["g_vars"] = [var for var in t_vars if second + '_g_' in var.name]
            B_g_vars = self.data_dict[second]["g_vars"]
            #self.B_g_vars = [var for var in t_vars if 'B_g_' in var.name]
            if self.d_vars is None:
                self.d_vars = A_d_vars + B_d_vars 
            else:
                if first not in models_done:
                    self.d_vars += A_d_vars
                if second not in models_done:
                    self.d_vars += B_d_vars

            if self.g_vars is None:
                self.g_vars = A_g_vars + B_g_vars 
            else:
                if first not in models_done:
                    self.g_vars += A_g_vars
                if second not in models_done:
                    self.g_vars += B_g_vars
            #self.g_vars = self.A_g_vars + self.B_g_vars
            self.saver = tf.train.Saver()
            models_done.add(first)
            models_done.add(second)


    def clip_trainable_vars(self, var_list):
        for var in var_list:
            self.sess.run(var.assign(tf.clip_by_value(var, -self.c, self.c)))

    def load_random_samples(self,first):

        sample_files =np.random.choice(glob('./datasets/' + self.dataset_name + '/val/' + 
                                        first + '/*.jpg'),self.batch_size)
        sample_A_imgs = [load_data(f, image_size =self.image_size, flip = False) for f in sample_files]
        
       

        sample_A_imgs = np.reshape(np.array(sample_A_imgs).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        
        return sample_A_imgs

    #Runs the learned model on a sample set of images to save results.
    def sample_shotcut(self, first, second, sample_dir, fnames, epoch_idx, batch_idx):
        imgBatchDict = {}
        for fname in fnames:
            imgBatchDict[fname] = self.load_random_samples(fname)
        
        tempname1 = first + '2' + second + '2' + first
        tempname2 = second + '2' + first + '2' + second
        combo = ''.join(sorted(first + second))
        A_loss = self.data_dict[first]["loss"] 
        A2B2A = self.combo_dict[combo][tempname1] 
        A2B = self.data_dict[first]["g_net"]
        B_loss = self.data_dict[second]["loss"] 
        B2A2B = self.combo_dict[combo][tempname2]
        B2A = self.data_dict[second]["g_net"]
        combo = ''.join(sorted(first + second))
        Ag, A2B2A_imgs, A2B_imgs = self.sess.run([self.data_dict[first]["loss"], self.combo_dict[combo][tempname1] , self.data_dict[first]["g_net"]], 
                                   feed_dict = {self.data_dict[fname]["real_images"]: imgBatchDict[fname] for fname in fnames})
        Bg, B2A2B_imgs, B2A_imgs = self.sess.run([self.data_dict[second]["loss"], self.combo_dict[combo][tempname2], self.data_dict[second]["g_net"]], 
                                   feed_dict = {self.data_dict[fname]["real_images"]: imgBatchDict[fname] for fname in fnames})
        save_images(A2B_imgs, [self.batch_size,1], './' + sample_dir +'/' + self.combo_dict[combo]["dir_name"] +'/' + 
                                str(epoch_idx) +'_' + str(batch_idx) +'_' + first +'2' + second +'.jpg')
        save_images(A2B2A_imgs, [self.batch_size,1], './' + sample_dir +'/' + self.combo_dict[combo]["dir_name"] +'/' + 
                                str(epoch_idx) +'_' + str(batch_idx) +'_' + tempname1 + '.jpg')
        save_images(B2A_imgs, [self.batch_size,1], './' + sample_dir +'/' + self.combo_dict[combo]["dir_name"] +'/' + 
                                str(epoch_idx) +'_' + str(batch_idx) +'_' + second +'2' + first +'.jpg')
        save_images(B2A2B_imgs, [self.batch_size,1], './' + sample_dir +'/' + self.combo_dict[combo]["dir_name"] +'/' + 
                                str(epoch_idx) +'_' + str(batch_idx) +'_' + tempname2 + '.jpg')
        
        print("[Sample] A_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))
    """
    This is the main training function. All other methods get called from here.
    """
    def train(self, args):
        """Train Dual GAN"""
        decay = 0.9
        ##Change to ADAM?
        self.d_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        ##Change to ADAM?
        self.g_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay) \
                          .minimize(self.g_loss, var_list=self.g_vars)          
        tf.global_variables_initializer().run()

        ##Logger
        self.writer = tf.summary.FileWriter("./logs/"+self.logdir, self.sess.graph)

        step = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" Load failed...ignored...")
            print(" start training...")

        for epoch_idx in xrange(args.epoch):
            ##Loading the datasets here. Can generalize to N datasets
            #Get the folder names from the train folder
            
            
            #Load each dataset and place it into a dictionary
            epoch_size = float("inf") #To find the epoch size which is the min of all the lengths of the datasets
            for i, folderName in enumerate(self.trainFolderNames):
                data = glob('./datasets/{}/train/{}/*.jpg'.format(self.dataset_name,folderName))
                np.random.shuffle(data)
                self.data_dict[folderName]["imgNames"] = data
                if len(data) < epoch_size:
                    epoch_size = len(data)
            epoch_size = epoch_size // (self.batch_size)
            print('[*] training data loaded successfully')
            print("Epoch size: " + str(epoch_size))
            #Print the lengths of each dataset
            print("There are %d datasets."%(len(self.data_dict.keys())))
            _ = [print("#" + folderName + str(len(self.data_dict[folderName]["imgNames"]))) 
                for folderName in sorted(self.data_dict.keys())]
            print("Loaded batches")
            print('[*] run optimizer...')

            for batch_idx in xrange(0, epoch_size):
                ##Load all the datasets
                imgBatchDict = {}
                #Load a batch of each dataset
                imgBatchDict = {folderName:self.load_training_imgs(self.data_dict[folderName]["imgNames"], batch_idx)
                                for folderName in self.trainFolderNames}
                for fname in self.trainFolderNames:
                    print(imgBatchDict[fname].shape)
                print("Epoch: [%2d] [%4d/%4d]"%(epoch_idx, batch_idx, epoch_size))
                step = step + 1

                #Run the optimizer for all the pairs of datasets, this is nC2 number of pairs. Ordering matters.
                
                
                for i, combo in enumerate(self.allFolderCombos):
                    #run the optimizer for all dissimilar folder pairs
                    print(combo)
                    if combo[0] != combo[1]:
                        self.run_optim(combo[0], combo[1], imgBatchDict, step, start_time,epoch_idx,batch_idx)
                    print("Ran optimizer")

                    
                
                if np.mod(step, 5) == 0:
                    self.sample_shotcut(combo[0], combo[1],args.sample_dir,imgBatchDict.keys(),epoch_idx, batch_idx)

                if np.mod(step, args.save_freq) == 0:
                    self.save(args.checkpoint_dir, step)

        #Now to store the losses into a text file.

        #Once all the generators and discriminators get updated, save the loss.
        #Add tf.Summary() later
        



    #Does not need to be modified
    def load_training_imgs(self, files, idx):
        batch_files = files[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_imgs = [load_data(f, image_size =self.image_size, flip = self.flip) for f in batch_files]
        batch_imgs = np.reshape(np.array(batch_imgs).astype(np.float32),(self.batch_size,self.image_size, self.image_size,-1))
        
        return batch_imgs
        
    def run_optim(self,first, second, imgBatchDict, counter, start_time,epoch_idx,batch_idx):

        #Run the discriminator once
        _, Adfake,Adreal,Bdfake,Bdreal, Ad, Bd = self.sess.run(
            [self.d_optim, self.data_dict[first]["d_loss_fake"], self.data_dict[first]["d_loss_real"], self.data_dict[second]["d_loss_fake"], 
             self.data_dict[second]["d_loss_real"],  self.data_dict[first]["d_loss"],  self.data_dict[second]["d_loss"]], 
            feed_dict = {self.data_dict[fname]["real_images"]: imgBatchDict[fname] for fname in imgBatchDict.keys()})


        ##Run the generator twice
        _, Ag, Bg, Aloss, Bloss = self.sess.run(
            [self.g_optim, self.data_dict[first]["g_loss"], self.data_dict[second]["g_loss"], 
            self.data_dict[first]["loss"], self.data_dict[second]["loss"]], 
            feed_dict = {self.data_dict[fname]["real_images"]: imgBatchDict[fname] for fname in imgBatchDict.keys()})

        _, Ag, Bg, Aloss, Bloss = self.sess.run(
            [self.g_optim, self.data_dict[first]["g_loss"], self.data_dict[second]["g_loss"], 
            self.data_dict[first]["loss"], self.data_dict[second]["loss"]], 
            feed_dict = {self.data_dict[fname]["real_images"]: imgBatchDict[fname] for fname in imgBatchDict.keys()})
        end_time = time.time()
        print("time: " + str(end_time - start_time) + ", " + first + "d: " + str(Ad) + ", " + first + "g: " + str(Ag) + ", " + 
            second + "d: " + str(Bd) + ", " + second + "g: " + str(Bg) + ",  U_diff: " + str(Aloss) + ", V_diff: " + str(Bloss))
        print(first+"d_fake: " + str(Adfake) + ", " + first + "d_real: " + str(Adreal) +", " + 
            second + "d_fake: " + str(Bdfake) + ", " + second + "g_real: " + str(Bdreal))
        #Save losses for plotting.
        self.loss_dict[first]["d_fake_loss"].append(Adfake) 
        self.loss_dict[first]["d_real_loss"].append(Adreal)
        self.loss_dict[first]["g_loss"].append(Ag)
        
        self.loss_dict[second]["d_fake_loss"].append(Bdfake)
        self.loss_dict[second]["d_real_loss"].append(Bdreal)
        self.loss_dict[second]["g_loss"].append(Bg)

        #Once all the generators and discriminators get updated, save the loss.
        #Add tf.Summary() later
        datasets = list(self.data_dict.keys())
        with open(os.path.join("./logs/",self.logdir,"logfile.txt"),"a") as f:
            for d in datasets:
                
                log_string =  "{0}_d_fake_loss:{1},{0}_d_real_loss:{2},{0}_g_loss:{3},epoch_id:{4},batch_id:{5},time:{6},diff_loss:{7}".format(first,
                    self.loss_dict[d]["d_fake_loss"][-1],
                    self.loss_dict[d]["d_real_loss"][-1],
                    self.loss_dict[d]["g_loss"][-1],
                    epoch_idx,batch_idx,end_time-start_time,Aloss)
                log_string2 =  "{0}_d_fake_loss:{1},{0}_d_real_loss:{2},{0}_g_loss:{3},epoch_id:{4},batch_id:{5},time:{6},diff_loss:{7}".format(second,
                    self.loss_dict[d]["d_fake_loss"][-1],
                    self.loss_dict[d]["d_real_loss"][-1],
                    self.loss_dict[d]["g_loss"][-1],
                    epoch_idx,batch_idx,end_time-start_time,Bloss)
                
                f.write(log_string)
                f.write("\n")
                f.write(log_string2)
                f.write("\n")

    def d_net(self, imgs, name ,y = None, reuse = False):
        return self.discriminator(imgs, prefix = name + '_d_', reuse = reuse)
    
    def discriminator(self, image,  prefix , y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name=prefix+'h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name=prefix+'h1_conv'), name = prefix+'bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name=prefix+'h2_conv'), name = prefix+ 'bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name=prefix+'h3_conv'), name = prefix+ 'bn3'))
            # h3 is (32 x 32 x self.df_dim*8)
            h4 = conv2d(h3, 1, d_h=1, d_w=1, name =prefix+'h4')
            return h4
    def g_net(self,imgs, name, reuse = False):    
        return self.fcn(imgs, name=name , reuse = reuse)
        
    def fcn(self, imgs, name = None , reuse = False):
        prefix = name + "_g_"
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False
            
            s = self.image_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # imgs is (256 x 256 x input_c_dim)
            e1 = conv2d(imgs, self.fcn_filter_dim, name=prefix+'e1_conv')
            # e1 is (128 x 128 x self.fcn_filter_dim)
            e2 = batch_norm(conv2d(lrelu(e1), self.fcn_filter_dim*2, name=prefix+'e2_conv'), name = prefix+'bn_e2')
            # e2 is (64 x 64 x self.fcn_filter_dim*2)
            e3 = batch_norm(conv2d(lrelu(e2), self.fcn_filter_dim*4, name=prefix+'e3_conv'), name = prefix+'bn_e3')
            # e3 is (32 x 32 x self.fcn_filter_dim*4)
            e4 = batch_norm(conv2d(lrelu(e3), self.fcn_filter_dim*8, name=prefix+'e4_conv'), name = prefix+'bn_e4')
            # e4 is (16 x 16 x self.fcn_filter_dim*8)
            e5 = batch_norm(conv2d(lrelu(e4), self.fcn_filter_dim*8, name=prefix+'e5_conv'), name = prefix+'bn_e5')
            # e5 is (8 x 8 x self.fcn_filter_dim*8)
            e6 = batch_norm(conv2d(lrelu(e5), self.fcn_filter_dim*8, name=prefix+'e6_conv'), name = prefix+'bn_e6')
            # e6 is (4 x 4 x self.fcn_filter_dim*8)
            e7 = batch_norm(conv2d(lrelu(e6), self.fcn_filter_dim*8, name=prefix+'e7_conv'), name = prefix+'bn_e7')
            # e7 is (2 x 2 x self.fcn_filter_dim*8)
            e8 = batch_norm(conv2d(lrelu(e7), self.fcn_filter_dim*8, name=prefix+'e8_conv'), name = prefix+'bn_e8')
            # e8 is (1 x 1 x self.fcn_filter_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.fcn_filter_dim*8], name=prefix+'d1', with_w=True)
            d1 = tf.nn.dropout(batch_norm(self.d1, name = prefix+'bn_d1'), 0.5)
            d1 = tf.concat([d1, e7],3)
            # d1 is (2 x 2 x self.fcn_filter_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.fcn_filter_dim*8], name=prefix+'d2', with_w=True)
            d2 = tf.nn.dropout(batch_norm(self.d2, name = prefix+'bn_d2'), 0.5)

            d2 = tf.concat([d2, e6],3)
            # d2 is (4 x 4 x self.fcn_filter_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.fcn_filter_dim*8], name=prefix+'d3', with_w=True)
            d3 = tf.nn.dropout(batch_norm(self.d3, name = prefix+'bn_d3'), 0.5)

            d3 = tf.concat([d3, e5],3)
            # d3 is (8 x 8 x self.fcn_filter_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.fcn_filter_dim*8], name=prefix+'d4', with_w=True)
            d4 = batch_norm(self.d4, name = prefix+'bn_d4')

            d4 = tf.concat([d4, e4],3)
            # d4 is (16 x 16 x self.fcn_filter_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.fcn_filter_dim*4], name=prefix+'d5', with_w=True)
            d5 = batch_norm(self.d5, name = prefix+'bn_d5')
            d5 = tf.concat([d5, e3],3)
            # d5 is (32 x 32 x self.fcn_filter_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.fcn_filter_dim*2], name=prefix+'d6', with_w=True)
            d6 = batch_norm(self.d6, name = prefix+'bn_d6')
            d6 = tf.concat([d6, e2],3)
            # d6 is (64 x 64 x self.fcn_filter_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.fcn_filter_dim], name=prefix+'d7', with_w=True)
            d7 = batch_norm(self.d7, name = prefix+'bn_d7')
            d7 = tf.concat([d7, e1],3)
            # d7 is (128 x 128 x self.fcn_filter_dim*1*2)

            
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),[self.batch_size, s, s, self.data_dict[name]["channels"]], name=prefix+'d8', with_w=True)
            
             # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(self.d8)
    
    def save(self, checkpoint_dir, step):
        model_name = "DualNet.model"
        model_dir = self.logdir
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print("Saving checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir =  self.logdir
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test DualNet"""
        start_time = time.time()
        tf.global_variables_initializer().run()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
            test_dir = './{}/{}'.format(args.test_dir, self.logdir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            test_log = open(test_dir+'evaluation.txt','a') 
            test_log.write(self.logdir)
            self.test_domain(args, test_log, first = 'A', second = 'B')
            self.test_domain(args, test_log, first = 'B',second = 'A')
            self.test_domain(args, test_log, first = 'C',second = 'A')
            self.test_domain(args, test_log, first = 'A',second = 'C')
            self.test_domain(args, test_log, first = 'B',second = 'C')
            self.test_domain(args, test_log, first = 'C',second = 'B')
            test_log.close()
        
    def test_domain(self, args, test_log, first,second):
        test_files = glob('./datasets/{}/val/{}/*.jpg'.format(self.dataset_name,first))
        # load testing input
        print("Loading testing images ...")
        test_imgs = [load_data(f, is_test=True, image_size =self.image_size, flip = args.flip) for f in test_files]
        print("#images loaded: %d"%(len(test_imgs)))
        test_imgs = np.reshape(np.asarray(test_imgs).astype(np.float32),(len(test_files),self.image_size, self.image_size,-1))
        test_imgs = [test_imgs[i*self.batch_size:(i+1)*self.batch_size]
                         for i in xrange(0, len(test_imgs)//self.batch_size)]
        test_imgs = np.asarray(test_imgs)
        test_path = './{}/{}/'.format(args.test_dir, self.logdir)
        # test input samples

        for i in xrange(0, len(test_files)//self.batch_size):
            filename_o = test_files[i*self.batch_size].split('/')[-1].split('.')[0]
            print(filename_o)
            idx = i+1
            A_imgs = np.reshape(np.array(test_imgs[i]), (self.batch_size,self.image_size, self.image_size,-1))
            print("testing ",first," image %d"%(idx))
            print(A_imgs.shape)
            combo = ''.join(sorted(first + second))
            A2B_imgs, A2B2A_imgs = self.sess.run(
                [self.data_dict[first]["g_net"], self.combo_dict[combo][first + '2' + second + '2' + first]],
                feed_dict={self.data_dict[first]["real_images"]: A_imgs}
                )
            save_images(A_imgs, [self.batch_size, 1], test_path+filename_o+'_real' + first + '.jpg')
            save_images(A2B_imgs, [self.batch_size, 1], test_path+filename_o+'_' + first + '2' + second + '.jpg')
            save_images(A2B2A_imgs, [self.batch_size, 1], test_path+filename_o+'_' + first + '2' + second + '2' + first + '.jpg')
        