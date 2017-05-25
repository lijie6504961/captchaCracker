from PIL import Image
from PIL import ImageFilter  
import numpy as np  
import tensorflow as tf  
import os
import random
import sys
import platform

sysstr = platform.system()
if(sysstr =="Windows"):
    print ("Call Windows tasks")
elif(sysstr == "Linux"):
    print ("Call Linux tasks")
LOC_TEST='/test_yzm/'
LOC_YZM='/yzm/'
LOC_PROCESSED='/processed/'
LOC_MODEL='./model/'    
number = ['0','1','2','3','4','5','6','7','8']#,'9']  
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']  
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  


def get_one_data(loc=LOC_YZM):
    dir=os.getcwd()+loc
    listimg=os.listdir(dir)
    lucky_data=random.sample(listimg,1)[0]
    image=Image.open(dir+lucky_data)
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            r,g,b =image.getpixel((j,i))
            if (r-(g+b))<100:
                r=255
                b=255
                g=255
            image.putpixel((j,i), (r,g,b))
    box=(20,20,300,100)
    image=image.crop(box)
    image=image.filter(ImageFilter.GaussianBlur(radius=2))
    image=image.resize((140, 40), Image.ANTIALIAS)
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            r,g,b =image.getpixel((j,i))
            if g>150 or b>150:
                r=0
                b=0
                g=0
            else:   
                r=255
                b=255
                g=255
            image.putpixel((j,i), (r,g,b))               
    image=np.array(image)
    text=lucky_data.split('.')[0]
    return text,image

    
def get_one_processed_data(loc=LOC_PROCESSED):    
    dir=os.getcwd()+loc
    listimg=os.listdir(dir)
    lucky_data=random.sample(listimg,1)[0]
    image=Image.open(dir+lucky_data)
    image=np.array(image)
    text=lucky_data.split('.')[0]
    if len(text)>5:
        print (dir+lucky_data)
    return text,image
    
text,image=get_one_processed_data()
print("验证码图像channel:",text,image.shape)  
# 图像大小  
IMAGE_HEIGHT = 40  
IMAGE_WIDTH = 140  
MAX_CAPTCHA = 5
char_set = number + alphabet  
CHAR_SET_LEN = len(char_set)
n_classes=MAX_CAPTCHA*CHAR_SET_LEN
n_input = IMAGE_HEIGHT*IMAGE_WIDTH
   
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）  
def convert2gray(img):  
    if len(img.shape) > 2:  
        gray = np.mean(img, -1)  
        # 上面的转法较快，正规转法如下  
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]  
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  
        return gray  
    else:  
        return img  
   
""" 
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。 
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行 
"""  
   
# 文本转向量   
def text2vec(text):  
    text_len = len(text)  
    if text_len > MAX_CAPTCHA:  
        raise ValueError('验证码最长4个字符',text)  
   
    vector = np.zeros(n_classes)  
    def char2pos(c):  
        k = ord(c)-48  
        if k > 8:  
            k = ord(c) - 87   
        return k  
    for i, c in enumerate(text):  
        idx = i * CHAR_SET_LEN + char2pos(c)  
        vector[idx] = 1  
    return vector  
# 向量转回文本  
def vec2text(vec):  
    char_pos = vec.nonzero()[0]  
    text=[]  
    for i, c in enumerate(char_pos):  
        char_at_pos = i #c/63  
        char_idx = c % CHAR_SET_LEN  
        if char_idx < 9:  
            char_code = char_idx + ord('0')  
        elif char_idx <35:  
            char_code = char_idx - 9 + ord('a')  
        else:  
            raise ValueError('error')  
        text.append(chr(char_code))  
    return "".join(text)  
   
""" 
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有 
vec = text2vec("F5Sd") 
text = vec2text(vec) 
print(text)  # F5Sd 
vec = text2vec("SFd5") 
text = vec2text(vec) 
print(text)  # SFd5 
"""  
   
# 生成一个训练batch  
def get_next_batch(batch_size=128,type='train'):  
    batch_x = np.zeros([batch_size, n_input])  
    batch_y = np.zeros([batch_size, n_classes])  
 
    if type=='train':
        for i in range(batch_size):  
            text, image = get_one_processed_data()  
            image = convert2gray(image)  
       
            batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0  
            batch_y[i,:] = text2vec(text)
    else:
        for i in range(batch_size):  
            text, image = get_one_data(loc=LOC_TEST)  
            image = convert2gray(image)  
       
            batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0  
            batch_y[i,:] = text2vec(text)        
   
    return batch_x, batch_y  
   
####################################################################  
   
X = tf.placeholder(tf.float32, [None, n_input])  
Y = tf.placeholder(tf.float32, [None, n_classes])  
keep_prob = tf.placeholder(tf.float32) # dropout  
   
# 定义CNN  
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):  
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  
   
    #w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #  
    #w_c2_alpha = np.sqrt(2.0/(3*3*32))   
    #w_c3_alpha = np.sqrt(2.0/(3*3*64))   
    #w_d1_alpha = np.sqrt(2.0/(8*32*64))  
    #out_alpha = np.sqrt(2.0/1024)  
   
    # 3 conv layer  
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))  
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    # w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 64]))  
    # b_c1 = tf.Variable(b_alpha*tf.random_normal([64]))      
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))  
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv1 = tf.nn.dropout(conv1, keep_prob)  
   
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))  
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    # w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 128]))  
    # b_c2 = tf.Variable(b_alpha*tf.random_normal([128]))     
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))  
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv2 = tf.nn.dropout(conv2, keep_prob)  
   
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))  
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    # w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 128, 256]))  
    # b_c3 = tf.Variable(b_alpha*tf.random_normal([256]))      
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))  
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv3 = tf.nn.dropout(conv3, keep_prob)  
   
    # Fully connected layer  (33)8*32*40 (55)20*8*64
    w_d = tf.Variable(w_alpha*tf.random_normal([18*5*64, 1024]))  
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))  
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])  
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))  
    dense = tf.nn.dropout(dense, keep_prob)
    w_d2=tf.Variable(w_alpha*tf.random_normal([1024, 1024]))
    b_d2 = tf.Variable(b_alpha*tf.random_normal([1024]))  
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense, w_d2), b_d2)) 
   
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, n_classes]))  
    b_out = tf.Variable(b_alpha*tf.random_normal([n_classes]))  
    out = tf.add(tf.matmul(dense2, w_out), b_out)  
    #out = tf.nn.softmax(out)  
    return out  
   
# 训练  
def train_crack_captcha_cnn():  
    output = crack_captcha_cnn()  
    # loss  
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))  
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y))  
        # 最后一层用来分类的softmax和sigmoid有什么不同？  
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰  
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  
   
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    max_idx_p = tf.argmax(predict, 2)  
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
        goal_acc=0.7
        step = 0  
        while step<10000:  
            batch_x, batch_y = get_next_batch(128)  
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})  
            print(step, loss_)  
              
            # 每100 step计算一次准确率  
            if step % 100 == 0:  
                batch_x_test, batch_y_test = get_next_batch(100,type='validation')  
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})  
                print(step, acc,goal_acc)  
                # 如果准确率大于50%,保存模型,完成训练  
                if acc > goal_acc:  
                    saver.save(sess, LOC_MODEL+"crack_capcha.model", global_step=step)  
                    goal_acc=acc
                    if goal_acc>0.8:
                        break
            if step==9999:
                saver.save(sess, LOC_MODEL+"crack_capcha.model", global_step=step)  
                break                
   
            step += 1  

            
def entropy_loss(logits, labels):
    cross_entropy_per_number = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy = tf.reduce_mean(cross_entropy_per_number)
    tf.add_to_collection("loss", cross_entropy)
    return cross_entropy            
            
           
#restore
def restore_train_crack_captcha_cnn(loc,now_steps,max_steps):  
    output = crack_captcha_cnn()  
    # loss  
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y)) 
    
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output, Y))  
        # 最后一层用来分类的softmax和sigmoid有什么不同？  
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰  
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)  
   
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    max_idx_p = tf.argmax(predict, 2)  
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        saver.restore(sess, loc) 
        goal_acc=0.8
        step = now_steps+1 
        while step<max_steps:  
            batch_x, batch_y = get_next_batch(300)  
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})  
            print(step, loss_)  
              
            # 每100 step计算一次准确率  
            if step % 100 == 0:  
                batch_x_test, batch_y_test = get_next_batch(100,type='validation')  
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})  
                print(step, acc,goal_acc)  
                # 如果准确率大于50%,保存模型,完成训练  
                if acc > goal_acc:  
                    saver.save(sess, LOC_MODEL+"crack_capcha.model", global_step=step)  
                    goal_acc=acc
                    if goal_acc>0.9:
                        break
            if step==max_steps-1:
                saver.save(sess, LOC_MODEL+"crack_capcha.model", global_step=step)  
                break                
   
            step += 1  

            
            
            
#以下为结果验证方法
def crack_captcha(captcha_image):  
    output = crack_captcha_cnn()  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        saver.restore(sess,LOC_MODEL+'crack_capcha.model-7200')#tf.train.latest_checkpoint(LOC_MODEL))  
   
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        print ('text_list',text_list)
        text = text_list[0].tolist()
        for i in range(len(text)):
            if text[i]>8:
               text[i]=text[i]-1 
        text=''.join([char_set[i] for i in text])        
        return text  


def predict_captcha(num=100):
    output = crack_captcha_cnn()  
   
    saver = tf.train.Saver()        
    validation=np.zeros(100)
    #valid_text=[]
    #valid_img=[]
    for i in range(num):
        valid_text, image = get_one_data(loc=LOC_TEST)
        image = convert2gray(image) 
        image = image.flatten() / 255 
        with tf.Session() as sess:  
            saver.restore(sess,LOC_MODEL+'crack_capcha.model-18800')#tf.train.latest_checkpoint(LOC_MODEL))  
       
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            #print ('text_list',text_list)
            text = text_list[0].tolist()
            for j in range(len(text)):
                if text[j]>8:
                   text[j]=text[j]-1 
            text=''.join([char_set[x] for x in text])
            if text==valid_text:
                validation[i]=1
                print("True 正确: {}  预测: {}".format(valid_text, text))
            else:
                print("False 正确: {}  预测: {}".format(valid_text, text))
    print('正确率:',np.sum(validation)/num)            
    #print("正确: {}  预测: {}".format(valid_text, text))

    
    
if __name__ == '__main__':
    if len(sys.argv)>1:
        if sys.argv[1]=='train':
            print ("mode:",sys.argv[1])
            train_crack_captcha_cnn()
        elif sys.argv[1]=='predict':
            print ("mode:",sys.argv[1])
            predict_captcha() 
        elif sys.argv[1]=='restore':   
            print ("mode:",sys.argv[1])
            restore_train_crack_captcha_cnn(LOC_MODEL+'crack_capcha.model-7200',7200,30000)
        else:
            print ("no such mode!!")