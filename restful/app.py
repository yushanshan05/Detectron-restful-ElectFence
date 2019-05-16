#coding=utf-8
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from model import Model
from flask import request,Response
from scipy import misc
import json
import urllib,urllib2
import cv2
import os
import time
#import core.infer_simple_test as infer_test
#from infer_simple_test import Model
from model import Model
app = Flask(__name__)


@app.route('/user', methods=['POST'])
def info():
    # logger add
    
    formatter = logging.Formatter("[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/oilsteal.log', maxBytes=2000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    ip = request.remote_addr
    info_str = 'IP:' + ip
    logger.info(info_str)
    info_str = 'model_path:' + cfg_path + '-' + weights_path
    logger.info(info_str)
    #imagepath = request.form.getlist('data')
    #imagepath = request.form.get("data",type=str,default=None)
    start_time = time.time()
    js = request.get_json()
    #return json init
    out_json = {"data":[]}
    
    #js is not dict return []
    info_str = ''
    if isinstance(js,dict) and js.has_key('data') :
        imagepath = js.get('data',None)
        info_str = 'images path:' + imagepath
        logger.info(info_str)
    else:
        logger.warning('post has not data or data-key!!!')
        end_time = time.time() - start_time
        logger.info('predict time:{}'.format(end_time))
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
    if (imagepath.startswith("https://") or imagepath.startswith("http://") or imagepath.startswith("file://")):
        imagefile = urllib.urlopen(imagepath)
        status=imagefile.code
        # url
        if(status==200): 
            image_data = imagefile.read()
            image_name = os.path.basename(imagepath)
            #new_imagepath = filepath+"/"+image_name
            new_imagepath = image_name
            with open(new_imagepath, 'wb') as code:
                code.write(image_data)
            #img_np = misc.imread(new_imagepath)
            img_np = cv2.imread(new_imagepath)  #read image by cv2 ,the same as /tool/test_net.py
            if img_np is None:
                logger.warning('the images is NONE!!!')
                end_time = time.time() - start_time
                logger.info('predict time:{}'.format(end_time))
                logger.removeHandler(handler)
                handler.close()
                return json.dumps(out_json)
        else:
            logger.warning('the image is not download on internet!!!')
            end_time = time.time() - start_time
            logger.info('predict time:{}'.format(end_time))
            logger.removeHandler(handler)
            handler.close()
            return json.dumps(out_json)
    # path 
    else:
        if not os.path.exists(imagepath):
            logger.warning('the image is not exists!!!')
            end_time = time.time() - start_time
            logger.info('predict time:{}'.format(end_time))
            logger.removeHandler(handler)
            handler.close()
            return json.dumps(out_json)
        else:
            #img_np = misc.imread(imagepath)
            start_readimage_time = time.time()
            img_np = cv2.imread(imagepath)  #read image by cv2 ,the same as /tool/test_net.py
            end_readimage_time = time.time() - start_readimage_time
            logger.info('readimage time:{}'.format(end_readimage_time))
            if img_np is None:
                logger.warning('the images is NONE!!!')
                end_time = time.time() - start_time
                logger.info('predict time:{}'.format(end_time))
                
                logger.removeHandler(handler)
                handler.close()
                return json.dumps(out_json)
    
    x_position = js.get('x_position',None)    
    y_position = js.get('y_position',None)
       
      
    if type(x_position) == type(y_position) and (type(x_position)!=type(list())):        
        x_position=[]
        y_position=[]        
        logger.info("no electronic fence!")        
    elif len(x_position)!=len(y_position)  or (len(x_position) ==len(y_position) and len(x_position) < 3):
        logger.warning('electronic fence is invalid!!!')
        x_position=[]
        y_position=[]        
    else:         
        logger.info("electronic fence:"+str(len(y_position))+"poly")
    #info_str = 'x_position:' + str(x_position) + '   y_position:' + str(y_position)
    logger.info('x_position:' + str(x_position) + '   y_position:' + str(y_position)) 
    
    start_model_time = time.time()
    predict_datalist = mm.predict(img_np, x_position, y_position)
    end_model_time = time.time() - start_model_time
    logger.info('model time:{}'.format(end_model_time))
    if len(predict_datalist) > 0:
        logger.info('the images predict completed!!!')
        res_log = []
        res_log.append(info_str)
        for i in range(len(predict_datalist)):
            single_data = {}
            single_data = predict_datalist[i]
            res_log.append(single_data['cls'])
        logger.info(res_log)
        out_json["data"] = predict_datalist
    else:
        logger.warning('the images has not right bbox!!!')
    end_time = time.time() - start_time
    logger.info('predict time:{}'.format(end_time))
    logger.removeHandler(handler)
    handler.close()
    return json.dumps(out_json)
    
    
if __name__ == '__main__':

    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    logger = logging.getLogger('oilsteal')    #set root level , default is WRAINING
    logger.setLevel(logging.DEBUG)
    
    '''
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/oilsteal.log', maxBytes=10000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)     #ok  start root log
    #app.logger.addHandler(handler)  #ok  start private log
    '''
    
    cfg_path = '/opt/oilstealing_model/retinanet.yaml'
    weights_path = '/opt/oilstealing_model/model.pkl'
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        cfg_path = './model/retinanet.yaml'
        weights_path = './model/model.pkl'
    mm = Model(cfg_path,weights_path)

    app.run(host="0.0.0.0",port=8080,debug=False)   #threaded=True

    
