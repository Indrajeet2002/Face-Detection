import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def detectDNN(imgpath):
    modelFile = "DNN/model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "DNN/model/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    start = time.time()

    frame = cv2.imread(imgpath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    end = time.time()
    #[,frame,number of detection,[classid,class score,conf,x,y,h,w]]
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 3)

    # show the output image
    plt.imshow( frame)
    plt.show()
    print(f"{end-start:.3f}s")

# detectDNN('low_res_dataset/img1.jpg')
# detectDNN('low_res_dataset/img2.jpg')
# detectDNN('low_res_dataset/img3.jpg')
# detectDNN('low_res_dataset/img4.jpg')
# detectDNN('low_res_dataset/img5.jpg')
# detectDNN('low_res_dataset/img6.jpg')
# detectDNN('low_res_dataset/img7.jpg')
# detectDNN('low_res_dataset/img8.jpg')
# detectDNN('low_res_dataset/img9.jpg')
# detectDNN('low_res_dataset/img10.jpg')
# detectDNN('low_res_dataset/img11.jpg')
# detectDNN('low_res_dataset/img12.jpg')
# detectDNN('low_res_dataset/img13.jpg')
# detectDNN('low_res_dataset/img14.jpg')
# detectDNN('low_res_dataset/img15.jpg')
detectDNN('low_res_dataset/img16.jpg')
# detectDNN('low_res_dataset/img17.jpg')
# detectDNN('low_res_dataset/img18.jpg')
# detectDNN('low_res_dataset/img19.jpg')
# detectDNN('low_res_dataset/img20.jpg')
# detectDNN('low_res_dataset/img21.jpg')
# detectDNN('low_res_dataset/img22.jpg')
# detectDNN('low_res_dataset/img23.jpg')
# detectDNN('low_res_dataset/img24.jpg')
# detectDNN('low_res_dataset/img25.jpg')
# detectDNN('low_res_dataset/img26.jpg')
# detectDNN('low_res_dataset/img27.jpg')
# detectDNN('low_res_dataset/img28.jpg')
# detectDNN('low_res_dataset/img29.jpg')
# detectDNN('low_res_dataset/img30.jpg')


# detectDNN('Fam2a/1000306439_a1744969b8_1369_11099615@N00.jpg')
# detectDNN('Fam2a/1001738864_d4cb853874_1186_40124061@N00.jpg')
# detectDNN('Fam2a/1002585912_42de35c492_1405_44711882@N00.jpg')
# detectDNN('Fam2a/1003692712_3e4e298009_1180_7467212@N04.jpg')
# detectDNN('Fam2a/1011925598_aa9e47e53e_1366_69215398@N00.jpg')
# detectDNN('Fam2a/1012374989_23b999f21e_1116_8450063@N08.jpg')
# detectDNN('Fam2a/1012809184_655c160dbf_1287_18962472@N00.jpg')
# detectDNN('Fam2a/1016545105_873840a508_1150_92729647@N00.jpg')
# detectDNN('Fam2a/1017120681_e00da7dcef_1316_10975509@N03.jpg')
# detectDNN('Fam2a/1019616561_feed94da1f_1398_60792219@N00.jpg')
# detectDNN('Fam2a/1021735753_12d6e849e9_1052_17012619@N00.jpg')
# detectDNN('Fam2a/1022854323_d2835addda_1045_45795741@N00.jpg')
# detectDNN('Fam2a/1022990207_bf29b1d778_1214_50224145@N00.jpg')
# detectDNN('Fam2a/1022998857_bd42ab85a1_1210_50224145@N00.jpg')
# detectDNN('Fam2a/1023023109_e1c3fd3ecf_1195_51783879@N00.jpg')
# detectDNN('Fam2a/1023859710_4fa2ceea46_1125_50224145@N00.jpg')
# detectDNN('Fam2a/1024417205_acaf9c5938_1288_96603368@N00.jpg')
# detectDNN('Fam2a/1024436545_147e615323_1170_96603368@N00.jpg')
# detectDNN('Fam2a/1024440819_146afc75db_1390_96603368@N00.jpg')
# detectDNN('Fam2a/1024919820_be0ff5b20a_1055_93205202@N00.jpg')
# detectDNN('Fam2a/1025286614_336f91364e_1306_96603368@N00.jpg')
# detectDNN('Fam2a/1025292092_325b3df405_1153_96603368@N00.jpg')
# detectDNN('Fam2a/1025296488_4712c26a4f_1160_96603368@N00.jpg')
# detectDNN('Fam2a/1025297682_1c31d18f0a_1370_96603368@N00.jpg')
# detectDNN('Fam2a/1025838985_3ec98cd09c_1127_59496256@N00.jpg')
# detectDNN('Fam2a/1029898092_aa087e9e63_1409_58385743@N00.jpg')
# detectDNN('Fam2a/1030741878_6399e18431_1337_50361459@N00.jpg')
# detectDNN('Fam2a/1031582237_8c2f40b7cb_1143_17284432@N00.jpg')
# detectDNN('Fam2a/1031755873_0b5712d533_1342_90322259@N00.jpg')
# detectDNN('Fam2a/1031954059_d2a86414ab_1288_26801567@N00.jpg')


# detectDNN('Masked dataset/images/maksssksksss1.png')
# detectDNN('Masked dataset/images/maksssksksss2.png')
# detectDNN('Masked dataset/images/maksssksksss3.png')
# detectDNN('Masked dataset/images/maksssksksss4.png')
# detectDNN('Masked dataset/images/maksssksksss5.png')
# detectDNN('Masked dataset/images/maksssksksss6.png')
# detectDNN('Masked dataset/images/maksssksksss7.png')
# detectDNN('Masked dataset/images/maksssksksss8.png')
# detectDNN('Masked dataset/images/maksssksksss9.png')
# detectDNN('Masked dataset/images/maksssksksss10.png')
# detectDNN('Masked dataset/images/maksssksksss11.png')
# detectDNN('Masked dataset/images/maksssksksss12.png')
# detectDNN('Masked dataset/images/maksssksksss13.png')
# detectDNN('Masked dataset/images/maksssksksss14.png')
# detectDNN('Masked dataset/images/maksssksksss15.png')
# detectDNN('Masked dataset/images/maksssksksss16.png')
# detectDNN('Masked dataset/images/maksssksksss17.png')
# detectDNN('Masked dataset/images/maksssksksss18.png')
# detectDNN('Masked dataset/images/maksssksksss19.png')
# detectDNN('Masked dataset/images/maksssksksss20.png')
# detectDNN('Masked dataset/images/maksssksksss21.png')
# detectDNN('Masked dataset/images/maksssksksss22.png')
# detectDNN('Masked dataset/images/maksssksksss23.png')
# detectDNN('Masked dataset/images/maksssksksss24.png')
# detectDNN('Masked dataset/images/maksssksksss25.png')
# detectDNN('Masked dataset/images/maksssksksss26.png')
# detectDNN('Masked dataset/images/maksssksksss27.png')
# detectDNN('Masked dataset/images/maksssksksss28.png')
# detectDNN('Masked dataset/images/maksssksksss29.png')
# detectDNN('Masked dataset/images/maksssksksss30.png')

