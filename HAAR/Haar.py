import cv2
import matplotlib.pyplot as plt
import time

def detectHC(imgpath):
    cascPath = "HAAR/haarcascades/haarcascade_frontalface_default.xml"
    # cascPath = "HAAR/haarcascades/haarcascade_frontalface_alt_tree.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    start = time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX

    frame = cv2.imread(imgpath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    end = time.time()
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame,'Face',(x, y), font, 2,(255,0,0),5)

    cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),3)
    plt.imshow(frame)
    plt.show()
    print(f"{end-start:.3f}s")

# detectHC('low_res_dataset/img1.jpg')
# detectHC('low_res_dataset/img2.jpg')
# detectHC('low_res_dataset/img3.jpg')
# detectHC('low_res_dataset/img4.jpg')
# detectHC('low_res_dataset/img5.jpg')
# detectHC('low_res_dataset/img6.jpg')
# detectHC('low_res_dataset/img7.jpg')
# detectHC('low_res_dataset/img8.jpg')
# detectHC('low_res_dataset/img9.jpg')
# detectHC('low_res_dataset/img10.jpg')
# detectHC('low_res_dataset/img11.jpg')
# detectHC('low_res_dataset/img12.jpg')
# detectHC('low_res_dataset/img13.jpg')
# detectHC('low_res_dataset/img14.jpg')
# detectHC('low_res_dataset/img15.jpg')
# detectHC('low_res_dataset/img16.jpg')
# detectHC('low_res_dataset/img17.jpg')
# detectHC('low_res_dataset/img18.jpg')
# detectHC('low_res_dataset/img19.jpg')
# detectHC('low_res_dataset/img20.jpg')
# detectHC('low_res_dataset/img21.jpg')
# detectHC('low_res_dataset/img22.jpg')
# detectHC('low_res_dataset/img23.jpg')
# detectHC('low_res_dataset/img24.jpg')
# detectHC('low_res_dataset/img25.jpg')
# detectHC('low_res_dataset/img26.jpg')
# detectHC('low_res_dataset/img27.jpg')
# detectHC('low_res_dataset/img28.jpg')
# detectHC('low_res_dataset/img29.jpg')
# detectHC('low_res_dataset/img30.jpg')



# detectHC('Fam2a/1000306439_a1744969b8_1369_11099615@N00.jpg')
# detectHC('Fam2a/1001738864_d4cb853874_1186_40124061@N00.jpg')
# detectHC('Fam2a/1002585912_42de35c492_1405_44711882@N00.jpg')
# detectHC('Fam2a/1003692712_3e4e298009_1180_7467212@N04.jpg')
# detectHC('Fam2a/1011925598_aa9e47e53e_1366_69215398@N00.jpg')
# detectHC('Fam2a/1012374989_23b999f21e_1116_8450063@N08.jpg')
# detectHC('Fam2a/1012809184_655c160dbf_1287_18962472@N00.jpg')
# detectHC('Fam2a/1016545105_873840a508_1150_92729647@N00.jpg')
# detectHC('Fam2a/1017120681_e00da7dcef_1316_10975509@N03.jpg')
# detectHC('Fam2a/1019616561_feed94da1f_1398_60792219@N00.jpg')
# detectHC('Fam2a/1021735753_12d6e849e9_1052_17012619@N00.jpg')
# detectHC('Fam2a/1022854323_d2835addda_1045_45795741@N00.jpg')
# detectHC('Fam2a/1022990207_bf29b1d778_1214_50224145@N00.jpg')
# detectHC('Fam2a/1022998857_bd42ab85a1_1210_50224145@N00.jpg')
# detectHC('Fam2a/1023023109_e1c3fd3ecf_1195_51783879@N00.jpg')
# detectHC('Fam2a/1023859710_4fa2ceea46_1125_50224145@N00.jpg')
# detectHC('Fam2a/1024417205_acaf9c5938_1288_96603368@N00.jpg')
# detectHC('Fam2a/1024436545_147e615323_1170_96603368@N00.jpg')
# detectHC('Fam2a/1024440819_146afc75db_1390_96603368@N00.jpg')
# detectHC('Fam2a/1024919820_be0ff5b20a_1055_93205202@N00.jpg')
# detectHC('Fam2a/1025286614_336f91364e_1306_96603368@N00.jpg')
# detectHC('Fam2a/1025292092_325b3df405_1153_96603368@N00.jpg')
# detectHC('Fam2a/1025296488_4712c26a4f_1160_96603368@N00.jpg')
# detectHC('Fam2a/1025297682_1c31d18f0a_1370_96603368@N00.jpg')
# detectHC('Fam2a/1025838985_3ec98cd09c_1127_59496256@N00.jpg')
# detectHC('Fam2a/1029898092_aa087e9e63_1409_58385743@N00.jpg')
# detectHC('Fam2a/1030741878_6399e18431_1337_50361459@N00.jpg')
# detectHC('Fam2a/1031582237_8c2f40b7cb_1143_17284432@N00.jpg')
# detectHC('Fam2a/1031755873_0b5712d533_1342_90322259@N00.jpg')
# detectHC('Fam2a/1031954059_d2a86414ab_1288_26801567@N00.jpg') 


detectHC('Masked dataset/images/maksssksksss1.png')
detectHC('Masked dataset/images/maksssksksss2.png')
detectHC('Masked dataset/images/maksssksksss3.png')
detectHC('Masked dataset/images/maksssksksss4.png')
detectHC('Masked dataset/images/maksssksksss5.png')
detectHC('Masked dataset/images/maksssksksss6.png')
detectHC('Masked dataset/images/maksssksksss7.png')
detectHC('Masked dataset/images/maksssksksss8.png')
detectHC('Masked dataset/images/maksssksksss9.png')
detectHC('Masked dataset/images/maksssksksss10.png')
detectHC('Masked dataset/images/maksssksksss11.png')
detectHC('Masked dataset/images/maksssksksss12.png')
detectHC('Masked dataset/images/maksssksksss13.png')
detectHC('Masked dataset/images/maksssksksss14.png')
detectHC('Masked dataset/images/maksssksksss15.png')
detectHC('Masked dataset/images/maksssksksss16.png')
detectHC('Masked dataset/images/maksssksksss17.png')
detectHC('Masked dataset/images/maksssksksss18.png')
detectHC('Masked dataset/images/maksssksksss19.png')
detectHC('Masked dataset/images/maksssksksss20.png')
detectHC('Masked dataset/images/maksssksksss21.png')
detectHC('Masked dataset/images/maksssksksss22.png')
detectHC('Masked dataset/images/maksssksksss23.png')
detectHC('Masked dataset/images/maksssksksss24.png')
detectHC('Masked dataset/images/maksssksksss25.png')
detectHC('Masked dataset/images/maksssksksss26.png')
detectHC('Masked dataset/images/maksssksksss27.png')
detectHC('Masked dataset/images/maksssksksss28.png')
detectHC('Masked dataset/images/maksssksksss29.png')
detectHC('Masked dataset/images/maksssksksss30.png')