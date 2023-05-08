import dlib
import argparse
import cv2
import sys
import time


def process_dlib_boxes(box):
    xmin = box.left()
    ymin = box.top()
    xmax = box.right()
    ymax = box.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def face_det_image(imgpath):
    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='../../input/test_data/image1.jpg',
                        help='path to the input image')
    parser.add_argument('-u', '--upsample', default=None, type=float,
                        help='factor by which to upsample the image, default None, ' + \
                            'pass 1, 2, 3, ...')
    args = vars(parser.parse_args())
    # read the image and convert to RGB color format
    # image = cv2.imread(args['input'])
    image = cv2.imread(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # path for saving the result image
    save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}_u{args['upsample']}.jpg"
    # initilaize the Dlib face detector according to the upsampling value
    detector = dlib.get_frontal_face_detector()
    # carry out the face detection in the image
    if args['upsample'] == None:
        start = time.time()
        detected_boxes = detector(image_rgb)
        end = time.time()
    elif args['upsample'] > 0 and args['upsample'] < 4:
        start = time.time()
        detected_boxes = detector(image_rgb, int(args['upsample']))
        end = time.time()
    # prematurely exit the program when upsample value is >= 4
    else:
        warn_string = 'Please provide usample value > 1 and < 4.' + \
                ' Else it might lock your CPU.'
        print(warn_string)
        sys.exit(0)
    # process the detection boxes
    for box in detected_boxes:
        res_box = process_dlib_boxes(box)
        cv2.rectangle(image, (res_box[0], res_box[1]),
                    (res_box[2], res_box[3]), (0, 255, 0), 
                    2)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    # cv2.imwrite(save_name, image)
    # print(f"Total faces detected: {len(detected_boxes)}")
    print(f"{end-start:.3f}s")
    # print(f"FPS: {1/(end-start):.3f}")


# face_det_image("low_res_dataset/img1.jpg")
# face_det_image("low_res_dataset/img2.jpg")
# face_det_image("low_res_dataset/img3.jpg")
# face_det_image("low_res_dataset/img4.jpg")
# face_det_image("low_res_dataset/img5.jpg")
# face_det_image("low_res_dataset/img6.jpg")
# face_det_image("low_res_dataset/img7.jpg")
# face_det_image("low_res_dataset/img8.jpg")
# face_det_image("low_res_dataset/img9.jpg")
# face_det_image("low_res_dataset/img10.jpg")
# face_det_image("low_res_dataset/img11.jpg")
# face_det_image("low_res_dataset/img12.jpg")
# face_det_image("low_res_dataset/img13.jpg")
# face_det_image("low_res_dataset/img14.jpg")
# face_det_image("low_res_dataset/img15.jpg")
# face_det_image("low_res_dataset/img16.jpg")
# face_det_image("low_res_dataset/img17.jpg")
# face_det_image("low_res_dataset/img18.jpg")
# face_det_image("low_res_dataset/img19.jpg")
# face_det_image("low_res_dataset/img20.jpg")
# face_det_image("low_res_dataset/img21.jpg")
# face_det_image("low_res_dataset/img22.jpg")
# face_det_image("low_res_dataset/img23.jpg")
# face_det_image("low_res_dataset/img24.jpg")
# face_det_image("low_res_dataset/img25.jpg")
# face_det_image("low_res_dataset/img26.jpg")
# face_det_image("low_res_dataset/img27.jpg")
# face_det_image("low_res_dataset/img28.jpg")
# face_det_image("low_res_dataset/img29.jpg")
# face_det_image("low_res_dataset/img30.jpg")


# face_det_image('Fam2a/1000306439_a1744969b8_1369_11099615@N00.jpg')
# face_det_image('Fam2a/1001738864_d4cb853874_1186_40124061@N00.jpg')
# face_det_image('Fam2a/1002585912_42de35c492_1405_44711882@N00.jpg')
# face_det_image('Fam2a/1003692712_3e4e298009_1180_7467212@N04.jpg')
# face_det_image('Fam2a/1011925598_aa9e47e53e_1366_69215398@N00.jpg')
# face_det_image('Fam2a/1012374989_23b999f21e_1116_8450063@N08.jpg')
# face_det_image('Fam2a/1012809184_655c160dbf_1287_18962472@N00.jpg')
# face_det_image('Fam2a/1016545105_873840a508_1150_92729647@N00.jpg')
# face_det_image('Fam2a/1017120681_e00da7dcef_1316_10975509@N03.jpg')
# face_det_image('Fam2a/1019616561_feed94da1f_1398_60792219@N00.jpg')
# face_det_image('Fam2a/1021735753_12d6e849e9_1052_17012619@N00.jpg')
# face_det_image('Fam2a/1022854323_d2835addda_1045_45795741@N00.jpg')
# face_det_image('Fam2a/1022990207_bf29b1d778_1214_50224145@N00.jpg')
# face_det_image('Fam2a/1022998857_bd42ab85a1_1210_50224145@N00.jpg')
# face_det_image('Fam2a/1023023109_e1c3fd3ecf_1195_51783879@N00.jpg')
# face_det_image('Fam2a/1023859710_4fa2ceea46_1125_50224145@N00.jpg')
# face_det_image('Fam2a/1024417205_acaf9c5938_1288_96603368@N00.jpg')
# face_det_image('Fam2a/1024436545_147e615323_1170_96603368@N00.jpg')
# face_det_image('Fam2a/1024440819_146afc75db_1390_96603368@N00.jpg')
# face_det_image('Fam2a/1024919820_be0ff5b20a_1055_93205202@N00.jpg')
# face_det_image('Fam2a/1025286614_336f91364e_1306_96603368@N00.jpg')
# face_det_image('Fam2a/1025292092_325b3df405_1153_96603368@N00.jpg')
# face_det_image('Fam2a/1025296488_4712c26a4f_1160_96603368@N00.jpg')
# face_det_image('Fam2a/1025297682_1c31d18f0a_1370_96603368@N00.jpg')
# face_det_image('Fam2a/1025838985_3ec98cd09c_1127_59496256@N00.jpg')
# face_det_image('Fam2a/1029898092_aa087e9e63_1409_58385743@N00.jpg')
# face_det_image('Fam2a/1030741878_6399e18431_1337_50361459@N00.jpg')
# face_det_image('Fam2a/1031582237_8c2f40b7cb_1143_17284432@N00.jpg')
# face_det_image('Fam2a/1031755873_0b5712d533_1342_90322259@N00.jpg')
# face_det_image('Fam2a/1031954059_d2a86414ab_1288_26801567@N00.jpg')


face_det_image('Masked dataset/images/maksssksksss1.png')
face_det_image('Masked dataset/images/maksssksksss2.png')
face_det_image('Masked dataset/images/maksssksksss3.png')
face_det_image('Masked dataset/images/maksssksksss4.png')
face_det_image('Masked dataset/images/maksssksksss5.png')
face_det_image('Masked dataset/images/maksssksksss6.png')
face_det_image('Masked dataset/images/maksssksksss7.png')
face_det_image('Masked dataset/images/maksssksksss8.png')
face_det_image('Masked dataset/images/maksssksksss9.png')
face_det_image('Masked dataset/images/maksssksksss10.png')
face_det_image('Masked dataset/images/maksssksksss11.png')
face_det_image('Masked dataset/images/maksssksksss12.png')
face_det_image('Masked dataset/images/maksssksksss13.png')
face_det_image('Masked dataset/images/maksssksksss14.png')
face_det_image('Masked dataset/images/maksssksksss15.png')
face_det_image('Masked dataset/images/maksssksksss16.png')
face_det_image('Masked dataset/images/maksssksksss17.png')
face_det_image('Masked dataset/images/maksssksksss18.png')
face_det_image('Masked dataset/images/maksssksksss19.png')
face_det_image('Masked dataset/images/maksssksksss20.png')
face_det_image('Masked dataset/images/maksssksksss21.png')
face_det_image('Masked dataset/images/maksssksksss22.png')
face_det_image('Masked dataset/images/maksssksksss23.png')
face_det_image('Masked dataset/images/maksssksksss24.png')
face_det_image('Masked dataset/images/maksssksksss25.png')
face_det_image('Masked dataset/images/maksssksksss26.png')
face_det_image('Masked dataset/images/maksssksksss27.png')
face_det_image('Masked dataset/images/maksssksksss28.png')
face_det_image('Masked dataset/images/maksssksksss29.png')
face_det_image('Masked dataset/images/maksssksksss30.png')