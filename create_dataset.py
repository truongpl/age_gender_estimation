import mtcnn
import pandas as pd
import cv2
import glob
from utils import Params
import argparse

def create_annotate(dir_list, write_path, write_file):
    for d in dir_list:
        file_list = glob.glob(d+'*.jpg')
        for file in file_list:
            # Parse annotation
            file_name = file.split('/')[-1]

            try:
                age = int(file_name.split('_')[0])
                gender = 1 - int(file_name.split('_')[1]) # to make gender suitable with own label: 0=female, 1=male
                ethics = int(file_name.split('_')[2])
            except:
                continue

            # Use mtcnn to detect face
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w, _ = image.shape

            faces = detector.detect_faces(image)
            if len(faces) == 0:
                continue

            sorted(faces, key=lambda x: x['box'][2]*x['box'][3], reverse=False)
            largest_face = faces[0]

            if largest_face['confidence'] > 0.9 and len(largest_face['keypoints']) >= 4:
                box = largest_face['box']
                # Crop the frame
                width = box[3]
                height = box[2]

                # Get face margin
                mar_x = width * 0.2
                mar_y = height * 0.2

                xmin = box[1] - int(mar_x/2)
                if xmin < 0:
                    xmin = box[1]

                xmax = box[3] + box[1] + int(mar_x/2)
                if xmax > w:
                    xmax = box[3]+box[1]

                ymin = box[0] - int(mar_y/2)
                if ymin < 0:
                    ymin = box[0]

                ymax = box[2] + box[0] + int(mar_y/2)
                if ymax > h:
                    ymax = box[2]+box[0]

                face_img = image[xmin:xmax,ymin:ymax]

                # Drop small face
                if face_img.shape[0] > 30 and face_img.shape[1] > 30:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    write_string = write_path + file_name

                    cv2.imwrite(write_string, face_img)

                    annotate_string = write_string +',' + str(age) + ',' + str(gender) + ',' + str(ethics)
                    write_file.write(annotate_string+'\n')
                else:
                    print('Drop small face ', file_name)
            else:
                print('Not enough face information ')

if __name__ == '__main__':
    params = Params('./params.json')

    detector = mtcnn.MTCNN()

    train_dir = params.train_dir
    val_dir = params.val_dir

    # Create annotation file
    print('Create train file')
    train_file = open('train_utk.csv','w')
    train_file.write('image,age,gender,ethic\n')
    create_annotate(train_dir, './data/train_utk/', train_file)
    train_file.close()

    print('Create val file')
    val_file = open('val_utk.csv','w')
    val_file.write('image,age,gender,ethic\n')
    create_annotate(val_dir, './data/val_utk/', val_file)
    val_file.close()
