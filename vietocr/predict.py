import argparse
from PIL import Image

from VietOCR.vietocr.tool.predictor import Predictor
from VietOCR.vietocr.tool.config import Cfg
import os 

def get_listImage(test_folder_path):
    """Get all filenames in folder

    :return: dict : { filename1 : "fakelabel" filename2 : "fakelabel" ...}
    """
    image_files = sorted(os.listdir(test_folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    results = []

    for image_file in image_files:
        results.append(Image.open(os.path.join(test_folder_path, image_file)))

    return image_files , results

def write_file(filenames, labels,path):
  l = len(filenames)
  with open(path, "w") as file:
      for i in range(l):
            file.write(f"{filenames[i]} {labels[i]}\n")

def main():

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder_path', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')
    parser.add_argument('--path', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)
    image_files, images = get_listImage(args.test_folder_path)
    s = detector.predict_batch(imgs=images)
    for i in range(len(s)) :
        if not s[i]:
            print(image_files[i])
    write_file(image_files,s,args.path)

    # print(s)

if __name__ == '__main__':
    main()
