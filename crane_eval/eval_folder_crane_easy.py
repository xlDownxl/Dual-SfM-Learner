import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('tum_dir',)
args = parser.parse_args()
import subprocess

def iterate(dir):
    if os.path.isfile(dir+"/dispnet_model_best.pth.tar"):
        output = subprocess.getoutput("python3 test_disp.py --pretrained-dispnet {}/dispnet_model_best.pth.tar --pretrained-posenet {}/exp_pose_model_best.pth.tar --img-height 180 --img-width 320 --dataset crane --dataset-dir /wrk/users/nicjosw/crane_dataset_final --dataset-list crane_eval/crane_val_list_easy.txt --max-depth 10".format(dir,dir))
        text_file = open(args.tum_dir+"/"+dir.split("/")[-1]+"_crane_easy_best.txt", "w")
        n = text_file.write(output)
        text_file.close()
        print(output)
        output = subprocess.getoutput("python3 test_disp.py --pretrained-dispnet {}/dispnet_checkpoint.pth.tar --pretrained-posenet {}/exp_pose_checkpoint.pth.tar --img-height 180 --img-width 320 --dataset crane --dataset-dir /wrk/users/nicjosw/crane_dataset_final --dataset-list crane_eval/crane_val_list_easy.txt --max-depth 10".format(dir,dir))
        text_file = open(args.tum_dir+"/"+dir.split("/")[-1]+"_crane_easy_checkpoint.txt", "w")
        n = text_file.write(output)
        text_file.close()
        print(output)
        print(dir)
    else:
        for subfolder in os.listdir(dir):
            if os.path.isdir(dir+"/"+subfolder):
                iterate(dir+"/"+subfolder) 


iterate(args.tum_dir)