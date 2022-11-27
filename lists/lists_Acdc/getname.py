import os

List = os.listdir("/ai/jt/project_TransUNet/data/Acdc/test_vol_h5")

with open("test_vol.txt","w") as f:
    for name in List:
        rename = name.replace(".npy.h5", "")
        print(rename)
        
        f.write(rename)
        f.write("\n")
        
