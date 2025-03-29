from models.CascadeNetwork import create_cascade_nets, single_image_inference

import torch
import utils
import os

# python -m evaluate.eval_cascade

def main():
    img_path = '/home/gxy/cv/dataset/submission'  
    output_path = 'images/submission/'
    cascade_path_day = 'Param/RainDrop/Cascade/epoch10.pth.tar'
    cascade_path_night = 'Param/NightRaindrop/Cascade/Night-epoch10.pth.tar'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    # Load and sort image files
    img_files = sorted(
        [f for f in os.listdir(img_path) if f.endswith('.png')],
        key=lambda x: x
    )
    first_120 = img_files[:120]
    last_120 = img_files[-120:]

    # Load cascade networks
    cascade_day = create_cascade_nets()
    checkpoint_day = torch.load(cascade_path_day, map_location=device, weights_only=False)
    cascade_day.load_state_dict(checkpoint_day['state_dict'], strict=True)

    cascade_night = create_cascade_nets()
    checkpoint_night = torch.load(cascade_path_night, map_location=device, weights_only=False)
    cascade_night.load_state_dict(checkpoint_night['state_dict'], strict=True)

    # Process images
    for idx, img_name in enumerate(first_120 + last_120):
        img_full_path = os.path.join(img_path, img_name)
        img = utils.image.imread(img_full_path)
        img_tensor = utils.image.img2tensor(img).to(device)

        cascade = cascade_day if idx < 120 else cascade_night
        cascade = cascade.to(device)
        with torch.no_grad():
            output = single_image_inference(cascade, img_tensor)

        output_file_path = os.path.join(output_path, img_name)
        utils.logging.save_image(output, output_file_path)
        print(f"Processed and saved: {output_file_path}")

if __name__ == '__main__':
    main()