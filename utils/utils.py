import lpips
import torch
from insightface.app import FaceAnalysis

from PIL import ImageFont, ImageDraw, Image
from models.swinface_project.inference import SwinFaceCfg, load_model
from models.FIQA.utils import construct_full_model

def save_combined_images_and_individual_metrics(combined_og_inverted, metrics,path_combined, image_name):
    font = ImageFont.truetype("Roboto-Regular.ttf", 20)

    # ADD values and save combined image
    combined_og_inverted_img = Image.fromarray(combined_og_inverted)
    text_lines = []
    for k,v in metrics.items():
        if k != "FAC":
            text_lines.append(k + " : " + str(round(v[-1],2)))
    text = "\n".join(text_lines)

    draw = ImageDraw.Draw(combined_og_inverted_img)
    draw.text((0, 0), text, font=font, fill=(255, 0, 0, 255))

    combined_og_inverted_img.save(path_combined + image_name.split("/")[-1])

class FaceAnalysis2(FaceAnalysis):
    # NOTE: allows setting det_size for each detection call.
    # the model allows it but the wrapping code from insightface
    # doesn't show it, and people end up loading duplicate models
    # for different sizes where there is absolutely no need to
    def get(self, img, max_num=0, det_size=(640, 640)):
        if det_size is not None:
            self.det_model.input_size = det_size

        return super().get(img, max_num)

def load_models(opts):
    models = {}
    if opts.metrics.lpips:
        # Load LPIPS
        models["lpips"] = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    if opts.metrics.fic_swinface or opts.fac:
        # Load Swinface
        model_path = "pretrained_models/checkpoint_step_79999_gpu_0.pt"
        cfg = SwinFaceCfg()
        models["swinface"] = load_model(cfg, model_path)

    if opts.metrics.fic_arcface:
        # Load Arcface
        models["arcface"] = FaceAnalysis2(providers=['CUDAExecutionProvider'])
        models["arcface"].prepare(ctx_id=0, det_size=(704,704))  # images at 1024, face not found, need to do incrementale search loading a model each time, all faces are below 700, 1024->704 less try to find face

    if opts.metrics.lfiq:
        # Load FIQA
        models["fiqa"], _, models["fiqa_trans"] = construct_full_model("models/FIQA/configs/model_config.yaml")
        models["fiqa"].load_state_dict(torch.load("pretrained_models/aikd_diffiqar_model.pth"))
        models["fiqa"].to("cuda").eval()

    return models