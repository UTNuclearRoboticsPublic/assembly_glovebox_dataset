import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch import loggers as pl_loggers
import torchmetrics
import numpy as np
import torch
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
from statistics import mean
from PIL import Image
import PIL
import cv2
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from mobile_sam.utils.transforms import ResizeLongestSide
import time

from training.metrics import *
from transfer_learning.tl_dataloaders.bin_datamodule import BinAssemblyDataModule
from ..tl_models.UNET import UNET
from training.lightning_trainers.lightning_model import LitModel

### ADJUST THE DECODER CODE WHEN PIP INSTALLING
### ADD MSE LOSS INSTEAD

class MobileSamLitModel(LitModel):
    def __init__(self, learning_rate=0.001, weight_decay=0.1, test_dropout=False):
        super(MobileSamLitModel, self).__init__()
        
        model_type = "vit_t"
        sam_checkpoint = "/home/slwanna/FIRETEAM/Code/assembly_glovebox_dataset/training/MobileSAM/weights/mobile_sam.pt"

        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.resize_transform = ResizeLongestSide(self.model.image_encoder.img_size)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.test_dropout = test_dropout



        # freezing the encoder of the model
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

    def predict_step(self, batch, batch_idx):
        # used only when loading a model from a checkpoint/checking predictions
        x, y = batch

        input_box = torch.from_numpy(np.array([[0, 0, x.shape[2], x.shape[3]]])).float()
        input_boxes = input_box.repeat(x.shape[0], 1, 1)
        
        sparse_embeddings, dense_embeddings = self.get_embeds(self.model, x, input_boxes = input_boxes)

        raw_preds = self.get_prediction(model=self.model,
                                    image=x,
                                    sparse_embeddings=sparse_embeddings,
                                    dense_embeddings=dense_embeddings,
                                    image_embedding=x,
                                    multimask_output=True)

        probs = F.softmax(raw_preds, dim=1)
        preds = torch.argmax(probs, dim=1)

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.mask_decoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    
    def _get_preds(self, x):
        with torch.no_grad():
            
            input_image, pixel_values = self.prepare_image(x, self.resize_transform)

            input_boxes = torch.from_numpy(np.array([[0, 0, x.shape[2], x.shape[3]]])).float()
            input_boxes = input_boxes.to(self.device)
            rep_boxes = input_boxes.repeat(x.shape[0], 1, 1) # shape of each box -> 0, 0, 768, 768

            sparse_embeddings, dense_embeddings = self.get_embeds(self.model, x, input_boxes = rep_boxes)

        raw_preds = self.get_prediction(
                            model=self.model,
                            input_image=input_image,
                            original_image = x,
                            sparse_embeddings=sparse_embeddings,
                            dense_embeddings=dense_embeddings,
                            image_embedding=pixel_values,
                            multimask_output=True)
        print(f"raw_preds shape in _get_preds: {raw_preds.shape}")
        return raw_preds

    
    def _common_set(self, batch, batch_idx):
        """Run a forward pass and calculate training loss"""
        x, y = batch

        start_time = time.perf_counter()

        # with torch.no_grad():
            
        #     input_image, pixel_values = self.prepare_image(x, self.resize_transform)

        #     input_boxes = torch.from_numpy(np.array([[0, 0, x.shape[2], x.shape[3]]])).float()
        #     input_boxes = input_boxes.to(self.device)
        #     rep_boxes = input_boxes.repeat(x.shape[0], 1, 1) # shape of each box -> 0, 0, 768, 768

        #     sparse_embeddings, dense_embeddings = self.get_embeds(self.model, x, input_boxes = rep_boxes)

        # raw_preds = self.get_prediction(
        #                     model=self.model,
        #                     input_image=input_image,
        #                     original_image = x,
        #                     sparse_embeddings=sparse_embeddings,
        #                     dense_embeddings=dense_embeddings,
        #                     image_embedding=pixel_values,
        #                     multimask_output=True)

        raw_preds = self._get_preds(x)
        
        end_time = time.perf_counter()
        pred_time = end_time - start_time

        self.avg_pred_time = pred_time / x.shape[0]

        super()._set_time(self.avg_pred_time)

        loss = self.get_loss(raw_preds, y)
        # loss = F.cross_entropy(raw_preds, y.long())


        # adjust 0 if this is not working
        return loss, raw_preds
    
    def get_loss(self, raw_preds, y):
        y1, y2 = y
        # loss_1 = F.cross_entropy(raw_preds, y1.long())
        # loss_2 = F.cross_entropy(raw_preds, y2.long())
        y1 = torch.nn.functional.one_hot(y1.long(), num_classes=3)
        y1 = y1.permute(0,-1, 1, 2)
        y2 = torch.nn.functional.one_hot(y2.long(), num_classes=3)
        y2 = y2.permute(0,-1, 1, 2)

        print(f"shape of raw preds: {raw_preds.shape}")
        raw_preds = torch.squeeze(raw_preds, 1)
        print(f"shape of raw preds after squeeze: {raw_preds.shape}")

        print(f"y1 shape: {y1.shape}")
        print(f"y2 shape: {y2.shape}")

        loss_1 = F.mse_loss(raw_preds, y1.float()) # error bc this wasn't float earlier
        loss_2 = F.mse_loss(raw_preds, y2.float())
        loss = (loss_1 + loss_2) / 2
        
        return loss

    def prepare_image(self, image, transform):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img_tensor = transform.apply_image_torch(image)
        # resize_img_tensor = image.permute(2, 0, 1).contiguous()
        # resize_img_tensor = image.permute(2, 0, 1)
        input_image = self.model.preprocess(resized_img_tensor) # (B, 3, 1024, 1024)
        image_embedding = self.model.image_encoder(input_image)
        return input_image, image_embedding
    
    # not using the predictor because that doesn't allow us to compute gradients
    def get_embeds(self, model, image, points=None, mask_input=None, return_logits=False, input_boxes=None):
    # setting the embeddings and encoders

        # requires original image in (H, W) format

        transformed_boxes = self.resize_transform.apply_boxes_torch(input_boxes, image.shape[-2:])

        # Embed prompts
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=transformed_boxes,
            masks=mask_input,
        )

        return sparse_embeddings, dense_embeddings

    def get_prediction(self, model, input_image, original_image, sparse_embeddings, dense_embeddings, image_embedding, multimask_output=False, return_logits=False):
        # Predice Masks
        print(f"input image shape: {input_image.shape}")
        print(f"input image embedding: {image_embedding.shape}")

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embedding.to("cuda:2"),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        # 1 -> low res masks from the mask decoder
        # 2 -> input size to the model, in (H, W)
        # 3-> original size of image before resizing for input to the model
        masks = model.postprocess_masks(low_res_masks, input_image.shape[-2:], original_image.shape[-2:]).to("cuda:2")

        return masks


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    model = MobileSamLitModel()
    dm = BinAssemblyDataModule(
        fit_query= {
            'participants': ['Test_Subject_3'], # 1, 2, 3, 4, 6, 7, 9, 10, 11, 12
            'distribution':['ood'], # ood, id
            'task': ['J'], # J, TB
            'view': ['Top_View'] # Top_View, Side_View
        },
        test_query= {
            # queries can be the same as above
            'participants': ['Test_Subject_3'], 
            'distribution':['ood'],
            'task': ['J'], 
            'view': ['Top_View'],
        }
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./tb_logs', name="sam")
    # this is where checkpoints will be stored
    # look under root_dir here -> https://pytorch-lightning.readthedocs.io/en/0.10.0/logging.html


    trainer = pl.Trainer(accelerator="gpu", max_epochs=150, logger=tensorboard, fast_dev_run=True,
                         profiler="pytorch")

    # trainer.fit(model, dm, ckpt_path='./logs/lightning_logs/version_11/checkpoints/epoch=114-step=115.ckpt')
    trainer.fit(model, dm)

    trainer.test(model, dm)